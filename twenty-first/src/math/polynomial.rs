use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Rem;
use std::ops::Sub;

use arbitrary::Arbitrary;
use itertools::EitherOrBoth;
use itertools::Itertools;
use num_bigint::BigInt;
use num_traits::One;
use num_traits::Zero;
use rayon::prelude::*;

use crate::math::ntt::intt;
use crate::math::ntt::ntt;
use crate::math::traits::FiniteField;
use crate::math::traits::ModPowU32;

use super::b_field_element::BFieldElement;
use super::traits::PrimitiveRootOfUnity;

impl<FF: FiniteField> Zero for Polynomial<FF> {
    fn zero() -> Self {
        Self {
            coefficients: vec![],
        }
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<FF: FiniteField> One for Polynomial<FF> {
    fn one() -> Self {
        Self {
            coefficients: vec![FF::one()],
        }
    }

    fn is_one(&self) -> bool {
        self.degree() == 0 && self.coefficients[0].is_one()
    }
}

#[derive(Clone, Arbitrary)]
pub struct Polynomial<FF: FiniteField> {
    pub coefficients: Vec<FF>,
}

impl<FF: FiniteField> Debug for Polynomial<FF> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polynomial")
            .field("coefficients", &self.coefficients)
            .finish()
    }
}

impl<FF: FiniteField> Hash for Polynomial<FF> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
    }
}

impl<FF: FiniteField> Display for Polynomial<FF> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        let degree = match self.degree() {
            -1 => return write!(f, "0"),
            d => d as usize,
        };

        for pow in (0..=degree).rev() {
            let coeff = self.coefficients[pow];
            if coeff.is_zero() {
                continue;
            }

            if pow != degree {
                write!(f, " + ")?;
            }
            if !coeff.is_one() || pow == 0 {
                write!(f, "{coeff}")?;
            }
            match pow {
                0 => (),
                1 => write!(f, "x")?,
                _ => write!(f, "x^{pow}")?,
            }
        }

        Ok(())
    }
}

impl<FF: FiniteField> PartialEq for Polynomial<FF> {
    fn eq(&self, other: &Self) -> bool {
        if self.degree() != other.degree() {
            return false;
        }

        self.coefficients
            .iter()
            .zip(other.coefficients.iter())
            .all(|(x, y)| x == y)
    }
}

impl<FF: FiniteField> Eq for Polynomial<FF> {}

impl<FF> Polynomial<FF>
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    /// [Fast multiplication](Self::multiply) is slower than [naïve multiplication](Self::mul)
    /// for polynomials of degree less than this threshold.
    ///
    /// Extracted from `cargo bench --bench poly_mul` on mjolnir.
    const FAST_MULTIPLY_CUTOFF_THRESHOLD: isize = 1 << 8;

    /// [Fast division](Self::fast_divide) is slower than [naïve divison](Self::naive_divide) for
    /// polynomials of degree less than this threshold.
    ///
    /// Extracted from `cargo bench --bench poly_div` on mjolnir.
    const FAST_DIVIDE_CUTOFF_THRESHOLD: isize = isize::MAX - 1;

    /// Computing the [fast zerofier][fast] is slower than computing the [smart zerofier][smart] for
    /// domain sizes smaller than this threshold. The [naïve zerofier][naive] is always slower to
    /// compute than the [smart zerofier][smart] for domain sizes smaller than the threshold.
    ///
    /// Extracted from `cargo bench --bench zerofier` on mjolnir.
    ///
    /// [naive]: Self::naive_zerofier
    /// [smart]: Self::smart_zerofier
    /// [fast]: Self::fast_zerofier
    const FAST_ZEROFIER_CUTOFF_THRESHOLD: usize = 200;

    /// [Fast interpolation](Self::fast_interpolate) is slower than
    /// [Lagrange interpolation](Self::lagrange_interpolate) below this threshold.
    ///
    /// Extracted from `cargo bench --bench interpolation` on mjolnir.
    const FAST_INTERPOLATE_CUTOFF_THRESHOLD: usize = 1 << 9;

    /// [Fast evaluation](Self::fast_evaluate) is slower than evaluating every point in parallel
    /// below this threshold.
    ///
    /// Extracted from `cargo bench --bench evaluation` on mjolnir.
    const FAST_EVALUATE_CUTOFF_THRESHOLD: usize = usize::MAX - 42;

    /// Return the polynomial which corresponds to the transformation `x → α·x`.
    ///
    /// Given a polynomial P(x), produce P'(x) := P(α·x). Evaluating P'(x) then corresponds to
    /// evaluating P(α·x).
    #[must_use]
    pub fn scale<S, XF>(&self, alpha: S) -> Polynomial<XF>
    where
        S: Clone,
        FF: Mul<XF, Output = XF>,
        XF: FiniteField + Mul<S, Output = XF>,
    {
        let mut power_of_alpha = XF::one();
        let mut return_coefficients = Vec::with_capacity(self.coefficients.len());
        for &coefficient in &self.coefficients {
            return_coefficients.push(coefficient * power_of_alpha);
            power_of_alpha = power_of_alpha * alpha.clone();
        }
        Polynomial::new(return_coefficients)
    }

    /// It is the caller's responsibility that this function is called with sufficiently large input
    /// to be safe and to be faster than `square`.
    #[must_use]
    pub fn fast_square(&self) -> Self {
        let degree = self.degree();
        if degree == -1 {
            return Self::zero();
        }
        if degree == 0 {
            return Self::from_constant(self.coefficients[0] * self.coefficients[0]);
        }

        let result_degree: u64 = 2 * self.degree() as u64;
        let order = (result_degree + 1).next_power_of_two();
        let root_res = BFieldElement::primitive_root_of_unity(order);
        let root = match root_res {
            Some(n) => n,
            None => panic!("Failed to find primitive root for order = {order}"),
        };

        let mut coefficients = self.coefficients.to_vec();
        coefficients.resize(order as usize, FF::zero());
        let log_2_of_n = coefficients.len().ilog2();
        ntt::<FF>(&mut coefficients, root, log_2_of_n);

        for element in coefficients.iter_mut() {
            *element = element.to_owned() * element.to_owned();
        }

        intt::<FF>(&mut coefficients, root, log_2_of_n);
        coefficients.truncate(result_degree as usize + 1);

        Polynomial { coefficients }
    }

    #[must_use]
    pub fn square(&self) -> Self {
        let degree = self.degree();
        if degree == -1 {
            return Self::zero();
        }

        // A benchmark run on sword_smith's PC revealed that `fast_square` was faster when the input
        // size exceeds a length of 64.
        let squared_coefficient_len = self.degree() as usize * 2 + 1;
        if squared_coefficient_len > 64 {
            return self.fast_square();
        }

        let zero = FF::zero();
        let one = FF::one();
        let two = one + one;
        let mut squared_coefficients = vec![zero; squared_coefficient_len];

        // TODO: Review.
        for i in 0..self.coefficients.len() {
            let ci = self.coefficients[i];
            squared_coefficients[2 * i] += ci * ci;

            for j in i + 1..self.coefficients.len() {
                let cj = self.coefficients[j];
                squared_coefficients[i + j] += two * ci * cj;
            }
        }

        Self {
            coefficients: squared_coefficients,
        }
    }

    #[must_use]
    pub fn fast_mod_pow(&self, pow: BigInt) -> Self {
        let one = FF::one();

        // Special case to handle 0^0 = 1
        if pow.is_zero() {
            return Self::from_constant(one);
        }

        if self.is_zero() {
            return Self::zero();
        }

        if pow.is_one() {
            return self.clone();
        }

        let mut acc = Polynomial::from_constant(one);
        let bit_length: u64 = pow.bits();
        for i in 0..bit_length {
            acc = acc.square();
            let set: bool =
                !(pow.clone() & Into::<BigInt>::into(1u128 << (bit_length - 1 - i))).is_zero();
            if set {
                acc = self.to_owned() * acc;
            }
        }

        acc
    }

    /// Multiply `self` by `other`.
    ///
    /// Prefer this over [`self * other`](Self::mul) since it chooses the fastest multiplication
    /// strategy.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        if self.degree() + other.degree() < Self::FAST_MULTIPLY_CUTOFF_THRESHOLD {
            self.naive_multiply(other)
        } else {
            self.fast_multiply(other)
        }
    }

    /// Use [Self::multiply] instead. Only `pub` to allow benchmarking; not considered part of the
    /// public API.
    ///
    /// This method is asymptotically faster than [naive multiplication](Self::naive_multiply). For
    /// small instances, _i.e._, polynomials of low degree, it is slower.
    ///
    /// The time complexity of this method is in O(n·log(n)), where `n` is the sum of the degrees
    /// of the operands. The time complexity of the naive multiplication is in O(n^2).
    #[doc(hidden)]
    pub fn fast_multiply(&self, other: &Self) -> Self {
        let Ok(degree) = usize::try_from(self.degree() + other.degree()) else {
            return Self::zero();
        };
        let order = (degree + 1).next_power_of_two();
        let order_u64 = u64::try_from(order).unwrap();
        let root = BFieldElement::primitive_root_of_unity(order_u64).unwrap();

        let mut lhs_coefficients = self.coefficients.to_vec();
        let mut rhs_coefficients = other.coefficients.to_vec();

        lhs_coefficients.resize(order, FF::zero());
        rhs_coefficients.resize(order, FF::zero());

        ntt::<FF>(&mut lhs_coefficients, root, order.ilog2());
        ntt::<FF>(&mut rhs_coefficients, root, order.ilog2());

        let mut hadamard_product: Vec<FF> = rhs_coefficients
            .into_iter()
            .zip(lhs_coefficients)
            .map(|(r, l)| r * l)
            .collect();

        intt::<FF>(&mut hadamard_product, root, order.ilog2());
        hadamard_product.truncate(degree + 1);
        Self::new(hadamard_product)
    }

    /// Compute the lowest degree polynomial with the provided roots.
    pub fn zerofier(roots: &[FF]) -> Self {
        if roots.len() < Self::FAST_ZEROFIER_CUTOFF_THRESHOLD {
            Self::smart_zerofier(roots)
        } else {
            Self::fast_zerofier(roots)
        }
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn smart_zerofier(roots: &[FF]) -> Self {
        let mut zerofier = vec![FF::zero(); roots.len() + 1];
        zerofier[0] = FF::one();
        let mut num_coeffs = 1;
        for &root in roots {
            for k in (1..=num_coeffs).rev() {
                zerofier[k] = zerofier[k - 1] - root * zerofier[k];
            }
            zerofier[0] = -root * zerofier[0];
            num_coeffs += 1;
        }
        Self::new(zerofier)
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn fast_zerofier(roots: &[FF]) -> Self {
        let mid_point = roots.len() / 2;
        let left_half = &roots[..mid_point];
        let right_half = &roots[mid_point..];
        let mut zerofier_halves = [left_half, right_half]
            .into_par_iter()
            .map(|half_domain| Self::zerofier(half_domain))
            .collect::<Vec<_>>();
        let right = zerofier_halves.pop().unwrap();
        let left = zerofier_halves.pop().unwrap();

        Self::multiply(&left, &right)
    }

    /// Construct the lowest-degree polynomial interpolating the given points.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let domain = bfe_vec![0, 1, 2, 3];
    /// let values = bfe_vec![1, 3, 5, 7];
    /// let polynomial = Polynomial::interpolate(&domain, &values);
    ///
    /// assert_eq!(1, polynomial.degree());
    /// assert_eq!(bfe!(9), polynomial.evaluate(&bfe!(4)));
    /// ```
    ///
    /// # Panics
    ///
    /// - Panics if the provided domain is empty.
    /// - Panics if the provided domain and values are not of the same length.
    pub fn interpolate(domain: &[FF], values: &[FF]) -> Self {
        assert!(
            !domain.is_empty(),
            "interpolation must happen through more than zero points"
        );
        assert_eq!(
            domain.len(),
            values.len(),
            "The domain and values lists have to be of equal length."
        );

        if domain.len() <= Self::FAST_INTERPOLATE_CUTOFF_THRESHOLD {
            Self::lagrange_interpolate(domain, values)
        } else {
            Self::fast_interpolate(domain, values)
        }
    }

    /// Any fast interpolation will use NTT, so this is mainly used for testing/integrity
    /// purposes. This also means that it is not pivotal that this function has an optimal
    /// runtime.
    #[doc(hidden)]
    pub fn lagrange_interpolate_zipped(points: &[(FF, FF)]) -> Self {
        assert!(
            !points.is_empty(),
            "interpolation must happen through more than zero points"
        );
        assert!(
            points.iter().map(|x| x.0).all_unique(),
            "Repeated x values received. Got: {points:?}",
        );

        let xs: Vec<FF> = points.iter().map(|x| x.0.to_owned()).collect();
        let ys: Vec<FF> = points.iter().map(|x| x.1.to_owned()).collect();
        Self::lagrange_interpolate(&xs, &ys)
    }

    #[doc(hidden)]
    pub fn lagrange_interpolate(domain: &[FF], values: &[FF]) -> Self {
        debug_assert!(
            !domain.is_empty(),
            "interpolation domain cannot have zero points"
        );
        debug_assert_eq!(domain.len(), values.len());

        let zero = FF::zero();
        let zerofier = Self::zerofier(domain).coefficients;

        // In each iteration of this loop, accumulate into the sum one polynomial that evaluates
        // to some abscis (y-value) in the given ordinate (domain point), and to zero in all other
        // ordinates.
        let mut lagrange_sum_array = vec![zero; domain.len()];
        let mut summand_array = vec![zero; domain.len()];
        for (i, &abscis) in values.iter().enumerate() {
            // divide (X - domain[i]) out of zerofier to get unweighted summand
            let mut leading_coefficient = zerofier[domain.len()];
            let mut supporting_coefficient = zerofier[domain.len() - 1];
            for k in (0..domain.len()).rev() {
                summand_array[k] = leading_coefficient;
                leading_coefficient = supporting_coefficient + leading_coefficient * domain[i];
                if k != 0 {
                    supporting_coefficient = zerofier[k - 1];
                }
            }

            // summand does not necessarily evaluate to 1 in domain[i]: correct for this value
            let mut summand_eval = zero;
            for &s in summand_array.iter().rev() {
                summand_eval = summand_eval * domain[i] + s;
            }
            let corrected_abscis = abscis / summand_eval;

            // accumulate term
            for j in 0..domain.len() {
                lagrange_sum_array[j] += corrected_abscis * summand_array[j];
            }
        }

        Self::new(lagrange_sum_array)
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn fast_interpolate(domain: &[FF], values: &[FF]) -> Self {
        debug_assert!(
            !domain.is_empty(),
            "interpolation domain cannot have zero points"
        );
        debug_assert_eq!(domain.len(), values.len());

        let hadamard_mul = |x: &[_], y: Vec<_>| x.iter().zip(y).map(|(&n, d)| n * d).collect_vec();

        let mid_point = domain.len() / 2;
        let left_domain_half = &domain[..mid_point];
        let left_values_half = &values[..mid_point];
        let right_domain_half = &domain[mid_point..];
        let right_values_half = &values[mid_point..];

        let left_zerofier = Self::zerofier(left_domain_half);
        let right_zerofier = Self::zerofier(right_domain_half);

        let left_offset = right_zerofier.fast_evaluate(left_domain_half);
        let right_offset = left_zerofier.fast_evaluate(right_domain_half);

        let left_offset_inverse = FF::batch_inversion(left_offset);
        let left_targets: Vec<FF> = hadamard_mul(left_values_half, left_offset_inverse);
        let left_interpolant = Self::interpolate(left_domain_half, &left_targets);

        let right_offset_inverse = FF::batch_inversion(right_offset);
        let right_targets: Vec<FF> = hadamard_mul(right_values_half, right_offset_inverse);
        let right_interpolant = Self::interpolate(right_domain_half, &right_targets);

        let left_term = left_interpolant.multiply(&right_zerofier);
        let right_term = right_interpolant.multiply(&left_zerofier);
        left_term + right_term
    }

    pub fn batch_fast_interpolate(
        domain: &[FF],
        values_matrix: &Vec<Vec<FF>>,
        primitive_root: BFieldElement,
        root_order: usize,
    ) -> Vec<Self> {
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            BFieldElement::one(),
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {primitive_root:?}\
            Supplied order was: {root_order:?}"
        );

        assert!(
            !domain.is_empty(),
            "Cannot fast interpolate through zero points.",
        );

        let mut zerofier_dictionary: HashMap<(FF, FF), Polynomial<FF>> = HashMap::default();
        let mut offset_inverse_dictionary: HashMap<(FF, FF), Vec<FF>> = HashMap::default();

        Self::batch_fast_interpolate_with_memoization(
            domain,
            values_matrix,
            &mut zerofier_dictionary,
            &mut offset_inverse_dictionary,
        )
    }

    fn batch_fast_interpolate_with_memoization(
        domain: &[FF],
        values_matrix: &Vec<Vec<FF>>,
        zerofier_dictionary: &mut HashMap<(FF, FF), Polynomial<FF>>,
        offset_inverse_dictionary: &mut HashMap<(FF, FF), Vec<FF>>,
    ) -> Vec<Self> {
        // This value of 16 was found to be optimal through a benchmark on sword_smith's
        // machine.
        const OPTIMAL_CUTOFF_POINT_FOR_BATCHED_INTERPOLATION: usize = 16;
        if domain.len() < OPTIMAL_CUTOFF_POINT_FOR_BATCHED_INTERPOLATION {
            return values_matrix
                .iter()
                .map(|values| Self::lagrange_interpolate(domain, values))
                .collect();
        }

        // calculate everything related to the domain
        let half = domain.len() / 2;

        let left_key = (domain[0], domain[half - 1]);
        let left_zerofier = match zerofier_dictionary.get(&left_key) {
            Some(z) => z.to_owned(),
            None => {
                let left_zerofier = Self::zerofier(&domain[..half]);
                zerofier_dictionary.insert(left_key, left_zerofier.clone());
                left_zerofier
            }
        };
        let right_key = (domain[half], *domain.last().unwrap());
        let right_zerofier = match zerofier_dictionary.get(&right_key) {
            Some(z) => z.to_owned(),
            None => {
                let right_zerofier = Self::zerofier(&domain[half..]);
                zerofier_dictionary.insert(right_key, right_zerofier.clone());
                right_zerofier
            }
        };

        let left_offset_inverse = match offset_inverse_dictionary.get(&left_key) {
            Some(vector) => vector.to_owned(),
            None => {
                let left_offset: Vec<FF> = Self::fast_evaluate(&right_zerofier, &domain[..half]);
                let left_offset_inverse = FF::batch_inversion(left_offset);
                offset_inverse_dictionary.insert(left_key, left_offset_inverse.clone());
                left_offset_inverse
            }
        };
        let right_offset_inverse = match offset_inverse_dictionary.get(&right_key) {
            Some(vector) => vector.to_owned(),
            None => {
                let right_offset: Vec<FF> = Self::fast_evaluate(&left_zerofier, &domain[half..]);
                let right_offset_inverse = FF::batch_inversion(right_offset);
                offset_inverse_dictionary.insert(right_key, right_offset_inverse.clone());
                right_offset_inverse
            }
        };

        // prepare target matrices
        let all_left_targets: Vec<_> = values_matrix
            .par_iter()
            .map(|values| {
                values[..half]
                    .iter()
                    .zip(left_offset_inverse.iter())
                    .map(|(n, d)| n.to_owned() * *d)
                    .collect()
            })
            .collect();
        let all_right_targets: Vec<_> = values_matrix
            .par_iter()
            .map(|values| {
                values[half..]
                    .par_iter()
                    .zip(right_offset_inverse.par_iter())
                    .map(|(n, d)| n.to_owned() * *d)
                    .collect()
            })
            .collect();

        // recurse
        let left_interpolants = Self::batch_fast_interpolate_with_memoization(
            &domain[..half],
            &all_left_targets,
            zerofier_dictionary,
            offset_inverse_dictionary,
        );
        let right_interpolants = Self::batch_fast_interpolate_with_memoization(
            &domain[half..],
            &all_right_targets,
            zerofier_dictionary,
            offset_inverse_dictionary,
        );

        // add vectors of polynomials
        let interpolants = left_interpolants
            .par_iter()
            .zip(right_interpolants.par_iter())
            .map(|(left_interpolant, right_interpolant)| {
                let left_term = left_interpolant.multiply(&right_zerofier);
                let right_term = right_interpolant.multiply(&left_zerofier);

                left_term + right_term
            })
            .collect();

        interpolants
    }

    pub fn batch_evaluate(&self, domain: &[FF]) -> Vec<FF> {
        if domain.len() <= Self::FAST_EVALUATE_CUTOFF_THRESHOLD {
            domain.par_iter().map(|p| self.evaluate(p)).collect()
        } else {
            self.fast_evaluate(domain)
        }
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn vector_batch_evaluate(&self, domain: &[FF]) -> Vec<FF> {
        let mut accumulators = vec![FF::zero(); domain.len()];
        for &c in self.coefficients.iter().rev() {
            accumulators
                .par_iter_mut()
                .zip(domain)
                .for_each(|(acc, &x)| *acc = *acc * x + c);
        }

        accumulators
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn fast_evaluate(&self, domain: &[FF]) -> Vec<FF> {
        let mid_point = domain.len() / 2;
        let left_half = &domain[..mid_point];
        let right_half = &domain[mid_point..];

        [left_half, right_half]
            .into_par_iter()
            .map(|half_domain| {
                let zerofier = Self::zerofier(half_domain);
                let (_, zerofier_inverse, _) = Self::xgcd(zerofier.clone(), Self::zero());
                let quotient = self.multiply(&zerofier_inverse);
                let remainder = self.clone() - quotient.multiply(&zerofier);
                remainder.batch_evaluate(half_domain)
            })
            .flatten()
            .collect()
    }

    /// Fast evaluate on a coset domain, which is the group generated by `generator^i * offset`.
    ///
    /// ### Current limitations
    ///
    /// - The order of the domain must be greater than the degree of `self`.
    pub fn fast_coset_evaluate(
        &self,
        offset: FF,
        generator: BFieldElement,
        order: usize,
    ) -> Vec<FF> {
        // NTT's input and output are of the same size. For domains of an order that is larger than
        // or equal to the number of coefficients of the polynomial, padding with leading zeros
        // (a no-op to the polynomial) achieves this requirement. However, if the order is smaller
        // than the number of coefficients in the polynomial, this would mean chopping off leading
        // coefficients, which changes the polynomial. Therefore, this method is currently limited
        // to domain orders greater than the degree of the polynomial.
        assert!(
            (order as isize) > self.degree(),
            "`Polynomial::fast_coset_evaluate` is currently limited to domains of order \
            greater than the degree of the polynomial."
        );

        let mut coefficients = self.scale(offset).coefficients;
        coefficients.resize(order, FF::zero());

        let log_2_of_n = coefficients.len().ilog2();
        ntt::<FF>(&mut coefficients, generator, log_2_of_n);
        coefficients
    }

    /// The inverse of `fast_coset_evaluate`. The number of provided values must equal the order
    /// of the generator, _i.e._, the size of the domain.
    pub fn fast_coset_interpolate(offset: FF, generator: BFieldElement, values: &[FF]) -> Self {
        let length = values.len();
        let mut mut_values = values.to_vec();

        intt(&mut mut_values, generator, length.ilog2());
        let poly = Polynomial::new(mut_values);

        poly.scale(offset.inverse())
    }

    /// Divide `self` by some `divisor`.
    ///
    /// # Panics
    ///
    /// Panics if the `divisor` is zero.
    pub fn divide(&self, divisor: &Self) -> Self {
        let quotient_degree = self.degree() - divisor.degree();
        if quotient_degree < Self::FAST_DIVIDE_CUTOFF_THRESHOLD {
            let (quotient, _) = self.naive_divide(divisor);
            quotient
        } else {
            self.fast_divide(divisor)
        }
    }

    /// Polynomial long division with `self` as the dividend, divided by some `divisor`.
    /// Only `pub` to allow benchmarking; not considered part of the public API.
    ///
    /// As the name implies, the advantage of this method over [`divide`](Self::naive_divide) is
    /// runtime complexity. Concretely, this method has time complexity in O(n·log(n)), whereas
    /// [`divide`](Self::naive_divide) has time complexity in O(n^2).
    #[doc(hidden)]
    pub fn fast_divide(&self, divisor: &Self) -> Self {
        // The math for this function: [0]. There are very slight deviations, for example around the
        // requirement that the divisor is monic.
        //
        // [0] https://cs.uwaterloo.ca/~r5olivei/courses/2021-winter-cs487/lecture5-post.pdf

        let Ok(quotient_degree) = usize::try_from(self.degree() - divisor.degree()) else {
            return Self::zero();
        };

        let divisor_lc = divisor.leading_coefficient();
        let divisor_lc_inv = divisor_lc.expect("divisor should be non-zero").inverse();

        // Reverse coefficient vectors to move into formal power series ring over FF, i.e., FF[[x]].
        // Re-interpret as a polynomial to benefit from the already-implemented multiplication
        // method, which mechanically work the same in FF[X] and FF[[x]].
        let reverse = |poly: &Self| Self::new(poly.coefficients.iter().copied().rev().collect());
        let rev_divisor = reverse(divisor);

        // Newton iteration to invert divisor up to required precision. Why is this the required
        // precision? Good question.
        // The initialization of `f` makes up for the divisor not necessarily being monic.
        let precision = (quotient_degree + 1).next_power_of_two();
        let num_rounds = precision.ilog2();
        let mut f = Self::from_constant(divisor_lc_inv);
        for _ in 0..num_rounds {
            f = f.scalar_mul(FF::from(2)) - rev_divisor.multiply(&f).multiply(&f);
        }
        let rev_divisor_inverse = f;

        let rev_quotient = reverse(self).multiply(&rev_divisor_inverse);
        reverse(&rev_quotient).truncate(quotient_degree)
    }

    /// The degree-`k` polynomial with the same `k + 1` leading coefficients as `self`. To be more
    /// precise: The degree of the result will be the minimum of `k` and [`Self::degree()`]. This
    /// implies, among other things, that if `self` [is zero](Self::is_zero()), the result will also
    /// be zero, independent of `k`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let f = Polynomial::new(bfe_vec![0, 1, 2, 3, 4]); // 4x⁴ + 3x³ + 2x² + 1x¹ + 0
    /// let g = f.truncate(2);                            // 4x² + 3x¹ + 2
    /// assert_eq!(Polynomial::new(bfe_vec![2, 3, 4]), g);
    /// ```
    pub fn truncate(&self, k: usize) -> Self {
        let coefficients = self.coefficients.iter().copied();
        let coefficients = coefficients.rev().take(k + 1).rev().collect();
        Self::new(coefficients)
    }

    /// `self % x^n`
    ///
    /// A special case of [Self::rem], and faster.
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let f = Polynomial::new(bfe_vec![0, 1, 2, 3, 4]); // 4x⁴ + 3x³ + 2x² + 1x¹ + 0
    /// let g = f.mod_x_to_the_n(2);                      // 1x¹ + 0
    /// assert_eq!(Polynomial::new(bfe_vec![0, 1]), g);
    /// ```
    pub fn mod_x_to_the_n(&self, n: usize) -> Self {
        let num_coefficients_to_retain = n.min(self.coefficients.len());
        Self::new(self.coefficients[..num_coefficients_to_retain].into())
    }
}

impl<const N: usize, FF, E> From<[E; N]> for Polynomial<FF>
where
    FF: FiniteField,
    E: Into<FF>,
{
    fn from(coefficients: [E; N]) -> Self {
        Self::new(coefficients.into_iter().map(|x| x.into()).collect())
    }
}

impl<FF, E> From<&[E]> for Polynomial<FF>
where
    FF: FiniteField,
    E: Into<FF> + Clone,
{
    fn from(coefficients: &[E]) -> Self {
        Self::from(coefficients.to_vec())
    }
}

impl<FF, E> From<Vec<E>> for Polynomial<FF>
where
    FF: FiniteField,
    E: Into<FF>,
{
    fn from(coefficients: Vec<E>) -> Self {
        Self::new(coefficients.into_iter().map(|c| c.into()).collect())
    }
}

impl<FF, E> From<&Vec<E>> for Polynomial<FF>
where
    FF: FiniteField,
    E: Into<FF> + Clone,
{
    fn from(coefficients: &Vec<E>) -> Self {
        Self::from(coefficients.to_vec())
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    pub const fn new(coefficients: Vec<FF>) -> Self {
        Self { coefficients }
    }

    pub fn normalize(&mut self) {
        while !self.coefficients.is_empty() && self.coefficients.last().unwrap().is_zero() {
            self.coefficients.pop();
        }
    }

    pub fn from_constant(constant: FF) -> Self {
        Self {
            coefficients: vec![constant],
        }
    }

    pub fn is_x(&self) -> bool {
        self.degree() == 1 && self.coefficients[0].is_zero() && self.coefficients[1].is_one()
    }

    pub fn evaluate(&self, &x: &FF) -> FF {
        let mut acc = FF::zero();
        for &c in self.coefficients.iter().rev() {
            acc = c + x * acc;
        }

        acc
    }

    /// The coefficient of the polynomial's term of highest power. `None` if (and only if) `self`
    /// [is zero](Self::is_zero).
    ///
    /// Furthermore, is never `Some(FF::zero())`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// # use num_traits::Zero;
    /// let f = Polynomial::new(bfe_vec![1, 2, 3]);
    /// assert_eq!(Some(bfe!(3)), f.leading_coefficient());
    /// assert_eq!(None, Polynomial::<XFieldElement>::zero().leading_coefficient());
    /// ```
    pub fn leading_coefficient(&self) -> Option<FF> {
        match self.degree() {
            -1 => None,
            n => Some(self.coefficients[n as usize]),
        }
    }

    pub fn are_colinear_3(p0: (FF, FF), p1: (FF, FF), p2: (FF, FF)) -> bool {
        if p0.0 == p1.0 || p1.0 == p2.0 || p2.0 == p0.0 {
            return false;
        }

        let dy = p0.1 - p1.1;
        let dx = p0.0 - p1.0;

        dx * (p2.1 - p0.1) == dy * (p2.0 - p0.0)
    }

    pub fn get_colinear_y(p0: (FF, FF), p1: (FF, FF), p2_x: FF) -> FF {
        assert_ne!(p0.0, p1.0, "Line must not be parallel to y-axis");
        let dy = p0.1 - p1.1;
        let dx = p0.0 - p1.0;
        let p2_y_times_dx = dy * (p2_x - p0.0) + dx * p0.1;

        // Can we implement this without division?
        p2_y_times_dx / dx
    }

    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn naive_zerofier(domain: &[FF]) -> Self {
        domain
            .iter()
            .map(|&r| Self::new(vec![-r, FF::one()]))
            .reduce(|accumulator, linear_poly| accumulator * linear_poly)
            .unwrap_or_else(Self::one)
    }

    /// Slow square implementation that does not use NTT
    #[must_use]
    pub fn slow_square(&self) -> Self {
        let degree = self.degree();
        if degree == -1 {
            return Self::zero();
        }

        let squared_coefficient_len = self.degree() as usize * 2 + 1;
        let zero = FF::zero();
        let one = FF::one();
        let two = one + one;
        let mut squared_coefficients = vec![zero; squared_coefficient_len];

        for i in 0..self.coefficients.len() {
            let ci = self.coefficients[i];
            squared_coefficients[2 * i] += ci * ci;

            // TODO: Review.
            for j in i + 1..self.coefficients.len() {
                let cj = self.coefficients[j];
                squared_coefficients[i + j] += two * ci * cj;
            }
        }

        Self {
            coefficients: squared_coefficients,
        }
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    pub fn are_colinear(points: &[(FF, FF)]) -> bool {
        if points.len() < 3 {
            return false;
        }

        if !points.iter().map(|(x, _)| x).all_unique() {
            return false;
        }

        // Find 1st degree polynomial through first two points
        let (p0_x, p0_y) = points[0];
        let (p1_x, p1_y) = points[1];
        let a = (p0_y - p1_y) / (p0_x - p1_x);
        let b = p0_y - a * p0_x;

        points.iter().skip(2).all(|&(x, y)| a * x + b == y)
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn naive_multiply(&self, other: &Self) -> Self {
        let Ok(degree_lhs) = usize::try_from(self.degree()) else {
            return Self::zero();
        };
        let Ok(degree_rhs) = usize::try_from(other.degree()) else {
            return Self::zero();
        };

        let mut product = vec![FF::zero(); degree_lhs + degree_rhs + 1];
        for i in 0..=degree_lhs {
            for j in 0..=degree_rhs {
                product[i + j] += self.coefficients[i] * other.coefficients[j];
            }
        }

        Self::new(product)
    }

    /// Multiply a polynomial with itself `pow` times
    #[must_use]
    pub fn mod_pow(&self, pow: BigInt) -> Self {
        let one = FF::one();

        // Special case to handle 0^0 = 1
        if pow.is_zero() {
            return Self::from_constant(one);
        }

        if self.is_zero() {
            return Self::zero();
        }

        let mut acc = Polynomial::from_constant(one);
        let bit_length: u64 = pow.bits();
        for i in 0..bit_length {
            acc = acc.slow_square();
            let set: bool =
                !(pow.clone() & Into::<BigInt>::into(1u128 << (bit_length - 1 - i))).is_zero();
            if set {
                acc = acc * self.clone();
            }
        }

        acc
    }

    pub fn shift_coefficients_mut(&mut self, power: usize, zero: FF) {
        self.coefficients.splice(0..0, vec![zero; power]);
    }

    /// Multiply a polynomial with x^power
    #[must_use]
    pub fn shift_coefficients(&self, power: usize) -> Self {
        let zero = FF::zero();

        let mut coefficients: Vec<FF> = self.coefficients.clone();
        coefficients.splice(0..0, vec![zero; power]);
        Self { coefficients }
    }

    // TODO: Review
    #[deprecated(since = "0.39.0", note = "use `.scalar_mul()` instead")]
    pub fn scalar_mul_mut(&mut self, scalar: FF) {
        for coefficient in &mut self.coefficients {
            *coefficient *= scalar;
        }
    }

    #[must_use]
    pub fn scalar_mul(&self, scalar: FF) -> Self {
        Self::new(self.coefficients.iter().map(|&c| c * scalar).collect())
    }

    /// Return (quotient, remainder). Prefer [`Self::divide()`].
    ///
    /// Only `pub` to allow benchmarking; not considered part of the public API.
    #[doc(hidden)]
    pub fn naive_divide(&self, divisor: &Self) -> (Self, Self) {
        let divisor_lc_inv = divisor
            .leading_coefficient()
            .expect("divisor should be non-zero")
            .inverse();

        let Ok(quotient_degree) = usize::try_from(self.degree() - divisor.degree()) else {
            // self.degree() < divisor.degree()
            return (Self::zero(), self.to_owned());
        };
        debug_assert!(!self.is_zero());

        // quotient is built from back to front, must be reversed later
        let mut rev_quotient = Vec::with_capacity(quotient_degree + 1);
        let mut remainder = self.clone();
        remainder.normalize();

        // The divisor is also iterated back to front.
        // It is normalized manually to avoid it being a `&mut` argument.
        let rev_divisor = divisor.coefficients.iter().rev();
        let normal_rev_divisor = rev_divisor.skip_while(|c| c.is_zero());

        for _ in 0..=quotient_degree {
            let remainder_lc = remainder.coefficients.pop().unwrap();
            let quotient_coeff = remainder_lc * divisor_lc_inv;
            rev_quotient.push(quotient_coeff);

            if quotient_coeff.is_zero() {
                continue;
            }

            // don't use `.degree()` to still count leading zeros in intermittent remainders
            let remainder_degree = remainder.coefficients.len().saturating_sub(1);

            // skip divisor's leading coefficient: it has already been dealt with
            for (i, &divisor_coeff) in normal_rev_divisor.clone().skip(1).enumerate() {
                remainder.coefficients[remainder_degree - i] -= quotient_coeff * divisor_coeff;
            }
        }

        rev_quotient.reverse();
        let quotient = Self::new(rev_quotient);

        (quotient, remainder)
    }
}

impl<FF: FiniteField> Div for Polynomial<FF> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let (quotient, _): (Self, Self) = self.naive_divide(&other);
        quotient
    }
}

impl<FF: FiniteField> Rem for Polynomial<FF> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        let (_, remainder): (Self, Self) = self.naive_divide(&other);
        remainder
    }
}

impl<FF: FiniteField> Add for Polynomial<FF> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let summed: Vec<FF> = self
            .coefficients
            .into_iter()
            .zip_longest(other.coefficients)
            .map(|a| match a {
                EitherOrBoth::Both(l, r) => l.to_owned() + r.to_owned(),
                EitherOrBoth::Left(l) => l.to_owned(),
                EitherOrBoth::Right(r) => r.to_owned(),
            })
            .collect();

        Self {
            coefficients: summed,
        }
    }
}

impl<FF: FiniteField> AddAssign for Polynomial<FF> {
    fn add_assign(&mut self, rhs: Self) {
        let rhs_len = rhs.coefficients.len();
        let self_len = self.coefficients.len();
        for i in 0..std::cmp::min(self_len, rhs_len) {
            self.coefficients[i] = self.coefficients[i] + rhs.coefficients[i];
        }

        if rhs_len > self_len {
            self.coefficients
                .append(&mut rhs.coefficients[self_len..].to_vec());
        }
    }
}

impl<FF: FiniteField> Sub for Polynomial<FF> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let coefficients = self
            .coefficients
            .into_iter()
            .zip_longest(other.coefficients)
            .map(|a| match a {
                EitherOrBoth::Both(l, r) => l - r,
                EitherOrBoth::Left(l) => l,
                EitherOrBoth::Right(r) => FF::zero() - r,
            })
            .collect();

        Self { coefficients }
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    /// Extended Euclidean algorithm with polynomials. Computes the greatest
    /// common divisor `gcd` as a monic polynomial, as well as the corresponding
    /// Bézout coefficients `a` and `b`, satisfying `gcd = a·x + b·y`
    ///
    /// # Example
    ///
    /// ```
    /// # use twenty_first::prelude::Polynomial;
    /// # use twenty_first::prelude::BFieldElement;
    /// let x = Polynomial::<BFieldElement>::from([1, 0, 1]);
    /// let y = Polynomial::<BFieldElement>::from([1, 1]);
    /// let (gcd, a, b) = Polynomial::xgcd(x.clone(), y.clone());
    /// assert_eq!(gcd, a * x + b * y);
    /// ```
    pub fn xgcd(mut x: Self, mut y: Self) -> (Self, Self, Self) {
        let (mut a_factor, mut a1) = (Self::one(), Self::zero());
        let (mut b_factor, mut b1) = (Self::zero(), Self::one());

        while !y.is_zero() {
            let quotient = x.clone() / y.clone();
            let remainder = x % y.clone();
            let c = a_factor - quotient.clone() * a1.clone();
            let d = b_factor - quotient * b1.clone();

            x = y;
            y = remainder;
            a_factor = a1;
            a1 = c;
            b_factor = b1;
            b1 = d;
        }

        // normalize result to ensure the gcd, _i.e._, `x` has leading coefficient 1
        let lc = x.leading_coefficient().unwrap_or_else(FF::one);
        let normalize = |poly: Self| poly.scalar_mul(lc.inverse());

        let [x, a, b] = [x, a_factor, b_factor].map(normalize);
        (x, a, b)
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    pub fn degree(&self) -> isize {
        let mut deg = self.coefficients.len() as isize - 1;
        while deg >= 0 && self.coefficients[deg as usize].is_zero() {
            deg -= 1;
        }

        deg // -1 for the zero polynomial
    }

    pub fn formal_derivative(&self) -> Self {
        // not `enumerate()`ing: `FiniteField` is trait-bound to `From<u64>` but not `From<usize>`
        let coefficients = (0..)
            .zip(&self.coefficients)
            .map(|(i, &coefficient)| FF::from(i) * coefficient)
            .skip(1)
            .collect();

        Self { coefficients }
    }
}

impl<FF: FiniteField> Mul for Polynomial<FF> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.naive_multiply(&other)
    }
}

impl<FF: FiniteField> Neg for Polynomial<FF> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.scalar_mul(-FF::one())
    }
}

#[cfg(test)]
mod test_polynomials {
    use proptest::collection::size_range;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::prelude::*;

    use super::*;

    impl proptest::arbitrary::Arbitrary for Polynomial<BFieldElement> {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb().boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    impl proptest::arbitrary::Arbitrary for Polynomial<XFieldElement> {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb().boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    #[test]
    fn polynomial_display_test() {
        let polynomial = |cs: &[u64]| Polynomial::<BFieldElement>::from(cs);

        assert_eq!("0", polynomial(&[]).to_string());
        assert_eq!("0", polynomial(&[0]).to_string());
        assert_eq!("0", polynomial(&[0, 0]).to_string());

        assert_eq!("1", polynomial(&[1]).to_string());
        assert_eq!("2", polynomial(&[2, 0]).to_string());
        assert_eq!("3", polynomial(&[3, 0, 0]).to_string());

        assert_eq!("x", polynomial(&[0, 1]).to_string());
        assert_eq!("2x", polynomial(&[0, 2]).to_string());
        assert_eq!("3x", polynomial(&[0, 3]).to_string());

        assert_eq!("5x + 2", polynomial(&[2, 5]).to_string());
        assert_eq!("9x + 7", polynomial(&[7, 9, 0, 0, 0]).to_string());

        assert_eq!("4x^4 + 3x^3", polynomial(&[0, 0, 0, 3, 4]).to_string());
        assert_eq!("2x^4 + 1", polynomial(&[1, 0, 0, 0, 2]).to_string());
    }

    #[proptest]
    fn leading_coefficient_of_zero_polynomial_is_none(#[strategy(0usize..30)] num_zeros: usize) {
        let coefficients = vec![BFieldElement::zero(); num_zeros];
        let polynomial = Polynomial { coefficients };
        prop_assert!(polynomial.leading_coefficient().is_none());
    }

    #[proptest]
    fn leading_coefficient_of_non_zero_polynomial_is_some(
        polynomial: Polynomial<BFieldElement>,
        leading_coefficient: BFieldElement,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let mut coefficients = polynomial.coefficients;
        coefficients.push(leading_coefficient);
        coefficients.extend(vec![BFieldElement::zero(); num_leading_zeros]);
        let polynomial_with_leading_zeros = Polynomial { coefficients };
        prop_assert_eq!(
            leading_coefficient,
            polynomial_with_leading_zeros.leading_coefficient().unwrap()
        );
    }

    #[test]
    fn normalizing_canonical_zero_polynomial_has_no_effect() {
        let mut zero_polynomial = Polynomial::<BFieldElement>::zero();
        zero_polynomial.normalize();
        assert_eq!(Polynomial::zero(), zero_polynomial);
    }

    #[proptest]
    fn spurious_leading_zeros_dont_affect_equality(
        polynomial: Polynomial<BFieldElement>,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let mut coefficients = polynomial.coefficients.clone();
        coefficients.extend(vec![BFieldElement::zero(); num_leading_zeros]);
        let polynomial_with_leading_zeros = Polynomial { coefficients };

        prop_assert_eq!(polynomial, polynomial_with_leading_zeros);
    }

    #[proptest]
    fn normalizing_removes_spurious_leading_zeros(
        polynomial: Polynomial<BFieldElement>,
        #[filter(!#leading_coefficient.is_zero())] leading_coefficient: BFieldElement,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let mut coefficients = polynomial.coefficients.clone();
        coefficients.push(leading_coefficient);
        coefficients.extend(vec![BFieldElement::zero(); num_leading_zeros]);
        let mut polynomial_with_leading_zeros = Polynomial { coefficients };
        polynomial_with_leading_zeros.normalize();

        let num_inserted_coefficients = 1;
        let expected_num_coefficients = polynomial.coefficients.len() + num_inserted_coefficients;
        let num_coefficients = polynomial_with_leading_zeros.coefficients.len();

        prop_assert_eq!(expected_num_coefficients, num_coefficients);
    }

    #[test]
    fn scaling_a_polynomial_works_with_different_fields_as_the_offset() {
        let bfe_poly = Polynomial::new(bfe_vec![0, 1, 2]);
        let _: Polynomial<BFieldElement> = bfe_poly.scale(bfe!(42));
        let _: Polynomial<XFieldElement> = bfe_poly.scale(bfe!(42));
        let _: Polynomial<XFieldElement> = bfe_poly.scale(xfe!(42));

        let xfe_poly = Polynomial::new(xfe_vec![0, 1, 2]);
        let _: Polynomial<XFieldElement> = xfe_poly.scale(bfe!(42));
        let _: Polynomial<XFieldElement> = xfe_poly.scale(xfe!(42));
    }

    #[proptest]
    fn slow_lagrange_interpolation(
        polynomial: Polynomial<BFieldElement>,
        #[strategy(Just(#polynomial.coefficients.len().max(1)))] _min_points: usize,
        #[any(size_range(#_min_points..8 * #_min_points).lift())] points: Vec<BFieldElement>,
    ) {
        let evaluations = points
            .into_iter()
            .map(|x| (x, polynomial.evaluate(&x)))
            .collect_vec();
        let interpolation_polynomial = Polynomial::lagrange_interpolate_zipped(&evaluations);
        prop_assert_eq!(polynomial, interpolation_polynomial);
    }

    #[proptest]
    fn three_colinear_points_are_colinear(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p2_x && #p1.0 != #p2_x)] p2_x: BFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let p2 = (p2_x, line.evaluate(&p2_x));
        prop_assert!(Polynomial::are_colinear_3(p0, p1, p2));
    }

    #[proptest]
    fn three_non_colinear_points_are_not_colinear(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p2_x && #p1.0 != #p2_x)] p2_x: BFieldElement,
        #[filter(!#disturbance.is_zero())] disturbance: BFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let p2 = (p2_x, line.evaluate(&p2_x) + disturbance);
        prop_assert!(!Polynomial::are_colinear_3(p0, p1, p2));
    }

    #[proptest]
    fn colinearity_check_needs_at_least_three_points(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
    ) {
        prop_assert!(!Polynomial::<BFieldElement>::are_colinear(&[]));
        prop_assert!(!Polynomial::are_colinear(&[p0]));
        prop_assert!(!Polynomial::are_colinear(&[p0, p1]));
    }

    #[proptest]
    fn colinearity_check_with_repeated_points_fails(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
    ) {
        prop_assert!(!Polynomial::are_colinear(&[p0, p1, p1]));
    }

    #[proptest]
    fn colinear_points_are_colinear(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
        #[filter(!#additional_points_xs.contains(&#p0.0))]
        #[filter(!#additional_points_xs.contains(&#p1.0))]
        #[filter(#additional_points_xs.iter().unique().count() == #additional_points_xs.len())]
        #[any(size_range(1..100).lift())]
        additional_points_xs: Vec<BFieldElement>,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let additional_points = additional_points_xs
            .into_iter()
            .map(|x| (x, line.evaluate(&x)))
            .collect_vec();
        let all_points = [p0, p1].into_iter().chain(additional_points).collect_vec();
        prop_assert!(Polynomial::are_colinear(&all_points));
    }

    #[test]
    #[should_panic(expected = "Line must not be parallel to y-axis")]
    fn getting_point_on_invalid_line_fails() {
        let one = BFieldElement::one();
        let two = one + one;
        let three = two + one;
        Polynomial::<BFieldElement>::get_colinear_y((one, one), (one, three), two);
    }

    #[proptest]
    fn point_on_line_and_colinear_point_are_identical(
        p0: (BFieldElement, BFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (BFieldElement, BFieldElement),
        x: BFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let y = line.evaluate(&x);
        let y_from_get_point_on_line = Polynomial::get_colinear_y(p0, p1, x);
        prop_assert_eq!(y, y_from_get_point_on_line);
    }

    #[proptest]
    fn point_on_line_and_colinear_point_are_identical_in_extension_field(
        p0: (XFieldElement, XFieldElement),
        #[filter(#p0.0 != #p1.0)] p1: (XFieldElement, XFieldElement),
        x: XFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let y = line.evaluate(&x);
        let y_from_get_point_on_line = Polynomial::get_colinear_y(p0, p1, x);
        prop_assert_eq!(y, y_from_get_point_on_line);
    }

    #[proptest]
    fn shifting_polynomial_coefficients_by_zero_is_the_same_as_not_shifting_it(
        poly: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(poly.clone(), poly.shift_coefficients(0));
    }

    #[proptest]
    fn shifting_polynomial_one_is_equivalent_to_raising_polynomial_x_to_the_power_of_the_shift(
        #[strategy(0usize..30)] shift: usize,
    ) {
        let shifted_one = Polynomial::one().shift_coefficients(shift);
        let x_to_the_shift = Polynomial::<BFieldElement>::from([0, 1]).mod_pow(shift.into());
        prop_assert_eq!(shifted_one, x_to_the_shift);
    }

    #[test]
    fn polynomial_shift_test() {
        let polynomial = Polynomial::<BFieldElement>::from([17, 14]);
        assert_eq!(
            bfe_vec![17, 14],
            polynomial.shift_coefficients(0).coefficients
        );
        assert_eq!(
            bfe_vec![0, 17, 14],
            polynomial.shift_coefficients(1).coefficients
        );
        assert_eq!(
            bfe_vec![0, 0, 0, 0, 17, 14],
            polynomial.shift_coefficients(4).coefficients
        );
    }

    #[proptest]
    fn shifting_a_polynomial_means_prepending_zeros_to_its_coefficients(
        polynomial: Polynomial<BFieldElement>,
        #[strategy(0usize..30)] shift: usize,
    ) {
        let shifted_polynomial = polynomial.shift_coefficients(shift);
        let mut expected_coefficients = vec![BFieldElement::zero(); shift];
        expected_coefficients.extend(polynomial.coefficients);
        prop_assert_eq!(expected_coefficients, shifted_polynomial.coefficients);
    }

    #[proptest]
    fn any_polynomial_to_the_power_of_zero_is_one(poly: Polynomial<BFieldElement>) {
        let poly_to_the_zero = poly.mod_pow(0.into());
        prop_assert_eq!(Polynomial::one(), poly_to_the_zero);
    }

    #[proptest]
    fn any_polynomial_to_the_power_one_is_itself(poly: Polynomial<BFieldElement>) {
        let poly_to_the_one = poly.mod_pow(1.into());
        prop_assert_eq!(poly, poly_to_the_one);
    }

    #[proptest]
    fn polynomial_one_to_any_power_is_one(#[strategy(0u64..30)] exponent: u64) {
        let one_to_the_exponent = Polynomial::<BFieldElement>::one().mod_pow(exponent.into());
        prop_assert_eq!(Polynomial::one(), one_to_the_exponent);
    }

    #[test]
    fn mod_pow_test() {
        let polynomial = |cs: &[u64]| Polynomial::<BFieldElement>::from(cs);

        let pol = polynomial(&[0, 14, 0, 4, 0, 8, 0, 3]);
        let pol_squared = polynomial(&[0, 0, 196, 0, 112, 0, 240, 0, 148, 0, 88, 0, 48, 0, 9]);
        let pol_cubed = polynomial(&[
            0, 0, 0, 2744, 0, 2352, 0, 5376, 0, 4516, 0, 4080, 0, 2928, 0, 1466, 0, 684, 0, 216, 0,
            27,
        ]);

        assert_eq!(pol_squared, pol.mod_pow(2.into()));
        assert_eq!(pol_cubed, pol.mod_pow(3.into()));

        let parabola = polynomial(&[5, 41, 19]);
        let parabola_squared = polynomial(&[25, 410, 1871, 1558, 361]);
        assert_eq!(parabola_squared, parabola.mod_pow(2.into()));
    }

    #[proptest]
    fn mod_pow_arbitrary_test(
        poly: Polynomial<BFieldElement>,
        #[strategy(0u32..15)] exponent: u32,
    ) {
        let actual = poly.mod_pow(exponent.into());
        let fast_actual = poly.fast_mod_pow(exponent.into());
        let mut expected = Polynomial::one();
        for _ in 0..exponent {
            expected = expected.clone() * poly.clone();
        }

        prop_assert_eq!(expected.clone(), actual);
        prop_assert_eq!(expected, fast_actual);
    }

    #[proptest]
    fn polynomial_zero_is_neutral_element_for_addition(a: Polynomial<BFieldElement>) {
        prop_assert_eq!(a.clone() + Polynomial::zero(), a.clone());
        prop_assert_eq!(Polynomial::zero() + a.clone(), a);
    }

    #[proptest]
    fn polynomial_one_is_neutral_element_for_multiplication(a: Polynomial<BFieldElement>) {
        prop_assert_eq!(a.clone() * Polynomial::one(), a.clone());
        prop_assert_eq!(Polynomial::one() * a.clone(), a);
    }

    #[proptest]
    fn multiplication_by_zero_is_zero(a: Polynomial<BFieldElement>) {
        prop_assert_eq!(Polynomial::zero(), a.clone() * Polynomial::zero());
        prop_assert_eq!(Polynomial::zero(), Polynomial::zero() * a.clone());
    }

    #[proptest]
    fn polynomial_addition_is_commutative(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.clone() + b.clone(), b + a);
    }

    #[proptest]
    fn polynomial_multiplication_is_commutative(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.clone() * b.clone(), b * a);
    }

    #[proptest]
    fn polynomial_addition_is_associative(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
        c: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!((a.clone() + b.clone()) + c.clone(), a + (b + c));
    }

    #[proptest]
    fn polynomial_multiplication_is_associative(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
        c: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!((a.clone() * b.clone()) * c.clone(), a * (b * c));
    }

    #[proptest]
    fn polynomial_multiplication_is_distributive(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
        c: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(
            (a.clone() + b.clone()) * c.clone(),
            (a * c.clone()) + (b * c)
        );
    }

    #[proptest]
    fn polynomial_subtraction_of_self_is_zero(a: Polynomial<BFieldElement>) {
        prop_assert_eq!(Polynomial::zero(), a.clone() - a);
    }

    #[proptest]
    fn polynomial_division_by_self_is_one(#[filter(!#a.is_zero())] a: Polynomial<BFieldElement>) {
        prop_assert_eq!(Polynomial::one(), a.clone() / a);
    }

    #[proptest]
    fn polynomial_division_removes_common_factors(
        a: Polynomial<BFieldElement>,
        #[filter(!#b.is_zero())] b: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.clone(), a * b.clone() / b);
    }

    #[proptest]
    fn polynomial_multiplication_raises_degree_at_maximum_to_sum_of_degrees(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        let sum_of_degrees = (a.degree() + b.degree()).max(-1);
        prop_assert!((a * b).degree() <= sum_of_degrees);
    }

    #[test]
    fn leading_zeros_dont_affect_polynomial_division() {
        // This test was used to catch a bug where the polynomial division was wrong when the
        // divisor has a leading zero coefficient, i.e. when it was not normalized

        let polynomial = |cs: &[u64]| Polynomial::<BFieldElement>::from(cs);

        // x^3 - x + 1 / y = x
        let numerator = polynomial(&[1, BFieldElement::P - 1, 0, 1]);
        let numerator_with_leading_zero = polynomial(&[1, BFieldElement::P - 1, 0, 1, 0]);

        let divisor_normalized = polynomial(&[0, 1]);
        let divisor_not_normalized = polynomial(&[0, 1, 0]);
        let divisor_more_leading_zeros = polynomial(&[0, 1, 0, 0, 0, 0, 0, 0, 0]);

        let expected = polynomial(&[BFieldElement::P - 1, 0, 1]);

        // Verify that the divisor need not be normalized
        assert_eq!(expected, numerator.clone() / divisor_normalized.clone());
        assert_eq!(expected, numerator.clone() / divisor_not_normalized.clone());
        assert_eq!(expected, numerator / divisor_more_leading_zeros.clone());

        // Verify that numerator need not be normalized
        let res_numerator_not_normalized_0 =
            numerator_with_leading_zero.clone() / divisor_normalized;
        let res_numerator_not_normalized_1 =
            numerator_with_leading_zero.clone() / divisor_not_normalized;
        let res_numerator_not_normalized_2 =
            numerator_with_leading_zero / divisor_more_leading_zeros;
        assert_eq!(expected, res_numerator_not_normalized_0);
        assert_eq!(expected, res_numerator_not_normalized_1);
        assert_eq!(expected, res_numerator_not_normalized_2);
    }

    #[proptest]
    fn leading_coefficient_of_truncated_polynomial_is_same_as_original_leading_coefficient(
        poly: Polynomial<BFieldElement>,
        #[strategy(..50_usize)] truncation_point: usize,
    ) {
        let Some(lc) = poly.leading_coefficient() else {
            let reason = "test is only sensible if polynomial has a leading coefficient";
            return Err(TestCaseError::Reject(reason.into()));
        };
        let truncated_poly = poly.truncate(truncation_point);
        let Some(trunc_lc) = truncated_poly.leading_coefficient() else {
            let reason = "test is only sensible if truncated polynomial has a leading coefficient";
            return Err(TestCaseError::Reject(reason.into()));
        };
        prop_assert_eq!(lc, trunc_lc);
    }

    #[proptest]
    fn truncated_polynomial_is_of_degree_min_of_truncation_point_and_poly_degree(
        poly: Polynomial<BFieldElement>,
        #[strategy(..50_usize)] truncation_point: usize,
    ) {
        let expected_degree = poly.degree().min(truncation_point.try_into().unwrap());
        prop_assert_eq!(expected_degree, poly.truncate(truncation_point).degree());
    }

    #[proptest]
    fn truncating_zero_polynomial_gives_zero_polynomial(
        #[strategy(..50_usize)] truncation_point: usize,
    ) {
        let poly = Polynomial::<BFieldElement>::zero().truncate(truncation_point);
        prop_assert!(poly.is_zero());
    }

    #[proptest]
    fn truncation_negates_degree_shifting(
        #[strategy(0_usize..30)] shift: usize,
        #[strategy(..50_usize)] truncation_point: usize,
        #[filter(#poly.degree() >= #truncation_point as isize)] poly: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(
            poly.truncate(truncation_point),
            poly.shift_coefficients(shift).truncate(truncation_point)
        );
    }

    #[proptest]
    fn zero_polynomial_mod_any_power_of_x_is_zero_polynomial(power: usize) {
        let must_be_zero = Polynomial::<BFieldElement>::zero().mod_x_to_the_n(power);
        prop_assert!(must_be_zero.is_zero());
    }

    #[proptest]
    fn polynomial_mod_some_power_of_x_results_in_polynomial_of_degree_one_less_than_power(
        #[filter(!#poly.is_zero())] poly: Polynomial<BFieldElement>,
        #[strategy(..=usize::try_from(#poly.degree()).unwrap())] power: usize,
    ) {
        let remainder = poly.mod_x_to_the_n(power);
        prop_assert_eq!(isize::try_from(power).unwrap() - 1, remainder.degree());
    }

    #[proptest]
    fn polynomial_mod_some_power_of_x_shares_low_degree_terms_coefficients_with_original_polynomial(
        #[filter(!#poly.is_zero())] poly: Polynomial<BFieldElement>,
        power: usize,
    ) {
        let remainder = poly.mod_x_to_the_n(power);
        let min_num_coefficients = poly.coefficients.len().min(remainder.coefficients.len());
        prop_assert_eq!(
            &poly.coefficients[..min_num_coefficients],
            &remainder.coefficients[..min_num_coefficients]
        );
    }

    #[proptest]
    fn fast_multiplication_by_zero_gives_zero(poly: Polynomial<BFieldElement>) {
        let product = poly.fast_multiply(&Polynomial::zero());
        prop_assert_eq!(Polynomial::zero(), product);
    }

    #[proptest]
    fn fast_multiplication_by_one_gives_self(poly: Polynomial<BFieldElement>) {
        let product = poly.fast_multiply(&Polynomial::one());
        prop_assert_eq!(poly, product);
    }

    #[proptest]
    fn fast_multiplication_is_commutative(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.fast_multiply(&b), b.fast_multiply(&a));
    }

    #[proptest]
    fn fast_multiplication_and_normal_multiplication_are_equivalent(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        let product = a.fast_multiply(&b);
        prop_assert_eq!(a * b, product);
    }

    #[proptest(cases = 50)]
    fn naive_zerofier_and_fast_zerofier_are_identical(
        #[any(size_range(..Polynomial::<BFieldElement>::FAST_ZEROFIER_CUTOFF_THRESHOLD * 2).lift())]
        roots: Vec<BFieldElement>,
    ) {
        let naive_zerofier = Polynomial::naive_zerofier(&roots);
        let fast_zerofier = Polynomial::fast_zerofier(&roots);
        prop_assert_eq!(naive_zerofier, fast_zerofier);
    }

    #[proptest(cases = 50)]
    fn smart_zerofier_and_fast_zerofier_are_identical(
        #[any(size_range(..Polynomial::<BFieldElement>::FAST_ZEROFIER_CUTOFF_THRESHOLD * 2).lift())]
        roots: Vec<BFieldElement>,
    ) {
        let smart_zerofier = Polynomial::smart_zerofier(&roots);
        let fast_zerofier = Polynomial::fast_zerofier(&roots);
        prop_assert_eq!(smart_zerofier, fast_zerofier);
    }

    #[proptest(cases = 50)]
    fn zerofier_and_naive_zerofier_are_identical(
        #[any(size_range(..Polynomial::<BFieldElement>::FAST_ZEROFIER_CUTOFF_THRESHOLD * 2).lift())]
        roots: Vec<BFieldElement>,
    ) {
        let zerofier = Polynomial::zerofier(&roots);
        let naive_zerofier = Polynomial::naive_zerofier(&roots);
        prop_assert_eq!(zerofier, naive_zerofier);
    }

    #[proptest(cases = 50)]
    fn zerofier_is_zero_only_on_domain(
        #[any(size_range(..1024).lift())] domain: Vec<BFieldElement>,
        #[filter(#out_of_domain_points.iter().all(|p| !#domain.contains(p)))]
        out_of_domain_points: Vec<BFieldElement>,
    ) {
        let zerofier = Polynomial::zerofier(&domain);
        for point in domain {
            prop_assert_eq!(BFieldElement::zero(), zerofier.evaluate(&point));
        }
        for point in out_of_domain_points {
            prop_assert_ne!(BFieldElement::zero(), zerofier.evaluate(&point));
        }
    }

    #[proptest]
    fn zerofier_has_leading_coefficient_one(domain: Vec<BFieldElement>) {
        let zerofier = Polynomial::zerofier(&domain);
        prop_assert_eq!(
            BFieldElement::one(),
            zerofier.leading_coefficient().unwrap()
        );
    }

    #[test]
    fn fast_evaluate_on_hardcoded_domain_and_polynomial() {
        // x^5 + x^3
        let poly = Polynomial::<BFieldElement>::from([0, 0, 0, 1, 0, 1]);
        let domain = [6, 12].map(BFieldElement::new);
        let evaluation = poly.fast_evaluate(&domain);

        let expected_0 = domain[0].mod_pow(5u64) + domain[0].mod_pow(3u64);
        assert_eq!(expected_0, evaluation[0]);

        let expected_1 = domain[1].mod_pow(5u64) + domain[1].mod_pow(3u64);
        assert_eq!(expected_1, evaluation[1]);
    }

    #[proptest]
    fn slow_and_fast_polynomial_evaluation_are_equivalent(
        poly: Polynomial<BFieldElement>,
        #[any(size_range(..1024).lift())] domain: Vec<BFieldElement>,
    ) {
        let evaluations = domain.iter().map(|x| poly.evaluate(x)).collect_vec();
        let fast_evaluations = poly.fast_evaluate(&domain);
        prop_assert_eq!(evaluations, fast_evaluations);
    }

    #[test]
    #[should_panic(expected = "zero points")]
    fn lagrange_interpolation_through_no_points_is_impossible() {
        let _ = Polynomial::<BFieldElement>::lagrange_interpolate(&[], &[]);
    }

    #[proptest]
    fn interpolating_through_one_point_gives_constant_polynomial(
        x: BFieldElement,
        y: BFieldElement,
    ) {
        let interpolant = Polynomial::lagrange_interpolate(&[x], &[y]);
        let polynomial = Polynomial::from_constant(y);
        prop_assert_eq!(polynomial, interpolant);
    }

    #[test]
    #[should_panic(expected = "zero points")]
    fn fast_interpolation_through_no_points_is_impossible() {
        let _ = Polynomial::<BFieldElement>::fast_interpolate(&[], &[]);
    }

    #[proptest(cases = 10)]
    fn lagrange_and_fast_interpolation_are_identical(
        #[any(size_range(1..2048).lift())]
        #[filter(#domain.iter().unique().count() == #domain.len())]
        domain: Vec<BFieldElement>,
        #[strategy(vec(arb(), #domain.len()))] values: Vec<BFieldElement>,
    ) {
        let lagrange_interpolant = Polynomial::lagrange_interpolate(&domain, &values);
        let fast_interpolant = Polynomial::fast_interpolate(&domain, &values);
        prop_assert_eq!(lagrange_interpolant, fast_interpolant);
    }

    #[proptest(cases = 20)]
    fn interpolation_then_evaluation_is_identity(
        #[any(size_range(1..2048).lift())]
        #[filter(#domain.iter().unique().count() == #domain.len())]
        domain: Vec<BFieldElement>,
        #[strategy(vec(arb(), #domain.len()))] values: Vec<BFieldElement>,
    ) {
        let interpolant = Polynomial::fast_interpolate(&domain, &values);
        let evaluations = interpolant.fast_evaluate(&domain);
        prop_assert_eq!(values, evaluations);
    }

    #[proptest(cases = 1)]
    fn fast_batch_interpolation_is_equivalent_to_fast_interpolation(
        #[any(size_range(1..2048).lift())]
        #[filter(#domain.iter().unique().count() == #domain.len())]
        domain: Vec<BFieldElement>,
        #[strategy(vec(vec(arb(), #domain.len()), 0..10))] value_vecs: Vec<Vec<BFieldElement>>,
    ) {
        let root_order = domain.len().next_power_of_two();
        let root_of_unity = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();

        let interpolants = value_vecs
            .iter()
            .map(|values| Polynomial::fast_interpolate(&domain, values))
            .collect_vec();

        let batched_interpolants =
            Polynomial::batch_fast_interpolate(&domain, &value_vecs, root_of_unity, root_order);
        prop_assert_eq!(interpolants, batched_interpolants);
    }

    fn coset_domain_of_size_from_generator_with_offset(
        size: usize,
        generator: BFieldElement,
        offset: BFieldElement,
    ) -> Vec<BFieldElement> {
        let mut domain = vec![offset];
        for _ in 1..size {
            domain.push(domain.last().copied().unwrap() * generator);
        }
        domain
    }

    #[proptest]
    fn fast_coset_evaluation_and_fast_evaluation_on_coset_are_identical(
        polynomial: Polynomial<BFieldElement>,
        offset: BFieldElement,
        #[strategy(0..8usize)]
        #[map(|x: usize| 1 << x)]
        // due to current limitation in `Polynomial::fast_coset_evaluate`
        #[filter((#root_order as isize) > #polynomial.degree())]
        root_order: usize,
    ) {
        let root_of_unity = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();
        let domain =
            coset_domain_of_size_from_generator_with_offset(root_order, root_of_unity, offset);

        let fast_values = polynomial.fast_evaluate(&domain);
        let fast_coset_values = polynomial.fast_coset_evaluate(offset, root_of_unity, root_order);
        prop_assert_eq!(fast_values, fast_coset_values);
    }

    #[proptest]
    fn fast_coset_interpolation_and_and_fast_interpolation_on_coset_are_identical(
        #[filter(!#offset.is_zero())] offset: BFieldElement,
        #[strategy(1..8usize)]
        #[map(|x: usize| 1 << x)]
        root_order: usize,
        #[strategy(vec(arb(), #root_order))] values: Vec<BFieldElement>,
    ) {
        let root_of_unity = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();
        let domain =
            coset_domain_of_size_from_generator_with_offset(root_order, root_of_unity, offset);

        let fast_interpolant = Polynomial::fast_interpolate(&domain, &values);
        let fast_coset_interpolant =
            Polynomial::fast_coset_interpolate(offset, root_of_unity, &values);
        prop_assert_eq!(fast_interpolant, fast_coset_interpolant);
    }

    #[proptest]
    fn naive_division_gives_quotient_and_remainder_with_expected_properties(
        a: Polynomial<BFieldElement>,
        #[filter(!#b.is_zero())] b: Polynomial<BFieldElement>,
    ) {
        let (quot, rem) = a.naive_divide(&b);
        prop_assert!(rem.degree() < b.degree());
        prop_assert_eq!(a, quot * b + rem);
    }

    #[proptest]
    fn clean_naive_division_gives_quotient_and_remainder_with_expected_properties(
        #[filter(!#a_roots.is_empty())] a_roots: Vec<BFieldElement>,
        #[strategy(vec(0..#a_roots.len(), 1..=#a_roots.len()))]
        #[filter(#b_root_indices.iter().all_unique())]
        b_root_indices: Vec<usize>,
    ) {
        let b_roots = b_root_indices.into_iter().map(|i| a_roots[i]).collect_vec();
        let a = Polynomial::zerofier(&a_roots);
        let b = Polynomial::zerofier(&b_roots);
        let (quot, rem) = a.naive_divide(&b);
        prop_assert!(rem.is_zero());
        prop_assert_eq!(a, quot * b);
    }

    #[proptest]
    fn naive_division_and_fast_division_are_equivalent(
        a: Polynomial<BFieldElement>,
        #[filter(!#b.is_zero())] b: Polynomial<BFieldElement>,
    ) {
        let quotient = a.fast_divide(&b);
        prop_assert_eq!(a / b, quotient);
    }

    #[proptest]
    fn dividing_constant_polynomials_is_equivalent_to_dividing_constants(
        a: BFieldElement,
        #[filter(!#b.is_zero())] b: BFieldElement,
    ) {
        let a_poly = Polynomial::from_constant(a);
        let b_poly = Polynomial::from_constant(b);
        let expected_quotient = Polynomial::from_constant(a / b);
        prop_assert_eq!(expected_quotient, a_poly / b_poly);
    }

    #[proptest]
    fn dividing_any_polynomial_by_a_constant_polynomial_results_in_remainder_zero(
        a: Polynomial<BFieldElement>,
        #[filter(!#b.is_zero())] b: BFieldElement,
    ) {
        let b_poly = Polynomial::from_constant(b);
        let (_, remainder) = a.naive_divide(&b_poly);
        prop_assert_eq!(Polynomial::zero(), remainder);
    }

    #[test]
    fn polynomial_division_by_and_with_shah_polynomial() {
        let polynomial = |cs: &[u64]| Polynomial::<BFieldElement>::from(cs);

        let shah = XFieldElement::shah_polynomial();
        let x_to_the_3 = polynomial(&[1]).shift_coefficients(3);
        let (shah_div_x_to_the_3, shah_mod_x_to_the_3) = shah.naive_divide(&x_to_the_3);
        assert_eq!(polynomial(&[1]), shah_div_x_to_the_3);
        assert_eq!(polynomial(&[1, BFieldElement::P - 1]), shah_mod_x_to_the_3);

        let x_to_the_6 = polynomial(&[1]).shift_coefficients(6);
        let (x_to_the_6_div_shah, x_to_the_6_mod_shah) = x_to_the_6.naive_divide(&shah);

        // x^3 + x - 1
        let expected_quot = polynomial(&[BFieldElement::P - 1, 1, 0, 1]);
        assert_eq!(expected_quot, x_to_the_6_div_shah);

        // x^2 - 2x + 1
        let expected_rem = polynomial(&[1, BFieldElement::P - 2, 1]);
        assert_eq!(expected_rem, x_to_the_6_mod_shah);
    }

    #[test]
    fn xgcd_does_not_panic_on_input_zero() {
        let zero = Polynomial::<BFieldElement>::zero;
        let (gcd, a, b) = Polynomial::xgcd(zero(), zero());
        assert_eq!(zero(), gcd);
        println!("a = {a}");
        println!("b = {b}");
    }

    #[proptest]
    fn xgcd_b_field_pol_test(x: Polynomial<BFieldElement>, y: Polynomial<BFieldElement>) {
        let (gcd, a, b) = Polynomial::xgcd(x.clone(), y.clone());
        // Bezout relation
        prop_assert_eq!(gcd, a * x + b * y);
    }

    #[proptest]
    fn xgcd_x_field_pol_test(x: Polynomial<XFieldElement>, y: Polynomial<XFieldElement>) {
        let (gcd, a, b) = Polynomial::xgcd(x.clone(), y.clone());
        // Bezout relation
        prop_assert_eq!(gcd, a * x + b * y);
    }

    #[proptest]
    fn add_assign_is_equivalent_to_adding_and_assigning(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        let mut c = a.clone();
        c += b.clone();
        prop_assert_eq!(a + b, c);
    }

    #[test]
    fn only_monic_polynomial_of_degree_1_is_x() {
        let polynomial = |cs: &[u64]| Polynomial::<BFieldElement>::from(cs);

        assert!(polynomial(&[0, 1]).is_x());
        assert!(polynomial(&[0, 1, 0]).is_x());
        assert!(polynomial(&[0, 1, 0, 0]).is_x());

        assert!(!polynomial(&[]).is_x());
        assert!(!polynomial(&[0]).is_x());
        assert!(!polynomial(&[1]).is_x());
        assert!(!polynomial(&[1, 0]).is_x());
        assert!(!polynomial(&[0, 2]).is_x());
        assert!(!polynomial(&[0, 0, 1]).is_x());
    }

    #[test]
    fn hardcoded_polynomial_squaring() {
        let polynomial = |cs: &[u64]| Polynomial::<BFieldElement>::from(cs);

        assert_eq!(Polynomial::zero(), polynomial(&[]).square());

        let x_plus_1 = polynomial(&[1, 1]);
        assert_eq!(polynomial(&[1, 2, 1]), x_plus_1.square());

        let x_to_the_15 = polynomial(&[1]).shift_coefficients(15);
        let x_to_the_30 = polynomial(&[1]).shift_coefficients(30);
        assert_eq!(x_to_the_30, x_to_the_15.square());

        let some_poly = polynomial(&[14, 1, 3, 4]);
        assert_eq!(
            polynomial(&[196, 28, 85, 118, 17, 24, 16]),
            some_poly.square()
        );
    }

    #[proptest]
    fn polynomial_squaring_is_equivalent_to_multiplication_with_self(
        poly: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(poly.clone() * poly.clone(), poly.square());
    }

    #[proptest]
    fn slow_and_normal_squaring_are_equivalent(poly: Polynomial<BFieldElement>) {
        prop_assert_eq!(poly.slow_square(), poly.square());
    }

    #[proptest]
    fn normal_and_fast_squaring_are_equivalent(poly: Polynomial<BFieldElement>) {
        prop_assert_eq!(poly.square(), poly.fast_square());
    }

    #[test]
    fn constant_zero_eq_constant_zero() {
        let zero_polynomial1 = Polynomial::<BFieldElement>::zero();
        let zero_polynomial2 = Polynomial::<BFieldElement>::zero();

        assert_eq!(zero_polynomial1, zero_polynomial2)
    }

    #[test]
    fn zero_polynomial_is_zero() {
        assert!(Polynomial::<BFieldElement>::zero().is_zero());
        assert!(Polynomial::<XFieldElement>::zero().is_zero());
    }

    #[proptest]
    fn zero_polynomial_is_zero_independent_of_spurious_leading_zeros(
        #[strategy(..500usize)] num_zeros: usize,
    ) {
        let coefficients = vec![0; num_zeros];
        prop_assert_eq!(
            Polynomial::zero(),
            Polynomial::<BFieldElement>::from(coefficients)
        );
    }

    #[proptest]
    fn no_constant_polynomial_with_non_zero_coefficient_is_zero(
        #[filter(!#constant.is_zero())] constant: BFieldElement,
    ) {
        let constant_polynomial = Polynomial::from_constant(constant);
        prop_assert!(!constant_polynomial.is_zero());
    }

    #[test]
    fn constant_one_eq_constant_one() {
        let one_polynomial1 = Polynomial::<BFieldElement>::one();
        let one_polynomial2 = Polynomial::<BFieldElement>::one();

        assert_eq!(one_polynomial1, one_polynomial2)
    }

    #[test]
    fn one_polynomial_is_one() {
        assert!(Polynomial::<BFieldElement>::one().is_one());
        assert!(Polynomial::<XFieldElement>::one().is_one());
    }

    #[proptest]
    fn one_polynomial_is_one_independent_of_spurious_leading_zeros(
        #[strategy(..500usize)] num_leading_zeros: usize,
    ) {
        let spurious_leading_zeros = vec![0; num_leading_zeros];
        let mut coefficients = vec![1];
        coefficients.extend(spurious_leading_zeros);
        prop_assert_eq!(
            Polynomial::one(),
            Polynomial::<BFieldElement>::from(coefficients)
        );
    }

    #[proptest]
    fn no_constant_polynomial_with_non_one_coefficient_is_one(
        #[filter(!#constant.is_one())] constant: BFieldElement,
    ) {
        let constant_polynomial = Polynomial::from_constant(constant);
        prop_assert!(!constant_polynomial.is_one());
    }

    #[test]
    fn formal_derivative_of_zero_is_zero() {
        assert!(Polynomial::<BFieldElement>::zero()
            .formal_derivative()
            .is_zero());
        assert!(Polynomial::<XFieldElement>::zero()
            .formal_derivative()
            .is_zero());
    }

    #[proptest]
    fn formal_derivative_of_constant_polynomial_is_zero(constant: BFieldElement) {
        let formal_derivative = Polynomial::from_constant(constant).formal_derivative();
        prop_assert!(formal_derivative.is_zero());
    }

    #[proptest]
    fn formal_derivative_of_non_zero_polynomial_is_of_degree_one_less_than_the_polynomial(
        #[filter(!#poly.is_zero())] poly: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(poly.degree() - 1, poly.formal_derivative().degree());
    }

    #[proptest]
    fn formal_derivative_of_product_adheres_to_the_leibniz_product_rule(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        let product_formal_derivative = (a.clone() * b.clone()).formal_derivative();
        let product_rule = a.formal_derivative() * b.clone() + a * b.formal_derivative();
        prop_assert_eq!(product_rule, product_formal_derivative);
    }
}
