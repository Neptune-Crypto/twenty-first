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
use std::ops::Rem;
use std::ops::Sub;

use arbitrary::Arbitrary;
use itertools::EitherOrBoth;
use itertools::Itertools;
use num_bigint::BigInt;
use num_traits::One;
use num_traits::Zero;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use rayon::prelude::ParallelIterator;

use crate::shared_math::ntt::intt;
use crate::shared_math::ntt::ntt;
use crate::shared_math::traits::FiniteField;
use crate::shared_math::traits::ModPowU32;

use super::b_field_element::BFieldElement;
use super::b_field_element::BFIELD_ONE;
use super::other;
use super::traits::Inverse;
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
    /// Fast division ([`Self::fast_divide`] and [`Self::fast_coset_divide`]) is slower for
    /// polynomials of degree less than this threshold.
    /// todo: Benchmark and find the optimal value.
    const FAST_DIVIDE_CUTOFF_THRESHOLD: isize = 8;

    /// Return the polynomial which corresponds to the transformation `x -> alpha * x`
    /// Given a polynomial P(x), produce P'(x) := P(alpha * x). Evaluating P'(x) then corresponds to
    /// evaluating P(alpha * x).
    #[must_use]
    pub fn scale(&self, alpha: BFieldElement) -> Self {
        let mut acc = FF::one();
        let mut return_coefficients = self.coefficients.clone();
        for elem in return_coefficients.iter_mut() {
            *elem *= acc;
            acc *= alpha;
        }

        Self::new(return_coefficients)
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

    // FIXME: lhs -> &self. FIXME: Change root_order: usize into : u32.
    pub fn fast_multiply(
        lhs: &Self,
        rhs: &Self,
        primitive_root: BFieldElement,
        root_order: usize,
    ) -> Self {
        assert!(
            primitive_root.mod_pow_u32(root_order as u32).is_one(),
            "provided primitive root must have the provided power."
        );
        assert!(
            !primitive_root.mod_pow_u32(root_order as u32 / 2).is_one(),
            "provided primitive root must be primitive in the right power."
        );

        if lhs.is_zero() || rhs.is_zero() {
            return Self::zero();
        }

        let mut root: BFieldElement = primitive_root.to_owned();
        let mut order = root_order;
        let lhs_degree = lhs.degree() as usize;
        let rhs_degree = rhs.degree() as usize;
        let degree = lhs_degree + rhs_degree;

        if degree < 8 {
            return lhs.to_owned() * rhs.to_owned();
        }

        while degree < order / 2 {
            root *= root;
            order /= 2;
        }

        let mut lhs_coefficients: Vec<FF> = lhs.coefficients[0..lhs_degree + 1].to_vec();
        let mut rhs_coefficients: Vec<FF> = rhs.coefficients[0..rhs_degree + 1].to_vec();
        while lhs_coefficients.len() < order {
            lhs_coefficients.push(FF::zero());
        }
        while rhs_coefficients.len() < order {
            rhs_coefficients.push(FF::zero());
        }

        let lhs_log_2_of_n = lhs_coefficients.len().ilog2();
        let rhs_log_2_of_n = rhs_coefficients.len().ilog2();
        ntt::<FF>(&mut lhs_coefficients, root, lhs_log_2_of_n);
        ntt::<FF>(&mut rhs_coefficients, root, rhs_log_2_of_n);

        let mut hadamard_product: Vec<FF> = rhs_coefficients
            .into_iter()
            .zip(lhs_coefficients)
            .map(|(r, l)| r * l)
            .collect();

        let log_2_of_n = hadamard_product.len().ilog2();
        intt::<FF>(&mut hadamard_product, root, log_2_of_n);
        hadamard_product.truncate(degree + 1);

        Self::new(hadamard_product)
    }

    /// Extracted from `cargo bench --bench zerofier` on mjolnir.
    const CUTOFF_POINT_FOR_FAST_ZEROFIER: usize = 200;

    /// Compute the lowest degree polynomial with the provided roots.
    ///
    /// Uses the fastest version of zerofier available, depending on the size of the domain.
    /// Should be preferred over [`Self::fast_zerofier`] and [`Self::naive_zerofier`].
    pub fn zerofier(roots: &[FF]) -> Self {
        if roots.len() < Self::CUTOFF_POINT_FOR_FAST_ZEROFIER {
            return Self::naive_zerofier(roots);
        }
        Self::fast_zerofier(roots)
    }

    pub fn fast_zerofier(domain: &[FF]) -> Self {
        let dedup_domain = domain.iter().copied().unique().collect::<Vec<_>>();
        let root_order = (dedup_domain.len() + 1).next_power_of_two();
        let primitive_root = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();
        Self::fast_zerofier_inner(&dedup_domain, primitive_root, root_order)
    }

    fn fast_zerofier_inner(
        domain: &[FF],
        primitive_root: BFieldElement,
        root_order: usize,
    ) -> Self {
        if domain.is_empty() {
            return Self::one();
        }
        if domain.len() == 1 {
            return Self::new(vec![-domain[0], FF::one()]);
        }

        let mid_point = domain.len() / 2;
        let left = Self::fast_zerofier_inner(&domain[..mid_point], primitive_root, root_order);
        let right = Self::fast_zerofier_inner(&domain[mid_point..], primitive_root, root_order);
        Self::fast_multiply(&left, &right, primitive_root, root_order)
    }

    pub fn fast_evaluate(&self, domain: &[FF]) -> Vec<FF> {
        let root_order = (domain.len() + 1).next_power_of_two();
        let root_order_u64 = u64::try_from(root_order).unwrap();
        let primitive_root = BFieldElement::primitive_root_of_unity(root_order_u64).unwrap();
        self.fast_evaluate_inner(domain, primitive_root, root_order)
    }

    fn fast_evaluate_inner(
        &self,
        domain: &[FF],
        primitive_root: BFieldElement,
        root_order: usize,
    ) -> Vec<FF> {
        if domain.is_empty() {
            return vec![];
        }

        if domain.len() == 1 {
            return vec![self.evaluate(&domain[0])];
        }

        let half = domain.len() / 2;

        let left_zerofier = Self::fast_zerofier_inner(&domain[..half], primitive_root, root_order);
        let right_zerofier = Self::fast_zerofier_inner(&domain[half..], primitive_root, root_order);

        let mut left = (self.clone() % left_zerofier).fast_evaluate_inner(
            &domain[..half],
            primitive_root,
            root_order,
        );
        let mut right = (self.clone() % right_zerofier).fast_evaluate_inner(
            &domain[half..],
            primitive_root,
            root_order,
        );

        left.append(&mut right);
        left
    }

    /// # Panics
    ///
    /// - Panics if the provided domain is empty.
    /// - Panics if the provided domain and values are not of the same length.
    pub fn fast_interpolate(domain: &[FF], values: &[FF]) -> Self {
        assert_eq!(domain.len(), values.len());
        assert!(
            !domain.is_empty(),
            "Cannot interpolate through zero points.",
        );

        let root_order = (domain.len() + 1).next_power_of_two();
        let root_order_u64 = u64::try_from(root_order).unwrap();
        let primitive_root = BFieldElement::primitive_root_of_unity(root_order_u64).unwrap();
        Self::fast_interpolate_inner(domain, values, primitive_root, root_order)
    }

    fn fast_interpolate_inner(
        domain: &[FF],
        values: &[FF],
        primitive_root: BFieldElement,
        root_order: usize,
    ) -> Self {
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            BFieldElement::one(),
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {primitive_root:?}\
            Supplied order was: {root_order:?}"
        );

        const CUTOFF_POINT_FOR_FAST_INTERPOLATION: usize = 1024;
        if domain.len() < CUTOFF_POINT_FOR_FAST_INTERPOLATION {
            return Self::lagrange_interpolate(domain, values);
        }

        let half = domain.len() / 2;

        let left_zerofier = Self::fast_zerofier_inner(&domain[..half], primitive_root, root_order);
        let right_zerofier = Self::fast_zerofier_inner(&domain[half..], primitive_root, root_order);

        let left_offset: Vec<FF> =
            Self::fast_evaluate_inner(&right_zerofier, &domain[..half], primitive_root, root_order);
        let right_offset: Vec<FF> =
            Self::fast_evaluate_inner(&left_zerofier, &domain[half..], primitive_root, root_order);

        let left_offset_inverse = FF::batch_inversion(left_offset);
        let right_offset_inverse = FF::batch_inversion(right_offset);
        let left_targets: Vec<FF> = values[..half]
            .iter()
            .zip(left_offset_inverse)
            .map(|(n, d)| n.to_owned() * d)
            .collect();
        let right_targets: Vec<FF> = values[half..]
            .iter()
            .zip(right_offset_inverse)
            .map(|(n, d)| n.to_owned() * d)
            .collect();

        let left_interpolant = Self::fast_interpolate_inner(
            &domain[..half],
            &left_targets,
            primitive_root,
            root_order,
        );
        let right_interpolant = Self::fast_interpolate_inner(
            &domain[half..],
            &right_targets,
            primitive_root,
            root_order,
        );

        let left_term = Self::fast_multiply(
            &left_interpolant,
            &right_zerofier,
            primitive_root,
            root_order,
        );
        let right_term = Self::fast_multiply(
            &right_interpolant,
            &left_zerofier,
            primitive_root,
            root_order,
        );
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
            primitive_root,
            root_order,
            &mut zerofier_dictionary,
            &mut offset_inverse_dictionary,
        )
    }

    fn batch_fast_interpolate_with_memoization(
        domain: &[FF],
        values_matrix: &Vec<Vec<FF>>,
        primitive_root: BFieldElement,
        root_order: usize,
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
                let left_zerofier =
                    Self::fast_zerofier_inner(&domain[..half], primitive_root, root_order);
                zerofier_dictionary.insert(left_key, left_zerofier.clone());
                left_zerofier
            }
        };
        let right_key = (domain[half], *domain.last().unwrap());
        let right_zerofier = match zerofier_dictionary.get(&right_key) {
            Some(z) => z.to_owned(),
            None => {
                let right_zerofier =
                    Self::fast_zerofier_inner(&domain[half..], primitive_root, root_order);
                zerofier_dictionary.insert(right_key, right_zerofier.clone());
                right_zerofier
            }
        };

        let left_offset_inverse = match offset_inverse_dictionary.get(&left_key) {
            Some(vector) => vector.to_owned(),
            None => {
                let left_offset: Vec<FF> = Self::fast_evaluate_inner(
                    &right_zerofier,
                    &domain[..half],
                    primitive_root,
                    root_order,
                );
                let left_offset_inverse = FF::batch_inversion(left_offset);
                offset_inverse_dictionary.insert(left_key, left_offset_inverse.clone());
                left_offset_inverse
            }
        };
        let right_offset_inverse = match offset_inverse_dictionary.get(&right_key) {
            Some(vector) => vector.to_owned(),
            None => {
                let right_offset: Vec<FF> = Self::fast_evaluate_inner(
                    &left_zerofier,
                    &domain[half..],
                    primitive_root,
                    root_order,
                );
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
            primitive_root,
            root_order,
            zerofier_dictionary,
            offset_inverse_dictionary,
        );
        let right_interpolants = Self::batch_fast_interpolate_with_memoization(
            &domain[half..],
            &all_right_targets,
            primitive_root,
            root_order,
            zerofier_dictionary,
            offset_inverse_dictionary,
        );

        // add vectors of polynomials
        let interpolants = left_interpolants
            .par_iter()
            .zip(right_interpolants.par_iter())
            .map(|(left_interpolant, right_interpolant)| {
                let left_term = Self::fast_multiply(
                    left_interpolant,
                    &right_zerofier,
                    primitive_root,
                    root_order,
                );
                let right_term = Self::fast_multiply(
                    right_interpolant,
                    &left_zerofier,
                    primitive_root,
                    root_order,
                );

                left_term + right_term
            })
            .collect();

        interpolants
    }

    /// Fast evaluate on a coset domain, which is the group generated by `generator^i * offset`.
    ///
    /// ### Current limitations
    ///
    /// - The order of the domain must be greater than the degree of `self`.
    pub fn fast_coset_evaluate(
        &self,
        offset: BFieldElement,
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
    pub fn fast_coset_interpolate(
        offset: BFieldElement,
        generator: BFieldElement,
        values: &[FF],
    ) -> Self {
        let length = values.len();
        let mut mut_values = values.to_vec();

        intt(&mut mut_values, generator, length.ilog2());
        let poly = Polynomial::new(mut_values);

        poly.scale(offset.inverse())
    }

    /// Divide `self` by some `divisor`.
    ///
    /// As the name implies, the advantage of this method over [`divide`](Self::divide) is runtime
    /// complexity. Concretely, this method has time complexity in O(n·log(n)), whereas
    /// [`divide`](Self::divide) has time complexity in O(n^2).
    ///
    /// The disadvantage of this method is its incompleteness. In some very specific cases, the
    /// division cannot be performed and `panic!`s. More concretely:
    /// Let `r` be the [root of unity][root_unit] of order `n`, where `n` is the next power of two
    /// of ([`self.degree()`](Self::degree) `+ 1`).
    /// If the `divisor` has any root equal to any power of `r`, this method will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// let a = Polynomial::<BFieldElement>::from([1, 2, 3]);
    /// let b = Polynomial::from([42, 53]);
    /// let c = a.fast_divide(&b);
    /// ```
    ///
    /// ## Incompleteness
    ///
    /// The divisor `(x - 1)` has [root of unity][root_unit] `1` as its root, triggering a panic.
    ///
    /// ```should_panic
    /// # use twenty_first::prelude::*;
    /// # // surpass `Polynomial::FAST_DIVIDE_CUTOFF_THRESHOLD`
    /// # const DEGREE: u64 = 8;
    /// # let roots = (0..DEGREE).map(BFieldElement::new).collect::<Vec<_>>();
    /// # let dividend = Polynomial::zerofier(&roots);
    /// let divisor = Polynomial::zerofier(&[bfe!(1)]);
    /// let quotient = dividend.fast_divide(&divisor); // panic!
    /// ```
    ///
    /// Should speed be of no concern, for example because the degrees of the polynomials involved
    /// are known to be small, use [`divide`](Self::divide) instead.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// # // surpass `Polynomial::FAST_DIVIDE_CUTOFF_THRESHOLD`
    /// # const DEGREE: u64 = 8;
    /// # let roots = (0..DEGREE).map(BFieldElement::new).collect::<Vec<_>>();
    /// # let dividend = Polynomial::zerofier(&roots);
    /// # let divisor = Polynomial::zerofier(&[bfe!(1)]);
    /// let quotient = dividend / divisor;
    /// ```
    ///
    /// Alternatively, if the roots of the divisor are known or can be anticipated, _and_ are known
    /// to be problematic, use [`fast_coset_divide`](Self::fast_coset_divide) to perform division in
    /// a coset.
    ///
    /// ```
    /// # use twenty_first::prelude::*;
    /// # // surpass `Polynomial::FAST_DIVIDE_CUTOFF_THRESHOLD`
    /// # const DEGREE: u64 = 8;
    /// # let roots = (0..DEGREE).map(BFieldElement::new).collect::<Vec<_>>();
    /// # let dividend = Polynomial::zerofier(&roots);
    /// # let divisor = Polynomial::zerofier(&[bfe!(1)]);
    /// let quotient = dividend.fast_coset_divide(&divisor, bfe!(7));
    /// ```
    ///
    /// # Panics
    ///
    /// - if the `divisor` is zero
    /// - if the degree of the `divisor` is greater than the degree of `self`
    /// - if the `divisor` has a [root of unity][root_unit] as a root, as discussed above
    ///
    /// [root_unit]: BFieldElement::primitive_root_of_unity
    pub fn fast_divide(&self, divisor: &Self) -> Self {
        self.fast_coset_divide(divisor, BFIELD_ONE)
    }

    /// Divide `self` by some `divisor`.
    /// Generally like [`fast_divide`](Self::fast_divide).
    ///
    /// The additional `offset` grants finer control over the coset domain used internally to
    /// perform the divison. Intimate understanding of [roots of unity][root_unit], cosets, and the
    /// `divisor`'s shape are required to select an appropriate `offset`. For a discussion, see
    /// [`fast_divide`](Self::fast_divide).
    ///
    /// # Panics
    ///
    /// - if the `divisor` is zero
    /// - if the `offset` is zero
    /// - if the degree of the `divisor` is greater than the degree of `self`
    /// - if the offset `divisor` has a [root of unity][root_unit] as a root
    ///
    /// [root_unit]: BFieldElement::primitive_root_of_unity
    pub fn fast_coset_divide(&self, divisor: &Self, offset: BFieldElement) -> Self {
        // Uses the homomorphism of evaluation, i.e., NTT + batch inversion + iNTT.

        assert!(!divisor.is_zero(), "divisor must be non-zero");
        if self.is_zero() {
            return Self::zero();
        }

        assert!(
            divisor.degree() <= self.degree(),
            "divisor degree must be at most that of dividend"
        );

        // See the comment in `fast_coset_evaluate` why this bound is necessary.
        let order = (self.degree() as usize + 1).next_power_of_two();
        let order_u64 = u64::try_from(order).unwrap();
        let root = BFieldElement::primitive_root_of_unity(order_u64).unwrap();

        if self.degree() < Self::FAST_DIVIDE_CUTOFF_THRESHOLD {
            return self.to_owned() / divisor.to_owned();
        }

        let mut dividend_coefficients = self.scale(offset).coefficients;
        let mut divisor_coefficients = divisor.scale(offset).coefficients;

        dividend_coefficients.resize(order, FF::zero());
        divisor_coefficients.resize(order, FF::zero());

        ntt(&mut dividend_coefficients, root, order.ilog2());
        ntt(&mut divisor_coefficients, root, order.ilog2());

        let divisor_inverses = FF::batch_inversion(divisor_coefficients);
        let mut quotient_codeword = dividend_coefficients
            .into_iter()
            .zip(divisor_inverses)
            .map(|(l, r)| l * r)
            .collect_vec();

        intt(&mut quotient_codeword, root, order.ilog2());
        Self::new(quotient_codeword).scale(offset.inverse())
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

    pub fn leading_coefficient(&self) -> Option<FF> {
        match self.degree() {
            -1 => None,
            n => Some(self.coefficients[n as usize]),
        }
    }

    pub fn lagrange_interpolate(domain: &[FF], values: &[FF]) -> Self {
        assert_eq!(
            domain.len(),
            values.len(),
            "The domain and values lists have to be of equal length."
        );
        assert!(
            !domain.is_empty(),
            "Trying to interpolate through 0 points."
        );

        let zero = FF::zero();
        let one = FF::one();

        // precompute the coefficient vector of the zerofier,
        // which is the monic lowest-degree polynomial that evaluates
        // to zero in all domain points, also prod_i (X - d_i).
        let mut zerofier_array = vec![zero; domain.len() + 1];
        zerofier_array[0] = one;
        let mut num_coeffs = 1;
        for &d in domain.iter() {
            for k in (1..num_coeffs + 1).rev() {
                zerofier_array[k] = zerofier_array[k - 1] - d * zerofier_array[k];
            }
            zerofier_array[0] = -d * zerofier_array[0];
            num_coeffs += 1;
        }

        // in each iteration of this loop, we accumulate into the sum
        // one polynomial that evaluates to some abscis (y-value) in
        // the given ordinate (domain point), and to zero in all other
        // ordinates.
        let mut lagrange_sum_array = vec![zero; domain.len()];
        let mut summand_array = vec![zero; domain.len()];
        for (i, &abscis) in values.iter().enumerate() {
            // divide (X - domain[i]) out of zerofier to get unweighted summand
            let mut leading_coefficient = zerofier_array[domain.len()];
            let mut supporting_coefficient = zerofier_array[domain.len() - 1];
            for k in (0..domain.len()).rev() {
                summand_array[k] = leading_coefficient;
                leading_coefficient = supporting_coefficient + leading_coefficient * domain[i];
                if k != 0 {
                    supporting_coefficient = zerofier_array[k - 1];
                }
            }

            // summand does not necessarily evaluate to 1 in domain[i],
            // so we need to correct for this value
            let mut summand_eval = zero;
            for s in summand_array.iter().rev() {
                summand_eval = summand_eval * domain[i] + *s;
            }
            let corrected_abscis = abscis / summand_eval;

            // accumulate term
            for j in 0..domain.len() {
                lagrange_sum_array[j] += corrected_abscis * summand_array[j];
            }
        }
        Polynomial {
            coefficients: lagrange_sum_array,
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

    pub fn naive_zerofier(domain: &[FF]) -> Self {
        domain
            .iter()
            .unique()
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
            println!("Too few points received. Got: {} points", points.len());
            return false;
        }

        if !points.iter().map(|p| p.0).all_unique() {
            println!("Non-unique element spotted Got: {points:?}");
            return false;
        }

        // Find 1st degree polynomial from first two points
        let one: FF = FF::one();
        let x_diff: FF = points[0].0 - points[1].0;
        let x_diff_inv = one / x_diff;
        let a = (points[0].1 - points[1].1) * x_diff_inv;
        let b = points[0].1 - a * points[0].0;
        for point in points.iter().skip(2) {
            let expected = a * point.0 + b;
            if point.1 != expected {
                println!(
                    "L({}) = {}, expected L({}) = {}, Found: L(x) = {}x + {} from {{({},{}),({},{})}}",
                    point.0,
                    point.1,
                    point.0,
                    expected,
                    a,
                    b,
                    points[0].0,
                    points[0].1,
                    points[1].0,
                    points[1].1
                );
                return false;
            }
        }

        true
    }

    /// Any fast interpolation will use NTT, so this is mainly used for testing/integrity
    /// purposes. This also means that it is not pivotal that this function has an optimal
    /// runtime.
    pub fn lagrange_interpolate_zipped(points: &[(FF, FF)]) -> Self {
        if points.is_empty() {
            panic!("Cannot interpolate through zero points.");
        }
        if !points.iter().map(|x| x.0).all_unique() {
            panic!("Repeated x values received. Got: {points:?}");
        }

        let xs: Vec<FF> = points.iter().map(|x| x.0.to_owned()).collect();
        let ys: Vec<FF> = points.iter().map(|x| x.1.to_owned()).collect();
        Self::lagrange_interpolate(&xs, &ys)
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    pub fn multiply(self, other: Self) -> Self {
        let degree_lhs = self.degree();
        let degree_rhs = other.degree();

        if degree_lhs < 0 || degree_rhs < 0 {
            return Self::zero();
            // return self.zero();
        }

        // allocate right number of coefficients, initialized to zero
        let mut result_coeff: Vec<FF> =
            //vec![U::zero_from_field(field: U); degree_lhs as usize + degree_rhs as usize + 1];
            vec![FF::zero(); degree_lhs as usize + degree_rhs as usize + 1];

        // TODO: Review this.
        // for all pairs of coefficients, add product to result vector in appropriate coordinate
        for i in 0..=degree_lhs as usize {
            for j in 0..=degree_rhs as usize {
                let mul: FF = self.coefficients[i] * other.coefficients[j];
                result_coeff[i + j] += mul;
            }
        }

        // build and return Polynomial object
        Self {
            coefficients: result_coeff,
        }
    }

    // Multiply a polynomial with itself `pow` times
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

    // Multiply a polynomial with x^power
    #[must_use]
    pub fn shift_coefficients(&self, power: usize) -> Self {
        let zero = FF::zero();

        let mut coefficients: Vec<FF> = self.coefficients.clone();
        coefficients.splice(0..0, vec![zero; power]);
        Polynomial { coefficients }
    }

    // TODO: Review
    pub fn scalar_mul_mut(&mut self, scalar: FF) {
        for coefficient in self.coefficients.iter_mut() {
            *coefficient *= scalar;
        }
    }

    #[must_use]
    pub fn scalar_mul(&self, scalar: FF) -> Self {
        let mut coefficients: Vec<FF> = vec![];
        for i in 0..self.coefficients.len() {
            coefficients.push(self.coefficients[i] * scalar);
        }

        Self { coefficients }
    }

    /// Return (quotient, remainder)
    pub fn divide(&self, divisor: Self) -> (Self, Self) {
        let degree_lhs = self.degree();
        let degree_rhs = divisor.degree();
        // cannot divide by zero
        if degree_rhs < 0 {
            panic!("Cannot divide polynomial by zero. Got: ({self:?})/({divisor:?})");
        }

        // zero divided by anything gives zero. degree == -1 <=> polynomial = 0
        if self.is_zero() {
            return (Self::zero(), Self::zero());
        }

        // quotient is built from back to front so must be reversed
        // Preallocate space for quotient coefficients
        let mut quotient: Vec<FF> = if degree_lhs - degree_rhs >= 0 {
            Vec::with_capacity((degree_lhs - degree_rhs + 1) as usize)
        } else {
            vec![]
        };
        let mut remainder = self.clone();
        remainder.normalize();

        // a divisor coefficient is guaranteed to exist since the divisor is non-zero
        let dlc: FF = divisor.leading_coefficient().unwrap();
        let inv = FF::one() / dlc;

        let mut i = 0;
        while i + degree_rhs <= degree_lhs {
            // calculate next quotient coefficient, and set leading coefficient
            // of remainder is 0 by removing it
            let rlc: FF = remainder.coefficients.last().unwrap().to_owned();
            let q: FF = rlc * inv;
            quotient.push(q);
            remainder.coefficients.pop();
            if q.is_zero() {
                i += 1;
                continue;
            }

            // TODO: Review that this loop body was correctly modified.
            // Calculate the new remainder
            for j in 0..degree_rhs as usize {
                let rem_length = remainder.coefficients.len();
                remainder.coefficients[rem_length - j - 1] -=
                    q * divisor.coefficients[(degree_rhs + 1) as usize - j - 2];
            }

            i += 1;
        }

        quotient.reverse();
        let quotient_pol = Self {
            coefficients: quotient,
        };

        (quotient_pol, remainder)
    }
}

impl<FF: FiniteField> Div for Polynomial<FF> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let (quotient, _): (Self, Self) = self.divide(other);
        quotient
    }
}

impl<FF: FiniteField> Rem for Polynomial<FF> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        let (_, remainder): (Self, Self) = self.divide(other);
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
    pub fn xgcd(x: Self, y: Self) -> (Self, Self, Self) {
        let (x, a, b) = other::xgcd(x, y);

        // normalize result to ensure `x` has leading coefficient 1
        let lc = x.leading_coefficient().unwrap_or_else(FF::one);
        let normalize = |poly: Self| poly.scalar_mul(lc.inverse());

        let [x, a, b] = [x, a, b].map(normalize);
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
        let coefficients = self
            .clone()
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, &coefficient)| FF::new_from_usize(&coefficient, i) * coefficient)
            .skip(1)
            .collect_vec();

        Self { coefficients }
    }
}

impl<FF: FiniteField> Mul for Polynomial<FF> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::multiply(self, other)
    }
}

#[cfg(test)]
mod test_polynomials {
    use proptest::collection::size_range;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::shared_math::traits::PrimitiveRootOfUnity;
    use crate::shared_math::x_field_element::XFieldElement;

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
        leading_coefficient: BFieldElement,
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
        let to_bfe_vec = |a: &[u64]| a.iter().copied().map(BFieldElement::new).collect_vec();

        let polynomial = Polynomial::<BFieldElement>::from([17, 14]);
        assert_eq!(
            to_bfe_vec(&[17, 14]),
            polynomial.shift_coefficients(0).coefficients
        );
        assert_eq!(
            to_bfe_vec(&[0, 17, 14]),
            polynomial.shift_coefficients(1).coefficients
        );
        assert_eq!(
            to_bfe_vec(&[0, 0, 0, 0, 17, 14]),
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
    fn fast_multiplication_by_zero_gives_zero(poly: Polynomial<BFieldElement>) {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        let product = Polynomial::fast_multiply(&Polynomial::zero(), &poly, primitive_root, 32);
        prop_assert_eq!(Polynomial::zero(), product);
    }

    #[proptest]
    fn fast_multiplication_by_one_gives_self(poly: Polynomial<BFieldElement>) {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        let product = Polynomial::fast_multiply(&Polynomial::one(), &poly, primitive_root, 32);
        prop_assert_eq!(poly, product);
    }

    #[proptest]
    fn fast_multiplication_is_commutative(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        let product = Polynomial::fast_multiply(&a, &b, primitive_root, 32);
        let product_commutative = Polynomial::fast_multiply(&b, &a, primitive_root, 32);
        prop_assert_eq!(product, product_commutative);
    }

    #[proptest]
    fn fast_multiplication_and_normal_multiplication_are_equivalent(
        a: Polynomial<BFieldElement>,
        b: Polynomial<BFieldElement>,
    ) {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        let product = Polynomial::fast_multiply(&a, &b, primitive_root, 32);
        prop_assert_eq!(a * b, product);
    }

    #[proptest(cases = 50)]
    fn naive_zerofier_and_fast_zerofier_are_identical(
        #[any(size_range(..1024).lift())] domain: Vec<BFieldElement>,
    ) {
        let zerofier = Polynomial::naive_zerofier(&domain);
        let fast_zerofier = Polynomial::fast_zerofier(&domain);
        prop_assert_eq!(zerofier, fast_zerofier);
    }

    #[proptest(cases = 50)]
    fn zerofier_and_naive_zerofier_are_identical(
        #[any(size_range(..1024).lift())] domain: Vec<BFieldElement>,
    ) {
        let zerofier = Polynomial::zerofier(&domain);
        let naive_zerofier = Polynomial::naive_zerofier(&domain);
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
    #[should_panic(expected = "0 points")]
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
        #[strategy(0..8usize)]
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
    fn fast_coset_division_and_division_are_equivalent(
        a: Polynomial<BFieldElement>,
        #[filter(!#b.is_zero())] b: Polynomial<BFieldElement>,
        #[filter(!#offset.is_zero())] offset: BFieldElement,
    ) {
        let product = a.clone() * b.clone();
        let quotient = product.fast_coset_divide(&b, offset);
        prop_assert_eq!(product / b, quotient);
    }

    #[proptest]
    fn fast_coset_division_and_fast_division_are_equivalent(
        a: Polynomial<BFieldElement>,
        #[filter(!#b.is_zero())] b: Polynomial<BFieldElement>,
        #[filter(!#offset.is_zero())] offset: BFieldElement,
    ) {
        let product = a.clone() * b.clone();
        let quotient_0 = product.fast_divide(&b);
        let quotient_1 = product.fast_coset_divide(&b, offset);
        prop_assert_eq!(quotient_0, quotient_1);
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
        let (_, remainder) = a.divide(b_poly);
        prop_assert_eq!(Polynomial::zero(), remainder);
    }

    #[test]
    fn polynomial_division_by_and_with_shah_polynomial() {
        let polynomial = |cs: &[u64]| Polynomial::<BFieldElement>::from(cs);

        let shah = XFieldElement::shah_polynomial();
        let x_to_the_3 = polynomial(&[1]).shift_coefficients(3);
        let (shah_div_x_to_the_3, shah_mod_x_to_the_3) = shah.divide(x_to_the_3);
        assert_eq!(polynomial(&[1]), shah_div_x_to_the_3);
        assert_eq!(polynomial(&[1, BFieldElement::P - 1]), shah_mod_x_to_the_3);

        let x_to_the_6 = polynomial(&[1]).shift_coefficients(6);
        let (x_to_the_6_div_shah, x_to_the_6_mod_shah) = x_to_the_6.divide(shah);

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
