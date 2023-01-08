use crate::shared_math::ntt::{intt, ntt};
use crate::shared_math::other::{log_2_floor, roundup_npo2};
use crate::shared_math::traits::{FiniteField, ModPowU32};
use crate::utils::has_unique_elements;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use num_bigint::BigInt;
use num_traits::{One, Zero};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::convert::From;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Rem, Sub};

use super::b_field_element::BFieldElement;
use super::other::{self, log_2_ceil};
use super::traits::{Inverse, PrimitiveRootOfUnity};

fn degree_raw<T: Add + Div + Mul + Sub + Display + Zero>(coefficients: &[T]) -> isize {
    let mut deg = coefficients.len() as isize - 1;
    while deg >= 0 && coefficients[deg as usize].is_zero() {
        deg -= 1;
    }

    deg // -1 for the zero polynomial
}

fn pretty_print_coefficients_generic<FF: FiniteField>(coefficients: &[FF]) -> String {
    let degree = degree_raw(coefficients);
    if degree == -1 {
        return String::from("0");
    }

    // for every nonzero term, in descending order
    let mut outputs: Vec<String> = Vec::new();
    let pol_degree = degree as usize;
    for i in 0..=pol_degree {
        let pow = pol_degree - i;
        if coefficients[pow].is_zero() {
            continue;
        }

        outputs.push(format!(
            "{}{}{}", // { + } { 7 } { x^3 }
            if i == 0 { "" } else { " + " },
            if coefficients[pow].is_one() {
                String::from("")
            } else {
                coefficients[pow].to_string()
            },
            if pow == 0 && coefficients[pow].is_one() {
                let one: FF = FF::one();
                one.to_string()
            } else if pow == 0 {
                String::from("")
            } else if pow == 1 {
                String::from("x")
            } else {
                let mut result = "x^".to_owned();
                let borrowed_string = pow.to_string().to_owned();
                result.push_str(&borrowed_string);
                result
            }
        ));
    }
    outputs.join("")
}

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

#[derive(Clone)]
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
        write!(
            f,
            "{}",
            pretty_print_coefficients_generic::<FF>(&self.coefficients)
        )
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
    // Return the polynomial which corresponds to the transformation `x -> alpha * x`
    // Given a polynomial P(x), produce P'(x) := P(alpha * x). Evaluating P'(x)
    // then corresponds to evaluating P(alpha * x).
    #[must_use]
    pub fn scale(&self, &alpha: &BFieldElement) -> Self {
        let mut acc = FF::one();
        let mut return_coefficients = self.coefficients.clone();
        for elem in return_coefficients.iter_mut() {
            *elem *= acc;
            acc *= alpha;
        }

        Self {
            coefficients: return_coefficients,
        }
    }

    // It is the caller's responsibility that this function
    // is called with sufficiently large input to be safe
    // and to be faster than `square`.
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
        let order = roundup_npo2(result_degree + 1);
        let root_res = BFieldElement::primitive_root_of_unity(order);
        let root = match root_res {
            Some(n) => n,
            None => panic!("Failed to find primitive root for order = {}", order),
        };

        let mut coefficients = self.coefficients.to_vec();
        coefficients.resize(order as usize, FF::zero());
        let log_2_of_n = log_2_floor(coefficients.len() as u128) as u32;
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

        // A benchmark run on sword_smith's PC revealed that
        // `fast_square` was faster when the input size exceeds
        // a length of 64.
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
        primitive_root: &BFieldElement,
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

        let lhs_log_2_of_n = log_2_floor(lhs_coefficients.len() as u128) as u32;
        let rhs_log_2_of_n = log_2_floor(rhs_coefficients.len() as u128) as u32;
        ntt::<FF>(&mut lhs_coefficients, root, lhs_log_2_of_n);
        ntt::<FF>(&mut rhs_coefficients, root, rhs_log_2_of_n);

        let mut hadamard_product: Vec<FF> = rhs_coefficients
            .into_iter()
            .zip(lhs_coefficients.into_iter())
            .map(|(r, l)| r * l)
            .collect();

        let log_2_of_n = log_2_floor(hadamard_product.len() as u128) as u32;
        intt::<FF>(&mut hadamard_product, root, log_2_of_n);
        hadamard_product.truncate(degree + 1);

        Polynomial {
            coefficients: hadamard_product,
        }
    }

    // domain: polynomial roots
    pub fn fast_zerofier(domain: &[FF], primitive_root: &BFieldElement, root_order: usize) -> Self {
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            BFieldElement::one(),
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {:?}\
            Supplied order was: {:?}",
            primitive_root,
            root_order
        );

        if domain.is_empty() {
            return Self::zero();
        }

        if domain.len() == 1 {
            return Self {
                coefficients: vec![-domain[0], FF::one()],
            };
        }

        // This assertion must come after above recursion-ending cases have been dealt with.
        // Otherwise, the supplied primitive_root will (at some point) equal 1 with correct
        // root_order = 1, incorrectly failing the assertion.
        debug_assert_ne!(
            primitive_root.mod_pow_u32((root_order / 2) as u32),
            BFieldElement::one(),
            "Supplied element “primitive_root” must be primitive root of supplied order.\
            Supplied element was: {:?}\
            Supplied order was: {:?}",
            primitive_root,
            root_order
        );

        let half = domain.len() / 2;

        let left = Self::fast_zerofier(&domain[..half], primitive_root, root_order);
        let right = Self::fast_zerofier(&domain[half..], primitive_root, root_order);
        Self::fast_multiply(&left, &right, primitive_root, root_order)
    }

    pub fn fast_evaluate(
        &self,
        domain: &[FF],
        primitive_root: &BFieldElement,
        root_order: usize,
    ) -> Vec<FF> {
        if domain.is_empty() {
            return vec![];
        }

        if domain.len() == 1 {
            return vec![self.evaluate(&domain[0])];
        }

        let half = domain.len() / 2;

        let left_zerofier = Self::fast_zerofier(&domain[..half], primitive_root, root_order);
        let right_zerofier = Self::fast_zerofier(&domain[half..], primitive_root, root_order);

        let mut left = (self.clone() % left_zerofier).fast_evaluate(
            &domain[..half],
            primitive_root,
            root_order,
        );
        let mut right = (self.clone() % right_zerofier).fast_evaluate(
            &domain[half..],
            primitive_root,
            root_order,
        );

        left.append(&mut right);
        left
    }

    pub fn fast_interpolate(
        domain: &[FF],
        values: &[FF],
        primitive_root: &BFieldElement,
        root_order: usize,
    ) -> Self {
        assert_eq!(
            domain.len(),
            values.len(),
            "Domain and values lengths must match"
        );
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            BFieldElement::one(),
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {:?}\
            Supplied order was: {:?}",
            primitive_root,
            root_order
        );

        assert!(
            !domain.is_empty(),
            "Cannot fast interpolate through zero points.",
        );

        const CUTOFF_POINT_FOR_FAST_INTERPOLATION: usize = 1024;
        if domain.len() < CUTOFF_POINT_FOR_FAST_INTERPOLATION {
            return Self::lagrange_interpolate(domain, values);
        }

        let half = domain.len() / 2;

        let left_zerofier = Self::fast_zerofier(&domain[..half], primitive_root, root_order);
        let right_zerofier = Self::fast_zerofier(&domain[half..], primitive_root, root_order);

        let left_offset: Vec<FF> =
            Self::fast_evaluate(&right_zerofier, &domain[..half], primitive_root, root_order);
        let right_offset: Vec<FF> =
            Self::fast_evaluate(&left_zerofier, &domain[half..], primitive_root, root_order);

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

        let left_interpolant =
            Self::fast_interpolate(&domain[..half], &left_targets, primitive_root, root_order);
        let right_interpolant =
            Self::fast_interpolate(&domain[half..], &right_targets, primitive_root, root_order);

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
        primitive_root: &BFieldElement,
        root_order: usize,
    ) -> Vec<Self> {
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            BFieldElement::one(),
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {:?}\
            Supplied order was: {:?}",
            primitive_root,
            root_order
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
        primitive_root: &BFieldElement,
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
                    Self::fast_zerofier(&domain[..half], primitive_root, root_order);
                zerofier_dictionary.insert(left_key, left_zerofier.clone());
                left_zerofier
            }
        };
        let right_key = (domain[half], *domain.last().unwrap());
        let right_zerofier = match zerofier_dictionary.get(&right_key) {
            Some(z) => z.to_owned(),
            None => {
                let right_zerofier =
                    Self::fast_zerofier(&domain[half..], primitive_root, root_order);
                zerofier_dictionary.insert(right_key, right_zerofier.clone());
                right_zerofier
            }
        };

        let left_offset_inverse = match offset_inverse_dictionary.get(&left_key) {
            Some(vector) => vector.to_owned(),
            None => {
                let left_offset: Vec<FF> = Self::fast_evaluate(
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
                let right_offset: Vec<FF> = Self::fast_evaluate(
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

    /// Fast evaluate on a coset domain, which is the group generated by `generator^i * offset`
    pub fn fast_coset_evaluate(
        &self,
        offset: &BFieldElement,
        generator: BFieldElement,
        order: usize,
    ) -> Vec<FF> {
        let mut coefficients = self.scale(offset).coefficients;
        coefficients.append(&mut vec![FF::zero(); order - coefficients.len()]);
        let log_2_of_n = log_2_floor(coefficients.len() as u128) as u32;
        ntt::<FF>(&mut coefficients, generator, log_2_of_n);
        coefficients
    }

    /// The inverse of `fast_coset_evaluate`
    pub fn fast_coset_interpolate(
        offset: &BFieldElement,
        generator: BFieldElement,
        values: &[FF],
    ) -> Self {
        let length = values.len();
        let mut mut_values = values.to_vec();
        intt(
            &mut mut_values,
            generator,
            log_2_ceil(length as u128) as u32,
        );
        let poly = Polynomial::new(mut_values);

        poly.scale(&offset.inverse())
    }

    /// Divide two polynomials under the homomorphism of evaluation for a N^2 -> N*log(N) speedup
    /// Since we often want to use this fast division for numerators and divisors that evaluate
    /// to zero in their domain, we do the division with an offset from the polynomials' original
    /// domains. The issue of zero in the numerator and divisor arises when we divide a transition
    /// polynomial with a zerofier.
    pub fn fast_coset_divide(
        lhs: &Polynomial<FF>,
        rhs: &Polynomial<FF>,
        offset: BFieldElement,
        primitive_root: BFieldElement,
        root_order: usize,
    ) -> Polynomial<FF> {
        assert!(
            primitive_root.mod_pow_u32(root_order as u32).is_one(),
            "primitive root raised to given order must yield 1"
        );
        assert!(
            !primitive_root.mod_pow_u32(root_order as u32 / 2).is_one(),
            "primitive root raised to half of given order must not yield 1"
        );
        assert!(!rhs.is_zero(), "cannot divide by zero polynomial");

        if lhs.is_zero() {
            return Polynomial {
                coefficients: vec![],
            };
        }

        assert!(
            rhs.degree() <= lhs.degree(),
            "in polynomial division, right hand side degree must be at most that of left hand side"
        );

        let zero = FF::zero();
        let mut root: BFieldElement = primitive_root.to_owned();
        let mut order = root_order;
        let degree: usize = lhs.degree() as usize;

        if degree < 8 {
            return lhs.to_owned() / rhs.to_owned();
        }

        while degree < order / 2 {
            root *= root;
            order /= 2;
        }

        let mut scaled_lhs_coefficients: Vec<FF> = lhs.scale(&offset).coefficients;
        scaled_lhs_coefficients.append(&mut vec![zero; order - scaled_lhs_coefficients.len()]);

        let mut scaled_rhs_coefficients: Vec<FF> = rhs.scale(&offset).coefficients;
        scaled_rhs_coefficients.append(&mut vec![zero; order - scaled_rhs_coefficients.len()]);

        let lhs_log_2_of_n = log_2_floor(scaled_lhs_coefficients.len() as u128) as u32;
        let rhs_log_2_of_n = log_2_floor(scaled_rhs_coefficients.len() as u128) as u32;

        ntt::<FF>(&mut scaled_lhs_coefficients, root, lhs_log_2_of_n);
        ntt::<FF>(&mut scaled_rhs_coefficients, root, rhs_log_2_of_n);

        let rhs_inverses = FF::batch_inversion(scaled_rhs_coefficients);
        let mut quotient_codeword: Vec<FF> = scaled_lhs_coefficients
            .iter()
            .zip(rhs_inverses)
            .map(|(l, r)| l.to_owned() * r)
            .collect();

        let log_2_of_n = log_2_floor(quotient_codeword.len() as u128) as u32;
        intt::<FF>(&mut quotient_codeword, root, log_2_of_n);

        let scaled_quotient = Polynomial {
            coefficients: quotient_codeword,
        };

        scaled_quotient.scale(&offset.inverse())
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    pub fn new(coefficients: Vec<FF>) -> Self {
        Self { coefficients }
    }

    pub fn new_const(element: FF) -> Self {
        Self {
            coefficients: vec![element],
        }
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
        debug_assert_ne!(p0.0, p1.0, "Line must not be parallel to y-axis");
        let dy = p0.1 - p1.1;
        let dx = p0.0 - p1.0;
        let p2_y_times_dx = dy * (p2_x - p0.0) + dx * p0.1;

        // Can we implement this without division?
        p2_y_times_dx / dx
    }

    pub fn zerofier(domain: &[FF]) -> Self {
        if domain.is_empty() {
            return Self {
                coefficients: vec![FF::one()],
            };
        }
        let mut zerofier_array = vec![FF::zero(); domain.len() + 1];
        zerofier_array[0] = FF::one();
        let mut num_coeffs = 1;
        for &d in domain.iter() {
            for k in (1..num_coeffs + 1).rev() {
                zerofier_array[k] = zerofier_array[k - 1] - d * zerofier_array[k];
            }
            zerofier_array[0] = -d * zerofier_array[0];
            num_coeffs += 1;
        }
        Self {
            coefficients: zerofier_array,
        }
    }

    // Slow square implementation that does not use NTT
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

        if !has_unique_elements(points.iter().map(|p| p.0)) {
            println!("Non-unique element spotted Got: {:?}", points);
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

    // Any fast interpolation will use NTT, so this is mainly used for testing/integrity
    // purposes. This also means that it is not pivotal that this function has an optimal
    // runtime.
    pub fn lagrange_interpolate_zipped(points: &[(FF, FF)]) -> Self {
        if points.is_empty() {
            panic!("Cannot interpolate through zero points.");
        }
        if !has_unique_elements(points.iter().map(|x| x.0)) {
            panic!("Repeated x values received. Got: {:?}", points);
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
            panic!(
                "Cannot divide polynomial by zero. Got: ({:?})/({:?})",
                self, divisor
            );
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
            // of remainder remainder is 0 by removing it
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

    // fn add(self, other: Self) -> Self {
    //     let (mut longest, mut shortest) = if self.coefficients.len() < other.coefficients.len() {
    //         (other, self)
    //     } else {
    //         (self, other)
    //     };

    //     let mut summed = longest.clone();
    //     for i in 0..shortest.coefficients.len() {
    //         summed.coefficients[i] += shortest.coefficients[i];
    //     }

    //     summed
    // }

    fn add(self, other: Self) -> Self {
        let summed: Vec<FF> = self
            .coefficients
            .into_iter()
            .zip_longest(other.coefficients.into_iter())
            .map(|a: itertools::EitherOrBoth<FF, FF>| match a {
                Both(l, r) => l.to_owned() + r.to_owned(),
                Left(l) => l.to_owned(),
                Right(r) => r.to_owned(),
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
        let summed: Vec<FF> = self
            .coefficients
            .into_iter()
            .zip_longest(other.coefficients.into_iter())
            .map(|a: itertools::EitherOrBoth<FF, FF>| match a {
                Both(l, r) => l - r,
                Left(l) => l,
                Right(r) => FF::zero() - r,
            })
            .collect();

        Self {
            coefficients: summed,
        }
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    /// Extended Euclidean algorithm with polynomials. Computes the greatest
    /// common divisor `gcd` as a monic polynomial, as well as the corresponding
    /// Bézout coefficients `a` and `b`, satisfying `gcd = a·x + b·y`
    pub fn xgcd(
        x: Polynomial<FF>,
        y: Polynomial<FF>,
    ) -> (Polynomial<FF>, Polynomial<FF>, Polynomial<FF>) {
        let (x, a_factor, b_factor) = other::xgcd(x, y);

        // The result is valid up to a coefficient, so we normalize the result,
        // to ensure that x has a leading coefficient of 1.
        // x cannot be zero here since polynomials form a group and all elements,
        // except the zero polynomial, have an inverse.
        let lc = x.leading_coefficient().unwrap();
        let scale = lc.inverse();
        (
            x.scalar_mul(scale),
            a_factor.scalar_mul(scale),
            b_factor.scalar_mul(scale),
        )
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    pub fn degree(&self) -> isize {
        degree_raw(&self.coefficients)
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
    #![allow(clippy::just_underscores_and_digits)]

    use rand::Rng;
    use rand_distr::Standard;

    use super::*;
    use crate::shared_math::other::{random_elements, random_elements_distinct};
    use crate::shared_math::traits::PrimitiveRootOfUnity;
    use crate::shared_math::x_field_element::XFieldElement;

    #[test]
    fn polynomial_display_test() {
        let empty = Polynomial::<BFieldElement> {
            coefficients: vec![],
        };
        assert_eq!("0", empty.to_string());

        let zero = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(0u64)],
        };
        assert_eq!("0", zero.to_string());

        let double_zero = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(0u64), BFieldElement::from(0u64)],
        };
        assert_eq!("0", double_zero.to_string());

        let one = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(1u64)],
        };
        assert_eq!("1", one.to_string());

        let zero_one = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(1u64), BFieldElement::from(0u64)],
        };
        assert_eq!("1", zero_one.to_string());

        let zero_zero_one = Polynomial::<BFieldElement> {
            coefficients: vec![
                BFieldElement::from(1u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
            ],
        };
        assert_eq!("1", zero_zero_one.to_string());

        let one_zero = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(0u64), BFieldElement::from(1u64)],
        };
        assert_eq!("x", one_zero.to_string());
        assert_eq!("1", one.to_string());
        let x_plus_one = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(1u64), BFieldElement::from(1u64)],
        };
        assert_eq!("x + 1", x_plus_one.to_string());
        let many_zeros = Polynomial::<BFieldElement> {
            coefficients: vec![
                BFieldElement::from(1u64),
                BFieldElement::from(1u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
            ],
        };
        assert_eq!("x + 1", many_zeros.to_string());
        let also_many_zeros = Polynomial::<BFieldElement> {
            coefficients: vec![
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(1u64),
                BFieldElement::from(1u64),
            ],
        };
        assert_eq!("x^4 + x^3", also_many_zeros.to_string());
        let yet_many_zeros = Polynomial::<BFieldElement> {
            coefficients: vec![
                BFieldElement::from(1u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(1u64),
            ],
        };
        assert_eq!("x^4 + 1", yet_many_zeros.to_string());
    }

    #[test]
    fn leading_coefficient_test() {
        // Verify that the leading coefficient for the zero-polynomial is `None`
        let _14 = BFieldElement::new(14);
        let _0 = BFieldElement::zero();
        let _1 = BFieldElement::one();
        let _max = BFieldElement::new(BFieldElement::MAX);
        let lc_0_0: Polynomial<BFieldElement> = Polynomial::new(vec![]);
        let lc_0_1: Polynomial<BFieldElement> = Polynomial::new(vec![_0]);
        let lc_0_2: Polynomial<BFieldElement> = Polynomial::new(vec![_0, _0, _0]);
        assert_eq!(None, lc_0_0.leading_coefficient());
        assert_eq!(None, lc_0_1.leading_coefficient());
        assert_eq!(None, lc_0_2.leading_coefficient());

        // Other numbers as LC
        let lc_1_0: Polynomial<BFieldElement> = Polynomial::new(vec![_1]);
        let lc_1_1: Polynomial<BFieldElement> = Polynomial::new(vec![_0, _0, _0, _1]);
        let lc_1_2: Polynomial<BFieldElement> = Polynomial::new(vec![_max, _14, _0, _max, _1]);
        let lc_1_3: Polynomial<BFieldElement> = Polynomial::new(vec![_max, _14, _0, _max, _1, _0]);
        let lc_1_4: Polynomial<BFieldElement> = Polynomial::new(vec![
            _max, _14, _0, _max, _1, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0,
        ]);
        assert_eq!(Some(_1), lc_1_0.leading_coefficient());
        assert_eq!(Some(_1), lc_1_1.leading_coefficient());
        assert_eq!(Some(_1), lc_1_2.leading_coefficient());
        assert_eq!(Some(_1), lc_1_3.leading_coefficient());
        assert_eq!(Some(_1), lc_1_4.leading_coefficient());

        let lc_14_0: Polynomial<BFieldElement> = Polynomial::new(vec![_14]);
        let lc_14_1: Polynomial<BFieldElement> = Polynomial::new(vec![_0, _0, _0, _14]);
        let lc_14_2: Polynomial<BFieldElement> =
            Polynomial::new(vec![_max, _14, _0, _max, _14, _0, _0, _0]);
        let lc_14_3: Polynomial<BFieldElement> = Polynomial::new(vec![_14, _0]);
        assert_eq!(Some(_14), lc_14_0.leading_coefficient());
        assert_eq!(Some(_14), lc_14_1.leading_coefficient());
        assert_eq!(Some(_14), lc_14_2.leading_coefficient());
        assert_eq!(Some(_14), lc_14_3.leading_coefficient());

        let lc_max_0: Polynomial<BFieldElement> = Polynomial::new(vec![_max]);
        let lc_max_1: Polynomial<BFieldElement> = Polynomial::new(vec![_0, _0, _0, _max]);
        let lc_max_2: Polynomial<BFieldElement> =
            Polynomial::new(vec![_max, _14, _0, _max, _max, _0, _0, _0]);
        assert_eq!(Some(_max), lc_max_0.leading_coefficient());
        assert_eq!(Some(_max), lc_max_1.leading_coefficient());
        assert_eq!(Some(_max), lc_max_2.leading_coefficient());
    }

    #[test]
    fn normalize_test() {
        let _0_71 = BFieldElement::from(0u64);
        let _1_71 = BFieldElement::from(1u64);
        let _6_71 = BFieldElement::from(6u64);
        let _12_71 = BFieldElement::from(12u64);
        let zero: Polynomial<BFieldElement> = Polynomial::zero();
        let mut mut_one: Polynomial<BFieldElement> = Polynomial::<BFieldElement> {
            coefficients: vec![_1_71],
        };
        let one: Polynomial<BFieldElement> = Polynomial::<BFieldElement> {
            coefficients: vec![_1_71],
        };
        let mut a = Polynomial::<BFieldElement> {
            coefficients: vec![],
        };
        a.normalize();
        assert_eq!(zero, a);
        mut_one.normalize();
        assert_eq!(one, mut_one);

        // trailing zeros are removed
        a = Polynomial::<BFieldElement> {
            coefficients: vec![_1_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<BFieldElement> {
                coefficients: vec![_1_71],
            },
            a
        );
        a = Polynomial::<BFieldElement> {
            coefficients: vec![_1_71, _0_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<BFieldElement> {
                coefficients: vec![_1_71],
            },
            a
        );

        // but leading zeros are not removed
        a = Polynomial::<BFieldElement> {
            coefficients: vec![_0_71, _1_71, _0_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<BFieldElement> {
                coefficients: vec![_0_71, _1_71],
            },
            a
        );
        a = Polynomial::<BFieldElement> {
            coefficients: vec![_0_71, _1_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<BFieldElement> {
                coefficients: vec![_0_71, _1_71],
            },
            a
        );
        a = Polynomial::<BFieldElement> {
            coefficients: vec![_0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<BFieldElement> {
                coefficients: vec![],
            },
            a
        );
    }

    #[should_panic]
    #[test]
    fn panic_when_one_is_not_one() {
        assert_eq!(
            Polynomial::<BFieldElement>::new(vec![
                BFieldElement::from(30u64),
                BFieldElement::from(0u64),
                BFieldElement::from(1u64)
            ]),
            Polynomial::zerofier(&[BFieldElement::from(1u64), BFieldElement::from(30u64)])
        );
    }

    #[test]
    fn property_based_slow_lagrange_interpolation_test() {
        // Autogenerate a `number_of_points - 1` degree polynomial
        // We start by autogenerating the polynomial, as we would get a polynomial
        // with fractional coefficients if we autogenerated the points and derived the polynomium
        // from that.
        let number_of_points = 50usize;

        let coefficients: Vec<BFieldElement> = random_elements(number_of_points);
        let pol: Polynomial<BFieldElement> = Polynomial { coefficients };

        // Evaluate polynomial in `number_of_points` points
        let points = (0..number_of_points)
            .map(|x| {
                let x = BFieldElement::from(x as u64);
                (x, pol.evaluate(&x))
            })
            .collect::<Vec<(BFieldElement, BFieldElement)>>();

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result: Polynomial<BFieldElement> =
            Polynomial::lagrange_interpolate_zipped(&points);
        assert_eq!(interpolation_result, pol);
        for point in points {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }
    }

    #[test]
    fn property_based_slow_lagrange_interpolation_test_big() {
        // Autogenerate a `number_of_points - 1` degree polynomial
        // We start by autogenerating the polynomial, as we would get a polynomial
        // with fractional coefficients if we autogenerated the points and derived the polynomium
        // from that.
        let number_of_points = 50usize;
        // FIXME: I'm not sure why BFieldElements were converted to u64 and back here.
        let coefficients: Vec<BFieldElement> = random_elements(number_of_points);
        let pol: Polynomial<BFieldElement> = Polynomial { coefficients };

        // Evaluate polynomial in `number_of_points` points
        let points: Vec<(BFieldElement, BFieldElement)> = (0..number_of_points)
            .map(|x| {
                let x = BFieldElement::from(x as u64);
                (x, pol.evaluate(&x))
            })
            .collect();

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result: Polynomial<BFieldElement> =
            Polynomial::lagrange_interpolate_zipped(&points);
        assert_eq!(interpolation_result, pol);
        for point in points {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }
    }

    #[test]
    fn polynomial_are_colinear_3_test() {
        assert!(Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ));
        assert!(!Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(7u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ));
        assert!(Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(1u64), BFieldElement::from(3u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(1u64))
        ));
        assert!(Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(7u64), BFieldElement::from(7u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ));
        assert!(!Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(4u64))
        ));
        assert!(!Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(3u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ));
        assert!(!Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(1u64), BFieldElement::from(0u64)),
            (BFieldElement::from(2u64), BFieldElement::from(3u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ));
        assert!(Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(15u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(19u64), BFieldElement::from(108u64))
        ));
        assert!(!Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(12u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(19u64), BFieldElement::from(108u64))
        ));

        // Disallow repeated x-values
        assert!(!Polynomial::<BFieldElement>::are_colinear_3(
            (BFieldElement::from(12u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(11u64), BFieldElement::from(108u64))
        ));
    }

    #[test]
    fn polynomial_are_colinear_test() {
        assert!(Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(7u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(3u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(1u64))
        ]));
        assert!(Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(7u64), BFieldElement::from(7u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(4u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(3u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(0u64)),
            (BFieldElement::from(2u64), BFieldElement::from(3u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(15u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(19u64), BFieldElement::from(108u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(12u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(19u64), BFieldElement::from(108u64))
        ]));

        // Disallow repeated x-values
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(12u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(11u64), BFieldElement::from(108u64))
        ]));

        // Disallow args with less than three points
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(12u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64))
        ]));
    }

    #[test]
    fn polynomial_are_colinear_test_big() {
        assert!(Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(7u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(3u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(1u64))
        ]));
        assert!(Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(7u64), BFieldElement::from(7u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(2u64)),
            (BFieldElement::from(3u64), BFieldElement::from(4u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(1u64)),
            (BFieldElement::from(2u64), BFieldElement::from(3u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(1u64), BFieldElement::from(0u64)),
            (BFieldElement::from(2u64), BFieldElement::from(3u64)),
            (BFieldElement::from(3u64), BFieldElement::from(3u64))
        ]));
        assert!(Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(15u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(19u64), BFieldElement::from(108u64))
        ]));
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(12u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(19u64), BFieldElement::from(108u64))
        ]));

        // Disallow repeated x-values
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(12u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64)),
            (BFieldElement::from(11u64), BFieldElement::from(108u64))
        ]));

        // Disallow args with less than three points
        assert!(!Polynomial::<BFieldElement>::are_colinear(&[
            (BFieldElement::from(12u64), BFieldElement::from(92u64)),
            (BFieldElement::from(11u64), BFieldElement::from(76u64))
        ]));
    }

    #[test]
    fn polynomial_shift_test() {
        let pol = Polynomial::<BFieldElement>::new(vec![
            BFieldElement::from(17u64),
            BFieldElement::from(14u64),
        ]);
        assert_eq!(
            vec![
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(17u64),
                BFieldElement::from(14u64)
            ],
            pol.shift_coefficients(4).coefficients
        );
        assert_eq!(
            vec![BFieldElement::from(17u64), BFieldElement::from(14u64)],
            pol.shift_coefficients(0).coefficients
        );
        assert_eq!(
            vec![
                BFieldElement::from(0u64),
                BFieldElement::from(17u64),
                BFieldElement::from(14u64)
            ],
            pol.shift_coefficients(1).coefficients
        );
    }

    #[test]
    fn mod_pow_test() {
        let zero = BFieldElement::from(0u64);
        let one = BFieldElement::from(1u64);
        let one_pol = Polynomial::<BFieldElement>::from_constant(one);

        assert_eq!(one_pol, one_pol.mod_pow(0.into()));
        assert_eq!(one_pol, one_pol.mod_pow(1.into()));
        assert_eq!(one_pol, one_pol.mod_pow(2.into()));
        assert_eq!(one_pol, one_pol.mod_pow(3.into()));

        let x = one_pol.shift_coefficients(1);
        let x_squared = one_pol.shift_coefficients(2);
        let x_cubed = one_pol.shift_coefficients(3);
        assert_eq!(x, x.mod_pow(1.into()));
        assert_eq!(x_squared, x.mod_pow(2.into()));
        assert_eq!(x_cubed, x.mod_pow(3.into()));

        let pol = Polynomial {
            coefficients: vec![
                zero,
                BFieldElement::from(14u64),
                zero,
                BFieldElement::from(4u64),
                zero,
                BFieldElement::from(8u64),
                zero,
                BFieldElement::from(3u64),
            ],
        };
        let pol_squared = Polynomial {
            coefficients: vec![
                zero,
                zero,
                BFieldElement::from(196u64),
                zero,
                BFieldElement::from(112u64),
                zero,
                BFieldElement::from(240u64),
                zero,
                BFieldElement::from(148u64),
                zero,
                BFieldElement::from(88u64),
                zero,
                BFieldElement::from(48u64),
                zero,
                BFieldElement::from(9u64),
            ],
        };
        let pol_cubed = Polynomial {
            coefficients: vec![
                zero,
                zero,
                zero,
                BFieldElement::from(2744u64),
                zero,
                BFieldElement::from(2352u64),
                zero,
                BFieldElement::from(5376u64),
                zero,
                BFieldElement::from(4516u64),
                zero,
                BFieldElement::from(4080u64),
                zero,
                BFieldElement::from(2928u64),
                zero,
                BFieldElement::from(1466u64),
                zero,
                BFieldElement::from(684u64),
                zero,
                BFieldElement::from(216u64),
                zero,
                BFieldElement::from(27u64),
            ],
        };

        assert_eq!(one_pol, pol.mod_pow(0.into()));
        assert_eq!(pol, pol.mod_pow(1.into()));
        assert_eq!(pol_squared, pol.mod_pow(2.into()));
        assert_eq!(pol_cubed, pol.mod_pow(3.into()));

        let parabola = Polynomial {
            coefficients: vec![
                BFieldElement::from(5u64),
                BFieldElement::from(41u64),
                BFieldElement::from(19u64),
            ],
        };
        let parabola_squared = Polynomial {
            coefficients: vec![
                BFieldElement::from(25u64),
                BFieldElement::from(410u64),
                BFieldElement::from(1871u64),
                BFieldElement::from(1558u64),
                BFieldElement::from(361u64),
            ],
        };
        assert_eq!(one_pol, parabola.mod_pow(0.into()));
        assert_eq!(parabola, parabola.mod_pow(1.into()));
        assert_eq!(parabola_squared, parabola.mod_pow(2.into()));
    }

    #[test]
    fn mod_pow_arbitrary_test() {
        for _ in 0..20 {
            let poly = gen_polynomial();
            for i in 0..15 {
                let actual = poly.mod_pow(i.into());
                let fast_actual = poly.fast_mod_pow(i.into());
                let mut expected = Polynomial::from_constant(BFieldElement::one());
                for _ in 0..i {
                    expected = expected.clone() * poly.clone();
                }

                assert_eq!(expected, actual);
                assert_eq!(expected, fast_actual);
            }
        }
    }

    #[test]
    fn polynomial_arithmetic_property_based_test() {
        let a_degree = 20;
        for i in 0..20 {
            // FIXME: I'm not sure why BFieldElements were converted to u64 and back here.
            let a = Polynomial::<BFieldElement> {
                coefficients: random_elements(a_degree),
            };
            let b = Polynomial::<BFieldElement> {
                coefficients: random_elements(a_degree + i),
            };

            let mul_a_b = a.clone() * b.clone();
            let mul_b_a = b.clone() * a.clone();
            let add_a_b = a.clone() + b.clone();
            let add_b_a = b.clone() + a.clone();
            let sub_a_b = a.clone() - b.clone();
            let sub_b_a = b.clone() - a.clone();

            let mut res = mul_a_b.clone() / b.clone();
            assert_eq!(res, a);
            res = mul_b_a.clone() / a.clone();
            assert_eq!(res, b);
            res = add_a_b.clone() - b.clone();
            assert_eq!(res, a);
            res = sub_a_b.clone() + b.clone();
            assert_eq!(res, a);
            res = add_b_a.clone() - a.clone();
            assert_eq!(res, b);
            res = sub_b_a.clone() + a.clone();
            assert_eq!(res, b);
            assert_eq!(add_a_b, add_b_a);
            assert_eq!(mul_a_b, mul_b_a);
            assert!(a.degree() < a_degree as isize);
            assert!(b.degree() < (a_degree + i) as isize);
            assert!(mul_a_b.degree() <= ((a_degree - 1) * 2 + i) as isize);
            assert!(add_a_b.degree() < (a_degree + i) as isize);

            let mut one = mul_a_b.clone() / mul_a_b.clone();
            assert!(one.is_one());
            one = a.clone() / a.clone();
            assert!(one.is_one());
            one = b.clone() / b.clone();
            assert!(one.is_one());
        }
    }

    #[test]
    fn polynomial_arithmetic_property_based_test_big() {
        let a_degree = 20;
        for i in 0..20 {
            // FIXME: I'm not sure why BFieldElements were converted to u64 and back here.
            let a = Polynomial::<BFieldElement> {
                coefficients: random_elements(a_degree),
            };
            let b = Polynomial::<BFieldElement> {
                coefficients: random_elements(a_degree + i),
            };

            let mul_a_b = a.clone() * b.clone();
            let mul_b_a = b.clone() * a.clone();
            let add_a_b = a.clone() + b.clone();
            let add_b_a = b.clone() + a.clone();
            let sub_a_b = a.clone() - b.clone();
            let sub_b_a = b.clone() - a.clone();

            let mut res = mul_a_b.clone() / b.clone();
            assert_eq!(res, a);
            res = mul_b_a.clone() / a.clone();
            assert_eq!(res, b);
            res = add_a_b.clone() - b.clone();
            assert_eq!(res, a);
            res = sub_a_b.clone() + b.clone();
            assert_eq!(res, a);
            res = add_b_a.clone() - a.clone();
            assert_eq!(res, b);
            res = sub_b_a.clone() + a.clone();
            assert_eq!(res, b);
            assert_eq!(add_a_b, add_b_a);
            assert_eq!(mul_a_b, mul_b_a);
            assert!(a.degree() < a_degree as isize);
            assert!(b.degree() < (a_degree + i) as isize);
            assert!(mul_a_b.degree() <= ((a_degree - 1) * 2 + i) as isize);
            assert!(add_a_b.degree() < (a_degree + i) as isize);

            let one = mul_a_b.clone() / mul_a_b.clone();
            assert!(one.is_one());
        }
    }

    // This test was used to catch a bug where the polynomial division
    // was wrong when the divisor has a leading zero coefficient, i.e.
    // when it was not normalized
    #[test]
    fn pol_div_bug_detection_test() {
        // x^3 + 18446744069414584320x + 1 / y = x
        let numerator: Polynomial<BFieldElement> = Polynomial::new(vec![
            BFieldElement::new(1),
            -BFieldElement::new(1),
            BFieldElement::new(0),
            BFieldElement::new(1),
        ]);
        let divisor_normalized =
            Polynomial::new(vec![BFieldElement::new(0), BFieldElement::new(1)]);
        let divisor_not_normalized = Polynomial::new(vec![
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(0),
        ]);

        let divisor_more_leading_zeros = Polynomial::new(vec![
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ]);

        let numerator_with_leading_zero = Polynomial::new(vec![
            BFieldElement::new(1),
            -BFieldElement::new(1),
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(0),
        ]);

        let expected = Polynomial::new(vec![
            -BFieldElement::new(1),
            BFieldElement::new(0),
            BFieldElement::new(1),
        ]);

        // Verify that the divisor need not be normalized
        let res_correct = numerator.clone() / divisor_normalized.clone();
        let res_not_normalized = numerator.clone() / divisor_not_normalized.clone();
        assert_eq!(expected, res_correct);
        assert_eq!(res_correct, res_not_normalized);
        let res_more_leading_zeros = numerator / divisor_more_leading_zeros.clone();
        assert_eq!(expected, res_more_leading_zeros);

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

    #[test]
    fn polynomial_arithmetic_division_test() {
        let a = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(17u64)],
        };
        let b = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(17u64)],
        };
        let one = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(1u64)],
        };
        let zero = Polynomial::<BFieldElement> {
            coefficients: vec![],
        };
        let zero_alt = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(0u64)],
        };
        let zero_alt_alt = Polynomial::<BFieldElement> {
            coefficients: vec![BFieldElement::from(0u64); 4],
        };
        assert_eq!(one, a / b.clone());
        let div_with_zero = zero.clone() / b.clone();
        let div_with_zero_alt = zero_alt / b.clone();
        let div_with_zero_alt_alt = zero_alt_alt / b.clone();
        assert!(div_with_zero.is_zero());
        assert!(!div_with_zero.is_one());
        assert!(div_with_zero_alt.is_zero());
        assert!(!div_with_zero_alt.is_one());
        assert!(div_with_zero_alt_alt.is_zero());
        assert!(!div_with_zero_alt_alt.is_one());
        assert!(div_with_zero.coefficients.is_empty());
        assert!(div_with_zero_alt.coefficients.is_empty());
        assert!(div_with_zero_alt_alt.coefficients.is_empty());

        let x: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![BFieldElement::from(0u64), BFieldElement::from(1u64)],
        };
        let mut prod_x = Polynomial {
            coefficients: vec![BFieldElement::from(0u64), BFieldElement::from(1u64)],
        };
        let mut expected_quotient = Polynomial {
            coefficients: vec![BFieldElement::from(1u64)],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());
        assert_eq!(zero, zero.clone() / b);

        prod_x = Polynomial {
            coefficients: vec![
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(1u64),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![BFieldElement::from(1u64)],
        };
        assert_eq!(expected_quotient, prod_x / (x.clone() * x.clone()));

        prod_x = Polynomial {
            coefficients: vec![
                BFieldElement::from(0u64),
                BFieldElement::from(1u64),
                BFieldElement::from(2u64),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![BFieldElement::from(1u64), BFieldElement::from(2u64)],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());

        prod_x = Polynomial {
            coefficients: vec![
                BFieldElement::from(1u64),
                BFieldElement::from(0u64),
                BFieldElement::from(2u64),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![BFieldElement::from(0u64), BFieldElement::from(2u64)],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());

        prod_x = Polynomial {
            coefficients: vec![
                BFieldElement::from(0u64),
                BFieldElement::from(48u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(25u64),
                BFieldElement::from(11u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(64u64),
                BFieldElement::from(16u64),
                BFieldElement::from(0u64),
                BFieldElement::from(30u64),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![
                BFieldElement::from(48u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(25u64),
                BFieldElement::from(11u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(64u64),
                BFieldElement::from(16u64),
                BFieldElement::from(0u64),
                BFieldElement::from(30u64),
            ],
        };
        assert_eq!(expected_quotient, prod_x.clone() / x.clone());

        expected_quotient = Polynomial {
            coefficients: vec![
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(25u64),
                BFieldElement::from(11u64),
                BFieldElement::from(0u64),
                BFieldElement::from(0u64),
                BFieldElement::from(64u64),
                BFieldElement::from(16u64),
                BFieldElement::from(0u64),
                BFieldElement::from(30u64),
            ],
        };
        assert_eq!(expected_quotient, prod_x.clone() / (x.clone() * x.clone()));
        assert_eq!(
            Polynomial {
                coefficients: vec![BFieldElement::from(0u64), BFieldElement::from(48u64),],
            },
            prod_x % (x.clone() * x)
        );
    }

    #[test]
    fn fast_multiply_test() {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        println!("primitive_root = {}", primitive_root);
        let a: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::from(1u64),
                BFieldElement::from(2u64),
                BFieldElement::from(3u64),
                BFieldElement::from(4u64),
                BFieldElement::from(5u64),
                BFieldElement::from(6u64),
                BFieldElement::from(7u64),
                BFieldElement::from(8u64),
                BFieldElement::from(9u64),
                BFieldElement::from(10u64),
            ],
        };
        let b: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::from(1u64),
                BFieldElement::from(2u64),
                BFieldElement::from(3u64),
                BFieldElement::from(4u64),
                BFieldElement::from(5u64),
                BFieldElement::from(6u64),
                BFieldElement::from(7u64),
                BFieldElement::from(8u64),
                BFieldElement::from(9u64),
                BFieldElement::from(17u64),
            ],
        };
        let c_fast = Polynomial::fast_multiply(&a, &b, &primitive_root, 32);
        let c_normal = a.clone() * b.clone();
        println!("c_normal = {}", c_normal);
        println!("c_fast = {}", c_fast);
        assert_eq!(c_normal, c_fast);
        assert_eq!(
            Polynomial::zero(),
            Polynomial::fast_multiply(&Polynomial::zero(), &b, &primitive_root, 32)
        );
        assert_eq!(
            Polynomial::zero(),
            Polynomial::fast_multiply(&a, &Polynomial::zero(), &primitive_root, 32)
        );

        let one: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![BFieldElement::from(1u64)],
        };
        assert_eq!(a, Polynomial::fast_multiply(&a, &one, &primitive_root, 32));
        assert_eq!(a, Polynomial::fast_multiply(&one, &a, &primitive_root, 32));
        assert_eq!(b, Polynomial::fast_multiply(&b, &one, &primitive_root, 32));
        assert_eq!(b, Polynomial::fast_multiply(&one, &b, &primitive_root, 32));
        let x: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![BFieldElement::from(0u64), BFieldElement::from(1u64)],
        };
        assert_eq!(
            a.shift_coefficients(1),
            Polynomial::fast_multiply(&x, &a, &primitive_root, 32)
        );
        assert_eq!(
            a.shift_coefficients(1),
            Polynomial::fast_multiply(&a, &x, &primitive_root, 32)
        );
        assert_eq!(
            b.shift_coefficients(1),
            Polynomial::fast_multiply(&x, &b, &primitive_root, 32)
        );
        assert_eq!(
            b.shift_coefficients(1),
            Polynomial::fast_multiply(&b, &x, &primitive_root, 32)
        );
    }

    #[test]
    fn fast_zerofier_test() {
        let _1_17 = BFieldElement::from(1u64);
        let _5_17 = BFieldElement::from(5u64);
        let root_order: usize = 8;
        let omega = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();
        let domain = vec![_1_17, _5_17];
        let actual = Polynomial::<BFieldElement>::fast_zerofier(&domain, &omega, root_order);
        assert!(
            actual.evaluate(&_1_17).is_zero(),
            "expecting {} = 0 when x = 1",
            actual
        );
        assert!(
            actual.evaluate(&_5_17).is_zero(),
            "expecting {} = 0 when x = 5",
            actual
        );
        assert!(
            !actual.evaluate(&omega).is_zero(),
            "expecting {} != 0 when x = 9",
            actual
        );

        let _7_17 = BFieldElement::from(7u64);
        let _10_17 = BFieldElement::from(10u64);
        let root_order_2 = 16;
        let omega2 = BFieldElement::primitive_root_of_unity(root_order_2 as u64).unwrap();
        let domain_2 = vec![_7_17, _10_17];
        let actual_2 = Polynomial::<BFieldElement>::fast_zerofier(&domain_2, &omega2, root_order_2);
        assert!(
            actual_2.evaluate(&_7_17).is_zero(),
            "expecting {} = 0 when x = 7",
            actual_2
        );
        assert!(
            actual_2.evaluate(&_10_17).is_zero(),
            "expecting {} = 0 when x = 10",
            actual_2
        );
    }

    #[test]
    fn fast_zerofier_pb_test() {
        let mut rng = rand::thread_rng();
        for _trial_index in 0..100 {
            let num_points: usize = rng.gen_range(1..=200);

            // sample random but distinct domain points
            let domain: Vec<BFieldElement> = random_elements_distinct(num_points);

            // prepare NTT-based methods

            // find order by rounding num_points up to the next power of 2
            let mut order = num_points << 1;
            while (order & (order - 1)) != 0 {
                order &= order - 1;
            }

            // get matching primitive nth root of unity
            let maybe_omega = BFieldElement::primitive_root_of_unity(order as u64);
            let omega = maybe_omega.unwrap();

            // compute zerofier
            let zerofier = Polynomial::<BFieldElement>::fast_zerofier(&domain, &omega, order);

            // evaluate in all domain points and match against zero
            for d in domain.iter() {
                assert_eq!(zerofier.evaluate(d), BFieldElement::zero());
            }

            // evaluate in non domain points and match against nonzer
            for point in random_elements(num_points).iter() {
                if domain.contains(point) {
                    continue;
                }
                assert_ne!(zerofier.evaluate(point), BFieldElement::zero());
            }

            // verify leading coefficient
            assert_eq!(
                zerofier.leading_coefficient().unwrap(),
                BFieldElement::one()
            );
        }
    }

    #[test]
    fn fast_evaluate_test() {
        let _0_17 = BFieldElement::from(0u64);
        let _1_17 = BFieldElement::from(1u64);
        let omega = BFieldElement::primitive_root_of_unity(16).unwrap();
        let _5_17 = BFieldElement::from(5u64);

        // x^5 + x^3
        let poly = Polynomial::<BFieldElement>::new(vec![_0_17, _0_17, _0_17, _1_17, _0_17, _1_17]);

        let _6_17 = BFieldElement::from(6u64);
        let _12_17 = BFieldElement::from(12u64);
        let domain = vec![_6_17, _12_17];

        let actual = poly.fast_evaluate(&domain, &omega, 16);
        let expected_6 = _6_17.mod_pow(5u64) + _6_17.mod_pow(3u64);
        assert_eq!(expected_6, actual[0]);

        let expected_12 = _12_17.mod_pow(5u64) + _12_17.mod_pow(3u64);
        assert_eq!(expected_12, actual[1]);
    }

    #[test]
    fn fast_evaluate_pb_test() {
        let mut rng = rand::thread_rng();
        for _trial_index in 0..100 {
            let num_points: usize = rng.gen_range(1..=200);

            // sample random but distinct domain points
            let domain = random_elements_distinct(num_points);

            // sample polynomial
            let degree: usize = rng.gen_range(0..200);
            let coefficients: Vec<BFieldElement> = random_elements(degree);
            let poly = Polynomial::<BFieldElement> { coefficients };

            // slow evaluate
            let slow_eval = domain.iter().map(|d| poly.evaluate(d)).collect_vec();

            // prepare NTT-based methods

            // find order by rounding num_points up to the next power of 2
            let mut order = num_points << 1;
            while (order & (order - 1)) != 0 {
                order &= order - 1;
            }

            // get matching primitive nth root of unity
            let maybe_omega = BFieldElement::primitive_root_of_unity(order as u64);
            let omega = maybe_omega.unwrap();

            // fast evaluate
            let fast_eval = poly.fast_evaluate(&domain, &omega, order);

            // match evaluations
            assert_eq!(slow_eval, fast_eval);
        }
    }

    #[test]
    fn fast_interpolate_test() {
        let _0_17 = BFieldElement::from(0u64);
        let _1_17 = BFieldElement::from(1u64);
        let omega = BFieldElement::primitive_root_of_unity(4).unwrap();
        let _5_17 = BFieldElement::from(5u64);

        // x^3 + x^1
        let poly = Polynomial::<BFieldElement>::new(vec![_0_17, _1_17, _0_17, _1_17]);

        let _6_17 = BFieldElement::from(6u64);
        let _7_17 = BFieldElement::from(7u64);
        let _8_17 = BFieldElement::from(8u64);
        let _9_17 = BFieldElement::from(9u64);
        let domain = vec![_6_17, _7_17, _8_17, _9_17];

        let evals = poly.fast_evaluate(&domain, &omega, 4);
        let reinterp = Polynomial::fast_interpolate(&domain, &evals, &omega, 4);
        assert_eq!(poly, reinterp);

        let reinterps_batch: Vec<Polynomial<BFieldElement>> =
            Polynomial::batch_fast_interpolate(&domain, &vec![evals], &omega, 4);
        assert_eq!(poly, reinterps_batch[0]);
    }

    #[test]
    fn fast_interpolate_pbt() {
        for num_points in [1, 2, 4, 8, 16, 32, 64, 128, 2000] {
            let domain: Vec<BFieldElement> = random_elements(num_points);
            let values: Vec<BFieldElement> = random_elements(num_points);
            let order_of_omega = other::roundup_npo2(num_points as u64) as usize;
            let omega = BFieldElement::primitive_root_of_unity(order_of_omega as u64).unwrap();

            // Unbatched fast interpolation
            let interpolant =
                Polynomial::fast_interpolate(&domain, &values, &omega, order_of_omega);

            for (x, y) in domain.iter().zip(values) {
                assert_eq!(y, interpolant.evaluate(x));
            }

            // Batched fast interpolation
            let values_vec: Vec<Vec<BFieldElement>> = vec![
                random_elements(num_points),
                random_elements(num_points),
                random_elements(num_points),
                random_elements(num_points),
                random_elements(num_points),
            ];

            let batch_interpolated =
                Polynomial::batch_fast_interpolate(&domain, &values_vec, &omega, order_of_omega);
            for (y_values, interpolant_from_batch_function) in
                values_vec.into_iter().zip(batch_interpolated.into_iter())
            {
                for (x, y) in domain.iter().zip(y_values) {
                    assert_eq!(y, interpolant_from_batch_function.evaluate(x));
                }
            }
        }
    }

    #[test]
    fn interpolate_pb_test() {
        let mut rng = rand::thread_rng();
        for _trial_index in 0..100 {
            let num_points: usize = rng.gen_range(1..=200);

            // sample random but distinct domain points
            let domain = random_elements_distinct(num_points);

            // sample random values
            let values = random_elements(num_points);

            // use lagrange interpolation
            let lagrange_interpolant =
                Polynomial::<BFieldElement>::lagrange_interpolate(&domain, &values);

            // re-evaluate and match against values
            let lagrange_re_eval = domain
                .iter()
                .map(|d| lagrange_interpolant.evaluate(d))
                .collect_vec();
            for (v, r) in values.iter().zip(lagrange_re_eval.iter()) {
                assert_eq!(v, r);
            }

            // prepare NTT-based methods

            // find order by rounding num_points up to the next power of 2
            let mut order = num_points << 1;
            while (order & (order - 1)) != 0 {
                order &= order - 1;
            }

            // get matching primitive nth root of unity
            let maybe_omega = BFieldElement::primitive_root_of_unity(order as u64);
            let omega = maybe_omega.unwrap();

            // use NTT-based interpolation
            let interpolant =
                Polynomial::<BFieldElement>::fast_interpolate(&domain, &values, &omega, order);

            // re-evaluate and match against sampled values
            let re_eval = interpolant.fast_evaluate(&domain, &omega, order);
            for (v, r) in values.iter().zip(re_eval.iter()) {
                assert_eq!(v, r);
            }

            // match against lagrange interpolation
            assert_eq!(interpolant, lagrange_interpolant);

            // Use batched-NTT-based interpolation
            let batched_interpolants = Polynomial::<BFieldElement>::batch_fast_interpolate(
                &domain,
                &vec![values],
                &omega,
                order,
            );

            // match against lagrange interpolation
            assert_eq!(batched_interpolants[0], lagrange_interpolant);
            assert_eq!(1, batched_interpolants.len())
        }
    }

    #[test]
    fn fast_coset_evaluate_test() {
        let _1 = BFieldElement::from(1u64);
        let _0 = BFieldElement::from(0u64);

        // x^5 + x^3
        let poly = Polynomial::<BFieldElement>::new(vec![_0, _0, _0, _1, _0, _1]);

        let offset = BFieldElement::generator();
        let omega = BFieldElement::primitive_root_of_unity(8).unwrap();

        let values = poly.fast_coset_evaluate(&offset, omega, 8);

        let mut domain = vec![_0; 8];
        domain[0] = offset;
        for i in 1..8 {
            domain[i] = domain[i - 1].to_owned() * omega.to_owned();
        }

        let reinterp = Polynomial::fast_interpolate(&domain, &values, &omega, 8);
        assert_eq!(reinterp, poly);

        let poly_interpolated = Polynomial::fast_coset_interpolate(&offset, omega, &values);
        assert_eq!(poly, poly_interpolated);
    }

    #[test]
    fn fast_coset_divide_test() {
        let offset = BFieldElement::primitive_root_of_unity(64).unwrap();
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        println!("primitive_root = {}", primitive_root);
        let a: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::from(1u64),
                BFieldElement::from(2u64),
                BFieldElement::from(3u64),
                BFieldElement::from(4u64),
                BFieldElement::from(5u64),
                BFieldElement::from(6u64),
                BFieldElement::from(7u64),
                BFieldElement::from(8u64),
                BFieldElement::from(9u64),
                BFieldElement::from(10u64),
            ],
        };
        let b: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::from(1u64),
                BFieldElement::from(2u64),
                BFieldElement::from(3u64),
                BFieldElement::from(4u64),
                BFieldElement::from(5u64),
                BFieldElement::from(6u64),
                BFieldElement::from(7u64),
                BFieldElement::from(8u64),
                BFieldElement::from(9u64),
                BFieldElement::from(17u64),
            ],
        };
        let c_fast = Polynomial::fast_multiply(&a, &b, &primitive_root, 32);

        let mut quotient = Polynomial::fast_coset_divide(&c_fast, &b, offset, primitive_root, 32);
        assert_eq!(a, quotient);

        quotient = Polynomial::fast_coset_divide(&c_fast, &a, offset, primitive_root, 32);
        assert_eq!(b, quotient);
    }

    #[test]
    pub fn polynomial_divide_test() {
        let minus_one = BFieldElement::P - 1;
        let zero = BFieldElement::zero();
        let one = BFieldElement::one();
        let two = BFieldElement::new(2);

        let a: Polynomial<BFieldElement> = Polynomial::new_const(BFieldElement::new(30));
        let b: Polynomial<BFieldElement> = Polynomial::new_const(BFieldElement::new(5));

        {
            let (actual_quot, actual_rem) = a.divide(b);
            let expected_quot: Polynomial<BFieldElement> =
                Polynomial::new_const(BFieldElement::new(6));
            assert_eq!(expected_quot, actual_quot);
            assert!(actual_rem.is_zero());
        }

        // Shah-polynomial test
        let shah = XFieldElement::shah_polynomial();
        let c = Polynomial::new(vec![
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::one(),
        ]);
        {
            let (actual_quot, actual_rem) = shah.divide(c);
            let expected_quot = Polynomial::new_const(BFieldElement::new(1));
            let expected_rem =
                Polynomial::new(vec![BFieldElement::one(), BFieldElement::new(minus_one)]);
            assert_eq!(expected_quot, actual_quot);
            assert_eq!(expected_rem, actual_rem);
        }

        // x^6
        let d: Polynomial<BFieldElement> = Polynomial::new(vec![one]).shift_coefficients(6);
        let (actual_sixth_quot, actual_sixth_rem) = d.divide(shah);

        // x^3 + x - 1
        let expected_sixth_quot: Polynomial<BFieldElement> =
            Polynomial::new(vec![-one, one, zero, one]);
        // x^2 - 2x + 1
        let expected_sixth_rem: Polynomial<BFieldElement> = Polynomial::new(vec![one, -two, one]);

        assert_eq!(expected_sixth_quot, actual_sixth_quot);
        assert_eq!(expected_sixth_rem, actual_sixth_rem);
    }

    #[test]
    pub fn xgcd_b_field_pol_test() {
        for _ in 0..100 {
            let x: Polynomial<BFieldElement> = gen_polynomial_non_zero();
            let y: Polynomial<BFieldElement> = gen_polynomial_non_zero();
            let (gcd, a, b): (
                Polynomial<BFieldElement>,
                Polynomial<BFieldElement>,
                Polynomial<BFieldElement>,
            ) = Polynomial::xgcd(x.clone(), y.clone());
            assert!(gcd.is_one());

            // Verify Bezout relations: ax + by = gcd
            assert_eq!(gcd, a * x + b * y);
        }
    }

    #[test]
    pub fn xgcd_x_field_pol_test() {
        for _ in 0..50 {
            let x: Polynomial<XFieldElement> = gen_polynomial_non_zero();
            let y: Polynomial<XFieldElement> = gen_polynomial_non_zero();
            let (gcd, a, b): (
                Polynomial<XFieldElement>,
                Polynomial<XFieldElement>,
                Polynomial<XFieldElement>,
            ) = Polynomial::xgcd(x.clone(), y.clone());
            assert!(gcd.is_one());

            // Verify Bezout relations: ax + by = gcd
            assert_eq!(gcd, a * x + b * y);
        }
    }

    #[test]
    fn add_assign_test() {
        for _ in 0..10 {
            let poly1: Polynomial<BFieldElement> = gen_polynomial();
            let poly2 = gen_polynomial();
            let expected = poly1.clone() + poly2.clone();
            let mut actual = poly1.clone();
            actual += poly2.clone();

            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn is_x_test() {
        let zero: Polynomial<BFieldElement> = Polynomial::zero();
        assert!(!zero.is_x());

        let one: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![BFieldElement::one()],
        };
        assert!(!one.is_x());
        let x: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![BFieldElement::zero(), BFieldElement::one()],
        };
        assert!(x.is_x());
        let x_alt: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::zero(),
                BFieldElement::one(),
                BFieldElement::zero(),
            ],
        };
        assert!(x_alt.is_x());
        let x_alt_alt: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::zero(),
                BFieldElement::one(),
                BFieldElement::zero(),
                BFieldElement::zero(),
            ],
        };
        assert!(x_alt_alt.is_x());
        let _2x: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::zero(),
                BFieldElement::one() + BFieldElement::one(),
            ],
        };
        assert!(!_2x.is_x());
        let not_x = Polynomial::<BFieldElement>::new(
            vec![14, 1, 3, 4]
                .into_iter()
                .map(BFieldElement::new)
                .collect::<Vec<BFieldElement>>(),
        );
        assert!(!not_x.is_x());
    }

    #[test]
    fn square_simple_test() {
        let coefficients = vec![14, 1, 3, 4]
            .into_iter()
            .map(BFieldElement::new)
            .collect::<Vec<BFieldElement>>();
        let poly: Polynomial<BFieldElement> = Polynomial { coefficients };
        let expected = Polynomial {
            coefficients: vec![
                14 * 14,            // 0th degree
                2 * 14,             // 1st degree
                2 * 3 * 14 + 1,     // 2nd degree
                2 * 3 + 2 * 4 * 14, // 3rd degree
                3 * 3 + 2 * 4,      // 4th degree
                2 * 3 * 4,          // 5th degree
                4 * 4,              // 6th degree
            ]
            .into_iter()
            .map(BFieldElement::new)
            .collect::<Vec<BFieldElement>>(),
        };

        assert_eq!(expected, poly.square());
        assert_eq!(expected, poly.slow_square());
    }

    #[test]
    fn fast_square_test() {
        let mut poly: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![],
        };
        assert!(poly.fast_square().is_zero());

        // square P(x) = x + 1; (P(x))^2 = (x + 1)^2 = x^2 + 2x + 1
        poly.coefficients = vec![1, 1].into_iter().map(BFieldElement::new).collect();
        let mut expected: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![1, 2, 1].into_iter().map(BFieldElement::new).collect(),
        };
        assert_eq!(expected, poly.fast_square());

        // square P(x) = x^15; (P(x))^2 = (x^15)^2 = x^30
        poly.coefficients = vec![0; 16].into_iter().map(BFieldElement::new).collect();
        poly.coefficients[15] = BFieldElement::one();
        expected.coefficients = vec![0; 32].into_iter().map(BFieldElement::new).collect();
        expected.coefficients[30] = BFieldElement::one();
        assert_eq!(expected, poly.fast_square());
    }

    #[test]
    fn square_test() {
        let one_pol = Polynomial {
            coefficients: vec![BFieldElement::one()],
        };
        for _ in 0..1000 {
            let poly = gen_polynomial() + one_pol.clone();
            let actual = poly.square();
            let fast_square_actual = poly.fast_square();
            let slow_square_actual = poly.slow_square();
            let expected = poly.clone() * poly;
            assert_eq!(expected, actual);
            assert_eq!(expected, fast_square_actual);
            assert_eq!(expected, slow_square_actual);
        }
    }

    #[test]
    fn mul_commutative_test() {
        for _ in 0..10 {
            let a: Polynomial<BFieldElement> = gen_polynomial();
            let b = gen_polynomial();
            let ab = a.clone() * b.clone();
            let ba = b.clone() * a.clone();
            assert_eq!(ab, ba);
        }
    }

    #[test]
    fn constant_zero_eq_constant_zero() {
        let zero_polynomial1 = Polynomial::<BFieldElement>::zero();
        let zero_polynomial2 = Polynomial::<BFieldElement>::zero();

        assert!(zero_polynomial1 == zero_polynomial2)
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn get_point_on_invalid_line_test() {
        let one = BFieldElement::one();
        let two = one + one;
        let three = two + one;
        Polynomial::<BFieldElement>::get_colinear_y((one, one), (one, three), two);
    }

    fn get_point_on_line_prop<FF: FiniteField>() {
        let one = FF::one();
        let two = one + one;
        let three = two + one;

        let colinear_y_1 = Polynomial::<FF>::get_colinear_y((one, one), (three, three), two);
        assert_eq!(two, colinear_y_1);

        let colinear_y_2 = Polynomial::<FF>::get_colinear_y((three, three), (one, one), two);
        assert_eq!(two, colinear_y_2);

        let colinear_y_3 = Polynomial::<FF>::get_colinear_y((one, one), (three, one), two);
        assert_eq!(one, colinear_y_3);
    }

    #[test]
    fn get_point_on_line_tests() {
        get_point_on_line_prop::<BFieldElement>();
        get_point_on_line_prop::<XFieldElement>();
    }

    fn gen_polynomial_non_zero<T: FiniteField>() -> Polynomial<T>
    where
        Standard: rand_distr::Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        let coefficient_count: usize = rng.gen_range(1..40);

        Polynomial {
            coefficients: random_elements(coefficient_count),
        }
    }

    fn gen_polynomial<T: FiniteField>() -> Polynomial<T>
    where
        Standard: rand_distr::Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        let coefficient_count: usize = rng.gen_range(0..40);

        Polynomial {
            coefficients: random_elements(coefficient_count),
        }
    }

    #[test]
    fn zero_test() {
        let mut zero_pol: Polynomial<BFieldElement> = Polynomial::zero();
        assert!(zero_pol.is_zero());

        // Verify that trailing zeros in the coefficients does not affect the `is_zero` result
        for _ in 0..12 {
            zero_pol.coefficients.push(BFieldElement::zero());
            assert!(zero_pol.is_zero());
        }

        // Verify that other constant-polynomials are not `zero`
        let rand_bs: Vec<BFieldElement> = random_elements(10);
        for rand_b in rand_bs {
            let pol: Polynomial<BFieldElement> = Polynomial {
                coefficients: vec![rand_b],
            };
            assert!(
                !pol.is_zero() || rand_b.is_zero(),
                "Pol is not zero if constant coefficient is not zero"
            );
        }
    }

    #[test]
    fn one_test() {
        let mut one_pol: Polynomial<BFieldElement> = Polynomial::one();
        assert!(one_pol.is_one(), "One must be one");

        // Verify that trailing zeros in the coefficients does not affect the `is_zero` result
        let one_pol_original = one_pol.clone();
        for _ in 0..12 {
            one_pol.coefficients.push(BFieldElement::zero());
            assert!(
                one_pol.is_one(),
                "One must be one, also with trailing zeros"
            );
            assert_eq!(
                one_pol_original, one_pol,
                "One must be equal to one with trailing zeros"
            );
        }

        // Verify that other constant-polynomials are not `one`
        let rand_bs: Vec<BFieldElement> = random_elements(10);
        for rand_b in rand_bs {
            let pol: Polynomial<BFieldElement> = Polynomial {
                coefficients: vec![rand_b],
            };
            assert!(
                !pol.is_one() || rand_b.is_one(),
                "Pol is not one if constant coefficient is not one"
            );
            assert!(0 == pol.degree() || -1 == pol.degree());
        }
    }

    #[test]
    fn lagrange_interpolate_size_one_test() {
        type BPoly = Polynomial<BFieldElement>;
        let interpoly =
            BPoly::lagrange_interpolate(&[BFieldElement::new(14)], &[BFieldElement::new(7888854)]);
        assert_eq!(
            BFieldElement::new(7888854),
            interpoly.evaluate(&BFieldElement::new(5))
        );
        assert!(interpoly.degree().is_zero());
    }

    #[test]
    fn lagrange_interpolate_test() {
        type BPoly = Polynomial<BFieldElement>;
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let num_points: usize = rng.gen_range(2..10);
            let domain = random_elements_distinct(num_points);
            let values: Vec<BFieldElement> = random_elements(num_points);
            let interpoly = BPoly::lagrange_interpolate(&domain, &values);

            assert!(num_points as isize > interpoly.degree());
            for (i, y) in values.into_iter().enumerate() {
                assert_eq!(y, interpoly.evaluate(&domain[i]));
            }
        }
    }

    #[test]
    fn fast_lagrange_interpolate_test() {
        type BPoly = Polynomial<BFieldElement>;
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let num_points: usize = rng.gen_range(2..10);
            let domain = random_elements_distinct(num_points);
            let values: Vec<BFieldElement> = random_elements(num_points);
            let interpoly = BPoly::lagrange_interpolate(&domain, &values);

            assert!(num_points as isize > interpoly.degree());
            for (i, y) in values.into_iter().enumerate() {
                assert_eq!(y, interpoly.evaluate(&domain[i]));
            }
        }
    }

    #[test]
    fn zerofier_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let num_samples: usize = rng.gen_range(2..100);
            let domain = random_elements_distinct(num_samples);

            // zerofier method
            let zerofier_polynomial = Polynomial::<BFieldElement>::zerofier(&domain);

            // verify zeros
            for domain_value in domain.iter() {
                assert!(
                    zerofier_polynomial.evaluate(domain_value).is_zero(),
                    "The zerofier polynomial evaluates to zero in the entire domain"
                );
            }

            // verify non-zeros
            for _ in 0..num_samples {
                let elem = rng.gen();
                if domain.contains(&elem) {
                    continue;
                }
                assert_ne!(zerofier_polynomial.evaluate(&elem), BFieldElement::zero())
            }

            // NTT-based fast zerofier
            let mut next_po2 = domain.len() << 1;
            while next_po2 & (next_po2 - 1) != 0 {
                next_po2 = next_po2 & (next_po2 - 1);
            }

            let omega = BFieldElement::primitive_root_of_unity(next_po2 as u64).unwrap();

            let fast_zerofier_polynomial =
                Polynomial::<BFieldElement>::fast_zerofier(&domain, &omega, next_po2);

            assert_eq!(zerofier_polynomial, fast_zerofier_polynomial);
        }
    }

    #[test]
    fn differentiate_zero() {
        let elm = BFieldElement::new(0);
        let p = Polynomial::new_const(elm);
        let q = p.formal_derivative();

        assert!(q.is_zero());
        assert_eq!(q.degree(), -1)
    }
    #[test]

    fn differentiate_const() {
        let elm = BFieldElement::new(42);
        let p = Polynomial::new_const(elm);
        let q = p.formal_derivative();

        assert!(q.is_zero());
        assert_eq!(q.degree(), -1)
    }

    #[test]
    fn differentiate_quartic() {
        let elm = BFieldElement::new(42);
        let coeffs = vec![elm, elm, elm, elm, elm];
        let p = Polynomial::new(coeffs);
        let q = p.formal_derivative();

        assert!(!q.is_zero());
        assert_eq!(q.degree(), 3);

        let manual_result = Polynomial::new(vec![
            elm,
            BFieldElement::new(2) * elm,
            BFieldElement::new(3) * elm,
            BFieldElement::new(4) * elm,
        ]);

        assert_eq!(q, manual_result)
    }

    #[test]
    fn differentiate_leibniz() {
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let terms_count_p = rng.gen_range(2..10);
            let terms_count_q = rng.gen_range(2..10);

            let rnd_coeffs_p: Vec<BFieldElement> = random_elements(terms_count_p);
            let rnd_coeffs_q: Vec<BFieldElement> = random_elements(terms_count_q);

            let p = Polynomial::new(rnd_coeffs_p);

            let q = Polynomial::new(rnd_coeffs_q);

            let pq_prime = (p.clone() * q.clone()).formal_derivative();

            let leibniz = p.formal_derivative() * q.clone() + p * q.formal_derivative();

            assert_eq!(pq_prime, leibniz)
        }
    }

    #[test]
    fn equality() {
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let terms_count_p = rng.gen_range(2..10);

            let rnd_coeffs_p: Vec<BFieldElement> = random_elements(terms_count_p);

            let mut p = Polynomial::new(rnd_coeffs_p);
            let original_p = p.clone();

            for _ in 0..4 {
                p.coefficients.push(BFieldElement::new(0));
                assert_eq!(p, original_p);
            }
        }
    }
}
