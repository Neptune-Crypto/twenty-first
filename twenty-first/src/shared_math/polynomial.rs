use crate::shared_math::ntt::{intt, ntt};
use crate::shared_math::other::{log_2_floor, roundup_npo2};
use crate::shared_math::traits::{FiniteField, ModPowU32};
use crate::utils::has_unique_elements;
use arbitrary::Arbitrary;
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
    pub fn scale(&self, alpha: BFieldElement) -> Self {
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
            None => panic!("Failed to find primitive root for order = {order}"),
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

        let lhs_log_2_of_n = log_2_floor(lhs_coefficients.len() as u128) as u32;
        let rhs_log_2_of_n = log_2_floor(rhs_coefficients.len() as u128) as u32;
        ntt::<FF>(&mut lhs_coefficients, root, lhs_log_2_of_n);
        ntt::<FF>(&mut rhs_coefficients, root, rhs_log_2_of_n);

        let mut hadamard_product: Vec<FF> = rhs_coefficients
            .into_iter()
            .zip(lhs_coefficients)
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
    pub fn fast_zerofier(domain: &[FF], primitive_root: BFieldElement, root_order: usize) -> Self {
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            BFieldElement::one(),
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {primitive_root:?}\
            Supplied order was: {root_order:?}"
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
            Supplied element was: {primitive_root:?}\
            Supplied order was: {root_order:?}"
        );

        let half = domain.len() / 2;

        let left = Self::fast_zerofier(&domain[..half], primitive_root, root_order);
        let right = Self::fast_zerofier(&domain[half..], primitive_root, root_order);
        Self::fast_multiply(&left, &right, primitive_root, root_order)
    }

    pub fn fast_evaluate(
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
        primitive_root: BFieldElement,
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
            Supplied element was: {primitive_root:?}\
            Supplied order was: {root_order:?}"
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
        offset: BFieldElement,
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
        offset: BFieldElement,
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

        poly.scale(offset.inverse())
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

        let mut scaled_lhs_coefficients: Vec<FF> = lhs.scale(offset).coefficients;
        scaled_lhs_coefficients.append(&mut vec![zero; order - scaled_lhs_coefficients.len()]);

        let mut scaled_rhs_coefficients: Vec<FF> = rhs.scale(offset).coefficients;
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

        scaled_quotient.scale(offset.inverse())
    }
}

impl<FF: FiniteField> Polynomial<FF> {
    pub const fn new(coefficients: Vec<FF>) -> Self {
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

    // Any fast interpolation will use NTT, so this is mainly used for testing/integrity
    // purposes. This also means that it is not pivotal that this function has an optimal
    // runtime.
    pub fn lagrange_interpolate_zipped(points: &[(FF, FF)]) -> Self {
        if points.is_empty() {
            panic!("Cannot interpolate through zero points.");
        }
        if !has_unique_elements(points.iter().map(|x| x.0)) {
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
            .zip_longest(other.coefficients)
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
            .zip_longest(other.coefficients)
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
        let lc = x.leading_coefficient().unwrap_or_else(FF::one);
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

    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::Rng;
    use rand_distr::Standard;
    use test_strategy::proptest;

    use crate::shared_math::other::{random_elements, random_elements_distinct};
    use crate::shared_math::traits::PrimitiveRootOfUnity;
    use crate::shared_math::x_field_element::XFieldElement;

    use super::*;

    #[test]
    fn polynomial_display_test() {
        let polynomial =
            |cs: &[u64]| Polynomial::new(cs.iter().copied().map(BFieldElement::new).collect());

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
        #[strategy(arb())] polynomial: Polynomial<BFieldElement>,
        #[strategy(arb())] leading_coefficient: BFieldElement,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let coefficients = polynomial
            .coefficients
            .into_iter()
            .chain([leading_coefficient])
            .chain([BFieldElement::zero()].repeat(num_leading_zeros))
            .collect();
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
    fn normalizing_removes_spurious_leading_zeros(
        #[strategy(arb())] polynomial: Polynomial<BFieldElement>,
        #[strategy(arb())] leading_coefficient: BFieldElement,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let coefficients = polynomial
            .coefficients
            .iter()
            .copied()
            .chain([leading_coefficient])
            .chain([BFieldElement::zero()].repeat(num_leading_zeros))
            .collect();
        let mut polynomial_with_leading_zeros = Polynomial { coefficients };
        polynomial_with_leading_zeros.normalize();

        let num_inserted_coefficients = 1;
        let expected_num_coefficients = polynomial.coefficients.len() + num_inserted_coefficients;
        let num_coefficients = polynomial_with_leading_zeros.coefficients.len();

        prop_assert_eq!(expected_num_coefficients, num_coefficients);
    }

    #[proptest]
    fn slow_lagrange_interpolation(
        #[strategy(arb())] polynomial: Polynomial<BFieldElement>,
        #[strategy(Just(#polynomial.coefficients.len().max(1)))] _min_num_points: usize,
        #[strategy(#_min_num_points..8 * #_min_num_points)] _num_points: usize,
        #[strategy(vec(arb(), #_num_points))] points: Vec<BFieldElement>,
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
        #[strategy(arb())] p0: (BFieldElement, BFieldElement),
        #[strategy(arb())]
        #[filter(#p0.0 != #p1.0)]
        p1: (BFieldElement, BFieldElement),
        #[strategy(arb())]
        #[filter(#p0.0 != #p2_x)]
        #[filter(#p1.0 != #p2_x)]
        p2_x: BFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let p2 = (p2_x, line.evaluate(&p2_x));
        prop_assert!(Polynomial::are_colinear_3(p0, p1, p2));
    }

    #[proptest]
    fn three_non_colinear_points_are_not_colinear(
        #[strategy(arb())] p0: (BFieldElement, BFieldElement),
        #[strategy(arb())]
        #[filter(#p0.0 != #p1.0)]
        p1: (BFieldElement, BFieldElement),
        #[strategy(arb())]
        #[filter(#p0.0 != #p2_x)]
        #[filter(#p1.0 != #p2_x)]
        p2_x: BFieldElement,
        #[strategy(arb())]
        #[filter(!#disturbance.is_zero())]
        disturbance: BFieldElement,
    ) {
        let line = Polynomial::lagrange_interpolate_zipped(&[p0, p1]);
        let p2 = (p2_x, line.evaluate(&p2_x) + disturbance);
        prop_assert!(!Polynomial::are_colinear_3(p0, p1, p2));
    }

    #[proptest]
    fn colinearity_check_needs_at_least_three_points(
        #[strategy(arb())] p0: (BFieldElement, BFieldElement),
        #[strategy(arb())]
        #[filter(#p0.0 != #p1.0)]
        p1: (BFieldElement, BFieldElement),
    ) {
        prop_assert!(!Polynomial::<BFieldElement>::are_colinear(&[]));
        prop_assert!(!Polynomial::are_colinear(&[p0]));
        prop_assert!(!Polynomial::are_colinear(&[p0, p1]));
    }

    #[proptest]
    fn colinearity_check_with_repeated_points_fails(
        #[strategy(arb())] p0: (BFieldElement, BFieldElement),
        #[strategy(arb())]
        #[filter(#p0.0 != #p1.0)]
        p1: (BFieldElement, BFieldElement),
    ) {
        prop_assert!(!Polynomial::are_colinear(&[p0, p1, p1]));
    }

    #[proptest]
    fn colinear_points_are_colinear(
        #[strategy(arb())] p0: (BFieldElement, BFieldElement),
        #[strategy(arb())]
        #[filter(#p0.0 != #p1.0)]
        p1: (BFieldElement, BFieldElement),
        #[strategy(1usize..50)] _num_additional_points: usize,
        #[strategy(vec(arb(), #_num_additional_points))]
        #[filter(!#additional_points_xs.contains(&#p0.0))]
        #[filter(!#additional_points_xs.contains(&#p1.0))]
        #[filter(#additional_points_xs.iter().unique().count() == #_num_additional_points)]
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

    #[proptest]
    fn shifting_polynomial_coefficients_by_zero_is_the_same_as_not_shifting_it(
        #[strategy(arb())] poly: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(poly.clone(), poly.shift_coefficients(0));
    }

    #[proptest]
    fn shifting_polynomial_one_is_equivalent_to_raising_polynomial_x_to_the_power_of_the_shift(
        #[strategy(0usize..30)] shift: usize,
    ) {
        let polynomial =
            |cs: &[u64]| Polynomial::new(cs.iter().copied().map(BFieldElement::new).collect());

        let shifted_one = Polynomial::one().shift_coefficients(shift);
        let x_to_the_shift = polynomial(&[0, 1]).mod_pow(shift.into());
        prop_assert_eq!(shifted_one, x_to_the_shift);
    }

    #[test]
    fn polynomial_shift_test() {
        let to_bfe_vec = |a: &[u64]| a.iter().copied().map(BFieldElement::new).collect_vec();

        let polynomial = Polynomial::new(to_bfe_vec(&[17, 14]));
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
        #[strategy(arb())] polynomial: Polynomial<BFieldElement>,
        #[strategy(0usize..30)] shift: usize,
    ) {
        let shifted_polynomial = polynomial.shift_coefficients(shift);
        let expected_coefficients = [0]
            .repeat(shift)
            .into_iter()
            .map(BFieldElement::new)
            .chain(polynomial.coefficients.iter().copied())
            .collect_vec();
        prop_assert_eq!(expected_coefficients, shifted_polynomial.coefficients);
    }

    #[proptest]
    fn any_polynomial_to_the_power_of_zero_is_one(
        #[strategy(arb())] poly: Polynomial<BFieldElement>,
    ) {
        let poly_to_the_zero = poly.mod_pow(0.into());
        prop_assert_eq!(Polynomial::one(), poly_to_the_zero);
    }

    #[proptest]
    fn any_polynomial_to_the_power_one_is_itself(
        #[strategy(arb())] poly: Polynomial<BFieldElement>,
    ) {
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
        let polynomial =
            |cs: &[u64]| Polynomial::new(cs.iter().copied().map(BFieldElement::new).collect());

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
        #[strategy(arb())] poly: Polynomial<BFieldElement>,
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
    fn polynomial_zero_is_neutral_element_for_addition(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.clone() + Polynomial::zero(), a);
    }

    #[proptest]
    fn polynomial_one_is_neutral_element_for_multiplication(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.clone() * Polynomial::one(), a);
    }

    #[proptest]
    fn multiplication_by_zero_is_zero(#[strategy(arb())] a: Polynomial<BFieldElement>) {
        prop_assert_eq!(Polynomial::zero(), a.clone() * Polynomial::zero());
    }

    #[proptest]
    fn polynomial_addition_is_commutative(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())] b: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.clone() + b.clone(), b + a);
    }

    #[proptest]
    fn polynomial_multiplication_is_commutative(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())] b: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.clone() * b.clone(), b * a);
    }

    #[proptest]
    fn polynomial_addition_is_associative(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())] b: Polynomial<BFieldElement>,
        #[strategy(arb())] c: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!((a.clone() + b.clone()) + c.clone(), a + (b + c));
    }

    #[proptest]
    fn polynomial_multiplication_is_associative(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())] b: Polynomial<BFieldElement>,
        #[strategy(arb())] c: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!((a.clone() * b.clone()) * c.clone(), a * (b * c));
    }

    #[proptest]
    fn polynomial_multiplication_is_distributive(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())] b: Polynomial<BFieldElement>,
        #[strategy(arb())] c: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(
            (a.clone() + b.clone()) * c.clone(),
            (a * c.clone()) + (b * c)
        );
    }

    #[proptest]
    fn polynomial_subtraction_of_self_is_zero(#[strategy(arb())] a: Polynomial<BFieldElement>) {
        prop_assert_eq!(Polynomial::zero(), a.clone() - a);
    }

    #[proptest]
    fn polynomial_division_by_self_is_one(
        #[strategy(arb())]
        #[filter(!#a.is_zero())]
        a: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(Polynomial::one(), a.clone() / a);
    }

    #[proptest]
    fn polynomial_division_removes_common_factors(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())]
        #[filter(!#b.is_zero())]
        b: Polynomial<BFieldElement>,
    ) {
        prop_assert_eq!(a.clone(), a * b.clone() / b);
    }

    #[proptest]
    fn polynomial_multiplication_raises_degree_at_maximum_to_sum_of_degrees(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())] b: Polynomial<BFieldElement>,
    ) {
        let sum_of_degrees = (a.degree() + b.degree()).max(-1);
        prop_assert!((a * b).degree() <= sum_of_degrees);
    }

    #[test]
    fn leading_zeros_dont_affect_polynomial_division() {
        let polynomial =
            |cs: &[u64]| Polynomial::new(cs.iter().copied().map(BFieldElement::new).collect());

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
    fn fast_multiplication_by_zero_gives_zero(#[strategy(arb())] poly: Polynomial<BFieldElement>) {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        let product = Polynomial::fast_multiply(&Polynomial::zero(), &poly, primitive_root, 32);
        prop_assert_eq!(Polynomial::zero(), product);
    }

    #[proptest]
    fn fast_multiplication_by_one_gives_self(#[strategy(arb())] poly: Polynomial<BFieldElement>) {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        let product = Polynomial::fast_multiply(&Polynomial::one(), &poly, primitive_root, 32);
        prop_assert_eq!(poly, product);
    }

    #[proptest]
    fn fast_multiplication_is_commutative(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())] b: Polynomial<BFieldElement>,
    ) {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        let product = Polynomial::fast_multiply(&a, &b, primitive_root, 32);
        let product_commutative = Polynomial::fast_multiply(&b, &a, primitive_root, 32);
        prop_assert_eq!(product, product_commutative);
    }

    #[proptest]
    fn fast_multiplication_and_normal_multiplication_are_equivalent(
        #[strategy(arb())] a: Polynomial<BFieldElement>,
        #[strategy(arb())] b: Polynomial<BFieldElement>,
    ) {
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        let product = Polynomial::fast_multiply(&a, &b, primitive_root, 32);
        prop_assert_eq!(a * b, product);
    }

    #[test]
    fn fast_zerofier_test() {
        let _1_17 = BFieldElement::from(1u64);
        let _5_17 = BFieldElement::from(5u64);
        let root_order: usize = 8;
        let omega = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();
        let domain = vec![_1_17, _5_17];
        let actual = Polynomial::<BFieldElement>::fast_zerofier(&domain, omega, root_order);
        assert!(
            actual.evaluate(&_1_17).is_zero(),
            "expecting {actual} = 0 when x = 1"
        );
        assert!(
            actual.evaluate(&_5_17).is_zero(),
            "expecting {actual} = 0 when x = 5"
        );
        assert!(
            !actual.evaluate(&omega).is_zero(),
            "expecting {actual} != 0 when x = 9"
        );

        let _7_17 = BFieldElement::from(7u64);
        let _10_17 = BFieldElement::from(10u64);
        let root_order_2 = 16;
        let omega2 = BFieldElement::primitive_root_of_unity(root_order_2 as u64).unwrap();
        let domain_2 = vec![_7_17, _10_17];
        let actual_2 = Polynomial::<BFieldElement>::fast_zerofier(&domain_2, omega2, root_order_2);
        assert!(
            actual_2.evaluate(&_7_17).is_zero(),
            "expecting {actual_2} = 0 when x = 7"
        );
        assert!(
            actual_2.evaluate(&_10_17).is_zero(),
            "expecting {actual_2} = 0 when x = 10"
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
            let zerofier = Polynomial::<BFieldElement>::fast_zerofier(&domain, omega, order);

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

        let actual = poly.fast_evaluate(&domain, omega, 16);
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
            let fast_eval = poly.fast_evaluate(&domain, omega, order);

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

        let evals = poly.fast_evaluate(&domain, omega, 4);
        let reinterp = Polynomial::fast_interpolate(&domain, &evals, omega, 4);
        assert_eq!(poly, reinterp);

        let reinterps_batch: Vec<Polynomial<BFieldElement>> =
            Polynomial::batch_fast_interpolate(&domain, &vec![evals], omega, 4);
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
            let interpolant = Polynomial::fast_interpolate(&domain, &values, omega, order_of_omega);

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
                Polynomial::batch_fast_interpolate(&domain, &values_vec, omega, order_of_omega);
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
                Polynomial::<BFieldElement>::fast_interpolate(&domain, &values, omega, order);

            // re-evaluate and match against sampled values
            let re_eval = interpolant.fast_evaluate(&domain, omega, order);
            for (v, r) in values.iter().zip(re_eval.iter()) {
                assert_eq!(v, r);
            }

            // match against lagrange interpolation
            assert_eq!(interpolant, lagrange_interpolant);

            // Use batched-NTT-based interpolation
            let batched_interpolants = Polynomial::<BFieldElement>::batch_fast_interpolate(
                &domain,
                &vec![values],
                omega,
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

        let values = poly.fast_coset_evaluate(offset, omega, 8);

        let mut domain = vec![_0; 8];
        domain[0] = offset;
        for i in 1..8 {
            domain[i] = domain[i - 1].to_owned() * omega.to_owned();
        }

        let reinterp = Polynomial::fast_interpolate(&domain, &values, omega, 8);
        assert_eq!(reinterp, poly);

        let poly_interpolated = Polynomial::fast_coset_interpolate(offset, omega, &values);
        assert_eq!(poly, poly_interpolated);
    }

    #[test]
    fn fast_coset_divide_test() {
        let offset = BFieldElement::primitive_root_of_unity(64).unwrap();
        let primitive_root = BFieldElement::primitive_root_of_unity(32).unwrap();
        println!("primitive_root = {primitive_root}");
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
        let c_fast = Polynomial::fast_multiply(&a, &b, primitive_root, 32);

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
    fn xgcd_does_not_panic_on_input_zero() {
        let zero = Polynomial::<BFieldElement>::zero;
        let (gcd, a, b) = Polynomial::xgcd(zero(), zero());
        assert_eq!(zero(), gcd);
        println!("a = {a}");
        println!("b = {b}");
    }

    #[proptest]
    fn xgcd_b_field_pol_test(
        #[strategy(arb())] x: Polynomial<BFieldElement>,
        #[strategy(arb())] y: Polynomial<BFieldElement>,
    ) {
        let (gcd, a, b) = Polynomial::xgcd(x.clone(), y.clone());
        // Bezout relation
        prop_assert_eq!(gcd, a * x + b * y);
    }

    #[proptest]
    fn xgcd_x_field_pol_test(
        #[strategy(arb())] x: Polynomial<XFieldElement>,
        #[strategy(arb())] y: Polynomial<XFieldElement>,
    ) {
        let (gcd, a, b) = Polynomial::xgcd(x.clone(), y.clone());
        // Bezout relation
        prop_assert_eq!(gcd, a * x + b * y);
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

        assert_eq!(zero_polynomial1, zero_polynomial2)
    }

    #[test]
    #[should_panic(expected = "assertion `left != right` failed")]
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
                Polynomial::<BFieldElement>::fast_zerofier(&domain, omega, next_po2);

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
