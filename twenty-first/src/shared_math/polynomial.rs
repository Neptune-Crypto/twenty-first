use crate::shared_math::ntt::{intt, ntt};
use crate::shared_math::other::{log_2_floor, roundup_npo2};
use crate::shared_math::traits::{IdentityValues, PrimeField};
use crate::utils::has_unique_elements;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::convert::From;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, Rem, Sub};

use super::b_field_element::BFieldElement;
use super::other::log_2_ceil;
use super::x_field_element::XFieldElement;

fn degree_raw<T: Add + Div + Mul + Sub + IdentityValues + Display>(coefficients: &[T]) -> isize {
    let mut deg = coefficients.len() as isize - 1;
    while deg >= 0 && coefficients[deg as usize].is_zero() {
        deg -= 1;
    }

    deg // -1 for the zero polynomial
}

fn pretty_print_coefficients_generic<PFElem: PrimeField>(coefficients: &[PFElem]) -> String {
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
                let one: PFElem = coefficients[pow].ring_one();
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

pub struct Polynomial<PFElem: PrimeField> {
    pub coefficients: Vec<PFElem>,
}

impl<PFElem: PrimeField> Clone for Polynomial<PFElem> {
    fn clone(&self) -> Self {
        Self {
            coefficients: self.coefficients.clone(),
        }
    }
}

impl<PFElem: PrimeField> Debug for Polynomial<PFElem> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polynomial")
            .field("coefficients", &self.coefficients)
            .finish()
    }
}

impl<PFElem: PrimeField> Hash for Polynomial<PFElem> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
    }
}

impl<PFElem: PrimeField> Display for Polynomial<PFElem> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            pretty_print_coefficients_generic::<PFElem>(&self.coefficients)
        )
    }
}

impl<PFElem: PrimeField> PartialEq for Polynomial<PFElem> {
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

impl<PFElem: PrimeField> Eq for Polynomial<PFElem> {}

impl<PFElem: PrimeField> Polynomial<PFElem> {
    pub fn new(coefficients: Vec<PFElem>) -> Self {
        Self { coefficients }
    }

    pub fn new_const(element: PFElem) -> Self {
        Self {
            coefficients: vec![element],
        }
    }

    // FIXME: Can be done with traits instead of explicitly mentioning B and X.
    // Thor does not agree that this is a good path to venture down
    pub fn lift_b_x(b_poly: &Polynomial<BFieldElement>) -> Polynomial<XFieldElement> {
        let x_field_coefficients = b_poly.coefficients.iter().map(|b| b.lift()).collect();
        Polynomial::new(x_field_coefficients)
    }

    pub fn normalize(&mut self) {
        while !self.coefficients.is_empty() && self.coefficients.last().unwrap().is_zero() {
            self.coefficients.pop();
        }
    }

    pub fn ring_zero() -> Self {
        Self {
            coefficients: vec![],
        }
    }

    pub fn from_constant(constant: PFElem) -> Self {
        Self {
            coefficients: vec![constant],
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty() || self.coefficients.iter().all(|x| x.is_zero())
    }

    pub fn is_one(&self) -> bool {
        self.degree() == 0 && self.coefficients[0].is_one()
    }

    pub fn is_x(&self) -> bool {
        self.degree() == 1 && self.coefficients[0].is_zero() && self.coefficients[1].is_one()
    }

    pub fn evaluate(&self, &x: &PFElem) -> PFElem {
        let mut acc = x.ring_zero();
        for &c in self.coefficients.iter().rev() {
            acc = c + x * acc;
        }

        acc
    }

    pub fn leading_coefficient(&self) -> Option<PFElem> {
        match self.degree() {
            -1 => None,
            n => Some(self.coefficients[n as usize]),
        }
    }

    // Return the polynomial which corresponds to the transformation `x -> alpha * x`
    // Given a polynomial P(x), produce P'(x) := P(alpha * x). Evaluating P'(x)
    // then corresponds to evaluating P(alpha * x).
    #[must_use]
    pub fn scale(&self, &alpha: &PFElem) -> Self {
        let mut acc = alpha.ring_one();
        let mut return_coefficients = self.coefficients.clone();
        for elem in return_coefficients.iter_mut() {
            *elem *= acc;
            acc *= alpha;
        }

        Self {
            coefficients: return_coefficients,
        }
    }

    pub fn lagrange_interpolation_2(
        p0: &(PFElem, PFElem),
        p1: &(PFElem, PFElem),
    ) -> (PFElem, PFElem) {
        let x_diff = p0.0 - p1.0;
        let x_diff_inv = p0.0.ring_one() / x_diff;
        let a = (p0.1 - p1.1) * x_diff_inv;
        let b = p0.1 - a * p0.0;

        (a, b)
    }

    pub fn lagrange_interpolate(domain: &[PFElem], values: &[PFElem]) -> Self {
        assert_eq!(
            domain.len(),
            values.len(),
            "The domain and values lists have to be of equal length."
        );
        assert!(
            !domain.is_empty(),
            "Trying to interpolate through 0 points."
        );

        let zero = domain[0].ring_zero();
        let one = domain[0].ring_one();
        let x = Self {
            coefficients: vec![zero, one],
        };

        let mut acc = Self::ring_zero();
        for (i, &y) in values.iter().enumerate() {
            let x_i = domain[i];
            let mut summand = Self::from_constant(y);
            for (j, &x_j) in domain.iter().enumerate() {
                if i == j {
                    continue;
                }
                summand = summand * (x.clone() - Self::from_constant(x_j))
                    / Self::from_constant(x_i - x_j);
            }
            acc += summand
        }
        acc
    }

    // fast_lagrange_interpolate
    // Faster than lagrange_interpolate, but less readable.
    pub fn fast_lagrange_interpolate(domain: &[PFElem], values: &[PFElem]) -> Self {
        assert_eq!(
            domain.len(),
            values.len(),
            "The domain and values lists have to be of equal length."
        );
        assert!(
            !domain.is_empty(),
            "Trying to interpolate through 0 points."
        );

        let zero = domain[0].ring_zero();
        let one = domain[0].ring_one();

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

    pub fn slow_lagrange_interpolation_internal(xs: &[PFElem], ys: &[PFElem]) -> Self {
        assert_eq!(
            xs.len(),
            ys.len(),
            "x and y values must have the same length"
        );
        let roots: Vec<PFElem> = xs.to_vec();
        let mut big_pol_coeffs = Self::prod_helper(&roots);
        big_pol_coeffs.reverse();
        let big_pol = Self {
            coefficients: big_pol_coeffs,
        };
        let zero: PFElem = xs[0].ring_zero();
        let one: PFElem = xs[0].ring_one();
        let mut coefficients: Vec<PFElem> = vec![zero; xs.len()];
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            // create a PrimeFieldPolynomial that is zero at all other points than this
            // coeffs_j = prod_{i=0, i != j}^{N}((x- q_i))
            let my_div_coefficients = vec![zero - x, one];
            let mut my_pol = Self {
                coefficients: my_div_coefficients,
            };
            my_pol = big_pol.clone() / my_pol;
            let mut divisor = one;
            for &root in roots.iter() {
                if root == x {
                    continue;
                }
                divisor *= x - root;
            }
            // TODO: Review.
            let mut my_coeffs: Vec<PFElem> = my_pol.coefficients.clone();
            for coeff in my_coeffs.iter_mut() {
                *coeff = coeff.to_owned() * y;
                *coeff = coeff.to_owned() / divisor;
            }
            for i in 0..my_coeffs.len() {
                coefficients[i] += my_coeffs[i];
            }
        }
        Self { coefficients }
    }

    pub fn are_colinear_3(
        p0: (PFElem, PFElem),
        p1: (PFElem, PFElem),
        p2: (PFElem, PFElem),
    ) -> bool {
        if p0.0 == p1.0 || p1.0 == p2.0 || p2.0 == p0.0 {
            return false;
        }

        let dy = p0.1 - p1.1;
        let dx = p0.0 - p1.0;

        dx * (p2.1 - p0.1) == dy * (p2.0 - p0.0)
    }

    pub fn get_colinear_y(p0: (PFElem, PFElem), p1: (PFElem, PFElem), p2_x: PFElem) -> PFElem {
        debug_assert_ne!(p0.0, p1.0, "Line must not be parallel to y-axis");
        let dy = p0.1 - p1.1;
        let dx = p0.0 - p1.0;
        let p2_y_times_dx = dy * (p2_x - p0.0) + dx * p0.1;

        // Can we implement this without division?
        p2_y_times_dx / dx
    }

    // Calculates a reversed representation of the coefficients of
    // prod_{i=0}^{N}((x- q_i))
    #[allow(clippy::assign_op_pattern)]
    fn prod_helper(input: &[PFElem]) -> Vec<PFElem> {
        if let Some((&q_j, elements)) = input.split_first() {
            let one: PFElem = q_j.ring_one();
            let zero: PFElem = q_j.ring_zero();
            let minus_q_j = zero - q_j;
            match elements {
                // base case is `x - q_j` := [1, -q_j]
                [] => vec![one, minus_q_j],
                _ => {
                    // The recursive call calculates (x-q_j)*rec = x*rec - q_j*rec := [0, rec] .- q_j*[rec]
                    let mut rec = Self::prod_helper(elements);
                    rec.push(zero);
                    let mut i = rec.len() - 1;
                    while i > 0 {
                        // The linter thinks we should fix this line, but the borrow-checker objects.
                        rec[i] = rec[i] - q_j * rec[i - 1];
                        i -= 1;
                    }
                    rec
                }
            }
        } else {
            // TODO: Shouldn't we just return one here?
            // That would require a `one` element as input
            // but maybe more correct/elegant than current
            // implementaation
            panic!("Empty array received");
        }
    }

    pub fn get_polynomial_with_roots(roots: &[PFElem], one: PFElem) -> Self {
        assert!(one.is_one(), "Provided one must be one");
        if roots.is_empty() {
            Polynomial {
                coefficients: vec![one],
            }
        } else {
            let mut coefficients = Self::prod_helper(roots);
            coefficients.reverse();
            Polynomial { coefficients }
        }
    }

    // Slow square implementation that does not use NTT
    #[must_use]
    pub fn slow_square(&self) -> Self {
        let degree = self.degree();
        if degree == -1 {
            return Self::ring_zero();
        }

        let squared_coefficient_len = self.degree() as usize * 2 + 1;
        let zero = self.coefficients[0].ring_zero();
        let one = zero.ring_one();
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

impl<PFElem: PrimeField> Polynomial<PFElem> {
    pub fn are_colinear(points: &[(PFElem, PFElem)]) -> bool {
        if points.len() < 3 {
            println!("Too few points received. Got: {} points", points.len());
            return false;
        }

        if !has_unique_elements(points.iter().map(|p| p.0)) {
            println!("Non-unique element spotted Got: {:?}", points);
            return false;
        }

        // Find 1st degree polynomial from first two points
        let one: PFElem = points[0].0.ring_one();
        let x_diff: PFElem = points[0].0 - points[1].0;
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
    pub fn lagrange_interpolate_zipped(points: &[(PFElem, PFElem)]) -> Self {
        if points.is_empty() {
            panic!("Cannot interpolate through zero points.");
        }
        if !has_unique_elements(points.iter().map(|x| x.0)) {
            panic!("Repeated x values received. Got: {:?}", points);
        }

        let xs: Vec<PFElem> = points.iter().map(|x| x.0.to_owned()).collect();
        let ys: Vec<PFElem> = points.iter().map(|x| x.1.to_owned()).collect();
        Self::lagrange_interpolate(&xs, &ys)
    }
}

impl<PFElem: PrimeField> Polynomial<PFElem> {
    // It is the caller's responsibility that this function
    // is called with sufficiently large input to be safe
    // and to be faster than `square`.
    #[must_use]
    pub fn fast_square(&self) -> Self {
        let degree = self.degree();
        if degree == -1 {
            return Self::ring_zero();
        }
        if degree == 0 {
            return Self::from_constant(self.coefficients[0] * self.coefficients[0]);
        }

        let result_degree: u64 = 2 * self.degree() as u64;
        let order = roundup_npo2(result_degree + 1);
        let (root_res, _) = self.coefficients[0].get_primitive_root_of_unity(order);
        let root = match root_res {
            Some(n) => n,
            None => panic!("Failed to find primitive root for order = {}", order),
        };

        let mut coefficients = self.coefficients.to_vec();
        coefficients.resize(order as usize, root.ring_zero());
        let log_2_of_n = log_2_floor(coefficients.len() as u128) as u32;
        ntt::<PFElem>(&mut coefficients, root, log_2_of_n);

        for element in coefficients.iter_mut() {
            *element = element.to_owned() * element.to_owned();
        }

        intt::<PFElem>(&mut coefficients, root, log_2_of_n);
        coefficients.truncate(result_degree as usize + 1);

        Polynomial { coefficients }
    }

    #[must_use]
    pub fn square(&self) -> Self {
        let degree = self.degree();
        if degree == -1 {
            return Self::ring_zero();
        }

        // A benchmark run on sword_smith's PC revealed that
        // `fast_square` was faster when the input size exceeds
        // a length of 64.
        let squared_coefficient_len = self.degree() as usize * 2 + 1;
        if squared_coefficient_len > 64 {
            return self.fast_square();
        }

        let zero = self.coefficients[0].ring_zero();
        let one = zero.ring_one();
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
    pub fn fast_mod_pow(&self, pow: BigInt, one: PFElem) -> Self {
        assert!(one.is_one(), "Provided one must be one");

        // Special case to handle 0^0 = 1
        if pow.is_zero() {
            return Self::from_constant(one);
        }

        if self.is_zero() {
            return Self::ring_zero();
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
}

impl<PFElem: PrimeField> Polynomial<PFElem> {
    // FIXME: lhs -> &self. FIXME: Change root_order: usize into : u32.
    pub fn fast_multiply(
        lhs: &Self,
        rhs: &Self,
        primitive_root: &PFElem,
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
            return Self::ring_zero();
        }

        let mut root: PFElem = primitive_root.to_owned();
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

        let mut lhs_coefficients: Vec<PFElem> = lhs.coefficients[0..lhs_degree + 1].to_vec();
        let mut rhs_coefficients: Vec<PFElem> = rhs.coefficients[0..rhs_degree + 1].to_vec();
        while lhs_coefficients.len() < order {
            lhs_coefficients.push(root.ring_zero());
        }
        while rhs_coefficients.len() < order {
            rhs_coefficients.push(root.ring_zero());
        }

        let lhs_log_2_of_n = log_2_floor(lhs_coefficients.len() as u128) as u32;
        let rhs_log_2_of_n = log_2_floor(rhs_coefficients.len() as u128) as u32;
        ntt::<PFElem>(&mut lhs_coefficients, root, lhs_log_2_of_n);
        ntt::<PFElem>(&mut rhs_coefficients, root, rhs_log_2_of_n);

        let mut hadamard_product: Vec<PFElem> = rhs_coefficients
            .into_iter()
            .zip(lhs_coefficients.into_iter())
            .map(|(r, l)| r * l)
            .collect();

        let log_2_of_n = log_2_floor(hadamard_product.len() as u128) as u32;
        intt::<PFElem>(&mut hadamard_product, root, log_2_of_n);
        hadamard_product.truncate(degree + 1);

        Polynomial {
            coefficients: hadamard_product,
        }
    }

    // domain: polynomial roots
    pub fn fast_zerofier(domain: &[PFElem], primitive_root: &PFElem, root_order: usize) -> Self {
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            primitive_root.ring_one(),
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {:?}\
            Supplied order was: {:?}",
            primitive_root,
            root_order
        );

        if domain.is_empty() {
            return Self::ring_zero();
        }

        if domain.len() == 1 {
            return Self {
                coefficients: vec![-domain[0], primitive_root.ring_one()],
            };
        }

        // This assertion must come after above recursion-ending cases have been dealt with.
        // Otherwise, the supplied primitive_root will (at some point) equal 1 with correct
        // root_order = 1, incorrectly failing the assertion.
        debug_assert_ne!(
            primitive_root.mod_pow_u32((root_order / 2) as u32),
            primitive_root.ring_one(),
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
        domain: &[PFElem],
        primitive_root: &PFElem,
        root_order: usize,
    ) -> Vec<PFElem> {
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
        domain: &[PFElem],
        values: &[PFElem],
        primitive_root: &PFElem,
        root_order: usize,
    ) -> Self {
        assert_eq!(
            domain.len(),
            values.len(),
            "Domain and values lengths must match"
        );
        debug_assert_eq!(
            primitive_root.mod_pow_u32(root_order as u32),
            primitive_root.ring_one(),
            "Supplied element “primitive_root” must have supplied order.\
            Supplied element was: {:?}\
            Supplied order was: {:?}",
            primitive_root,
            root_order
        );

        if domain.is_empty() {
            return Self::ring_zero();
        }

        if domain.len() == 1 {
            return Polynomial {
                coefficients: vec![values[0]],
            };
        }

        let half = domain.len() / 2;

        let left_zerofier = Self::fast_zerofier(&domain[..half], primitive_root, root_order);
        let right_zerofier = Self::fast_zerofier(&domain[half..], primitive_root, root_order);

        let left_offset: Vec<PFElem> =
            Self::fast_evaluate(&right_zerofier, &domain[..half], primitive_root, root_order);
        let right_offset: Vec<PFElem> =
            Self::fast_evaluate(&left_zerofier, &domain[half..], primitive_root, root_order);

        let left_targets: Vec<PFElem> = values[..half]
            .iter()
            .zip(left_offset)
            .map(|(n, d)| n.to_owned() / d)
            .collect();
        let right_targets: Vec<PFElem> = values[half..]
            .iter()
            .zip(right_offset)
            .map(|(n, d)| n.to_owned() / d)
            .collect();

        let left_interpolant =
            Self::fast_interpolate(&domain[..half], &left_targets, primitive_root, root_order);
        let right_interpolant =
            Self::fast_interpolate(&domain[half..], &right_targets, primitive_root, root_order);

        left_interpolant * right_zerofier + right_interpolant * left_zerofier
    }

    /// Fast evaluate on a coset domain, which is the group generated by `generator^i * offset`
    pub fn fast_coset_evaluate(
        &self,
        offset: &PFElem,
        generator: PFElem,
        order: usize,
    ) -> Vec<PFElem> {
        let mut coefficients = self.scale(offset).coefficients;
        coefficients.append(&mut vec![generator.ring_zero(); order - coefficients.len()]);
        let log_2_of_n = log_2_floor(coefficients.len() as u128) as u32;
        ntt::<PFElem>(&mut coefficients, generator, log_2_of_n);
        coefficients
    }

    /// The inverse of `fast_coset_evaluate`
    pub fn fast_coset_interpolate(offset: &PFElem, generator: PFElem, values: &[PFElem]) -> Self {
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
        lhs: &Polynomial<PFElem>,
        rhs: &Polynomial<PFElem>,
        offset: PFElem,
        primitive_root: PFElem,
        root_order: usize,
    ) -> Polynomial<PFElem> {
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

        let zero = lhs.coefficients[0].ring_zero();
        let mut root: PFElem = primitive_root.to_owned();
        let mut order = root_order;
        let degree: usize = lhs.degree() as usize;

        if degree < 8 {
            return lhs.to_owned() / rhs.to_owned();
        }

        while degree < order / 2 {
            root *= root;
            order /= 2;
        }

        let mut scaled_lhs_coefficients: Vec<PFElem> = lhs.scale(&offset).coefficients;
        scaled_lhs_coefficients.append(&mut vec![zero; order - scaled_lhs_coefficients.len()]);

        let mut scaled_rhs_coefficients: Vec<PFElem> = rhs.scale(&offset).coefficients;
        scaled_rhs_coefficients.append(&mut vec![zero; order - scaled_rhs_coefficients.len()]);

        let lhs_log_2_of_n = log_2_floor(scaled_lhs_coefficients.len() as u128) as u32;
        let rhs_log_2_of_n = log_2_floor(scaled_rhs_coefficients.len() as u128) as u32;

        ntt::<PFElem>(&mut scaled_lhs_coefficients, root, lhs_log_2_of_n);
        ntt::<PFElem>(&mut scaled_rhs_coefficients, root, rhs_log_2_of_n);

        let rhs_inverses = PFElem::batch_inversion(scaled_rhs_coefficients);
        let mut quotient_codeword: Vec<PFElem> = scaled_lhs_coefficients
            .iter()
            .zip(rhs_inverses)
            .map(|(l, r)| l.to_owned() * r)
            .collect();

        let log_2_of_n = log_2_floor(quotient_codeword.len() as u128) as u32;
        intt::<PFElem>(&mut quotient_codeword, root, log_2_of_n);

        let scaled_quotient = Polynomial {
            coefficients: quotient_codeword,
        };

        scaled_quotient.scale(&(zero.ring_one() / offset.to_owned()))
    }
}

impl<PFElem: PrimeField> Polynomial<PFElem> {
    pub fn multiply(self, other: Self) -> Self {
        let degree_lhs = self.degree();
        let degree_rhs = other.degree();

        if degree_lhs < 0 || degree_rhs < 0 {
            return Self::ring_zero();
            // return self.zero();
        }

        // allocate right number of coefficients, initialized to zero
        let elem = self.coefficients[0];
        let mut result_coeff: Vec<PFElem> =
            //vec![U::zero_from_field(field: U); degree_lhs as usize + degree_rhs as usize + 1];
            vec![elem.ring_zero(); degree_lhs as usize + degree_rhs as usize + 1];

        // TODO: Review this.
        // for all pairs of coefficients, add product to result vector in appropriate coordinate
        for i in 0..=degree_lhs as usize {
            for j in 0..=degree_rhs as usize {
                let mul: PFElem = self.coefficients[i] * other.coefficients[j];
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
    pub fn mod_pow(&self, pow: BigInt, one: PFElem) -> Self {
        assert!(one.is_one(), "Provided one must be one");

        // Special case to handle 0^0 = 1
        if pow.is_zero() {
            return Self::from_constant(one);
        }

        if self.is_zero() {
            return Self::ring_zero();
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

    pub fn shift_coefficients_mut(&mut self, power: usize, zero: PFElem) {
        self.coefficients.splice(0..0, vec![zero; power]);
    }

    // Multiply a polynomial with x^power
    #[must_use]
    pub fn shift_coefficients(&self, power: usize, zero: PFElem) -> Self {
        if !zero.is_zero() {
            panic!("`zero` was not zero. Don't do this.");
        }

        let mut coefficients: Vec<PFElem> = self.coefficients.clone();
        coefficients.splice(0..0, vec![zero; power]);
        Polynomial { coefficients }
    }

    // TODO: Review
    pub fn scalar_mul_mut(&mut self, scalar: PFElem) {
        for coefficient in self.coefficients.iter_mut() {
            *coefficient *= scalar;
        }
    }

    #[must_use]
    pub fn scalar_mul(&self, scalar: PFElem) -> Self {
        let mut coefficients: Vec<PFElem> = vec![];
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
            return (Self::ring_zero(), Self::ring_zero());
        }

        // quotient is built from back to front so must be reversed
        // Preallocate space for quotient coefficients
        let mut quotient: Vec<PFElem> = if degree_lhs - degree_rhs >= 0 {
            Vec::with_capacity((degree_lhs - degree_rhs + 1) as usize)
        } else {
            vec![]
        };
        let mut remainder = self.clone();
        remainder.normalize();

        // a divisor coefficient is guaranteed to exist since the divisor is non-zero
        let dlc: PFElem = divisor.leading_coefficient().unwrap();
        let inv = dlc.ring_one() / dlc;

        let mut i = 0;
        while i + degree_rhs <= degree_lhs {
            // calculate next quotient coefficient, and set leading coefficient
            // of remainder remainder is 0 by removing it
            let rlc: PFElem = remainder.coefficients.last().unwrap().to_owned();
            let q: PFElem = rlc * inv;
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

impl<PFElem: PrimeField> Div for Polynomial<PFElem> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let (quotient, _): (Self, Self) = self.divide(other);
        quotient
    }
}

impl<PFElem: PrimeField> Rem for Polynomial<PFElem> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        let (_, remainder): (Self, Self) = self.divide(other);
        remainder
    }
}

impl<PFElem: PrimeField> Add for Polynomial<PFElem> {
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
        let summed: Vec<PFElem> = self
            .coefficients
            .into_iter()
            .zip_longest(other.coefficients.into_iter())
            .map(|a: itertools::EitherOrBoth<PFElem, PFElem>| match a {
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

impl<PFElem: PrimeField> AddAssign for Polynomial<PFElem> {
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

impl<PFElem: PrimeField> Sub for Polynomial<PFElem> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let summed: Vec<PFElem> = self
            .coefficients
            .into_iter()
            .zip_longest(other.coefficients.into_iter())
            .map(|a: itertools::EitherOrBoth<PFElem, PFElem>| match a {
                Both(l, r) => l - r,
                Left(l) => l,
                Right(r) => r.ring_zero() - r,
            })
            .collect();

        Self {
            coefficients: summed,
        }
    }
}

impl<PFElem: PrimeField> Polynomial<PFElem> {
    pub fn degree(&self) -> isize {
        degree_raw(&self.coefficients)
    }
}

impl<PFElem: PrimeField> Mul for Polynomial<PFElem> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::multiply(self, other)
    }
}

#[cfg(test)]
mod test_polynomials {
    #![allow(clippy::just_underscores_and_digits)]

    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::prime_field_element_flexible::PrimeFieldElementFlexible;
    use crate::shared_math::traits::GetRandomElements;
    use crate::shared_math::traits::{GetPrimitiveRootOfUnity, IdentityValues};
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::timing_reporter::TimingReporter;
    use crate::utils::generate_random_numbers;
    use primitive_types::U256;
    use rand::RngCore;
    use std::cmp::max;
    use std::vec;

    fn pfb(n: i64, q: u64) -> PrimeFieldElementFlexible {
        let q_u256: U256 = q.into();
        if n < 0 {
            let positive_n: U256 = (-n).into();
            let field_element_n: U256 = positive_n % q_u256;

            -PrimeFieldElementFlexible::new(field_element_n, q_u256)
        } else {
            let positive_n: U256 = n.into();
            let field_element_n: U256 = positive_n % q_u256;
            PrimeFieldElementFlexible::new(field_element_n, q_u256)
        }
    }

    #[test]
    fn polynomial_display_test() {
        let q = 71;
        let empty = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![],
        };
        assert_eq!("0", empty.to_string());
        let zero = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(0, q)],
        };
        assert_eq!("0", zero.to_string());
        let double_zero = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(0, q), pfb(0, q)],
        };
        assert_eq!("0", double_zero.to_string());
        let one = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(1, q)],
        };
        assert_eq!("1", one.to_string());
        let zero_one = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(1, q), pfb(0, q)],
        };
        assert_eq!("1", zero_one.to_string());
        let zero_zero_one = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(1, q), pfb(0, q), pfb(0, q)],
        };
        assert_eq!("1", zero_zero_one.to_string());
        let one_zero = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(0, q), pfb(1, q)],
        };
        assert_eq!("x", one_zero.to_string());
        let one = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(1, q)],
        };
        assert_eq!("1", one.to_string());
        let x_plus_one = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(1, q), pfb(1, q)],
        };
        assert_eq!("x + 1", x_plus_one.to_string());
        let many_zeros = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(1, q), pfb(1, q), pfb(0, q), pfb(0, q), pfb(0, q)],
        };
        assert_eq!("x + 1", many_zeros.to_string());
        let also_many_zeros = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(0, q), pfb(0, q), pfb(0, q), pfb(1, q), pfb(1, q)],
        };
        assert_eq!("x^4 + x^3", also_many_zeros.to_string());
        let yet_many_zeros = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(1, q), pfb(0, q), pfb(0, q), pfb(0, q), pfb(1, q)],
        };
        assert_eq!("x^4 + 1", yet_many_zeros.to_string());
    }

    #[test]
    fn polynomial_evaluate_test_big() {
        let q = 71;
        let parabola = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(7, q), pfb(3, q), pfb(2, q)],
        };
        assert_eq!(pfb(7, q), parabola.evaluate(&pfb(0, q)));
        assert_eq!(pfb(12, q), parabola.evaluate(&pfb(1, q)));
        assert_eq!(pfb(21, q), parabola.evaluate(&pfb(2, q)));
        assert_eq!(pfb(34, q), parabola.evaluate(&pfb(3, q)));
        assert_eq!(pfb(51, q), parabola.evaluate(&pfb(4, q)));
        assert_eq!(pfb(1, q), parabola.evaluate(&pfb(5, q)));
        assert_eq!(pfb(26, q), parabola.evaluate(&pfb(6, q)));
    }

    #[test]
    fn polynomial_evaluate_test() {
        let q = 71;
        let parabola = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(7, q), pfb(3, q), pfb(2, q)],
        };
        assert_eq!(pfb(7, q), parabola.evaluate(&pfb(0, q)));
        assert_eq!(pfb(12, q), parabola.evaluate(&pfb(1, q)));
        assert_eq!(pfb(21, q), parabola.evaluate(&pfb(2, q)));
        assert_eq!(pfb(34, q), parabola.evaluate(&pfb(3, q)));
        assert_eq!(pfb(51, q), parabola.evaluate(&pfb(4, q)));
        assert_eq!(pfb(1, q), parabola.evaluate(&pfb(5, q)));
        assert_eq!(pfb(26, q), parabola.evaluate(&pfb(6, q)));
    }

    #[test]
    fn leading_coefficient_test() {
        // Verify that the leading coefficient for the zero-polynomial is `None`
        let _14 = BFieldElement::new(14);
        let _0 = BFieldElement::ring_zero();
        let _1 = BFieldElement::ring_one();
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
    fn scale_test() {
        let q = 71;
        let _0_71 = pfb(0, q);
        let _1_71 = pfb(1, q);
        let _2_71 = pfb(2, q);
        let _3_71 = pfb(3, q);
        let _6_71 = pfb(6, q);
        let _12_71 = pfb(12, q);
        let _36_71 = pfb(36, q);
        let _37_71 = pfb(37, q);
        let _40_71 = pfb(40, q);
        let mut input: Polynomial<PrimeFieldElementFlexible> = Polynomial::new(vec![_1_71, _6_71]);

        let mut expected = Polynomial {
            coefficients: vec![_1_71, _12_71],
        };

        assert_eq!(expected, input.scale(&_2_71));

        input = Polynomial::ring_zero();
        expected = Polynomial::ring_zero();
        assert_eq!(expected, input.scale(&_2_71));

        input = Polynomial {
            coefficients: vec![_12_71, _12_71, _12_71, _12_71],
        };
        expected = Polynomial {
            coefficients: vec![_12_71, _36_71, _37_71, _40_71],
        };
        assert_eq!(expected, input.scale(&_3_71));
    }

    #[test]
    fn normalize_test() {
        let q = 71;
        let _0_71 = pfb(0, q);
        let _1_71 = pfb(1, q);
        let _6_71 = pfb(6, q);
        let _12_71 = pfb(12, q);
        let zero: Polynomial<PrimeFieldElementFlexible> = Polynomial::ring_zero();
        let mut mut_one: Polynomial<PrimeFieldElementFlexible> =
            Polynomial::<PrimeFieldElementFlexible> {
                coefficients: vec![_1_71],
            };
        let one: Polynomial<PrimeFieldElementFlexible> = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_1_71],
        };
        let mut a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![],
        };
        a.normalize();
        assert_eq!(zero, a);
        mut_one.normalize();
        assert_eq!(one, mut_one);

        // trailing zeros are removed
        a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_1_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElementFlexible> {
                coefficients: vec![_1_71],
            },
            a
        );
        a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_1_71, _0_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElementFlexible> {
                coefficients: vec![_1_71],
            },
            a
        );

        // but leading zeros are not removed
        a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_0_71, _1_71, _0_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElementFlexible> {
                coefficients: vec![_0_71, _1_71],
            },
            a
        );
        a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_0_71, _1_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElementFlexible> {
                coefficients: vec![_0_71, _1_71],
            },
            a
        );
        a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElementFlexible> {
                coefficients: vec![],
            },
            a
        );
    }

    fn poly_flex(
        coefficients: Vec<PrimeFieldElementFlexible>,
    ) -> Polynomial<PrimeFieldElementFlexible> {
        Polynomial::<PrimeFieldElementFlexible>::new(coefficients)
    }

    #[test]
    fn get_polynomial_with_roots_test() {
        let q = 31;
        assert_eq!(
            poly_flex(vec![pfb(30, q), pfb(0, q), pfb(1, q)]),
            Polynomial::get_polynomial_with_roots(&[pfb(1, q), pfb(30, q)], pfb(1, q))
        );
        assert_eq!(
            poly_flex(vec![pfb(0, q), pfb(30, q), pfb(0, q), pfb(1, q)]),
            Polynomial::get_polynomial_with_roots(&[pfb(1, q), pfb(30, q), pfb(0, q)], pfb(1, q))
        );
        assert_eq!(
            poly_flex(vec![pfb(25, q), pfb(11, q), pfb(25, q), pfb(1, q)]),
            Polynomial::get_polynomial_with_roots(&[pfb(1, q), pfb(2, q), pfb(3, q)], pfb(1, q))
        );
        assert_eq!(
            poly_flex(vec![pfb(1, q)]),
            Polynomial::get_polynomial_with_roots(&[], pfb(1, q))
        );
    }

    #[should_panic]
    #[test]
    fn panic_when_one_is_not_one() {
        let q = 31;
        assert_eq!(
            poly_flex(vec![pfb(30, q), pfb(0, q), pfb(1, q)]),
            Polynomial::get_polynomial_with_roots(&[pfb(1, q), pfb(30, q)], pfb(14, q))
        );
    }

    #[test]
    fn slow_lagrange_interpolation_test() {
        let q = 7;

        // Verify that interpolation works with just one point
        let one_point = &[(pfb(2, q), pfb(5, q))];
        let mut interpolation_result = Polynomial::lagrange_interpolate_zipped(one_point);
        println!("interpolation_result = {}", interpolation_result);
        let mut expected_result = Polynomial {
            coefficients: vec![pfb(5, q)],
        };
        assert_eq!(expected_result, interpolation_result);

        // Test with three points
        let points = &[
            (pfb(0, q), pfb(6, q)),
            (pfb(1, q), pfb(6, q)),
            (pfb(2, q), pfb(2, q)),
        ];

        interpolation_result = Polynomial::lagrange_interpolate_zipped(points);
        expected_result = Polynomial {
            coefficients: vec![pfb(6, q), pfb(2, q), pfb(5, q)],
        };
        assert_eq!(expected_result, interpolation_result);

        // Use the same numbers to test evaluations
        for point in points.iter() {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }

        // Test linear interpolation, when there are only two points given as input
        let two_points = &[(pfb(0, q), pfb(6, q)), (pfb(2, q), pfb(2, q))];
        interpolation_result = Polynomial::lagrange_interpolate_zipped(two_points);
        expected_result = Polynomial {
            coefficients: vec![pfb(6, q), pfb(5, q)],
        };
        assert_eq!(expected_result, interpolation_result);
    }

    #[test]
    fn slow_lagrange_interpolation_test_big() {
        let q = 7;
        let points = &[
            (pfb(0, q), pfb(6, q)),
            (pfb(1, q), pfb(6, q)),
            (pfb(2, q), pfb(2, q)),
        ];

        let interpolation_result: Polynomial<PrimeFieldElementFlexible> =
            Polynomial::lagrange_interpolate_zipped(points);
        let expected_result = Polynomial {
            coefficients: vec![pfb(6, q), pfb(2, q), pfb(5, q)],
        };
        assert_eq!(expected_result, interpolation_result);

        // Use the same numbers to test evaluation
        for point in points.iter() {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }
    }

    #[test]
    fn property_based_slow_lagrange_interpolation_test() {
        // Autogenerate a `number_of_points - 1` degree polynomial
        // We start by autogenerating the polynomial, as we would get a polynomial
        // with fractional coefficients if we autogenerated the points and derived the polynomium
        // from that.
        let q: u64 = 999983;
        let number_of_points = 50usize;
        let coefficients: Vec<PrimeFieldElementFlexible> =
            generate_random_numbers(number_of_points, q as i128)
                .iter()
                .map(|x| pfb(*x as i64, q))
                .collect();

        let pol: Polynomial<PrimeFieldElementFlexible> = Polynomial { coefficients };

        // Evaluate polynomial in `number_of_points` points
        let points = (0..number_of_points)
            .map(|x| {
                let x = pfb(x as i64, q);
                (x, pol.evaluate(&x))
            })
            .collect::<Vec<(PrimeFieldElementFlexible, PrimeFieldElementFlexible)>>();

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result: Polynomial<PrimeFieldElementFlexible> =
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
        let q: u64 = 999983;
        let number_of_points = 50usize;
        let coefficients: Vec<PrimeFieldElementFlexible> =
            generate_random_numbers(number_of_points, q as i128)
                .iter()
                .map(|x| pfb(*x as i64, q))
                .collect();

        let pol: Polynomial<PrimeFieldElementFlexible> = Polynomial { coefficients };

        // Evaluate polynomial in `number_of_points` points
        let points: Vec<(PrimeFieldElementFlexible, PrimeFieldElementFlexible)> = (0
            ..number_of_points)
            .map(|x| {
                let x = pfb(x as i64, q);
                (x, pol.evaluate(&x))
            })
            .collect();

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result: Polynomial<PrimeFieldElementFlexible> =
            Polynomial::lagrange_interpolate_zipped(&points);
        assert_eq!(interpolation_result, pol);
        for point in points {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }
    }

    #[test]
    fn lagrange_interpolation_2_test() {
        let q = 5;
        assert_eq!(
            (pfb(1, q), pfb(0, q)),
            Polynomial::<PrimeFieldElementFlexible>::lagrange_interpolation_2(
                &(pfb(1, q), pfb(1, q)),
                &(pfb(2, q), pfb(2, q))
            )
        );
        assert_eq!(
            (pfb(4, q), pfb(4, q)),
            Polynomial::<PrimeFieldElementFlexible>::lagrange_interpolation_2(
                &(pfb(1, q), pfb(3, q)),
                &(pfb(2, q), pfb(2, q))
            )
        );
        assert_eq!(
            (pfb(4, q), pfb(2, q)),
            Polynomial::<PrimeFieldElementFlexible>::lagrange_interpolation_2(
                &(pfb(15, q), pfb(92, q)),
                &(pfb(19, q), pfb(108, q))
            )
        );

        assert_eq!(
            (pfb(3, q), pfb(2, q)),
            Polynomial::<PrimeFieldElementFlexible>::lagrange_interpolation_2(
                &(pfb(1, q), pfb(0, q)),
                &(pfb(2, q), pfb(3, q))
            )
        );

        let q = 5;
        assert_eq!(
            (pfb(1, q), pfb(0, q)),
            Polynomial::<PrimeFieldElementFlexible>::lagrange_interpolation_2(
                &(pfb(1, q), pfb(1, q)),
                &(pfb(2, q), pfb(2, q))
            )
        );
        assert_eq!(
            (pfb(4, q), pfb(4, q)),
            Polynomial::<PrimeFieldElementFlexible>::lagrange_interpolation_2(
                &(pfb(1, q), pfb(3, q)),
                &(pfb(2, q), pfb(2, q))
            )
        );
        assert_eq!(
            (pfb(4, q), pfb(2, q)),
            Polynomial::<PrimeFieldElementFlexible>::lagrange_interpolation_2(
                &(pfb(15, q), pfb(92, q)),
                &(pfb(19, q), pfb(108, q))
            )
        );
        assert_eq!(
            (pfb(3, q), pfb(2, q)),
            Polynomial::<PrimeFieldElementFlexible>::lagrange_interpolation_2(
                &(pfb(1, q), pfb(0, q)),
                &(pfb(2, q), pfb(3, q))
            )
        );
    }

    #[test]
    fn polynomial_are_colinear_3_test() {
        let q = 5;
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(3, q))
        ));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(7, q)),
            (pfb(3, q), pfb(3, q))
        ));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(1, q), pfb(3, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(1, q))
        ));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(1, q), pfb(1, q)),
            (pfb(7, q), pfb(7, q)),
            (pfb(3, q), pfb(3, q))
        ));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(4, q))
        ));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(3, q)),
            (pfb(3, q), pfb(3, q))
        ));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(1, q), pfb(0, q)),
            (pfb(2, q), pfb(3, q)),
            (pfb(3, q), pfb(3, q))
        ));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(15, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(19, q), pfb(108, q))
        ));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(12, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(19, q), pfb(108, q))
        ));

        // Disallow repeated x-values
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear_3(
            (pfb(12, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(11, q), pfb(108, q))
        ));
    }

    #[test]
    fn polynomial_are_colinear_test() {
        let q = 5;
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(7, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(3, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(1, q))
        ]));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(7, q), pfb(7, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(4, q))
        ]));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(3, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(0, q)),
            (pfb(2, q), pfb(3, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(15, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(19, q), pfb(108, q))
        ]));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(12, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(19, q), pfb(108, q))
        ]));

        // Disallow repeated x-values
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(12, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(11, q), pfb(108, q))
        ]));

        // Disallow args with less than three points
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(12, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q))
        ]));
    }

    #[test]
    fn polynomial_are_colinear_test_big() {
        let q = 5;
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(7, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(3, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(1, q))
        ]));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(7, q), pfb(7, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(2, q)),
            (pfb(3, q), pfb(4, q))
        ]));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(1, q)),
            (pfb(2, q), pfb(3, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(1, q), pfb(0, q)),
            (pfb(2, q), pfb(3, q)),
            (pfb(3, q), pfb(3, q))
        ]));
        assert!(Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(15, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(19, q), pfb(108, q))
        ]));
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(12, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(19, q), pfb(108, q))
        ]));

        // Disallow repeated x-values
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(12, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q)),
            (pfb(11, q), pfb(108, q))
        ]));

        // Disallow args with less than three points
        assert!(!Polynomial::<PrimeFieldElementFlexible>::are_colinear(&[
            (pfb(12, q), pfb(92, q)),
            (pfb(11, q), pfb(76, q))
        ]));
    }

    #[test]
    fn polynomial_shift_test() {
        let q = 71;
        let pol = poly_flex(vec![pfb(17, q), pfb(14, q)]);
        assert_eq!(
            vec![
                pfb(0, q),
                pfb(0, q),
                pfb(0, q),
                pfb(0, q),
                pfb(17, q),
                pfb(14, q)
            ],
            pol.shift_coefficients(4, pfb(0, q)).coefficients
        );
        assert_eq!(
            vec![pfb(17, q), pfb(14, q)],
            pol.shift_coefficients(0, pfb(0, q)).coefficients
        );
        assert_eq!(
            vec![pfb(0, q), pfb(17, q), pfb(14, q)],
            pol.shift_coefficients(1, pfb(0, q)).coefficients
        );
    }

    #[test]
    fn mod_pow_test() {
        let q = 71;
        let zero = pfb(0, q);
        let one = pfb(1, q);
        let one_pol = Polynomial::<PrimeFieldElementFlexible>::from_constant(one);

        assert_eq!(one_pol, one_pol.mod_pow(0.into(), one));
        assert_eq!(one_pol, one_pol.mod_pow(1.into(), one));
        assert_eq!(one_pol, one_pol.mod_pow(2.into(), one));
        assert_eq!(one_pol, one_pol.mod_pow(3.into(), one));

        let x = one_pol.shift_coefficients(1, zero);
        let x_squared = one_pol.shift_coefficients(2, zero);
        let x_cubed = one_pol.shift_coefficients(3, zero);
        assert_eq!(x, x.mod_pow(1.into(), one));
        assert_eq!(x_squared, x.mod_pow(2.into(), one));
        assert_eq!(x_cubed, x.mod_pow(3.into(), one));

        let pol = Polynomial {
            coefficients: vec![
                zero,
                pfb(14, q),
                zero,
                pfb(4, q),
                zero,
                pfb(8, q),
                zero,
                pfb(3, q),
            ],
        };
        let pol_squared = Polynomial {
            coefficients: vec![
                zero,
                zero,
                pfb(196, q),
                zero,
                pfb(112, q),
                zero,
                pfb(240, q),
                zero,
                pfb(148, q),
                zero,
                pfb(88, q),
                zero,
                pfb(48, q),
                zero,
                pfb(9, q),
            ],
        };
        let pol_cubed = Polynomial {
            coefficients: vec![
                zero,
                zero,
                zero,
                pfb(2744, q),
                zero,
                pfb(2352, q),
                zero,
                pfb(5376, q),
                zero,
                pfb(4516, q),
                zero,
                pfb(4080, q),
                zero,
                pfb(2928, q),
                zero,
                pfb(1466, q),
                zero,
                pfb(684, q),
                zero,
                pfb(216, q),
                zero,
                pfb(27, q),
            ],
        };

        assert_eq!(one_pol, pol.mod_pow(0.into(), one));
        assert_eq!(pol, pol.mod_pow(1.into(), one));
        assert_eq!(pol_squared, pol.mod_pow(2.into(), one));
        assert_eq!(pol_cubed, pol.mod_pow(3.into(), one));

        let parabola = Polynomial {
            coefficients: vec![pfb(5, q), pfb(41, q), pfb(19, q)],
        };
        let parabola_squared = Polynomial {
            coefficients: vec![
                pfb(25, q),
                pfb(410, q),
                pfb(1871, q),
                pfb(1558, q),
                pfb(361, q),
            ],
        };
        assert_eq!(one_pol, parabola.mod_pow(0.into(), one));
        assert_eq!(parabola, parabola.mod_pow(1.into(), one));
        assert_eq!(parabola_squared, parabola.mod_pow(2.into(), one));
    }

    #[test]
    fn mod_pow_arbitrary_test() {
        for _ in 0..20 {
            let poly = gen_polynomial();
            for i in 0..15 {
                let actual = poly.mod_pow(i.into(), BFieldElement::ring_one());
                let fast_actual = poly.fast_mod_pow(i.into(), BFieldElement::ring_one());
                let mut expected = Polynomial::from_constant(BFieldElement::ring_one());
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
        let q: u64 = 71;
        let a_degree = 20;
        for i in 0..20 {
            let a = Polynomial::<PrimeFieldElementFlexible> {
                coefficients: generate_random_numbers(a_degree, q as i128)
                    .iter()
                    .map(|x| pfb(*x as i64, q))
                    .collect(),
            };
            let b = Polynomial::<PrimeFieldElementFlexible> {
                coefficients: generate_random_numbers(a_degree + i, q as i128)
                    .iter()
                    .map(|x| pfb(*x as i64, q))
                    .collect(),
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
        let q: u64 = 71;
        let a_degree = 20;
        for i in 0..20 {
            let a = Polynomial::<PrimeFieldElementFlexible> {
                coefficients: generate_random_numbers(a_degree, q as i128)
                    .iter()
                    .map(|x| pfb(*x as i64, q))
                    .collect(),
            };
            let b = Polynomial::<PrimeFieldElementFlexible> {
                coefficients: generate_random_numbers(a_degree + i, q as i128)
                    .iter()
                    .map(|x| pfb(*x as i64, q))
                    .collect(),
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
        let q = 71;
        let a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(17, q)],
        };
        let b = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(17, q)],
        };
        let one = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(1, q)],
        };
        let zero = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![],
        };
        let zero_alt = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(0, q)],
        };
        let zero_alt_alt = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(0, q); 4],
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

        let x: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![pfb(0, q), pfb(1, q)],
        };
        let mut prod_x = Polynomial {
            coefficients: vec![pfb(0, q), pfb(1, q)],
        };
        let mut expected_quotient = Polynomial {
            coefficients: vec![pfb(1, q)],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());
        assert_eq!(zero, zero.clone() / b);

        prod_x = Polynomial {
            coefficients: vec![pfb(0, q), pfb(0, q), pfb(1, q)],
        };
        expected_quotient = Polynomial {
            coefficients: vec![pfb(1, q)],
        };
        assert_eq!(expected_quotient, prod_x / (x.clone() * x.clone()));

        prod_x = Polynomial {
            coefficients: vec![pfb(0, q), pfb(1, q), pfb(2, q)],
        };
        expected_quotient = Polynomial {
            coefficients: vec![pfb(1, q), pfb(2, q)],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());

        prod_x = Polynomial {
            coefficients: vec![pfb(1, q), pfb(0, q), pfb(2, q)],
        };
        expected_quotient = Polynomial {
            coefficients: vec![pfb(0, q), pfb(2, q)],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());

        prod_x = Polynomial {
            coefficients: vec![
                pfb(0, q),
                pfb(48, q),
                pfb(0, q),
                pfb(0, q),
                pfb(0, q),
                pfb(25, q),
                pfb(11, q),
                pfb(0, q),
                pfb(0, q),
                pfb(64, q),
                pfb(16, q),
                pfb(0, q),
                pfb(30, q),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![
                pfb(48, q),
                pfb(0, q),
                pfb(0, q),
                pfb(0, q),
                pfb(25, q),
                pfb(11, q),
                pfb(0, q),
                pfb(0, q),
                pfb(64, q),
                pfb(16, q),
                pfb(0, q),
                pfb(30, q),
            ],
        };
        assert_eq!(expected_quotient, prod_x.clone() / x.clone());

        expected_quotient = Polynomial {
            coefficients: vec![
                pfb(0, q),
                pfb(0, q),
                pfb(0, q),
                pfb(25, q),
                pfb(11, q),
                pfb(0, q),
                pfb(0, q),
                pfb(64, q),
                pfb(16, q),
                pfb(0, q),
                pfb(30, q),
            ],
        };
        assert_eq!(expected_quotient, prod_x.clone() / (x.clone() * x.clone()));
        assert_eq!(
            Polynomial {
                coefficients: vec![pfb(0, q), pfb(48, q),],
            },
            prod_x % (x.clone() * x)
        );
    }

    #[test]
    fn polynomial_arithmetic_test_linear_combination() {
        let q = 167772161;
        let tq = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![
                pfb(76432291, q),
                pfb(6568597, q),
                pfb(37593670, q),
                pfb(164656139, q),
                pfb(100728053, q),
                pfb(8855557, q),
                pfb(84827854, q),
            ],
        };
        let ti = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![
                pfb(137616711, q),
                pfb(15613095, q),
                pfb(114041830, q),
                pfb(68272686, q),
            ],
        };
        let bq = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(43152288, q), pfb(68272686, q)],
        };
        let x_to_3 = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![pfb(0, q), pfb(0, q), pfb(0, q), pfb(1, q)],
        };
        let ks = vec![
            pfb(132934501, q),
            pfb(57662258, q),
            pfb(76229169, q),
            pfb(82319948, q),
        ];
        let expected_lc = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![
                pfb(2792937, q),
                pfb(39162406, q),
                pfb(7217300, q),
                pfb(58955792, q),
                pfb(3275580, q),
                pfb(58708383, q),
                pfb(3119620, q),
            ],
        };
        let linear_combination = tq
            + ti.scalar_mul(ks[0])
            + (ti * x_to_3.clone()).scalar_mul(ks[1])
            + bq.scalar_mul(ks[2])
            + (bq * x_to_3).scalar_mul(ks[3]);
        assert_eq!(expected_lc, linear_combination);

        let x_values: Vec<PrimeFieldElementFlexible> = vec![
            pfb(1, q),
            pfb(116878283, q),
            pfb(71493608, q),
            pfb(131850885, q),
            pfb(65249968, q),
            pfb(26998229, q),
            pfb(30406922, q),
            pfb(40136459, q),
            pfb(167772160, q),
            pfb(50893878, q),
            pfb(96278553, q),
            pfb(35921276, q),
            pfb(102522193, q),
            pfb(140773932, q),
            pfb(137365239, q),
            pfb(127635702, q),
        ];
        let expected_y_values: Vec<PrimeFieldElementFlexible> = vec![
            pfb(5459857, q),
            pfb(148657471, q),
            pfb(30002611, q),
            pfb(66137138, q),
            pfb(8094868, q),
            pfb(56386222, q),
            pfb(156375138, q),
            pfb(54481212, q),
            pfb(27351017, q),
            pfb(142491681, q),
            pfb(27138843, q),
            pfb(146662298, q),
            pfb(151140487, q),
            pfb(131629901, q),
            pfb(120097158, q),
            pfb(114758378, q),
        ];
        for i in 0..16 {
            assert_eq!(
                expected_y_values[i],
                linear_combination.evaluate(&x_values[i])
            );
        }
    }

    #[test]
    fn fast_multiply_test() {
        let q = 65537;
        let primitive_root = pfb(1, q).get_primitive_root_of_unity(32).0.unwrap();
        println!("primitive_root = {}", primitive_root);
        let a: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![
                pfb(1, q),
                pfb(2, q),
                pfb(3, q),
                pfb(4, q),
                pfb(5, q),
                pfb(6, q),
                pfb(7, q),
                pfb(8, q),
                pfb(9, q),
                pfb(10, q),
            ],
        };
        let b: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![
                pfb(1, q),
                pfb(2, q),
                pfb(3, q),
                pfb(4, q),
                pfb(5, q),
                pfb(6, q),
                pfb(7, q),
                pfb(8, q),
                pfb(9, q),
                pfb(17, q),
            ],
        };
        let c_fast = Polynomial::fast_multiply(&a, &b, &primitive_root, 32);
        let c_normal = a.clone() * b.clone();
        println!("c_normal = {}", c_normal);
        println!("c_fast = {}", c_fast);
        assert_eq!(c_normal, c_fast);
        assert_eq!(
            Polynomial::ring_zero(),
            Polynomial::fast_multiply(&Polynomial::ring_zero(), &b, &primitive_root, 32)
        );
        assert_eq!(
            Polynomial::ring_zero(),
            Polynomial::fast_multiply(&a, &Polynomial::ring_zero(), &primitive_root, 32)
        );

        let one: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![pfb(1, q)],
        };
        assert_eq!(a, Polynomial::fast_multiply(&a, &one, &primitive_root, 32));
        assert_eq!(a, Polynomial::fast_multiply(&one, &a, &primitive_root, 32));
        assert_eq!(b, Polynomial::fast_multiply(&b, &one, &primitive_root, 32));
        assert_eq!(b, Polynomial::fast_multiply(&one, &b, &primitive_root, 32));
        let x: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![pfb(0, q), pfb(1, q)],
        };
        assert_eq!(
            a.shift_coefficients(1, pfb(0, q)),
            Polynomial::fast_multiply(&x, &a, &primitive_root, 32)
        );
        assert_eq!(
            a.shift_coefficients(1, pfb(0, q)),
            Polynomial::fast_multiply(&a, &x, &primitive_root, 32)
        );
        assert_eq!(
            b.shift_coefficients(1, pfb(0, q)),
            Polynomial::fast_multiply(&x, &b, &primitive_root, 32)
        );
        assert_eq!(
            b.shift_coefficients(1, pfb(0, q)),
            Polynomial::fast_multiply(&b, &x, &primitive_root, 32)
        );
    }

    #[test]
    fn fast_zerofier_test() {
        let q = 17;
        let _1_17 = pfb(1, q);
        let _5_17 = pfb(5, q);
        let _9_17 = pfb(9, q);
        let root_order = 8;
        let domain = vec![_1_17, _5_17];
        let actual =
            Polynomial::<PrimeFieldElementFlexible>::fast_zerofier(&domain, &_9_17, root_order);
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
            !actual.evaluate(&_9_17).is_zero(),
            "expecting {} != 0 when x = 9",
            actual
        );

        let _3_17 = pfb(3, q);
        let _7_17 = pfb(7, q);
        let _10_17 = pfb(10, q);
        let root_order_2 = 16;
        let domain_2 = vec![_7_17, _10_17];
        let actual_2 =
            Polynomial::<PrimeFieldElementFlexible>::fast_zerofier(&domain_2, &_3_17, root_order_2);
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
    fn fast_evaluate_test() {
        let q = 17;
        let _0_17 = pfb(0, q);
        let _1_17 = pfb(1, q);
        let _3_17 = pfb(3, q);
        let _5_17 = pfb(5, q);

        // x^5 + x^3
        let poly = poly_flex(vec![_0_17, _0_17, _0_17, _1_17, _0_17, _1_17]);

        let _6_17 = pfb(6, q);
        let _12_17 = pfb(12, q);
        let domain = vec![_6_17, _12_17];

        let actual = poly.fast_evaluate(&domain, &_3_17, 16);
        let expected_6 = _6_17.mod_pow(5.into()) + _6_17.mod_pow(3.into());
        assert_eq!(expected_6, actual[0]);

        let expected_12 = _12_17.mod_pow(5.into()) + _12_17.mod_pow(3.into());
        assert_eq!(expected_12, actual[1]);
    }

    #[test]
    fn fast_interpolate_test() {
        let q = 17;
        let _0_17 = pfb(0, q);
        let _1_17 = pfb(1, q);
        let _13_17 = pfb(13, q);
        let _5_17 = pfb(5, q);

        // x^3 + x^1
        let poly = poly_flex(vec![_0_17, _1_17, _0_17, _1_17]);

        let _6_17 = pfb(6, q);
        let _7_17 = pfb(7, q);
        let _8_17 = pfb(8, q);
        let _9_17 = pfb(9, q);
        let domain = vec![_6_17, _7_17, _8_17, _9_17];

        let evals = poly.fast_evaluate(&domain, &_13_17, 4);
        let reinterp = Polynomial::fast_interpolate(&domain, &evals, &_13_17, 4);
        assert_eq!(poly, reinterp);
    }

    #[test]
    fn fast_interpolate_pbt() {
        let mut rng = rand::thread_rng();
        for num_points in [1, 2, 4, 8, 16, 32, 64, 128] {
            let domain = BFieldElement::random_elements(num_points, &mut rng);
            let values = BFieldElement::random_elements(num_points, &mut rng);
            let omega = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(num_points as u64)
                .0
                .unwrap();

            let interpolant = Polynomial::fast_interpolate(&domain, &values, &omega, num_points);

            for (x, y) in domain.iter().zip(values) {
                assert_eq!(y, interpolant.evaluate(x));
            }
        }
    }

    #[test]
    fn interpolate_pb_test() {
        let mut rng = rand::thread_rng();
        for _trial_index in 0..1 {
            let num_points = 128; //(rng.next_u32() % 200) as usize;

            // sample random but distinct domain points
            let mut domain: Vec<BFieldElement> = Vec::<BFieldElement>::with_capacity(num_points);
            for _i in 0..num_points {
                let mut x = BFieldElement::new(rng.next_u64());
                while domain.contains(&x) {
                    x = BFieldElement::new(rng.next_u64());
                }
                domain.push(x);
            }

            // sample random values
            let values = (0..num_points)
                .map(|_| BFieldElement::new(rng.next_u64()))
                .collect_vec();

            // use lagrange interpolation
            let lagrange_interpolant =
                Polynomial::<BFieldElement>::fast_lagrange_interpolate(&domain, &values);

            // re-evaluate and match against values
            let lagrange_re_eval = domain
                .iter()
                .map(|d| lagrange_interpolant.evaluate(&d))
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
            let maybe_omega = BFieldElement::ring_zero().get_primitive_root_of_unity(order as u64);
            let omega = maybe_omega.0.unwrap();

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
        }
    }

    #[test]
    fn fast_coset_evaluate_test() {
        let q = 17;
        let _0_17 = pfb(0, q);
        let _1_17 = pfb(1, q);
        let _3_17 = pfb(3, q);
        let _9_17 = pfb(9, q);

        // x^5 + x^3
        let poly = poly_flex(vec![_0_17, _0_17, _0_17, _1_17, _0_17, _1_17]);

        let values = poly.fast_coset_evaluate(&_3_17, _9_17, 8);

        let mut domain = vec![_0_17; 8];
        domain[0] = _3_17;
        for i in 1..8 {
            domain[i] = domain[i - 1].to_owned() * _9_17.to_owned();
        }

        let reinterp = Polynomial::fast_interpolate(&domain, &values, &_9_17, 8);
        assert_eq!(reinterp, poly);

        let poly_interpolated = Polynomial::fast_coset_interpolate(&_3_17, _9_17, &values);
        assert_eq!(poly, poly_interpolated);
    }

    #[test]
    fn fast_coset_divide_test() {
        let q = 65537;
        let offset = pfb(1, q).get_primitive_root_of_unity(64).0.unwrap();
        let primitive_root = pfb(1, q).get_primitive_root_of_unity(32).0.unwrap();
        println!("primitive_root = {}", primitive_root);
        let a: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![
                pfb(1, q),
                pfb(2, q),
                pfb(3, q),
                pfb(4, q),
                pfb(5, q),
                pfb(6, q),
                pfb(7, q),
                pfb(8, q),
                pfb(9, q),
                pfb(10, q),
            ],
        };
        let b: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![
                pfb(1, q),
                pfb(2, q),
                pfb(3, q),
                pfb(4, q),
                pfb(5, q),
                pfb(6, q),
                pfb(7, q),
                pfb(8, q),
                pfb(9, q),
                pfb(17, q),
            ],
        };
        let c_fast = Polynomial::fast_multiply(&a, &b, &primitive_root, 32);

        let mut quotient = Polynomial::fast_coset_divide(&c_fast, &b, offset, primitive_root, 32);
        assert_eq!(a, quotient);

        quotient = Polynomial::fast_coset_divide(&c_fast, &a, offset, primitive_root, 32);
        assert_eq!(b, quotient);
    }

    #[test]
    fn polynomial_arithmetic_test() {
        let q = 71;
        let _6_71 = pfb(6, q);
        let _12_71 = pfb(12, q);
        let _16_71 = pfb(16, q);
        let _17_71 = pfb(17, q);
        let _22_71 = pfb(22, q);
        let _28_71 = pfb(28, q);
        let _33_71 = pfb(33, q);
        let _38_71 = pfb(38, q);
        let _49_71 = pfb(49, q);
        let _60_71 = pfb(60, q);
        let _64_71 = pfb(64, q);
        let _65_71 = pfb(65, q);
        let _66_71 = pfb(66, q);
        let mut a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_17_71],
        };
        let mut b = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_16_71],
        };
        let mut sum = a + b;
        let mut expected_sum = Polynomial {
            coefficients: vec![_33_71],
        };
        assert_eq!(expected_sum, sum);

        // Verify overflow handling
        a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_66_71],
        };
        b = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_65_71],
        };
        sum = a + b;
        expected_sum = Polynomial {
            coefficients: vec![_60_71],
        };
        assert_eq!(expected_sum, sum);

        // Verify handling of multiple indices
        a = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_66_71, _66_71, _66_71],
        };
        b = Polynomial::<PrimeFieldElementFlexible> {
            coefficients: vec![_33_71, _33_71, _17_71, _65_71],
        };
        sum = a.clone() + b.clone();
        expected_sum = Polynomial {
            coefficients: vec![_28_71, _28_71, _12_71, _65_71],
        };
        assert_eq!(expected_sum, sum);

        let mut diff = a.clone() - b.clone();
        let mut expected_diff = Polynomial {
            coefficients: vec![_33_71, _33_71, _49_71, _6_71],
        };
        assert_eq!(expected_diff, diff);

        diff = b.clone() - a.clone();
        expected_diff = Polynomial {
            coefficients: vec![_38_71, _38_71, _22_71, _65_71],
        };
        assert_eq!(expected_diff, diff);

        // Test multiplication
        let mut prod = a.clone() * b.clone();
        let mut expected_prod = Polynomial {
            coefficients: vec![
                pfb(48, q),
                pfb(25, q),
                pfb(11, q),
                pfb(64, q),
                pfb(16, q),
                pfb(30, q),
            ],
        };
        assert_eq!(expected_prod, prod);
        assert_eq!(5, prod.degree());
        assert_eq!(2, a.degree());
        assert_eq!(3, b.degree());

        let zero: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![],
        };
        let zero_alt: Polynomial<PrimeFieldElementFlexible> = Polynomial::ring_zero();
        assert_eq!(zero, zero_alt);
        let one: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![pfb(1, q)],
        };
        let x: Polynomial<PrimeFieldElementFlexible> = Polynomial {
            coefficients: vec![pfb(0, q), pfb(1, q)],
        };
        assert_eq!(-1, zero.degree());
        assert_eq!(0, one.degree());
        assert_eq!(1, x.degree());
        assert_eq!(zero, prod.clone() * zero.clone());
        assert_eq!(prod, prod.clone() * one.clone());
        assert_eq!(x, x.clone() * one.clone());

        assert_eq!("0", zero.to_string());
        assert_eq!("1", one.to_string());
        assert_eq!("x", x.to_string());
        assert_eq!("66x^2 + 66x + 66", a.to_string());

        expected_prod = Polynomial {
            coefficients: vec![
                pfb(0, q),
                pfb(48, q),
                pfb(25, q),
                pfb(11, q),
                pfb(64, q),
                pfb(16, q),
                pfb(30, q),
            ],
        };
        prod = prod.clone() * x;
        assert_eq!(expected_prod, prod);
        assert_eq!(
            "30x^6 + 16x^5 + 64x^4 + 11x^3 + 25x^2 + 48x",
            prod.to_string()
        );
    }

    #[test]
    pub fn polynomial_divide_test() {
        let minus_one = BFieldElement::QUOTIENT - 1;
        let zero = BFieldElement::ring_zero();
        let one = BFieldElement::ring_one();
        let two = BFieldElement::new(2);

        let a: Polynomial<BFieldElement> = Polynomial::new_const(BFieldElement::new(30));
        let b: Polynomial<BFieldElement> = Polynomial::new_const(BFieldElement::new(5));

        let (actual_quot, actual_rem) = a.divide(b);
        let expected_quot: Polynomial<BFieldElement> = Polynomial::new_const(BFieldElement::new(6));

        assert_eq!(expected_quot, actual_quot);
        assert!(actual_rem.is_zero());

        // Shah-polynomial test
        let shah = XFieldElement::shah_polynomial();
        let c = Polynomial::new(vec![
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_one(),
        ]);
        let (actual_quot, actual_rem) = shah.divide(c);
        println!("actual_quot = {}", actual_quot);
        println!("actual_rem = {}", actual_rem);

        let expected_quot = Polynomial::new_const(BFieldElement::new(1));
        let expected_rem = Polynomial::new(vec![
            BFieldElement::ring_one(),
            BFieldElement::new(minus_one),
        ]);
        assert_eq!(expected_quot, actual_quot);
        assert_eq!(expected_rem, actual_rem);

        // x^6
        let c: Polynomial<BFieldElement> = Polynomial::new(vec![one]).shift_coefficients(6, zero);

        let (actual_sixth_quot, actual_sixth_rem) = c.divide(shah);

        // x^3 + x - 1
        let expected_sixth_quot: Polynomial<BFieldElement> =
            Polynomial::new(vec![-one, one, zero, one]);
        // x^2 - 2x + 1
        let expected_sixth_rem: Polynomial<BFieldElement> = Polynomial::new(vec![one, -two, one]);

        assert_eq!(expected_sixth_quot, actual_sixth_quot);
        assert_eq!(expected_sixth_rem, actual_sixth_rem);
    }

    #[test]
    fn add_assign_test() {
        for _ in 0..10 {
            let poly1 = gen_polynomial();
            let poly2 = gen_polynomial();
            let expected = poly1.clone() + poly2.clone();
            let mut actual = poly1.clone();
            actual += poly2.clone();

            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn is_x_test() {
        let zero: Polynomial<BFieldElement> = Polynomial::ring_zero();
        assert!(!zero.is_x());

        let one: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![BFieldElement::ring_one()],
        };
        assert!(!one.is_x());
        let x: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![BFieldElement::ring_zero(), BFieldElement::ring_one()],
        };
        assert!(x.is_x());
        let x_alt: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::ring_zero(),
                BFieldElement::ring_one(),
                BFieldElement::ring_zero(),
            ],
        };
        assert!(x_alt.is_x());
        let x_alt_alt: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::ring_zero(),
                BFieldElement::ring_one(),
                BFieldElement::ring_zero(),
                BFieldElement::ring_zero(),
            ],
        };
        assert!(x_alt_alt.is_x());
        let _2x: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::ring_zero(),
                BFieldElement::ring_one() + BFieldElement::ring_one(),
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
        poly.coefficients[15] = BFieldElement::ring_one();
        expected.coefficients = vec![0; 32].into_iter().map(BFieldElement::new).collect();
        expected.coefficients[30] = BFieldElement::ring_one();
        assert_eq!(expected, poly.fast_square());
    }

    #[test]
    fn square_test() {
        let one_pol = Polynomial {
            coefficients: vec![BFieldElement::ring_one()],
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
            let a = gen_polynomial();
            let b = gen_polynomial();
            let ab = a.clone() * b.clone();
            let ba = b.clone() * a.clone();
            assert_eq!(ab, ba);
        }
    }

    #[test]
    fn lift_b_x_test() {
        for _ in 0..5 {
            let pol = gen_polynomial();
            let lifted_pol: Polynomial<XFieldElement> = Polynomial::<BFieldElement>::lift_b_x(&pol);
            for (coefficient, lifted_coefficient) in
                pol.coefficients.iter().zip(lifted_pol.coefficients.iter())
            {
                assert_eq!(Some(*coefficient), lifted_coefficient.unlift());
            }
        }
    }

    #[test]
    fn constant_zero_eq_constant_zero() {
        let zero_polynomial1 = Polynomial::<BFieldElement>::ring_zero();
        let zero_polynomial2 = Polynomial::<BFieldElement>::ring_zero();

        assert!(zero_polynomial1 == zero_polynomial2)
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn get_point_on_invalid_line_test() {
        let one = BFieldElement::ring_one();
        let two = one + one;
        let three = two + one;
        Polynomial::<BFieldElement>::get_colinear_y((one, one), (one, three), two);
    }

    #[test]
    fn get_point_on_line_test() {
        type BPoly = Polynomial<BFieldElement>;
        let one = BFieldElement::ring_one();
        let two = one + one;
        let three = two + one;
        assert_eq!(two, BPoly::get_colinear_y((one, one), (three, three), two));
        assert_eq!(two, BPoly::get_colinear_y((three, three), (one, one), two));
        assert_eq!(one, BPoly::get_colinear_y((one, one), (three, one), two));
        type XPoly = Polynomial<XFieldElement>;
        let one = XFieldElement::ring_one();
        let two = one + one;
        let three = two + one;
        assert_eq!(two, XPoly::get_colinear_y((one, one), (three, three), two));
        assert_eq!(two, XPoly::get_colinear_y((three, three), (one, one), two));
        assert_eq!(one, XPoly::get_colinear_y((one, one), (three, one), two));
    }

    fn gen_polynomial() -> Polynomial<BFieldElement> {
        let mut rng = rand::thread_rng();
        let coefficient_count = rng.next_u64() as usize % 40;

        Polynomial {
            coefficients: BFieldElement::random_elements(coefficient_count, &mut rng),
        }
    }

    #[test]
    fn lagrange_interpolate_test() {
        type BPoly = Polynomial<BFieldElement>;
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let num_points = max(2, rng.next_u32() as usize % 10);
            let domain = {
                let mut domain = vec![];
                while domain.len() < num_points {
                    let new_domain_candidate = rng.next_u64();
                    if !domain.contains(&new_domain_candidate) {
                        domain.push(new_domain_candidate)
                    }
                }
                domain
                    .into_iter()
                    .map(|x| BFieldElement::new(x))
                    .collect_vec()
            };
            let values = BFieldElement::random_elements(num_points, &mut rng);
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
            let num_points = max(2, rng.next_u32() as usize % 10);
            let domain = {
                let mut domain = vec![];
                while domain.len() < num_points {
                    let new_domain_candidate = rng.next_u64();
                    if !domain.contains(&new_domain_candidate) {
                        domain.push(new_domain_candidate)
                    }
                }
                domain
                    .into_iter()
                    .map(|x| BFieldElement::new(x))
                    .collect_vec()
            };
            let values = BFieldElement::random_elements(num_points, &mut rng);
            let interpoly = BPoly::fast_lagrange_interpolate(&domain, &values);

            assert!(num_points as isize > interpoly.degree());
            for (i, y) in values.into_iter().enumerate() {
                assert_eq!(y, interpoly.evaluate(&domain[i]));
            }
        }
    }

    #[test]
    fn lagrange_bench_test() {
        type BPoly = Polynomial<BFieldElement>;
        let mut rng = rand::thread_rng();
        for num_points in [8, 16, 32, 64, 128] {
            let domain = {
                let mut domain = vec![];
                while domain.len() < num_points {
                    let new_domain_candidate = rng.next_u64();
                    if !domain.contains(&new_domain_candidate) {
                        domain.push(new_domain_candidate)
                    }
                }
                domain
                    .into_iter()
                    .map(|x| BFieldElement::new(x))
                    .collect_vec()
            };
            let values = BFieldElement::random_elements(num_points, &mut rng);

            let mut timer = TimingReporter::start();

            // readable
            let readable_poly = BPoly::lagrange_interpolate(&domain, &values);
            timer.elapsed(&format!(
                "Readable lagrange interpolation ({} points).",
                num_points
            ));

            // fast
            let fast_poly = BPoly::fast_lagrange_interpolate(&domain, &values);
            timer.elapsed(&format!(
                "Fast lagrange interpolation ({} points).",
                num_points
            ));

            // slowinternal
            let slow_internal_poly = BPoly::slow_lagrange_interpolation_internal(&domain, &values);
            timer.elapsed(&format!(
                "SlowInternal lagrange interpolation ({} points).",
                num_points
            ));

            // ntt-based
            let ntt_based_poly = Polynomial::fast_interpolate(
                &domain,
                &values,
                &domain[0]
                    .get_primitive_root_of_unity(domain.len() as u64)
                    .0
                    .unwrap(),
                domain.len(),
            );
            timer.elapsed(&format!("NTT-Based interpolation ({} points).", num_points));

            let report = timer.finish();
            println!("{}", report);

            for (d, v) in domain.iter().zip(values.iter()) {
                assert_eq!(readable_poly.evaluate(d), *v);
                assert_eq!(fast_poly.evaluate(d), *v);
                assert_eq!(slow_internal_poly.evaluate(d), *v);
                assert_eq!(ntt_based_poly.evaluate(d), *v);
            }
        }
    }
}
