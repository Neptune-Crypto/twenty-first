use super::fraction::Fraction;
use super::polynomial_quotient_ring::PolynomialQuotientRing;
use super::prime_field_element::{PrimeField, PrimeFieldElement};
use crate::utils::{generate_random_numbers, has_unique_elements};
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use num_traits::Zero;
use rand::Rng;
use rand_distr::Normal;
use std::convert::From;
use std::fmt;

fn pretty_print_coefficients(coefficients: &[i128]) -> String {
    if coefficients.is_empty() {
        return String::from("0");
    }

    let trailing_zeros_warning: &str = match coefficients.last() {
        Some(0) => &"Trailing zeros!",
        _ => &"",
    };

    let mut outputs: Vec<String> = Vec::new();
    let pol_degree = coefficients.len() - 1;
    for i in 0..=pol_degree {
        let pow = pol_degree - i;

        if coefficients[pow] != 0 {
            // operator: plus or minus
            let mut operator_str = "";
            if i != 0 && coefficients[pow] != 0 {
                operator_str = if coefficients[pow] > 0 { " + " } else { " - " };
            }

            let abs_val = if coefficients[pow] < 0 {
                -coefficients[pow]
            } else {
                coefficients[pow]
            };
            outputs.push(format!(
                "{}{}{}",
                operator_str,
                if abs_val == 1 {
                    String::from("")
                } else {
                    abs_val.to_string()
                },
                if pow == 0 && abs_val == 1 {
                    String::from("1")
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
    }
    format!("{}{}", trailing_zeros_warning, outputs.join(""))
}

// All structs holding references must have lifetime annotations in their definition.
// This <'a> annotation means that an instance of Polynomial cannot outlive the reference it holds
// in its `pqr` field.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimeFieldPolynomial<'a> {
    pub coefficients: Vec<i128>,
    pub pqr: &'a PolynomialQuotientRing,
}

impl std::fmt::Display for PrimeFieldPolynomial<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let res: String = pretty_print_coefficients(&self.coefficients);
        return write!(f, "{}", res);
    }
}

impl<'a> PrimeFieldPolynomial<'a> {
    // Verify that N > 2 points are colinear
    // Also demand that all x-values are unique??
    pub fn are_colinear_raw(points: &[(i128, i128)], modulus: i128) -> bool {
        if points.len() < 3 {
            println!("Too few points received. Got: {}", points.len());
            return false;
        }

        // if !has_unique_elements(points.iter().map(|p| p.0)) {
        //     println!("Non-unique element spotted Got: {:?}", points);
        //     return false;
        // }

        // Find 1st degree polynomial from first two points
        let field = PrimeField::new(modulus);
        let x_diff = PrimeFieldElement::new(points[0].0 - points[1].0, &field);
        let x_diff_inv = x_diff.inv();
        // println!("x_diff = {} => x_diff_inv = {}", x_diff.value, x_diff_inv);
        let a = ((points[0].1 - points[1].1) * x_diff_inv.value % modulus + modulus) % modulus;
        let b = ((points[0].1 - a * points[0].0) % modulus + modulus) % modulus;
        for point in points.iter().skip(2) {
            // A decent speedup could be achieved by removing the two last modulus
            // expressions here and demand that the input x-values are all elements
            // in the finite field
            let expected = ((a * point.0 + b) % modulus + modulus) % modulus;
            if (point.1 % modulus + modulus) % modulus != expected {
                println!(
                    "L({}) = {}, expected L({}) = {}, Found: L(x) = {}x + {} mod {} from {{({},{}),({},{})}}",
                    point.0,
                    (point.1 % modulus + modulus) % modulus,
                    point.0,
                    expected,
                    a,
                    b,
                    modulus,
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

    pub fn additive_identity(pqr: &'a PolynomialQuotientRing) -> Self {
        Self {
            coefficients: Vec::new(),
            pqr,
        }
    }

    pub fn polynomium_from_int(val: i128, pqr: &'a PolynomialQuotientRing) -> Self {
        Self {
            coefficients: vec![val],
            pqr,
        }
    }

    // This function assumes that the polynomial is already normalized
    pub fn degree(&self) -> usize {
        match self.coefficients[..] {
            [] => 0,
            [_, ..] => self.coefficients.len() - 1,
        }
    }

    pub fn evaluate<'d>(&self, x: &'d PrimeFieldElement) -> PrimeFieldElement<'d> {
        let zero = PrimeFieldElement::new(0, x.field);
        self.coefficients
            .iter()
            .enumerate()
            .map(|(i, &c)| PrimeFieldElement::new(c, x.field) * x.mod_pow(i as i128))
            .fold(zero, |sum, val| sum + val)
    }

    pub fn finite_field_lagrange_interpolation<'b>(
        points: &'b [(PrimeFieldElement, PrimeFieldElement)],
        pqr: &'a PolynomialQuotientRing,
    ) -> Self {
        // calculate a reversed representation of the coefficients of
        // prod_{i=0}^{N}((x- q_i))
        fn prod_helper<'c>(input: &[PrimeFieldElement<'c>]) -> Vec<PrimeFieldElement<'c>> {
            if let Some((q_j, elements)) = input.split_first() {
                match elements {
                    // base case is `x - q_j` := [1, -q_j]
                    [] => vec![
                        PrimeFieldElement::new(1, q_j.field),
                        PrimeFieldElement::new(-q_j.value, q_j.field),
                    ],
                    _ => {
                        // The recursive call calculates (x-q_j)*rec = x*rec - q_j*rec := [0, rec] .- q_j*[rec]
                        let mut rec = prod_helper(elements);
                        rec.push(PrimeFieldElement::new(0, q_j.field));
                        let mut i = rec.len() - 1;
                        while i > 0 {
                            rec[i] =
                                rec[i] - PrimeFieldElement::new(q_j.value, q_j.field) * rec[i - 1];
                            i -= 1;
                        }
                        rec
                    }
                }
            } else {
                panic!("Empty array received");
            }
        }

        if !has_unique_elements(points.iter().map(|&x| x.0.value)) {
            panic!("Repeated x values received. Got: {:?}", points);
        }

        let roots: Vec<PrimeFieldElement> = points.iter().map(|x| x.0).collect();
        let mut big_pol_coeffs = prod_helper(&roots);

        big_pol_coeffs.reverse();
        let big_pol = Self {
            coefficients: big_pol_coeffs.iter().map(|&x| x.value).collect(),
            pqr,
        };
        let mut coefficients: Vec<PrimeFieldElement> =
            vec![PrimeFieldElement::new(0, points[0].0.field); points.len()];
        for point in points.iter() {
            // create a PrimeFieldPolynomial that is zero at all other points than this
            // coeffs_j = prod_{i=0, i != j}^{N}((x- q_i))
            let my_div_coefficients = vec![
                PrimeFieldElement::new(-point.0.value, point.0.field),
                PrimeFieldElement::new(1, point.0.field),
            ];
            let mut my_pol = Self {
                coefficients: my_div_coefficients.iter().map(|&x| x.value).collect(),
                pqr,
            };
            my_pol = big_pol.div(&my_pol).0; // my_pol = big_pol / my_pol

            let mut divisor: PrimeFieldElement = PrimeFieldElement::new(1, point.0.field);
            for root in roots.iter() {
                if root.value == point.0.value {
                    continue;
                }
                divisor = divisor * (point.0 - *root);
            }

            let mut my_coeffs: Vec<PrimeFieldElement> = my_pol
                .coefficients
                .iter()
                .map(|&x| PrimeFieldElement::new(x, point.0.field))
                .collect();
            for coeff in my_coeffs.iter_mut() {
                *coeff = *coeff * point.1;
                *coeff = *coeff / divisor;
            }

            for i in 0..my_coeffs.len() {
                coefficients[i] = coefficients[i] + my_coeffs[i];
            }
        }

        Self {
            coefficients: coefficients.iter().map(|&x| x.value).collect(),
            pqr,
        }
    }

    pub fn gen_binary_poly(pqr: &'a PolynomialQuotientRing) -> Self {
        let mut res = Self {
            coefficients: generate_random_numbers(pqr.n as usize, 2i128),
            pqr,
        };
        // TODO: Do we take the modulus here?
        res.normalize();
        res
    }

    pub fn gen_uniform_poly(pqr: &'a PolynomialQuotientRing) -> Self {
        let mut res = Self {
            coefficients: generate_random_numbers(pqr.n as usize, pqr.q),
            pqr,
        };
        res = res.modulus();
        res.normalize();
        res
    }

    pub fn gen_normal_poly(pqr: &'a PolynomialQuotientRing) -> Self {
        // Alan recommends N(0, 4) for this
        let mut coefficients: Vec<i128> = vec![0i128; pqr.n as usize];
        let mut prng = rand::thread_rng();
        let normal_dist = Normal::new(0.0, 4.0).unwrap();
        for elem in coefficients.iter_mut() {
            let val: f64 = prng.sample(normal_dist);
            let int_val = ((val.round() as i128 % pqr.q) + pqr.q) % pqr.q;
            *elem = int_val;
        }
        let mut ret = Self { coefficients, pqr };
        ret = ret.modulus();
        ret.normalize();
        ret
    }

    // Remove trailing zeros from coefficients
    // Borrow mutably when you mutate the data, but don't want to consume it
    pub fn normalize(&mut self) {
        while let Some(0) = self.coefficients.last() {
            self.coefficients.pop();
        }
    }

    pub fn get_constant_term(&self) -> i128 {
        match self.coefficients[..] {
            [] => 0,
            [i, ..] => i,
        }
    }

    fn div(&self, divisor: &PrimeFieldPolynomial<'a>) -> (Self, Self) {
        if divisor.coefficients.is_empty() {
            panic!(
                "Cannot divide polynomial by zero. Got: ({})/({})",
                self, divisor
            );
        }

        if self.coefficients.is_empty() {
            return (
                PrimeFieldPolynomial::additive_identity(self.pqr),
                PrimeFieldPolynomial::additive_identity(self.pqr),
            );
        }

        let mut remainder = self.coefficients.clone();
        let mut quotient = Vec::with_capacity(self.pqr.n as usize); // built from back to front so must be reversed

        let dividend_coeffs: &Vec<i128> = &self.coefficients;
        let divisor_coeffs: &Vec<i128> = &divisor.coefficients;
        let divisor_degree = divisor_coeffs.len() - 1;
        let dividend_degree = dividend_coeffs.len() - 1;
        let dominant_divisor: i128 = *divisor_coeffs.last().unwrap();
        let mut inv: i128 = 1;
        if dominant_divisor != 1 {
            let (_, inv0, _) = PrimeFieldElement::eea(dominant_divisor, self.pqr.q);
            inv = inv0;
        }
        let mut i = 0;
        while i + divisor_degree <= dividend_degree {
            // calculate next quotient coefficient
            let mut res: i128 = *remainder.last().unwrap();
            if inv != 1 {
                res = (inv * res % self.pqr.q + self.pqr.q) % self.pqr.q;
            }
            quotient.push(res);
            remainder.pop(); // remove highest order coefficient

            // Calculate rem = rem - res * divisor
            // We need to manipulate divisor_degree + 1 values in remainder, but we get one for free through
            // the pop() above, so we only need to manipulate divisor_degree values. For a divisor of degree
            // 1, we need to manipulate 1 element.
            for j in 0..divisor_degree {
                // TODO: Rewrite this in terms of i, j, and poly degrees
                let rem_length = remainder.len();
                let divisor_length = divisor_coeffs.len();
                remainder[rem_length - j - 1] -= res * divisor_coeffs[divisor_length - j - 2];
                remainder[rem_length - j - 1] =
                    (remainder[rem_length - j - 1] % self.pqr.q + self.pqr.q) % self.pqr.q;
            }
            i += 1;
        }

        quotient.reverse();
        let mut quotient_pol = Self {
            coefficients: quotient,
            pqr: self.pqr,
        };
        let mut remainder_pol = Self {
            coefficients: remainder,
            pqr: self.pqr,
        };
        quotient_pol.normalize();
        remainder_pol.normalize();

        (quotient_pol, remainder_pol)
    }

    pub fn modulus(&self) -> Self {
        let polynomial_modulus = Self {
            coefficients: self.pqr.get_polynomial_modulus(),
            pqr: self.pqr,
        };
        self.div(&polynomial_modulus).1
    }

    pub fn mul(&self, other: &PrimeFieldPolynomial<'a>) -> Self {
        // If either polynomial is zero, return zero
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return PrimeFieldPolynomial::additive_identity(self.pqr);
        }

        let self_degree = self.coefficients.len() - 1;
        let other_degree = other.coefficients.len() - 1;
        let mut result_coeff: Vec<i128> = vec![0i128; self_degree + other_degree + 1];
        for i in 0..=self_degree {
            for j in 0..=other_degree {
                let mul = self.coefficients[i] * other.coefficients[j];
                result_coeff[i + j] += mul;
                result_coeff[i + j] =
                    ((result_coeff[i + j] % self.pqr.q) + self.pqr.q) % self.pqr.q;
            }
        }
        let mut ret = Self {
            coefficients: result_coeff,
            pqr: self.pqr,
        };
        ret.normalize();
        ret
    }

    pub fn balance(&self) -> Self {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            if coefficients[i] > self.pqr.q / 2 {
                coefficients[i] -= self.pqr.q;
            }
        }
        Self {
            coefficients,
            pqr: self.pqr,
        }
    }

    pub fn scalar_mul(&self, scalar: i128) -> Self {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] = (((coefficients[i] * scalar) % self.pqr.q) + self.pqr.q) % self.pqr.q;
        }
        Self {
            coefficients,
            pqr: self.pqr,
        }
    }

    pub fn scalar_modulus(&self, modulus: i128) -> Self {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] %= modulus;
        }
        Self {
            coefficients,
            pqr: self.pqr,
        }
    }

    pub fn scalar_mul_float(&self, float_scalar: f64) -> Self {
        // Function does not map coefficients into finite field. Should it?
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] = (coefficients[i] as f64 * float_scalar).round() as i128;
        }
        Self {
            coefficients,
            pqr: self.pqr,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let summed: Vec<i128> = self
            .coefficients
            .iter()
            .zip_longest(other.coefficients.iter())
            .map(|a: itertools::EitherOrBoth<&i128, &i128>| match a {
                Both(l, r) => (((*l + *r) % self.pqr.q) + self.pqr.q) % self.pqr.q,
                Left(l) => *l,
                Right(r) => *r,
            })
            .collect();

        let mut res = Self {
            coefficients: summed,
            pqr: self.pqr,
        };
        res.normalize();
        res
    }

    pub fn sub(&self, other: &Self) -> Self {
        let diff: Vec<i128> = self
            .coefficients
            .iter()
            .zip_longest(other.coefficients.iter())
            .map(|a: itertools::EitherOrBoth<&i128, &i128>| match a {
                Both(l, r) => (((*l - *r) % self.pqr.q) + self.pqr.q) % self.pqr.q,
                Left(l) => *l,
                Right(r) => -*r + self.pqr.q,
            })
            .collect();

        let mut res = Self {
            coefficients: diff,
            pqr: self.pqr,
        };
        res.normalize();
        res
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntegerRingPolynomial {
    pub coefficients: Vec<i128>,
}

impl fmt::Display for IntegerRingPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let res: String = pretty_print_coefficients(&self.coefficients);
        return write!(f, "{}", res);
    }
}

impl IntegerRingPolynomial {
    pub fn additive_identity() -> Self {
        Self {
            coefficients: Vec::new(),
        }
    }

    pub fn polynomium_from_int(val: i128) -> Self {
        Self {
            coefficients: vec![val],
        }
    }

    pub fn evaluate(&self, x: i128) -> i128 {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(i, &c)| c * x.pow(i as u32))
            .sum()
    }

    pub fn integer_lagrange_interpolation(points: &[(i128, i128)]) -> Self {
        // calculate a reversed representation of the coefficients of
        // prod_{i=0}^{N}((x- q_i))
        fn prod_helper(input: &[i128]) -> Vec<i128> {
            if let Some((q_j, elements)) = input.split_first() {
                match elements {
                    // base case is `x - q_j` := [1, -q_j]
                    [] => vec![1, -q_j],
                    _ => {
                        // The recursive call calculates (x-q_j)*rec = x*rec - q_j*rec := [0, rec] .- q_j*[rec]
                        let mut rec = prod_helper(elements);
                        rec.push(0);
                        let mut i = rec.len() - 1;
                        while i > 0 {
                            rec[i] -= q_j * rec[i - 1];
                            i -= 1;
                        }
                        rec
                    }
                }
            } else {
                panic!("Empty array received");
            }
        }

        if !has_unique_elements(points.iter().map(|&x| x.0)) {
            panic!("Repeated x values received. Got: {:?}", points);
        }

        let zero_frac = Fraction::zero();
        let mut fracs: Vec<Fraction<i128>> = vec![zero_frac; points.len()];
        let roots: Vec<i128> = points.iter().map(|x| x.0).collect();
        let mut big_pol_coeffs = prod_helper(&roots);
        big_pol_coeffs.reverse();
        let big_pol = Self {
            coefficients: big_pol_coeffs,
        };
        for point in points.iter() {
            // create a polynomial that is zero at all other points than this
            // coeffs_j = prod_{i=0, i != j}^{N}((x- q_i))
            let mut my_pol = Self {
                coefficients: vec![-point.0, 1],
            };
            my_pol = big_pol.div(&my_pol).0;

            let mut divisor: i128 = 1;
            for root in roots.iter() {
                if *root == point.0 {
                    continue;
                }
                // Beware that this value does not overflow
                // In release/benchmark mode an overflow will not get caught
                divisor *= point.0 - root;
            }

            let mut frac_coeffs: Vec<Fraction<i128>> = my_pol
                .coefficients
                .iter()
                .map(|&x| Fraction::<i128>::from(x))
                .collect();
            for coeff in frac_coeffs.iter_mut() {
                *coeff = coeff.scalar_mul(point.1);
                *coeff = coeff.scalar_div(divisor);
            }

            for i in 0..frac_coeffs.len() {
                fracs[i] = fracs[i] + frac_coeffs[i];
            }
        }

        // This assumes that all divisors are 1 and that all coefficients are integers
        let coefficients = fracs.iter().map(|&x| x.get_dividend()).collect();

        Self { coefficients }
    }

    // Remove trailing zeros from coefficients
    // Borrow mutably when you mutate the data, but don't want to consume it
    pub fn normalize(&mut self) {
        while let Some(0) = self.coefficients.last() {
            self.coefficients.pop();
        }
    }

    pub fn get_constant_term(&self) -> i128 {
        match self.coefficients[..] {
            [] => 0,
            [i, ..] => i,
        }
    }

    fn div(&self, divisor: &Self) -> (Self, Self) {
        if divisor.coefficients.is_empty() {
            panic!(
                "Cannot divide polynomial by zero. Got: ({})/({})",
                self, divisor
            );
        }

        if self.coefficients.is_empty() {
            return (Self::additive_identity(), Self::additive_identity());
        }

        let mut remainder = self.coefficients.clone();
        let mut quotient = Vec::with_capacity(divisor.coefficients.len() as usize); // built from back to front so must be reversed

        let dividend_coeffs: &Vec<i128> = &self.coefficients;
        let divisor_coeffs: &Vec<i128> = &divisor.coefficients;
        let divisor_degree = divisor_coeffs.len() - 1;
        let dividend_degree = dividend_coeffs.len() - 1;
        let dominant_divisor = divisor_coeffs.last().unwrap();
        let mut i = 0;
        while i + divisor_degree <= dividend_degree {
            // calculate next quotient coefficient
            // TODO: This is wrong for finite fields! But works if dominant_divisor is 1 (which it always is in Lagrange interpolation)
            let res = remainder.last().unwrap() / dominant_divisor;
            quotient.push(res);
            remainder.pop(); // remove highest order coefficient

            // Calculate rem = rem - res * divisor
            // We need to manipulate divisor_degree + 1 values in remainder, but we get one for free through
            // the pop() above, so we only need to manipulate divisor_degree values. For a divisor of degree
            // 1, we need to manipulate 1 element.
            for j in 0..divisor_degree {
                // TODO: Rewrite this in terms of i, j, and poly degrees
                let rem_length = remainder.len();
                let divisor_length = divisor_coeffs.len();
                remainder[rem_length - j - 1] -= res * divisor_coeffs[divisor_length - j - 2];
            }
            i += 1;
        }

        quotient.reverse();
        let quotient_pol = Self {
            coefficients: quotient,
        };
        let remainder_pol = Self {
            coefficients: remainder,
        };
        (quotient_pol, remainder_pol)
    }

    pub fn mul(&self, other: &Self) -> Self {
        // If either polynomial is zero, return zero
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Self::additive_identity();
        }

        let self_degree = self.coefficients.len() - 1;
        let other_degree = other.coefficients.len() - 1;
        let mut result_coeff: Vec<i128> = vec![0i128; self_degree + other_degree + 1];
        for i in 0..=self_degree {
            for j in 0..=other_degree {
                let mul = self.coefficients[i] * other.coefficients[j];
                result_coeff[i + j] += mul;
            }
        }
        let mut ret = Self {
            coefficients: result_coeff,
        };
        ret.normalize();
        ret
    }

    pub fn scalar_mul(&self, scalar: i128) -> Self {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] *= scalar;
        }
        Self { coefficients }
    }

    pub fn scalar_modulus(&self, modulus: i128) -> Self {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] %= modulus;
        }
        Self { coefficients }
    }

    pub fn scalar_mul_float(&self, float_scalar: f64) -> Self {
        // Function does not map coefficients into finite field. Should it?
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] = (coefficients[i] as f64 * float_scalar).round() as i128;
        }
        Self { coefficients }
    }

    pub fn add(&self, other: &Self) -> Self {
        let summed: Vec<i128> = self
            .coefficients
            .iter()
            .zip_longest(other.coefficients.iter())
            .map(|a: itertools::EitherOrBoth<&i128, &i128>| match a {
                Both(l, r) => *l + *r,
                Left(l) => *l,
                Right(r) => *r,
            })
            .collect();

        Self {
            coefficients: summed,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let diff: Vec<i128> = self
            .coefficients
            .iter()
            .zip_longest(other.coefficients.iter())
            .map(|a: itertools::EitherOrBoth<&i128, &i128>| match a {
                Both(l, r) => *l - *r,
                Left(l) => *l,
                Right(r) => -*r,
            })
            .collect();

        Self { coefficients: diff }
    }

    // This function assumes that the polynomial is already normalized
    pub fn degree(&self) -> usize {
        match self.coefficients[..] {
            [] => 0,
            [_, ..] => self.coefficients.len() - 1,
        }
    }
}

#[cfg(test)]
mod test_polynomials {
    #![allow(clippy::just_underscores_and_digits)]
    use super::super::prime_field_element::{PrimeField, PrimeFieldElement};
    use super::*;

    #[test]
    fn are_colinear_raw_test() {
        assert!(PrimeFieldPolynomial::are_colinear_raw(
            &[(1, 1), (2, 2), (3, 3)],
            5
        ));
        assert!(PrimeFieldPolynomial::are_colinear_raw(
            &[(1, 1), (2, 7), (3, 3)],
            5
        ));
        assert!(PrimeFieldPolynomial::are_colinear_raw(
            &[(1, 1), (7, 7), (3, 3)],
            5
        ));
        assert!(PrimeFieldPolynomial::are_colinear_raw(
            &[(1, 1), (7, 2), (3, 3)],
            5
        ));
        assert!(PrimeFieldPolynomial::are_colinear_raw(
            &[(-4, 1), (7, 2), (3, -2)],
            5
        ));
        assert!(!PrimeFieldPolynomial::are_colinear_raw(
            &[(1, 1), (2, 2), (3, 4)],
            5
        ));
        assert!(!PrimeFieldPolynomial::are_colinear_raw(
            &[(1, 1), (2, 3), (3, 3)],
            5
        ));
        assert!(!PrimeFieldPolynomial::are_colinear_raw(
            &[(1, 0), (2, 2), (3, 3)],
            5
        ));
        assert!(PrimeFieldPolynomial::are_colinear_raw(
            &[(15, 92), (11, 76), (19, 108)],
            193
        ));
        assert!(!PrimeFieldPolynomial::are_colinear_raw(
            &[(12, 92), (11, 76), (19, 108)],
            193
        ));
    }

    #[test]
    fn modular_arithmetic_mul_and_div0() {
        let pqr = PolynomialQuotientRing::new(4, 7); // degree: 4, mod prime: 7
        let a = PrimeFieldPolynomial {
            coefficients: vec![6, 2, 5],
            pqr: &pqr,
        };
        let b = PrimeFieldPolynomial {
            coefficients: vec![4, 2],
            pqr: &pqr,
        };
        let mul: PrimeFieldPolynomial = a.mul(&b);
        let div: PrimeFieldPolynomial = mul.div(&b).0;
        assert_eq!(a, div);
    }

    #[test]
    fn modular_arithmetic_mul_and_div1() {
        let pqr = PolynomialQuotientRing::new(4, 7);
        let a = PrimeFieldPolynomial {
            coefficients: vec![6, 2, 5, 1, 2, 3, 5, 3, 1, 0, 2],
            pqr: &pqr,
        };
        let b = PrimeFieldPolynomial {
            coefficients: vec![5, 2],
            pqr: &pqr,
        };
        let mul: PrimeFieldPolynomial = a.mul(&b);
        let div: PrimeFieldPolynomial = mul.div(&b).0;
        assert_eq!(a, div);
    }

    #[test]
    fn modular_arithmetic_mul_and_div2() {
        let pqr = PolynomialQuotientRing::new(4, 5); // degree: 4, mod prime: 5
        let a = PrimeFieldPolynomial {
            coefficients: vec![1, 0, 2],
            pqr: &pqr,
        };
        let b = PrimeFieldPolynomial {
            coefficients: vec![2, 4],
            pqr: &pqr,
        };
        let expected_quotient = PrimeFieldPolynomial {
            coefficients: vec![1, 3],
            pqr: &pqr,
        };
        let expected_remainder = PrimeFieldPolynomial {
            coefficients: vec![4],
            pqr: &pqr,
        };
        let (quotient, remainder) = a.div(&b);
        assert_eq!(expected_quotient, quotient);
        assert_eq!(expected_remainder, remainder);
        assert_eq!(a.degree(), 2);
        assert_eq!(b.degree(), 1);
        assert_eq!(expected_remainder.degree(), 0);
    }

    #[test]
    fn modular_arithmetic_polynomial_property_based_test() {
        let prime_modulus = 53;
        let pqr = PolynomialQuotientRing::new(16, prime_modulus); // degree: 16, mod prime: prime_modulus
        let a_degree = 20;
        for i in 0..20 {
            let mut a = PrimeFieldPolynomial {
                coefficients: generate_random_numbers(a_degree, prime_modulus),
                pqr: &pqr,
            };
            a.normalize();
            let mut b = PrimeFieldPolynomial {
                coefficients: generate_random_numbers(a_degree + i, prime_modulus),
                pqr: &pqr,
            };
            b.normalize();

            let mul_a_b = a.mul(&b);
            let mul_b_a = b.mul(&a);
            let add_a_b = a.add(&b);
            let sub_a_b = a.sub(&b);
            let sub_b_a = b.sub(&a);
            let add_b_a = b.add(&a);
            assert_eq!(mul_a_b.div(&b).0, a);
            assert_eq!(add_a_b.sub(&b), a);
            assert_eq!(sub_a_b.add(&b), a);
            assert_eq!(mul_b_a.div(&a).0, b);
            assert_eq!(add_b_a.sub(&a), b);
            assert_eq!(sub_b_a.add(&a), b);
            assert_eq!(add_a_b, add_b_a);
            assert_eq!(mul_a_b, mul_b_a);
            assert!(a.degree() < a_degree);
            assert!(b.degree() < a_degree + i);
            assert!(mul_a_b.degree() <= (a_degree - 1) * 2 + i);
            assert!(add_a_b.degree() < a_degree + i);
        }
    }

    #[test]
    fn degree() {
        let pqr = PolynomialQuotientRing::new(4, 5); // degree: 4, mod prime: 5
        let a = PrimeFieldPolynomial {
            coefficients: vec![],
            pqr: &pqr,
        };
        assert_eq!(0, a.degree());
    }

    #[test]
    fn finite_field_lagrange_interpolation() {
        let pqr = PolynomialQuotientRing::new(4, 7); // degree: 4, mod prime: 7

        let field = PrimeField::new(7);
        let points = &[
            (
                PrimeFieldElement::new(0, &field),
                PrimeFieldElement::new(6, &field),
            ),
            (
                PrimeFieldElement::new(1, &field),
                PrimeFieldElement::new(6, &field),
            ),
            (
                PrimeFieldElement::new(2, &field),
                PrimeFieldElement::new(2, &field),
            ),
        ];
        let interpolation_result =
            PrimeFieldPolynomial::finite_field_lagrange_interpolation(points, &pqr);
        let expected_interpolation_result = PrimeFieldPolynomial {
            coefficients: vec![6, 2, 5],
            pqr: &pqr,
        };
        for point in points {
            assert_eq!(interpolation_result.evaluate(&point.0).value, point.1.value);
        }
        assert_eq!(expected_interpolation_result, interpolation_result);
    }

    #[test]
    fn property_based_test_finite_field_lagrange_interpolation() {
        // Autogenerate a `number_of_points - 1` degree polynomial
        // We start by autogenerating the polynomial, as we would get a polynomial
        // with fractional coefficients if we autogenerated the points and derived the polynomium
        // from that.
        let field = PrimeField::new(999983i128);
        let number_of_points = 50usize;
        let pqr = PolynomialQuotientRing::new(256, field.q);
        let mut coefficients: Vec<i128> = Vec::with_capacity(number_of_points);
        for _ in 0..number_of_points {
            coefficients.push((rand::random::<i128>() % field.q + field.q) % field.q);
        }

        let pol = PrimeFieldPolynomial {
            coefficients,
            pqr: &pqr,
        };

        // Evaluate this in `number_of_points` points
        // Pretty ugly since I don't yet understand Rust lifetime parameters. Sue me.
        let points: Vec<(PrimeFieldElement, PrimeFieldElement)> = (0..number_of_points)
            .map(|x| {
                let x = PrimeFieldElement::new(x as i128, &field);
                (x, PrimeFieldElement::new(0, &field))
            })
            .collect();
        let mut new_points = points.clone();
        for i in 0..new_points.len() {
            new_points[i].1 = pol.evaluate(&points[i].0);
        }

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result =
            PrimeFieldPolynomial::finite_field_lagrange_interpolation(&new_points, &pqr);
        assert_eq!(interpolation_result, pol);
        for point in new_points {
            assert_eq!(interpolation_result.evaluate(&point.0), point.1);
        }
        println!("interpolation_result = {}", interpolation_result);
        println!("pol = {}", pol);
    }

    #[test]
    fn integer_lagrange_interpolation() {
        let points = &[(-1, 1), (0, 0), (1, 1)];
        let interpolation_result = IntegerRingPolynomial::integer_lagrange_interpolation(points);
        let expected_interpolation_result = IntegerRingPolynomial {
            coefficients: vec![0, 0, 1],
        };
        assert_eq!(expected_interpolation_result, interpolation_result);
        for point in points {
            assert_eq!(interpolation_result.evaluate(point.0), point.1);
        }
    }

    #[test]
    fn property_based_test_integer_lagrange_interpolation() {
        // Autogenerate a `number_of_points - 1` degree polynomial
        // We start by autogenerating the polynomial, as we would get a polynomial
        // with fractional coefficients if we autogenerated the points and derived the polynomium
        // from that. And we cannot currently handle non-integer coefficients.
        let number_of_points = 17;
        let mut coefficients: Vec<i128> = Vec::with_capacity(number_of_points);
        for _ in 0..number_of_points {
            coefficients.push(rand::random::<i128>() % 16i128);
        }

        let pol = IntegerRingPolynomial { coefficients };

        // Evaluate this in `number_of_points` points
        let mut points: Vec<(i128, i128)> = Vec::with_capacity(number_of_points);
        for i in 0..number_of_points {
            let x = i as i128 - number_of_points as i128 / 2;
            let y = pol.evaluate(x);
            points.push((x as i128, y));
        }

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result = IntegerRingPolynomial::integer_lagrange_interpolation(&points);
        for point in points {
            assert_eq!(interpolation_result.evaluate(point.0), point.1);
        }
        assert_eq!(interpolation_result, pol);
    }

    #[test]
    fn new_test() {
        let pqr = PolynomialQuotientRing::new(4, 7); // degree: 4, mod prime: 7
        let mut a: PrimeFieldPolynomial = PrimeFieldPolynomial {
            coefficients: vec![1, 0],
            pqr: &pqr,
        };
        let mut b: PrimeFieldPolynomial = PrimeFieldPolynomial {
            coefficients: vec![1, 1],
            pqr: &pqr,
        };
        let mut sum: PrimeFieldPolynomial = PrimeFieldPolynomial {
            coefficients: vec![2, 1],
            pqr: &pqr,
        };
        assert_eq!(a.add(&b), sum);

        let mut diff: PrimeFieldPolynomial = PrimeFieldPolynomial {
            coefficients: vec![0, 6],
            pqr: &pqr,
        };
        assert_eq!(a.sub(&b), diff);

        let mut mul: PrimeFieldPolynomial = PrimeFieldPolynomial {
            coefficients: vec![1, 1],
            pqr: &pqr,
        };
        assert_eq!(a.mul(&b), mul);

        a = PrimeFieldPolynomial {
            coefficients: vec![1, 0, 5],
            pqr: &pqr,
        };
        b = PrimeFieldPolynomial {
            coefficients: vec![1],
            pqr: &pqr,
        };
        sum = PrimeFieldPolynomial {
            coefficients: vec![2, 0, 5],
            pqr: &pqr,
        };
        assert_eq!(a.add(&b), sum);

        diff = PrimeFieldPolynomial {
            coefficients: vec![0, 0, 5],
            pqr: &pqr,
        };
        assert_eq!(a.sub(&b), diff);

        mul = PrimeFieldPolynomial {
            coefficients: vec![1, 0, 5],
            pqr: &pqr,
        };
        assert_eq!(a.mul(&b), mul);

        a = PrimeFieldPolynomial {
            coefficients: vec![1, 5, 6],
            pqr: &pqr,
        };
        b = PrimeFieldPolynomial {
            coefficients: vec![3, 2, 4, 3],
            pqr: &pqr,
        };
        sum = PrimeFieldPolynomial {
            coefficients: vec![4, 0, 3, 3],
            pqr: &pqr,
        };
        assert_eq!(a.add(&b), sum);

        diff = PrimeFieldPolynomial {
            coefficients: vec![5, 3, 2, 4],
            pqr: &pqr,
        };
        assert_eq!(a.sub(&b), diff);

        mul = PrimeFieldPolynomial {
            coefficients: vec![3, 3, 4, 0, 4, 4],
            pqr: &pqr,
        };
        assert_eq!(a.mul(&b), mul);

        a = PrimeFieldPolynomial {
            coefficients: vec![0, 0, 1],
            pqr: &pqr,
        };
        b = PrimeFieldPolynomial {
            coefficients: vec![0, 1],
            pqr: &pqr,
        };
        let mut quotient = PrimeFieldPolynomial {
            coefficients: vec![0, 1],
            pqr: &pqr,
        };
        let mut remainder = PrimeFieldPolynomial {
            coefficients: vec![],
            pqr: &pqr,
        };
        let mut div_result = a.div(&b);
        div_result.0.normalize();
        div_result.1.normalize();
        assert_eq!(div_result.0, quotient);
        assert_eq!(div_result.1, remainder);
        assert_eq!(remainder.to_string(), "0");
        assert_eq!(a.to_string(), "x^2");
        assert_eq!(b.to_string(), "x");

        a = PrimeFieldPolynomial {
            coefficients: vec![-4, 0, -2, 1],
            pqr: &pqr,
        };
        b = PrimeFieldPolynomial {
            coefficients: vec![-3, 1],
            pqr: &pqr,
        };
        quotient = PrimeFieldPolynomial {
            coefficients: vec![3, 1, 1],
            pqr: &pqr,
        };
        remainder = PrimeFieldPolynomial {
            coefficients: vec![5],
            pqr: &pqr,
        };
        let mut div_result = a.div(&b);
        div_result.0.normalize();
        div_result.1.normalize();
        assert_eq!(div_result.0, quotient);
        assert_eq!(div_result.1, remainder);

        a = PrimeFieldPolynomial {
            coefficients: vec![-10, 0, 1, 3, -4, 5],
            pqr: &pqr,
        };
        b = PrimeFieldPolynomial {
            coefficients: vec![2, -1, 1],
            pqr: &pqr,
        };
        quotient = PrimeFieldPolynomial {
            coefficients: vec![0, 1, 1, 5],
            pqr: &pqr,
        };
        remainder = PrimeFieldPolynomial {
            coefficients: vec![4, 5],
            pqr: &pqr,
        };
        div_result = a.div(&b);
        div_result.0.normalize();
        div_result.1.normalize();
        assert_eq!(div_result.0, quotient);
        assert_eq!(div_result.1, remainder);

        let empty_pol = PrimeFieldPolynomial {
            coefficients: vec![],
            pqr: &pqr,
        };
        assert_eq!(a.get_constant_term(), -10);
        assert_eq!(empty_pol.get_constant_term(), 0);
        assert_eq!(a.to_string(), "5x^5 - 4x^4 + 3x^3 + x^2 - 10");
        assert_eq!(b.to_string(), "x^2 - x + 2");
        assert_eq!(quotient.to_string(), "5x^3 + x^2 + x");
        assert_eq!(remainder.to_string(), "5x + 4");

        // Let's test: normalize, additive_identitive, polynomial_from_int
        // random_generators, balance, scalar_mul, scalar_modulus, scalar_mul_float
        b = PrimeFieldPolynomial {
            coefficients: vec![0, 1, 0],
            pqr: &pqr,
        };
        let normalized = PrimeFieldPolynomial {
            coefficients: vec![0, 1],
            pqr: &pqr,
        };
        let five = PrimeFieldPolynomial {
            coefficients: vec![5],
            pqr: &pqr,
        };
        assert_ne!(b, normalized);
        b.normalize();
        assert_eq!(b, normalized);
        assert_eq!(PrimeFieldPolynomial::additive_identity(&pqr), empty_pol);
        assert_eq!(PrimeFieldPolynomial::polynomium_from_int(5, &pqr), five);

        // Test binary polynomial generation
        let binary = PrimeFieldPolynomial::gen_binary_poly(&pqr);
        assert!(binary.coefficients.len() <= pqr.n as usize);
        for elem in binary.coefficients.iter() {
            assert!(*elem == 0 || *elem == 1);
        }

        // Test balance on quotient = 5x^3 + x^2 + x
        let balanced = quotient.balance();
        let expected_balanced = PrimeFieldPolynomial {
            coefficients: vec![0, 1, 1, -2],
            pqr: &pqr,
        };
        assert!(balanced.coefficients.len() == quotient.coefficients.len());
        assert_eq!(balanced, expected_balanced);

        // Test scalar_mul
        let scalar_mul = quotient.scalar_mul(3);
        let expected_scalar_mul = PrimeFieldPolynomial {
            coefficients: vec![0, 3, 3, 1],
            pqr: &pqr,
        };
        assert!(quotient.coefficients.len() == scalar_mul.coefficients.len());
        assert_eq!(scalar_mul, expected_scalar_mul);

        // Test scalar_modulus
        let scalar_modulus = scalar_mul.scalar_modulus(2);
        let expected_scalar_modulus = PrimeFieldPolynomial {
            coefficients: vec![0, 1, 1, 1],
            pqr: &pqr,
        };
        assert!(scalar_mul.coefficients.len() == scalar_modulus.coefficients.len());
        assert_eq!(expected_scalar_modulus, scalar_modulus);

        // Test scalar_mul_float
        let scalar_mul_float = scalar_mul.scalar_mul_float(3.4f64);
        let expected_scalar_mul_float = PrimeFieldPolynomial {
            coefficients: vec![0, 10, 10, 3],
            pqr: &pqr,
        };
        assert!(scalar_mul.coefficients.len() == scalar_mul_float.coefficients.len());
        assert_eq!(expected_scalar_mul_float, scalar_mul_float);
    }
}
