use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use num_traits::Zero;
use rand::Rng;
use rand::RngCore;
use rand_distr::Normal;
use std::collections::HashSet;
use std::convert::From;
use std::fmt;
use std::hash::Hash;

use super::fraction::Fraction;
use super::polynomial_quotient_ring::PolynomialQuotientRing;

// All structs holding references must have lifetime annotations in their definition.
// This <'a> annotation means that an instance of Polynomial cannot outlive the reference it holds
// in its `pqr` field.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<'a> {
    // I think we cannot use arrays to store the coefficients, since array sizes must be known at compile time
    // So we use vectors instead.
    pub coefficients: Vec<i128>,
    pub pqr: &'a PolynomialQuotientRing,
}

impl fmt::Display for Polynomial<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coefficients.is_empty() {
            return write!(f, "0");
        }

        let trailing_zeros_warning: &str = match self.coefficients.last() {
            Some(0) => &"Leading zeros!",
            _ => &"",
        };

        let mut outputs: Vec<String> = Vec::new();
        let pol_degree = self.coefficients.len() - 1;
        for i in 0..=pol_degree {
            let pow = pol_degree - i;

            if self.coefficients[pow] != 0 {
                // operator: plus or minus
                let mut operator_str = "";
                if i != 0 && self.coefficients[pow] != 0 {
                    operator_str = if self.coefficients[pow] > 0 {
                        " + "
                    } else {
                        " - "
                    };
                }

                let abs_val = if self.coefficients[pow] < 0 {
                    -self.coefficients[pow]
                } else {
                    self.coefficients[pow]
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
        write!(f, "{}{}", trailing_zeros_warning, outputs.join(""))
    }
}

impl<'a> Polynomial<'a> {
    pub fn additive_identity(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        Polynomial {
            coefficients: Vec::new(),
            pqr,
        }
    }

    pub fn polynomium_from_int(val: i128, pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        Polynomial {
            coefficients: vec![val],
            pqr,
        }
    }

    pub fn evaluate(&self, x: i128) -> i128 {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(i, &c)| c * x.pow(i as u32))
            .sum()
    }

    pub fn lagrange_interpolation(
        points: &[(i128, i128)],
        pqr: &'a PolynomialQuotientRing,
    ) -> Polynomial<'a> {
        fn has_unique_elements<T>(iter: T) -> bool
        where
            T: IntoIterator,
            T::Item: Eq + Hash,
        {
            let mut uniq = HashSet::new();
            iter.into_iter().all(move |x| uniq.insert(x))
        }
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
        let big_pol = Polynomial {
            coefficients: big_pol_coeffs,
            pqr,
        };
        for point in points.iter() {
            // create a polynomial that is zero at all other points than this
            // coeffs_j = prod_{i=0, i != j}^{N}((x- q_i))
            let mut my_pol = Polynomial {
                coefficients: vec![-point.0, 1],
                pqr,
            };
            my_pol = big_pol.div_without_modulus(&my_pol).0;

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

        let coefficients = fracs.iter().map(|&x| x.get_dividend()).collect();

        Polynomial { coefficients, pqr }
    }

    // should be a private function
    fn generate_random_numbers(size: usize, modulus: i128) -> Vec<i128> {
        let mut prng = rand::thread_rng();
        let mut rand = vec![0u8; size];
        prng.fill_bytes(rand.as_mut_slice());

        // This looks pretty inefficient
        // How is this done with a map instead?
        let mut coefficients: Vec<i128> = vec![0i128; size];
        for i in 0..size {
            // The modulus operator should give the remainder, as
            // all rand[i] are positive.
            coefficients[i] = rand[i] as i128 % modulus;
        }
        coefficients
    }

    pub fn gen_binary_poly(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        let mut res = Polynomial {
            coefficients: Polynomial::generate_random_numbers(pqr.n as usize, 2i128),
            pqr,
        };
        // TODO: Do we take the modulus here?
        res.normalize();
        res
    }

    pub fn gen_uniform_poly(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        let mut res = Polynomial {
            coefficients: Polynomial::generate_random_numbers(pqr.n as usize, pqr.q),
            pqr,
        };
        res = res.modulus();
        res.normalize();
        res
    }

    pub fn gen_normal_poly(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        // Alan recommends N(0, 4) for this
        let mut coefficients: Vec<i128> = vec![0i128; pqr.n as usize];
        let mut prng = rand::thread_rng();
        let normal_dist = Normal::new(0.0, 4.0).unwrap();
        for elem in coefficients.iter_mut() {
            let val: f64 = prng.sample(normal_dist);
            let int_val = ((val.round() as i128 % pqr.q) + pqr.q) % pqr.q;
            *elem = int_val;
        }
        let mut ret = Polynomial { coefficients, pqr };
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

    fn div_internal(
        &self,
        divisor: &Polynomial<'a>,
        finite_field: bool,
    ) -> (Polynomial<'a>, Polynomial<'a>) {
        if divisor.coefficients.is_empty() {
            panic!(
                "Cannot divide polynomial by zero. Got: ({})/({})",
                self, divisor
            );
        }

        if self.coefficients.is_empty() {
            return (
                Polynomial::additive_identity(self.pqr),
                Polynomial::additive_identity(self.pqr),
            );
        }

        let mut remainder = self.coefficients.clone();
        let mut quotient = Vec::with_capacity(self.pqr.n as usize); // built from back to front so must be reversed

        let dividend_coeffs: &Vec<i128> = &self.coefficients;
        let divisor_coeffs: &Vec<i128> = &divisor.coefficients;
        let divisor_degree = divisor_coeffs.len() - 1;
        let dividend_degree = dividend_coeffs.len() - 1;
        let dominant_divisor = divisor_coeffs.last().unwrap();
        let mut i = 0;
        while i + divisor_degree <= dividend_degree {
            // calculate next quotient coefficient
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

        // Map all coefficients into the finite field mod p
        if finite_field {
            for elem in quotient.iter_mut() {
                *elem = (*elem % self.pqr.q + self.pqr.q) % self.pqr.q;
            }
            for elem in remainder.iter_mut() {
                *elem = (*elem % self.pqr.q + self.pqr.q) % self.pqr.q;
            }
        }

        quotient.reverse();
        let quotient_pol = Polynomial {
            coefficients: quotient,
            pqr: self.pqr,
        };
        let remainder_pol = Polynomial {
            coefficients: remainder,
            pqr: self.pqr,
        };
        (quotient_pol, remainder_pol)
    }

    // Mainly just written for benchmarking/testing/prototyping
    pub fn div_without_modulus(
        &self,
        divisor: &Polynomial<'a>,
    ) -> (Polynomial<'a>, Polynomial<'a>) {
        self.div_internal(divisor, false)
    }

    // Doing long division, return the pair (div, mod), coefficients are always floored!
    // TODO: Verify lifetime parameters here!
    pub fn div(&self, divisor: &Polynomial<'a>) -> (Polynomial<'a>, Polynomial<'a>) {
        self.div_internal(divisor, true)
    }

    pub fn modulus(&self) -> Polynomial<'a> {
        let polynomial_modulus = Polynomial {
            coefficients: self.pqr.get_polynomial_modulus(),
            pqr: self.pqr,
        };
        self.div(&polynomial_modulus).1
    }

    pub fn mul(&self, other: &Polynomial<'a>) -> Polynomial<'a> {
        // If either polynomial is zero, return zero
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Polynomial::additive_identity(self.pqr);
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
        let mut ret = Polynomial {
            coefficients: result_coeff,
            pqr: self.pqr,
        };
        ret.normalize();
        ret
    }

    pub fn balance(&self) -> Polynomial<'a> {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            if coefficients[i] > self.pqr.q / 2 {
                coefficients[i] -= self.pqr.q;
            }
        }
        Polynomial {
            coefficients,
            pqr: self.pqr,
        }
    }

    pub fn scalar_mul(&self, scalar: i128) -> Polynomial<'a> {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] = (((coefficients[i] * scalar) % self.pqr.q) + self.pqr.q) % self.pqr.q;
        }
        Polynomial {
            coefficients,
            pqr: self.pqr,
        }
    }

    pub fn scalar_modulus(&self, modulus: i128) -> Polynomial<'a> {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] %= modulus;
        }
        Polynomial {
            coefficients,
            pqr: self.pqr,
        }
    }

    pub fn scalar_mul_float(&self, float_scalar: f64) -> Polynomial<'a> {
        // Function does not map coefficients into finite field. Should it?
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] = (coefficients[i] as f64 * float_scalar).round() as i128;
        }
        Polynomial {
            coefficients,
            pqr: self.pqr,
        }
    }

    pub fn add(&self, other: &Polynomial<'a>) -> Polynomial<'a> {
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

        Polynomial {
            coefficients: summed,
            pqr: self.pqr,
        }
    }

    pub fn sub(&self, other: &Polynomial<'a>) -> Polynomial<'a> {
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

        Polynomial {
            coefficients: diff,
            pqr: self.pqr,
        }
    }
}

#[cfg(test)]
mod test_polynomials {
    use super::*;

    #[test]
    fn lagrange_interpolation() {
        let pqr = PolynomialQuotientRing::new(4, 101); // degree: 4, mod prime: 101

        let points = &[(-1, 1), (0, 0), (1, 1)];
        let interpolation_result = Polynomial::lagrange_interpolation(points, &pqr);
        let expected_interpolation_result = Polynomial {
            coefficients: vec![0, 0, 1],
            pqr: &pqr,
        };
        assert_eq!(expected_interpolation_result, interpolation_result);
        for point in points {
            assert_eq!(interpolation_result.evaluate(point.0), point.1);
        }
    }

    #[test]
    fn property_based_test_lagrange_interpolation() {
        // Autogenerate a `number_of_points - 1` degree polynomial
        // We start by autogenerating the polynomial, as we would get a polynomial
        // with fractional coefficients if we autogenerated the points and derived the polynomium
        // from that. And we cannot currently handle non-integer coefficients.
        let number_of_points = 12;
        let pqr = PolynomialQuotientRing::new(256, std::i64::MAX as i128);
        let mut coefficients: Vec<i128> = Vec::with_capacity(number_of_points);
        for _ in 0..number_of_points {
            coefficients.push(rand::random::<i128>() % 20i128);
        }

        let pol = Polynomial {
            coefficients,
            pqr: &pqr,
        };

        // Evaluate this in `number_of_points` points
        let mut points: Vec<(i128, i128)> = Vec::with_capacity(number_of_points);
        for i in 0..number_of_points {
            let x = i as i128 - 5;
            let y = pol.evaluate(x);
            points.push((x as i128, y));
        }

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result = Polynomial::lagrange_interpolation(&points, &pqr);
        for point in points {
            assert_eq!(interpolation_result.evaluate(point.0), point.1);
        }
        assert_eq!(interpolation_result, pol);
    }

    #[test]
    fn new_test() {
        let pqr = PolynomialQuotientRing::new(4, 7); // degree: 4, mod prime: 7
        let mut a: Polynomial = Polynomial {
            coefficients: vec![1, 0],
            pqr: &pqr,
        };
        let mut b: Polynomial = Polynomial {
            coefficients: vec![1, 1],
            pqr: &pqr,
        };
        let mut sum: Polynomial = Polynomial {
            coefficients: vec![2, 1],
            pqr: &pqr,
        };
        assert_eq!(a.add(&b), sum);

        let mut diff: Polynomial = Polynomial {
            coefficients: vec![0, 6],
            pqr: &pqr,
        };
        assert_eq!(a.sub(&b), diff);

        let mut mul: Polynomial = Polynomial {
            coefficients: vec![1, 1],
            pqr: &pqr,
        };
        assert_eq!(a.mul(&b), mul);

        a = Polynomial {
            coefficients: vec![1, 0, 5],
            pqr: &pqr,
        };
        b = Polynomial {
            coefficients: vec![1],
            pqr: &pqr,
        };
        sum = Polynomial {
            coefficients: vec![2, 0, 5],
            pqr: &pqr,
        };
        assert_eq!(a.add(&b), sum);

        diff = Polynomial {
            coefficients: vec![0, 0, 5],
            pqr: &pqr,
        };
        assert_eq!(a.sub(&b), diff);

        mul = Polynomial {
            coefficients: vec![1, 0, 5],
            pqr: &pqr,
        };
        assert_eq!(a.mul(&b), mul);

        a = Polynomial {
            coefficients: vec![1, 5, 6],
            pqr: &pqr,
        };
        b = Polynomial {
            coefficients: vec![3, 2, 4, 3],
            pqr: &pqr,
        };
        sum = Polynomial {
            coefficients: vec![4, 0, 3, 3],
            pqr: &pqr,
        };
        assert_eq!(a.add(&b), sum);

        diff = Polynomial {
            coefficients: vec![5, 3, 2, 4],
            pqr: &pqr,
        };
        assert_eq!(a.sub(&b), diff);

        mul = Polynomial {
            coefficients: vec![3, 3, 4, 0, 4, 4],
            pqr: &pqr,
        };
        assert_eq!(a.mul(&b), mul);

        a = Polynomial {
            coefficients: vec![0, 0, 1],
            pqr: &pqr,
        };
        b = Polynomial {
            coefficients: vec![0, 1],
            pqr: &pqr,
        };
        let mut quotient = Polynomial {
            coefficients: vec![0, 1],
            pqr: &pqr,
        };
        let mut remainder = Polynomial {
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

        a = Polynomial {
            coefficients: vec![-4, 0, -2, 1],
            pqr: &pqr,
        };
        b = Polynomial {
            coefficients: vec![-3, 1],
            pqr: &pqr,
        };
        quotient = Polynomial {
            coefficients: vec![3, 1, 1],
            pqr: &pqr,
        };
        remainder = Polynomial {
            coefficients: vec![5],
            pqr: &pqr,
        };
        let mut div_result = a.div(&b);
        div_result.0.normalize();
        div_result.1.normalize();
        assert_eq!(div_result.0, quotient);
        assert_eq!(div_result.1, remainder);

        a = Polynomial {
            coefficients: vec![-10, 0, 1, 3, -4, 5],
            pqr: &pqr,
        };
        b = Polynomial {
            coefficients: vec![2, -1, 1],
            pqr: &pqr,
        };
        quotient = Polynomial {
            coefficients: vec![0, 1, 1, 5],
            pqr: &pqr,
        };
        remainder = Polynomial {
            coefficients: vec![4, 5],
            pqr: &pqr,
        };
        div_result = a.div(&b);
        div_result.0.normalize();
        div_result.1.normalize();
        assert_eq!(div_result.0, quotient);
        assert_eq!(div_result.1, remainder);

        let empty_pol = Polynomial {
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
        b = Polynomial {
            coefficients: vec![0, 1, 0],
            pqr: &pqr,
        };
        let normalized = Polynomial {
            coefficients: vec![0, 1],
            pqr: &pqr,
        };
        let five = Polynomial {
            coefficients: vec![5],
            pqr: &pqr,
        };
        assert_ne!(b, normalized);
        b.normalize();
        assert_eq!(b, normalized);
        assert_eq!(Polynomial::additive_identity(&pqr), empty_pol);
        assert_eq!(Polynomial::polynomium_from_int(5, &pqr), five);

        // Test binary polynomial generation
        let binary = Polynomial::gen_binary_poly(&pqr);
        assert!(binary.coefficients.len() <= pqr.n as usize);
        for elem in binary.coefficients.iter() {
            assert!(*elem == 0 || *elem == 1);
        }

        // Test balance on quotient = 5x^3 + x^2 + x
        let balanced = quotient.balance();
        let expected_balanced = Polynomial {
            coefficients: vec![0, 1, 1, -2],
            pqr: &pqr,
        };
        assert!(balanced.coefficients.len() == quotient.coefficients.len());
        assert_eq!(balanced, expected_balanced);

        // Test scalar_mul
        let scalar_mul = quotient.scalar_mul(3);
        let expected_scalar_mul = Polynomial {
            coefficients: vec![0, 3, 3, 1],
            pqr: &pqr,
        };
        assert!(quotient.coefficients.len() == scalar_mul.coefficients.len());
        assert_eq!(scalar_mul, expected_scalar_mul);

        // Test scalar_modulus
        let scalar_modulus = scalar_mul.scalar_modulus(2);
        let expected_scalar_modulus = Polynomial {
            coefficients: vec![0, 1, 1, 1],
            pqr: &pqr,
        };
        assert!(scalar_mul.coefficients.len() == scalar_modulus.coefficients.len());
        assert_eq!(expected_scalar_modulus, scalar_modulus);

        // Test scalar_mul_float
        let scalar_mul_float = scalar_mul.scalar_mul_float(3.4f64);
        let expected_scalar_mul_float = Polynomial {
            coefficients: vec![0, 10, 10, 3],
            pqr: &pqr,
        };
        assert!(scalar_mul.coefficients.len() == scalar_mul_float.coefficients.len());
        assert_eq!(expected_scalar_mul_float, scalar_mul_float);
    }
}
