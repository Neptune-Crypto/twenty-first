use super::fraction::Fraction;
use crate::utils::has_unique_elements;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use num_traits::Zero;
use std::convert::From;
use std::fmt;

fn pretty_print_coefficients(coefficients: &[i128]) -> String {
    if coefficients.is_empty() {
        return String::from("0");
    }

    let trailing_zeros_warning: &str = match coefficients.last() {
        Some(0) => "Trailing zeros!",
        _ => "",
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

    #[must_use]
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

    #[must_use]
    pub fn scalar_mul(&self, scalar: i128) -> Self {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] *= scalar;
        }
        Self { coefficients }
    }

    #[must_use]
    pub fn scalar_modulus(&self, modulus: i128) -> Self {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] %= modulus;
        }
        Self { coefficients }
    }

    #[must_use]
    pub fn scalar_mul_float(&self, float_scalar: f64) -> Self {
        // Function does not map coefficients into finite field. Should it?
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] = (coefficients[i] as f64 * float_scalar).round() as i128;
        }
        Self { coefficients }
    }

    #[must_use]
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

    #[must_use]
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
mod test_integer_ring_polynomials {
    use super::*;
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
}
