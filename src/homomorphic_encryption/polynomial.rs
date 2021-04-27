use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use rand::Rng;
use rand::RngCore;
use rand_distr::Normal;
use std::convert::TryInto;
use std::fmt;

use super::polynomial_quotient_ring::PolynomialQuotientRing;

// All structs holding references must have lifetime annotations in their definition.
// This <'a> annotation means that an instance of Polynomial cannot outlive the reference it holds
// in its `pqr` field.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<'a> {
    // I think we cannot use arrays to store the coefficients, since array sizes must be known at compile time
    pub coefficients: Vec<i128>,
    pub pqr: &'a PolynomialQuotientRing,
}

impl fmt::Display for Polynomial<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coefficients.is_empty() {
            return write!(f, "0");
        }

        let leading_zeros_warning: &str = match self.coefficients.as_slice() {
            [0, ..] => &"Leading zeros!",
            _ => &"",
        };
        let mut outputs: Vec<String> = Vec::new();
        let degree = self.coefficients.len() - 1;
        for i in 0..=degree {
            if self.coefficients[i] != 0 {
                outputs.push(format!(
                    "{}{}",
                    if self.coefficients[i] == 1 {
                        String::from("")
                    } else {
                        self.coefficients[i].to_string()
                    },
                    // TODO: Replace by match!
                    if degree - i == 0 && self.coefficients[i] == 1 {
                        String::from("1")
                    } else if degree - i == 0 {
                        String::from("")
                    } else if degree - i == 1 {
                        String::from("x")
                    } else {
                        let mut result = "x^".to_owned();
                        let borrowed_string = (degree - i).to_string().to_owned();
                        result.push_str(&borrowed_string);
                        result
                    }
                ));
            }
        }
        write!(f, "{}{}", leading_zeros_warning, outputs.join("+"))
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
        let res = Polynomial {
            coefficients: Polynomial::generate_random_numbers(pqr.n as usize, 2i128),
            pqr,
        };
        // TODO: Do we take the modulus here?
        res.normalize()
    }

    pub fn gen_uniform_poly(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        let res = Polynomial {
            coefficients: Polynomial::generate_random_numbers(pqr.n as usize, pqr.q),
            pqr,
        };
        res.modulus().normalize()
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
        let ret = Polynomial { coefficients, pqr };
        ret.modulus().normalize()
    }

    // Remove leading zeros from coefficients
    pub fn normalize(self) -> Polynomial<'a> {
        let mut a: Vec<i128> = self.coefficients.clone();
        while let [0, ..] = a.as_slice() {
            a.remove(0);
        }
        Polynomial {
            coefficients: a,
            pqr: self.pqr,
        }
    }

    // Doing long division, return the pair (div, mod), coefficients are always floored!
    // TODO: Verify lifetime parameters here!
    pub fn div(&self, divisor: &Polynomial<'a>) -> (Polynomial<'a>, Polynomial<'a>) {
        if self.coefficients.is_empty() {
            return (
                Polynomial::additive_identity(self.pqr),
                Polynomial::additive_identity(self.pqr),
            );
        }

        let dividend_coeffs: &Vec<i128> = &self.coefficients;
        let divisor_coeffs: &Vec<i128> = &divisor.coefficients;
        let mut quotient: Vec<i128> = Vec::with_capacity(self.pqr.n.try_into().unwrap());
        let mut remainder: Vec<i128> = self.coefficients.clone();
        let dividend_degree = dividend_coeffs.len() - 1;
        let divisor_degree = divisor_coeffs.len() - 1;
        let mut i = 0;
        while i + divisor_degree <= dividend_degree {
            let mut res = remainder[0] / divisor_coeffs[0]; // result is always floored!
            res = ((res % self.pqr.q) + self.pqr.q) % self.pqr.q;
            quotient.push(res);

            // multiply divisor by result just obtained
            // Create the correction value which is subtracted from the remainder
            let mut correction = Vec::new();
            for div_coeff in divisor_coeffs.iter() {
                correction.push(div_coeff * res);
            }
            let zeros = &mut vec![0i128; dividend_degree - divisor_degree - i];
            correction.append(zeros);

            // calculate rem - correction
            remainder.remove(0); // TODO: This is probably slow!
            correction.remove(0); // TODO: This is probably slow!
            for j in 0..correction.len() {
                remainder[j] =
                    (((remainder[j] - correction[j]) % self.pqr.q) + self.pqr.q) % self.pqr.q;
            }

            i += 1;
        }

        // Divide the first coefficient in the dividend w/ the first non-zero coefficient of the divisor.
        // Store the result in the quotient.
        let quotient_pol = Polynomial {
            coefficients: quotient,
            pqr: self.pqr,
        };
        let remainder_pol = Polynomial {
            coefficients: remainder,
            pqr: self.pqr,
        };
        (quotient_pol.normalize(), remainder_pol.normalize())
    }

    // TODO: Rewrite to return new polynomial instead of manipulating it inplace
    // The reason that this is better is that it allows for chaining of operators
    pub fn modulus(&self) -> Polynomial<'a> {
        let polynomial_modulus = Polynomial {
            coefficients: self.pqr.get_polynomial_modulus(),
            pqr: self.pqr,
        };
        self.div(&polynomial_modulus).1
        // self.coefficients = result.coefficients;
    }

    pub fn mul(&self, other: &Polynomial<'a>) -> Polynomial<'a> {
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Polynomial::additive_identity(self.pqr);
        }
        let self_degree = self.coefficients.len() - 1;
        let other_degree = other.coefficients.len() - 1;
        let mut self_rev = self.coefficients.clone();
        self_rev.reverse();
        let mut other_rev = other.coefficients.clone();
        other_rev.reverse();
        let mut result_coeff: Vec<i128> = vec![0i128; self_degree + other_degree + 1];
        for i in 0..=self_degree {
            for j in 0..=other_degree {
                let mul = self_rev[i] * other_rev[j];
                result_coeff[i + j] += mul;
                result_coeff[i + j] =
                    ((result_coeff[i + j] % self.pqr.q) + self.pqr.q) % self.pqr.q;
            }
        }
        result_coeff.reverse();
        let ret = Polynomial {
            coefficients: result_coeff,
            pqr: self.pqr,
        };

        ret.normalize()
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
        // Create reversed copy of self.coefficients and of other.coefficients
        let mut self_reversed = self.coefficients.clone();
        self_reversed.reverse();
        let mut other_reversed = other.coefficients.clone();
        other_reversed.reverse();
        let mut summed: Vec<i128> = self_reversed
            .iter()
            .zip_longest(other_reversed.iter())
            .map(|a: itertools::EitherOrBoth<&i128, &i128>| match a {
                Both(l, r) => (((*l + *r) % self.pqr.q) + self.pqr.q) % self.pqr.q,
                Left(l) => *l,
                Right(r) => *r,
            })
            .collect();
        summed.reverse();
        let ret = Polynomial {
            coefficients: summed,
            pqr: self.pqr,
        };

        ret.normalize()
    }

    pub fn sub(&self, other: &Polynomial<'a>) -> Polynomial<'a> {
        let negative_coefficients = other.coefficients.clone().iter().map(|x| -x).collect();
        let neg_pol = Polynomial {
            coefficients: negative_coefficients,
            pqr: self.pqr,
        };
        // The adding should normalize the result, so we don't need to do it here too.
        self.add(&neg_pol)
    }
}

#[cfg(test)]
mod test_polynomial {
    use super::*;

    #[test]
    fn internal() {
        let pqr = PolynomialQuotientRing::new(4, 7);
        let a: Polynomial = Polynomial {
            coefficients: vec![1],
            pqr: &pqr,
        };
        let b: Polynomial = Polynomial {
            coefficients: vec![2],
            pqr: &pqr,
        };
        let big: Polynomial = Polynomial {
            coefficients: vec![3, 5, 1, 6],
            pqr: &pqr,
        };
        let big_squared = Polynomial {
            coefficients: vec![4, 3, 3, 5],
            pqr: &pqr,
        };
        let zero = Polynomial::additive_identity(&pqr);
        assert!(a.add(&a) == b);
        assert!(a.sub(&a) == zero);
        assert!(big.mul(&a) == big);
        assert!(big.mul(&big).modulus() == big_squared);
    }
}
