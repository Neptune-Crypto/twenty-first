use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use rand::Rng;
use rand::RngCore;
use rand_distr::StandardNormal;
use std::convert::TryInto;
use std::fmt;

#[derive(Debug, Clone)]
struct PolynomialQuotientRing {
    degree: i128,
    n: i128, // n = 2^p, always a power of 2.
    q: i128, // The prime modulus, satisfies q = 1 mod 2n
    // polynomial_modulus should be an immutable reference to a Polynomial whose lifetime is the
    // same as this object -- it should die, if this object dies.
    polynomial_modulus: Vec<i128>,
}

// All structs holding references must have lifetime annotations in their definition.
// This <'a> annotation means that an instance of Polynomial cannot outlive the reference it holds
// in its `pqr` field.
#[derive(Debug, Clone)]
struct Polynomial<'a> {
    // I think we cannot use arrays to store the coefficients, since array sizes must be known at compile time
    coefficients: Vec<i128>,
    pqr: &'a PolynomialQuotientRing,
}

// Public key should maybe also have a `size` field
#[derive(Debug)]
struct PublicKey<'a> {
    a: Polynomial<'a>,
    b: Polynomial<'a>,
}

impl fmt::Display for PublicKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(b={}, a={})", self.b, self.a)
    }
}

// KeyPair should maybe also have a `size` field
#[derive(Debug)]
struct KeyPair<'a> {
    pk: PublicKey<'a>,
    sk: Polynomial<'a>,
}

impl fmt::Display for KeyPair<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pk: {}, sk: {}", self.pk, self.sk)
    }
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
    fn additive_identity(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        Polynomial {
            coefficients: Vec::new(),
            pqr,
        }
    }

    fn polynomium_from_int(val: i128, pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
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

    fn gen_binary_poly(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        let res = Polynomial {
            coefficients: Polynomial::generate_random_numbers(pqr.n as usize, 2i128),
            pqr,
        };
        println!("{:?}", res);
        // TODO: Do we take the modulus here?
        res.normalize()
    }

    fn gen_uniform_poly(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        let res = Polynomial {
            coefficients: Polynomial::generate_random_numbers(pqr.n as usize, pqr.q),
            pqr,
        };
        res.modulus().normalize()
    }

    fn gen_normal_poly(pqr: &'a PolynomialQuotientRing) -> Polynomial<'a> {
        // The tutorial recommends N(0, 2) for this
        let mut coefficients: Vec<i128> = vec![0i128; pqr.n as usize];
        let mut prng = rand::thread_rng();
        for elem in coefficients.iter_mut() {
            let val: f64 = prng.sample(StandardNormal);
            let int_val = ((val.round() as i128 % pqr.q) + pqr.q) % pqr.q;
            *elem = int_val;
        }
        let ret = Polynomial { coefficients, pqr };
        ret.modulus().normalize()
    }

    fn keygen(pqr: &'a PolynomialQuotientRing) -> KeyPair {
        let sk: Polynomial = Polynomial::gen_binary_poly(pqr);
        let a: Polynomial = Polynomial::gen_uniform_poly(pqr);
        let e: Polynomial = Polynomial::gen_normal_poly(pqr);
        let zero: Polynomial = Polynomial::additive_identity(pqr);
        let b: Polynomial = zero.sub(&a).mul(&sk).sub(&e).modulus();
        let pk = PublicKey { a, b };
        KeyPair { pk, sk }
    }

    fn encrypt(
        pk: &'a PublicKey<'a>,
        plain_text_modulus: i128,
        pt: i128,
    ) -> (Polynomial, Polynomial) {
        let e1 = Polynomial::gen_normal_poly(pk.a.pqr);
        println!("e1: {}", e1);
        let e2 = Polynomial::gen_normal_poly(pk.a.pqr);
        println!("e2: {}", e2);
        let delta = pk.a.pqr.q / plain_text_modulus;
        let m = Polynomial::polynomium_from_int(pt, pk.a.pqr).scalar_modulus(plain_text_modulus);
        println!("m: {}", m);
        let scaled_m = m.scalar_mul(delta).scalar_modulus(pk.a.pqr.q);
        println!("scaled_m: {}", scaled_m);
        // TODO: Not sure if we take modulus here!
        let u = Polynomial::gen_binary_poly(pk.a.pqr).modulus().normalize();
        // let u = Polynomial::gen_binary_poly(pk.a.pqr)
        println!("u: {}", u);
        let ct0 = pk.b.mul(&u).add(&e1).add(&scaled_m).modulus().normalize();
        let ct1 = pk.a.mul(&u).add(&e2).modulus().normalize();
        (ct0, ct1)
    }

    fn decrypt(
        sk: &'a Polynomial<'a>,
        plain_text_modulus: i128,
        ciphertext: &(Polynomial<'a>, Polynomial<'a>),
    ) -> Polynomial<'a> {
        let t_over_q: f64 = plain_text_modulus as f64 / sk.pqr.q as f64;
        println!("t_over_q = {}", t_over_q);
        let scaled_pt = ciphertext.1.mul(&sk).add(&ciphertext.0);
        println!("scaled_pt = {}", scaled_pt);
        let unscaled_pt = scaled_pt
            .scalar_mul_float(t_over_q)
            .modulus()
            .normalize()
            .scalar_modulus(plain_text_modulus);
        println!("unscaled_pt = {}", unscaled_pt);
        unscaled_pt.normalize()
    }

    // Remove leading zeros from coefficients
    fn normalize(self) -> Polynomial<'a> {
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
    fn div(&self, divisor: &Polynomial<'a>) -> (Polynomial<'a>, Polynomial<'a>) {
        if self.coefficients.is_empty() {
            return (
                Polynomial::additive_identity(self.pqr),
                Polynomial::additive_identity(self.pqr),
            );
        }

        let dividend_coeffs: &Vec<i128> = &self.coefficients;
        let divisor_coeffs: &Vec<i128> = &divisor.coefficients;
        let mut quotient: Vec<i128> = Vec::with_capacity(self.pqr.degree.try_into().unwrap());
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
    fn modulus(&self) -> Polynomial<'a> {
        let polynomial_modulus = Polynomial {
            coefficients: self.pqr.polynomial_modulus.clone(),
            pqr: self.pqr,
        };
        self.div(&polynomial_modulus).1
        // self.coefficients = result.coefficients;
    }

    fn mul(&self, other: &Polynomial<'a>) -> Polynomial<'a> {
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

    fn scalar_mul(&self, scalar: i128) -> Polynomial<'a> {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] = (((coefficients[i] * scalar) % self.pqr.q) + self.pqr.q) % self.pqr.q;
        }
        Polynomial {
            coefficients,
            pqr: self.pqr,
        }
    }

    fn scalar_modulus(&self, modulus: i128) -> Polynomial<'a> {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] %= modulus;
        }
        Polynomial {
            coefficients,
            pqr: self.pqr,
        }
    }

    fn scalar_mul_float(&self, float_scalar: f64) -> Polynomial<'a> {
        let mut coefficients = self.coefficients.clone();
        for i in 0..self.coefficients.len() {
            coefficients[i] = ((((coefficients[i] as f64 * float_scalar).round() as i128)
                % self.pqr.q)
                + self.pqr.q)
                % self.pqr.q;
        }
        Polynomial {
            coefficients,
            pqr: self.pqr,
        }
    }

    fn add(&self, other: &Polynomial<'a>) -> Polynomial<'a> {
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

    fn sub(&self, other: &Polynomial<'a>) -> Polynomial<'a> {
        let negative_coefficients = other.coefficients.clone().iter().map(|x| -x).collect();
        let neg_pol = Polynomial {
            coefficients: negative_coefficients,
            pqr: self.pqr,
        };
        // The adding should normalize the result, so we don't need to do it here too.
        self.add(&neg_pol)
    }
}

pub fn test() {
    let pqr = PolynomialQuotientRing {
        degree: 4i128,
        n: 4i128,
        q: 11i128,
        polynomial_modulus: vec![1, 0, 0, 0, 1],
    };
    let pqr_weird = PolynomialQuotientRing {
        degree: 4i128,
        n: 4i128,
        q: 999983i128,
        polynomial_modulus: vec![1, 0, 4, 3, 8],
    };
    let long_quotient = Polynomial {
        coefficients: vec![7, 0, 23, 65, 1, 2, 14, 14, 14, 14, 3, 19, 6, 20],
        pqr: &pqr_weird,
    };
    let pqr_weird_polynomial = Polynomial {
        coefficients: pqr_weird.polynomial_modulus.clone(),
        pqr: &pqr_weird,
    };
    println!(
        "{} / {} = {}",
        long_quotient,
        pqr_weird_polynomial,
        long_quotient.modulus()
    );
    let pol = Polynomial {
        coefficients: vec![4, 9, 9, 1, 0, 0],
        pqr: &pqr,
    };
    let a = Polynomial {
        coefficients: vec![4, 9, 9, 1, 0, 0],
        pqr: &pqr,
    };
    let pol_mod = Polynomial {
        coefficients: pqr.polynomial_modulus.clone(),
        pqr: &pqr,
    };
    let c = Polynomial {
        coefficients: vec![5, 0, 3],
        pqr: &pqr,
    };
    let d = Polynomial {
        coefficients: vec![3, 4, 0, 0],
        pqr: &pqr,
    };
    let leading_zeros = Polynomial {
        coefficients: vec![0, 0, 0, 0, 4, 2],
        pqr: &pqr,
    };
    let leading_zeros_normalized = leading_zeros.clone().normalize();
    let mul_result = c.mul(&d);
    mul_result.modulus();
    // Verify that leading zeros warner works!
    println!("Polynomial with leading zeros: {:?}", leading_zeros);
    println!(
        "A polynomial with leading zeros is printed as: {}",
        leading_zeros
    );
    println!(
        "normalize({:?}) = {:?}",
        leading_zeros, leading_zeros_normalized,
    );
    println!("({}) * ({}) = {}", a, pol_mod, a.mul(&pol_mod));
    println!("{} / {} = {}", a, pol_mod, pol.modulus());
    println!("({}) * ({}) = {} = {}", c, d, c.mul(&d), mul_result);
    println!("{} + {} = {}", a, pol_mod, a.add(&pol_mod));
    println!("{} - ({}) = {}", a, pol_mod, a.sub(&pol_mod));
    println!(
        "Random binary polynomial: {}",
        Polynomial::gen_binary_poly(a.pqr),
    );
    println!(
        "Random uniform polynomial of size 7: {}",
        Polynomial::gen_uniform_poly(a.pqr),
    );
    println!(
        "Random normal distributed polynomial of size 7: {}",
        Polynomial::gen_normal_poly(a.pqr),
    );
    let key_pair = Polynomial::keygen(a.pqr);
    println!("A randomly generated keypair is: {}", key_pair);
    let plain_text = 2i128;
    let pt_modulus = 5i128;
    let ciphertext = Polynomial::encrypt(&key_pair.pk, pt_modulus, plain_text);
    println!(
        "{} encrypted under this key is: ct0={}, ct1={}",
        plain_text, ciphertext.0, ciphertext.1
    );
    println!(
        "Decrypting this, we get: {}",
        Polynomial::decrypt(&key_pair.sk, pt_modulus, &ciphertext)
    );

    let test_pqr = PolynomialQuotientRing {
        polynomial_modulus: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        degree: 16i128,
        n: 16i128,
        q: 32768i128,
    };
    let pt_modulus_test = 256;
    let pt_test_1 = 73;
    let pt_test_2 = 20;
    let key_pair_test = Polynomial::keygen(&test_pqr);
    println!("A new randomly generated keypair is: {}", key_pair_test);
    let ct1_test = Polynomial::encrypt(&key_pair_test.pk, pt_modulus_test, pt_test_1);
    let ct2_test = Polynomial::encrypt(&key_pair_test.pk, pt_modulus_test, pt_test_2);
    println!(
        "{} encrypted under this key is: ct0={}, ct1={}",
        pt_test_1, ct1_test.0, ct1_test.1
    );
    println!(
        "{} encrypted under this key is: ct0={}, ct1={}",
        pt_test_2, ct2_test.0, ct2_test.1
    );
    let decrypted_ct1_test = Polynomial::decrypt(&key_pair_test.sk, pt_modulus_test, &ct1_test);
    let decrypted_ct2_test = Polynomial::decrypt(&key_pair_test.sk, pt_modulus_test, &ct2_test);
    println!("Decrypting this, we get: {}", decrypted_ct1_test);
    println!(
        "Leaving us with the number: {}",
        (&decrypted_ct1_test.coefficients).last().unwrap()
    );
    println!("Decrypting this, we get: {}", decrypted_ct2_test);
    println!(
        "Leaving us with the number: {}",
        (&decrypted_ct2_test.coefficients).last().unwrap()
    );
    println!(
        "Taking a modulus here, we get: {}, {}",
        (&decrypted_ct1_test.coefficients).last().unwrap(),
        (&decrypted_ct2_test.coefficients).last().unwrap()
    );
}
