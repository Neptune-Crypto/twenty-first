use super::public_key::PublicKey;
use super::secret_key::SecretKey;
use crate::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;
use std::fmt;

// KeyPair should maybe also have a `size` field
#[derive(Debug)]
pub struct KeyPair<'a> {
    pub pk: PublicKey<'a>,
    pub sk: SecretKey<'a>,
}

impl fmt::Display for KeyPair<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pk: {}, sk: {}", self.pk, self.sk.value)
    }
}

impl<'a> KeyPair<'a> {
    pub fn keygen(pqr: &'a PolynomialQuotientRing) -> Self {
        let secret_key: PrimeFieldPolynomial = PrimeFieldPolynomial::gen_binary_poly(pqr);
        // println!("secret_key = {}", secret_key);
        let a: PrimeFieldPolynomial = PrimeFieldPolynomial::gen_uniform_poly(pqr);
        // println!("{}", a);
        let mut e: PrimeFieldPolynomial = PrimeFieldPolynomial::gen_normal_poly(pqr);
        e.normalize();
        // println!("{}", e);
        let zero: PrimeFieldPolynomial = PrimeFieldPolynomial::additive_identity(pqr);
        // println!("{}", zero);
        let mut b: PrimeFieldPolynomial = zero.sub(&a).mul(&secret_key).sub(&e).modulus();
        b.normalize();
        // println!("{}", b);
        let pk = PublicKey { a, b };
        KeyPair {
            pk,
            sk: SecretKey { value: secret_key },
        }
    }
}
