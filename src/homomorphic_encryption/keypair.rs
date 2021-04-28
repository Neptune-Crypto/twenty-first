use super::polynomial::Polynomial;
use super::polynomial_quotient_ring::PolynomialQuotientRing;
use super::public_key::PublicKey;
use super::secret_key::SecretKey;
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
        let secret_key: Polynomial = Polynomial::gen_binary_poly(pqr);
        // println!("secret_key = {}", secret_key);
        let a: Polynomial = Polynomial::gen_uniform_poly(pqr);
        // println!("{}", a);
        let e: Polynomial = Polynomial::gen_normal_poly(pqr).normalize();
        // println!("{}", e);
        let zero: Polynomial = Polynomial::additive_identity(pqr);
        // println!("{}", zero);
        let b: Polynomial = zero.sub(&a).mul(&secret_key).sub(&e).modulus().normalize();
        // println!("{}", b);
        let pk = PublicKey { a, b };
        KeyPair {
            pk,
            sk: SecretKey { value: secret_key },
        }
    }
}
