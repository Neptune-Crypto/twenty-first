use super::polynomial::Polynomial;
use super::polynomial_quotient_ring::PolynomialQuotientRing;
use super::public_key::PublicKey;
use std::fmt;

// KeyPair should maybe also have a `size` field
#[derive(Debug)]
pub struct KeyPair<'a> {
    pub pk: PublicKey<'a>,
    pub sk: Polynomial<'a>,
}

impl fmt::Display for KeyPair<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pk: {}, sk: {}", self.pk, self.sk)
    }
}

impl<'a> KeyPair<'a> {
    pub fn keygen(pqr: &'a PolynomialQuotientRing) -> Self {
        let sk: Polynomial = Polynomial::gen_binary_poly(pqr);
        let a: Polynomial = Polynomial::gen_uniform_poly(pqr);
        let e: Polynomial = Polynomial::gen_normal_poly(pqr);
        let zero: Polynomial = Polynomial::additive_identity(pqr);
        let b: Polynomial = zero.sub(&a).mul(&sk).sub(&e).modulus();
        let pk = PublicKey { a, b };
        KeyPair { pk, sk }
    }
}
