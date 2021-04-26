#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialQuotientRing {
    // polynomial_modulus should be an immutable reference to a Polynomial whose lifetime is the
    // same as this object -- it should die, if this object dies.
    pub n: i128, // n = 2^p, always a power of 2.
    pub q: i128,
    polynomial_modulus: Vec<i128>,
}

impl PolynomialQuotientRing {
    pub fn new(n: usize, q: i128) -> Self {
        let mut polynomial_modulus = vec![0i128; n + 1];
        polynomial_modulus[0] = 1;
        polynomial_modulus[n] = 1;
        PolynomialQuotientRing {
            n: n as i128,
            q,
            polynomial_modulus,
        }
    }

    pub fn get_polynomial_modulus(&self) -> Vec<i128> {
        self.polynomial_modulus.clone()
    }
}
