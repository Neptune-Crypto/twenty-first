use super::polynomial::Polynomial;
use std::fmt;

// Public key should maybe also have a `size` field
#[derive(Debug)]
pub struct PublicKey<'a> {
    pub a: Polynomial<'a>,
    pub b: Polynomial<'a>,
}

impl fmt::Display for PublicKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(b={}, a={})", self.b, self.a)
    }
}
