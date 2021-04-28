use super::polynomial::Polynomial;
use std::fmt;

#[derive(Debug)]
pub struct Ciphertext<'a> {
    pub ct0: Polynomial<'a>,
    pub ct1: Polynomial<'a>,
}

impl fmt::Display for Ciphertext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ct_0 = {}; ct_1 = {}", self.ct0, self.ct1)
    }
}

impl<'a> Ciphertext<'a> {
    // inline manipulation of the ciphertext
    pub fn add_plain(&mut self, pt: i128, pt_modulus: i128) {
        let pt_pol = Polynomial::polynomium_from_int(pt, self.ct0.pqr).scalar_modulus(pt_modulus);
        let delta = self.ct0.pqr.q / pt_modulus;
        let scaled_m = pt_pol.scalar_mul(delta).scalar_modulus(self.ct0.pqr.q);
        let new_ct0 = self.ct0.add(&scaled_m).modulus();
        self.ct0 = new_ct0;
    }

    // Inline manipulation of the ciphertext
    pub fn mul_plain(&mut self, pt: i128, pt_modulus: i128) {
        let pt_pol = Polynomial::polynomium_from_int(pt, self.ct0.pqr).scalar_modulus(pt_modulus);
        self.ct0 = self.ct0.mul(&pt_pol).modulus();
        self.ct1 = self.ct1.mul(&pt_pol).modulus();
    }

    // Since the decryption function dec :: Z_q^2 -> Z_q = ct_0 + sk*ct_1
    // is linear s.t. dec(ct + ct') = dec(ct) + dec(ct'), two cipher texts
    // can simply be added, and the decryption of this will be the sum of the
    // decrypted individual parts.
    pub fn add_cipher(&mut self, other: &Self) {
        self.ct0 = self.ct0.add(&other.ct0).modulus();
        self.ct1 = self.ct1.add(&other.ct1).modulus();
    }
}
