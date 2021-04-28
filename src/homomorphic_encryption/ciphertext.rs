use super::polynomial::Polynomial;

#[derive(Debug)]
pub struct Ciphertext<'a> {
    pub ct0: Polynomial<'a>,
    pub ct1: Polynomial<'a>,
}

impl<'a> Ciphertext<'a> {
    // inline manipulation of the ciphertext
    pub fn add_plain(&mut self, pt: i128, pt_modulus: i128) {
        let pt_pol = Polynomial::polynomium_from_int(pt, self.ct0.pqr).scalar_modulus(pt_modulus);
        let delta = self.ct0.pqr.q / pt_modulus;
        let scaled_m = pt_pol.scalar_mul(delta).scalar_modulus(self.ct0.pqr.q);
        let new_ct0 = self.ct0.add(&scaled_m);
        self.ct0 = new_ct0;
    }

    // Inline manipulation of the ciphertext
    pub fn mul_plain(&mut self, pt: i128, pt_modulus: i128) {
        let pt_pol = Polynomial::polynomium_from_int(pt, self.ct0.pqr).scalar_modulus(pt_modulus);
        self.ct0 = self.ct0.mul(&pt_pol);
        self.ct1 = self.ct1.mul(&pt_pol);
    }
}
