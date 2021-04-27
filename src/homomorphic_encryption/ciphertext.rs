use super::polynomial::Polynomial;

#[derive(Debug)]
pub struct Ciphertext<'a> {
    pub ct0: Polynomial<'a>,
    pub ct1: Polynomial<'a>,
}

impl<'a> Ciphertext<'a> {
    pub fn add_plain(&self, pt: i128, pt_modulus: i128) -> Ciphertext {
        let pt_pol = Polynomial::polynomium_from_int(pt, self.ct0.pqr).scalar_modulus(pt_modulus);
        let delta = self.ct0.pqr.q / pt_modulus;
        let scaled_m = pt_pol.scalar_mul(delta).scalar_modulus(self.ct0.pqr.q);
        let new_ct0 = self.ct0.add(&scaled_m);
        Ciphertext {
            ct0: new_ct0,
            ct1: self.ct1.clone(),
        }
    }

    pub fn mul_plain(&self, pt: i128, pt_modulus: i128) -> Ciphertext {
        let pt_pol = Polynomial::polynomium_from_int(pt, self.ct0.pqr).scalar_modulus(pt_modulus);
        let new_c0 = self.ct0.mul(&pt_pol);
        let new_c1 = self.ct1.mul(&pt_pol);
        Ciphertext {
            ct0: new_c0,
            ct1: new_c1,
        }
    }
}
