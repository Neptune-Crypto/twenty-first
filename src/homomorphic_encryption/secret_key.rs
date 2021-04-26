use super::ciphertext::Ciphertext;
use super::polynomial::Polynomial;

#[derive(Debug)]
pub struct SecretKey<'a> {
    pub value: Polynomial<'a>,
}

impl<'a> SecretKey<'a> {
    pub fn decrypt(&self, plain_text_modulus: i128, ciphertext: &'a Ciphertext) -> Polynomial<'a> {
        let t_over_q: f64 = plain_text_modulus as f64 / self.value.pqr.q as f64;
        // println!("t_over_q = {}", t_over_q);
        let scaled_pt = ciphertext.ct1.mul(&self.value).add(&ciphertext.ct0);
        // println!("scaled_pt = {}", scaled_pt);
        let unscaled_pt = scaled_pt
            .scalar_mul_float(t_over_q)
            .modulus()
            .normalize()
            .scalar_modulus(plain_text_modulus);
        // println!("unscaled_pt = {}", unscaled_pt);
        unscaled_pt.normalize()
    }
}
