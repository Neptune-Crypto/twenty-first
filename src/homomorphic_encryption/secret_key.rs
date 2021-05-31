use super::ciphertext::Ciphertext;
use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;

#[derive(Debug)]
pub struct SecretKey<'a> {
    pub value: PrimeFieldPolynomial<'a>,
}

impl<'a> SecretKey<'a> {
    pub fn decrypt(&self, plain_text_modulus: i128, ciphertext: &'a Ciphertext) -> i128 {
        let t_over_q: f64 = plain_text_modulus as f64 / self.value.pqr.q as f64;
        // println!("t_over_q = {}", t_over_q);
        let scaled_pt = ciphertext
            .ct1
            .mul(&self.value)
            .add(&ciphertext.ct0)
            .modulus();

        // If scaled_pt coeffs is larger than q/2, subtract q
        let scaled_pt_balanced = scaled_pt.balance();

        // println!("scaled_pt = {}", scaled_pt);
        let unscaled_pt: PrimeFieldPolynomial = scaled_pt_balanced.scalar_mul_float(t_over_q);

        // Extract the constant term and map into [0; ptm - 1]
        (unscaled_pt.get_constant_term() % plain_text_modulus + plain_text_modulus)
            % plain_text_modulus
    }
}
