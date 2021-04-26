mod ciphertext;
use ciphertext::Ciphertext;
mod polynomial;
use polynomial::Polynomial;
mod polynomial_quotient_ring;
use polynomial_quotient_ring::PolynomialQuotientRing;
mod public_key;
use public_key::PublicKey;
mod keypair;
use keypair::KeyPair;

fn encrypt<'a>(pk: &'a PublicKey<'a>, plain_text_modulus: i128, pt: i128) -> Ciphertext<'a> {
    let e1 = Polynomial::gen_normal_poly(pk.a.pqr);
    // println!("e1: {}", e1);
    let e2 = Polynomial::gen_normal_poly(pk.a.pqr);
    // println!("e2: {}", e2);
    let delta = pk.a.pqr.q / plain_text_modulus;
    let m = Polynomial::polynomium_from_int(pt, pk.a.pqr).scalar_modulus(plain_text_modulus);
    // println!("m: {}", m);
    let scaled_m = m.scalar_mul(delta).scalar_modulus(pk.a.pqr.q);
    // println!("scaled_m: {}", scaled_m);
    let u = Polynomial::gen_binary_poly(pk.a.pqr).modulus().normalize();
    // println!("u: {}", u);
    let ct0 = pk.b.mul(&u).add(&e1).add(&scaled_m).modulus().normalize();
    let ct1 = pk.a.mul(&u).add(&e2).modulus().normalize();
    Ciphertext { ct0, ct1 }
}

fn decrypt<'a>(
    sk: &'a Polynomial<'a>,
    plain_text_modulus: i128,
    ciphertext: &'a Ciphertext,
) -> Polynomial<'a> {
    let t_over_q: f64 = plain_text_modulus as f64 / sk.pqr.q as f64;
    // println!("t_over_q = {}", t_over_q);
    let scaled_pt = ciphertext.ct1.mul(&sk).add(&ciphertext.ct0);
    // println!("scaled_pt = {}", scaled_pt);
    let unscaled_pt = scaled_pt
        .scalar_mul_float(t_over_q)
        .modulus()
        .normalize()
        .scalar_modulus(plain_text_modulus);
    // println!("unscaled_pt = {}", unscaled_pt);
    unscaled_pt.normalize()
}

pub fn test() {
    let pqr = PolynomialQuotientRing::new(4, 11);
    let pqr_weird = PolynomialQuotientRing::new(4, 999983i128);
    let long_quotient = Polynomial {
        coefficients: vec![7, 0, 23, 65, 1, 2, 14, 14, 14, 14, 3, 19, 6, 20],
        pqr: &pqr_weird,
    };
    let pqr_weird_polynomial = Polynomial {
        coefficients: pqr_weird.get_polynomial_modulus(),
        pqr: &pqr_weird,
    };
    println!(
        "{} / {} = {}",
        long_quotient,
        pqr_weird_polynomial,
        long_quotient.modulus()
    );
    let pol = Polynomial {
        coefficients: vec![4, 9, 9, 1, 0, 0],
        pqr: &pqr,
    };
    let a = Polynomial {
        coefficients: vec![4, 9, 9, 1, 0, 0],
        pqr: &pqr,
    };
    let pol_mod = Polynomial {
        coefficients: pqr.get_polynomial_modulus(),
        pqr: &pqr,
    };
    let c = Polynomial {
        coefficients: vec![5, 0, 3],
        pqr: &pqr,
    };
    let d = Polynomial {
        coefficients: vec![3, 4, 0, 0],
        pqr: &pqr,
    };
    let leading_zeros = Polynomial {
        coefficients: vec![0, 0, 0, 0, 4, 2],
        pqr: &pqr,
    };
    let leading_zeros_normalized = leading_zeros.clone().normalize();
    let mul_result = c.mul(&d);
    mul_result.modulus();
    // Verify that leading zeros warner works!
    println!("Polynomial with leading zeros: {:?}", leading_zeros);
    println!(
        "A polynomial with leading zeros is printed as: {}",
        leading_zeros
    );
    println!(
        "normalize({:?}) = {:?}",
        leading_zeros, leading_zeros_normalized,
    );
    println!("({}) * ({}) = {}", a, pol_mod, a.mul(&pol_mod));
    println!("{} / {} = {}", a, pol_mod, pol.modulus());
    println!("({}) * ({}) = {} = {}", c, d, c.mul(&d), mul_result);
    println!("{} + {} = {}", a, pol_mod, a.add(&pol_mod));
    println!("{} - ({}) = {}", a, pol_mod, a.sub(&pol_mod));
    println!(
        "Random binary polynomial: {}",
        Polynomial::gen_binary_poly(a.pqr),
    );
    println!(
        "Random uniform polynomial of size 7: {}",
        Polynomial::gen_uniform_poly(a.pqr),
    );
    println!(
        "Random normal distributed polynomial of size 7: {}",
        Polynomial::gen_normal_poly(a.pqr),
    );
    let key_pair = KeyPair::keygen(a.pqr);
    println!("A randomly generated keypair is: {}", key_pair);
    let plain_text = 2i128;
    let pt_modulus = 5i128;
    let ciphertext = encrypt(&key_pair.pk, pt_modulus, plain_text);
    println!(
        "{} encrypted under this key is: ct0={}, ct1={}",
        plain_text, ciphertext.ct0, ciphertext.ct1
    );
    println!(
        "Decrypting this, we get: {}",
        decrypt(&key_pair.sk, pt_modulus, &ciphertext)
    );

    let test_pqr = PolynomialQuotientRing::new(16, 32768i128);
    let pt_modulus_test = 256;
    let pt_test_1 = 73;
    let pt_test_2 = 20;
    let key_pair_test = KeyPair::keygen(&test_pqr);
    println!("A new randomly generated keypair is: {}", key_pair_test);
    let ct1_test = encrypt(&key_pair_test.pk, pt_modulus_test, pt_test_1);
    let ct2_test = encrypt(&key_pair_test.pk, pt_modulus_test, pt_test_2);
    println!(
        "\n\n\n\n{} encrypted under this key is: ct0={}, ct1={}",
        pt_test_1, ct1_test.ct0, ct1_test.ct1
    );
    println!(
        "{} encrypted under this key is: ct0={}, ct1={}",
        pt_test_2, ct2_test.ct0, ct2_test.ct1
    );
    let decrypted_ct1_test = decrypt(&key_pair_test.sk, pt_modulus_test, &ct1_test);
    let decrypted_ct2_test = decrypt(&key_pair_test.sk, pt_modulus_test, &ct2_test);
    println!("Decrypting this, we get: {}", decrypted_ct1_test);
    println!(
        "Leaving us with the number: {}",
        (&decrypted_ct1_test.coefficients).last().unwrap()
    );
    println!("Decrypting this, we get: {}", decrypted_ct2_test);
    println!(
        "Leaving us with the number: {}",
        (&decrypted_ct2_test.coefficients).last().unwrap()
    );
    println!(
        "Taking a modulus here, we get: {}, {}",
        (&decrypted_ct1_test.coefficients).last().unwrap(),
        (&decrypted_ct2_test.coefficients).last().unwrap()
    );

    let pt_real = 79;
    let pt_modulus_real = 256i128;
    let pqr_real = PolynomialQuotientRing::new(1024, 786433);
    let kp_real = KeyPair::keygen(&pqr_real);
    let ct_real = encrypt(&kp_real.pk, pt_modulus_real, pt_real);
    let dec_real = decrypt(&kp_real.sk, pt_modulus_real, &ct_real);
    println!(
        "Encrypting and decrypting {}, we get: {}",
        pt_real, dec_real
    );
}
