mod ciphertext;
use crate::shared_math::fraction::Fraction;
use crate::shared_math::integer_ring_polynomial::IntegerRingPolynomial;
use crate::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;
use ciphertext::Ciphertext;
use rand::RngCore;
mod keypair;
mod public_key;
use keypair::KeyPair;
mod secret_key;

pub fn test() {
    let pqr = PolynomialQuotientRing::new(4, 11);
    let pqr_weird = PolynomialQuotientRing::new(4, 999983i128);
    let long_quotient = PrimeFieldPolynomial {
        coefficients: vec![7, 0, 23, 65, 1, 2, 14, 14, 14, 14, 3, 19, 6, 20],
        pqr: &pqr_weird,
    };
    let pqr_weird_polynomial = PrimeFieldPolynomial {
        coefficients: pqr_weird.get_polynomial_modulus(),
        pqr: &pqr_weird,
    };
    println!(
        "{} / {} = {}",
        long_quotient,
        pqr_weird_polynomial,
        long_quotient.modulus()
    );
    let pol = PrimeFieldPolynomial {
        coefficients: vec![4, 9, 9, 1, 0, 0],
        pqr: &pqr,
    };
    let a = PrimeFieldPolynomial {
        coefficients: vec![4, 9, 9, 1, 0, 0],
        pqr: &pqr,
    };
    let pol_mod = PrimeFieldPolynomial {
        coefficients: pqr.get_polynomial_modulus(),
        pqr: &pqr,
    };
    let c = PrimeFieldPolynomial {
        coefficients: vec![5, 0, 3],
        pqr: &pqr,
    };
    let d = PrimeFieldPolynomial {
        coefficients: vec![3, 4, 0, 0],
        pqr: &pqr,
    };
    let leading_zeros = PrimeFieldPolynomial {
        coefficients: vec![0, 0, 0, 0, 4, 2],
        pqr: &pqr,
    };
    let leading_zeros_normalized = leading_zeros.clone().normalize();
    let mul_result = c.mul(&d);
    mul_result.modulus();
    // Verify that leading zeros warner works!
    println!(
        "PrimeFieldPolynomial with leading zeros: {:?}",
        leading_zeros
    );
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
        PrimeFieldPolynomial::gen_binary_poly(a.pqr),
    );
    println!(
        "Random uniform polynomial of size 7: {}",
        PrimeFieldPolynomial::gen_uniform_poly(a.pqr),
    );
    println!(
        "Random normal distributed polynomial of size 7: {}",
        PrimeFieldPolynomial::gen_normal_poly(a.pqr),
    );
    let key_pair = KeyPair::keygen(a.pqr);
    println!("A randomly generated keypair is: {}", key_pair);
    let plain_text = 2i128;
    let pt_modulus = 5i128;
    let ciphertext = key_pair.pk.encrypt(pt_modulus, plain_text);
    println!(
        "{} encrypted under this key is: ct0={}, ct1={}",
        plain_text, ciphertext.ct0, ciphertext.ct1
    );
    println!(
        "Decrypting this, we get: {}",
        key_pair.sk.decrypt(pt_modulus, &ciphertext)
    );

    // Lagrange interpolation
    let interpolation =
        IntegerRingPolynomial::integer_lagrange_interpolation(&[(0, 0), (1, 1), (-1, 1)]);
    println!(
        "interpolation result of points (0, 0), (1, 1), (-1, 1) is: {}",
        interpolation
    );

    // Fractions
    let frac = Fraction::new(5, 2);
    let sum = frac + frac;
    println!("{} + {} = {}", frac, frac, sum);
    println!("dividend of {} = {}", frac, frac.get_dividend());
    println!("divisor of {} = {}", frac, frac.get_divisor());
    println!("{} * {} = {}", 6, frac, frac.scalar_mul(6));
    println!("{} / {} = {}", frac, 7, frac.scalar_div(7));

    let test_pqr = PolynomialQuotientRing::new(16, 32768i128);
    let pt_modulus_test = 256;
    let pt_test_1 = 73;
    let pt_test_2 = 20;
    let key_pair_test = KeyPair::keygen(&test_pqr);
    println!("A new randomly generated keypair is: {}", key_pair_test);
    let ct1_test = key_pair_test.pk.encrypt(pt_modulus_test, pt_test_1);
    let ct2_test = key_pair_test.pk.encrypt(pt_modulus_test, pt_test_2);
    println!(
        "\n\n\n\n{} encrypted under this key is: ct0={}, ct1={}",
        pt_test_1, ct1_test.ct0, ct1_test.ct1
    );
    println!(
        "{} encrypted under this key is: ct0={}, ct1={}",
        pt_test_2, ct2_test.ct0, ct2_test.ct1
    );
    let decrypted_ct1_test = key_pair_test.sk.decrypt(pt_modulus_test, &ct1_test);
    let decrypted_ct2_test = key_pair_test.sk.decrypt(pt_modulus_test, &ct2_test);
    println!("Decrypting this, we get: {}", decrypted_ct1_test);
    println!("Decrypting this, we get: {}", decrypted_ct2_test);

    let pt_real = 39;
    let pt_modulus_real = 64i128;
    let pqr_real = PolynomialQuotientRing::new(1024, 786433);
    let kp_real = KeyPair::keygen(&pqr_real);
    let mut ct_real = kp_real.pk.encrypt(pt_modulus_real, pt_real);
    let dec_real = kp_real.sk.decrypt(pt_modulus_real, &ct_real);
    println!(
        "Encrypting and decrypting {}, we get: {}",
        pt_real, dec_real
    );

    ct_real.add_plain(7, pt_modulus_real);
    let decrypted_ct3 = kp_real.sk.decrypt(pt_modulus_real, &ct_real);
    println!("Encrypting adding 7 and decrypting: {}", decrypted_ct3);

    ct_real.mul_plain(3, pt_modulus_real);
    let decrypted_ct4 = kp_real.sk.decrypt(pt_modulus_real, &ct_real);
    println!(
        "Encrypting adding 7, multiplying by 3 and decrypting: {}",
        decrypted_ct4
    );

    let mut k = 0;
    while k < 1 {
        println!("k = {}", k);
        for i in 0..3 {
            let pt_new = i;
            let ct_new = kp_real.pk.encrypt(pt_modulus_real, pt_new);
            let res: i128 = kp_real.sk.decrypt(pt_modulus_real, &ct_new);
            println!("Encrypting and decrypting {}, we get: {}", pt_new, res);
            if res != i {
                println!("Failed on {}, got: {}", i, res);
                break;
            } else {
                println!("Success on {}", i);
            }
        }
        k += 1;
    }

    // encrypting and decrypting with plaintext mul and addition on the ciphertext
    let mut prng = rand::thread_rng();
    k = 0;
    while k < 1 {
        println!("k = {}", k);
        for i in 1..10 {
            let pt_new = i;
            let mut ct_new = kp_real.pk.encrypt(pt_modulus_real, pt_new);

            // multiply ciphertext with x and add y
            let mul_value: i128 = (prng.next_u32() % (pt_modulus_real as u32)) as i128;
            let add_value: i128 = (prng.next_u32() % (pt_modulus_real as u32)) as i128;
            ct_new.mul_plain(mul_value, pt_modulus_real);
            ct_new.add_plain(add_value, pt_modulus_real);

            let res: i128 = kp_real.sk.decrypt(pt_modulus_real, &ct_new);
            let expected: i128 = (i * mul_value + add_value) % pt_modulus_real;
            println!(
                "{} * {} + {} mod {} on the ciphertext gives us: {}",
                i, mul_value, add_value, pt_modulus_real, res
            );
            if res != expected {
                println!("Failed on {}, got: {}, expected: {}", i, res, expected);
                break;
            } else {
                println!("Success on {}", i);
            }
        }
        k += 1;
    }

    // Encrypting and decrypting with sum of ciphertexts
    let pt0 = 7;
    let pt1 = 11;
    let mut ct_new0: Ciphertext = kp_real.pk.encrypt(pt_modulus_real, pt0);
    let ct_new1 = kp_real.pk.encrypt(pt_modulus_real, pt1);
    ct_new0.add_cipher(&ct_new1);
    let res = kp_real.sk.decrypt(pt_modulus_real, &ct_new0);
    println!("Decrypting {} + {} = {}", pt0, pt1, res);

    // Add encrypted zero to a plaintext and verify that it the ciphertext changes
    println!(
        "18 encrypted ends in : {} and in {}",
        ct_new0.ct0.get_constant_term(),
        ct_new0.ct1.get_constant_term(),
    );
    let ct_new_zero: Ciphertext = kp_real.pk.encrypt(pt_modulus_real, 0);
    ct_new0.add_cipher(&ct_new_zero);
    println!(
        "18 encrypted plus encrypted 0 ends in: {} and in {}",
        ct_new0.ct0.get_constant_term(),
        ct_new0.ct1.get_constant_term(),
    );
}
