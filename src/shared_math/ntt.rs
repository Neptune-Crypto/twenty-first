use crate::shared_math::traits::ModPowU32;
use crate::shared_math::traits::PrimeField;
use crate::shared_math::traits::{IdentityValues, New};

// This NTT implementation is adapted from inspired by Longa and Naehrig[0]
// and from dusk network/Plon[1]
// [0]: https://eprint.iacr.org/2016/504.pdf
// [1]: https://github.com/dusk-network/plonk/blob/d3412cec5fa5c2e720f848a6fd8db96d663e92a9/src/fft/domain.rs#L310

#[allow(clippy::many_single_char_names)]
pub fn ntt<PF: PrimeField>(x: &mut [PF::Elem], omega: PF::Elem, log_2_of_n: u32) {
    let n = x.len() as u32;
    assert_eq!(
        n,
        1 << log_2_of_n,
        "Order must match length of input vector"
    );
    debug_assert!(omega.mod_pow_u32(n).is_one());
    debug_assert!(!omega.mod_pow_u32(n / 2).is_one());

    for k in 0..n {
        let rk = bitreverse(k, log_2_of_n);
        if k < rk {
            x.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_2_of_n {
        // let w_m = omega.mod_pow_u32(&[(n / (2 * m)) as u64, 0, 0, 0]);
        let w_m = omega.mod_pow_u32(n / (2 * m));

        let mut k = 0;
        while k < n {
            let mut w = omega.ring_one();
            for j in 0..m {
                let mut t = x[(k + j + m) as usize];
                t *= w;
                let mut tmp = x[(k + j) as usize];
                tmp -= t;
                x[(k + j + m) as usize] = tmp;
                x[(k + j) as usize] += t;
                w *= w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

pub fn intt<PF: PrimeField>(x: &mut [PF::Elem], omega: PF::Elem, log_2_of_n: u32) {
    let n: PF::Elem = omega.new_from_usize(x.len());
    let n_inv: PF::Elem = omega.ring_one() / n;
    ntt::<PF>(x, omega.ring_one() / omega, log_2_of_n);
    for elem in x.iter_mut() {
        *elem *= n_inv;
    }
}

#[inline]
fn bitreverse(mut n: u32, l: u32) -> u32 {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

#[cfg(test)]
mod fast_ntt_attempt_tests {
    use crate::shared_math::{
        b_field_element::BFieldElement,
        traits::{GetPrimitiveRootOfUnity, GetRandomElements},
        x_field_element::XFieldElement,
    };

    use super::*;

    #[test]
    fn chu_ntt_b_field_prop_test() {
        let mut rng = rand::thread_rng();
        for log_2_n in 1..10 {
            let n = 1 << log_2_n;
            for _ in 0..10 {
                let mut values = BFieldElement::random_elements(n, &mut rng);
                let original_values = values.clone();
                let omega = BFieldElement::ring_one()
                    .get_primitive_root_of_unity(n as u128)
                    .0
                    .unwrap();
                ntt::<BFieldElement>(&mut values, omega, log_2_n);
                assert_ne!(original_values, values);
                intt::<BFieldElement>(&mut values, omega, log_2_n);
                assert_eq!(original_values, values);

                values[0] = BFieldElement::new(BFieldElement::MAX);
                let original_values_with_max_element = values.clone();
                ntt::<BFieldElement>(&mut values, omega, log_2_n);
                assert_ne!(original_values, values);
                intt::<BFieldElement>(&mut values, omega, log_2_n);
                assert_eq!(original_values_with_max_element, values);
            }
        }
    }

    #[test]
    fn chu_ntt_x_field_prop_test() {
        let mut rng = rand::thread_rng();
        for log_2_n in 1..10 {
            let n = 1 << log_2_n;
            for _ in 0..10 {
                let mut values = XFieldElement::random_elements(n, &mut rng);
                let original_values = values.clone();
                let omega = XFieldElement::ring_one()
                    .get_primitive_root_of_unity(n as u128)
                    .0
                    .unwrap();
                ntt::<XFieldElement>(&mut values, omega, log_2_n);
                assert_ne!(original_values, values);
                intt::<XFieldElement>(&mut values, omega, log_2_n);
                assert_eq!(original_values, values);

                // Verify that we are not just operating in the B-field
                // statistically this should hold except one out of
                // ~ (2^64)^2 times this test runs
                assert!(
                    !original_values[1].coefficients[1].is_zero()
                        || !original_values[1].coefficients[2].is_zero()
                );

                values[0] = XFieldElement::new([
                    BFieldElement::new(BFieldElement::MAX),
                    BFieldElement::new(BFieldElement::MAX),
                    BFieldElement::new(BFieldElement::MAX),
                ]);
                let original_values_with_max_element = values.clone();
                ntt::<XFieldElement>(&mut values, omega, log_2_n);
                assert_ne!(original_values, values);
                intt::<XFieldElement>(&mut values, omega, log_2_n);
                assert_eq!(original_values_with_max_element, values);
            }
        }
    }

    #[test]
    fn xfield_basic_test_of_chu_ntt() {
        let mut input_output = vec![
            XFieldElement::new_const(BFieldElement::ring_one()),
            XFieldElement::new_const(BFieldElement::ring_zero()),
            XFieldElement::new_const(BFieldElement::ring_zero()),
            XFieldElement::new_const(BFieldElement::ring_zero()),
        ];
        let original_input = input_output.clone();
        let expected = vec![
            XFieldElement::new_const(BFieldElement::ring_one()),
            XFieldElement::new_const(BFieldElement::ring_one()),
            XFieldElement::new_const(BFieldElement::ring_one()),
            XFieldElement::new_const(BFieldElement::ring_one()),
        ];
        let omega = XFieldElement::ring_one()
            .get_primitive_root_of_unity(4)
            .0
            .unwrap();

        println!("input_output = {:?}", input_output);
        ntt::<XFieldElement>(&mut input_output, omega, 2);
        assert_eq!(expected, input_output);
        println!("input_output = {:?}", input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<XFieldElement>(&mut input_output, omega, 2);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn bfield_basic_test_of_chu_ntt() {
        let mut input_output = vec![
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        let original_input = input_output.clone();
        let expected = vec![
            BFieldElement::new(5),
            BFieldElement::new(1125899906842625),
            BFieldElement::new(18446744069414584318),
            BFieldElement::new(18445618169507741698),
        ];
        let omega = BFieldElement::ring_one()
            .get_primitive_root_of_unity(4)
            .0
            .unwrap();

        ntt::<BFieldElement>(&mut input_output, omega, 2);
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output, omega, 2);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn bfield_max_value_test_of_chu_ntt() {
        let mut input_output = vec![
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        let original_input = input_output.clone();
        let expected = vec![
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::new(BFieldElement::MAX),
        ];
        let omega = BFieldElement::ring_one()
            .get_primitive_root_of_unity(4)
            .0
            .unwrap();

        ntt::<BFieldElement>(&mut input_output, omega, 2);
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output, omega, 2);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn b_field_ntt_with_length_32() {
        let mut input_output = vec![
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        let original_input = input_output.clone();
        let omega = BFieldElement::ring_one()
            .get_primitive_root_of_unity(32)
            .0
            .unwrap();
        ntt::<BFieldElement>(&mut input_output, omega, 5);
        // let actual_output = ntt(&mut input_output, &omega, 5);
        println!("actual_output = {:?}", input_output);
        let expected = vec![
            BFieldElement::new(20),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(18446744069146148869),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(4503599627370500),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(18446726477228544005),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(18446744069414584309),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(268435460),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(18442240469787213829),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(17592186040324),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output, omega, 5);
        assert_eq!(original_input, input_output);
    }
}
