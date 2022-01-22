use crate::shared_math::traits::{IdentityValues, ModPowU32};

use super::traits::PrimeFieldElement;

// This NTT implementation is adapted from inspired by Longa and Naehrig[0]
// and from dusk network/Plon[1]
// [0]: https://eprint.iacr.org/2016/504.pdf
// [1]: https://github.com/dusk-network/plonk/blob/d3412cec5fa5c2e720f848a6fd8db96d663e92a9/src/fft/domain.rs#L310
#[inline]
fn bitreverse(mut n: u32, l: u32) -> u32 {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

pub fn chu_ntt<T: PrimeFieldElement>(a: &mut [T::Elem], omega: T::Elem, log_2_of_n: u32) {
    let n = a.len() as u32;
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
            a.swap(rk as usize, k as usize);
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
                let mut t = a[(k + j + m) as usize].clone();
                t *= w.clone();
                let mut tmp = a[(k + j) as usize].clone();
                tmp -= t.clone();
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize] += t;
                w *= w_m.clone();
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

#[cfg(test)]
mod fast_ntt_attempt_tests {
    use crate::shared_math::{b_field_element::BFieldElement, traits::GetPrimitiveRootOfUnity};

    use super::*;

    #[test]
    fn basic_test_of_fast_ntt() {
        let mut input_output = vec![
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
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

        chu_ntt::<BFieldElement>(&mut input_output, omega, 2);
        assert_eq!(expected, input_output);
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
        let omega = BFieldElement::ring_one()
            .get_primitive_root_of_unity(32)
            .0
            .unwrap();
        chu_ntt::<BFieldElement>(&mut input_output, omega, 5);
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
    }
}

// input
// BFieldElement::new(1),
// BFieldElement::new(4),
// BFieldElement::new(0),
// BFieldElement::new(0),

// output
// BFieldElement(5), BFieldElement(1125899906842625), BFieldElement(18446744069414584318), BFieldElement(18445618169507741698)
