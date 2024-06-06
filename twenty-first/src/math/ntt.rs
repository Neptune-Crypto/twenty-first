use std::ops::MulAssign;

use num_traits::ConstOne;
use num_traits::ConstZero;
use rand_distr::num_traits::One;

use super::b_field_element::BFieldElement;
use super::traits::FiniteField;
use super::traits::Inverse;
use super::traits::ModPowU32;

/// ## Perform NTT on slices of prime-field elements
///
/// NTTs are Number Theoretic Transforms, which are Discrete Fourier Transforms
/// (DFTs) over finite fields. This implementation specifically aims at being
/// used to compute polynomial multiplication over finite fields. NTT reduces
/// the complexity of such multiplication.
///
/// For a brief introduction to the math, see:
///
/// * <https://cgyurgyik.github.io/posts/2021/04/brief-introduction-to-ntt/>
/// * <https://www.nayuki.io/page/number-theoretic-transform-integer-dft>
///
/// The implementation is adapted from:
///
/// <pre>
/// Speeding up the Number Theoretic Transform
/// for Faster Ideal Lattice-Based Cryptography
/// Longa and Naehrig
/// https://eprint.iacr.org/2016/504.pdf
/// </pre>
///
/// as well as inspired by <https://github.com/dusk-network/plonk>
///
/// * `x` - a mutable slice of prime-field elements of length `n`
/// * `omega` - a primitive `n`th root of unity
/// * `log_2_of_n` - a precomputation of *log2(`n`)* to avoid repeating its
///   computation
///
/// A primitive `n`th root of unity means:
///
/// * `omega`^`n` = 1 (making it an `n`th root of unity), and
/// * `omega`^`k` ≠ 1 for all integers 1 ≤ k < n (making it a primitive `n`th root of unity)
///
/// This transform is performed in-place.
///
/// If called on an empty array, returns an empty array.
#[allow(clippy::many_single_char_names)]
pub fn ntt<FF: FiniteField + MulAssign<BFieldElement>>(
    x: &mut [FF],
    omega: BFieldElement,
    log_2_of_n: u32,
) {
    let n = x.len() as u32;

    // `n` must be a power of 2, or be zero
    debug_assert!(
        n == 1 << log_2_of_n || n == 0 && log_2_of_n == 0,
        "2^log2(n) == n || n == 0 && log_2_of_n == 0 must evaluate to true, but n was {n} and log_2_of_n was {log_2_of_n}"
    );

    // `omega` must be a primitive root of unity of order `n`
    debug_assert!(
        omega.mod_pow_u32(n).is_one(),
        "Got {omega} which is not a {n}th root of 1"
    );
    debug_assert!(!omega.mod_pow_u32(n / 2).is_one() || n == 0 || n == 1);

    for k in 0..n {
        let rk = bitreverse(k, log_2_of_n);
        if k < rk {
            x.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_2_of_n {
        let w_m = omega.mod_pow_u32(n / (2 * m));
        let mut k = 0;
        while k < n {
            let mut w = BFieldElement::ONE;
            for j in 0..m {
                let u = x[(k + j) as usize];
                let mut v = x[(k + j + m) as usize];
                v *= w;
                x[(k + j) as usize] = u + v;
                x[(k + j + m) as usize] = u - v;
                w *= w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

/// ## Perform INTT on slices of prime-field elements
///
/// INTT is the inverse NTT, so abstractly,
/// *intt(values, omega, log2(n)) = ntt(values, 1/omega, log2(n)) / n*.
///
/// ```ignore
/// let original_values: Vec<PF> = ...;
/// let mut transformed_values = original_values.clone();
/// ntt::<PF>(&mut values, omega, log_2_n);
/// intt::<PF>(&mut values, omega, log_2_n);
/// assert_eq!(original_values, transformed_values);
/// ```
///
/// This transform is performed in-place.
pub fn intt<FF: FiniteField + MulAssign<BFieldElement>>(
    x: &mut [FF],
    omega: BFieldElement,
    log_2_of_n: u32,
) {
    let n = BFieldElement::new(u64::try_from(x.len()).unwrap());
    let n_inv_or_zero = n.inverse_or_zero();
    ntt::<FF>(x, omega.inverse(), log_2_of_n);
    for elem in x.iter_mut() {
        *elem *= n_inv_or_zero
    }
}

#[inline]
pub fn bitreverse_usize(mut n: usize, l: usize) -> usize {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

pub fn bitreverse_order<FF>(array: &mut [FF]) {
    let mut logn = 0;
    while (1 << logn) < array.len() {
        logn += 1;
    }

    for k in 0..array.len() {
        let rk = bitreverse_usize(k, logn);
        if k < rk {
            array.swap(rk, k);
        }
    }
}

/// Compute the NTT, but leave the array in bitreversed order.
///
/// This method can be expected to outperform regular NTT when
///  - it is followed up by INTT (e.g. for fast multiplication)
///  - the powers_of_omega_bitreversed can be precomputed (which
///    is not the case here).
/// In that case, be sure to use the matching `intt_noswap` and
/// don't forget to unscale by n, e.g. using `unscale`.
pub fn ntt_noswap<FF: FiniteField + MulAssign<BFieldElement>>(x: &mut [FF], omega: BFieldElement) {
    let n: usize = x.len();

    // `n` must be a power of 2
    debug_assert_eq!(n & (n - 1), 0);

    // `omega` must be a primitive root of unity of order `n`
    debug_assert!(
        omega.mod_pow_u32(n as u32).is_one(),
        "Got {omega} which is not a {n}th root of 1"
    );
    debug_assert!(!omega.mod_pow_u32((n / 2).try_into().unwrap()).is_one());

    let mut logn: usize = 0;
    while (1 << logn) < x.len() {
        logn += 1;
    }

    let mut powers_of_omega_bitreversed = vec![BFieldElement::ZERO; n];
    let mut omegai = BFieldElement::ONE;
    for i in 0..n / 2 {
        powers_of_omega_bitreversed[bitreverse_usize(i, logn - 1)] = omegai;
        omegai *= omega;
    }

    let mut m: usize = 1;
    let mut t: usize = n;
    while m < n {
        t >>= 1;

        for (i, zeta) in powers_of_omega_bitreversed.iter().enumerate().take(m) {
            let s = i * t * 2;
            for j in s..(s + t) {
                let u = x[j];
                let mut v = x[j + t];
                v *= *zeta;
                x[j] = u + v;
                x[j + t] = u - v;
            }
        }

        m *= 2;
    }
}

/// Compute the inverse NTT, assuming that the array is presented in
/// bitreversed order. Also, don't unscale by n afterwards.
pub fn intt_noswap<FF: FiniteField + MulAssign<BFieldElement>>(x: &mut [FF], omega: BFieldElement) {
    let n = x.len();
    let omega_inverse = omega.inverse();

    // `n` must be a power of 2
    debug_assert_eq!(n & (n - 1), 0, "array length must be power of 2");

    // `omega` must be a primitive root of unity of order `n`
    debug_assert!(
        omega_inverse.mod_pow_u32(n.try_into().unwrap()).is_one(),
        "Got {omega_inverse} which is not a {n}th root of 1"
    );
    debug_assert!(!omega_inverse
        .mod_pow_u32((n / 2).try_into().unwrap())
        .is_one());

    let mut logn: usize = 0;
    while (1 << logn) < x.len() {
        logn += 1;
    }

    let mut m = 1;
    for _ in 0..logn {
        let w_m = omega_inverse.mod_pow_u32((n / (2 * m)).try_into().unwrap());
        let mut k = 0;
        while k < n {
            let mut w = BFieldElement::ONE;
            for j in 0..m {
                let u = x[k + j];
                let mut v = x[k + j + m];
                v *= w;
                x[k + j] = u + v;
                x[k + j + m] = u - v;
                w *= w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

/// Unscale the array by multiplying every element by the
/// inverse of the array's length. Useful for following up intt.
pub fn unscale(array: &mut [BFieldElement]) {
    let ninv = BFieldElement::new(array.len() as u64).inverse();
    for a in array.iter_mut() {
        *a *= ninv;
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
    use itertools::Itertools;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::math::other::random_elements;
    use crate::math::traits::PrimitiveRootOfUnity;
    use crate::math::x_field_element::EXTENSION_DEGREE;
    use crate::prelude::*;
    use crate::xfe;

    use super::*;

    #[test]
    fn chu_ntt_b_field_prop_test() {
        for log_2_n in 1..10 {
            let n = 1 << log_2_n;
            for _ in 0..10 {
                let mut values = random_elements(n);
                let original_values = values.clone();
                let omega = BFieldElement::primitive_root_of_unity(n as u64).unwrap();
                ntt::<BFieldElement>(&mut values, omega, log_2_n);
                assert_ne!(original_values, values);
                intt::<BFieldElement>(&mut values, omega, log_2_n);
                assert_eq!(original_values, values);

                values[0] = bfe!(BFieldElement::MAX);
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
        for log_2_n in 1..10 {
            let n = 1 << log_2_n;
            for _ in 0..10 {
                let mut values = random_elements(n);
                let original_values = values.clone();
                let omega = XFieldElement::primitive_root_of_unity(n as u64).unwrap();
                ntt::<XFieldElement>(&mut values, omega.unlift().unwrap(), log_2_n);
                assert_ne!(original_values, values);
                intt::<XFieldElement>(&mut values, omega.unlift().unwrap(), log_2_n);
                assert_eq!(original_values, values);

                // Verify that we are not just operating in the B-field
                // statistically this should hold except one out of
                // ~ (2^64)^2 times this test runs
                assert!(
                    !original_values[1].coefficients[1].is_zero()
                        || !original_values[1].coefficients[2].is_zero()
                );

                values[0] = xfe!([BFieldElement::MAX; EXTENSION_DEGREE]);
                let original_values_with_max_element = values.clone();
                ntt::<XFieldElement>(&mut values, omega.unlift().unwrap(), log_2_n);
                assert_ne!(original_values, values);
                intt::<XFieldElement>(&mut values, omega.unlift().unwrap(), log_2_n);
                assert_eq!(original_values_with_max_element, values);
            }
        }
    }

    #[test]
    fn xfield_basic_test_of_chu_ntt() {
        let mut input_output = vec![
            XFieldElement::new_const(BFieldElement::ONE),
            XFieldElement::new_const(BFieldElement::ZERO),
            XFieldElement::new_const(BFieldElement::ZERO),
            XFieldElement::new_const(BFieldElement::ZERO),
        ];
        let original_input = input_output.clone();
        let expected = vec![
            XFieldElement::new_const(BFieldElement::ONE),
            XFieldElement::new_const(BFieldElement::ONE),
            XFieldElement::new_const(BFieldElement::ONE),
            XFieldElement::new_const(BFieldElement::ONE),
        ];
        let omega = XFieldElement::primitive_root_of_unity(4).unwrap();

        println!("input_output = {input_output:?}");
        ntt::<XFieldElement>(&mut input_output, omega.unlift().unwrap(), 2);
        assert_eq!(expected, input_output);
        println!("input_output = {input_output:?}");

        // Verify that INTT(NTT(x)) = x
        intt::<XFieldElement>(&mut input_output, omega.unlift().unwrap(), 2);
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
        let omega = BFieldElement::primitive_root_of_unity(4).unwrap();

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
        let omega = BFieldElement::primitive_root_of_unity(4).unwrap();

        ntt::<BFieldElement>(&mut input_output, omega, 2);
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output, omega, 2);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn ntt_on_empty_input() {
        let mut input_output = vec![];
        let original_input = input_output.clone();
        let omega = BFieldElement::primitive_root_of_unity(0).unwrap();

        ntt::<BFieldElement>(&mut input_output, omega, 0);
        assert_eq!(0, input_output.len());

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output, omega, 0);
        assert_eq!(original_input, input_output);
    }

    #[proptest]
    fn ntt_on_input_of_length_one(bfe: BFieldElement) {
        let mut test_vector = vec![bfe];
        let root_of_unity = BFieldElement::ONE;

        ntt(&mut test_vector, root_of_unity, 0);
        assert_eq!(vec![bfe], test_vector);
    }

    #[proptest(cases = 10)]
    fn ntt_then_intt_is_identity_operation(
        #[strategy((0_usize..20).prop_map(|l| 1 << l))] _vector_length: usize,
        #[strategy(vec(arb(), #_vector_length))] mut input: Vec<BFieldElement>,
    ) {
        let original_input = input.clone();
        let log_2_of_input_length = input.len().ilog2();
        let root_of_unity = BFieldElement::primitive_root_of_unity(input.len() as u64).unwrap();

        ntt::<BFieldElement>(&mut input, root_of_unity, log_2_of_input_length);
        intt::<BFieldElement>(&mut input, root_of_unity, log_2_of_input_length);

        assert_eq!(original_input, input);
    }

    #[test]
    fn b_field_ntt_with_length_32() {
        let mut input_output = bfe_vec![
            1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0,
            0, 0, 0,
        ];
        let original_input = input_output.clone();
        let omega = BFieldElement::primitive_root_of_unity(32).unwrap();
        ntt::<BFieldElement>(&mut input_output, omega, 5);
        // let actual_output = ntt(&mut input_output, &omega, 5);
        println!("actual_output = {input_output:?}");
        let expected = bfe_vec![
            20,
            0,
            0,
            0,
            18446744069146148869_u64,
            0,
            0,
            0,
            4503599627370500_u64,
            0,
            0,
            0,
            18446726477228544005_u64,
            0,
            0,
            0,
            18446744069414584309_u64,
            0,
            0,
            0,
            268435460,
            0,
            0,
            0,
            18442240469787213829_u64,
            0,
            0,
            0,
            17592186040324_u64,
            0,
            0,
            0,
        ];
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output, omega, 5);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn test_compare_ntt_to_eval() {
        for log_size in 1..10 {
            let size = 1 << log_size;
            let mut array: Vec<BFieldElement> = random_elements(size);
            let polynomial = Polynomial::from(&array);

            let omega = BFieldElement::primitive_root_of_unity(size.try_into().unwrap()).unwrap();
            ntt(&mut array, omega, log_size.try_into().unwrap());

            let evals = (0..size)
                .map(|i| omega.mod_pow(i.try_into().unwrap()))
                .map(|p| polynomial.evaluate(p))
                .collect_vec();

            assert_eq!(evals, array);
        }
    }

    #[test]
    fn test_ntt_noswap() {
        for log_size in 1..8 {
            let size = 1 << log_size;
            println!("size: {size}");
            let a: Vec<BFieldElement> = random_elements(size);
            let omega = BFieldElement::primitive_root_of_unity(size.try_into().unwrap()).unwrap();
            let mut a1 = a.clone();
            ntt(&mut a1, omega, log_size);
            let mut a2 = a.clone();
            ntt_noswap(&mut a2, omega);
            bitreverse_order(&mut a2);
            assert_eq!(a1, a2);

            intt(&mut a1, omega, log_size);
            bitreverse_order(&mut a2);
            intt_noswap(&mut a2, omega);
            for a2e in a2.iter_mut() {
                *a2e *= BFieldElement::new(size.try_into().unwrap()).inverse();
            }
            assert_eq!(a1, a2);
        }
    }
}
