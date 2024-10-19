use std::ops::MulAssign;

use num_traits::ConstOne;
use num_traits::ConstZero;

use super::b_field_element::BFieldElement;
use super::traits::FiniteField;
use super::traits::Inverse;
use super::traits::ModPowU32;
use super::traits::PrimitiveRootOfUnity;

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
/// The transform is performed in-place.
/// If called on an empty array, returns an empty array.
///
/// For the inverse, see [iNTT][self::intt].
///
/// # Panics
///
/// Panics if the length of the input slice is
/// - not a power of two
/// - larger than [`u32::MAX`]
pub fn ntt<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let slice_len = u32::try_from(x.len()).expect("slice should be no longer than u32::MAX");

    assert!(slice_len == 0 || slice_len.is_power_of_two());
    let log2_slice_len = slice_len.checked_ilog2().unwrap_or(0);

    // `slice_len` is 0 or a power of two smaller than u32::MAX
    //  => `unwrap()` never panics
    let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
    ntt_unchecked(x, omega, log2_slice_len);
}

/// ## Perform INTT on slices of prime-field elements
///
/// INTT is the inverse [NTT][self::ntt], so abstractly,
/// *intt(values) = ntt(values) / n*.
///
/// This transform is performed in-place.
///
/// # Example
///
/// ```
/// # use twenty_first::prelude::*;
/// # use twenty_first::math::ntt::ntt;
/// # use twenty_first::math::ntt::intt;
/// let original_values = bfe_vec![0, 1, 1, 2, 3, 5, 8, 13];
/// let mut transformed_values = original_values.clone();
/// ntt(&mut transformed_values);
/// intt(&mut transformed_values);
/// assert_eq!(original_values, transformed_values);
/// ```
///
/// # Panics
///
/// Panics if the length of the input slice is
/// - not a power of two
/// - larger than [`u32::MAX`]
pub fn intt<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let slice_len = u32::try_from(x.len()).expect("slice should be no longer than u32::MAX");

    assert!(slice_len == 0 || slice_len.is_power_of_two());
    let log2_slice_len = slice_len.checked_ilog2().unwrap_or(0);

    // `slice_len` is 0 or a power of two smaller than u32::MAX
    //  => `unwrap()` never panics
    let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
    ntt_unchecked(x, omega.inverse(), log2_slice_len);

    let n_inv_or_zero = BFieldElement::from(x.len()).inverse_or_zero();
    for elem in x.iter_mut() {
        *elem *= n_inv_or_zero
    }
}

/// Like [NTT][self::ntt], but with greater control over the root of unity that
/// is to be used.
///
/// Does _not_ check whether
/// - the passed-in root of unity is indeed a primitive root of unity of the
///   appropriate order, or whether
/// - the passed-in log₂ of the slice length matches.
///
/// Use [NTT][self:ntt] if you want a nicer interface.
#[expect(clippy::many_single_char_names)]
#[inline]
fn ntt_unchecked<FF>(x: &mut [FF], omega: BFieldElement, log2_slice_len: u32)
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let slice_len = x.len() as u32;

    for k in 0..slice_len {
        let rk = bitreverse(k, log2_slice_len);
        if k < rk {
            x.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log2_slice_len {
        let w_m = omega.mod_pow_u32(slice_len / (2 * m));
        let mut k = 0;
        while k < slice_len {
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

/// Compute the [NTT][self::ntt], but leave the array in
/// [bitreversed order][self::bitreverse_order].
///
/// This method can be expected to outperform regular NTT when
///  - it is followed up by [INTT][self::intt] (e.g. for fast multiplication)
///  - the `powers_of_omega_bitreversed` can be precomputed (which is not the
///    case here).
///
/// In that case, be sure to use the matching [`intt_noswap`] and don't forget
/// to unscale by `n`, e.g. using [`unscale`].
pub fn ntt_noswap<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let n: usize = x.len();
    debug_assert!(n.is_power_of_two());

    let root_order = n.try_into().unwrap();
    let omega = BFieldElement::primitive_root_of_unity(root_order).unwrap();

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

/// Compute the [inverse NTT][self::intt], assuming that the array is presented
/// in [bitreversed order][self::bitreverse_order]. Also, don't unscale by `n`
/// afterward.
///
/// See also [`ntt_noswap`].
pub fn intt_noswap<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let n = x.len();
    debug_assert!(n.is_power_of_two());

    let root_order = n.try_into().unwrap();
    let omega = BFieldElement::primitive_root_of_unity(root_order).unwrap();
    let omega_inverse = omega.inverse();

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

    use super::*;
    use crate::math::other::random_elements;
    use crate::math::traits::PrimitiveRootOfUnity;
    use crate::math::x_field_element::EXTENSION_DEGREE;
    use crate::prelude::*;
    use crate::xfe;

    #[test]
    fn chu_ntt_b_field_prop_test() {
        for log_2_n in 1..10 {
            let n = 1 << log_2_n;
            for _ in 0..10 {
                let mut values = random_elements(n);
                let original_values = values.clone();
                ntt::<BFieldElement>(&mut values);
                assert_ne!(original_values, values);
                intt::<BFieldElement>(&mut values);
                assert_eq!(original_values, values);

                values[0] = bfe!(BFieldElement::MAX);
                let original_values_with_max_element = values.clone();
                ntt::<BFieldElement>(&mut values);
                assert_ne!(original_values, values);
                intt::<BFieldElement>(&mut values);
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
                ntt::<XFieldElement>(&mut values);
                assert_ne!(original_values, values);
                intt::<XFieldElement>(&mut values);
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
                ntt::<XFieldElement>(&mut values);
                assert_ne!(original_values, values);
                intt::<XFieldElement>(&mut values);
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

        println!("input_output = {input_output:?}");
        ntt::<XFieldElement>(&mut input_output);
        assert_eq!(expected, input_output);
        println!("input_output = {input_output:?}");

        // Verify that INTT(NTT(x)) = x
        intt::<XFieldElement>(&mut input_output);
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

        ntt::<BFieldElement>(&mut input_output);
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output);
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

        ntt::<BFieldElement>(&mut input_output);
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn ntt_on_empty_input() {
        let mut input_output = vec![];
        let original_input = input_output.clone();

        ntt::<BFieldElement>(&mut input_output);
        assert_eq!(0, input_output.len());

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[proptest]
    fn ntt_on_input_of_length_one(bfe: BFieldElement) {
        let mut test_vector = vec![bfe];
        ntt(&mut test_vector);
        assert_eq!(vec![bfe], test_vector);
    }

    #[proptest(cases = 10)]
    fn ntt_then_intt_is_identity_operation(
        #[strategy((0_usize..18).prop_map(|l| 1 << l))] _vector_length: usize,
        #[strategy(vec(arb(), #_vector_length))] mut input: Vec<BFieldElement>,
    ) {
        let original_input = input.clone();
        ntt::<BFieldElement>(&mut input);
        intt::<BFieldElement>(&mut input);
        assert_eq!(original_input, input);
    }

    #[test]
    fn b_field_ntt_with_length_32() {
        let mut input_output = bfe_vec![
            1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0,
            0, 0, 0,
        ];
        let original_input = input_output.clone();
        ntt::<BFieldElement>(&mut input_output);
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
        intt::<BFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn test_compare_ntt_to_eval() {
        for log_size in 1..10 {
            let size = 1 << log_size;
            let mut coefficients = random_elements(size);
            let polynomial = Polynomial::new(coefficients.clone());

            let omega = BFieldElement::primitive_root_of_unity(size.try_into().unwrap()).unwrap();
            ntt(&mut coefficients);

            let evals = (0..size)
                .map(|i| omega.mod_pow(i.try_into().unwrap()))
                .map(|p| polynomial.evaluate_in_same_field(p))
                .collect_vec();

            assert_eq!(evals, coefficients);
        }
    }

    #[test]
    fn test_ntt_noswap() {
        for log_size in 1..8 {
            let size = 1 << log_size;
            println!("size: {size}");
            let a: Vec<BFieldElement> = random_elements(size);
            let mut a1 = a.clone();
            ntt(&mut a1);
            let mut a2 = a.clone();
            ntt_noswap(&mut a2);
            bitreverse_order(&mut a2);
            assert_eq!(a1, a2);

            intt(&mut a1);
            bitreverse_order(&mut a2);
            intt_noswap(&mut a2);
            for a2e in a2.iter_mut() {
                *a2e *= BFieldElement::new(size.try_into().unwrap()).inverse();
            }
            assert_eq!(a1, a2);
        }
    }
}
