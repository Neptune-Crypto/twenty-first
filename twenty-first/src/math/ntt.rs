use std::num::NonZeroUsize;
use std::ops::MulAssign;
use std::sync::OnceLock;

use num_traits::ConstOne;

use super::b_field_element::BFieldElement;
use super::traits::FiniteField;
use super::traits::Inverse;
use super::traits::ModPowU32;
use super::traits::PrimitiveRootOfUnity;

/// The number of different domains over which this library can compute (i)NTT.
///
/// In particular, the maximum slice length for both [NTT][ntt] and [iNTT][intt]
/// supported by this library is 2^31 on 64 bit systems and 2^28 on 32 bit systems.
/// All domains of length some power of 2 smaller than this, plus the empty domain,
/// are supported as well.
#[cfg(target_pointer_width = "32")]
const NUM_DOMAINS: usize = 29; // On 32-bit, up to length 2^28 to avoid isize::MAX overflow.

#[cfg(not(target_pointer_width = "32"))]
const NUM_DOMAINS: usize = 32; // Default for 64-bit and other architectures.

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
    static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    let slice_len = slice_len(x);
    let twiddle_factors = ALL_TWIDDLE_FACTORS[slice_len.checked_ilog2().unwrap_or(0) as usize]
        .get_or_init(|| {
            let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
            twiddle_factors(slice_len, omega)
        });

    ntt_unchecked(x, twiddle_factors);
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
    static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    let slice_len = slice_len(x);
    let twiddle_factors = ALL_TWIDDLE_FACTORS[slice_len.checked_ilog2().unwrap_or(0) as usize]
        .get_or_init(|| {
            let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
            twiddle_factors(slice_len, omega.inverse())
        });

    ntt_unchecked(x, twiddle_factors);
    unscale(x);
}

/// Internal helper function to assert that the slice for [NTT][self::ntt] or
/// [iNTT][self::intt] is of a correct length.
///
/// # Panics
///
/// Panics if the slice length is
/// - neither 0 nor a power of two, or
/// - larger than [`u32::MAX`].
fn slice_len<FF>(x: &[FF]) -> u32 {
    let slice_len = u32::try_from(x.len()).expect("slice should be no longer than u32::MAX");
    assert!(slice_len == 0 || slice_len.is_power_of_two());

    slice_len
}

/// Internal helper function for [NTT][self::ntt] and [iNTT][self::intt].
///
/// Assumes that
/// - the passed-in twiddle factors are correct for the length of the slice,
/// - the length of the slice is a power of two, and
/// - the length of the slice is smaller than [`u32::MAX`].
///
/// If any of the above assumptions are violated, the function may panic or
/// produce incorrect results.
#[expect(clippy::many_single_char_names)]
#[inline]
fn ntt_unchecked<FF>(x: &mut [FF], twiddle_factors: &[Vec<BFieldElement>])
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    // It is possible to pre-compute all swap indices at compile time, but that
    // would incur a big compile time penalty.
    //
    // The type here is quite the mouthful. A short explainer is in order.
    // - `OnceLock` is used to ensure that the swap indices are computed only
    //   once per slice length, and that the computation is thread-safe. This
    //   cache significantly speeds up the computation.
    // - For the remaining `Vec<Option<NonZeroUsize>>`, see the documentation of
    //   `swap_indices`.
    static ALL_SWAP_INDICES: [OnceLock<Vec<Option<NonZeroUsize>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    let slice_len = x.len();
    let log2_slice_len = slice_len.checked_ilog2().unwrap_or(0);
    let swap_indices =
        ALL_SWAP_INDICES[log2_slice_len as usize].get_or_init(|| swap_indices(slice_len));
    debug_assert_eq!(swap_indices.len(), slice_len);

    // This is the most performant version of the code I can produce.
    // Things I've tried:
    // - swap_indices: Vec<(usize, usize)>, where each element in the vector
    //   is a pair of indices to swap. This vector is shorter than x, and the
    //   body of the loop is branch-free (at least on our end) so it seems like
    //   it should be faster, but I couldn't measure any difference.
    // - swap_indices: Vec<usize>, where the element equals its index for those
    //   indices that do not need to be swapped. Since core::slice::swap
    //   guarantees that elements don't get swapped if its two arguments are
    //   equal, the behavior is unchanged and removes the branching in the loop
    //   body, but resulted in a slowdown.
    for (k, maybe_rev_k) in swap_indices.iter().enumerate() {
        if let Some(rev_k) = maybe_rev_k {
            x.swap(k, rev_k.get());
        }
    }

    let slice_len = slice_len as u32;
    let mut m = 1;
    for twiddles in twiddle_factors {
        let mut k = 0;
        while k < slice_len {
            for j in 0..m {
                let idx1 = (k + j) as usize;
                let idx2 = (k + j + m) as usize;
                let u = x[idx1];
                let mut v = x[idx2];
                v *= twiddles[j as usize];
                x[idx1] = u + v;
                x[idx2] = u - v;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

/// Unscale the array by multiplying every element by the
/// inverse of the array's length. Useful for following up intt.
#[inline]
fn unscale<FF>(array: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let n_inv = BFieldElement::from(array.len()).inverse_or_zero();
    for elem in array {
        *elem *= n_inv;
    }
}

/// A list of options, where the `i`-th element is `Some(j)` if and only if
/// `i` and `j` are indices that should be swapped in the NTT.
//
// `Option<NonZeroUsize>` makes use of niche optimization, which means that
// the return value takes the same amount of space as a `Vec<usize>`, but
// allows us to use `None` as a marker for the case where no swap is needed.
//
// Only public for benchmarking purposes.
#[doc(hidden)]
pub fn swap_indices(len: usize) -> Vec<Option<NonZeroUsize>> {
    #[inline(always)]
    const fn bitreverse(mut k: u32, log2_n: u32) -> u32 {
        k = ((k & 0x55555555) << 1) | ((k & 0xaaaaaaaa) >> 1);
        k = ((k & 0x33333333) << 2) | ((k & 0xcccccccc) >> 2);
        k = ((k & 0x0f0f0f0f) << 4) | ((k & 0xf0f0f0f0) >> 4);
        k = ((k & 0x00ff00ff) << 8) | ((k & 0xff00ff00) >> 8);
        k = k.rotate_right(16);
        k >> ((32 - log2_n) & 0x1f)
    }

    // For large enough `len`, the computation benefits from parallelization.
    // However, if NTT is also being called from within a rayon-parallel
    // context, the potential parallelization here can lead to a deadlock.
    // The relevant issue is <https://github.com/rayon-rs/rayon/issues/592>.
    //
    // As a short summary, consider the following scenario.
    // 1. Some task on some rayon thread calls NTT's OnceLock::get_or_init.
    // 2. The initialization task, i.e., execution of swap_indices, is also done
    //    in parallel. Some of that work is stolen by other rayon threads.
    // 3. The task that originally called OnceLock::get_or_init finishes its
    //    work and starts looking for more work.
    // 4. It steals part of the _outer_ parallelization effort, which just so
    //    happens to be a call to an NTT with the same slice length.
    // 5. It calls OnceLock::get_or_init on the _same_ OnceLock.
    // 6. This, implicitly, is re-entrant initialization of the OnceLock, which
    //    is documented as resulting in a deadlock.
    //
    // While parallel initialization would benefit runtime, a deadlock clearly
    // does not. Because it's a reasonable assumption that NTT is being called
    // in a rayon-parallelized context, we avoid parallelization here for now.
    // Potential ways forward are:
    // - use <https://github.com/rayon-rs/rayon/pull/1175> once that is merged
    // - use a parallelization approach that does not perform or allow
    //   work-stealing, like <https://crates.io/crates/chili> (though this
    //   particular crate might not be the best fit – do some research first 🙂)
    let log_2_len = len.checked_ilog2().unwrap_or(0);
    (0..len)
        .map(|k| {
            let rev_k = bitreverse(k as u32, log_2_len);

            // 0 >= bitreverse(0, log_2_len) == 0 => unwrap is fine
            ((k as u32) < rev_k).then(|| NonZeroUsize::new(rev_k as usize).unwrap())
        })
        .collect()
}

/// Internal helper function to (pre-) compute the twiddle factors for use in
/// [NTT][ntt] and [iNTT][intt].
///
/// Assumes that the given root of unity and the slice length match.
//
// The runtime of this function, especially when seen in the larger context,
// could potentially still be improved. Since this function is run at most twice
// per slice length (once for NTT, once for iNTT), any runtime savings are
// amortized pretty quickly. Saving RAM might be more interesting.
//
// One difference to the Longa+Naehrig paper [0] is the return value of
// Vec<Vec<_>> instead of a single Vec<_>.
// Also note that the twiddle factors for smaller domains are a subset of those
// for larger domains. In order to save both space and time, what can be shared,
// should be shared. I think the engineering work to get this working with the
// current OnceLock-based lazy-initialization is non-trivial, considering that
// OnceLocks must not be re-entrantly initialized. I could be wrong and it's
// actually easy.
//
// [0] <https://eprint.iacr.org/2016/504.pdf>
//
// Only public for benchmarking purposes.
#[doc(hidden)]
pub fn twiddle_factors(slice_len: u32, root_of_unity: BFieldElement) -> Vec<Vec<BFieldElement>> {
    // For an explanation of why this is not parallelized, see `swap_indices`.
    (0..slice_len.checked_ilog2().unwrap_or(0))
        .map(|i| {
            let m = 1 << i;
            let exponent = slice_len / (2 * m);
            let w_m = root_of_unity.mod_pow_u32(exponent);
            let mut w_powers = vec![BFieldElement::ONE; m as usize];
            for j in 1..m as usize {
                w_powers[j] = w_powers[j - 1] * w_m;
            }

            w_powers
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::ConstZero;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    use super::*;
    use crate::math::other::random_elements;
    use crate::math::traits::PrimitiveRootOfUnity;
    use crate::math::x_field_element::EXTENSION_DEGREE;
    use crate::prelude::*;
    use crate::xfe;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[proptest]
    fn ntt_on_input_of_length_one(bfe: BFieldElement) {
        let mut test_vector = vec![bfe];
        ntt(&mut test_vector);
        assert_eq!(vec![bfe], test_vector);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn swap_indices_can_be_computed() {
        // exponential growth is powerful; cap the number of domains
        for log_size in 0..NUM_DOMAINS - 2 {
            swap_indices(1 << log_size);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[test]
    fn twiddle_factors_can_be_computed() {
        // exponential growth is powerful; cap the number of domains
        for log_size in 0..NUM_DOMAINS - 5 {
            let size = 1 << log_size;
            let root = BFieldElement::primitive_root_of_unity(size.into()).unwrap();
            twiddle_factors(size, root);
        }
    }
}
