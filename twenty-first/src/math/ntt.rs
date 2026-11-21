use std::mem::MaybeUninit;
use std::ops::Mul;
use std::ops::MulAssign;
use std::sync::OnceLock;

use num_traits::ConstOne;
use rayon::prelude::*;

use super::b_field_element::BFieldElement;
use super::traits::FiniteField;
use super::traits::Inverse;
use super::traits::ModPowU32;
use super::traits::PrimitiveRootOfUnity;

/// The number of different domains over which this library can compute (i)NTT.
///
/// In particular, the maximum slice length for both [NTT][ntt] and [iNTT][intt]
/// supported by this library is 2^31 on 64-bit systems and 2^28 on 32-bit
/// systems. All domains of length some power of 2 smaller than this, plus the
/// empty domain, are supported as well.
const NUM_DOMAINS: usize = {
    #[cfg(target_pointer_width = "16")]
    compile_error!("pointer width 16 is not supported");

    #[cfg(target_pointer_width = "32")]
    {
        29 // avoid isize::MAX overflow.
    }

    #[cfg(target_pointer_width = "64")]
    {
        32 // NTT currently relies on `usize` to `u32` `as`-casting
    }
};

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
    let twiddle_factors = forward_twiddle_factors(slice_len(x));
    ntt_unchecked(x, twiddle_factors);
}

/// Parallel version of [NTT][self::ntt].
///
/// Use this for a single, large NTT. If many NTTs are to be computed, for
/// example one per column of a table, it is generally more efficient to
/// compute them in parallel using the serial [`ntt`] for each.
///
/// # Panics
///
/// See [`ntt`].
pub fn par_ntt<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement> + Send + Sync,
{
    let twiddle_factors = forward_twiddle_factors(slice_len(x));
    par_ntt_unchecked(x, twiddle_factors);
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
    let twiddle_factors = inverse_twiddle_factors(slice_len(x));
    ntt_unchecked(x, twiddle_factors);
    unscale(x);
}

/// Parallel version of [iNTT][self::intt]. See also [`par_ntt`].
///
/// # Panics
///
/// See [`intt`].
pub fn par_intt<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement> + Send + Sync,
{
    let twiddle_factors = inverse_twiddle_factors(slice_len(x));
    par_ntt_unchecked(x, twiddle_factors);
    par_unscale(x);
}

/// The [NTT][self::ntt] of `values`, written to `output`, which may be
/// uninitialized. On return, every element of `output` is initialized.
///
/// For large inputs, this is faster than copying the values and transforming
/// the copy in place, since the passes over memory dominate the runtime of
/// large transforms.
///
/// # Panics
///
/// Panics if the lengths of `values` and `output` differ, and under the
/// conditions [`ntt`] panics.
pub fn ntt_into<FF>(values: &[FF], output: &mut [MaybeUninit<FF>])
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let twiddle_factors = forward_twiddle_factors(slice_len(values));
    ntt_into_unchecked(values, output, twiddle_factors, BFieldElement::ONE);
}

/// The [inverse NTT][self::intt] of `values`, written to `output`, which may
/// be uninitialized. On return, every element of `output` is initialized.
/// See also [`ntt_into`].
///
/// # Panics
///
/// Panics if the lengths of `values` and `output` differ, and under the
/// conditions [`intt`] panics.
pub fn intt_into<FF>(values: &[FF], output: &mut [MaybeUninit<FF>])
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let twiddle_factors = inverse_twiddle_factors(slice_len(values));
    let len_inverse = BFieldElement::from(values.len()).inverse_or_zero();
    ntt_into_unchecked(values, output, twiddle_factors, len_inverse);
}

/// The binary logarithm of the block length of [`ntt_into_unchecked`]'s
/// two-pass algorithm. The blocks that are gathered together, i.e., 8 blocks
/// of 32 KiB for base field elements or 4 blocks of 96 KiB for extension
/// field elements, must fit the L2 cache.
#[cfg(target_arch = "x86_64")]
const LOG_2_GATHERED_BLOCK_LEN: u32 = 12;

/// Internal helper function for [`ntt_into`] and [`intt_into`]. Makes the
/// same assumptions as [`ntt_unchecked`]. Every element of the result is
/// multiplied by `scale`.
fn ntt_into_unchecked<FF>(
    values: &[FF],
    output: &mut [MaybeUninit<FF>],
    twiddle_factors: TwiddleFactors,
    scale: BFieldElement,
) where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    assert_eq!(values.len(), output.len());

    #[cfg(target_arch = "x86_64")]
    if ntt_into_in_two_passes(
        values,
        output,
        twiddle_factors,
        scale,
        LOG_2_GATHERED_BLOCK_LEN,
    ) {
        return;
    }

    for (target, &value) in output.iter_mut().zip(values) {
        target.write(value);
    }
    // SAFETY: Every element was just initialized, and `MaybeUninit<FF>` has
    // the same layout as `FF`.
    let output =
        unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<FF>(), output.len()) };
    ntt_unchecked(output, twiddle_factors);
    if scale != BFieldElement::ONE {
        for element in output {
            *element *= scale;
        }
    }
}

/// Try to perform [`ntt_into_unchecked`] in two passes over the output and
/// one pass over the input. Returns `false`, without having done anything, if
/// that is not possible: the SIMD kernels must be usable, and there must be
/// more than one block of `2^log_2_block_len` elements.
///
/// The structure is the same as that of [`avx512::CosetEvaluation`], except
/// that elements are not grouped into vector elements:
///
/// 1. The inputs of a few blocks are gathered from the values, in
///    bit-reversed order, into scratch space. There, all layers whose
///    butterflies stay within a block are applied. Then, element `i` of the
///    block that holds the values with indices `≡ j` modulo the number of
///    blocks is multiplied by `ω^(i·j)`, where `ω` is the primitive root of
///    unity the twiddle factors are made of, and the block is written to
///    the output.
/// 2. After that, the remaining layers are transforms of the blocks as a
///    whole, i.e., the same twiddle factor applies to all elements of a
///    block. See [`avx512::transform_across_blocks`].
#[cfg(target_arch = "x86_64")]
fn ntt_into_in_two_passes<FF>(
    values: &[FF],
    output: &mut [MaybeUninit<FF>],
    twiddle_factors: TwiddleFactors,
    scale: BFieldElement,
    log_2_block_len: u32,
) -> bool
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    /// `avx512::scale_by_powers` processes this many elements at once.
    const MIN_LOG_2_BLOCK_LEN: u32 = 5;

    let Some(limbs) = simd_limbs::<FF>() else {
        return false;
    };
    let Some(log_2_len) = values.len().checked_ilog2() else {
        return false;
    };
    let log_2_block_len = log_2_block_len
        .max(log_2_len.saturating_sub(MAX_LOG_2_NUM_VECTOR_BLOCKS))
        .max(MIN_LOG_2_BLOCK_LEN);
    let Some(log_2_num_blocks) = log_2_len.checked_sub(log_2_block_len).filter(|&n| n > 0) else {
        return false;
    };
    let block_len = 1_usize << log_2_block_len;
    let num_blocks = 1_usize << log_2_num_blocks;
    let lines_per_block = block_len * limbs / avx512::LANES;
    let group_len = if limbs == 1 { 8 } else { 4 }.min(num_blocks);
    let omega = twiddle_factors.omega();

    let scratch_len = (group_len * lines_per_block).max(num_blocks);
    with_scratch_lines(scratch_len, |scratch| {
        // SAFETY: A line is 8 base field elements, and `FF` is laid out as
        // `limbs` of them (see `simd_limbs`), is `Copy`, and has no invalid
        // bit patterns for initialized limbs. All elements that are read
        // below are written first.
        let blocks = unsafe {
            std::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<FF>(), group_len * block_len)
        };

        // Pass 1
        for first_column in (0..num_blocks).step_by(group_len) {
            for row in 0..block_len {
                let position = bit_reverse(row, log_2_block_len);
                let first_index = row * num_blocks + first_column;

                // The values are read at a large stride, which hardware
                // prefetchers do not follow.
                let ahead = first_index + avx512::PREFETCH_DISTANCE * num_blocks;
                if let Some(ahead) = values.get(ahead..ahead + group_len) {
                    avx512::prefetch(ahead);
                }
                for g in 0..group_len {
                    blocks[g * block_len + position] = values[first_index + g];
                }
            }
            for (g, block) in blocks.chunks_exact_mut(block_len).enumerate() {
                apply_layers(block, twiddle_factors, 0, log_2_block_len);

                let column = first_column + g;
                let block_index = bit_reverse(column, log_2_num_blocks);
                let target = output[block_index * block_len..].as_mut_ptr();
                // SAFETY: AVX-512 is available, see `simd_limbs`. The block
                // has a multiple of 32 elements, starts at a cache line of
                // the scratch space, and is a whole number of cache lines
                // long. The target is one block of the output.
                unsafe {
                    if column != 0 || scale != BFieldElement::ONE {
                        let base = omega.mod_pow(column as u64);
                        let block = block.as_mut_ptr().cast();
                        avx512::scale_by_powers(block, limbs, block_len, scale, base);
                    }
                    avx512::copy_lines(block.as_ptr().cast(), target.cast(), lines_per_block);
                }
            }
        }

        // Pass 2
        // SAFETY: AVX-512 is available. Every block of the exclusively
        // borrowed output was written to above. The scratch space has at
        // least as many lines as there are blocks.
        unsafe {
            avx512::transform_across_blocks(
                output.as_mut_ptr().cast(),
                lines_per_block,
                log_2_num_blocks,
                0..lines_per_block,
                twiddle_factors,
                scratch,
            )
        };
    });

    true
}

/// The twiddle factors for a transform of some length, layer by layer.
///
/// The twiddle factors of layer `i` are the `2^i` first powers of the
/// primitive root of unity of order `2^(i+1)`. In particular, they do not
/// depend on the length of the transform, so that transforms of all lengths
/// share them. A layer's twiddle factors are computed when they are first
/// used. This matters because the top layers of long transforms are big,
/// take a while to compute, and are not needed by all algorithms.
#[derive(Debug, Clone, Copy)]
struct TwiddleFactors {
    layers: &'static [OnceLock<Vec<BFieldElement>>; NUM_DOMAINS],
    num_layers: usize,
    is_inverse: bool,
}

impl TwiddleFactors {
    /// The number of layers of the transform.
    fn len(self) -> usize {
        self.num_layers
    }

    /// The primitive root of unity whose order is `2^log_2_order`, or its
    /// inverse for the twiddle factors of the inverse transform.
    fn root_of_unity(self, log_2_order: u32) -> BFieldElement {
        let root = BFieldElement::primitive_root_of_unity(1 << log_2_order).unwrap();
        if self.is_inverse {
            root.inverse()
        } else {
            root
        }
    }

    /// The primitive root of unity whose order is the length of the
    /// transform; see also [`Self::root_of_unity`].
    #[cfg(any(test, target_arch = "x86_64"))]
    fn omega(self) -> BFieldElement {
        self.root_of_unity(self.num_layers as u32)
    }
}

impl std::ops::Index<usize> for TwiddleFactors {
    type Output = Vec<BFieldElement>;

    fn index(&self, layer: usize) -> &Self::Output {
        assert!(layer < self.num_layers);

        // For large layers, this computation could benefit from
        // parallelization. See `twiddle_factors` for why it is sequential.
        self.layers[layer].get_or_init(|| {
            let root = self.root_of_unity(layer as u32 + 1);
            let mut powers = vec![BFieldElement::ONE; 1 << layer];
            for j in 1..powers.len() {
                powers[j] = powers[j - 1] * root;
            }
            powers
        })
    }
}

/// The (cached) twiddle factors for the forward NTT of the given length.
fn forward_twiddle_factors(slice_len: u32) -> TwiddleFactors {
    static LAYERS: [OnceLock<Vec<BFieldElement>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    TwiddleFactors {
        layers: &LAYERS,
        num_layers: slice_len.checked_ilog2().unwrap_or(0) as usize,
        is_inverse: false,
    }
}

/// The (cached) twiddle factors for the inverse NTT of the given length.
fn inverse_twiddle_factors(slice_len: u32) -> TwiddleFactors {
    static LAYERS: [OnceLock<Vec<BFieldElement>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    TwiddleFactors {
        layers: &LAYERS,
        num_layers: slice_len.checked_ilog2().unwrap_or(0) as usize,
        is_inverse: true,
    }
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

/// The binary logarithm of the number of elements per block for the
/// cache-blocked part of the NTT. A block of 2^14 base field elements
/// occupies 128 KiB, a block of extension field elements 384 KiB. Either
/// fits the L2 cache of common x86-64 CPUs, also when a second thread on
/// the same core is working on a block of its own, which is the situation
/// when many transforms run in parallel. Larger blocks fit only in the
/// base field case, and only for one thread per core; the more numerous
/// cross-block layers of smaller blocks are still applied in a single pass
/// over memory.
const LOG_2_BLOCK_LEN: u32 = 14;

/// The binary logarithm of the tile side length used by the
/// [bit-reversal permutation][bit_reverse_permutation]. A tile row of 32
/// elements is 256 bytes for base field elements and 768 bytes for extension
/// field elements: long enough for the DRAM to serve it efficiently, while a
/// pair of tiles still fits the L1 cache.
const LOG_2_TILE_LEN: u32 = 5;

/// The number of adjacent “columns” processed together in the high
/// (cross-block) layers of the NTT. See [`apply_cross_block_layers`].
const CROSS_BLOCK_COLUMN_WIDTH: usize = 64;

/// Below this (binary logarithm of the) length, the parallel NTT falls back
/// to the serial one: the work is too little to be worth distributing. See
/// also [`par_min_log_2_len`].
const PAR_MIN_LOG_2_LEN: u32 = 12;

/// The binary logarithm of the minimum number of elements per thread for
/// the parallel NTT to pay off. Waking a sleeping thread and handing it
/// work costs on the order of a microsecond, which a few hundred
/// butterflies amortize.
const PAR_MIN_LOG_2_LEN_PER_THREAD: u32 = 9;

/// The (binary logarithm of the) length from which on the parallel NTT is
/// faster than the serial one: at least 2^[`PAR_MIN_LOG_2_LEN`], and at least
/// 2^[`PAR_MIN_LOG_2_LEN_PER_THREAD`] elements per thread. With few threads,
/// only the first bound matters; with many, the fixed cost of waking them
/// all dominates transforms that a single thread finishes in a millisecond.
pub(crate) fn par_min_log_2_len() -> u32 {
    let num_threads = rayon::current_num_threads().max(1);
    let log_2_num_threads = num_threads.next_power_of_two().ilog2();
    PAR_MIN_LOG_2_LEN.max(PAR_MIN_LOG_2_LEN_PER_THREAD + log_2_num_threads)
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
///
/// The transform is a decimation-in-time Cooley-Tukey NTT. It is organized
/// to minimize the number of passes over memory, which is the bottleneck for
/// large inputs, especially when many NTTs run in parallel and compete for
/// memory bandwidth:
/// 1. The bit-reversal permutation is performed tile by tile, such that every
///    cache line that is touched is used in its entirety.
/// 2. All layers whose butterflies stay within a block of
///    2^[`LOG_2_BLOCK_LEN`] elements are applied block by block, i.e., in
///    one pass over memory, with each block staying in cache.
/// 3. All remaining layers are applied in one more pass over memory. See
///    [`apply_cross_block_layers`] for details.
///
/// Additionally, two consecutive layers are fused into radix-4 butterflies
/// wherever possible.
///
/// For a slightly different perspective on the structure of this function,
/// consider that step 3 is the NTT of the “rows” of the input interpreted as
/// a matrix, and step 2 is the NTT of the “columns”.
#[inline]
fn ntt_unchecked<FF>(x: &mut [FF], twiddle_factors: TwiddleFactors)
where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let Some(log_2_len) = x.len().checked_ilog2() else {
        // if the slice is empty, there's nothing to do
        return;
    };
    debug_assert_eq!(log_2_len as usize, twiddle_factors.len());

    bit_reverse_permutation(x);

    let log_2_block_len = log_2_len.min(LOG_2_BLOCK_LEN);
    for block in x.chunks_exact_mut(1 << log_2_block_len) {
        apply_layers(block, twiddle_factors, 0, log_2_block_len);
    }
    if log_2_block_len < log_2_len {
        apply_cross_block_layers(x, twiddle_factors, log_2_block_len, log_2_len);
    }
}

/// Parallel version of [`ntt_unchecked`]. Same assumptions.
///
/// All three phases of [`ntt_unchecked`] consist of independent units of
/// work – tile pairs, blocks, and column groups, respectively – that are
/// distributed across threads. The block length is reduced for shorter
/// inputs so that there are enough blocks to keep all threads busy.
fn par_ntt_unchecked<FF>(x: &mut [FF], twiddle_factors: TwiddleFactors)
where
    FF: FiniteField + MulAssign<BFieldElement> + Send + Sync,
{
    let Some(log_2_len) = x.len().checked_ilog2() else {
        return;
    };
    if log_2_len < par_min_log_2_len() {
        return ntt_unchecked(x, twiddle_factors);
    }
    debug_assert_eq!(log_2_len as usize, twiddle_factors.len());

    par_bit_reverse_permutation(x);

    let log_2_num_threads = rayon::current_num_threads()
        .max(1)
        .next_power_of_two()
        .ilog2();
    let log_2_block_len = log_2_len
        .min(LOG_2_BLOCK_LEN)
        .min(log_2_len.saturating_sub(log_2_num_threads))
        .max(2);
    x.par_chunks_exact_mut(1 << log_2_block_len)
        .for_each(|block| apply_layers(block, twiddle_factors, 0, log_2_block_len));
    if log_2_block_len < log_2_len {
        par_apply_cross_block_layers(x, twiddle_factors, log_2_block_len, log_2_len);
    }
}

/// A pointer to a slice that can be shared between threads, in order to
/// mutate disjoint but interleaved index sets of the slice in parallel.
/// Every access is `unsafe`; the caller is responsible for ensuring that no
/// two threads access the same index concurrently.
#[derive(Debug, Clone, Copy)]
struct SharedSliceMut<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: The pointer is only ever dereferenced through `unsafe` methods
// whose contracts require exclusive access to the accessed index.
unsafe impl<T: Send> Send for SharedSliceMut<T> {}
unsafe impl<T: Send> Sync for SharedSliceMut<T> {}

impl<T: Copy> SharedSliceMut<T> {
    fn new(x: &mut [T]) -> Self {
        Self {
            ptr: x.as_mut_ptr(),
            len: x.len(),
        }
    }

    /// The pointer to the slice's first element. Accessing memory through it
    /// is subject to the same contract as the other methods.
    #[cfg(target_arch = "x86_64")]
    fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// # Safety
    ///
    /// No other thread may access index `i` concurrently.
    #[inline(always)]
    unsafe fn read(&self, i: usize) -> T {
        debug_assert!(i < self.len);
        unsafe { self.ptr.add(i).read() }
    }

    /// # Safety
    ///
    /// No other thread may access index `i` concurrently.
    #[inline(always)]
    unsafe fn write(&self, i: usize, value: T) {
        debug_assert!(i < self.len);
        unsafe { self.ptr.add(i).write(value) }
    }

    /// # Safety
    ///
    /// No other thread may access indices `i` or `j` concurrently.
    #[inline(always)]
    unsafe fn swap(&self, i: usize, j: usize) {
        debug_assert!(i < self.len);
        debug_assert!(j < self.len);
        unsafe { std::ptr::swap(self.ptr.add(i), self.ptr.add(j)) }
    }
}

/// Reverse the lowest `num_bits` bits of `i`.
#[inline(always)]
const fn bit_reverse(i: usize, num_bits: u32) -> usize {
    if num_bits == 0 {
        return 0;
    }
    (i as u32).reverse_bits() as usize >> (32 - num_bits)
}

/// Permute the slice such that the element at index `i` ends up at the index
/// obtained by reversing the bits of `i`.
///
/// For slices that are large compared to the cache line size, the
/// permutation is done tile by tile: interpreting an index as the
/// concatenation of bit strings `high || middle || low`, its bit-reversal is
/// `rev(low) || rev(middle) || rev(high)`. For fixed `middle`, all elements
/// of the tile `{high || middle || low}` are moved into the tile
/// `{· || rev(middle) || ·}`. Both tiles consist of contiguous runs of
/// elements, so all touched cache lines are used in their entirety. The
/// naïve approach touches a new cache line for (almost) every element, which
/// causes an order of magnitude more memory traffic.
///
/// # Panics
///
/// Panics if the slice length is not a power of 2.
//
// Only public for benchmarking purposes.
#[doc(hidden)]
pub fn bit_reverse_permutation<T: Copy>(x: &mut [T]) {
    let Some(log_2_len) = x.len().checked_ilog2() else {
        return;
    };
    assert!(x.len().is_power_of_two());

    if log_2_len <= 2 * LOG_2_TILE_LEN {
        return bit_reverse_permutation_naive(x, log_2_len);
    }

    let log_2_num_middle = log_2_len - 2 * LOG_2_TILE_LEN;
    let shared_x = SharedSliceMut::new(x);
    for middle in 0..1_usize << log_2_num_middle {
        // SAFETY: Different `middle`s address disjoint pairs of tiles, and
        // this loop is sequential anyway.
        unsafe { bit_reverse_tile_pair(shared_x, middle, log_2_num_middle) };
    }
}

/// Parallel version of [`bit_reverse_permutation`].
fn par_bit_reverse_permutation<T: Copy + Send + Sync>(x: &mut [T]) {
    let Some(log_2_len) = x.len().checked_ilog2() else {
        return;
    };
    assert!(x.len().is_power_of_two());

    if log_2_len <= 2 * LOG_2_TILE_LEN {
        return bit_reverse_permutation_naive(x, log_2_len);
    }

    let log_2_num_middle = log_2_len - 2 * LOG_2_TILE_LEN;
    let shared_x = SharedSliceMut::new(x);
    (0..1_usize << log_2_num_middle)
        .into_par_iter()
        .for_each(|middle| {
            // SAFETY: Different `middle`s address disjoint pairs of tiles: the
            // pair for `middle` is {middle, rev(middle)}, `rev` is an
            // involution, and the pair is only handled from its smaller
            // element. Hence, no two threads access the same index.
            unsafe { bit_reverse_tile_pair(shared_x, middle, log_2_num_middle) };
        });
}

fn bit_reverse_permutation_naive<T>(x: &mut [T], log_2_len: u32) {
    for i in 0..x.len() {
        let j = bit_reverse(i, log_2_len);
        if i < j {
            x.swap(i, j);
        }
    }
}

/// Move the elements of the tile identified by `middle` into the tile
/// identified by `rev(middle)`, and vice versa. See
/// [`bit_reverse_permutation`] for details. Does nothing if
/// `rev(middle) < middle`, since that pair of tiles is handled when called
/// with `rev(middle)`.
///
/// # Safety
///
/// No other thread may concurrently access the tiles identified by `middle`
/// and `rev(middle)`.
#[inline]
unsafe fn bit_reverse_tile_pair<T: Copy>(
    x: SharedSliceMut<T>,
    middle: usize,
    log_2_num_middle: u32,
) {
    let reversed_middle = bit_reverse(middle, log_2_num_middle);
    if reversed_middle < middle {
        return;
    }
    let is_self_paired_tile = reversed_middle == middle;
    let index = |high: usize, mid: usize, low: usize| {
        (high << (log_2_num_middle + LOG_2_TILE_LEN)) | (mid << LOG_2_TILE_LEN) | low
    };
    for high in 0..1_usize << LOG_2_TILE_LEN {
        let reversed_high = bit_reverse(high, LOG_2_TILE_LEN);
        for low in 0..1_usize << LOG_2_TILE_LEN {
            let i = index(high, middle, low);
            let j = index(
                bit_reverse(low, LOG_2_TILE_LEN),
                reversed_middle,
                reversed_high,
            );
            if !is_self_paired_tile || i < j {
                // SAFETY: Both `i` and `j` lie in the two tiles this call has
                // exclusive access to, and they are smaller than the slice
                // length since all index components are within their ranges.
                unsafe { x.swap(i, j) };
            }
        }
    }
}

/// Apply the decimation-in-time layers `first_layer..last_layer` to the
/// (bit-reversed) slice, where the butterflies of layer `i` have distance
/// 2^i. Requires the slice's length to be a multiple of 2^`last_layer`.
///
/// Consecutive layers are fused into radix-4 butterflies, which halves the
/// number of passes over the slice.
#[inline]
fn apply_layers<FF>(
    x: &mut [FF],
    twiddle_factors: TwiddleFactors,
    first_layer: u32,
    last_layer: u32,
) where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let mut layer = first_layer;
    #[cfg(target_arch = "x86_64")]
    let simd_limbs = simd_limbs::<FF>();

    // The twiddle factors of the first two layers are 1, 1, and a primitive
    // 4th root of unity: only one multiplication per radix-4 butterfly.
    if layer == 0 && last_layer >= 2 {
        let fourth_root_of_unity = twiddle_factors[1][1];
        for butterfly in x.as_chunks_mut::<4>().0 {
            let [t0, t1, t2, t3] = *butterfly;
            let y0 = t0 + t1;
            let y1 = t0 - t1;
            let y2 = t2 + t3;
            let mut y3 = t2 - t3;
            y3 *= fourth_root_of_unity;
            butterfly[0] = y0 + y2;
            butterfly[1] = y1 + y3;
            butterfly[2] = y0 - y2;
            butterfly[3] = y1 - y3;
        }
        layer = 2;
    }

    while layer + 1 < last_layer {
        let m = 1_usize << layer;
        let twiddles_1 = &twiddle_factors[layer as usize];
        let twiddles_2 = &twiddle_factors[layer as usize + 1];
        #[cfg(target_arch = "x86_64")]
        if let Some(limbs) = simd_limbs
            && m >= avx512::LANES
        {
            for k in (0..x.len()).step_by(4 * m) {
                // SAFETY: AVX-512 was detected (see `simd_limbs`), `FF` is
                // laid out as `limbs` base field elements (see
                // `FiniteField::NUM_BFE_LIMBS`), and all accessed indices
                // are below `x.len()`, which is a multiple of `4m`.
                unsafe {
                    avx512::radix_4_butterflies(
                        x.as_mut_ptr().cast(),
                        limbs,
                        k,
                        m,
                        &twiddles_1[..m],
                        &twiddles_2[..2 * m],
                    )
                };
            }
            layer += 2;
            continue;
        }
        for butterflies in x.chunks_exact_mut(4 * m) {
            let (ab, cd) = butterflies.split_at_mut(2 * m);
            let (a, b) = ab.split_at_mut(m);
            let (c, d) = cd.split_at_mut(m);
            for j in 0..m {
                let t0 = a[j];
                let mut t1 = b[j];
                let t2 = c[j];
                let mut t3 = d[j];
                t1 *= twiddles_1[j];
                t3 *= twiddles_1[j];
                let y0 = t0 + t1;
                let y1 = t0 - t1;
                let mut y2 = t2 + t3;
                let mut y3 = t2 - t3;
                y2 *= twiddles_2[j];
                y3 *= twiddles_2[j + m];
                a[j] = y0 + y2;
                b[j] = y1 + y3;
                c[j] = y0 - y2;
                d[j] = y1 - y3;
            }
        }
        layer += 2;
    }

    if layer < last_layer {
        let m = 1_usize << layer;
        let twiddles = &twiddle_factors[layer as usize];
        #[cfg(target_arch = "x86_64")]
        if let Some(limbs) = simd_limbs
            && m >= avx512::LANES
        {
            for k in (0..x.len()).step_by(2 * m) {
                // SAFETY: See the radix-4 case above.
                unsafe {
                    avx512::radix_2_butterflies(x.as_mut_ptr().cast(), limbs, k, m, &twiddles[..m])
                };
            }
            return;
        }
        for butterflies in x.chunks_exact_mut(2 * m) {
            let (a, b) = butterflies.split_at_mut(m);
            for j in 0..m {
                let u = a[j];
                let mut v = b[j];
                v *= twiddles[j];
                a[j] = u + v;
                b[j] = u - v;
            }
        }
    }
}

/// The number of base field limbs of `FF` if the SIMD kernels can be used
/// for it on this CPU, else `None`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn simd_limbs<FF: FiniteField>() -> Option<usize> {
    match FF::NUM_BFE_LIMBS {
        Some(limbs @ (1 | 3)) if avx512::is_available() => Some(limbs),
        _ => None,
    }
}

/// Apply the decimation-in-time layers `first_layer..last_layer`, where
/// `2^first_layer` is the block length used in [`ntt_unchecked`].
///
/// For these layers, the butterfly with index `j` within a butterfly group
/// only ever combines elements whose index is congruent to `j` modulo the
/// block length. Hence, the elements `{x[j + t · block_len] | t}` form an
/// independent sub-problem for each `j`. A few adjacent such sub-problems
/// fit into cache at once, so all layers are applied to them before moving
/// on to the next few. This applies all remaining layers in one pass over
/// memory, instead of one pass per (pair of) layer(s).
#[inline]
fn apply_cross_block_layers<FF>(
    x: &mut [FF],
    twiddle_factors: TwiddleFactors,
    first_layer: u32,
    last_layer: u32,
) where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let block_len = 1_usize << first_layer;
    let column_width = CROSS_BLOCK_COLUMN_WIDTH.min(block_len);
    let shared_x = SharedSliceMut::new(x);
    for column_start in (0..block_len).step_by(column_width) {
        let columns = column_start..column_start + column_width;
        // SAFETY: Different column groups address disjoint sets of indices,
        // and this loop is sequential anyway.
        unsafe {
            apply_cross_block_layers_to_columns(
                shared_x,
                twiddle_factors,
                first_layer,
                last_layer,
                columns,
            )
        };
    }
}

/// Parallel version of [`apply_cross_block_layers`].
fn par_apply_cross_block_layers<FF>(
    x: &mut [FF],
    twiddle_factors: TwiddleFactors,
    first_layer: u32,
    last_layer: u32,
) where
    FF: FiniteField + MulAssign<BFieldElement> + Send + Sync,
{
    let block_len = 1_usize << first_layer;
    let column_width = par_cross_block_column_width::<FF>(x.len(), block_len);
    let shared_x = SharedSliceMut::new(x);
    (0..block_len)
        .into_par_iter()
        .step_by(column_width)
        .for_each(|column_start| {
            let columns = column_start..column_start + column_width;
            // SAFETY: Different column groups address disjoint sets of
            // indices: the group starting at `column_start` only accesses
            // indices that are congruent to some `j` in `columns` modulo the
            // block length. Hence, no two threads access the same index.
            unsafe {
                apply_cross_block_layers_to_columns(
                    shared_x,
                    twiddle_factors,
                    first_layer,
                    last_layer,
                    columns,
                )
            };
        });
}

/// The number of adjacent columns processed together by
/// [`par_apply_cross_block_layers`].
///
/// A task's working set is its columns across all blocks; it should fit the
/// L2 cache alongside a second thread's, since all layers are applied to it
/// before moving on. Narrower columns also make more tasks, which balance
/// better across many threads. Columns are at least `avx512::LANES` wide
/// for the SIMD kernels and at most [`CROSS_BLOCK_COLUMN_WIDTH`].
fn par_cross_block_column_width<FF>(len: usize, block_len: usize) -> usize {
    const WORKING_SET_BUDGET: usize = 96 << 10;
    const MIN_TASKS_PER_THREAD: usize = 2;

    let num_blocks = len / block_len;
    let fits_cache = WORKING_SET_BUDGET / (num_blocks * size_of::<FF>()).max(1);
    let enough_tasks = block_len / (MIN_TASKS_PER_THREAD * rayon::current_num_threads().max(1));
    let width = fits_cache
        .min(enough_tasks)
        .clamp(8, CROSS_BLOCK_COLUMN_WIDTH);
    let width = 1 << width.ilog2(); // round down to a power of two
    width.min(block_len)
}

/// Apply the cross-block layers to the sub-problems identified by `columns`.
/// See [`apply_cross_block_layers`].
///
/// # Safety
///
/// No other thread may concurrently access any index congruent to any
/// element of `columns` modulo `2^first_layer`. The slice length must be a
/// multiple of `2^last_layer`, and `columns` must be a sub-range of
/// `0..2^first_layer`.
#[inline]
unsafe fn apply_cross_block_layers_to_columns<FF>(
    x: SharedSliceMut<FF>,
    twiddle_factors: TwiddleFactors,
    first_layer: u32,
    last_layer: u32,
    columns: std::ops::Range<usize>,
) where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let block_len = 1_usize << first_layer;
    let len = x.len;
    let mut layer = first_layer;
    #[cfg(target_arch = "x86_64")]
    let simd_limbs = simd_limbs::<FF>().filter(|_| columns.len().is_multiple_of(avx512::LANES));

    // SAFETY (for all accesses below): all accessed indices are of the form
    // k + block_start + j + c·m with k < len a multiple of 4m (or 2m),
    // block_start + j < m, and c < 4 (or 2), hence smaller than `len`, and
    // congruent to `j ∈ columns` modulo the block length, which the caller
    // guarantees exclusive access to.
    while layer + 1 < last_layer {
        let m = 1_usize << layer;
        let twiddles_1 = &twiddle_factors[layer as usize];
        let twiddles_2 = &twiddle_factors[layer as usize + 1];
        #[cfg(target_arch = "x86_64")]
        if let Some(limbs) = simd_limbs {
            for k in (0..len).step_by(4 * m) {
                for block_start in (0..m).step_by(block_len) {
                    for j in columns.clone().step_by(avx512::LANES) {
                        let j = block_start + j;
                        // SAFETY: See above, and `radix_4_butterflies`.
                        unsafe {
                            avx512::radix_4_butterflies_at(
                                x.ptr.cast(),
                                limbs,
                                k + j,
                                m,
                                twiddles_1.as_ptr().add(j),
                                twiddles_2.as_ptr().add(j),
                                twiddles_2.as_ptr().add(j + m),
                            )
                        };
                    }
                }
            }
            layer += 2;
            continue;
        }
        for k in (0..len).step_by(4 * m) {
            for block_start in (0..m).step_by(block_len) {
                for j in columns.clone() {
                    let j = block_start + j;
                    let i = k + j;
                    let (t0, mut t1, t2, mut t3) = unsafe {
                        (
                            x.read(i),
                            x.read(i + m),
                            x.read(i + 2 * m),
                            x.read(i + 3 * m),
                        )
                    };
                    t1 *= twiddles_1[j];
                    t3 *= twiddles_1[j];
                    let y0 = t0 + t1;
                    let y1 = t0 - t1;
                    let mut y2 = t2 + t3;
                    let mut y3 = t2 - t3;
                    y2 *= twiddles_2[j];
                    y3 *= twiddles_2[j + m];
                    unsafe {
                        x.write(i, y0 + y2);
                        x.write(i + m, y1 + y3);
                        x.write(i + 2 * m, y0 - y2);
                        x.write(i + 3 * m, y1 - y3);
                    }
                }
            }
        }
        layer += 2;
    }

    if layer < last_layer {
        let m = 1_usize << layer;
        let twiddles = &twiddle_factors[layer as usize];
        #[cfg(target_arch = "x86_64")]
        if let Some(limbs) = simd_limbs {
            for k in (0..len).step_by(2 * m) {
                for block_start in (0..m).step_by(block_len) {
                    for j in columns.clone().step_by(avx512::LANES) {
                        let j = block_start + j;
                        // SAFETY: See above, and `radix_2_butterflies`.
                        unsafe {
                            avx512::radix_2_butterflies_at(
                                x.ptr.cast(),
                                limbs,
                                k + j,
                                m,
                                twiddles.as_ptr().add(j),
                            )
                        };
                    }
                }
            }
            return;
        }
        for k in (0..len).step_by(2 * m) {
            for block_start in (0..m).step_by(block_len) {
                for j in columns.clone() {
                    let j = block_start + j;
                    let i = k + j;
                    let (u, mut v) = unsafe { (x.read(i), x.read(i + m)) };
                    v *= twiddles[j];
                    unsafe {
                        x.write(i, u + v);
                        x.write(i + m, u - v);
                    }
                }
            }
        }
    }
}

/// The number of blocks whose inputs are gathered together in
/// [`scaled_zero_padded_ntt`]: one cache line of base field elements.
const GATHER_GROUP_LEN: usize = 8;

/// The binary logarithm of the number of rows per tile in the gather of
/// [`scaled_zero_padded_ntt`]. A tile writes 2^6 cache lines, i.e., 4 KiB,
/// contiguously into every block of the group.
const LOG_2_GATHER_TILE_LEN: u32 = 6;

/// The NTT of `coefficients`, zero-padded to the length of `codeword`, after
/// scaling coefficient `i` by `offset^i`. In other words: the evaluations of
/// the polynomial with the given coefficients on the coset `offset · <ω>`,
/// where `ω` is a primitive root of unity of the codeword's length. The
/// codeword may be uninitialized; on return, every element is initialized.
///
/// Compared to scaling, zero-padding, and then transforming, this saves the
/// passes over memory that dominate the runtime of large transforms. Let `n`
/// be the codeword's length and `E` the expansion factor, i.e., the ratio of
/// `n` and the (power-of-two-padded) number of coefficients. The bit-reversal
/// permutation of the zero-padded input is nonzero only at every `E`-th
/// index, and the first `log₂(E)` layers of the transform turn each of these
/// values into `E` copies of itself. Hence, the input of every block (see
/// `ntt_unchecked`) can be gathered directly from the coefficients,
/// scaled on the fly. The gather reads the coefficients like the columns of
/// a matrix, so the blocks are processed in groups that share cache lines.
///
/// # Panics
///
/// Panics if the codeword's length is not a power of two, or if there are
/// more coefficients than the codeword is long.
pub fn scaled_zero_padded_ntt<FF>(
    coefficients: &[FF],
    offset: BFieldElement,
    codeword: &mut [MaybeUninit<FF>],
) where
    FF: FiniteField + MulAssign<BFieldElement> + Mul<BFieldElement, Output = FF>,
{
    #[cfg(target_arch = "x86_64")]
    if coset_evaluate_in_two_passes(
        coefficients,
        offset,
        codeword,
        LOG_2_VECTOR_BLOCK_LEN,
        MIN_LOG_2_LEN_FOR_TWO_PASSES,
    ) {
        return;
    }
    scaled_zero_padded_ntt_with_block_len(coefficients, offset, codeword, LOG_2_BLOCK_LEN);
}

/// The binary logarithm of the block length, in vector elements, of
/// [`coset_evaluate_in_two_passes`]. A block of 2^9 vector elements is
/// 32 KiB for base field elements, 96 KiB for extension field elements.
#[cfg(target_arch = "x86_64")]
const LOG_2_VECTOR_BLOCK_LEN: u32 = 9;

/// The binary logarithm of the maximum number of blocks: the vectors of one
/// position of all blocks, 256 KiB, must fit the L2 cache.
#[cfg(target_arch = "x86_64")]
const MAX_LOG_2_NUM_VECTOR_BLOCKS: u32 = 12;

/// The binary logarithm of the codeword length from which on
/// [`scaled_zero_padded_ntt`] uses [`coset_evaluate_in_two_passes`].
#[cfg(target_arch = "x86_64")]
const MIN_LOG_2_LEN_FOR_TWO_PASSES: u32 = 12;

/// Run the given function with scratch space of the given number of cache
/// lines.
///
/// The scratch space is allocated for the duration of the call only. Keeping
/// it around per thread would be a little faster for short transforms, but
/// would also permanently occupy memory on every thread of the pool.
#[cfg(target_arch = "x86_64")]
fn with_scratch_lines<T>(len: usize, f: impl FnOnce(&mut [avx512::Line]) -> T) -> T {
    let mut scratch = vec![avx512::Line([0; avx512::LANES]); len];
    f(&mut scratch)
}

/// The vectorized two-pass algorithm for [`scaled_zero_padded_ntt`], if it
/// is applicable. See [`avx512::CosetEvaluation`].
///
/// Blocks are `2^log_2_block_len` vector elements long, or longer if that is
/// needed to keep their number at most 2^[`MAX_LOG_2_NUM_VECTOR_BLOCKS`].
#[cfg(target_arch = "x86_64")]
fn coset_evaluation<FF>(
    coefficients: &[FF],
    offset: BFieldElement,
    codeword: &mut [MaybeUninit<FF>],
    log_2_block_len: u32,
    min_log_2_len: u32,
) -> Option<avx512::CosetEvaluation>
where
    FF: FiniteField,
{
    let limbs = simd_limbs::<FF>()?;
    let len = codeword.len();
    if !len.is_power_of_two() || coefficients.len() > len / 4 {
        return None;
    }
    let log_2_len = len.ilog2();
    if log_2_len < min_log_2_len.max(avx512::LOG_2_LANES) {
        return None;
    }

    let log_2_num_vector_elements = log_2_len - avx512::LOG_2_LANES;
    let min_log_2_block_len = log_2_num_vector_elements.saturating_sub(MAX_LOG_2_NUM_VECTOR_BLOCKS);
    let evaluation = avx512::CosetEvaluation::new(
        limbs,
        coefficients.as_ptr().cast(),
        coefficients.len(),
        offset,
        codeword.as_mut_ptr().cast(),
        log_2_len,
        log_2_block_len.max(min_log_2_block_len),
        forward_twiddle_factors(u32::try_from(len).unwrap()),
    );

    Some(evaluation)
}

/// Try to perform [`scaled_zero_padded_ntt`] with the vectorized two-pass
/// algorithm, see [`avx512::CosetEvaluation`]. Returns `false`, without
/// having done anything, if that algorithm is not applicable.
#[cfg(target_arch = "x86_64")]
fn coset_evaluate_in_two_passes<FF>(
    coefficients: &[FF],
    offset: BFieldElement,
    codeword: &mut [MaybeUninit<FF>],
    log_2_block_len: u32,
    min_log_2_len: u32,
) -> bool
where
    FF: FiniteField,
{
    let Some(evaluation) = coset_evaluation(
        coefficients,
        offset,
        codeword,
        log_2_block_len,
        min_log_2_len,
    ) else {
        return false;
    };

    // SAFETY:
    // - AVX-512 is available and `FF` is laid out as 1 or 3 base field
    //   elements, see `simd_limbs`.
    // - The evaluation was set up with the pointers and lengths of the
    //   slices, of which the codeword is exclusively borrowed, and the
    //   number of coefficients is small enough; see `coset_evaluation`.
    // - The ranges are the entire ranges, in the required order.
    // - The scratch space has the required length.
    with_scratch_lines(evaluation.scratch_len(), |scratch| unsafe {
        evaluation.transform_blocks(0..evaluation.num_groups(), scratch);
        evaluation.transform_across_blocks(0..evaluation.num_positions(), scratch);
    });

    true
}

/// Parallel version of [`coset_evaluate_in_two_passes`]: both passes consist
/// of independent units of work.
#[cfg(target_arch = "x86_64")]
fn par_coset_evaluate_in_two_passes<FF>(
    coefficients: &[FF],
    offset: BFieldElement,
    codeword: &mut [MaybeUninit<FF>],
    log_2_block_len: u32,
    min_log_2_len: u32,
) -> bool
where
    FF: FiniteField,
{
    const MAX_TASKS_PER_THREAD: usize = 4;

    /// Every task allocates its own scratch space, and a pass takes little
    /// time per unit of work: more tasks than this do not make a single
    /// evaluation faster, not even with hundreds of threads, and slow down
    /// evaluations that run in parallel to each other considerably.
    const MAX_NUM_TASKS: usize = 64;

    let Some(evaluation) = coset_evaluation(
        coefficients,
        offset,
        codeword,
        log_2_block_len,
        min_log_2_len,
    ) else {
        return false;
    };
    let evaluation = &evaluation;
    let max_num_tasks =
        (MAX_TASKS_PER_THREAD * rayon::current_num_threads().max(1)).min(MAX_NUM_TASKS);
    let scratch_len = evaluation.scratch_len();

    // SAFETY: Like in `coset_evaluate_in_two_passes`. Additionally, the
    // tasks of a pass work on disjoint ranges, and the second pass only
    // starts once the first pass has completed.
    let num_groups = evaluation.num_groups();
    let groups_per_task = num_groups.div_ceil(max_num_tasks).max(1);
    (0..num_groups)
        .into_par_iter()
        .step_by(groups_per_task)
        .for_each(|start| {
            let groups = start..(start + groups_per_task).min(num_groups);
            with_scratch_lines(scratch_len, |scratch| unsafe {
                evaluation.transform_blocks(groups, scratch)
            });
        });

    let num_positions = evaluation.num_positions();
    let positions_per_task = num_positions.div_ceil(max_num_tasks).max(1);
    (0..num_positions)
        .into_par_iter()
        .step_by(positions_per_task)
        .for_each(|start| {
            let positions = start..(start + positions_per_task).min(num_positions);
            with_scratch_lines(scratch_len, |scratch| unsafe {
                evaluation.transform_across_blocks(positions, scratch)
            });
        });

    true
}

/// Parallel version of [`scaled_zero_padded_ntt`]: the gathers of the block
/// groups, the blocks' layers, and the cross-block layers are each
/// distributed across threads.
///
/// # Panics
///
/// See [`scaled_zero_padded_ntt`].
pub fn par_scaled_zero_padded_ntt<FF>(
    coefficients: &[FF],
    offset: BFieldElement,
    codeword: &mut [MaybeUninit<FF>],
) where
    FF: FiniteField + MulAssign<BFieldElement> + Mul<BFieldElement, Output = FF>,
{
    let len = codeword.len();
    assert!(coefficients.len() <= len);
    let Some(log_2_len) = len.checked_ilog2() else {
        return;
    };
    assert!(len.is_power_of_two());
    if log_2_len < par_min_log_2_len() {
        return scaled_zero_padded_ntt(coefficients, offset, codeword);
    }
    #[cfg(target_arch = "x86_64")]
    if par_coset_evaluate_in_two_passes(
        coefficients,
        offset,
        codeword,
        LOG_2_VECTOR_BLOCK_LEN,
        MIN_LOG_2_LEN_FOR_TWO_PASSES,
    ) {
        return;
    }
    let twiddle_factors = forward_twiddle_factors(u32::try_from(len).unwrap());

    let padded_num_coefficients = coefficients.len().next_power_of_two();
    let log_2_expansion = log_2_len - padded_num_coefficients.ilog2();
    let expansion = 1_usize << log_2_expansion;
    let log_2_block_len = log_2_len.min(LOG_2_BLOCK_LEN).max(log_2_expansion);
    let block_len = 1_usize << log_2_block_len;
    let log_2_num_blocks = log_2_len - log_2_block_len;
    let num_blocks = 1_usize << log_2_num_blocks;
    let log_2_num_rows = log_2_block_len - log_2_expansion;
    let num_rows = 1_usize << log_2_num_rows;

    let offset_to_the_num_blocks = offset.mod_pow(num_blocks as u64);
    let mut row_powers = Vec::with_capacity(num_rows);
    let mut power = BFieldElement::ONE;
    for _ in 0..num_rows {
        row_powers.push(power);
        power *= offset_to_the_num_blocks;
    }

    #[cfg(target_arch = "x86_64")]
    let stream_limbs = simd_limbs::<FF>().filter(|limbs| {
        let line_len = avx512::CACHE_LINE_LEN / size_of::<u64>();
        (expansion * limbs).is_multiple_of(line_len)
            && codeword
                .as_ptr()
                .cast::<u8>()
                .addr()
                .is_multiple_of(avx512::CACHE_LINE_LEN)
    });

    let group_len = GATHER_GROUP_LEN.min(num_blocks);
    let log_2_tile_len = log_2_num_rows.min(LOG_2_GATHER_TILE_LEN);
    let tile_len = 1_usize << log_2_tile_len;
    let num_tiles = num_rows >> log_2_tile_len;

    // Phase 1: gather every group's blocks. The groups write disjoint sets of
    // blocks, so they can run concurrently; the shared slice pointer hands
    // each task its blocks.
    let shared_codeword = SharedSliceMut::new(codeword);
    (0..num_blocks)
        .into_par_iter()
        .step_by(group_len)
        .for_each(|first_column| {
            let columns = first_column..first_column + group_len;
            let blocks = columns
                .clone()
                .map(|column| bit_reverse(column, log_2_num_blocks))
                .collect::<Vec<_>>();
            let column_powers = columns
                .clone()
                .map(|column| offset.mod_pow(column as u64))
                .collect::<Vec<_>>();
            let rows =
                (0..num_tiles).flat_map(|tile| (0..tile_len).map(move |t| t * num_tiles + tile));
            for row in rows {
                let row_power = row_powers[row];
                let position_in_block = bit_reverse(row, log_2_num_rows) * expansion;
                for ((column, &block), &column_power) in
                    columns.clone().zip(&blocks).zip(&column_powers)
                {
                    let index = row * num_blocks + column;
                    let value = match coefficients.get(index) {
                        Some(&coefficient) => coefficient * (row_power * column_power),
                        None => FF::ZERO,
                    };
                    let start = block * block_len + position_in_block;
                    // SAFETY: Different groups write disjoint blocks, and
                    // within a group, every position is written exactly once
                    // (see `scaled_zero_padded_ntt`).
                    #[cfg(target_arch = "x86_64")]
                    if let Some(limbs) = stream_limbs {
                        unsafe {
                            avx512::stream_repeated(
                                shared_codeword.as_mut_ptr().add(start).cast(),
                                (&raw const value).cast(),
                                limbs,
                                expansion,
                            )
                        };
                        continue;
                    }
                    for i in 0..expansion {
                        unsafe { shared_codeword.write(start + i, MaybeUninit::new(value)) };
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            if stream_limbs.is_some() {
                // SAFETY: AVX-512 was detected.
                unsafe { avx512::fence() };
            }
        });

    // SAFETY: Every block was written to above, and the blocks partition
    // the codeword. See also the safety argument in `scaled_zero_padded_ntt`.
    let codeword =
        unsafe { std::slice::from_raw_parts_mut(codeword.as_mut_ptr().cast::<FF>(), len) };

    // Phases 2 and 3: the blocks' layers, then the cross-block layers.
    codeword
        .par_chunks_exact_mut(block_len)
        .for_each(|block| apply_layers(block, twiddle_factors, log_2_expansion, log_2_block_len));
    if log_2_block_len < log_2_len {
        par_apply_cross_block_layers(codeword, twiddle_factors, log_2_block_len, log_2_len);
    }
}

/// [`scaled_zero_padded_ntt`] with a configurable maximum block length, to
/// exercise the multi-block code paths in tests with small inputs.
fn scaled_zero_padded_ntt_with_block_len<FF>(
    coefficients: &[FF],
    offset: BFieldElement,
    codeword: &mut [MaybeUninit<FF>],
    max_log_2_block_len: u32,
) where
    FF: FiniteField + MulAssign<BFieldElement> + Mul<BFieldElement, Output = FF>,
{
    let len = codeword.len();
    assert!(coefficients.len() <= len);
    let Some(log_2_len) = len.checked_ilog2() else {
        return;
    };
    assert!(len.is_power_of_two());
    let twiddle_factors = forward_twiddle_factors(u32::try_from(len).unwrap());

    let padded_num_coefficients = coefficients.len().next_power_of_two();
    let log_2_expansion = log_2_len - padded_num_coefficients.ilog2();
    let expansion = 1_usize << log_2_expansion;

    // A block is at least one expanded coefficient, i.e., `expansion` long.
    let log_2_block_len = log_2_len.min(max_log_2_block_len).max(log_2_expansion);
    let block_len = 1_usize << log_2_block_len;
    let log_2_num_blocks = log_2_len - log_2_block_len;
    let num_blocks = 1_usize << log_2_num_blocks;
    // The coefficients block `j` needs: interpreting the coefficient indices as
    // the matrix `row · num_blocks + column`, it is column `rev(j)`, with the
    // rows in bit-reversed order.
    let log_2_num_rows = log_2_block_len - log_2_expansion;
    let num_rows = 1_usize << log_2_num_rows;

    // The scaling factor of coefficient `row · num_blocks + column` is
    // `(offset^num_blocks)^row · offset^column`.
    let offset_to_the_num_blocks = offset.mod_pow(num_blocks as u64);
    let mut row_powers = Vec::with_capacity(num_rows);
    let mut power = BFieldElement::ONE;
    for _ in 0..num_rows {
        row_powers.push(power);
        power *= offset_to_the_num_blocks;
    }

    // The gathered values are written as whole cache lines in a scattered
    // order. Streaming stores avoid the read-for-ownership of every line and
    // keep the lines out of the caches, where they would only evict data
    // that is still needed. This requires whole, aligned lines.
    #[cfg(target_arch = "x86_64")]
    let stream_limbs = simd_limbs::<FF>().filter(|limbs| {
        let line_len = avx512::CACHE_LINE_LEN / size_of::<u64>();
        (expansion * limbs).is_multiple_of(line_len)
            && codeword
                .as_ptr()
                .cast::<u8>()
                .addr()
                .is_multiple_of(avx512::CACHE_LINE_LEN)
    });

    let group_len = GATHER_GROUP_LEN.min(num_blocks);
    let log_2_tile_len = log_2_num_rows.min(LOG_2_GATHER_TILE_LEN);
    let tile_len = 1_usize << log_2_tile_len;
    let num_tiles = num_rows >> log_2_tile_len;
    for first_column in (0..num_blocks).step_by(group_len) {
        let columns = first_column..first_column + group_len;
        let blocks = columns
            .clone()
            .map(|column| bit_reverse(column, log_2_num_blocks))
            .collect::<Vec<_>>();
        let column_powers = columns
            .clone()
            .map(|column| offset.mod_pow(column as u64))
            .collect::<Vec<_>>();

        // The rows are visited in tiles such that every block receives a
        // contiguous run of positions per tile: for a fixed low part of the
        // row index, the rows with all high parts map to consecutive
        // positions. The rows of a tile are read at a constant stride.
        let rows = (0..num_tiles).flat_map(|tile| (0..tile_len).map(move |t| t * num_tiles + tile));
        for row in rows {
            let row_power = row_powers[row];
            let position_in_block = bit_reverse(row, log_2_num_rows) * expansion;
            for ((column, &block), &column_power) in
                columns.clone().zip(&blocks).zip(&column_powers)
            {
                let index = row * num_blocks + column;
                let value = match coefficients.get(index) {
                    Some(&coefficient) => coefficient * (row_power * column_power),
                    None => FF::ZERO,
                };
                let start = block * block_len + position_in_block;
                #[cfg(target_arch = "x86_64")]
                if let Some(limbs) = stream_limbs {
                    // SAFETY: AVX-512 was detected (see `simd_limbs`), `FF`
                    // is laid out as `limbs` base field elements, the
                    // `expansion` elements at `start` are in bounds, and the
                    // destination is a whole number of aligned cache lines
                    // (see `stream_limbs`).
                    unsafe {
                        avx512::stream_repeated(
                            codeword.as_mut_ptr().add(start).cast(),
                            (&raw const value).cast(),
                            limbs,
                            expansion,
                        )
                    };
                    continue;
                }
                for target in &mut codeword[start..start + expansion] {
                    target.write(value);
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        if stream_limbs.is_some() {
            // SAFETY: AVX-512 was detected.
            unsafe { avx512::fence() };
        }
        for &block in &blocks {
            let block = &mut codeword[block * block_len..(block + 1) * block_len];
            // SAFETY:
            // 1. Every element of the block was written to above: the rows
            //    and positions within a block are a bijection.
            // 2. `MaybeUninit<FF>` has the same layout as `FF`.
            // 3. The pointer and length are those of the exclusively borrowed
            //    `block`, and the resulting slice does not outlive it.
            let block = unsafe {
                std::slice::from_raw_parts_mut(block.as_mut_ptr().cast::<FF>(), block.len())
            };
            apply_layers(block, twiddle_factors, log_2_expansion, log_2_block_len);
        }
    }
    if log_2_block_len < log_2_len {
        // SAFETY: All blocks were written to above, and the blocks partition
        // the codeword. See also the safety argument above.
        let codeword = unsafe {
            std::slice::from_raw_parts_mut(codeword.as_mut_ptr().cast::<FF>(), codeword.len())
        };
        apply_cross_block_layers(codeword, twiddle_factors, log_2_block_len, log_2_len);
    }
}

/// AVX-512 kernels for the butterflies of the NTT.
///
/// The kernels operate on the base field limbs of the elements: a
/// [`BFieldElement`] is one limb, an [`XFieldElement`] three (see
/// [`FiniteField::NUM_BFE_LIMBS`]). Eight limbs fill one vector. A twiddle
/// factor multiplies every limb of its element, so for extension field
/// elements, the twiddles are expanded threefold: three base field
/// multiplications per element, exactly as in the scalar code.
///
/// The functions in this module are compiled for every x86-64 target, but
/// must only be called after checking `is_available` at runtime.
///
/// [`XFieldElement`]: crate::math::x_field_element::XFieldElement
#[cfg(target_arch = "x86_64")]
#[expect(unsafe_op_in_unsafe_fn)]
mod avx512 {
    use std::arch::x86_64::*;

    use num_traits::ConstOne;

    use super::BFieldElement;
    use super::TwiddleFactors;

    /// The number of 64-bit lanes in a vector.
    pub(super) const LANES: usize = 8;

    /// Whether the CPU supports the AVX-512 extensions used in this module.
    /// The result is cached by the standard library, so this is cheap.
    #[inline]
    pub(super) fn is_available() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    /// The length of a cache line, in bytes.
    pub(super) const CACHE_LINE_LEN: usize = 64;

    /// Lane indices repeating the three limbs of one extension field element
    /// across consecutive vectors.
    const REPETITION_3: [[i64; LANES]; 3] = [
        [0, 1, 2, 0, 1, 2, 0, 1],
        [2, 0, 1, 2, 0, 1, 2, 0],
        [1, 2, 0, 1, 2, 0, 1, 2],
    ];

    /// Write `repetitions` copies of the element at `value`, which has
    /// `limbs` limbs, to `dst`, with streaming (non-temporal) stores. Call
    /// [`fence`] before reading the written memory.
    ///
    /// # Safety
    ///
    /// `limbs` must be 1 or 3, `repetitions · limbs` a multiple of
    /// [`LANES`], `dst` aligned to [`CACHE_LINE_LEN`] with room for
    /// `repetitions · limbs` limbs, and `limbs` limbs readable from `value`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn stream_repeated(
        dst: *mut u64,
        value: *const u64,
        limbs: usize,
        repetitions: usize,
    ) {
        let num_vectors = repetitions * limbs / LANES;
        if limbs == 1 {
            let vector = _mm512_set1_epi64(*value as i64);
            for v in 0..num_vectors {
                _mm512_stream_si512(dst.add(v * LANES).cast(), vector);
            }
            return;
        }

        let value = _mm512_maskz_loadu_epi64(0b111, value.cast());
        let patterns = REPETITION_3.map(|indices| {
            let indices = _mm512_loadu_epi64(indices.as_ptr());
            _mm512_permutexvar_epi64(indices, value)
        });
        for v in 0..num_vectors {
            _mm512_stream_si512(dst.add(v * LANES).cast(), patterns[v % 3]);
        }
    }

    /// Order all preceding streaming stores before subsequent loads.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn fence() {
        _mm_sfence();
    }

    /// Lane indices expanding eight twiddle factors to the 24 limbs of eight
    /// extension field elements.
    const EXPANSION_3: [[i64; LANES]; 3] = [
        [0, 0, 0, 1, 1, 1, 2, 2],
        [2, 3, 3, 3, 4, 4, 4, 5],
        [5, 5, 6, 6, 6, 7, 7, 7],
    ];

    /// Montgomery reduction of the 128-bit product `hi · 2^64 + lo` in every
    /// lane; the vector version of [`BFieldElement::montyred`].
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn montyred(lo: __m512i, hi: __m512i) -> __m512i {
        let one = _mm512_set1_epi64(1);
        let epsilon = _mm512_set1_epi64(0xffff_ffff); // 2^64 - p

        // (a, e) = lo.overflowing_add(lo << 32)
        let a = _mm512_add_epi64(lo, _mm512_slli_epi64(lo, 32));
        let e = _mm512_cmplt_epu64_mask(a, lo);
        // b = a - (a >> 32) - e
        let b = _mm512_sub_epi64(a, _mm512_srli_epi64(a, 32));
        let b = _mm512_mask_sub_epi64(b, e, b, one);
        // (r, c) = hi.overflowing_sub(b)
        let r = _mm512_sub_epi64(hi, b);
        let c = _mm512_cmplt_epu64_mask(hi, b);
        // r - (2^64 - p) · c
        _mm512_mask_sub_epi64(r, c, r, epsilon)
    }

    /// The product of two field elements in Montgomery representation, in
    /// every lane. The inputs must be canonical, i.e., less than p; so is the
    /// output.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul(x: __m512i, y: __m512i) -> __m512i {
        let mask_32 = _mm512_set1_epi64(0xffff_ffff);
        let x_hi = _mm512_srli_epi64(x, 32);
        let y_hi = _mm512_srli_epi64(y, 32);

        // the 128-bit product from four 32×32-bit products, without overflow
        let p00 = _mm512_mul_epu32(x, y);
        let p01 = _mm512_mul_epu32(x, y_hi);
        let p10 = _mm512_mul_epu32(x_hi, y);
        let p11 = _mm512_mul_epu32(x_hi, y_hi);
        let t0 = _mm512_add_epi64(p10, _mm512_srli_epi64(p00, 32));
        let t1 = _mm512_add_epi64(p01, _mm512_and_epi64(t0, mask_32));
        let lo = _mm512_or_epi64(_mm512_slli_epi64(t1, 32), _mm512_and_epi64(p00, mask_32));
        let hi = _mm512_add_epi64(
            _mm512_add_epi64(p11, _mm512_srli_epi64(t0, 32)),
            _mm512_srli_epi64(t1, 32),
        );

        montyred(lo, hi)
    }

    /// The sum of two canonical field elements, canonical, in every lane.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add(x: __m512i, y: __m512i) -> __m512i {
        let p = _mm512_set1_epi64(BFieldElement::P as i64);
        let sum = _mm512_add_epi64(x, y);
        let overflow = _mm512_cmplt_epu64_mask(sum, x);
        let too_large = _mm512_cmpge_epu64_mask(sum, p);
        _mm512_mask_sub_epi64(sum, overflow | too_large, sum, p)
    }

    /// The difference of two canonical field elements, canonical, in every
    /// lane.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub(x: __m512i, y: __m512i) -> __m512i {
        let p = _mm512_set1_epi64(BFieldElement::P as i64);
        let difference = _mm512_sub_epi64(x, y);
        let underflow = _mm512_cmplt_epu64_mask(x, y);
        _mm512_mask_add_epi64(difference, underflow, difference, p)
    }

    /// The twiddle factors for the limbs of eight consecutive elements with
    /// `limbs` limbs each, given the eight elements' twiddle factors.
    ///
    /// # Safety
    ///
    /// `limbs` must be 1 or 3, and eight twiddle factors must be readable.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn expand_twiddles(twiddles: *const BFieldElement, limbs: usize) -> [__m512i; 3] {
        let twiddles = _mm512_loadu_epi64(twiddles.cast());
        if limbs == 1 {
            return [twiddles; 3];
        }
        EXPANSION_3.map(|indices| {
            let indices = _mm512_loadu_epi64(indices.as_ptr());
            _mm512_permutexvar_epi64(indices, twiddles)
        })
    }

    /// Two layers of butterflies on the eight elements at `i..i + 8` and
    /// their partners at distances `m`, `2m`, and `3m`.
    ///
    /// # Safety
    ///
    /// `limbs` must be 1 or 3, and `x` must point to elements with that many
    /// limbs, of which all indices up to `i + 3m + 8` must be valid and not
    /// concurrently accessed. `m` twiddles must be readable from
    /// `twiddles_1` and `twiddles_2`, and from `twiddles_2_shifted`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn radix_4_butterflies_at(
        x: *mut u64,
        limbs: usize,
        i: usize,
        m: usize,
        twiddles_1: *const BFieldElement,
        twiddles_2: *const BFieldElement,
        twiddles_2_shifted: *const BFieldElement,
    ) {
        let w1 = expand_twiddles(twiddles_1, limbs);
        let w2 = expand_twiddles(twiddles_2, limbs);
        let w2_shifted = expand_twiddles(twiddles_2_shifted, limbs);

        let a = x.add(i * limbs);
        let b = a.add(m * limbs);
        let c = b.add(m * limbs);
        let d = c.add(m * limbs);
        for v in 0..limbs {
            let offset = v * LANES;
            let (pa, pb, pc, pd) = (a.add(offset), b.add(offset), c.add(offset), d.add(offset));

            let t0 = _mm512_loadu_epi64(pa.cast());
            let t1 = mul(_mm512_loadu_epi64(pb.cast()), w1[v]);
            let t2 = _mm512_loadu_epi64(pc.cast());
            let t3 = mul(_mm512_loadu_epi64(pd.cast()), w1[v]);
            let y0 = add(t0, t1);
            let y1 = sub(t0, t1);
            let y2 = mul(add(t2, t3), w2[v]);
            let y3 = mul(sub(t2, t3), w2_shifted[v]);
            _mm512_storeu_epi64(pa.cast(), add(y0, y2));
            _mm512_storeu_epi64(pb.cast(), add(y1, y3));
            _mm512_storeu_epi64(pc.cast(), sub(y0, y2));
            _mm512_storeu_epi64(pd.cast(), sub(y1, y3));
        }
    }

    /// One layer of butterflies on the eight elements at `i..i + 8` and their
    /// partners at distance `m`.
    ///
    /// # Safety
    ///
    /// Like [`radix_4_butterflies_at`], with indices up to `i + m + 8`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn radix_2_butterflies_at(
        x: *mut u64,
        limbs: usize,
        i: usize,
        m: usize,
        twiddles: *const BFieldElement,
    ) {
        let w = expand_twiddles(twiddles, limbs);
        let a = x.add(i * limbs);
        let b = a.add(m * limbs);
        for (v, &twiddle) in w.iter().enumerate().take(limbs) {
            let offset = v * LANES;
            let (pa, pb) = (a.add(offset), b.add(offset));
            let u = _mm512_loadu_epi64(pa.cast());
            let t = mul(_mm512_loadu_epi64(pb.cast()), twiddle);
            _mm512_storeu_epi64(pa.cast(), add(u, t));
            _mm512_storeu_epi64(pb.cast(), sub(u, t));
        }
    }

    /// Two layers of butterflies on the group of `4m` elements starting at
    /// `k`; see [`super::apply_layers`].
    ///
    /// # Safety
    ///
    /// Like [`radix_4_butterflies_at`], for indices `k..k + 4m`. `m` must be
    /// a multiple of [`LANES`].
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn radix_4_butterflies(
        x: *mut u64,
        limbs: usize,
        k: usize,
        m: usize,
        twiddles_1: &[BFieldElement],
        twiddles_2: &[BFieldElement],
    ) {
        debug_assert_eq!(m, twiddles_1.len());
        debug_assert_eq!(2 * m, twiddles_2.len());
        for j in (0..m).step_by(LANES) {
            radix_4_butterflies_at(
                x,
                limbs,
                k + j,
                m,
                twiddles_1.as_ptr().add(j),
                twiddles_2.as_ptr().add(j),
                twiddles_2.as_ptr().add(j + m),
            );
        }
    }

    /// One layer of butterflies on the group of `2m` elements starting at
    /// `k`; see [`super::apply_layers`].
    ///
    /// # Safety
    ///
    /// Like [`radix_2_butterflies_at`], for indices `k..k + 2m`. `m` must be
    /// a multiple of [`LANES`].
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn radix_2_butterflies(
        x: *mut u64,
        limbs: usize,
        k: usize,
        m: usize,
        twiddles: &[BFieldElement],
    ) {
        debug_assert_eq!(m, twiddles.len());
        for j in (0..m).step_by(LANES) {
            radix_2_butterflies_at(x, limbs, k + j, m, twiddles.as_ptr().add(j));
        }
    }

    /// One cache line of limbs.
    #[repr(C, align(64))]
    #[derive(Debug, Clone, Copy)]
    pub(super) struct Line(pub(super) [u64; LANES]);

    /// The binary logarithm of the number of cosets evaluated side by side by
    /// [`coset_evaluate`]: one per lane.
    pub(super) const LOG_2_LANES: u32 = 3;

    /// The binary logarithm of the number of vector elements that
    /// [`blocked_broadcast_layers`] finishes the low layers of before moving
    /// on: 2^6 lines are 4 KiB, a fraction of the L1 cache.
    const LOG_2_CHUNK_LEN: u32 = 6;

    /// How many loads ahead of their use strided reads are prefetched.
    pub(super) const PREFETCH_DISTANCE: usize = 12;

    /// Prefetch the first and the last cache line of the slice.
    #[inline(always)]
    pub(super) fn prefetch<T>(slice: &[T]) {
        let range = slice.as_ptr_range();
        // SAFETY: Prefetching is supported by all x86-64 CPUs, and has no
        // architectural effect, whatever the address.
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(range.start.cast());
            _mm_prefetch::<_MM_HINT_T0>(range.end.cast::<i8>().wrapping_sub(1));
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn broadcast(x: BFieldElement) -> __m512i {
        _mm512_set1_epi64((&raw const x).cast::<i64>().read())
    }

    /// The decimation-in-time layers `first_layer..last_layer` of the NTT of
    /// `len` “vector elements”, each consisting of `lines` vectors. A twiddle
    /// factor multiplies all lanes of all vectors of a vector element.
    ///
    /// # Safety
    ///
    /// `len · lines` vectors must be readable and writable at `x`, `len` must
    /// be a multiple of `2^last_layer`, and `twiddle_factors` must have at
    /// least `last_layer` layers.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn broadcast_layers(
        x: *mut u64,
        lines: usize,
        len: usize,
        first_layer: u32,
        last_layer: u32,
        twiddle_factors: TwiddleFactors,
    ) {
        let stride = lines * LANES;
        let mut layer = first_layer;

        if layer == 0 && last_layer >= 2 {
            let fourth_root_of_unity = broadcast(twiddle_factors[1][1]);
            for k in (0..len).step_by(4) {
                for v in 0..lines {
                    let pa = x.add(k * stride + v * LANES);
                    let (pb, pc, pd) = (pa.add(stride), pa.add(2 * stride), pa.add(3 * stride));
                    let t0 = _mm512_loadu_epi64(pa.cast());
                    let t1 = _mm512_loadu_epi64(pb.cast());
                    let t2 = _mm512_loadu_epi64(pc.cast());
                    let t3 = _mm512_loadu_epi64(pd.cast());
                    let y0 = add(t0, t1);
                    let y1 = sub(t0, t1);
                    let y2 = add(t2, t3);
                    let y3 = mul(sub(t2, t3), fourth_root_of_unity);
                    _mm512_storeu_epi64(pa.cast(), add(y0, y2));
                    _mm512_storeu_epi64(pb.cast(), add(y1, y3));
                    _mm512_storeu_epi64(pc.cast(), sub(y0, y2));
                    _mm512_storeu_epi64(pd.cast(), sub(y1, y3));
                }
            }
            layer = 2;
        }

        while layer + 1 < last_layer {
            let m = 1_usize << layer;
            let twiddles_1 = &twiddle_factors[layer as usize][..m];
            let twiddles_2 = &twiddle_factors[layer as usize + 1][..2 * m];
            for k in (0..len).step_by(4 * m) {
                for j in 0..m {
                    let w1 = broadcast(twiddles_1[j]);
                    let w2 = broadcast(twiddles_2[j]);
                    let w2_shifted = broadcast(twiddles_2[j + m]);
                    for v in 0..lines {
                        let pa = x.add((k + j) * stride + v * LANES);
                        let pb = pa.add(m * stride);
                        let pc = pb.add(m * stride);
                        let pd = pc.add(m * stride);
                        let t0 = _mm512_loadu_epi64(pa.cast());
                        let t1 = mul(_mm512_loadu_epi64(pb.cast()), w1);
                        let t2 = _mm512_loadu_epi64(pc.cast());
                        let t3 = mul(_mm512_loadu_epi64(pd.cast()), w1);
                        let y0 = add(t0, t1);
                        let y1 = sub(t0, t1);
                        let y2 = mul(add(t2, t3), w2);
                        let y3 = mul(sub(t2, t3), w2_shifted);
                        _mm512_storeu_epi64(pa.cast(), add(y0, y2));
                        _mm512_storeu_epi64(pb.cast(), add(y1, y3));
                        _mm512_storeu_epi64(pc.cast(), sub(y0, y2));
                        _mm512_storeu_epi64(pd.cast(), sub(y1, y3));
                    }
                }
            }
            layer += 2;
        }

        if layer < last_layer {
            let m = 1_usize << layer;
            let twiddles = &twiddle_factors[layer as usize][..m];
            for k in (0..len).step_by(2 * m) {
                for (j, &twiddle) in twiddles.iter().enumerate() {
                    let w = broadcast(twiddle);
                    for v in 0..lines {
                        let pa = x.add((k + j) * stride + v * LANES);
                        let pb = pa.add(m * stride);
                        let u = _mm512_loadu_epi64(pa.cast());
                        let t = mul(_mm512_loadu_epi64(pb.cast()), w);
                        _mm512_storeu_epi64(pa.cast(), add(u, t));
                        _mm512_storeu_epi64(pb.cast(), sub(u, t));
                    }
                }
            }
        }
    }

    /// All `log_2_len` layers of the NTT of `2^log_2_len` vector elements,
    /// see [`broadcast_layers`]. The low layers are applied chunk by chunk,
    /// so that they run in the L1 cache.
    ///
    /// # Safety
    ///
    /// See [`broadcast_layers`].
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn blocked_broadcast_layers(
        x: *mut u64,
        lines: usize,
        log_2_len: u32,
        twiddle_factors: TwiddleFactors,
    ) {
        let len = 1_usize << log_2_len;
        let log_2_chunk_len = log_2_len.min(LOG_2_CHUNK_LEN);
        let chunk_len = 1_usize << log_2_chunk_len;
        for chunk in (0..len).step_by(chunk_len) {
            let chunk = x.add(chunk * lines * LANES);
            broadcast_layers(chunk, lines, chunk_len, 0, log_2_chunk_len, twiddle_factors);
        }
        broadcast_layers(x, lines, len, log_2_chunk_len, log_2_len, twiddle_factors);
    }

    /// Multiply element `i` of the `len` elements at `x` by `scale · base^i`.
    ///
    /// # Safety
    ///
    /// `limbs` must be 1 or 3, and `x` must point to `len` elements with that
    /// many limbs, which must not be concurrently accessed. `len` must be a
    /// multiple of 32.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn scale_by_powers(
        x: *mut u64,
        limbs: usize,
        len: usize,
        scale: BFieldElement,
        base: BFieldElement,
    ) {
        /// The number of independent chains of running powers, in order for
        /// the multiplications that advance them to overlap.
        const CHAINS: usize = 4;
        const ELEMENTS_PER_STEP: usize = CHAINS * LANES;
        debug_assert!(len.is_multiple_of(ELEMENTS_PER_STEP));

        let mut powers = [scale; ELEMENTS_PER_STEP];
        for i in 1..ELEMENTS_PER_STEP {
            powers[i] = powers[i - 1] * base;
        }
        let step = broadcast(base.mod_pow(ELEMENTS_PER_STEP as u64));
        let mut factors = [_mm512_setzero_si512(); CHAINS];
        for (chain, factor) in factors.iter_mut().enumerate() {
            *factor = _mm512_loadu_epi64(powers.as_ptr().add(chain * LANES).cast());
        }
        let expansion_3 = EXPANSION_3.map(|indices| _mm512_loadu_epi64(indices.as_ptr()));

        for first_element in (0..len).step_by(ELEMENTS_PER_STEP) {
            for (chain, factor) in factors.iter_mut().enumerate() {
                let elements = x.add((first_element + chain * LANES) * limbs);
                if limbs == 1 {
                    let scaled = mul(_mm512_loadu_epi64(elements.cast()), *factor);
                    _mm512_storeu_epi64(elements.cast(), scaled);
                } else {
                    for (v, &indices) in expansion_3.iter().enumerate() {
                        let line = elements.add(v * LANES);
                        let factor = _mm512_permutexvar_epi64(indices, *factor);
                        let scaled = mul(_mm512_loadu_epi64(line.cast()), factor);
                        _mm512_storeu_epi64(line.cast(), scaled);
                    }
                }
                *factor = mul(*factor, step);
            }
        }
    }

    /// Copy `num_lines` cache lines from `source`, which must be aligned to
    /// [`CACHE_LINE_LEN`], to `target`. If the target is aligned, too, the
    /// copy uses streaming stores, which neither read the target's lines
    /// before overwriting them nor leave them in the caches.
    ///
    /// # Safety
    ///
    /// The lines must be readable from `source` and writable to `target`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn copy_lines(source: *const u64, target: *mut u64, num_lines: usize) {
        let is_aligned = target.addr().is_multiple_of(CACHE_LINE_LEN);
        for line in 0..num_lines {
            let value = _mm512_load_epi64(source.add(line * LANES).cast());
            if is_aligned {
                _mm512_stream_si512(target.add(line * LANES).cast(), value);
            } else {
                _mm512_storeu_epi64(target.add(line * LANES).cast(), value);
            }
        }
        _mm_sfence();
    }

    /// For each of the given positions of vectors in a block, transform the
    /// vectors at that position in all `2^log_2_num_blocks` blocks, where a
    /// block is made of `lines_per_block` vectors.
    ///
    /// The vectors are copied into scratch space, transformed there, and
    /// written back. The copy matters: the vectors' addresses differ by a
    /// large power of two, so they compete for very few sets of the caches,
    /// and would evict each other long before the transform is done with
    /// them.
    ///
    /// # Safety
    ///
    /// - The blocks at `x` must be readable and writable.
    /// - No other thread may access the given positions of any block.
    /// - `positions` must be a sub-range of `0..lines_per_block`.
    /// - `scratch` must hold at least `2^log_2_num_blocks` lines.
    /// - `twiddle_factors` must have at least `log_2_num_blocks` layers.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn transform_across_blocks(
        x: *mut u64,
        lines_per_block: usize,
        log_2_num_blocks: u32,
        positions: std::ops::Range<usize>,
        twiddle_factors: TwiddleFactors,
        scratch: &mut [Line],
    ) {
        let num_blocks = 1_usize << log_2_num_blocks;
        debug_assert!(scratch.len() >= num_blocks);
        let block_stride = lines_per_block * LANES;
        let scratch = scratch.as_mut_ptr().cast::<u64>();

        for position in positions {
            let column = x.add(position * LANES);
            for block in 0..num_blocks {
                // Hardware prefetchers do not follow the large stride.
                let ahead = (block + PREFETCH_DISTANCE).min(num_blocks - 1);
                _mm_prefetch::<_MM_HINT_T0>(column.add(ahead * block_stride).cast());
                let value = _mm512_loadu_epi64(column.add(block * block_stride).cast());
                _mm512_store_epi64(scratch.add(block * LANES).cast(), value);
            }
            blocked_broadcast_layers(scratch, 1, log_2_num_blocks, twiddle_factors);
            for block in 0..num_blocks {
                let value = _mm512_load_epi64(scratch.add(block * LANES).cast());
                _mm512_storeu_epi64(column.add(block * block_stride).cast(), value);
            }
        }
    }

    /// The evaluation of a polynomial on the coset `offset · <ω>`, where
    /// `ω` is the primitive root of unity whose order is the length of the
    /// codeword, which the twiddle factors are made of.
    ///
    /// The codeword is treated as a sequence of “vector elements” of eight
    /// consecutive elements each. Vector element `q` holds the evaluations in
    /// `offset · ω^(8q + r)` for the lanes `r ∈ 0..8`; put differently, lane
    /// `r` of all the vector elements is the evaluation of the polynomial on
    /// the smaller coset `offset · ω^r · <ω^8>`. These eight evaluations are
    /// transforms of an eighth of the length that only differ in the scaling
    /// of their inputs, so they are performed side by side: every twiddle
    /// factor is the same for all lanes of a vector element.
    ///
    /// The transform of the vector elements needs only two passes over the
    /// codeword, which is what matters for transforms that exceed the caches,
    /// especially when many of them compete for memory bandwidth. With the
    /// number of vector elements `n = c·b`:
    ///
    /// 1. The inputs of a few blocks of `c` vector elements each are gathered
    ///    from the (scaled) coefficients into scratch space, transformed
    ///    there, multiplied by the factors that glue the blocks together
    ///    (vector element `i` of the block holding the coefficients with
    ///    indices `≡ j mod b` is multiplied by `ω_n^(i·j)`), and written to
    ///    the codeword. This writes every cache line of the codeword once,
    ///    without reading it. See [`Self::transform_blocks`].
    /// 2. For each position in a block, the vectors at that position in all
    ///    `b` blocks are copied into scratch space, transformed there, and
    ///    written back. This reads and writes every cache line of the
    ///    codeword once. See [`Self::transform_across_blocks`].
    ///
    /// This is the “four-step” structure of large transforms.
    #[derive(Debug)]
    pub(super) struct CosetEvaluation {
        /// The number of limbs per element, which is also the number of
        /// vectors per vector element.
        limbs: usize,
        coefficients: *const u64,
        num_coefficients: usize,
        codeword: *mut u64,
        twiddle_factors: TwiddleFactors,

        /// The binary logarithm of the number of vector elements.
        log_2_len: u32,

        /// The binary logarithm of the number of vector elements per block.
        log_2_block_len: u32,

        /// The binary logarithm of the number of blocks.
        log_2_num_blocks: u32,

        /// The number of blocks whose inputs are gathered together.
        group_len: usize,

        /// The offset of every lane's coset.
        lane_offsets: Line,

        /// For every “row” of coefficients, i.e., the coefficients whose
        /// index is `row · num_blocks + column`, the lanes' scaling factors
        /// `lane_offset^(row · num_blocks)`.
        row_scales: Vec<Line>,

        /// The lanes' factors for coefficients that wrap around: on the
        /// coset of a lane, `x^len` is the constant `lane_offset^len`.
        wrap_scale: Line,
    }

    // SAFETY: The pointers are only dereferenced in `unsafe` methods, whose
    // contracts regulate concurrent access.
    unsafe impl Send for CosetEvaluation {}
    unsafe impl Sync for CosetEvaluation {}

    impl CosetEvaluation {
        /// Prepare the evaluation of the polynomial with the given
        /// coefficients into the codeword of `2^log_2_codeword_len ≥ 8`
        /// elements, with blocks of (at most) `2^log_2_block_len` vector
        /// elements. All elements have `limbs` limbs.
        ///
        /// `twiddle_factors` must be the forward twiddle factors for the
        /// codeword's length.
        #[expect(clippy::too_many_arguments)]
        pub(super) fn new(
            limbs: usize,
            coefficients: *const u64,
            num_coefficients: usize,
            offset: BFieldElement,
            codeword: *mut u64,
            log_2_codeword_len: u32,
            log_2_block_len: u32,
            twiddle_factors: TwiddleFactors,
        ) -> Self {
            assert!(log_2_codeword_len >= LOG_2_LANES);
            assert_eq!(log_2_codeword_len as usize, twiddle_factors.len());
            let log_2_len = log_2_codeword_len - LOG_2_LANES;
            let log_2_block_len = log_2_block_len.min(log_2_len);
            let log_2_num_blocks = log_2_len - log_2_block_len;
            let max_group_len = if limbs == 1 { 8 } else { 4 };
            let group_len = max_group_len.min(1 << log_2_num_blocks);

            let omega = twiddle_factors.omega();
            let mut lane_offsets = [offset; LANES];
            for r in 1..LANES {
                lane_offsets[r] = lane_offsets[r - 1] * omega;
            }
            let lane_powers =
                |exponent: u64| Line(lane_offsets.map(|x| x.mod_pow(exponent).raw_u64()));

            let row_step = lane_offsets.map(|x| x.mod_pow(1 << log_2_num_blocks));
            let mut row_scale = [BFieldElement::ONE; LANES];
            let mut row_scales = Vec::with_capacity(1 << log_2_block_len);
            for _ in 0..1 << log_2_block_len {
                row_scales.push(Line(row_scale.map(|x| x.raw_u64())));
                for (scale, step) in row_scale.iter_mut().zip(row_step) {
                    *scale *= step;
                }
            }

            Self {
                limbs,
                coefficients,
                num_coefficients,
                codeword,
                twiddle_factors,
                log_2_len,
                log_2_block_len,
                log_2_num_blocks,
                group_len,
                lane_offsets: lane_powers(1),
                row_scales,
                wrap_scale: lane_powers(1 << log_2_len),
            }
        }

        /// The number of lines of scratch space the transforms require.
        pub(super) fn scratch_len(&self) -> usize {
            (self.group_len * self.lines_per_block()).max(1 << self.log_2_num_blocks)
        }

        /// The number of vectors per block.
        fn lines_per_block(&self) -> usize {
            self.limbs << self.log_2_block_len
        }

        /// The number of independent units of work of
        /// [`Self::transform_blocks`].
        pub(super) fn num_groups(&self) -> usize {
            (1 << self.log_2_num_blocks) / self.group_len
        }

        /// The number of independent units of work of
        /// [`Self::transform_across_blocks`].
        pub(super) fn num_positions(&self) -> usize {
            match self.log_2_num_blocks {
                0 => 0,
                _ => self.lines_per_block(),
            }
        }

        /// The lanes' scaling factors `lane_offset^exponent`.
        fn lane_powers(&self, exponent: usize) -> Line {
            Line(self.lane_offsets.0.map(|x| {
                BFieldElement::from_raw_u64(x)
                    .mod_pow(exponent as u64)
                    .raw_u64()
            }))
        }

        /// The first pass over the codeword, for the given groups of blocks.
        /// See [`Self`].
        ///
        /// # Safety
        ///
        /// - The CPU must support AVX-512.
        /// - `limbs` must be 1 or 3.
        /// - `num_coefficients` elements must be readable from
        ///   `coefficients`, and their number must not exceed a quarter of
        ///   the codeword's length.
        /// - The codeword must be writable, and no other thread may access
        ///   the blocks of the given groups.
        /// - `groups` must be a sub-range of `0..self.num_groups()`.
        /// - `scratch` must hold at least `self.scratch_len()` lines.
        #[target_feature(enable = "avx512f")]
        pub(super) unsafe fn transform_blocks(
            &self,
            groups: std::ops::Range<usize>,
            scratch: &mut [Line],
        ) {
            debug_assert!(scratch.len() >= self.scratch_len());
            let &Self {
                limbs,
                coefficients,
                num_coefficients,
                codeword,
                twiddle_factors,
                log_2_len,
                log_2_block_len,
                log_2_num_blocks,
                group_len,
                ..
            } = self;
            let len = 1_usize << log_2_len;
            let block_len = 1_usize << log_2_block_len;
            let num_blocks = 1_usize << log_2_num_blocks;
            let lines_per_block = self.lines_per_block();
            let scratch = scratch.as_mut_ptr().cast::<u64>();
            let is_aligned = codeword.addr().is_multiple_of(CACHE_LINE_LEN);

            // The primitive root of unity for the vector elements: `ω^8`.
            let omega = twiddle_factors.omega();
            let vector_omega = omega.mod_pow(LANES as u64);
            let lane_offsets = _mm512_load_epi64(self.lane_offsets.0.as_ptr().cast());
            let wrap_scale = _mm512_load_epi64(self.wrap_scale.0.as_ptr().cast());
            let repetition_3 = REPETITION_3.map(|indices| _mm512_loadu_epi64(indices.as_ptr()));
            let expansion_3 = EXPANSION_3.map(|indices| _mm512_loadu_epi64(indices.as_ptr()));
            let wrap_scale_3 = expansion_3.map(|i| _mm512_permutexvar_epi64(i, wrap_scale));

            for group in groups {
                let first_column = group * group_len;

                // Coefficient `row · num_blocks + column` is scaled by
                // `lane_offset^(row · num_blocks) · lane_offset^column`.
                let mut column_scales = [_mm512_setzero_si512(); LANES];
                let column_scale = self.lane_powers(first_column);
                let mut column_scale = _mm512_load_epi64(column_scale.0.as_ptr().cast());
                for scale in column_scales.iter_mut().take(group_len) {
                    *scale = column_scale;
                    column_scale = mul(column_scale, lane_offsets);
                }

                // The blocks' inputs, in bit-reversed order.
                debug_assert_eq!(block_len, self.row_scales.len());
                for (row, row_scale) in self.row_scales.iter().enumerate() {
                    let position = super::bit_reverse(row, log_2_block_len);
                    let row_scale = _mm512_load_epi64(row_scale.0.as_ptr().cast());
                    let first_index = row * num_blocks + first_column;

                    // The coefficients are read at a large stride, which
                    // hardware prefetchers do not follow.
                    let ahead = first_index + PREFETCH_DISTANCE * num_blocks;
                    if ahead < num_coefficients {
                        let ahead = coefficients.add(ahead * limbs);
                        _mm_prefetch::<_MM_HINT_T0>(ahead.cast());
                        _mm_prefetch::<_MM_HINT_T0>(ahead.add(group_len * limbs - 1).cast());
                    }

                    for (g, &scale_of_column) in column_scales.iter().enumerate().take(group_len) {
                        let index = first_index + g;
                        let target = scratch.add((g * block_len + position) * limbs * LANES);
                        if index >= num_coefficients {
                            for v in 0..limbs {
                                let target = target.add(v * LANES);
                                _mm512_store_epi64(target.cast(), _mm512_setzero_si512());
                            }
                            continue;
                        }
                        let scale = mul(row_scale, scale_of_column);
                        let wrapped_index = index + len;
                        let wraps = wrapped_index < num_coefficients;

                        if limbs == 1 {
                            let value = coefficients.add(index).read();
                            let mut value = _mm512_set1_epi64(value as i64);
                            if wraps {
                                let wrapped = coefficients.add(wrapped_index).read();
                                let wrapped = _mm512_set1_epi64(wrapped as i64);
                                value = add(value, mul(wrapped, wrap_scale));
                            }
                            _mm512_store_epi64(target.cast(), mul(value, scale));
                            continue;
                        }

                        let value = coefficients.add(index * limbs);
                        let value = _mm512_maskz_loadu_epi64(0b111, value.cast());
                        let wrapped = if wraps {
                            let wrapped = coefficients.add(wrapped_index * limbs);
                            _mm512_maskz_loadu_epi64(0b111, wrapped.cast())
                        } else {
                            _mm512_setzero_si512()
                        };
                        for v in 0..3 {
                            let mut value = _mm512_permutexvar_epi64(repetition_3[v], value);
                            if wraps {
                                let wrapped = _mm512_permutexvar_epi64(repetition_3[v], wrapped);
                                value = add(value, mul(wrapped, wrap_scale_3[v]));
                            }
                            let scale = _mm512_permutexvar_epi64(expansion_3[v], scale);
                            let target = target.add(v * LANES);
                            _mm512_store_epi64(target.cast(), mul(value, scale));
                        }
                    }
                }

                for g in 0..group_len {
                    let block = scratch.add(g * lines_per_block * LANES);
                    blocked_broadcast_layers(block, limbs, log_2_block_len, twiddle_factors);

                    // The block of column `j` holds the coefficients with
                    // indices `≡ j mod num_blocks`.
                    let column = first_column + g;
                    let block_index = super::bit_reverse(column, log_2_num_blocks);
                    let target = codeword.add(block_index * lines_per_block * LANES);
                    let glue = vector_omega.mod_pow(column as u64);
                    let mut glue_power = BFieldElement::ONE;
                    for i in 0..block_len {
                        let factor = broadcast(glue_power);
                        for v in 0..limbs {
                            let line = (i * limbs + v) * LANES;
                            let mut value = _mm512_load_epi64(block.add(line).cast());
                            if column != 0 {
                                value = mul(value, factor);
                            }

                            // Streaming stores avoid the read-for-ownership
                            // of every line, and keep the lines out of the
                            // caches, where they would only evict the scratch
                            // space.
                            if is_aligned {
                                _mm512_stream_si512(target.add(line).cast(), value);
                            } else {
                                _mm512_storeu_epi64(target.add(line).cast(), value);
                            }
                        }
                        glue_power *= glue;
                    }
                }
            }
            _mm_sfence();
        }

        /// The second pass over the codeword, for the given positions in the
        /// blocks. See [`Self`].
        ///
        /// # Safety
        ///
        /// - The CPU must support AVX-512.
        /// - [`Self::transform_blocks`] must have completed for all groups.
        /// - No other thread may access the given positions of any block.
        /// - `positions` must be a sub-range of `0..self.num_positions()`.
        /// - `scratch` must hold at least `self.scratch_len()` lines.
        #[target_feature(enable = "avx512f")]
        pub(super) unsafe fn transform_across_blocks(
            &self,
            positions: std::ops::Range<usize>,
            scratch: &mut [Line],
        ) {
            debug_assert!(scratch.len() >= self.scratch_len());
            transform_across_blocks(
                self.codeword,
                self.lines_per_block(),
                self.log_2_num_blocks,
                positions,
                self.twiddle_factors,
                scratch,
            );
        }
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

/// Parallel version of [`unscale`].
#[inline]
fn par_unscale<FF>(array: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement> + Send + Sync,
{
    let n_inv = BFieldElement::from(array.len()).inverse_or_zero();
    array.par_iter_mut().for_each(|elem| *elem *= n_inv);
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
    // For large enough `slice_len`, this computation could benefit from
    // parallelization. However, if NTT is also being called from within a
    // rayon-parallel context, parallelization here can lead to a deadlock.
    // The relevant issue is <https://github.com/rayon-rs/rayon/issues/592>.
    //
    // As a short summary, consider the following scenario.
    // 1. Some task on some rayon thread calls NTT's OnceLock::get_or_init.
    // 2. The initialization task, i.e., execution of this function, is also
    //    done in parallel. Some of that work is stolen by other rayon threads.
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
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use itertools::Itertools;
    use num_traits::ConstZero;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_adapter::arb;

    use super::*;
    use crate::math::other::random_elements;
    use crate::math::traits::PrimitiveRootOfUnity;
    use crate::math::x_field_element::EXTENSION_DEGREE;
    use crate::prelude::*;
    use crate::tests::proptest;
    use crate::tests::test;
    use crate::xfe;

    #[macro_rules_attr::apply(test)]
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

    #[macro_rules_attr::apply(test)]
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

    #[macro_rules_attr::apply(test)]
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

    #[macro_rules_attr::apply(test)]
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

    #[macro_rules_attr::apply(test)]
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

    #[macro_rules_attr::apply(test)]
    fn ntt_on_empty_input() {
        let mut input_output = vec![];
        let original_input = input_output.clone();

        ntt::<BFieldElement>(&mut input_output);
        assert_eq!(0, input_output.len());

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[macro_rules_attr::apply(proptest(cases = 10))]
    fn ntt_on_input_of_length_one(bfe: BFieldElement) {
        let mut test_vector = vec![bfe];
        ntt(&mut test_vector);
        assert_eq!(vec![bfe], test_vector);
    }

    // Make sure that caches are correctly populated in edge cases.
    #[macro_rules_attr::apply(test)]
    fn ntt_on_input_of_length_0_then_1_then_0() {
        let mut empty = Vec::<BFieldElement>::new();
        ntt(&mut empty);
        ntt(&mut [BFieldElement::new(0)]);
        ntt(&mut empty);
    }

    #[macro_rules_attr::apply(proptest(cases = 10))]
    fn ntt_then_intt_is_identity_operation(
        #[strategy((0_usize..18).prop_map(|l| 1 << l))] _vector_length: usize,
        #[strategy(vec(arb(), #_vector_length))] mut input: Vec<BFieldElement>,
    ) {
        let original_input = input.clone();
        ntt::<BFieldElement>(&mut input);
        intt::<BFieldElement>(&mut input);
        assert_eq!(original_input, input);
    }

    #[macro_rules_attr::apply(test)]
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

    #[macro_rules_attr::apply(test)]
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

    #[macro_rules_attr::apply(test)]
    fn bit_reverse_permutation_agrees_with_naive_permutation() {
        // small sizes use the naïve permutation, larger sizes the tiled one
        for log_size in 0..=(2 * LOG_2_TILE_LEN + 3) {
            let size = 1_usize << log_size;
            let mut permuted = (0..size).collect_vec();
            bit_reverse_permutation(&mut permuted);

            let expected = (0..size).map(|i| bit_reverse(i, log_size)).collect_vec();
            assert_eq!(expected, permuted, "log_size: {log_size}");
        }
    }

    /// The internal structure of the NTT depends on the input length: the
    /// bit-reversal permutation is tiled for inputs longer than
    /// 2^(2·[`LOG_2_TILE_LEN`]), and the butterfly layers are split into
    /// per-block and cross-block layers for inputs longer than
    /// 2^[`LOG_2_BLOCK_LEN`]. Check all of these code paths against the
    /// definition of the NTT, i.e., polynomial evaluation, at a few points.
    #[macro_rules_attr::apply(test)]
    fn ntt_agrees_with_polynomial_evaluation_on_large_inputs() {
        let log_sizes = [
            2 * LOG_2_TILE_LEN,
            2 * LOG_2_TILE_LEN + 1,
            LOG_2_BLOCK_LEN,
            LOG_2_BLOCK_LEN + 1,
            LOG_2_BLOCK_LEN + 2,
        ];
        for log_size in log_sizes {
            let size = 1_usize << log_size;
            let omega = BFieldElement::primitive_root_of_unity(size as u64).unwrap();
            let coefficients: Vec<XFieldElement> = random_elements(size);
            let polynomial = Polynomial::new(coefficients.clone());

            let mut evaluations = coefficients.clone();
            ntt(&mut evaluations);
            for i in [0, 1, 2, 3, size / 2 - 1, size / 2, size - 2, size - 1] {
                let point = omega.mod_pow(i as u64);
                let expected = polynomial.evaluate_in_same_field(point.into());
                assert_eq!(expected, evaluations[i], "log_size: {log_size}, i: {i}");
            }

            intt(&mut evaluations);
            assert_eq!(coefficients, evaluations, "log_size: {log_size}");
        }
    }

    #[macro_rules_attr::apply(test)]
    fn parallel_ntt_agrees_with_serial_ntt() {
        // covers the serial fallback, the tiled bit-reversal, and the
        // cross-block layers with a reduced block length
        for log_size in [
            0,
            1,
            5,
            PAR_MIN_LOG_2_LEN - 1,
            PAR_MIN_LOG_2_LEN,
            15,
            17,
            18,
        ] {
            let size = 1_usize << log_size;
            let original: Vec<XFieldElement> = random_elements(size);

            let mut serial = original.clone();
            ntt(&mut serial);
            let mut parallel = original.clone();
            par_ntt(&mut parallel);
            assert_eq!(serial, parallel, "log_size: {log_size}");

            par_intt(&mut parallel);
            assert_eq!(original, parallel, "log_size: {log_size}");
        }
    }

    #[macro_rules_attr::apply(proptest(cases = 200))]
    fn scaled_zero_padded_ntt_agrees_with_scaling_padding_and_transforming(
        #[strategy(0_u32..=12)] log_2_len: u32,
        #[strategy(0_usize..=(1 << #log_2_len))] _num_coefficients: usize,
        #[strategy(vec(arb(), #_num_coefficients))] coefficients: Vec<XFieldElement>,
        #[filter(!#offset.is_zero())] offset: BFieldElement,
        #[strategy(0_u32..=6)] max_log_2_block_len: u32,
    ) {
        let len = 1_usize << log_2_len;
        let mut expected = coefficients
            .iter()
            .zip(std::iter::successors(Some(BFieldElement::ONE), |&p| {
                Some(p * offset)
            }))
            .map(|(&c, p)| c * p)
            .collect_vec();
        expected.resize(len, XFieldElement::ZERO);
        ntt(&mut expected);

        let mut buffer = vec![MaybeUninit::<XFieldElement>::uninit(); len + 8];
        let codeword = cache_line_aligned(&mut buffer, len);
        scaled_zero_padded_ntt_with_block_len(&coefficients, offset, codeword, max_log_2_block_len);
        let codeword = codeword
            .iter()
            .map(|c| unsafe { c.assume_init() })
            .collect_vec();
        prop_assert_eq!(expected, codeword);
    }

    /// The first `len` elements of `buffer` starting at a cache line.
    fn cache_line_aligned<T>(buffer: &mut [MaybeUninit<T>], len: usize) -> &mut [MaybeUninit<T>] {
        let misalignment = buffer.as_ptr().cast::<u8>().addr() % 64;
        let skip = (64 - misalignment) % 64 / size_of::<T>().max(1);
        &mut buffer[skip..skip + len]
    }

    /// Compare both the sequential and the parallel two-pass coset
    /// evaluation to scaling, zero-padding, and transforming.
    #[cfg(target_arch = "x86_64")]
    fn assert_two_pass_coset_evaluation_is_correct<FF>(
        coefficients: &[FF],
        offset: BFieldElement,
        log_2_len: u32,
        log_2_block_len: u32,
        misalignment: usize,
    ) -> std::result::Result<(), TestCaseError>
    where
        FF: FiniteField + MulAssign<BFieldElement> + Mul<BFieldElement, Output = FF>,
    {
        let len = 1_usize << log_2_len;
        let mut expected = coefficients
            .iter()
            .zip(std::iter::successors(Some(BFieldElement::ONE), |&p| {
                Some(p * offset)
            }))
            .map(|(&c, p)| c * p)
            .collect_vec();
        expected.resize(len, FF::ZERO);
        ntt(&mut expected);

        type Evaluate<FF> = fn(&[FF], BFieldElement, &mut [MaybeUninit<FF>], u32, u32) -> bool;
        let evaluators: [Evaluate<FF>; 2] = [
            coset_evaluate_in_two_passes,
            par_coset_evaluate_in_two_passes,
        ];
        for evaluate in evaluators {
            let mut buffer = vec![MaybeUninit::<FF>::uninit(); len + 9];
            let codeword = cache_line_aligned(&mut buffer, len + 1);
            let codeword = &mut codeword[misalignment..misalignment + len];
            let is_applicable = evaluate(coefficients, offset, codeword, log_2_block_len, 0);
            prop_assert_eq!(avx512::is_available(), is_applicable);
            if !is_applicable {
                continue;
            }
            let codeword = codeword
                .iter()
                .map(|c| unsafe { c.assume_init() })
                .collect_vec();
            prop_assert_eq!(&expected, &codeword);
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[macro_rules_attr::apply(proptest(cases = 200))]
    fn two_pass_coset_evaluation_of_base_field_polynomial_is_correct(
        #[strategy(3_u32..=13)] log_2_len: u32,
        #[strategy(0_usize..=(1 << (#log_2_len - 2)))] _num_coefficients: usize,
        #[strategy(vec(arb(), #_num_coefficients))] coefficients: Vec<BFieldElement>,
        #[filter(!#offset.is_zero())] offset: BFieldElement,
        #[strategy(0_u32..=6)] log_2_block_len: u32,
        #[strategy(0_usize..=1)] misalignment: usize,
    ) {
        assert_two_pass_coset_evaluation_is_correct(
            &coefficients,
            offset,
            log_2_len,
            log_2_block_len,
            misalignment,
        )?;
    }

    #[cfg(target_arch = "x86_64")]
    #[macro_rules_attr::apply(proptest(cases = 200))]
    fn two_pass_coset_evaluation_of_extension_field_polynomial_is_correct(
        #[strategy(3_u32..=13)] log_2_len: u32,
        #[strategy(0_usize..=(1 << (#log_2_len - 2)))] _num_coefficients: usize,
        #[strategy(vec(arb(), #_num_coefficients))] coefficients: Vec<XFieldElement>,
        #[filter(!#offset.is_zero())] offset: BFieldElement,
        #[strategy(0_u32..=6)] log_2_block_len: u32,
        #[strategy(0_usize..=1)] misalignment: usize,
    ) {
        assert_two_pass_coset_evaluation_is_correct(
            &coefficients,
            offset,
            log_2_len,
            log_2_block_len,
            misalignment,
        )?;
    }

    #[cfg(target_arch = "x86_64")]
    #[macro_rules_attr::apply(proptest(cases = 6))]
    fn two_pass_coset_evaluation_with_default_geometry_is_correct(
        #[strategy(14_u32..=19)] log_2_len: u32,
        #[strategy(0_usize..500)] num_wrapping_coefficients: usize,
        #[filter(!#offset.is_zero())] offset: BFieldElement,
        seed: u64,
    ) {
        let num_coefficients = (1 << (log_2_len - 3)) + num_wrapping_coefficients;
        let mut coefficient = BFieldElement::new(seed);
        let mut next_coefficient = || {
            coefficient = coefficient * coefficient + BFieldElement::new(7);
            coefficient
        };
        let b_coefficients = (0..num_coefficients)
            .map(|_| next_coefficient())
            .collect_vec();
        let x_coefficients = (0..num_coefficients)
            .map(|_| XFieldElement::new([(); EXTENSION_DEGREE].map(|()| next_coefficient())))
            .collect_vec();

        let block_len = LOG_2_VECTOR_BLOCK_LEN;
        assert_two_pass_coset_evaluation_is_correct(
            &b_coefficients,
            offset,
            log_2_len,
            block_len,
            0,
        )?;
        assert_two_pass_coset_evaluation_is_correct(
            &x_coefficients,
            offset,
            log_2_len,
            block_len,
            0,
        )?;
    }

    fn assert_transforms_into_agree_with_transforms_in_place<FF>(
        values: &[FF],
    ) -> std::result::Result<(), TestCaseError>
    where
        FF: FiniteField + MulAssign<BFieldElement>,
    {
        type Transform<FF> = fn(&mut [FF]);
        type TransformInto<FF> = fn(&[FF], &mut [MaybeUninit<FF>]);
        let transforms: [(Transform<FF>, TransformInto<FF>); 2] =
            [(ntt, ntt_into), (intt, intt_into)];
        for (transform, transform_into) in transforms {
            let mut expected = values.to_vec();
            transform(&mut expected);

            for misalignment in [0, 1] {
                let mut buffer = vec![MaybeUninit::<FF>::uninit(); values.len() + 9];
                let output = cache_line_aligned(&mut buffer, values.len() + 1);
                let output = &mut output[misalignment..misalignment + values.len()];
                transform_into(values, output);
                let output = output
                    .iter()
                    .map(|c| unsafe { c.assume_init() })
                    .collect_vec();
                prop_assert_eq!(&expected, &output);
            }
        }

        Ok(())
    }

    #[macro_rules_attr::apply(proptest(cases = 20))]
    fn transforms_into_agree_with_transforms_in_place(
        #[strategy(0_u32..=15)] log_2_len: u32,
        seed: u64,
    ) {
        let mut element = BFieldElement::new(seed);
        let mut next_element = || {
            element = element * element + BFieldElement::new(7);
            element
        };
        let b_values = (0..1 << log_2_len).map(|_| next_element()).collect_vec();
        let x_values = (0..1 << log_2_len)
            .map(|_| XFieldElement::new([(); EXTENSION_DEGREE].map(|()| next_element())))
            .collect_vec();
        assert_transforms_into_agree_with_transforms_in_place(&b_values)?;
        assert_transforms_into_agree_with_transforms_in_place(&x_values)?;
    }

    #[cfg(target_arch = "x86_64")]
    #[macro_rules_attr::apply(proptest(cases = 100))]
    fn two_pass_transform_into_is_correct(
        #[strategy(6_u32..=12)] log_2_len: u32,
        #[strategy(vec(arb(), 1 << #log_2_len))] values: Vec<XFieldElement>,
        #[strategy(0_u32..=7)] log_2_block_len: u32,
        scale: BFieldElement,
    ) {
        let twiddle_factors = forward_twiddle_factors(values.len() as u32);
        let mut expected = values.clone();
        ntt(&mut expected);
        expected.iter_mut().for_each(|e| *e *= scale);

        let mut output = vec![MaybeUninit::<XFieldElement>::uninit(); values.len()];
        let is_applicable = ntt_into_in_two_passes(
            &values,
            &mut output,
            twiddle_factors,
            scale,
            log_2_block_len,
        );
        let has_many_blocks = log_2_len > log_2_block_len.max(5);
        prop_assert_eq!(avx512::is_available() && has_many_blocks, is_applicable);
        if is_applicable {
            let output = output
                .iter()
                .map(|c| unsafe { c.assume_init() })
                .collect_vec();
            prop_assert_eq!(expected, output);
        }
    }

    #[macro_rules_attr::apply(proptest(cases = 4))]
    fn par_scaled_zero_padded_ntt_agrees_with_sequential_version(
        #[strategy(17_u32..=19)] log_2_len: u32,
        #[strategy(0_u32..=4)] log_2_expansion: u32,
        #[filter(!#offset.is_zero())] offset: BFieldElement,
        seed: u64,
    ) {
        let len = 1_usize << log_2_len;
        let num_coefficients = len >> log_2_expansion;
        let coefficients = (0..num_coefficients as u64)
            .map(|i| XFieldElement::from(BFieldElement::new(i.wrapping_mul(seed | 1))))
            .collect_vec();

        let mut expected_buffer = vec![MaybeUninit::<XFieldElement>::uninit(); len + 8];
        let expected = cache_line_aligned(&mut expected_buffer, len);
        scaled_zero_padded_ntt(&coefficients, offset, expected);
        let expected = expected
            .iter()
            .map(|c| unsafe { c.assume_init() })
            .collect_vec();

        let mut codeword_buffer = vec![MaybeUninit::<XFieldElement>::uninit(); len + 8];
        let codeword = cache_line_aligned(&mut codeword_buffer, len);
        par_scaled_zero_padded_ntt(&coefficients, offset, codeword);
        let codeword = codeword
            .iter()
            .map(|c| unsafe { c.assume_init() })
            .collect_vec();
        prop_assert_eq!(expected, codeword);
    }

    #[macro_rules_attr::apply(proptest(cases = 4))]
    fn scaled_zero_padded_ntt_agrees_for_default_block_len(
        #[strategy(17_u32..=18)] log_2_len: u32,
        #[strategy(0_u32..=4)] log_2_expansion: u32,
        #[filter(!#offset.is_zero())] offset: BFieldElement,
        seed: u64,
    ) {
        let len = 1_usize << log_2_len;
        let num_coefficients = len >> log_2_expansion;
        let coefficients = (0..num_coefficients as u64)
            .map(|i| BFieldElement::new(i.wrapping_mul(seed | 1)))
            .collect_vec();
        let mut expected = coefficients
            .iter()
            .zip(std::iter::successors(Some(BFieldElement::ONE), |&p| {
                Some(p * offset)
            }))
            .map(|(&c, p)| c * p)
            .collect_vec();
        expected.resize(len, BFieldElement::ZERO);
        ntt(&mut expected);

        let mut buffer = vec![MaybeUninit::<BFieldElement>::uninit(); len + 8];
        let codeword = cache_line_aligned(&mut buffer, len);
        scaled_zero_padded_ntt(&coefficients, offset, codeword);
        let codeword = codeword
            .iter()
            .map(|c| unsafe { c.assume_init() })
            .collect_vec();
        prop_assert_eq!(expected, codeword);
    }

    #[macro_rules_attr::apply(test)]
    fn shared_twiddle_factors_equal_twiddle_factors_for_specific_length() {
        for log_2_len in 0..=12 {
            let len = 1 << log_2_len;
            let root = BFieldElement::primitive_root_of_unity(u64::from(len)).unwrap();
            for (shared, omega) in [
                (forward_twiddle_factors(len), root),
                (inverse_twiddle_factors(len), root.inverse()),
            ] {
                let expected = twiddle_factors(len, omega);
                assert_eq!(expected.len(), shared.len());
                for (layer, expected_layer) in expected.iter().enumerate() {
                    assert_eq!(expected_layer, &shared[layer]);
                }
                if log_2_len > 0 {
                    assert_eq!(omega, shared.omega());
                }
            }
        }
    }

    #[macro_rules_attr::apply(test)]
    fn primitive_roots_of_unity_of_powers_of_two_are_consistent() {
        for log_2_order in 1..=32 {
            let root = BFieldElement::primitive_root_of_unity(1 << log_2_order).unwrap();
            let smaller_root = BFieldElement::primitive_root_of_unity(1 << (log_2_order - 1));
            assert_eq!(smaller_root.unwrap(), root * root);
        }
    }

    #[macro_rules_attr::apply(test)]
    fn twiddle_factors_can_be_computed() {
        // exponential growth is powerful; cap the number of domains
        for log_size in 0..NUM_DOMAINS - 5 {
            let size = 1 << log_size;
            let root = BFieldElement::primitive_root_of_unity(size.into()).unwrap();
            twiddle_factors(size, root);
        }
    }
}
