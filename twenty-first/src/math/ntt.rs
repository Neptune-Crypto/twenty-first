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

/// The (cached) twiddle factors for the forward NTT of the given length.
fn forward_twiddle_factors(slice_len: u32) -> &'static [Vec<BFieldElement>] {
    static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    ALL_TWIDDLE_FACTORS[slice_len.checked_ilog2().unwrap_or(0) as usize].get_or_init(|| {
        let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
        twiddle_factors(slice_len, omega)
    })
}

/// The (cached) twiddle factors for the inverse NTT of the given length.
fn inverse_twiddle_factors(slice_len: u32) -> &'static [Vec<BFieldElement>] {
    static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    ALL_TWIDDLE_FACTORS[slice_len.checked_ilog2().unwrap_or(0) as usize].get_or_init(|| {
        let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
        twiddle_factors(slice_len, omega.inverse())
    })
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
/// cache-blocked part of the NTT. A block of 2^16 base field elements
/// occupies 512 KiB, which fits the L2 cache of common x86-64 CPUs.
const LOG_2_BLOCK_LEN: u32 = 16;

/// The binary logarithm of the tile side length used by the
/// [bit-reversal permutation][bit_reverse_permutation].
const LOG_2_TILE_LEN: u32 = 4;

/// The number of adjacent “columns” processed together in the high
/// (cross-block) layers of the NTT. See [`apply_cross_block_layers`].
const CROSS_BLOCK_COLUMN_WIDTH: usize = 64;

/// Below this (binary logarithm of the) length, the parallel NTT falls back
/// to the serial one: the work is too little to be worth distributing.
const PAR_MIN_LOG_2_LEN: u32 = 12;

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
fn ntt_unchecked<FF>(x: &mut [FF], twiddle_factors: &[Vec<BFieldElement>])
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
fn par_ntt_unchecked<FF>(x: &mut [FF], twiddle_factors: &[Vec<BFieldElement>])
where
    FF: FiniteField + MulAssign<BFieldElement> + Send + Sync,
{
    let Some(log_2_len) = x.len().checked_ilog2() else {
        return;
    };
    if log_2_len < PAR_MIN_LOG_2_LEN {
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
    twiddle_factors: &[Vec<BFieldElement>],
    first_layer: u32,
    last_layer: u32,
) where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let mut layer = first_layer;

    // The twiddle factors of the first two layers are 1, 1, and a primitive
    // 4th root of unity: only one multiplication per radix-4 butterfly.
    if layer == 0 && last_layer >= 2 {
        let fourth_root_of_unity = twiddle_factors[1][1];
        for butterfly in x.chunks_exact_mut(4) {
            let [t0, t1, t2, t3] = [butterfly[0], butterfly[1], butterfly[2], butterfly[3]];
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
    twiddle_factors: &[Vec<BFieldElement>],
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
    twiddle_factors: &[Vec<BFieldElement>],
    first_layer: u32,
    last_layer: u32,
) where
    FF: FiniteField + MulAssign<BFieldElement> + Send + Sync,
{
    let block_len = 1_usize << first_layer;
    let column_width = CROSS_BLOCK_COLUMN_WIDTH.min(block_len);
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
    twiddle_factors: &[Vec<BFieldElement>],
    first_layer: u32,
    last_layer: u32,
    columns: std::ops::Range<usize>,
) where
    FF: FiniteField + MulAssign<BFieldElement>,
{
    let block_len = 1_usize << first_layer;
    let len = x.len;
    let mut layer = first_layer;

    // SAFETY (for all accesses below): all accessed indices are of the form
    // k + block_start + j + c·m with k < len a multiple of 4m (or 2m),
    // block_start + j < m, and c < 4 (or 2), hence smaller than `len`, and
    // congruent to `j ∈ columns` modulo the block length, which the caller
    // guarantees exclusive access to.
    while layer + 1 < last_layer {
        let m = 1_usize << layer;
        let twiddles_1 = &twiddle_factors[layer as usize];
        let twiddles_2 = &twiddle_factors[layer as usize + 1];
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
/// [`ntt_unchecked`]) can be gathered directly from the coefficients,
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
    scaled_zero_padded_ntt_with_block_len(coefficients, offset, codeword, LOG_2_BLOCK_LEN);
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

    let group_len = GATHER_GROUP_LEN.min(num_blocks);
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

        for (row, &row_power) in row_powers.iter().enumerate() {
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
                for target in &mut codeword[start..start + expansion] {
                    target.write(value);
                }
            }
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

        let mut codeword = vec![MaybeUninit::<XFieldElement>::uninit(); len];
        scaled_zero_padded_ntt_with_block_len(
            &coefficients,
            offset,
            &mut codeword,
            max_log_2_block_len,
        );
        let codeword = codeword
            .into_iter()
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

        let mut codeword = vec![MaybeUninit::<BFieldElement>::uninit(); len];
        scaled_zero_padded_ntt(&coefficients, offset, &mut codeword);
        let codeword = codeword
            .into_iter()
            .map(|c| unsafe { c.assume_init() })
            .collect_vec();
        prop_assert_eq!(expected, codeword);
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
