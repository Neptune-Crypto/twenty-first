//! Make [`Tip5`] even faster by using SIMD, in particular, AVX-512.
//!
//! The functions in this module are compiled for every x86-64 target, but
//! must only be called after checking [`Tip5::avx512_is_available`] at
//! runtime. This way, binaries that are not compiled with AVX-512 enabled
//! still benefit on CPUs that support it.

use std::arch::x86_64::*;

use num_traits::ConstOne;
use num_traits::ConstZero;

use super::Column;
use super::Digest;
use super::LOOKUP_TABLE;
use super::NUM_ROUNDS;
use super::NUM_SPLIT_AND_LOOKUP;
use super::RATE;
use super::STATE_SIZE;
use super::Tip5;
use super::WrappingArithmetic;
use crate::prelude::BFieldElement;
use crate::util_types::sponge::Domain;

/// The round constants in Montgomery representation, split into their high
/// (`_U`) and low (`_L`) 32-bit limbs.
const RCS_MONT_U: [u64; 80] = [
    0x61ab60dc, 0xd9547ed0, 0xa1de063d, 0x876c8676, 0x889cfb95, 0x43699f00, 0x7190db57, 0xd2b0d4b0,
    0xd483cd36, 0x44882a55, 0x9f498aa3, 0x79338d4b, 0x52c5b216, 0x48adad93, 0xfec868b5, 0xfb6b0d8a,
    0x20ef0328, 0x5bba5802, 0x27287a26, 0x4e193411, 0xa977eae0, 0x63fc191a, 0xaf39b210, 0x5933202e,
    0xbfcf71e4, 0xcc520bfb, 0xf774f673, 0x0309bc69, 0x275f3cb2, 0x2c8f905a, 0x61e609b3, 0x5c92c93a,
    0x56411dbf, 0x5fc2a26b, 0x3d9f2bf2, 0x5ca88c43, 0x2e1c1552, 0x3220a672, 0x4b861c4d, 0xeb86ebd6,
    0xbc3902de, 0x516bcbc0, 0x738f27cf, 0xeac8ea36, 0x4bf937c4, 0x220e6746, 0x07e796f8, 0xf2f6dd71,
    0x7d6e3a40, 0xe73743d7, 0xef802e57, 0x336e6aa5, 0xf3c8b226, 0x6afb2112, 0x25531967, 0x3866d0ee,
    0xd2215022, 0x12ee85b1, 0xfcd23eb4, 0xd727752f, 0xaff543b3, 0x17f192d4, 0xb026adc0, 0xe35c1017,
    0x6080bd06, 0x0b8a28b7, 0xae9da4ca, 0xd9e5a26b, 0x2d337846, 0xb7eee345, 0x59dde50c, 0x5ee62a88,
    0xf6a203d0, 0x3b6ae69e, 0x2be69c37, 0xdfff43cb, 0x5f4fdc6a, 0x97c0d760, 0x14148eba, 0xf2f24472,
];
const RCS_MONT_L: [u64; 80] = [
    0xe12a6137, 0x3c2d8f14, 0xce16c34a, 0x5d4cf10b, 0xa3fe2af2, 0xe0086636, 0x5712e44b, 0x05bceb49,
    0xb29f2156, 0x88310f48, 0xb091da34, 0xf1ff20f5, 0xfc597178, 0xbe758d99, 0x9853d114, 0x2cc48735,
    0xebc0eeec, 0x5bdfe8e6, 0x02df87a9, 0x0c7397fa, 0xcf6133cb, 0x6bef3d61, 0x96b1f98d, 0xa3216fc1,
    0x029fd62d, 0xfb4ad152, 0xe0c840b1, 0xad2abfa1, 0x7a336665, 0xe6ad794b, 0x1a9aa328, 0xf0bb400b,
    0xe9bc674a, 0x895bd10c, 0x39dfe4f5, 0xf0c467e0, 0x35b5227b, 0xe82efadd, 0x0fdd1d04, 0x0308861f,
    0x832913f5, 0x1bf8f7c6, 0xac69f270, 0xe798f708, 0xaa81ef62, 0x9498717d, 0xf9fad5c4, 0xe16d8ff5,
    0x7aefd019, 0xd4c162e9, 0x717a8a87, 0x53bcde49, 0x5e71152a, 0xf02e0b04, 0x3d64ddb1, 0x91012a32,
    0x4702d633, 0x5e3f4dac, 0xc9b208c8, 0x3d490349, 0xb670e77e, 0xf48bc718, 0x0615dfdf, 0xdcab5e5b,
    0x71014a42, 0xfe9a2b22, 0xcc26240d, 0x732867a0, 0x92fe65b8, 0xdcb6de4c, 0x8f0c9826, 0xe059226d,
    0xa302d668, 0x93fb6a88, 0x53fb6dbf, 0x9f9a0f27, 0x15b64f4b, 0x903d0ed1, 0xdb21a28b, 0xb971e6c9,
];

#[expect(unsafe_op_in_unsafe_fn)]
impl Tip5 {
    /// Whether the CPU supports all AVX-512 extensions used in this module.
    ///
    /// The result is cached by the standard library, so calling this is
    /// cheap. If the crate is compiled with these features enabled, e.g.,
    /// through `-C target-cpu=native`, the check is resolved at compile time.
    #[inline]
    pub(super) fn avx512_is_available() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512ifma")
    }

    /// The Tip5 permutation, using AVX-512.
    ///
    /// # Safety
    ///
    /// The CPU must support the AVX-512 extensions “f”, “bw”, “dq”, “vbmi”,
    /// and “ifma”; see [`Self::avx512_is_available`].
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn permutation_avx512(&mut self) {
        let (mut a, mut b) = Self::load_state(&self.state);
        for round_index in 0..NUM_ROUNDS {
            (a, b) = Self::round_in_registers(a, b, round_index);
        }
        Self::store_state(&mut self.state, a, b);
    }

    /// One round of the Tip5 permutation, using AVX-512.
    ///
    /// # Safety
    ///
    /// See [`Self::permutation_avx512`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn round_avx512(&mut self, round_index: usize) {
        let (a, b) = Self::load_state(&self.state);
        let (a, b) = Self::round_in_registers(a, b, round_index);
        Self::store_state(&mut self.state, a, b);
    }

    /// The state's first and second half, as vectors.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn load_state(state: &[BFieldElement; STATE_SIZE]) -> (__m512i, __m512i) {
        let a = _mm512_load_epi64(state.as_ptr().offset(0x00) as *const i64);
        let b = _mm512_load_epi64(state.as_ptr().offset(0x08) as *const i64);
        (a, b)
    }

    /// The inverse of [`Self::load_state`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn store_state(state: &mut [BFieldElement; STATE_SIZE], a: __m512i, b: __m512i) {
        _mm512_store_epi64(state.as_mut_ptr().offset(0x00) as *mut i64, a);
        _mm512_store_epi64(state.as_mut_ptr().offset(0x08) as *mut i64, b);
    }

    /// One round on a state held in two vectors.
    ///
    /// A single permutation has little parallelism: the S-box's power maps
    /// are two dependency chains, and the MDS matrix's rows are only a few
    /// more. Hence, the code below cares about the lengths of those chains.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn round_in_registers(a: __m512i, b: __m512i, round_index: usize) -> (__m512i, __m512i) {
        let (a, b) = Self::sbox_layer_avx512(a, b);
        Self::mds_rcs_avx512(a, b, round_index)
    }

    /// The S-box layer. The input must be canonical; the output is in
    /// [limb form](Limbs32), which is what the MDS layer consumes.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn sbox_layer_avx512(a: __m512i, b: __m512i) -> (Limbs32, Limbs32) {
        /* S-BOX */
        let asbox = Self::split32(Self::lookup8(a));

        /* 7-th power */
        let [a7, b7] = Self::pow7_many([a, b]);

        let amix = Limbs32 {
            lo: _mm512_mask_blend_epi64(0x0f, a7.lo, asbox.lo),
            hi: _mm512_mask_blend_epi64(0x0f, a7.hi, asbox.hi),
        };
        (amix, b7)
    }

    /// Multiply the state, given in [limb form](Limbs32), by the MDS matrix
    /// and add the round constants. The result is canonical.
    #[inline]
    #[expect(clippy::identity_op)]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn mds_rcs_avx512(a: Limbs32, b: Limbs32, round_index: usize) -> (__m512i, __m512i) {
        const MDS_TRANS: [[u64; 8]; 16] = [
            [61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034],
            [56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845],
            [17845, 61402, 1108, 28750, 33823, 7454, 43244, 53865],
            [12034, 56951, 27521, 41351, 40901, 12021, 59689, 26798],
            [26798, 17845, 61402, 1108, 28750, 33823, 7454, 43244],
            [53865, 12034, 56951, 27521, 41351, 40901, 12021, 59689],
            [59689, 26798, 17845, 61402, 1108, 28750, 33823, 7454],
            [43244, 53865, 12034, 56951, 27521, 41351, 40901, 12021],
            [12021, 59689, 26798, 17845, 61402, 1108, 28750, 33823],
            [7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901],
            [40901, 12021, 59689, 26798, 17845, 61402, 1108, 28750],
            [33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351],
            [41351, 40901, 12021, 59689, 26798, 17845, 61402, 1108],
            [28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521],
            [27521, 41351, 40901, 12021, 59689, 26798, 17845, 61402],
            [1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951],
        ];
        /* Round Constants used to initialize 32-bit accumulators */
        let rc0lo = _mm512_loadu_epi64(
            RCS_MONT_L
                .as_ptr()
                .offset((round_index * 16 + 0).try_into().unwrap()) as *const i64,
        );
        let rc1lo = _mm512_loadu_epi64(
            RCS_MONT_L
                .as_ptr()
                .offset((round_index * 16 + 8).try_into().unwrap()) as *const i64,
        );
        let rc0hi = _mm512_loadu_epi64(
            RCS_MONT_U
                .as_ptr()
                .offset((round_index * 16 + 0).try_into().unwrap()) as *const i64,
        );
        let rc1hi = _mm512_loadu_epi64(
            RCS_MONT_U
                .as_ptr()
                .offset((round_index * 16 + 8).try_into().unwrap()) as *const i64,
        );

        /* Linear Diffusion */
        // Every output limb is the sum of 16 products. A single permutation
        // has little parallelism, so each sum is split over `NUM_ACCS`
        // independent accumulators to shorten the multiply-add chains.
        //
        // The limbs are broadcast with in-register permutations. Extracting
        // them one by one would go through general purpose registers, which
        // is slow.
        //
        // Every product is at most 16 + 33 bits wide, so the multiply-adds
        // are exact, and every output limb is at most 53 bits wide, as
        // [`Self::reduce2x32`] requires: the sum of 16 products is less than
        // 2^52 + 2^43, and the round constant's limb adds less than 2^32.
        const NUM_ACCS: usize = 4;
        let zero = _mm512_setzero_si512();
        let mut acc0lo = [zero; NUM_ACCS];
        let mut acc0hi = [zero; NUM_ACCS];
        let mut acc1lo = [zero; NUM_ACCS];
        let mut acc1hi = [zero; NUM_ACCS];
        acc0lo[0] = rc0lo;
        acc0hi[0] = rc0hi;
        acc1lo[0] = rc1lo;
        acc1hi[0] = rc1hi;
        for i in 0..8 {
            let c0 =
                _mm512_loadu_epi64(MDS_TRANS.as_ptr().offset(2 * i as isize + 0) as *const i64);
            let c1 =
                _mm512_loadu_epi64(MDS_TRANS.as_ptr().offset(2 * i as isize + 1) as *const i64);

            let lane = _mm512_set1_epi64(i as i64);
            let d0lo = _mm512_permutexvar_epi64(lane, a.lo);
            let d0hi = _mm512_permutexvar_epi64(lane, a.hi);
            let e0lo = _mm512_permutexvar_epi64(lane, b.lo);
            let e0hi = _mm512_permutexvar_epi64(lane, b.hi);

            let acc = i % NUM_ACCS;
            acc0lo[acc] = _mm512_madd52lo_epu64(acc0lo[acc], c0, d0lo);
            acc0hi[acc] = _mm512_madd52lo_epu64(acc0hi[acc], c0, d0hi);
            acc1lo[acc] = _mm512_madd52lo_epu64(acc1lo[acc], c1, d0lo);
            acc1hi[acc] = _mm512_madd52lo_epu64(acc1hi[acc], c1, d0hi);

            acc0lo[acc] = _mm512_madd52lo_epu64(acc0lo[acc], c1, e0lo);
            acc0hi[acc] = _mm512_madd52lo_epu64(acc0hi[acc], c1, e0hi);
            acc1lo[acc] = _mm512_madd52lo_epu64(acc1lo[acc], c0, e0lo);
            acc1hi[acc] = _mm512_madd52lo_epu64(acc1hi[acc], c0, e0hi);
        }
        let sum = |accs: [__m512i; NUM_ACCS]| {
            _mm512_add_epi64(
                _mm512_add_epi64(accs[0], accs[1]),
                _mm512_add_epi64(accs[2], accs[3]),
            )
        };
        let r0lo = sum(acc0lo);
        let r0hi = sum(acc0hi);
        let r1lo = sum(acc1lo);
        let r1hi = sum(acc1hi);

        (Self::reduce2x32(r0lo, r0hi), Self::reduce2x32(r1lo, r1hi))
    }

    /// Combine the two given limbs and fully reduce them modulo
    /// [`BFieldElement::P`]: the result is canonical, i.e., less than `P`.
    ///
    /// Each of the arguments must be at most 53 bits wide for this function to
    /// work correctly.
    //
    // The limbs are first combined into a single 64-bit value by
    // [`Self::pack`]; see there for the arithmetic. That value is congruent to
    // the input but might exceed `P`. Since it is smaller than 2^64 < 2·P,
    // subtracting `P` at most once completes the reduction.
    //
    // That subtraction is `min(r, r + ε)` with ε = 2^32 - 1: the sum r + ε
    // wraps around exactly if r ≥ 2^64 - ε = P, in which case it equals
    // r - P, which is smaller than r. Otherwise, r + ε > r and r stays. Note
    // that a value equal to `P` wraps around, too, and becomes zero: `P` is
    // the non-canonical representation of zero, and must not survive.
    //
    // The assumption that the arguments are at most 53 bits wide exists because
    // of the context this function is used in, in particular, it is (only) used
    // after MDS matrix multiplication and round constant addition. Each entry
    // in the MDS matrix has at most 16 bits, and is multiplied with one limb
    // of a state element, which is at most 32 bits wide, or marginally wider
    // (see [`Limbs32`]), resulting in a 48-bit intermediate result, or
    // marginally more. STATE_SIZE == 16 such intermediate results are summed
    // up, resulting in a 52-bit element, or marginally more. A 32-bit limb of
    // the round constant is added to get a limb of the final result, which is
    // at most 53 bits wide.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn reduce2x32(lo: __m512i, hi: __m512i) -> __m512i {
        // input must be at most 53 bits
        #[cfg(debug_assertions)]
        {
            let max = _mm512_set1_epi64((1_u64 << 53) as i64);
            let lo_lt_max = _mm512_cmplt_epu64_mask(lo, max);
            let hi_lt_max = _mm512_cmplt_epu64_mask(hi, max);
            debug_assert_eq!(0xff, lo_lt_max);
            debug_assert_eq!(0xff, hi_lt_max);
        }

        let epsilon = _mm512_set1_epi64(u32::MAX as i64);
        let r = Self::pack(lo, hi);
        let r = _mm512_min_epu64(r, _mm512_add_epi64(r, epsilon));

        #[cfg(debug_assertions)]
        {
            let p = _mm512_set1_epi64(BFieldElement::P as i64);
            let r_lt_p = _mm512_cmplt_epu64_mask(r, p);
            debug_assert_eq!(0xff, r_lt_p);
        }

        r
    }

    /// Apply the [lookup table](LOOKUP_TABLE) to every byte of the given
    /// vector.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn lookup8(x: __m512i) -> __m512i {
        let c64s = _mm512_set1_epi8(0x40);

        let s0 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x00) as *const i64);
        let s1 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x40) as *const i64);
        let s2 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x80) as *const i64);
        let s3 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0xc0) as *const i64);

        let i0 = x;
        let i1 = _mm512_sub_epi8(i0, c64s);
        let i2 = _mm512_sub_epi8(i1, c64s);
        let i3 = _mm512_sub_epi8(i2, c64s);

        let lt0 = _mm512_cmplt_epu8_mask(i0, c64s);
        let lt1 = _mm512_cmplt_epu8_mask(i1, c64s);
        let lt2 = _mm512_cmplt_epu8_mask(i2, c64s);
        let lt3 = _mm512_cmplt_epu8_mask(i3, c64s);

        let mut result = _mm512_setzero_si512();
        result = _mm512_mask_permutexvar_epi8(result, lt0, i0, s0);
        result = _mm512_mask_permutexvar_epi8(result, lt1, i1, s1);
        result = _mm512_mask_permutexvar_epi8(result, lt2, i2, s2);
        _mm512_mask_permutexvar_epi8(result, lt3, i3, s3)
    }

    /// The 7th power of every lane of `N` vectors, modulo
    /// [`BFieldElement::P`], in [limb form](Limbs32).
    ///
    /// The input lanes can be any 64-bit values; they are interpreted modulo
    /// `P`. The `N` vectors are independent; their instructions are
    /// interleaved to give the CPU independent work while it waits for the
    /// multiplications' results.
    //
    // No intermediate result is reduced any further than the next
    // multiplication requires; see [`Self::limbs`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn pow7_many<const N: usize>(xs: [__m512i; N]) -> [Limbs32; N] {
        let x1 = xs.map(|x| Self::split32(x));
        let x2 = x1.map(|x1| Self::square8(x1));
        let x3: [_; N] = std::array::from_fn(|i| Self::mul8(x1[i], x2[i]));
        let x4 = x2.map(|x2| Self::square8(x2));
        std::array::from_fn(|i| Self::mul8(x3[i], x4[i]))
    }

    /// Split every lane into its two 32-bit limbs; see [`Limbs32`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn split32(x: __m512i) -> Limbs32 {
        let u32_max = _mm512_set1_epi64(u32::MAX as i64);
        Limbs32 {
            lo: _mm512_and_epi64(x, u32_max),
            hi: _mm512_srli_epi64(x, 32),
        }
    }

    /// Multiply the lanes of `x` and `y` modulo [`BFieldElement::P`]. See
    /// [`Self::fold`] for the arithmetic.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn mul8(x: Limbs32, y: Limbs32) -> Limbs32 {
        x.debug_assert_in_range();
        y.debug_assert_in_range();

        let zero = _mm512_setzero_si512();
        let bias_lo = _mm512_set1_epi64(BIAS_LO as i64);
        let bias_hi = _mm512_set1_epi64(BIAS_HI as i64);

        let p00_lo = _mm512_madd52lo_epu64(bias_lo, x.lo, y.lo);
        let p00_hi = _mm512_madd52hi_epu64(zero, x.lo, y.lo);
        let p11_lo = _mm512_madd52lo_epu64(zero, x.hi, y.hi);
        let p11_hi = _mm512_madd52hi_epu64(zero, x.hi, y.hi);

        let mid_lo = _mm512_madd52lo_epu64(bias_hi, x.lo, y.hi);
        let mid_lo = _mm512_madd52lo_epu64(mid_lo, x.hi, y.lo);
        let mid_hi = _mm512_madd52hi_epu64(zero, x.lo, y.hi);
        let mid_hi = _mm512_madd52hi_epu64(mid_hi, x.hi, y.lo);

        let (l, h) = Self::fold(p00_lo, p00_hi, mid_lo, mid_hi, p11_lo, p11_hi);
        Self::limbs(l, h)
    }

    /// Square the lanes of `x` modulo [`BFieldElement::P`]. See
    /// [`Self::fold`] for the arithmetic.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn square8(x: Limbs32) -> Limbs32 {
        x.debug_assert_in_range();

        let zero = _mm512_setzero_si512();
        let bias_lo = _mm512_set1_epi64(BIAS_LO as i64);
        let bias_hi = _mm512_set1_epi64(BIAS_HI as i64);

        // The two cross products are equal. Doubling one operand (34 bits,
        // well within the multiplier's 52) computes their exact sum, split
        // consistently into its low 52 bits and the rest.
        let hi2 = _mm512_slli_epi64(x.hi, 1);

        let p00_lo = _mm512_madd52lo_epu64(bias_lo, x.lo, x.lo);
        let p00_hi = _mm512_madd52hi_epu64(zero, x.lo, x.lo);
        let mid_lo = _mm512_madd52lo_epu64(bias_hi, x.lo, hi2);
        let mid_hi = _mm512_madd52hi_epu64(zero, x.lo, hi2);
        let p11_lo = _mm512_madd52lo_epu64(zero, x.hi, x.hi);
        let p11_hi = _mm512_madd52hi_epu64(zero, x.hi, x.hi);

        let (l, h) = Self::fold(p00_lo, p00_hi, mid_lo, mid_hi, p11_lo, p11_hi);
        Self::limbs(l, h)
    }

    /// Reduce the partial products of a multiplication of two [`Limbs32`] to
    /// a pair `(l, h)` of lanes, each smaller than 2^54, such that the product
    /// is congruent to `l + h·2^32` modulo [`BFieldElement::P`].
    ///
    /// The pair is the input of [`Self::pack`] and [`Self::limbs`].
    //
    // With x = x_lo + x_hi·2^32 and y likewise, the product is
    //
    //   N = P00 + (P01 + P10)·2^32 + P11·2^64
    //
    // where P00 = x_lo·y_lo, P01 = x_lo·y_hi, P10 = x_hi·y_lo, P11 = x_hi·y_hi.
    // Each of these is smaller than 2^66 (see [`Limbs32`] for the limbs'
    // ranges); the multiplier splits them into their low 52 bits and the (at
    // most 14-bit) rest. The arguments are:
    //
    //   p00_lo + p00_hi·2^52 = P00,
    //   mid_lo + mid_hi·2^52 = P01 + P10 (sums of the split parts), and
    //   p11_lo + p11_hi·2^52 = P11,
    //
    // where p00_lo and p11_lo are less than 2^52, mid_lo is less than 2^53,
    // and p00_hi, p11_hi, and mid_hi are less than 2^14, 2^14, and 2^15,
    // respectively. Moreover, p00_lo includes BIAS_LO and mid_lo includes
    // BIAS_HI, see below.
    //
    // Using 2^52 = 2^20·2^32, 2^64 ≡ 2^32 - 1, 2^84 = 2^20·2^64, and
    // 2^116 = 2^20·2^96 ≡ -2^20 (all mod P), the product is congruent to
    // L + H·2^32 with
    //
    //   L = p00_lo - p11_lo - ((mid_hi + p11_hi) << 20)     and
    //   H = mid_lo + p11_lo + ((p00_hi + mid_hi) << 20).
    //
    // Both are computed in plain 64-bit arithmetic without any overflow or
    // borrow checks. This is sound because of the biases: the pair
    // (BIAS_LO, BIAS_HI) satisfies BIAS_LO + BIAS_HI·2^32 = P ≡ 0, so adding
    // them changes nothing modulo P, while BIAS_LO = 2^53 + 1 exceeds
    // everything subtracted from L (less than 2^52 + 2^36), and
    // BIAS_HI = 2^32 - 1 - 2^21 is positive. With all terms accounted for,
    //
    //   0 < L < 2^54      and      0 < H < 2^54.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn fold(
        p00_lo: __m512i,
        p00_hi: __m512i,
        mid_lo: __m512i,
        mid_hi: __m512i,
        p11_lo: __m512i,
        p11_hi: __m512i,
    ) -> (__m512i, __m512i) {
        let hi_to_l = _mm512_slli_epi64(_mm512_add_epi64(mid_hi, p11_hi), 20);
        let hi_to_h = _mm512_slli_epi64(_mm512_add_epi64(p00_hi, mid_hi), 20);
        let l = _mm512_sub_epi64(_mm512_sub_epi64(p00_lo, p11_lo), hi_to_l);
        let h = _mm512_add_epi64(_mm512_add_epi64(mid_lo, p11_lo), hi_to_h);

        #[cfg(debug_assertions)]
        {
            let max = _mm512_set1_epi64((1_u64 << 54) as i64);
            debug_assert_eq!(0xff, _mm512_cmplt_epu64_mask(l, max));
            debug_assert_eq!(0xff, _mm512_cmplt_epu64_mask(h, max));
        }

        (l, h)
    }

    /// Combine the two given lanes, `lo + hi·2^32`, into [`Limbs32`] that are
    /// congruent modulo [`BFieldElement::P`].
    ///
    /// Each of the arguments must be smaller than 2^54 for this function to
    /// work correctly. This is the cheaper sibling of [`Self::pack`]: nothing
    /// can overflow, so there is nothing to check or correct.
    //
    // With hi = hi_lo + hi_hi·2^32 and ε = 2^32 - 1 ≡ 2^64 (mod P):
    //
    //   lo + hi·2^32 ≡ lo + hi_hi·ε + hi_lo·2^32 = a + hi_lo·2^32
    //
    // where a = lo + hi_hi·ε < 2^55, see [`Self::pack`]. With
    // a = a_lo + a_hi·2^32, the result's limbs are a_lo < 2^32 and
    // a_hi + hi_lo < 2^23 + 2^32, within the ranges of [`Limbs32`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn limbs(lo: __m512i, hi: __m512i) -> Limbs32 {
        #[cfg(debug_assertions)]
        {
            let max = _mm512_set1_epi64((1_u64 << 54) as i64);
            debug_assert_eq!(0xff, _mm512_cmplt_epu64_mask(lo, max));
            debug_assert_eq!(0xff, _mm512_cmplt_epu64_mask(hi, max));
        }

        let u32_max = _mm512_set1_epi64(u32::MAX as i64);
        let epsilon = u32_max;

        let hi_hi = _mm512_srli_epi64(hi, 32);
        // `_mm512_mul_epu32` only reads the low 32 bits of each lane.
        let hi_hi_epsilon = _mm512_mul_epu32(hi_hi, epsilon);
        let a = _mm512_add_epi64(lo, hi_hi_epsilon);

        let hi_lo = _mm512_and_epi64(hi, u32_max);
        Limbs32 {
            lo: _mm512_and_epi64(a, u32_max),
            hi: _mm512_add_epi64(_mm512_srli_epi64(a, 32), hi_lo),
        }
    }

    /// Combine the two given limbs, `lo + hi·2^32`, into a single 64-bit
    /// value per lane that is congruent modulo [`BFieldElement::P`].
    ///
    /// Each of the arguments must be smaller than 2^54 for this function to
    /// work correctly. The result is smaller than 2^64 but not necessarily
    /// canonical.
    //
    // With hi = hi_lo + hi_hi·2^32 and ε = 2^32 - 1 ≡ 2^64 (mod P):
    //
    //   lo + hi·2^32 = lo + hi_lo·2^32 + hi_hi·2^64 ≡ lo + hi_hi·ε + hi_lo·2^32
    //
    // Since hi < 2^54, hi_hi < 2^22 and hi_hi·ε < 2^54, making
    // a = lo + hi_hi·ε smaller than 2^55. Adding hi_lo·2^32 < 2^64 overflows
    // at most once, by exactly 2^64 ≡ ε.
    //
    // The sum is computed, the overflow is detected by comparing with a
    // summand, and ε is added back. The corrected value is smaller than
    // 2^55 + ε, so that addition cannot overflow again.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn pack(lo: __m512i, hi: __m512i) -> __m512i {
        #[cfg(debug_assertions)]
        {
            let max = _mm512_set1_epi64((1_u64 << 54) as i64);
            debug_assert_eq!(0xff, _mm512_cmplt_epu64_mask(lo, max));
            debug_assert_eq!(0xff, _mm512_cmplt_epu64_mask(hi, max));
        }

        let epsilon = _mm512_set1_epi64(u32::MAX as i64);

        let hi_hi = _mm512_srli_epi64(hi, 32);
        // `_mm512_mul_epu32` only reads the low 32 bits of each lane.
        let hi_hi_epsilon = _mm512_mul_epu32(hi_hi, epsilon);
        let a = _mm512_add_epi64(lo, hi_hi_epsilon);

        let hi_lo_shl32 = _mm512_slli_epi64(hi, 32);
        let r = _mm512_add_epi64(a, hi_lo_shl32);
        let overflowed = _mm512_cmplt_epu64_mask(r, a);
        _mm512_mask_add_epi64(r, overflowed, r, epsilon)
    }
}

/// Eight field elements, each as a pair of limbs `lo + hi·2^32`, the operand
/// and result form of [`Tip5::mul8`] and [`Tip5::square8`].
///
/// The limbs are not fully normalized: `lo` is at most 32 bits wide, but `hi`
/// may be marginally wider, up to `2^32 + 2^23`. Everything that consumes
/// limbs is designed for these ranges: the multiplier's partial products stay
/// below 2^66, and the MDS layer's 64-bit accumulators keep enough headroom
/// (see [`Tip5::reduce2x32`]). The lazy upper bound is what makes the limbs
/// cheap to produce: see [`Tip5::limbs`].
///
/// [`Tip5::split32`] produces fully normalized limbs from any 64-bit lanes.
#[derive(Debug, Copy, Clone)]
struct Limbs32 {
    lo: __m512i,
    hi: __m512i,
}

/// The exclusive upper bound of [`Limbs32::hi`].
const LIMB_HI_MAX: u64 = (1 << 32) + (1 << 23);

// The analysis in `Tip5::fold` relies on all partial products being smaller
// than 2^66, i.e., on both limbs being smaller than 2^33.
const _: () = assert!(LIMB_HI_MAX <= 1 << 33);

impl Limbs32 {
    /// Check the limbs' ranges in debug builds.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn debug_assert_in_range(self) {
        #[cfg(debug_assertions)]
        {
            let lo_max = _mm512_set1_epi64(1 << 32);
            let hi_max = _mm512_set1_epi64(LIMB_HI_MAX as i64);
            debug_assert_eq!(0xff, _mm512_cmplt_epu64_mask(self.lo, lo_max));
            debug_assert_eq!(0xff, _mm512_cmplt_epu64_mask(self.hi, hi_max));
        }
    }
}

/// The bias added to the low limb in [`Tip5::fold`]. Together with
/// [`BIAS_HI`], it sums to [`BFieldElement::P`]: `BIAS_LO + BIAS_HI·2^32 == P`.
const BIAS_LO: u64 = (1 << 53) + 1;

/// The bias added to the high limb in [`Tip5::fold`]. See [`BIAS_LO`].
const BIAS_HI: u64 = (1 << 32) - 1 - (1 << 21);

const _: () = assert!(BIAS_LO as u128 + ((BIAS_HI as u128) << 32) == BFieldElement::P as u128);

/// Eight 64-bit integers, for use in [`Tip5::generated_function`].
///
/// The trait's functions can't have `#[target_feature]`s. Instead, they are
/// inlined into their callers, all of which enable the required ones. Values of
/// this private type are only created in such functions, which in turn may only
/// be called if the CPU supports the features.
#[derive(Debug, Copy, Clone)]
struct Lanes(__m512i);

impl WrappingArithmetic for Lanes {
    #[inline(always)]
    fn wrapping_add(self, rhs: Self) -> Self {
        // SAFETY: see the type's documentation
        Self(unsafe { _mm512_add_epi64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn wrapping_sub(self, rhs: Self) -> Self {
        // SAFETY: see the type's documentation
        Self(unsafe { _mm512_sub_epi64(self.0, rhs.0) })
    }

    #[inline(always)]
    fn wrapping_mul(self, rhs: u64) -> Self {
        // SAFETY: see the type's documentation
        Self(unsafe { _mm512_mullo_epi64(self.0, _mm512_set1_epi64(rhs as i64)) })
    }
}

/// The number of independent sponges processed side by side by the batched
/// functions: one per 64-bit lane of a 512-bit vector.
pub(super) const BATCH_SIZE: usize = 8;

/// The states of [`BATCH_SIZE`] independent Tip5 sponges, one per lane:
/// `state[i]` holds state element `i` of every sponge.
///
/// This layout is the transpose of [`Tip5`]'s: it is used to run the
/// permutation on many sponges at once, with every lane fully independent.
/// Compared to running one permutation at a time (see
/// [`Tip5::permutation_avx512`]), this keeps the S-box's power maps in many
/// independent dependency chains and never wastes lanes.
#[derive(Debug, Clone, Copy)]
#[repr(align(64))]
pub(super) struct Tip5Batch {
    state: [__m512i; STATE_SIZE],
}

#[expect(unsafe_op_in_unsafe_fn)]
impl Tip5Batch {
    /// All sponges in the given domain's initial state.
    ///
    /// # Safety
    ///
    /// See [`Tip5::permutation_avx512`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn new(domain: Domain) -> Self {
        let capacity = match domain {
            Domain::VariableLength => _mm512_setzero_si512(),
            Domain::FixedLength => _mm512_set1_epi64(BFieldElement::ONE.raw_u64() as i64),
        };
        let mut state = [_mm512_setzero_si512(); STATE_SIZE];
        for element in state.iter_mut().skip(RATE) {
            *element = capacity;
        }
        Self { state }
    }

    /// Set state element `index` of all sponges.
    ///
    /// # Safety
    ///
    /// See [`Tip5::permutation_avx512`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn set(&mut self, index: usize, elements: [BFieldElement; BATCH_SIZE]) {
        let raw = elements.map(|e| e.raw_u64() as i64);
        self.state[index] = _mm512_loadu_epi64(raw.as_ptr());
    }

    /// State element `index` of all sponges.
    ///
    /// # Safety
    ///
    /// See [`Tip5::permutation_avx512`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn get(&self, index: usize) -> [BFieldElement; BATCH_SIZE] {
        let mut raw = [0_i64; BATCH_SIZE];
        _mm512_storeu_epi64(raw.as_mut_ptr(), self.state[index]);
        raw.map(|e| BFieldElement::from_raw_u64(e as u64))
    }

    /// The Tip5 permutation on all sponges.
    ///
    /// # Safety
    ///
    /// See [`Tip5::permutation_avx512`].
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn permutation(&mut self) {
        for round_index in 0..NUM_ROUNDS {
            let limbs = self.sbox_layer();
            self.mds_rcs(limbs, round_index);
        }
    }

    /// The S-box layer. The state must be canonical; the result is in
    /// [limb form](Limbs32), which is what the MDS layer consumes.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn sbox_layer(&self) -> [Limbs32; STATE_SIZE] {
        // The power maps are computed several elements at a time, with their
        // instructions interleaved, so that the (long) dependency chains of
        // the multiplications overlap even on CPUs with small schedulers.
        // More interleaving than this exhausts the 32 vector registers.
        const INTERLEAVE: usize = 4;
        const NUM_POWER_MAPS: usize = STATE_SIZE - NUM_SPLIT_AND_LOOKUP;
        const NUM_CHUNKS: usize = NUM_POWER_MAPS / INTERLEAVE;
        const { assert!(NUM_CHUNKS * INTERLEAVE == NUM_POWER_MAPS) };

        let lookups: [_; NUM_SPLIT_AND_LOOKUP] =
            std::array::from_fn(|i| Tip5::split32(Tip5::lookup8(self.state[i])));
        let powers: [[_; INTERLEAVE]; NUM_CHUNKS] = std::array::from_fn(|chunk| {
            let first = NUM_SPLIT_AND_LOOKUP + chunk * INTERLEAVE;
            Tip5::pow7_many(std::array::from_fn(|j| self.state[first + j]))
        });
        std::array::from_fn(|i| match i.checked_sub(NUM_SPLIT_AND_LOOKUP) {
            None => lookups[i],
            Some(k) => powers[k / INTERLEAVE][k % INTERLEAVE],
        })
    }

    /// Multiply the states, given in [limb form](Limbs32), by the MDS matrix
    /// and add the round constants.
    ///
    /// For each limb, the product with the MDS matrix is computed exactly in
    /// 64 bits, using the same fast cyclic convolution as the scalar
    /// permutation. The round constant's limb is added; see
    /// [`Tip5::reduce2x32`] for the reduction, which makes the state
    /// canonical again.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn mds_rcs(&mut self, limbs: [Limbs32; STATE_SIZE], round_index: usize) {
        let lo = limbs.map(|limbs| Lanes(limbs.lo));
        let hi = limbs.map(|limbs| Lanes(limbs.hi));

        // The convolution is scaled by 16. Every entry of the MDS matrix has
        // at most 16 bits, making the unscaled limbs of the product at most 52
        // bits wide, or marginally more for the wider high limbs. Scaled, they
        // still fit into 64 bits, i.e., the wrapping arithmetic is exact.
        let lo = Tip5::generated_function(lo);
        let hi = Tip5::generated_function(hi);

        for r in 0..STATE_SIZE {
            let rc_index = round_index * STATE_SIZE + r;
            let rc_lo = _mm512_set1_epi64(RCS_MONT_L[rc_index] as i64);
            let rc_hi = _mm512_set1_epi64(RCS_MONT_U[rc_index] as i64);
            let acc_lo = _mm512_add_epi64(_mm512_srli_epi64(lo[r].0, 4), rc_lo);
            let acc_hi = _mm512_add_epi64(_mm512_srli_epi64(hi[r].0, 4), rc_hi);
            self.state[r] = Tip5::reduce2x32(acc_lo, acc_hi);
        }
    }
}

#[expect(unsafe_op_in_unsafe_fn)]
impl Tip5 {
    /// [`hash_varlen`](Self::hash_varlen) of [`BATCH_SIZE`] inputs of equal
    /// length at once.
    ///
    /// # Safety
    ///
    /// See [`Self::permutation_avx512`].
    ///
    /// # Panics
    ///
    /// Panics if the inputs differ in length.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn hash_varlen_batch(
        inputs: [&[BFieldElement]; BATCH_SIZE],
    ) -> [Digest; BATCH_SIZE] {
        let len = inputs[0].len();
        assert!(inputs.iter().all(|input| input.len() == len));

        let mut sponges = Tip5Batch::new(Domain::VariableLength);
        let num_full_chunks = len / RATE;
        for chunk in 0..num_full_chunks {
            for i in 0..RATE {
                sponges.set(i, inputs.map(|input| input[chunk * RATE + i]));
            }
            sponges.permutation();
        }

        // Pad input with [1, 0, 0, …] – padding is at least one element.
        let remainder_start = num_full_chunks * RATE;
        let remainder_len = len - remainder_start;
        for i in 0..RATE {
            let elements = match i.cmp(&remainder_len) {
                std::cmp::Ordering::Less => inputs.map(|input| input[remainder_start + i]),
                std::cmp::Ordering::Equal => [BFieldElement::ONE; BATCH_SIZE],
                std::cmp::Ordering::Greater => [BFieldElement::ZERO; BATCH_SIZE],
            };
            sponges.set(i, elements);
        }
        sponges.permutation();

        Self::digests(&sponges)
    }

    /// [`hash_pair`](Self::hash_pair) of [`BATCH_SIZE`] pairs at once.
    ///
    /// # Safety
    ///
    /// See [`Self::permutation_avx512`].
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn hash_pair_batch(
        pairs: &[[Digest; 2]; BATCH_SIZE],
    ) -> [Digest; BATCH_SIZE] {
        let mut sponges = Tip5Batch::new(Domain::FixedLength);
        for i in 0..Digest::LEN {
            sponges.set(i, pairs.map(|[left, _]| left.values()[i]));
            sponges.set(Digest::LEN + i, pairs.map(|[_, right]| right.values()[i]));
        }
        sponges.permutation();

        Self::digests(&sponges)
    }

    /// [`absorb`](crate::util_types::sponge::Sponge::absorb) on [`BATCH_SIZE`]
    /// sponges at once, each its own input.
    ///
    /// # Safety
    ///
    /// See [`Self::permutation_avx512`].
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn absorb_batch(
        sponges: &mut [Self; BATCH_SIZE],
        inputs: &[[BFieldElement; RATE]; BATCH_SIZE],
    ) {
        // Every state element is set below, so the domain does not matter.
        let mut batch = Tip5Batch::new(Domain::VariableLength);
        for i in 0..RATE {
            batch.set(i, inputs.map(|input| input[i]));
        }
        for i in RATE..STATE_SIZE {
            batch.set(i, std::array::from_fn(|j| sponges[j].state[i]));
        }
        batch.permutation();
        for i in 0..STATE_SIZE {
            for (sponge, element) in sponges.iter_mut().zip(batch.get(i)) {
                sponge.state[i] = element;
            }
        }
    }

    /// The first [`Digest::LEN`] state elements of every sponge.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn digests(sponges: &Tip5Batch) -> [Digest; BATCH_SIZE] {
        let mut digests = [[BFieldElement::ZERO; Digest::LEN]; BATCH_SIZE];
        for i in 0..Digest::LEN {
            for (digest, element) in digests.iter_mut().zip(sponges.get(i)) {
                digest[i] = element;
            }
        }
        digests.map(Digest::new)
    }
}

/// For every offset in `0..3`: the index vectors for two
/// [`_mm512_permutex2var_epi64`] steps that pick every third element out of
/// 24 contiguous ones (three vectors), starting at that offset.
///
/// The first step picks what it can from the first two vectors; the second
/// step keeps those and fills the rest from the third vector.
const EVERY_THIRD_INDICES: [([i64; BATCH_SIZE], [i64; BATCH_SIZE]); 3] = {
    let mut indices = [([0; BATCH_SIZE], [0; BATCH_SIZE]); 3];
    let mut offset = 0;
    while offset < 3 {
        let mut lane = 0;
        while lane < BATCH_SIZE {
            let source = offset + 3 * lane;
            if source < 2 * BATCH_SIZE {
                indices[offset].0[lane] = source as i64;
                indices[offset].1[lane] = lane as i64;
            } else {
                indices[offset].1[lane] = (source - BATCH_SIZE) as i64;
            }
            lane += 1;
        }
        offset += 1;
    }
    indices
};

#[expect(unsafe_op_in_unsafe_fn)]
impl Tip5Batch {
    /// Rows `first_row..first_row + BATCH_SIZE` of the column, one per lane.
    ///
    /// # Safety
    ///
    /// See [`Tip5::permutation_avx512`]. Additionally, the column must have at
    /// least `first_row + BATCH_SIZE` rows.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn load_column(column: Column, first_row: usize) -> __m512i {
        debug_assert!(first_row + BATCH_SIZE <= column.num_rows());
        let rows = column.elements.as_ptr().add(first_row * column.stride) as *const i64;
        match column.stride {
            1 => _mm512_loadu_epi64(rows),
            3 => {
                // The eight rows' elements span three vectors.
                let (indices_01, indices_2) = EVERY_THIRD_INDICES[column.offset];
                let v0 = _mm512_loadu_epi64(rows);
                let v1 = _mm512_loadu_epi64(rows.add(BATCH_SIZE));
                let v2 = _mm512_loadu_epi64(rows.add(2 * BATCH_SIZE));
                let v01 =
                    _mm512_permutex2var_epi64(v0, _mm512_loadu_epi64(indices_01.as_ptr()), v1);
                _mm512_permutex2var_epi64(v01, _mm512_loadu_epi64(indices_2.as_ptr()), v2)
            }
            _ => {
                let raw: [i64; BATCH_SIZE] =
                    std::array::from_fn(|lane| column.get(first_row + lane).raw_u64() as i64);
                _mm512_loadu_epi64(raw.as_ptr())
            }
        }
    }
}

/// The number of [batches](BATCH_SIZE) of rows that [`Tip5::hash_rows_block`]
/// hashes at once, i.e., 128 rows. Larger blocks read longer contiguous
/// pieces of every column, which helps the hardware prefetchers; the block's
/// sponge states (16 KiB) should still fit into the L1 cache.
pub(super) const ROW_BLOCK_BATCHES: usize = 16;

#[expect(unsafe_op_in_unsafe_fn)]
impl Tip5 {
    /// [`hash_varlen`](Self::hash_varlen) of rows `first_row..first_row +
    /// digests.len()` of a table given by its columns.
    ///
    /// The rows are hashed in batches of [`BATCH_SIZE`], all of which proceed
    /// through the columns in lockstep. This way, every column is read in one
    /// contiguous piece per block, which hardware prefetchers handle well.
    ///
    /// # Safety
    ///
    /// See [`Self::permutation_avx512`]. Additionally, every column must have
    /// at least `first_row + digests.len()` rows.
    ///
    /// # Panics
    ///
    /// Panics if the number of digests is not a multiple of [`BATCH_SIZE`] or
    /// exceeds [`ROW_BLOCK_BATCHES`] batches.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn hash_rows_block(
        columns: &[Column],
        first_row: usize,
        digests: &mut [Digest],
    ) {
        let (digest_batches, no_remainder) = digests.as_chunks_mut::<BATCH_SIZE>();
        assert!(no_remainder.is_empty());
        assert!(digest_batches.len() <= ROW_BLOCK_BATCHES);

        let mut sponges = [Tip5Batch::new(Domain::VariableLength); ROW_BLOCK_BATCHES];
        let sponges = &mut sponges[..digest_batches.len()];
        let batch_rows = |batch: usize| first_row + batch * BATCH_SIZE;

        let (full_chunks, remainder) = columns.as_chunks::<RATE>();
        for chunk in full_chunks {
            for (batch, sponge) in sponges.iter_mut().enumerate() {
                for (i, &column) in chunk.iter().enumerate() {
                    sponge.state[i] = Tip5Batch::load_column(column, batch_rows(batch));
                }
                sponge.permutation();
            }
        }

        // Pad input with [1, 0, 0, …] – padding is at least one element.
        for (batch, sponge) in sponges.iter_mut().enumerate() {
            for i in 0..RATE {
                sponge.state[i] = match remainder.get(i) {
                    Some(&column) => Tip5Batch::load_column(column, batch_rows(batch)),
                    None if i == remainder.len() => {
                        _mm512_set1_epi64(BFieldElement::ONE.raw_u64() as i64)
                    }
                    None => _mm512_setzero_si512(),
                };
            }
            sponge.permutation();
        }

        for (batch_digests, sponge) in digest_batches.iter_mut().zip(sponges.iter()) {
            *batch_digests = Self::digests(sponge);
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::super::naive::NaiveTip5;
    use super::*;

    const P: u64 = BFieldElement::P;

    /// Values at or next to the boundaries of the ranges the kernel relies
    /// on, including non-canonical representations (greater than or equal to
    /// `P`) that the multiplication routines accept as inputs.
    const BOUNDARY_VALUES: [u64; 23] = [
        0,
        1,
        2,
        3,
        7,
        (1 << 32) - 2,
        (1 << 32) - 1,
        1 << 32,
        (1 << 32) + 1,
        (1 << 48) - 1,
        1 << 48,
        (1 << 52) - 1,
        1 << 52,
        (1 << 53) - 1,
        1 << 53,
        (1 << 63) - 1,
        1 << 63,
        P - 2,
        P - 1,
        P,
        P + 1,
        u64::MAX - 1,
        u64::MAX,
    ];

    /// The canonical boundary values, i.e., valid raw state elements.
    fn canonical_boundary_values() -> Vec<u64> {
        BOUNDARY_VALUES.into_iter().filter(|&v| v < P).collect()
    }

    /// Pad with the default (zero) to a multiple of the batch size, then
    /// chunk into batches.
    fn batches<T: Copy + Default>(mut items: Vec<T>) -> Vec<[T; BATCH_SIZE]> {
        items.resize(items.len().next_multiple_of(BATCH_SIZE), T::default());
        let (batches, remainder) = items.as_chunks::<BATCH_SIZE>();
        assert!(remainder.is_empty());
        batches.to_vec()
    }

    fn mul_mod_p(x: u64, y: u64) -> u64 {
        ((u128::from(x) * u128::from(y)) % u128::from(P)) as u64
    }

    fn pow7_mod_p(x: u64) -> u64 {
        let x2 = mul_mod_p(x, x);
        let x4 = mul_mod_p(x2, x2);
        mul_mod_p(mul_mod_p(x, x2), x4)
    }

    /// Every pair of boundary values, in lanes of eight (padded with zeros).
    fn boundary_pairs() -> Vec<([u64; BATCH_SIZE], [u64; BATCH_SIZE])> {
        let mut pairs = vec![];
        for &x in &BOUNDARY_VALUES {
            for ys in BOUNDARY_VALUES.chunks(BATCH_SIZE) {
                let mut y = [0; BATCH_SIZE];
                y[..ys.len()].copy_from_slice(ys);
                pairs.push(([x; BATCH_SIZE], y));
            }
        }
        pairs
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn vector(lanes: [u64; BATCH_SIZE]) -> __m512i {
        unsafe { _mm512_loadu_epi64(lanes.map(|x| x as i64).as_ptr()) }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn lanes(vector: __m512i) -> [u64; BATCH_SIZE] {
        let mut raw = [0_i64; BATCH_SIZE];
        unsafe { _mm512_storeu_epi64(raw.as_mut_ptr(), vector) };
        raw.map(|x| x as u64)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn mul_square_pow7_on_boundary_values() {
        for (xs, ys) in boundary_pairs() {
            let x = unsafe { Tip5::split32(vector(xs)) };
            let y = unsafe { Tip5::split32(vector(ys)) };
            let products = unsafe { values(Tip5::mul8(x, y)) };
            let squares = unsafe { values(Tip5::square8(x)) };
            let powers = unsafe { values(Tip5::pow7_many([vector(xs)])[0]) };
            for lane in 0..BATCH_SIZE {
                let (xl, yl) = (xs[lane], ys[lane]);
                assert_eq!(mul_mod_p(xl, yl), products[lane], "{xl} * {yl}");
                assert_eq!(mul_mod_p(xl, xl), squares[lane], "{xl}^2");
                assert_eq!(pow7_mod_p(xl), powers[lane], "{xl}^7");
            }
        }
    }

    /// The limbs' values modulo `P`, after checking the limbs' ranges.
    #[target_feature(enable = "avx512f")]
    unsafe fn values(limbs: Limbs32) -> [u64; BATCH_SIZE] {
        let lo = unsafe { lanes(limbs.lo) };
        let hi = unsafe { lanes(limbs.hi) };
        std::array::from_fn(|lane| {
            assert!(lo[lane] <= u64::from(u32::MAX), "lo limb {}", lo[lane]);
            assert!(hi[lane] < LIMB_HI_MAX, "hi limb {}", hi[lane]);
            ((u128::from(lo[lane]) + (u128::from(hi[lane]) << 32)) % u128::from(P)) as u64
        })
    }

    /// Limb pairs at the boundaries of what [`Limbs32`] allows, including
    /// the lazy, wider-than-32-bit high limbs.
    fn boundary_limbs() -> Vec<(u64, u64)> {
        let los = [0, 1, (1 << 32) - 2, (1 << 32) - 1];
        let his = [
            0,
            1,
            (1 << 32) - 1,
            1 << 32,
            LIMB_HI_MAX - 2,
            LIMB_HI_MAX - 1,
        ];
        let mut limbs = vec![];
        for lo in los {
            for hi in his {
                limbs.push((lo, hi));
            }
        }
        limbs
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn mul_and_square_on_boundary_limbs() {
        let value = |(lo, hi): (u64, u64)| {
            ((u128::from(lo) + (u128::from(hi) << 32)) % u128::from(P)) as u64
        };
        let to_limbs = |limbs: [(u64, u64); BATCH_SIZE]| unsafe {
            Limbs32 {
                lo: vector(limbs.map(|(lo, _)| lo)),
                hi: vector(limbs.map(|(_, hi)| hi)),
            }
        };

        let all_limbs = boundary_limbs();
        for &x in &all_limbs {
            let xs = to_limbs([x; BATCH_SIZE]);
            let squares = unsafe { values(Tip5::square8(xs)) };
            assert_eq!(mul_mod_p(value(x), value(x)), squares[0], "{x:?}^2");

            for ys in batches(all_limbs.clone()) {
                let products = unsafe { values(Tip5::mul8(xs, to_limbs(ys))) };
                for (lane, &y) in ys.iter().enumerate() {
                    assert_eq!(
                        mul_mod_p(value(x), value(y)),
                        products[lane],
                        "{x:?} * {y:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn mul8_and_square8_are_correct_on_boundary_limbs() {
        if !Tip5::avx512_is_available() {
            return;
        }
        // SAFETY: The required CPU features were detected above.
        unsafe { mul_and_square_on_boundary_limbs() };
    }

    #[test]
    fn mul8_square8_and_pow7_are_correct_on_boundary_values() {
        if !Tip5::avx512_is_available() {
            return;
        }
        // SAFETY: The required CPU features were detected above.
        unsafe { mul_square_pow7_on_boundary_values() };
    }

    /// Lane pairs at the boundaries of what [`Tip5::reduce2x32`] accepts, and
    /// those whose combined value is a small multiple of `P`, or one off
    /// from one: the cases where a reduction that is merely congruent (for
    /// example, `P` instead of 0) would go unnoticed by arithmetic checks.
    fn boundary_lane_pairs() -> Vec<(u64, u64)> {
        const MAX_LIMB: u64 = (1 << 53) - 1;
        const EPSILON: u64 = (1 << 32) - 1;
        let limbs = [
            0,
            1,
            2,
            EPSILON - 1,
            EPSILON,
            EPSILON + 1,
            MAX_LIMB - 1,
            MAX_LIMB,
        ];
        let mut pairs = vec![];
        for lo in limbs {
            for hi in limbs {
                pairs.push((lo, hi));
            }
        }

        // k·P == k + k·ε·2^32, so (lo, hi) = (k, k·ε) combines to exactly k·P.
        // The largest k for which k·ε fits in 53 bits is 2^21.
        for k in [1, 2, 3, 2_u64.pow(20), 2_u64.pow(21) - 1, 2_u64.pow(21)] {
            let hi = k * EPSILON;
            pairs.push((k, hi)); // k·P
            pairs.push((k - 1, hi)); // k·P - 1
            pairs.push((k + 1, hi)); // k·P + 1
        }
        pairs
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn reduce2x32_is_canonical_on_boundary_limbs() {
        for pairs in batches(boundary_lane_pairs()) {
            let lo = unsafe { vector(pairs.map(|(lo, _)| lo)) };
            let hi = unsafe { vector(pairs.map(|(_, hi)| hi)) };
            let reduced = unsafe { lanes(Tip5::reduce2x32(lo, hi)) };
            let packed = unsafe { lanes(Tip5::pack(lo, hi)) };
            let limbs = unsafe { values(Tip5::limbs(lo, hi)) };
            for (lane, &(lo_lane, hi_lane)) in pairs.iter().enumerate() {
                let value = u128::from(lo_lane) + (u128::from(hi_lane) << 32);
                let expected = (value % u128::from(P)) as u64;
                assert_eq!(expected, reduced[lane], "reduce2x32({lo_lane}, {hi_lane})");
                assert_eq!(expected, packed[lane] % P, "pack({lo_lane}, {hi_lane})");
                assert_eq!(expected, limbs[lane], "limbs({lo_lane}, {hi_lane})");
            }
        }
    }

    #[test]
    fn reduce2x32_pack_and_limbs_are_correct_on_boundary_lanes() {
        if !Tip5::avx512_is_available() {
            return;
        }
        // SAFETY: The required CPU features were detected above.
        unsafe { reduce2x32_is_canonical_on_boundary_limbs() };
    }

    /// The MDS matrix's row `row`: the coefficient of every input element.
    fn mds_row(row: usize) -> [u64; STATE_SIZE] {
        use super::super::MDS_MATRIX_FIRST_COLUMN;
        std::array::from_fn(|col| {
            MDS_MATRIX_FIRST_COLUMN[(STATE_SIZE + row - col) % STATE_SIZE] as u64
        })
    }

    /// The MDS layer and round constant addition in plain integer arithmetic,
    /// on raw (Montgomery) values: for every row, the canonical result.
    fn mds_rcs_reference(limbs: [(u64, u64); STATE_SIZE], round_index: usize) -> [u64; STATE_SIZE] {
        std::array::from_fn(|row| {
            let rc_index = round_index * STATE_SIZE + row;
            let rc = u128::from(RCS_MONT_L[rc_index]) + (u128::from(RCS_MONT_U[rc_index]) << 32);
            let sum = mds_row(row)
                .iter()
                .zip(limbs)
                .map(|(&m, (lo, hi))| u128::from(m) * (u128::from(lo) + (u128::from(hi) << 32)))
                .sum::<u128>();
            ((sum + rc) % u128::from(P)) as u64
        })
    }

    /// Input limbs (one per element, all at most 32 bits wide) such that the
    /// MDS matrix's row `row` sums to exactly `target`, if the target is
    /// large enough: at least about 2^24, and at most about 2^50.
    ///
    /// Two columns with coprime coefficients make the sum exact; the others
    /// are filled greedily.
    fn limbs_summing_to(row: usize, target: u64) -> Option<[u64; STATE_SIZE]> {
        fn gcd(a: u64, b: u64) -> u64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        fn mod_inverse(a: u64, modulus: u64) -> u64 {
            (1..modulus).find(|&x| (a * x) % modulus == 1).unwrap()
        }

        let coeffs = mds_row(row);
        let (a, b) = (0..STATE_SIZE)
            .flat_map(|a| (a + 1..STATE_SIZE).map(move |b| (a, b)))
            .filter(|&(a, b)| gcd(coeffs[a], coeffs[b]) == 1)
            .min_by_key(|&(a, b)| coeffs[a] * coeffs[b])
            .expect("some pair of MDS entries is coprime");

        // With the coefficients c_a and c_b coprime, every remainder of at
        // least (c_a - 1)·(c_b - 1) is a non-negative combination of them.
        let frobenius = (coeffs[a] - 1) * (coeffs[b] - 1);
        if target < frobenius {
            return None;
        }
        let mut remaining = target;
        let mut limbs = [0; STATE_SIZE];
        for col in (0..STATE_SIZE).filter(|&col| col != a && col != b) {
            if remaining > frobenius {
                limbs[col] = ((remaining - frobenius) / coeffs[col]).min(u64::from(u32::MAX));
                remaining -= limbs[col] * coeffs[col];
            }
        }
        assert!(
            remaining >= frobenius,
            "target {target} is too small for row {row}"
        );
        assert!(
            remaining < frobenius + (1 << 16),
            "target {target} is too large for row {row}"
        );
        limbs[a] = (remaining * mod_inverse(coeffs[a] % coeffs[b], coeffs[b])) % coeffs[b];
        limbs[b] = (remaining - limbs[a] * coeffs[a]) / coeffs[b];

        let sum = coeffs.iter().zip(limbs).map(|(&c, l)| c * l).sum::<u64>();
        assert_eq!(target, sum);
        assert!(limbs.iter().all(|&l| l <= u64::from(u32::MAX)));
        Some(limbs)
    }

    /// MDS inputs, in limb form, that make row `row` of the MDS layer plus
    /// round constants come out as exactly `value`, before reduction: both of
    /// the row's limbs are controlled, so that the combined, unreduced result
    /// is `value` itself, not merely congruent to it.
    ///
    /// Fails (rarely) if the required high limb is too small for the matrix
    /// to produce.
    fn mds_inputs_hitting(
        row: usize,
        round_index: usize,
        value: u128,
    ) -> Option<[(u64, u64); STATE_SIZE]> {
        let rc_index = round_index * STATE_SIZE + row;
        let (rc_lo, rc_hi) = (RCS_MONT_L[rc_index], RCS_MONT_U[rc_index]);

        // Split the value into limbs. Moving one 2^32 from the high limb to
        // the low one keeps the latter's share of the sum comfortably large.
        let lo = (value & u128::from(u32::MAX)) as u64 + (1 << 32);
        let hi = (value >> 32) as u64 - 1;
        assert!(lo >= rc_lo && hi >= rc_hi);
        let conv_lo = limbs_summing_to(row, lo - rc_lo)?;
        let conv_hi = limbs_summing_to(row, hi - rc_hi)?;
        Some(std::array::from_fn(|col| (conv_lo[col], conv_hi[col])))
    }

    /// The values the MDS layer is made to hit: small multiples of `P`, and
    /// their neighbors. A multiple of `P` must come out as zero, not as `P`.
    fn mds_target_values() -> Vec<u128> {
        let mut values = vec![];
        for k in [1, 2, 3, 1000, 1 << 18] {
            let kp = k * u128::from(P);
            values.extend([kp - 1, kp, kp + 1]);
        }
        values
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vbmi,avx512ifma")]
    unsafe fn mds_layers_reduce_multiples_of_p_to_zero() {
        let mut num_traps = 0;
        for round_index in 0..NUM_ROUNDS {
            for row in [0, 5, 15] {
                for value in mds_target_values() {
                    let Some(inputs) = mds_inputs_hitting(row, round_index, value) else {
                        continue;
                    };
                    let expected = mds_rcs_reference(inputs, round_index);
                    assert_eq!((value % u128::from(P)) as u64, expected[row]);

                    // Make sure the trap is really sprung: for a multiple of
                    // `P`, the unreduced result must be `P` itself.
                    let rc_index = round_index * STATE_SIZE + row;
                    let unreduced = inputs
                        .iter()
                        .zip(mds_row(row))
                        .map(|(&(lo, hi), m)| {
                            (
                                u128::from(m) * u128::from(lo),
                                u128::from(m) * u128::from(hi),
                            )
                        })
                        .fold((0, 0), |(l, h), (dl, dh)| (l + dl, h + dh));
                    let lo = unsafe {
                        vector([(unreduced.0 as u64) + RCS_MONT_L[rc_index]; BATCH_SIZE])
                    };
                    let hi = unsafe {
                        vector([(unreduced.1 as u64) + RCS_MONT_U[rc_index]; BATCH_SIZE])
                    };
                    if value % u128::from(P) == 0 {
                        assert_eq!([P; BATCH_SIZE], unsafe { lanes(Tip5::pack(lo, hi)) });
                        num_traps += 1;
                    }

                    // The batched MDS layer.
                    let limbs = inputs.map(|(lo_limb, hi_limb)| unsafe {
                        Limbs32 {
                            lo: vector([lo_limb; BATCH_SIZE]),
                            hi: vector([hi_limb; BATCH_SIZE]),
                        }
                    });
                    let mut batch = unsafe { Tip5Batch::new(Domain::VariableLength) };
                    unsafe { batch.mds_rcs(limbs, round_index) };
                    for (i, &e) in expected.iter().enumerate() {
                        let raw = unsafe { batch.get(i) }.map(|x| x.raw_u64());
                        assert_eq!(
                            [e; BATCH_SIZE], raw,
                            "batched, round {round_index}, element {i}"
                        );
                    }

                    // The single permutation's MDS layer.
                    let a = Limbs32 {
                        lo: unsafe { vector(std::array::from_fn(|i| inputs[i].0)) },
                        hi: unsafe { vector(std::array::from_fn(|i| inputs[i].1)) },
                    };
                    let b = Limbs32 {
                        lo: unsafe { vector(std::array::from_fn(|i| inputs[BATCH_SIZE + i].0)) },
                        hi: unsafe { vector(std::array::from_fn(|i| inputs[BATCH_SIZE + i].1)) },
                    };
                    let (a, b) = unsafe { Tip5::mds_rcs_avx512(a, b, round_index) };
                    let single = [unsafe { lanes(a) }, unsafe { lanes(b) }].concat();
                    assert_eq!(expected.to_vec(), single, "single, round {round_index}");
                }
            }
        }
        // Almost every case is constructible; make sure enough of them were.
        assert!(
            num_traps >= NUM_ROUNDS * 3 * 5 - 5,
            "only {num_traps} traps sprung"
        );
    }

    #[test]
    fn mds_layers_are_canonical_on_multiples_of_p() {
        if !Tip5::avx512_is_available() {
            return;
        }
        // SAFETY: The required CPU features were detected above.
        unsafe { mds_layers_reduce_multiples_of_p_to_zero() };
    }

    /// States made of boundary values: constant states, and every rotation
    /// of the list of canonical boundary values, so that each one sits in
    /// every position, i.e., goes through both kinds of S-box.
    fn boundary_states() -> Vec<[BFieldElement; STATE_SIZE]> {
        let values = canonical_boundary_values();
        let mut states = values
            .iter()
            .map(|&v| [BFieldElement::from_raw_u64(v); STATE_SIZE])
            .collect::<Vec<_>>();
        for rotation in 0..values.len() {
            states.push(std::array::from_fn(|i| {
                BFieldElement::from_raw_u64(values[(i + rotation) % values.len()])
            }));
        }
        states
    }

    #[test]
    fn avx512_permutations_agree_with_naive_permutation_on_boundary_states() {
        if !Tip5::avx512_is_available() {
            return;
        }

        for states in batches(boundary_states()) {
            // SAFETY (all `unsafe` blocks): The required CPU features were
            // detected above.
            let mut batch = unsafe { Tip5Batch::new(Domain::VariableLength) };
            for i in 0..STATE_SIZE {
                unsafe { batch.set(i, states.map(|state| state[i])) };
            }
            unsafe { batch.permutation() };

            for (lane, &state) in states.iter().enumerate() {
                let mut naive = NaiveTip5 { state };
                naive.permutation();

                let mut single = Tip5 { state };
                unsafe { single.permutation_avx512() };
                assert_eq!(naive.state, single.state, "single, lane {lane}");

                for i in 0..STATE_SIZE {
                    let batched = unsafe { batch.get(i) }[lane];
                    assert_eq!(naive.state[i], batched, "batched, lane {lane}, element {i}");
                }
            }
        }
    }
}
