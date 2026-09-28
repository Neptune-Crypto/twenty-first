//! Make [`Tip5`] even faster by using SIMD, in particular, AVX-512.
//!
//! The functions in this module are compiled for every x86-64 target, but
//! must only be called after checking [`Tip5::avx512_is_available`] at
//! runtime. This way, binaries that are not compiled with AVX-512 enabled
//! still benefit on CPUs that support it.

use std::arch::x86_64::*;

use num_traits::ConstOne;
use num_traits::ConstZero;

use super::Digest;
use super::LOOKUP_TABLE;
use super::NUM_ROUNDS;
use super::NUM_SPLIT_AND_LOOKUP;
use super::RATE;
use super::STATE_SIZE;
use super::Tip5;
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
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512ifma")
    }

    /// The Tip5 permutation, using AVX-512.
    ///
    /// # Safety
    ///
    /// The CPU must support the AVX-512 extensions “f”, “bw”, “vbmi”, and
    /// “ifma”; see [`Self::avx512_is_available`].
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn permutation_avx512(&mut self) {
        for round_index in 0..NUM_ROUNDS {
            self.round_avx512(round_index);
        }
    }

    /// One round of the Tip5 permutation, using AVX-512.
    ///
    /// # Safety
    ///
    /// See [`Self::permutation_avx512`].
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn round_avx512(&mut self, round_index: usize) {
        Self::sbox_layer_avx512(&mut self.state);
        Self::mds_rcs_avx512(&mut self.state, round_index);
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    unsafe fn sbox_layer_avx512(state: &mut [BFieldElement; STATE_SIZE]) {
        let a = _mm512_load_epi64(state.as_mut_ptr().offset(0x00) as *mut i64);
        let b = _mm512_load_epi64(state.as_mut_ptr().offset(0x08) as *mut i64);

        /* S-BOX */
        let asbox = Self::lookup8(a);

        /* 7-th power */
        let a1 = a;
        let b1 = b;

        let a2 = Self::square8(a1);
        let b2 = Self::square8(b1);

        let a4 = Self::square8(a2);
        let b4 = Self::square8(b2);

        let a7 = Self::mul8(Self::mul8(a1, a2), a4);
        let b7 = Self::mul8(Self::mul8(b1, b2), b4);

        let amix = _mm512_mask_blend_epi64(0x0f, a7, asbox);

        _mm512_store_epi64(state.as_mut_ptr().offset(0x00) as *mut i64, amix);
        _mm512_store_epi64(state.as_mut_ptr().offset(0x08) as *mut i64, b7);
    }

    #[inline]
    #[expect(clippy::identity_op)]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    unsafe fn mds_rcs_avx512(state: &mut [BFieldElement; STATE_SIZE], round_index: usize) {
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
        union Vec512 {
            vector: __m512i,
            vals32: [u32; 16],
        }

        let a = _mm512_load_epi64(state.as_ptr().offset(0x00) as *const i64);
        let b = _mm512_load_epi64(state.as_ptr().offset(0x08) as *const i64);

        /* Round Constants used to initialize 32-bit accumulators */
        let mut r0lo = _mm512_loadu_epi64(
            RCS_MONT_L
                .as_ptr()
                .offset((round_index * 16 + 0).try_into().unwrap()) as *const i64,
        );
        let mut r1lo = _mm512_loadu_epi64(
            RCS_MONT_L
                .as_ptr()
                .offset((round_index * 16 + 8).try_into().unwrap()) as *const i64,
        );
        let mut r0hi = _mm512_loadu_epi64(
            RCS_MONT_U
                .as_ptr()
                .offset((round_index * 16 + 0).try_into().unwrap()) as *const i64,
        );
        let mut r1hi = _mm512_loadu_epi64(
            RCS_MONT_U
                .as_ptr()
                .offset((round_index * 16 + 8).try_into().unwrap()) as *const i64,
        );

        /* Linear Diffusion */
        for i in 0..8 {
            let c0 = _mm512_loadu_epi64(MDS_TRANS.as_ptr().offset(2 * i + 0) as *const i64);
            let c1 = _mm512_loadu_epi64(MDS_TRANS.as_ptr().offset(2 * i + 1) as *const i64);

            let d0lo = _mm512_set1_epi64(Vec512 { vector: a }.vals32[(2 * i + 0) as usize].into());
            let d0hi = _mm512_set1_epi64(Vec512 { vector: a }.vals32[(2 * i + 1) as usize].into());
            let e0lo = _mm512_set1_epi64(Vec512 { vector: b }.vals32[(2 * i + 0) as usize].into());
            let e0hi = _mm512_set1_epi64(Vec512 { vector: b }.vals32[(2 * i + 1) as usize].into());

            r0lo = _mm512_madd52lo_epu64(r0lo, c0, d0lo);
            r0hi = _mm512_madd52lo_epu64(r0hi, c0, d0hi);
            r1lo = _mm512_madd52lo_epu64(r1lo, c1, d0lo);
            r1hi = _mm512_madd52lo_epu64(r1hi, c1, d0hi);

            r0lo = _mm512_madd52lo_epu64(r0lo, c1, e0lo);
            r0hi = _mm512_madd52lo_epu64(r0hi, c1, e0hi);
            r1lo = _mm512_madd52lo_epu64(r1lo, c0, e0lo);
            r1hi = _mm512_madd52lo_epu64(r1hi, c0, e0hi);
        }

        _mm512_store_epi64(
            state.as_mut_ptr().offset(0x00) as *mut i64,
            Self::reduce2x32(r0lo, r0hi),
        );
        _mm512_store_epi64(
            state.as_mut_ptr().offset(0x08) as *mut i64,
            Self::reduce2x32(r1lo, r1hi),
        );
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    unsafe fn reduce3x48(ain: __m512i, bin: __m512i, cin: __m512i) -> __m512i {
        /* Combine and reduce a * 2**0 + b * 2**48 + c * 2**96 to F_P */

        let mask32 = _mm512_set1_epi64(0xffffffff);
        let mask48 = _mm512_set1_epi64(0xffffffffffff);

        let mut a = ain;
        let mut b = bin;
        let mut c = cin;

        /* Propagate carries */
        let ova = _mm512_srli_epi64(a, 48); // 1c/1.0c
        let ovb = _mm512_srli_epi64(b, 48); // 1c/1.0c

        b = _mm512_add_epi64(b, ova); // 1c/1.0c
        c = _mm512_add_epi64(c, ovb); // 1c/1.0c

        a = _mm512_and_epi64(a, mask48); // 1c/0.5c
        b = _mm512_and_epi64(b, mask48); // 1c/0.5c

        /* mod reduce */
        let abhi = _mm512_slli_epi64(b, 48); // 1c/1.0c
        let ab = _mm512_or_epi64(a, abhi); // 1c/0.5c

        let tmp0 = _mm512_sub_epi64(ab, c); // 1c/0.5c
        let mut ov = _mm512_cmp_epu64_mask(ab, tmp0, 1 /* lt */); // 3c/1.0c

        let mut tmp2 = _mm512_srli_epi64(b, 16); // 1c/1.0c
        let tmp3 = tmp2; //_mm512_and_epi64(tmp2, mask32);      // 1c/0.5c

        tmp2 = _mm512_slli_epi64(tmp2, 32); // 1c/1.0c
        tmp2 = _mm512_sub_epi64(tmp2, tmp3); // 1c/0.5c

        let tmp1 = _mm512_mask_sub_epi64(tmp0, ov, tmp0, mask32); // 1c/0.5c

        let r = _mm512_add_epi64(tmp1, tmp2); // 1c/0.5c
        ov = _mm512_cmp_epu64_mask(r, tmp1, 1 /* lt */); // 3c/1.0c

        _mm512_mask_add_epi64(r, ov, r, mask32) // 1c/0.5c
    }

    /// Combine and reduce the two given limbs modulo [BFieldElement::P].
    ///
    /// Each of the arguments must be at most 53 bits wide for this function to
    /// work correctly.
    //
    // This function uses the following equality:
    //
    //   x₂·2^64 + x₁·2^32 + x₀        | uses 2^64 == 2^32 - 1 (mod p)
    // =    (x₁ + x₂)·2^32 + x₀ - x₂     (mod p)
    //
    // Any given lane in the input is interpreted as follows:
    //
    //        ╭╴ lo ╶╮
    //  ╭╴ hi ╶╮
    //  x₂    x₁    x₀
    //
    // That is:
    // - The low 32 bits of the `lo` limb equal x₀.
    // - The sum of the high 32 bits of the `lo` limb and the low 32 bits of
    //   the `hi` limb equals x₁.
    // - The high 32 bits of the `hi` limb plus the carry from the previous sum
    //   equals x₂.
    //
    // This function uses the assumption that the input limbs are at most
    // 53 bits wide. This means that adding the (at most) 32-bit wide x₁ to the
    // (at most) 21-bit wide x₂, the result might be 33 bits wide. Therefore,
    // left-shifting (x₁ + x₂) by 32 bits cannot generally be stored in a u64
    // without loss. Under the current assumptions, this overflow is at most
    // 1 bit and is handled explicitly. The same assumption also implies that
    // the value (x₁ + x₂)·2^32 + x₀ - x₂ is strictly smaller than 2·p, i.e.,
    // subtracting p at most once completes modular reduction.
    //
    // Should the assumption that `lo` and `hi` are at most 53 bits wide change,
    // the above conclusions might not hold anymore, and the function must be
    // changed accordingly.
    //
    // The assumption that the arguments are at most 53 bits wide exists because
    // of the context this function is used in, in particular, it is (only) used
    // after MDS matrix multiplication and round constant addition. Each entry
    // in the MDS matrix has at most 16 bits, and is multiplied with one 32-bit
    // limb of a state element, resulting in a 48-bit intermediate result.
    // STATE_SIZE == 16 such intermediate results are summed up, resulting in a
    // 52-bit element. A 32-bit limb of the round constant is added to get a
    // limb of the final result, which is at most 53 bits wide.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
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

        let u32_max = _mm512_set1_epi64(u32::MAX as i64);

        let x0 = _mm512_and_epi64(lo, u32_max);

        let lo_shr32 = _mm512_srli_epi64(lo, 32);
        let x_tmp = _mm512_add_epi64(lo_shr32, hi);
        let x1 = _mm512_and_epi64(x_tmp, u32_max);
        let x2 = _mm512_srli_epi64(x_tmp, 32); // at most 21 bits (per lane)

        // r = ((x₁ + x₂) << 32) + x0 - x2
        let x1_plus_x2 = _mm512_add_epi64(x1, x2);
        let x1_plus_x2_shl32 = _mm512_slli_epi64(x1_plus_x2, 32);
        let x1_plus_x2_shl32_plus_x0 = _mm512_add_epi64(x1_plus_x2_shl32, x0);
        let r = _mm512_sub_epi64(x1_plus_x2_shl32_plus_x0, x2);

        // To guarantee a result that is less than p, subtract p if (any of):
        // - r >= p
        // - (x₁ + x₂) << 32 would overflow a u64, i.e., if (x₁ + x₂) > u32::MAX
        let p = _mm512_set1_epi64(BFieldElement::P as i64);
        let r_ge_p = _mm512_cmpge_epu64_mask(r, p);
        let x1_p_x2_gt_u32 = _mm512_cmpgt_epu64_mask(x1_plus_x2, u32_max);
        let ov_mask = r_ge_p | x1_p_x2_gt_u32;
        let r = _mm512_mask_sub_epi64(r, ov_mask, r, p);

        #[cfg(debug_assertions)]
        {
            let r_lt_p = _mm512_cmplt_epu64_mask(r, p);
            debug_assert_eq!(0xff, r_lt_p);
        }

        r
    }

    /// Apply the [lookup table](LOOKUP_TABLE) to every byte of the given
    /// vector.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
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

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    unsafe fn mul8(x: __m512i, y: __m512i) -> __m512i {
        let mask48 = _mm512_set1_epi64(0xffffffffffff);

        let mut a_0 = _mm512_setzero_si512();
        let mut b_0 = _mm512_setzero_si512();
        let mut b_4 = _mm512_setzero_si512();
        let mut c_0 = _mm512_setzero_si512();
        let mut c_4 = _mm512_setzero_si512();

        let xhi = _mm512_srli_epi64(x, 48);
        let yhi = _mm512_srli_epi64(y, 48);

        let xlo = _mm512_and_epi64(x, mask48);
        let ylo = _mm512_and_epi64(y, mask48);

        a_0 = _mm512_madd52lo_epu64(a_0, xlo, ylo);

        b_0 = _mm512_madd52lo_epu64(b_0, xhi, ylo);
        b_0 = _mm512_madd52lo_epu64(b_0, xlo, yhi);
        b_4 = _mm512_madd52hi_epu64(b_4, xlo, ylo);

        c_0 = _mm512_madd52lo_epu64(c_0, xhi, yhi);
        c_4 = _mm512_madd52hi_epu64(c_4, xhi, ylo);
        c_4 = _mm512_madd52hi_epu64(c_4, xlo, yhi);

        b_4 = _mm512_slli_epi64(b_4, 4);
        c_4 = _mm512_slli_epi64(c_4, 4);

        b_0 = _mm512_add_epi64(b_0, b_4);
        c_0 = _mm512_add_epi64(c_0, c_4);

        Self::reduce3x48(a_0, b_0, c_0)
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    unsafe fn square8(x: __m512i) -> __m512i {
        let mask48 = _mm512_set1_epi64(0xffffffffffff);

        let mut a_0 = _mm512_setzero_si512();
        let mut b_1 = _mm512_setzero_si512();
        let mut b_4 = _mm512_setzero_si512();
        let mut c_0 = _mm512_setzero_si512();
        let mut c_5 = _mm512_setzero_si512();

        let xhi = _mm512_srli_epi64(x, 48);
        let xlo = _mm512_and_epi64(x, mask48);

        a_0 = _mm512_madd52lo_epu64(a_0, xlo, xlo);

        b_1 = _mm512_madd52lo_epu64(b_1, xhi, xlo);
        b_4 = _mm512_madd52hi_epu64(b_4, xlo, xlo);

        c_0 = _mm512_madd52lo_epu64(c_0, xhi, xhi);
        c_5 = _mm512_madd52hi_epu64(c_5, xhi, xlo);

        b_1 = _mm512_slli_epi64(b_1, 1);
        b_4 = _mm512_slli_epi64(b_4, 4);
        c_5 = _mm512_slli_epi64(c_5, 5);

        let b_0 = _mm512_add_epi64(b_1, b_4);
        c_0 = _mm512_add_epi64(c_0, c_5);

        Self::reduce3x48(a_0, b_0, c_0)
    }
}

/// The (circulant) MDS matrix: `MDS_MATRIX[r][c]` is the entry in row `r`
/// and column `c`.
const MDS_MATRIX: [[u64; STATE_SIZE]; STATE_SIZE] = {
    let mut matrix = [[0; STATE_SIZE]; STATE_SIZE];
    let mut r = 0;
    while r < STATE_SIZE {
        let mut c = 0;
        while c < STATE_SIZE {
            matrix[r][c] = super::MDS_MATRIX_FIRST_COLUMN[(r + STATE_SIZE - c) % STATE_SIZE] as u64;
            c += 1;
        }
        r += 1;
    }
    matrix
};

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
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
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
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
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
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
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
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    pub(super) unsafe fn permutation(&mut self) {
        for round_index in 0..NUM_ROUNDS {
            self.sbox_layer();
            self.mds_rcs(round_index);
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    unsafe fn sbox_layer(&mut self) {
        for element in self.state.iter_mut().take(NUM_SPLIT_AND_LOOKUP) {
            *element = Tip5::lookup8(*element);
        }
        for element in self.state.iter_mut().skip(NUM_SPLIT_AND_LOOKUP) {
            let x1 = *element;
            let x2 = Tip5::square8(x1);
            let x4 = Tip5::square8(x2);
            *element = Tip5::mul8(Tip5::mul8(x1, x2), x4);
        }
    }

    /// Multiply the states by the MDS matrix and add the round constants.
    ///
    /// Every state element is split into its two 32-bit limbs, and every
    /// output limb accumulates the 16 products of an MDS row with the input
    /// limbs, starting from the round constant's limb. The accumulation is
    /// exact in 64 bits; see [`Tip5::reduce2x32`] for the reduction.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
    unsafe fn mds_rcs(&mut self, round_index: usize) {
        let u32_max = _mm512_set1_epi64(u32::MAX as i64);
        let mut lo = [_mm512_setzero_si512(); STATE_SIZE];
        let mut hi = [_mm512_setzero_si512(); STATE_SIZE];
        for c in 0..STATE_SIZE {
            lo[c] = _mm512_and_epi64(self.state[c], u32_max);
            hi[c] = _mm512_srli_epi64(self.state[c], 32);
        }

        // Half of the outputs at a time, to keep the accumulators in
        // registers.
        const HALF: usize = STATE_SIZE / 2;
        for half in 0..2 {
            let mut acc_lo = [_mm512_setzero_si512(); HALF];
            let mut acc_hi = [_mm512_setzero_si512(); HALF];
            for (i, r) in (half * HALF..(half + 1) * HALF).enumerate() {
                let rc_index = round_index * STATE_SIZE + r;
                acc_lo[i] = _mm512_set1_epi64(RCS_MONT_L[rc_index] as i64);
                acc_hi[i] = _mm512_set1_epi64(RCS_MONT_U[rc_index] as i64);
            }
            for c in 0..STATE_SIZE {
                for (i, r) in (half * HALF..(half + 1) * HALF).enumerate() {
                    let m = _mm512_set1_epi64(MDS_MATRIX[r][c] as i64);
                    acc_lo[i] = _mm512_madd52lo_epu64(acc_lo[i], m, lo[c]);
                    acc_hi[i] = _mm512_madd52lo_epu64(acc_hi[i], m, hi[c]);
                }
            }
            for (i, r) in (half * HALF..(half + 1) * HALF).enumerate() {
                self.state[r] = Tip5::reduce2x32(acc_lo[i], acc_hi[i]);
            }
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
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
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
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
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

    /// The first [`Digest::LEN`] state elements of every sponge.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512ifma")]
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
