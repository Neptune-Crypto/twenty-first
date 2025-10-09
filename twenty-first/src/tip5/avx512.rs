//! Make [`Tip5`] even faster by using SIMD, in particular, AVX-512.

use std::arch::x86_64::*;

use super::LOOKUP_TABLE;
use super::STATE_SIZE;
use super::Tip5;
use crate::prelude::BFieldElement;

#[expect(unsafe_op_in_unsafe_fn)]
impl Tip5 {
    #[inline(always)]
    pub fn round(&mut self, round_index: usize) {
        unsafe {
            Self::sbox_layer_avx512(&mut self.state);
            Self::mds_rcs_avx512(&mut self.state, round_index);
        }
    }

    #[inline(always)]
    pub fn round_x2(sponge0: &mut Tip5, sponge1: &mut Tip5, round_index: usize) {
        unsafe {
            Self::sbox_layer_x2_avx512(&mut sponge0.state, &mut sponge1.state);
            Self::mds_rcs_avx512(&mut sponge0.state, round_index);
            Self::mds_rcs_avx512(&mut sponge1.state, round_index);
        }
    }

    unsafe fn sbox_layer_avx512(state: &mut [BFieldElement; STATE_SIZE]) {
        let a = _mm512_load_epi64(state.as_mut_ptr().offset(0x00) as *mut i64);
        let b = _mm512_load_epi64(state.as_mut_ptr().offset(0x08) as *mut i64);

        /* S-BOX */
        let mut asbox = _mm512_setzero_si512();
        let c64s = _mm512_set1_epi8(0x40);

        let s0 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x00) as *const i64);
        let s1 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x40) as *const i64);
        let s2 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x80) as *const i64);
        let s3 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0xc0) as *const i64);

        let i0 = a;
        let i1 = _mm512_sub_epi8(i0, c64s);
        let i2 = _mm512_sub_epi8(i1, c64s);
        let i3 = _mm512_sub_epi8(i2, c64s);

        let lt0 = _mm512_cmplt_epu8_mask(i0, c64s);
        let lt1 = _mm512_cmplt_epu8_mask(i1, c64s);
        let lt2 = _mm512_cmplt_epu8_mask(i2, c64s);
        let lt3 = _mm512_cmplt_epu8_mask(i3, c64s);

        asbox = _mm512_mask_permutexvar_epi8(asbox, lt0, i0, s0);
        asbox = _mm512_mask_permutexvar_epi8(asbox, lt1, i1, s1);
        asbox = _mm512_mask_permutexvar_epi8(asbox, lt2, i2, s2);
        asbox = _mm512_mask_permutexvar_epi8(asbox, lt3, i3, s3);

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

    #[inline(always)]
    unsafe fn sbox_layer_x2_avx512(
        state0: &mut [BFieldElement; STATE_SIZE],
        state1: &mut [BFieldElement; STATE_SIZE],
    ) {
        let s0a = _mm512_load_epi64(state0.as_mut_ptr().offset(0x00) as *mut i64);
        let s0b = _mm512_load_epi64(state0.as_mut_ptr().offset(0x08) as *mut i64);

        let s1a = _mm512_load_epi64(state1.as_mut_ptr().offset(0x00) as *mut i64);
        let s1b = _mm512_load_epi64(state1.as_mut_ptr().offset(0x08) as *mut i64);

        /* S-BOX */
        let mut asbox = _mm512_setzero_si512();
        let c64s = _mm512_set1_epi8(0x40);

        let s0 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x00) as *const i64);
        let s1 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x40) as *const i64);
        let s2 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0x80) as *const i64);
        let s3 = _mm512_loadu_epi64(LOOKUP_TABLE.as_ptr().offset(0xc0) as *const i64);

        /* re-combine elements from state0 and state1 for subst/exp */
        let idx = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        let a1rev = _mm512_permutexvar_epi64(idx, s1a);
        let atosub = _mm512_mask_blend_epi64(0xf0, s0a, a1rev);
        let atoexp = _mm512_mask_blend_epi64(0x0f, s0a, a1rev);

        let i0 = atosub;
        let i1 = _mm512_sub_epi8(i0, c64s);
        let i2 = _mm512_sub_epi8(i1, c64s);
        let i3 = _mm512_sub_epi8(i2, c64s);

        let lt0 = _mm512_cmplt_epu8_mask(i0, c64s);
        let lt1 = _mm512_cmplt_epu8_mask(i1, c64s);
        let lt2 = _mm512_cmplt_epu8_mask(i2, c64s);
        let lt3 = _mm512_cmplt_epu8_mask(i3, c64s);

        asbox = _mm512_mask_permutexvar_epi8(asbox, lt0, i0, s0);
        asbox = _mm512_mask_permutexvar_epi8(asbox, lt1, i1, s1);
        asbox = _mm512_mask_permutexvar_epi8(asbox, lt2, i2, s2);
        asbox = _mm512_mask_permutexvar_epi8(asbox, lt3, i3, s3);

        let asboxrev = _mm512_permutexvar_epi64(idx, asbox);

        /* 7-th power */
        let a1 = atoexp;
        let b1 = s0b;
        let c1 = s1b;

        let a2 = Self::square8(a1);
        let b2 = Self::square8(b1);
        let c2 = Self::square8(c1);

        let a4 = Self::square8(a2);
        let b4 = Self::square8(b2);
        let c4 = Self::square8(c2);

        let a7 = Self::mul8(Self::mul8(a1, a2), a4);
        let b7 = Self::mul8(Self::mul8(b1, b2), b4);
        let c7 = Self::mul8(Self::mul8(c1, c2), c4);

        /* recombine substitution/exponentiation results and write back to state */
        let a7rev = _mm512_permutexvar_epi64(idx, a7);
        let out0 = _mm512_mask_blend_epi64(0x0f, a7, asbox);
        let out1 = _mm512_mask_blend_epi64(0x0f, a7rev, asboxrev);

        _mm512_store_epi64(state0.as_mut_ptr().offset(0x00) as *mut i64, out0);
        _mm512_store_epi64(state1.as_mut_ptr().offset(0x00) as *mut i64, out1);

        _mm512_store_epi64(state0.as_mut_ptr().offset(0x08) as *mut i64, b7);
        _mm512_store_epi64(state1.as_mut_ptr().offset(0x08) as *mut i64, c7);
    }

    #[inline(always)]
    #[expect(clippy::identity_op)]
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
        const RCS_MONT_U: [u64; 80] = [
            0x61ab60dc, 0xd9547ed0, 0xa1de063d, 0x876c8676, 0x889cfb95, 0x43699f00, 0x7190db57,
            0xd2b0d4b0, 0xd483cd36, 0x44882a55, 0x9f498aa3, 0x79338d4b, 0x52c5b216, 0x48adad93,
            0xfec868b5, 0xfb6b0d8a, 0x20ef0328, 0x5bba5802, 0x27287a26, 0x4e193411, 0xa977eae0,
            0x63fc191a, 0xaf39b210, 0x5933202e, 0xbfcf71e4, 0xcc520bfb, 0xf774f673, 0x0309bc69,
            0x275f3cb2, 0x2c8f905a, 0x61e609b3, 0x5c92c93a, 0x56411dbf, 0x5fc2a26b, 0x3d9f2bf2,
            0x5ca88c43, 0x2e1c1552, 0x3220a672, 0x4b861c4d, 0xeb86ebd6, 0xbc3902de, 0x516bcbc0,
            0x738f27cf, 0xeac8ea36, 0x4bf937c4, 0x220e6746, 0x07e796f8, 0xf2f6dd71, 0x7d6e3a40,
            0xe73743d7, 0xef802e57, 0x336e6aa5, 0xf3c8b226, 0x6afb2112, 0x25531967, 0x3866d0ee,
            0xd2215022, 0x12ee85b1, 0xfcd23eb4, 0xd727752f, 0xaff543b3, 0x17f192d4, 0xb026adc0,
            0xe35c1017, 0x6080bd06, 0x0b8a28b7, 0xae9da4ca, 0xd9e5a26b, 0x2d337846, 0xb7eee345,
            0x59dde50c, 0x5ee62a88, 0xf6a203d0, 0x3b6ae69e, 0x2be69c37, 0xdfff43cb, 0x5f4fdc6a,
            0x97c0d760, 0x14148eba, 0xf2f24472,
        ];
        const RCS_MONT_L: [u64; 80] = [
            0xe12a6137, 0x3c2d8f14, 0xce16c34a, 0x5d4cf10b, 0xa3fe2af2, 0xe0086636, 0x5712e44b,
            0x05bceb49, 0xb29f2156, 0x88310f48, 0xb091da34, 0xf1ff20f5, 0xfc597178, 0xbe758d99,
            0x9853d114, 0x2cc48735, 0xebc0eeec, 0x5bdfe8e6, 0x02df87a9, 0x0c7397fa, 0xcf6133cb,
            0x6bef3d61, 0x96b1f98d, 0xa3216fc1, 0x029fd62d, 0xfb4ad152, 0xe0c840b1, 0xad2abfa1,
            0x7a336665, 0xe6ad794b, 0x1a9aa328, 0xf0bb400b, 0xe9bc674a, 0x895bd10c, 0x39dfe4f5,
            0xf0c467e0, 0x35b5227b, 0xe82efadd, 0x0fdd1d04, 0x0308861f, 0x832913f5, 0x1bf8f7c6,
            0xac69f270, 0xe798f708, 0xaa81ef62, 0x9498717d, 0xf9fad5c4, 0xe16d8ff5, 0x7aefd019,
            0xd4c162e9, 0x717a8a87, 0x53bcde49, 0x5e71152a, 0xf02e0b04, 0x3d64ddb1, 0x91012a32,
            0x4702d633, 0x5e3f4dac, 0xc9b208c8, 0x3d490349, 0xb670e77e, 0xf48bc718, 0x0615dfdf,
            0xdcab5e5b, 0x71014a42, 0xfe9a2b22, 0xcc26240d, 0x732867a0, 0x92fe65b8, 0xdcb6de4c,
            0x8f0c9826, 0xe059226d, 0xa302d668, 0x93fb6a88, 0x53fb6dbf, 0x9f9a0f27, 0x15b64f4b,
            0x903d0ed1, 0xdb21a28b, 0xb971e6c9,
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

    #[inline(always)]
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
    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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
