//! Make [`Tip5`] even faster by using SIMD, in particular, AVX-512.

use std::arch::x86_64::*;

use super::LOOKUP_TABLE;
use super::NUM_ROUNDS;
use super::ROUND_CONSTANTS;
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
    pub(super) unsafe fn mds_rcs_avx512(
        state: &mut [BFieldElement; STATE_SIZE],
        round_index: usize,
    ) {
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
        const RCS_MONT_U: [u64; NUM_ROUNDS * STATE_SIZE] = {
            let mut constants = [0; NUM_ROUNDS * STATE_SIZE];
            let mut i = 0;
            while i < NUM_ROUNDS * STATE_SIZE {
                constants[i] = ROUND_CONSTANTS[i].raw_u64() >> 32;
                i += 1;
            }
            constants
        };
        const RCS_MONT_L: [u64; NUM_ROUNDS * STATE_SIZE] = {
            let mut constants = [0; NUM_ROUNDS * STATE_SIZE];
            let mut i = 0;
            while i < NUM_ROUNDS * STATE_SIZE {
                constants[i] = ROUND_CONSTANTS[i].raw_u64() & (u32::MAX as u64);
                i += 1;
            }
            constants
        };

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

    /// Combine and reduce `lo * 2**0 + hi * 2**32` to F_P
    #[inline(always)]
    unsafe fn reduce2x32(lo: __m512i, hi: __m512i) -> __m512i {
        /* Propagate carries */
        let lo_hi = _mm512_srli_epi64(lo, 32);
        let overflowing_hi = _mm512_add_epi64(lo_hi, hi);
        let carry = _mm512_srli_epi64(overflowing_hi, 32);
        let carry_shifted_left = _mm512_slli_epi64(carry, 32);

        /* mod reduce */
        let hi_shifted_left = _mm512_slli_epi64(hi, 32);
        let mut res = _mm512_add_epi64(lo, hi_shifted_left);
        res = _mm512_add_epi64(res, carry_shifted_left);
        res = _mm512_sub_epi64(res, carry);

        res
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

#[cfg(test)]
mod tests {
    use std::arch::x86_64::_mm512_cmpeq_epi64_mask;
    use std::mem::transmute;

    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::*;

    /// Helper trait to turn something into a compatible AVX-512 type. Only
    /// used for testing.
    //
    // Should you want to use this trait in a non-test environment, please
    // first check if this design is actually what you want. Not much thought
    // has gone into that part.
    trait Pack
    where
        Self: Sized,
    {
        type Packed;

        fn pack(self) -> Self::Packed;

        fn unpack(packed: Self::Packed) -> Self;
    }

    impl Pack for [i64; 8] {
        type Packed = __m512i;

        fn pack(self) -> Self::Packed {
            let s = self;
            unsafe { _mm512_set_epi64(s[7], s[6], s[5], s[4], s[3], s[2], s[1], s[0]) }
        }

        fn unpack(packed: Self::Packed) -> Self {
            // SAFETY: `Self` and `Self::Packed` have the same valid bit-patterns
            unsafe { transmute(packed) }
        }
    }

    impl Pack for [u64; 8] {
        type Packed = __m512i;

        fn pack(self) -> Self::Packed {
            self.map(|e| i64::from_ne_bytes(e.to_ne_bytes())).pack()
        }

        fn unpack(packed: Self::Packed) -> Self {
            // SAFETY: `Self` and `Self::Packed` have the same valid bit-patterns
            unsafe { transmute(packed) }
        }
    }

    /// A scalar version of [Tip5::reduce2x32].
    fn scalar_reduce2x32(lo: u64, hi: u64) -> u64 {
        /* Propagate carries */
        let overflowing_hi = hi.wrapping_add(lo >> 32);
        let carry = overflowing_hi >> 32;
        let carry_shifted_left = carry << 32;

        /* mod reduce */
        lo.wrapping_add(hi << 32)
            .wrapping_add(carry_shifted_left)
            .wrapping_sub(carry)
    }

    #[proptest]
    fn packing_and_unpacking_is_reciprocal(original: [u64; 8]) {
        let repacked = <[u64; 8]>::unpack(original.pack());
        prop_assert_eq!(original, repacked);
    }

    #[proptest]
    fn scalar_and_vectorized_reduce2x32_correspond(lo: u64, hi: u64) {
        let broadcast = |x: u64| {
            let x = i64::from_ne_bytes(x.to_ne_bytes());
            unsafe { _mm512_set1_epi64(x) }
        };

        let lo_vec = broadcast(lo);
        let hi_vec = broadcast(hi);
        let reduced_vec = unsafe { Tip5::reduce2x32(lo_vec, hi_vec) };

        let reduced_scalar = scalar_reduce2x32(lo, hi);
        let eq_mask = unsafe { _mm512_cmpeq_epi64_mask(reduced_vec, broadcast(reduced_scalar)) };
        prop_assert_eq!(0xFF, eq_mask);
    }

    #[proptest]
    fn reduce2x32_and_montgomery_reduction_correspond(a: BFieldElement, b: BFieldElement) {
        const LOW_BIT_MASK: u64 = u64::MAX >> 32;

        let a_raw_lo = a.raw_u64() & LOW_BIT_MASK;
        let a_raw_hi = a.raw_u64() >> 32;
        let b_raw_lo = b.raw_u64() & LOW_BIT_MASK;
        let b_raw_hi = b.raw_u64() >> 32;

        let c_raw_lo = a_raw_lo.wrapping_add(b_raw_lo);
        let c_raw_hi = a_raw_hi.wrapping_add(b_raw_hi);
        let c_raw = scalar_reduce2x32(c_raw_lo, c_raw_hi);
        let c = BFieldElement::from_raw_u64(c_raw);
        prop_assert_eq!(a + b, c, "add");

        let d_raw_lo = a_raw_lo.wrapping_mul(b_raw_lo);
        let d_raw_hi = a_raw_hi.wrapping_mul(b_raw_hi);
        let d_raw = scalar_reduce2x32(d_raw_lo, d_raw_hi);
        let d = BFieldElement::from_raw_u64(d_raw);
        prop_assert_eq!(a * b, d, "mul");
    }
}
