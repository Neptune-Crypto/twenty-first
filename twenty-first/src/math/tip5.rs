use std::hash::Hasher;

use arbitrary::Arbitrary;
use get_size2::GetSize;
use itertools::Itertools;
use num_traits::ConstOne;
use num_traits::ConstZero;
use serde::Deserialize;
use serde::Serialize;

use crate::math::b_field_element::BFieldElement;
pub use crate::math::digest::Digest;
use crate::math::mds::generated_function;
use crate::math::x_field_element::EXTENSION_DEGREE;
use crate::prelude::BFieldCodec;
use crate::prelude::XFieldElement;
use crate::util_types::sponge::Domain;
use crate::util_types::sponge::Sponge;

pub const STATE_SIZE: usize = 16;
pub const NUM_SPLIT_AND_LOOKUP: usize = 4;
pub const LOG2_STATE_SIZE: usize = 4;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;
pub const NUM_ROUNDS: usize = 5;

/// The lookup table with a high algebraic degree used in the TIP-5 permutation. To verify its
/// correctness, see the test “lookup_table_is_correct.”
pub const LOOKUP_TABLE: [u8; 256] = [
    0, 7, 26, 63, 124, 215, 85, 254, 214, 228, 45, 185, 140, 173, 33, 240, 29, 177, 176, 32, 8,
    110, 87, 202, 204, 99, 150, 106, 230, 14, 235, 128, 213, 239, 212, 138, 23, 130, 208, 6, 44,
    71, 93, 116, 146, 189, 251, 81, 199, 97, 38, 28, 73, 179, 95, 84, 152, 48, 35, 119, 49, 88,
    242, 3, 148, 169, 72, 120, 62, 161, 166, 83, 175, 191, 137, 19, 100, 129, 112, 55, 221, 102,
    218, 61, 151, 237, 68, 164, 17, 147, 46, 234, 203, 216, 22, 141, 65, 57, 123, 12, 244, 54, 219,
    231, 96, 77, 180, 154, 5, 253, 133, 165, 98, 195, 205, 134, 245, 30, 9, 188, 59, 142, 186, 197,
    181, 144, 92, 31, 224, 163, 111, 74, 58, 69, 113, 196, 67, 246, 225, 10, 121, 50, 60, 157, 90,
    122, 2, 250, 101, 75, 178, 159, 24, 36, 201, 11, 243, 132, 198, 190, 114, 233, 39, 52, 21, 209,
    108, 238, 91, 187, 18, 104, 194, 37, 153, 34, 200, 143, 126, 155, 236, 118, 64, 80, 172, 89,
    94, 193, 135, 183, 86, 107, 252, 13, 167, 206, 136, 220, 207, 103, 171, 160, 76, 182, 227, 217,
    158, 56, 174, 4, 66, 109, 139, 162, 184, 211, 249, 47, 125, 232, 117, 43, 16, 42, 127, 20, 241,
    25, 149, 105, 156, 51, 53, 168, 145, 247, 223, 79, 78, 226, 15, 222, 82, 115, 70, 210, 27, 41,
    1, 170, 40, 131, 192, 229, 248, 255,
];

/// The round constants used in the Tip5 permutation. To verify their correctness, see the test
/// “round_constants_are_correct.”
pub const ROUND_CONSTANTS: [BFieldElement; NUM_ROUNDS * STATE_SIZE] = [
    BFieldElement::new(13630775303355457758),
    BFieldElement::new(16896927574093233874),
    BFieldElement::new(10379449653650130495),
    BFieldElement::new(1965408364413093495),
    BFieldElement::new(15232538947090185111),
    BFieldElement::new(15892634398091747074),
    BFieldElement::new(3989134140024871768),
    BFieldElement::new(2851411912127730865),
    BFieldElement::new(8709136439293758776),
    BFieldElement::new(3694858669662939734),
    BFieldElement::new(12692440244315327141),
    BFieldElement::new(10722316166358076749),
    BFieldElement::new(12745429320441639448),
    BFieldElement::new(17932424223723990421),
    BFieldElement::new(7558102534867937463),
    BFieldElement::new(15551047435855531404),
    BFieldElement::new(17532528648579384106),
    BFieldElement::new(5216785850422679555),
    BFieldElement::new(15418071332095031847),
    BFieldElement::new(11921929762955146258),
    BFieldElement::new(9738718993677019874),
    BFieldElement::new(3464580399432997147),
    BFieldElement::new(13408434769117164050),
    BFieldElement::new(264428218649616431),
    BFieldElement::new(4436247869008081381),
    BFieldElement::new(4063129435850804221),
    BFieldElement::new(2865073155741120117),
    BFieldElement::new(5749834437609765994),
    BFieldElement::new(6804196764189408435),
    BFieldElement::new(17060469201292988508),
    BFieldElement::new(9475383556737206708),
    BFieldElement::new(12876344085611465020),
    BFieldElement::new(13835756199368269249),
    BFieldElement::new(1648753455944344172),
    BFieldElement::new(9836124473569258483),
    BFieldElement::new(12867641597107932229),
    BFieldElement::new(11254152636692960595),
    BFieldElement::new(16550832737139861108),
    BFieldElement::new(11861573970480733262),
    BFieldElement::new(1256660473588673495),
    BFieldElement::new(13879506000676455136),
    BFieldElement::new(10564103842682358721),
    BFieldElement::new(16142842524796397521),
    BFieldElement::new(3287098591948630584),
    BFieldElement::new(685911471061284805),
    BFieldElement::new(5285298776918878023),
    BFieldElement::new(18310953571768047354),
    BFieldElement::new(3142266350630002035),
    BFieldElement::new(549990724933663297),
    BFieldElement::new(4901984846118077401),
    BFieldElement::new(11458643033696775769),
    BFieldElement::new(8706785264119212710),
    BFieldElement::new(12521758138015724072),
    BFieldElement::new(11877914062416978196),
    BFieldElement::new(11333318251134523752),
    BFieldElement::new(3933899631278608623),
    BFieldElement::new(16635128972021157924),
    BFieldElement::new(10291337173108950450),
    BFieldElement::new(4142107155024199350),
    BFieldElement::new(16973934533787743537),
    BFieldElement::new(11068111539125175221),
    BFieldElement::new(17546769694830203606),
    BFieldElement::new(5315217744825068993),
    BFieldElement::new(4609594252909613081),
    BFieldElement::new(3350107164315270407),
    BFieldElement::new(17715942834299349177),
    BFieldElement::new(9600609149219873996),
    BFieldElement::new(12894357635820003949),
    BFieldElement::new(4597649658040514631),
    BFieldElement::new(7735563950920491847),
    BFieldElement::new(1663379455870887181),
    BFieldElement::new(13889298103638829706),
    BFieldElement::new(7375530351220884434),
    BFieldElement::new(3502022433285269151),
    BFieldElement::new(9231805330431056952),
    BFieldElement::new(9252272755288523725),
    BFieldElement::new(10014268662326746219),
    BFieldElement::new(15565031632950843234),
    BFieldElement::new(1209725273521819323),
    BFieldElement::new(6024642864597845108),
];

/// The defining, first column of the (circulant) MDS matrix.
/// Derived from the SHA-256 hash of the ASCII string “Tip5” by dividing the digest into 16-bit
/// chunks.
pub const MDS_MATRIX_FIRST_COLUMN: [i64; STATE_SIZE] = [
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901, 12021, 59689,
    26798, 17845,
];

#[derive(
    Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, GetSize, BFieldCodec, Arbitrary,
)]
pub struct Tip5 {
    pub state: [BFieldElement; STATE_SIZE],
}

impl Tip5 {
    #[inline]
    pub const fn new(domain: Domain) -> Self {
        use Domain::*;

        let mut state = [BFieldElement::ZERO; STATE_SIZE];

        match domain {
            VariableLength => (),
            FixedLength => {
                let mut i = RATE;
                while i < STATE_SIZE {
                    state[i] = BFieldElement::ONE;
                    i += 1;
                }
            }
        }

        Self { state }
    }

    #[inline]
    pub const fn offset_fermat_cube_map(x: u16) -> u16 {
        let xx = (x + 1) as u64;
        let xxx = xx * xx * xx;
        ((xxx + 256) % 257) as u16
    }

    #[inline]
    fn split_and_lookup(element: &mut BFieldElement) {
        // let value = element.value();
        let mut bytes = element.raw_bytes();

        #[allow(clippy::needless_range_loop)] // faster like so
        for i in 0..8 {
            // bytes[i] = Self::offset_fermat_cube_map(bytes[i].into()) as u8;
            bytes[i] = LOOKUP_TABLE[bytes[i] as usize];
        }

        *element = BFieldElement::from_raw_bytes(&bytes);
    }

    #[inline(always)]
    fn fast_cyclomul16(f: [i64; 16], g: [i64; 16]) -> [i64; 16] {
        const N: usize = 8;
        let mut ff_lo = [0i64; N];
        let mut gg_lo = [0i64; N];
        let mut ff_hi = [0i64; N];
        let mut gg_hi = [0i64; N];
        for i in 0..N {
            ff_lo[i] = f[i] + f[i + N];
            ff_hi[i] = f[i] - f[i + N];
            gg_lo[i] = g[i] + g[i + N];
            gg_hi[i] = g[i] - g[i + N];
        }

        let hh_lo = Self::fast_cyclomul8(ff_lo, gg_lo);
        let hh_hi = Self::complex_negacyclomul8(ff_hi, gg_hi);

        let mut hh = [0i64; 2 * N];
        for i in 0..N {
            hh[i] = (hh_lo[i] + hh_hi[i]) >> 1;
            hh[i + N] = (hh_lo[i] - hh_hi[i]) >> 1;
        }

        hh
    }

    #[inline(always)]
    fn complex_sum<const N: usize>(f: [(i64, i64); N], g: [(i64, i64); N]) -> [(i64, i64); N] {
        let mut h = [(0i64, 0i64); N];
        for i in 0..N {
            h[i].0 = f[i].0 + g[i].0;
            h[i].1 = f[i].1 + g[i].1;
        }
        h
    }

    #[inline(always)]
    fn complex_diff<const N: usize>(f: [(i64, i64); N], g: [(i64, i64); N]) -> [(i64, i64); N] {
        let mut h = [(0i64, 0i64); N];
        for i in 0..N {
            h[i].0 = f[i].0 - g[i].0;
            h[i].1 = f[i].1 - g[i].1;
        }
        h
    }

    #[inline(always)]
    fn complex_product(f: (i64, i64), g: (i64, i64)) -> (i64, i64) {
        // don't karatsuba; this is faster
        (f.0 * g.0 - f.1 * g.1, f.0 * g.1 + f.1 * g.0)
    }

    #[inline(always)]
    fn complex_karatsuba2(f: [(i64, i64); 2], g: [(i64, i64); 2]) -> [(i64, i64); 3] {
        const N: usize = 1;

        let ff = (f[0].0 + f[1].0, f[0].1 + f[1].1);
        let gg = (g[0].0 + g[1].0, g[0].1 + g[1].1);

        let lo = Self::complex_product(f[0], g[0]);
        let hi = Self::complex_product(f[1], g[1]);

        let ff_times_gg = Self::complex_product(ff, gg);
        let lo_plus_hi = (lo.0 + hi.0, lo.1 + hi.1);

        let li = (ff_times_gg.0 - lo_plus_hi.0, ff_times_gg.1 - lo_plus_hi.1);

        let mut result = [(0i64, 0i64); 4 * N - 1];
        result[0].0 += lo.0;
        result[0].1 += lo.1;
        result[N].0 += li.0;
        result[N].1 += li.1;
        result[2 * N].0 += hi.0;
        result[2 * N].1 += hi.1;

        result
    }

    #[inline(always)]
    fn complex_karatsuba4(f: [(i64, i64); 4], g: [(i64, i64); 4]) -> [(i64, i64); 7] {
        const N: usize = 2;

        let ff = Self::complex_sum::<2>(f[..N].try_into().unwrap(), f[N..].try_into().unwrap());
        let gg = Self::complex_sum::<2>(g[..N].try_into().unwrap(), g[N..].try_into().unwrap());

        let lo = Self::complex_karatsuba2(f[..N].try_into().unwrap(), g[..N].try_into().unwrap());
        let hi = Self::complex_karatsuba2(f[N..].try_into().unwrap(), g[N..].try_into().unwrap());

        let li = Self::complex_diff::<3>(
            Self::complex_karatsuba2(ff, gg),
            Self::complex_sum::<3>(lo, hi),
        );

        let mut result = [(0i64, 0i64); 4 * N - 1];
        for i in 0..(2 * N - 1) {
            result[i].0 = lo[i].0;
            result[i].1 = lo[i].1;
        }
        for i in 0..(2 * N - 1) {
            result[N + i].0 += li[i].0;
            result[N + i].1 += li[i].1;
        }
        for i in 0..(2 * N - 1) {
            result[2 * N + i].0 += hi[i].0;
            result[2 * N + i].1 += hi[i].1;
        }

        result
    }

    #[inline(always)]
    fn complex_negacyclomul8(f: [i64; 8], g: [i64; 8]) -> [i64; 8] {
        const N: usize = 4;

        let mut f0 = [(0i64, 0i64); N];
        // let mut f1 = [(0i64,0i64); N];
        let mut g0 = [(0i64, 0i64); N];
        // let mut g1 = [(0i64,0i64); N];

        for i in 0..N {
            f0[i] = (f[i], -f[N + i]);
            // f1[i] = (f[i],  f[N+i]);
            g0[i] = (g[i], -g[N + i]);
            // g1[i] = (g[i],  g[N+i]);
        }

        let h0 = Self::complex_karatsuba4(f0, g0);
        // h1 = complex_karatsuba(f1, g1)

        // h = a * h0 + b * h1
        // where a = 2^-1 * (i*X^(n/2) + 1)
        // and  b = 2^-1 * (-i*X^(n/2) + 1)

        let mut h = [0i64; 3 * N - 1];
        for i in 0..(2 * N - 1) {
            h[i] += h0[i].0;
            h[i + N] -= h0[i].1;
            // h[i] += h0[i].0 / 2
            // h[i+N] -= h0[i].1 / 2
            // h[i] += h1[i].0 / 2
            // h[i+N] -= h1[i].1 / 2
        }

        let mut hh = [0i64; 2 * N];
        for i in 0..(2 * N) {
            hh[i] += h[i];
        }
        for i in (2 * N)..(3 * N - 1) {
            hh[i - 2 * N] -= h[i];
        }

        hh
    }

    #[inline(always)]
    fn complex_negacyclomul4(f: [i64; 4], g: [i64; 4]) -> [i64; 4] {
        const N: usize = 2;

        let mut f0 = [(0i64, 0i64); N];
        // let mut f1 = [(0i64,0i64); N];
        let mut g0 = [(0i64, 0i64); N];
        // let mut g1 = [(0i64,0i64); N];

        for i in 0..N {
            f0[i] = (f[i], -f[N + i]);
            // f1[i] = (f[i],  f[N+i]);
            g0[i] = (g[i], -g[N + i]);
            // g1[i] = (g[i],  g[N+i]);
        }

        let h0 = Self::complex_karatsuba2(f0, g0);
        // h1 = complex_karatsuba(f1, g1)

        // h = a * h0 + b * h1
        // where a = 2^-1 * (i*X^(n/2) + 1)
        // and  b = 2^-1 * (-i*X^(n/2) + 1)

        let mut h = [0i64; 4 * N - 1];
        for i in 0..(2 * N - 1) {
            h[i] += h0[i].0;
            h[i + N] -= h0[i].1;
            // h[i] += h0[i].0 / 2
            // h[i+N] -= h0[i].1 / 2
            // h[i] += h1[i].0 / 2
            // h[i+N] -= h1[i].1 / 2
        }

        let mut hh = [0i64; 2 * N];
        for i in 0..(2 * N) {
            hh[i] += h[i];
        }
        for i in (2 * N)..(4 * N - 1) {
            hh[i - 2 * N] -= h[i];
        }

        hh
    }

    #[inline(always)]
    fn complex_negacyclomul2(f: [i64; 2], g: [i64; 2]) -> [i64; 2] {
        let f0 = (f[0], -f[1]);
        let g0 = (g[0], -g[1]);

        let h0 = Self::complex_product(f0, g0);

        [h0.0, -h0.1]
    }

    #[inline(always)]
    fn fast_cyclomul8(f: [i64; 8], g: [i64; 8]) -> [i64; 8] {
        const N: usize = 4;
        let mut ff_lo = [0i64; N];
        let mut gg_lo = [0i64; N];
        let mut ff_hi = [0i64; N];
        let mut gg_hi = [0i64; N];
        for i in 0..N {
            ff_lo[i] = f[i] + f[i + N];
            ff_hi[i] = f[i] - f[i + N];
            gg_lo[i] = g[i] + g[i + N];
            gg_hi[i] = g[i] - g[i + N];
        }

        let hh_lo = Self::fast_cyclomul4(ff_lo, gg_lo);
        let hh_hi = Self::complex_negacyclomul4(ff_hi, gg_hi);

        let mut hh = [0i64; 2 * N];
        for i in 0..N {
            hh[i] = (hh_lo[i] + hh_hi[i]) >> 1;
            hh[i + N] = (hh_lo[i] - hh_hi[i]) >> 1;
        }

        hh
    }

    #[inline(always)]
    fn fast_cyclomul4(f: [i64; 4], g: [i64; 4]) -> [i64; 4] {
        const N: usize = 2;
        let mut ff_lo = [0i64; N];
        let mut gg_lo = [0i64; N];
        let mut ff_hi = [0i64; N];
        let mut gg_hi = [0i64; N];
        for i in 0..N {
            ff_lo[i] = f[i] + f[i + N];
            ff_hi[i] = f[i] - f[i + N];
            gg_lo[i] = g[i] + g[i + N];
            gg_hi[i] = g[i] - g[i + N];
        }

        let hh_lo = Self::fast_cyclomul2(ff_lo, gg_lo);
        let hh_hi = Self::complex_negacyclomul2(ff_hi, gg_hi);

        let mut hh = [0i64; 2 * N];
        for i in 0..N {
            hh[i] = (hh_lo[i] + hh_hi[i]) >> 1;
            hh[i + N] = (hh_lo[i] - hh_hi[i]) >> 1;
        }

        hh
    }

    #[inline(always)]
    fn fast_cyclomul2(f: [i64; 2], g: [i64; 2]) -> [i64; 2] {
        let ff_lo = f[0] + f[1];
        let ff_hi = f[0] - f[1];
        let gg_lo = g[0] + g[1];
        let gg_hi = g[0] - g[1];

        let hh_lo = ff_lo * gg_lo;
        let hh_hi = ff_hi * gg_hi;

        let mut hh = [0i64; 2];
        hh[0] = (hh_lo + hh_hi) >> 1;
        hh[1] = (hh_lo - hh_hi) >> 1;

        hh
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn mds_cyclomul(&mut self) {
        let mut result = [BFieldElement::ZERO; STATE_SIZE];

        let mut lo: [i64; STATE_SIZE] = [0; STATE_SIZE];
        let mut hi: [i64; STATE_SIZE] = [0; STATE_SIZE];
        for (i, b) in self.state.iter().enumerate() {
            hi[i] = (b.raw_u64() >> 32) as i64;
            lo[i] = (b.raw_u64() as u32) as i64;
        }

        lo = Self::fast_cyclomul16(lo, MDS_MATRIX_FIRST_COLUMN);
        hi = Self::fast_cyclomul16(hi, MDS_MATRIX_FIRST_COLUMN);

        for r in 0..STATE_SIZE {
            let s = lo[r] as u128 + ((hi[r] as u128) << 32);
            let s_hi = (s >> 64) as u64;
            let s_lo = s as u64;
            let z = (s_hi << 32) - s_hi;
            let (res, over) = s_lo.overflowing_add(z);

            result[r] = BFieldElement::from_raw_u64(
                res.wrapping_add(0u32.wrapping_sub(over as u32) as u64),
            );
        }
        self.state = result;
    }

    #[inline(always)]
    fn mds_generated(&mut self) {
        let mut lo: [u64; STATE_SIZE] = [0; STATE_SIZE];
        let mut hi: [u64; STATE_SIZE] = [0; STATE_SIZE];
        for i in 0..STATE_SIZE {
            let b = self.state[i].raw_u64();
            hi[i] = b >> 32;
            lo[i] = b & 0xffffffffu64;
        }

        lo = generated_function(&lo);
        hi = generated_function(&hi);

        for r in 0..STATE_SIZE {
            let s = (lo[r] >> 4) as u128 + ((hi[r] as u128) << 28);

            let s_hi = (s >> 64) as u64;
            let s_lo = s as u64;

            let (res, over) = s_lo.overflowing_add(s_hi * 0xffffffffu64);

            self.state[r] =
                BFieldElement::from_raw_u64(if over { res + 0xffffffffu64 } else { res });
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn sbox_layer(&mut self) {
        for i in 0..NUM_SPLIT_AND_LOOKUP {
            Self::split_and_lookup(&mut self.state[i]);
        }

        for i in NUM_SPLIT_AND_LOOKUP..STATE_SIZE {
            let sq = self.state[i] * self.state[i];
            let qu = sq * sq;
            self.state[i] *= sq * qu;
        }
    }

    #[inline(always)]
    fn round(&mut self, round_index: usize) {
        self.sbox_layer();
        self.mds_generated();
        for i in 0..STATE_SIZE {
            self.state[i] += ROUND_CONSTANTS[round_index * STATE_SIZE + i];
        }
    }

    #[inline(always)]
    pub fn permutation(&mut self) {
        for i in 0..NUM_ROUNDS {
            self.round(i);
        }
    }

    /// Functionally equivalent to [`permutation`](Self::permutation). Returns the trace of
    /// applying the permutation; that is, the initial state of the sponge as well as its state
    /// after each round.
    pub fn trace(&mut self) -> [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS] {
        let mut trace = [[BFieldElement::ZERO; STATE_SIZE]; 1 + NUM_ROUNDS];

        trace[0] = self.state;
        for i in 0..NUM_ROUNDS {
            self.round(i);
            trace[1 + i] = self.state;
        }

        trace
    }

    /// Hash 10 elements, or two digests. There is no padding because
    /// the input length is fixed.
    pub fn hash_10(input: &[BFieldElement; 10]) -> [BFieldElement; Digest::LEN] {
        let mut sponge = Self::new(Domain::FixedLength);

        // absorb once
        sponge.state[..10].copy_from_slice(input);

        sponge.permutation();

        // squeeze once
        sponge.state[..Digest::LEN].try_into().unwrap()
    }

    pub fn hash_pair(left: Digest, right: Digest) -> Digest {
        let mut sponge = Self::new(Domain::FixedLength);
        sponge.state[..Digest::LEN].copy_from_slice(&left.values());
        sponge.state[Digest::LEN..2 * Digest::LEN].copy_from_slice(&right.values());

        sponge.permutation();

        let digest_values = sponge.state[..Digest::LEN].try_into().unwrap();
        Digest::new(digest_values)
    }

    /// Thin wrapper around [`hash_varlen`](Self::hash_varlen).
    pub fn hash<T: BFieldCodec>(value: &T) -> Digest {
        Self::hash_varlen(&value.encode())
    }

    /// Hash a variable-length sequence of [`BFieldElement`].
    ///
    /// - Apply the correct padding
    /// - [Sponge::pad_and_absorb_all()]
    /// - [Sponge::squeeze()] once.
    pub fn hash_varlen(input: &[BFieldElement]) -> Digest {
        let mut sponge = Self::init();
        sponge.pad_and_absorb_all(input);
        let produce: [BFieldElement; crate::util_types::sponge::RATE] = sponge.squeeze();

        Digest::new((&produce[..Digest::LEN]).try_into().unwrap())
    }

    /// Produce `num_indices` random integer values in the range `[0, upper_bound)`. The
    /// `upper_bound` must be a power of 2.
    ///
    /// This method uses von Neumann rejection sampling.
    /// Specifically, if the top 32 bits of a BFieldElement are all ones, then the bottom 32 bits
    /// are not uniformly distributed, and so they are dropped. This method invokes squeeze until
    /// enough uniform u32s have been sampled.
    pub fn sample_indices(&mut self, upper_bound: u32, num_indices: usize) -> Vec<u32> {
        debug_assert!(upper_bound.is_power_of_two());
        let mut indices = vec![];
        let mut squeezed_elements = vec![];
        while indices.len() != num_indices {
            if squeezed_elements.is_empty() {
                squeezed_elements = self.squeeze().into_iter().rev().collect_vec();
            }
            let element = squeezed_elements.pop().unwrap();
            if element != BFieldElement::new(BFieldElement::MAX) {
                indices.push(element.value() as u32 % upper_bound);
            }
        }
        indices
    }

    /// Produce `num_elements` random [`XFieldElement`] values.
    ///
    /// If `num_elements` is not divisible by [`RATE`][rate], spill the remaining elements of the
    /// last [`squeeze`][Sponge::squeeze].
    ///
    /// [rate]: Sponge::RATE
    pub fn sample_scalars(&mut self, num_elements: usize) -> Vec<XFieldElement> {
        let num_squeezes = (num_elements * EXTENSION_DEGREE).div_ceil(Self::RATE);
        debug_assert!(
            num_elements * EXTENSION_DEGREE <= num_squeezes * Self::RATE,
            "need {} elements but getting {}",
            num_elements * EXTENSION_DEGREE,
            num_squeezes * Self::RATE
        );
        (0..num_squeezes)
            .flat_map(|_| self.squeeze())
            .collect_vec()
            .chunks(3)
            .take(num_elements)
            .map(|elem| XFieldElement::new([elem[0], elem[1], elem[2]]))
            .collect()
    }
}

impl Sponge for Tip5 {
    const RATE: usize = RATE;

    fn init() -> Self {
        Self::new(Domain::VariableLength)
    }

    fn absorb(&mut self, input: [BFieldElement; RATE]) {
        self.state[..RATE]
            .iter_mut()
            .zip_eq(&input)
            .for_each(|(a, &b)| *a = b);

        self.permutation();
    }

    fn squeeze(&mut self) -> [BFieldElement; RATE] {
        let produce: [BFieldElement; RATE] = (&self.state[..RATE]).try_into().unwrap();
        self.permutation();

        produce
    }
}

impl Hasher for Tip5 {
    fn finish(&self) -> u64 {
        self.state[0].value()
    }

    fn write(&mut self, bytes: &[u8]) {
        let bfield_elements = bytes.chunks(BFieldElement::BYTES).map(|chunk| {
            let mut buffer = [0u8; BFieldElement::BYTES];
            buffer[..chunk.len()].copy_from_slice(chunk);
            BFieldElement::new(u64::from_le_bytes(buffer))
        });

        for chunk in bfield_elements.chunks(Tip5::RATE).into_iter() {
            let mut buffer = [BFieldElement::ZERO; Tip5::RATE];
            for (buffer_elem, chunk_elem) in buffer.iter_mut().zip(chunk) {
                *buffer_elem = chunk_elem;
            }
            self.absorb(buffer)
        }
    }
}

#[cfg(test)]
pub(crate) mod tip5_tests {
    use std::hash::Hash;
    use std::ops::Mul;

    use insta::assert_snapshot;
    use prop::sample::size_range;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::thread_rng;
    use rand::Rng;
    use rand::RngCore;
    use rayon::prelude::IntoParallelIterator;
    use rayon::prelude::ParallelIterator;
    use test_strategy::proptest;

    use super::*;
    use crate::math::other::random_elements;
    use crate::math::x_field_element::XFieldElement;

    impl Tip5 {
        pub(crate) fn randomly_seeded() -> Self {
            let mut sponge = Self::init();
            let mut rng = thread_rng();
            sponge.absorb(rng.gen());
            sponge
        }
    }

    #[test]
    fn get_size_test() {
        assert_eq!(
            STATE_SIZE * BFieldElement::ZERO.get_size(),
            Tip5::randomly_seeded().get_size()
        );
    }

    #[test]
    fn lookup_table_is_correct() {
        let table: [u8; 256] = (0..256)
            .map(|t| Tip5::offset_fermat_cube_map(t as u16) as u8)
            .collect_vec()
            .try_into()
            .unwrap();

        println!(
            "Entire lookup table:\n{}",
            table.iter().map(|t| format!("{t:02x}")).join(", ")
        );

        (0_usize..256).for_each(|i| {
            assert_eq!(
                LOOKUP_TABLE[i], table[i],
                "Lookup tables must agree at every index, including index {i}."
            )
        });
    }

    #[test]
    fn round_constants_are_correct() {
        let to_int = |bytes: &[u8]| {
            bytes
                .iter()
                .take(16)
                .enumerate()
                .map(|(i, b)| (*b as u128) << (8 * i))
                .sum::<u128>()
        };
        let round_constants = (0..NUM_ROUNDS * STATE_SIZE)
            .map(|i| ["Tip5".to_string().as_bytes(), &[i as u8]].concat())
            .map(|bytes| blake3::hash(&bytes))
            .map(|hash| *hash.as_bytes())
            .map(|bytes| to_int(&bytes))
            .map(|i| (i % BFieldElement::P as u128) as u64)
            .map(BFieldElement::from_raw_u64)
            .collect_vec();

        println!(
            "In case you changed something, here are all round constants:\n{}",
            round_constants.iter().map(|c| format!("{c}")).join(", ")
        );

        (0_usize..NUM_ROUNDS * STATE_SIZE).for_each(|i| {
            assert_eq!(
                ROUND_CONSTANTS[i], round_constants[i],
                "Round constants must agree at every index, including index {i}."
            )
        });
    }

    #[test]
    fn test_fermat_cube_map_is_permutation() {
        let mut touched = [false; 256];
        for i in 0..256 {
            touched[Tip5::offset_fermat_cube_map(i) as usize] = true;
        }
        assert!(touched.iter().all(|t| *t));
        assert_eq!(Tip5::offset_fermat_cube_map(0), 0);
        assert_eq!(Tip5::offset_fermat_cube_map(255), 255);
    }

    #[test]
    fn calculate_differential_uniformity() {
        // cargo test calculate_differential_uniformity -- --include-ignored --nocapture
        // addition-differential
        let count_satisfiers_fermat = |a, b| {
            (0..(1 << 8))
                .map(|x| {
                    u16::from(
                        (256 + Tip5::offset_fermat_cube_map((x + a) & 0xff)
                            - Tip5::offset_fermat_cube_map(x))
                            & 0xff
                            == b,
                    )
                })
                .sum()
        };
        let du_fermat: u16 = (1..256)
            .into_par_iter()
            .map(|a| {
                (1..256)
                    .map(|b| count_satisfiers_fermat(a, b))
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        println!("additive differential uniformity for fermat cube map: {du_fermat}");

        // bitwise-differential
        let count_satisfiers_fermat_bitwise = |a: u16, b: u16| {
            (0..(1 << 8))
                .map(|x| {
                    u16::from(
                        (Tip5::offset_fermat_cube_map(x ^ a) ^ Tip5::offset_fermat_cube_map(x))
                            == b,
                    )
                })
                .sum::<u16>()
        };
        for a in 1..256 {
            for b in 1..256 {
                let num_satisfiers = count_satisfiers_fermat_bitwise(a, b);
                if num_satisfiers == 256 {
                    println!("a: {a}, b: {b} -> 256 satisfiers");
                }
            }
        }
        let du_fermat_bitwise: u16 = (1..256)
            .into_par_iter()
            .map(|a| {
                (1..256)
                    .map(|b| count_satisfiers_fermat_bitwise(a, b))
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        println!("bitwise differential uniformity for fermat cube map: {du_fermat_bitwise}");
    }

    #[test]
    fn calculate_approximation_quality() {
        let mut fermat_cubed = [0u16; 65536];
        let mut bfield_cubed = [0u16; 65536];
        for i in 0..65536 {
            let cubed = (i as u64) * (i as u64) * (i as u64);
            fermat_cubed[i] = (cubed % 65537) as u16;
            bfield_cubed[i] = (cubed & 0xffff) as u16;
        }
        let equal_count = fermat_cubed
            .iter()
            .zip(bfield_cubed.iter())
            .filter(|(a, b)| a == b)
            .count();
        println!("agreement with low-degree function: {equal_count}");
    }

    #[test]
    fn hash10_test_vectors() {
        let mut preimage = [BFieldElement::ZERO; RATE];
        let mut digest: [BFieldElement; Digest::LEN];
        for i in 0..6 {
            digest = Tip5::hash_10(&preimage);
            println!(
                "{:?} -> {:?}",
                preimage.iter().map(|b| b.value()).collect_vec(),
                digest.iter().map(|b| b.value()).collect_vec()
            );
            preimage[i..Digest::LEN + i].copy_from_slice(&digest);
        }
        digest = Tip5::hash_10(&preimage);
        println!(
            "{:?} -> {:?}",
            preimage.iter().map(|b| b.value()).collect_vec(),
            digest.iter().map(|b| b.value()).collect_vec()
        );
        let final_digest = [
            10869784347448351760,
            1853783032222938415,
            6856460589287344822,
            17178399545409290325,
            7650660984651717733,
        ]
        .map(BFieldElement::new);
        assert_eq!(
            digest,
            final_digest,
            "expected: {:?}\nbut got: {:?}",
            final_digest.map(|d| d.value()),
            digest.map(|d| d.value()),
        )
    }

    #[test]
    fn hash_varlen_test_vectors() {
        let mut digest_sum = [BFieldElement::ZERO; Digest::LEN];
        for i in 0..20 {
            let preimage = (0..i).map(BFieldElement::new).collect_vec();
            let digest = Tip5::hash_varlen(&preimage);
            println!(
                "{:?} -> {:?}",
                preimage.iter().map(|b| b.value()).collect_vec(),
                digest.values().iter().map(|b| b.value()).collect_vec()
            );
            digest_sum
                .iter_mut()
                .zip(digest.values().iter())
                .for_each(|(s, d)| *s += *d);
        }
        println!(
            "sum of digests: {:?}",
            digest_sum.iter().map(|b| b.value()).collect_vec()
        );
        let expected_sum = [
            7610004073009036015,
            5725198067541094245,
            4721320565792709122,
            1732504843634706218,
            259800783350288362,
        ]
        .map(BFieldElement::new);
        assert_eq!(
            expected_sum,
            digest_sum,
            "expected: {:?}\nbut got: {:?}",
            expected_sum.map(|s| s.value()),
            digest_sum.map(|s| s.value())
        );
    }

    fn manual_hash_varlen(preimage: &[BFieldElement]) -> Digest {
        let mut sponge = Tip5::init();
        sponge.pad_and_absorb_all(preimage);
        let squeeze_result = sponge.squeeze();

        Digest::new((&squeeze_result[..Digest::LEN]).try_into().unwrap())
    }

    #[test]
    fn hash_var_len_equivalence_corner_cases() {
        for preimage_length in 0..=11 {
            let preimage = vec![BFieldElement::new(42); preimage_length];
            let hash_varlen_digest = Tip5::hash_varlen(&preimage);

            let digest_through_pad_squeeze_absorb = manual_hash_varlen(&preimage);
            assert_eq!(digest_through_pad_squeeze_absorb, hash_varlen_digest);
        }
    }

    #[proptest]
    fn hash_var_len_equivalence(#[strategy(arb())] preimage: Vec<BFieldElement>) {
        let hash_varlen_digest = Tip5::hash_varlen(&preimage);
        let digest_through_pad_squeeze_absorb = manual_hash_varlen(&preimage);
        prop_assert_eq!(digest_through_pad_squeeze_absorb, hash_varlen_digest);
    }

    #[test]
    fn test_linearity_of_mds() {
        type SpongeState = [BFieldElement; STATE_SIZE];

        let mds_procedure = |state| {
            let mut sponge = Tip5 { state };
            sponge.mds_cyclomul();
            sponge.state
        };

        let a: BFieldElement = random_elements(1)[0];
        let b: BFieldElement = random_elements(1)[0];

        let mul_procedure = |u: SpongeState, v: SpongeState| -> SpongeState {
            let mul_result = u.iter().zip(&v).map(|(&uu, &vv)| a * uu + b * vv);
            mul_result.collect_vec().try_into().unwrap()
        };

        let u: SpongeState = random_elements(STATE_SIZE).try_into().unwrap();
        let v: SpongeState = random_elements(STATE_SIZE).try_into().unwrap();
        let w = mul_procedure(u, v);

        let u = mds_procedure(u);
        let v = mds_procedure(v);
        let w = mds_procedure(w);

        let w_ = mul_procedure(u, v);

        assert_eq!(w, w_);
    }

    #[test]
    fn test_mds_circulancy() {
        let mut sponge = Tip5::init();
        sponge.state = [BFieldElement::ZERO; STATE_SIZE];
        sponge.state[0] = BFieldElement::ONE;

        sponge.mds_generated();

        let mut mat_first_row = [BFieldElement::ZERO; STATE_SIZE];
        mat_first_row[0] = sponge.state[0];
        for (i, first_row_elem) in mat_first_row.iter_mut().enumerate().skip(1) {
            *first_row_elem = sponge.state[STATE_SIZE - i];
        }

        println!(
            "mds matrix first row: {:?}",
            mat_first_row.map(|b| b.value())
        );

        let initial_state: [BFieldElement; STATE_SIZE] =
            random_elements(STATE_SIZE).try_into().unwrap();

        let mut mv = [BFieldElement::ZERO; STATE_SIZE];
        for i in 0..STATE_SIZE {
            for j in 0..STATE_SIZE {
                mv[i] += mat_first_row[(STATE_SIZE - i + j) % STATE_SIZE] * initial_state[j];
            }
        }

        let mut sponge_2 = Tip5::init();
        sponge_2.state = initial_state;
        sponge_2.mds_generated();

        assert_eq!(sponge_2.state, mv);
    }

    #[test]
    fn test_complex_karatsuba() {
        const N: usize = 4;
        let mut f = [(0i64, 0i64); N];
        let mut g = [(0i64, 0i64); N];
        for i in 0..N {
            f[i].0 = i as i64;
            g[i].0 = 1;
            f[i].1 = 1;
            g[i].1 = i as i64;
        }

        let h0 = Tip5::complex_karatsuba4(f, g);
        let h1 = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 13), (0, 14), (0, 10)];

        assert_eq!(h0, h1);
    }

    #[test]
    fn test_complex_product() {
        let mut rng = thread_rng();
        let mut random_small_i64 = || (rng.next_u32() % (1 << 16)) as i64;
        for _ in 0..1000 {
            let f = (random_small_i64(), random_small_i64());
            let g = (random_small_i64(), random_small_i64());
            let h0 = Tip5::complex_product(f, g);
            let h1 = (f.0 * g.0 - f.1 * g.1, f.0 * g.1 + f.1 * g.0);
            assert_eq!(h1, h0);
        }
    }

    #[test]
    fn sample_scalars_test() {
        let mut sponge = Tip5::randomly_seeded();
        let mut product = XFieldElement::ONE;
        for amount in 0..=4 {
            let scalars = sponge.sample_scalars(amount);
            assert_eq!(amount, scalars.len());
            product *= scalars
                .into_iter()
                .fold(XFieldElement::ONE, XFieldElement::mul);
        }
        assert_ne!(product, XFieldElement::ZERO); // false failure with prob ~2^{-192}
    }

    #[test]
    fn test_mds_agree() {
        let mut rng = thread_rng();
        let initial_state: [BFieldElement; STATE_SIZE] = (0..STATE_SIZE)
            .map(|_| BFieldElement::new(rng.gen_range(0..10)))
            .collect_vec()
            .try_into()
            .unwrap();

        let mut sponge_cyclomut = Tip5 {
            state: initial_state,
        };
        let mut sponge_generated = Tip5 {
            state: initial_state,
        };

        sponge_cyclomut.mds_cyclomul();
        sponge_generated.mds_generated();

        assert_eq!(
            sponge_cyclomut,
            sponge_generated,
            "cyclomul =/= generated\n{}\n{}",
            sponge_cyclomut.state.into_iter().join(","),
            sponge_generated.state.into_iter().join(",")
        );
    }

    #[test]
    fn tip5_hasher_trait_test() {
        let mut hasher = Tip5::init();
        let data = b"hello world";
        hasher.write(data);
        assert_snapshot!(hasher.finish(), @"2267905471610932299");
    }

    #[proptest]
    fn tip5_hasher_consumes_small_data(#[filter(!#bytes.is_empty())] bytes: Vec<u8>) {
        let mut hasher = Tip5::init();
        bytes.hash(&mut hasher);

        prop_assert_ne!(Tip5::init().finish(), hasher.finish());
    }

    #[proptest]
    fn appending_small_data_to_big_data_changes_tip5_hash(
        #[any(size_range(2_000..8_000).lift())] big_data: Vec<u8>,
        #[filter(!#small_data.is_empty())] small_data: Vec<u8>,
    ) {
        let mut hasher = Tip5::init();
        big_data.hash(&mut hasher);
        let big_data_hash = hasher.finish();

        // finish doesn't terminate the hasher; see it's documentation
        small_data.hash(&mut hasher);
        let all_data_hash = hasher.finish();

        prop_assert_ne!(big_data_hash, all_data_hash);
    }

    #[proptest]
    fn tip5_trace_starts_with_initial_state_and_is_equivalent_to_permutation(
        #[strategy(arb())] mut tip5: Tip5,
    ) {
        let [first, .., last] = tip5.clone().trace();
        prop_assert_eq!(first, tip5.state);

        tip5.permutation();
        prop_assert_eq!(last, tip5.state);
    }
}
