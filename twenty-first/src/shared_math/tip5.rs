use itertools::Itertools;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
pub use crate::shared_math::rescue_prime_digest::{Digest, DIGEST_LENGTH};

use crate::util_types::algebraic_hasher::{AlgebraicHasher, Domain, SpongeHasher};

use super::{
    mds::generated_function,
    x_field_element::{XFieldElement, EXTENSION_DEGREE},
};

pub const STATE_SIZE: usize = 16;
pub const NUM_SPLIT_AND_LOOKUP: usize = 4;
pub const LOG2_STATE_SIZE: usize = 4;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;
pub const NUM_ROUNDS: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tip5State {
    pub state: [BFieldElement; STATE_SIZE],
}

impl Tip5State {
    #[inline]
    pub const fn new(domain: Domain) -> Self {
        use Domain::*;

        let mut state = [BFIELD_ZERO; STATE_SIZE];

        match domain {
            VariableLength => (),
            FixedLength => {
                let mut i = RATE;
                while i < STATE_SIZE {
                    state[i] = BFIELD_ONE;
                    i += 1;
                }
            }
        }

        Self { state }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tip5 {}

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

impl Tip5 {
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
    fn mds_cyclomul(state: &mut [BFieldElement; STATE_SIZE]) {
        let mut result = [BFieldElement::zero(); STATE_SIZE];

        let mut lo: [i64; STATE_SIZE] = [0; STATE_SIZE];
        let mut hi: [i64; STATE_SIZE] = [0; STATE_SIZE];
        for (i, b) in state.iter().enumerate() {
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
        *state = result;
    }

    #[inline(always)]
    fn mds_generated(state: &mut [BFieldElement; STATE_SIZE]) {
        let mut lo: [u64; STATE_SIZE] = [0; STATE_SIZE];
        let mut hi: [u64; STATE_SIZE] = [0; STATE_SIZE];
        for i in 0..STATE_SIZE {
            let b = state[i].raw_u64();
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

            state[r] = BFieldElement::from_raw_u64(if over { res + 0xffffffffu64 } else { res });
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn sbox_layer(state: &mut [BFieldElement; STATE_SIZE]) {
        // lookup
        // state.iter_mut().take(NUM_SPLIT_AND_LOOKUP).for_each(|s| {
        //     Self::split_and_lookup(s);
        // });
        for i in 0..NUM_SPLIT_AND_LOOKUP {
            Self::split_and_lookup(&mut state[i]);
        }

        // power
        // for st in state.iter_mut().skip(NUM_SPLIT_AND_LOOKUP) {
        //     let sq = *st * *st;
        //     let qu = sq * sq;
        //     *st *= sq * qu;
        // }
        for i in NUM_SPLIT_AND_LOOKUP..STATE_SIZE {
            let sq = state[i] * state[i];
            let qu = sq * sq;
            state[i] *= sq * qu;
        }
    }

    #[inline(always)]
    fn round(sponge: &mut Tip5State, round_index: usize) {
        Self::sbox_layer(&mut sponge.state);

        // Self::mds_cyclomul(&mut sponge.state);
        Self::mds_generated(&mut sponge.state);

        for i in 0..STATE_SIZE {
            sponge.state[i] += ROUND_CONSTANTS[round_index * STATE_SIZE + i];
        }
    }

    // permutation
    #[inline(always)]
    fn permutation(sponge: &mut Tip5State) {
        for i in 0..NUM_ROUNDS {
            Self::round(sponge, i);
        }
    }

    /// Functionally equivalent to [`permutation`](Self::permutation). Returns the trace of
    /// applying the permutation; that is, the initial state of the sponge as well as its state
    /// after each round.
    pub fn trace(sponge: &mut Tip5State) -> [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS] {
        let mut trace = [[BFIELD_ZERO; STATE_SIZE]; 1 + NUM_ROUNDS];

        trace[0] = sponge.state;
        for i in 0..NUM_ROUNDS {
            Self::round(sponge, i);
            trace[1 + i] = sponge.state;
        }

        trace
    }

    /// hash_10
    /// Hash 10 elements, or two digests. There is no padding because
    /// the input length is fixed.
    pub fn hash_10(input: &[BFieldElement; 10]) -> [BFieldElement; DIGEST_LENGTH] {
        let mut sponge = Tip5State::new(Domain::FixedLength);

        // absorb once
        sponge.state[..10].copy_from_slice(input);

        // apply permutation
        Self::permutation(&mut sponge);

        // squeeze once
        sponge.state[..DIGEST_LENGTH].try_into().unwrap()
    }
}

impl AlgebraicHasher for Tip5 {
    fn hash_pair(left: &Digest, right: &Digest) -> Digest {
        let mut input = [BFIELD_ZERO; 10];
        input[..DIGEST_LENGTH].copy_from_slice(&left.values());
        input[DIGEST_LENGTH..].copy_from_slice(&right.values());
        Digest::new(Tip5::hash_10(&input))
    }

    /// Produce `num_elements` random [XFieldElement] values.
    ///
    /// - The randomness depends on `state`.
    ///
    /// Since [RATE] is not divisible by [EXTENSION_DEGREE], produce as many [XFieldElement] per
    /// `squeeze` as possible, and spill the remaining element(s). This causes some internal
    /// fragmentation, but it greatly simplifies building [AlgebraicHasher::sample_xfield()] on
    /// Triton VM.
    ///
    fn sample_scalars(state: &mut Self::SpongeState, num_elements: usize) -> Vec<XFieldElement> {
        let xfes_per_squeeze = Self::RATE / EXTENSION_DEGREE; // 3
        let num_squeezes = (num_elements + xfes_per_squeeze - 1) / xfes_per_squeeze;
        (0..num_squeezes)
            .flat_map(|_| {
                Self::squeeze(state)
                    .into_iter()
                    .take(xfes_per_squeeze * EXTENSION_DEGREE)
                    .collect_vec()
            })
            .collect_vec()
            .chunks(3)
            .take(num_elements)
            .map(|elem| XFieldElement::new([elem[0], elem[1], elem[2]]))
            .collect_vec()
    }
}

impl SpongeHasher for Tip5 {
    const RATE: usize = RATE;
    type SpongeState = Tip5State;

    fn init() -> Self::SpongeState {
        Tip5State::new(Domain::VariableLength)
    }

    fn absorb(sponge: &mut Self::SpongeState, input: &[BFieldElement; RATE]) {
        // absorb
        sponge.state[..RATE]
            .iter_mut()
            .zip_eq(input.iter())
            .for_each(|(a, &b)| *a += b);

        Tip5::permutation(sponge);
    }

    fn squeeze(sponge: &mut Self::SpongeState) -> [BFieldElement; RATE] {
        // squeeze
        let produce: [BFieldElement; RATE] = (&sponge.state[..RATE]).try_into().unwrap();

        Tip5::permutation(sponge);

        produce
    }
}

#[cfg(test)]
mod tip5_tests {

    use itertools::Itertools;
    use num_traits::One;
    use num_traits::Zero;
    use rand::thread_rng;
    use rand::RngCore;
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::other::random_elements;
    use crate::shared_math::rescue_prime_digest::DIGEST_LENGTH;
    use crate::shared_math::tip5::Tip5;
    use crate::shared_math::tip5::LOOKUP_TABLE;
    use crate::shared_math::tip5::NUM_ROUNDS;
    use crate::shared_math::tip5::ROUND_CONSTANTS;
    use crate::shared_math::tip5::STATE_SIZE;
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::util_types::algebraic_hasher::AlgebraicHasher;
    use crate::util_types::algebraic_hasher::SpongeHasher;
    use std::ops::Mul;

    use super::Tip5State;
    use super::RATE;

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
            .map(|i| ["Tip5".to_string().as_bytes(), &[(i as u8)]].concat())
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
    #[ignore = "used for calculating parameters"]
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
    #[ignore = "used for calculating parameters"]
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
    #[ignore = "used for calculating parameters"]
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
        let mut preimage = [BFieldElement::zero(); RATE];
        let mut digest: [BFieldElement; DIGEST_LENGTH];
        for i in 0..6 {
            digest = Tip5::hash_10(&preimage);
            println!(
                "{:?} -> {:?}",
                preimage.iter().map(|b| b.value()).collect_vec(),
                digest.iter().map(|b| b.value()).collect_vec()
            );
            preimage[i..DIGEST_LENGTH + i].copy_from_slice(&digest);
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
        let mut digest_sum = [BFieldElement::zero(); DIGEST_LENGTH];
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
            6483667016211232820,
            1120398765245047030,
            9375424207996641714,
            17770540514093105302,
            17391179748947955,
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

    #[test]
    fn test_linearity_of_mds() {
        let mds_procedure = Tip5::mds_cyclomul;
        // let mds_procedure = Tip5::mds_noswap;
        let a: BFieldElement = random_elements(1)[0];
        let b: BFieldElement = random_elements(1)[0];
        let mut u: [BFieldElement; STATE_SIZE] = random_elements(STATE_SIZE).try_into().unwrap();
        let mut v: [BFieldElement; STATE_SIZE] = random_elements(STATE_SIZE).try_into().unwrap();

        let mut w: [BFieldElement; STATE_SIZE] = u
            .iter()
            .zip(v.iter())
            .map(|(uu, vv)| a * *uu + b * *vv)
            .collect::<Vec<BFieldElement>>()
            .try_into()
            .unwrap();

        mds_procedure(&mut u);
        mds_procedure(&mut v);
        mds_procedure(&mut w);

        let w_: [BFieldElement; STATE_SIZE] = u
            .iter()
            .zip(v.iter())
            .map(|(uu, vv)| a * *uu + b * *vv)
            .collect::<Vec<BFieldElement>>()
            .try_into()
            .unwrap();

        assert_eq!(w, w_);
    }

    #[test]
    fn test_mds_circulancy() {
        let mut e1 = [BFieldElement::zero(); STATE_SIZE];
        e1[0] = BFieldElement::one();

        // let mds_procedure = Tip5::mds_al_kindi;
        // let mds_procedure = Tip5::mds_cyclomul;
        let mds_procedure = Tip5::mds_generated;

        mds_procedure(&mut e1);

        let mut mat_first_row = [BFieldElement::zero(); STATE_SIZE];
        mat_first_row[0] = e1[0];
        for i in 1..STATE_SIZE {
            mat_first_row[i] = e1[STATE_SIZE - i];
        }

        println!(
            "mds matrix first row: {:?}",
            mat_first_row.map(|b| b.value())
        );

        let mut vec: [BFieldElement; STATE_SIZE] = random_elements(STATE_SIZE).try_into().unwrap();

        let mut mv = [BFieldElement::zero(); STATE_SIZE];
        for i in 0..STATE_SIZE {
            for j in 0..STATE_SIZE {
                mv[i] += mat_first_row[(STATE_SIZE - i + j) % STATE_SIZE] * vec[j];
            }
        }

        mds_procedure(&mut vec);

        assert_eq!(vec, mv);
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
        for _ in 0..1000 {
            let f = (rng.next_u32() as i64, rng.next_u32() as i64);
            let g = (rng.next_u32() as i64, rng.next_u32() as i64);
            let h0 = Tip5::complex_product(f, g);
            let h1 = (f.0 * g.0 - f.1 * g.1, f.0 * g.1 + f.1 * g.0);
            assert_eq!(h1, h0);
        }
    }

    fn seed_tip5(sponge: &mut Tip5State) {
        let mut rng = thread_rng();
        Tip5::absorb(
            sponge,
            &(0..RATE)
                .map(|_| BFieldElement::new(rng.next_u64()))
                .collect_vec()
                .try_into()
                .unwrap(),
        );
    }

    #[test]
    fn sample_scalars_test() {
        let amounts = [0, 1, 2, 3, 4];
        let mut sponge = Tip5::init();
        seed_tip5(&mut sponge);
        let mut product = XFieldElement::one();
        for amount in amounts {
            let scalars = Tip5::sample_scalars(&mut sponge, amount);
            assert_eq!(amount, scalars.len());
            product *= scalars
                .into_iter()
                .fold(XFieldElement::one(), XFieldElement::mul);
        }
        assert_ne!(product, XFieldElement::zero()); // false failure with prob ~2^{-192}
    }

    #[test]
    fn test_mds_agree() {
        let mut rng = thread_rng();
        // let vector: [i64; 16] = (0..16)
        //     .map(|_| (rng.next_u64() & 0xffffffff) as i64)
        //     .collect_vec()
        //     .try_into()
        //     .unwrap();
        let vector: [BFieldElement; 16] = (0..16)
            .map(|_| BFieldElement::new(rng.next_u64() % 10))
            .collect_vec()
            .try_into()
            .unwrap();
        // let mut vector = [BFieldElement::zero(); 16];
        // vector[0] = BFieldElement::one();

        let mut cyclomul = vector;
        Tip5::mds_cyclomul(&mut cyclomul);
        let mut generated = vector;
        Tip5::mds_generated(&mut generated);

        assert_eq!(
            cyclomul,
            generated,
            "cyclomul =/= generated\n{}\n{}",
            cyclomul.map(|c| c.to_string()).join(","),
            generated.map(|c| c.to_string()).join(",")
        );
    }
}
