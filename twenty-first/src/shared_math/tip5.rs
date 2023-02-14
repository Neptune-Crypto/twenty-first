use itertools::Itertools;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use crate::shared_math::rescue_prime_digest::{Digest, DIGEST_LENGTH};

use crate::util_types::algebraic_hasher::{AlgebraicHasher, Domain, SpongeHasher};

pub const STATE_SIZE: usize = 16;
pub const NUM_SPLIT_AND_LOOKUP: usize = 4;
pub const LOG2_STATE_SIZE: usize = 4;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;
pub const NUM_ROUNDS: usize = 5;

// Note this NOT an MDS matrix. Finding an MDS matrix which has small powers of two partial FFT16
// is a requirement for what follows.
// Al's magic constants:
// Partial FFT16 of circular([3, 1, 11, 6, 10, 4, 2, 2, 5, 3, 13, 4, 14, 8, 6, 4])
// const MDS_FREQ_BLOCK_ONE: [i64; 4] = [8, 4, 8, 4];
// const MDS_FREQ_BLOCK_TWO: [(i64, i64); 4] = [(-1, 2), (-1, 2), (-1, 2), (1, 1)];
// const MDS_FREQ_BLOCK_THREE: [i64; 4] = [-4, -2, 4, 1];

// Alan's magic constants:
// corresponds to matrix.circulant( [ 61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845 ] )
// which is the SHA256 of "Tip5"
const MDS_FREQ_BLOCK_ONE: [i64; 4] = [137828, 131562, 150764, 104603];
const MDS_FREQ_BLOCK_TWO: [(i64, i64); 4] = [
    (4451, 4567),
    (-26413, 16445),
    (-12601, -27067),
    (-7078, 5811),
];
const MDS_FREQ_BLOCK_THREE: [i64; 4] = [98878, -74304, -10562, 44845];

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

    #[allow(clippy::many_single_char_names)]
    #[inline]
    fn ntt_noswap(x: &mut [BFieldElement]) {
        // const POWERS_OF_OMEGA_BITREVERSED: [BFieldElement; 8] = [
        //     BFieldElement::new(1),
        //     BFieldElement::new(281474976710656),
        //     BFieldElement::new(18446744069397807105),
        //     BFieldElement::new(18446742969902956801),
        //     BFieldElement::new(17293822564807737345),
        //     BFieldElement::new(4096),
        //     BFieldElement::new(4503599626321920),
        //     BFieldElement::new(18446744000695107585),
        // ];
        const POWERS_OF_OMEGA_BITREVERSED: [BFieldElement; 8] = [
            BFieldElement::new(1),
            BFieldElement::new(281474976710656),
            BFieldElement::new(16777216),
            BFieldElement::new(1099511627520),
            BFieldElement::new(4096),
            BFieldElement::new(1152921504606846976),
            BFieldElement::new(68719476736),
            BFieldElement::new(4503599626321920),
        ];
        const SHIFTS: [usize; 8] = [0, 48, 24, 72, 12, 60, 36, 84];
        let powers_of_omega = SHIFTS.map(|sh| BFieldElement::new(1) << sh);
        assert_eq!(powers_of_omega, POWERS_OF_OMEGA_BITREVERSED);

        // outer loop iteration 1
        for j in 0..8 {
            let u = x[j];
            let v = x[j + 8] * BFieldElement::one();
            x[j] = u + v;
            x[j + 8] = u - v;
        }

        // outer loop iteration 2
        for (i, zeta) in SHIFTS.iter().enumerate().take(2) {
            let s = i * 8;
            for j in s..(s + 4) {
                let u = x[j];
                let v = x[j + 4] << *zeta;
                x[j] = u + v;
                x[j + 4] = u - v;
            }
        }

        // outer loop iteration 3
        for (i, zeta) in SHIFTS.iter().enumerate().take(4) {
            let s = i * 4;
            for j in s..(s + 2) {
                let u = x[j];
                let v = x[j + 2] << *zeta;
                x[j] = u + v;
                x[j + 2] = u - v;
            }
        }

        // outer loop iteration 4
        for (i, zeta) in SHIFTS.iter().enumerate().take(8) {
            let s = i * 2;
            let u = x[s];
            let v = x[s + 1] << *zeta;
            x[s] = u + v;
            x[s + 1] = u - v;
        }
    }

    #[allow(clippy::many_single_char_names)]
    #[inline]
    fn intt_noswap(x: &mut [BFieldElement]) {
        // const POWERS_OF_OMEGA_INVERSE: [BFieldElement; 8] = [
        //     BFieldElement::new(1),
        //     BFieldElement::new(68719476736),
        //     BFieldElement::new(1099511627520),
        //     BFieldElement::new(18446744069414580225),
        //     BFieldElement::new(18446462594437873665),
        //     BFieldElement::new(18442240469788262401),
        //     BFieldElement::new(16777216),
        //     BFieldElement::new(1152921504606846976),
        // ];
        const POWERS_OF_OMEGA_INVERSE: [BFieldElement; 8] = [
            BFieldElement::new(1),
            BFieldElement::new(18442240469788262401),
            BFieldElement::new(18446742969902956801),
            BFieldElement::new(17293822564807737345),
            BFieldElement::new(18446462594437873665),
            BFieldElement::new(18446744000695107585),
            BFieldElement::new(18446744069397807105),
            BFieldElement::new(18446744069414580225),
        ];
        const SHIFTS: [usize; 8] = [0, 12, 24, 36, 48, 60, 72, 84];
        let powers_of_omega = SHIFTS.map(|sh| BFieldElement::new(1) >> sh);
        assert_eq!(
            powers_of_omega, POWERS_OF_OMEGA_INVERSE,
            "left hand side: {powers_of_omega:?}, versus right hand side: {POWERS_OF_OMEGA_INVERSE:?}"
        );

        // outer loop iteration 1
        {
            // while k < STATE_SIZE as usize
            // inner loop iteration 1
            {
                let u = x[1];
                let v = x[0];
                x[1] = v - u;
                x[0] = v + u;
            }

            // inner loop iteration 2
            {
                let u = x[2 + 1];
                let v = x[2];
                x[2 + 1] = v - u;
                x[2] = v + u;
            }

            // inner loop iteration 3
            {
                let u = x[4 + 1];
                let v = x[4];
                x[4 + 1] = v - u;
                x[4] = v + u;
            }

            // inner loop iteration 4
            {
                let u = x[6 + 1];
                let v = x[6];
                x[6 + 1] = v - u;
                x[6] = v + u;
            }

            // inner loop iteration 5
            {
                let u = x[8 + 1];
                let v = x[8];
                x[8 + 1] = v - u;
                x[8] = v + u;
            }

            // inner loop iteration 6
            {
                let u = x[10 + 1];
                let v = x[10];
                x[10 + 1] = v - u;
                x[10] = v + u;
            }

            // inner loop iteration 7
            {
                let u = x[12 + 1];
                let v = x[12];
                x[12 + 1] = v - u;
                x[12] = v + u;
            }

            // inner loop iteration 7
            {
                let u = x[14 + 1];
                let v = x[14];
                x[14 + 1] = v - u;
                x[14] = v + u;
            }
        }

        // outer loop iteration 2
        {
            // while k < STATE_SIZE as usize
            // inner loop iteration 1
            {
                for j in 0..2 {
                    let zeta = SHIFTS[4 * j];
                    {
                        let u = x[j + 2] >> zeta;
                        let v = x[j];
                        x[j + 2] = v - u;
                        x[j] = v + u;
                    }
                    // inner loop iteration 2
                    {
                        let u = x[4 + j + 2] >> zeta;
                        let v = x[4 + j];
                        x[4 + j + 2] = v - u;
                        x[4 + j] = v + u;
                    }
                    // inner loop iteration 3
                    {
                        let u = x[8 + j + 2] >> zeta;
                        let v = x[8 + j];
                        x[8 + j + 2] = v - u;
                        x[8 + j] = v + u;
                    }
                    // inner loop iteration 4
                    {
                        let u = x[12 + j + 2] >> zeta;
                        let v = x[12 + j];
                        x[12 + j + 2] = v - u;
                        x[12 + j] = v + u;
                    }
                }
            }
        }

        // outer loop iteration 3
        {
            // while k < STATE_SIZE as usize
            {
                for j in 0..4 {
                    let zeta = SHIFTS[2 * j];
                    // inner loop iteration 1
                    {
                        let u = x[j + 4] >> zeta;
                        let v = x[j];
                        x[j + 4] = v - u;
                        x[j] = v + u;
                    }
                    // inner loop iteration 2
                    {
                        let u = x[8 + j + 4] >> zeta;
                        let v = x[8 + j];
                        x[8 + j + 4] = v - u;
                        x[8 + j] = v + u;
                    }
                }
            }
        }

        // outer loop iteration 4
        {
            for j in 0..8 {
                let zeta = SHIFTS[j];
                let u = x[j + 8] >> zeta;
                let v = x[j];
                x[j + 8] = v - u;
                x[j] = v + u;
            }
        }
    }

    #[inline]
    pub fn mds_noswap(state: &mut [BFieldElement; STATE_SIZE]) {
        // without bitreverse: [4, 1, 3, 6, 4, 0, 0, 4, 1, 5, 7, 2, 3, 2, 5, 1]
        const SHIFTS: [u8; STATE_SIZE] = [4, 1, 4, 3, 3, 7, 0, 5, 1, 5, 0, 2, 6, 2, 4, 1];
        let mut array: [u128; STATE_SIZE] = [0; STATE_SIZE];
        Self::ntt_noswap(state);

        for i in 0..STATE_SIZE {
            array[i] = state[i].raw_u128() << SHIFTS[i];
        }
        let mut reduced = [0u64; STATE_SIZE];
        for i in 0..STATE_SIZE {
            reduced[i] = BFieldElement::montyred(array[i]);
        }
        for i in 0..16 {
            state[i] = BFieldElement::from_raw_u64(reduced[i]);
        }

        Self::intt_noswap(state);
    }

    #[inline(always)]
    #[allow(clippy::shadow_unrelated)]
    pub fn mds_multiply_freq(state: [i64; 16]) -> [i64; 16] {
        let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15] = state;

        let (u0, u1, u2) = Self::fft4_real([s0, s4, s8, s12]);
        let (u4, u5, u6) = Self::fft4_real([s1, s5, s9, s13]);
        let (u8, u9, u10) = Self::fft4_real([s2, s6, s10, s14]);
        let (u12, u13, u14) = Self::fft4_real([s3, s7, s11, s15]);

        let [v0, v4, v8, v12] = Self::block1([u0, u4, u8, u12], MDS_FREQ_BLOCK_ONE);
        let [v1, v5, v9, v13] = Self::block2([u1, u5, u9, u13], MDS_FREQ_BLOCK_TWO);
        let [v2, v6, v10, v14] = Self::block3([u2, u6, u10, u14], MDS_FREQ_BLOCK_THREE);
        // The 4th block is not computed as it is similar to the 2nd one, up to complex conjugation,
        // and is, due to the use of the real FFT and iFFT, redundant.

        let [s0, s4, s8, s12] = Self::ifft4_real((v0, v1, v2));
        let [s1, s5, s9, s13] = Self::ifft4_real((v4, v5, v6));
        let [s2, s6, s10, s14] = Self::ifft4_real((v8, v9, v10));
        let [s3, s7, s11, s15] = Self::ifft4_real((v12, v13, v14));

        [
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
        ]
    }

    // #[inline(always)]
    // fn karatsuba2(x: [i64; 2], y: [i64; 2]) -> [i64; 3] {
    //     let z0 = x[0] * y[0];
    //     let z2 = x[1] * y[1];
    //     let z1 = (x[0] + x[1]) * (y[0] + y[1]) - z0 - z2;
    //     [z0, z1, z2]
    // }

    #[inline(always)]
    fn karatsuba2(a: [i64; 2], b: [i64; 2]) -> [i64; 3] {
        // let lo = a[0] * b[0];
        // let hi = a[1] * b[1];
        // [lo, (a[0] + a[1]) * (b[0] + b[1]) - lo - hi, hi]
        [a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[1] * b[1]]
    }

    #[inline(always)]
    fn karatsuba4(a: [i64; 4], b: [i64; 4]) -> [i64; 7] {
        let lo = Self::karatsuba2([a[0], a[1]], [b[0], b[1]]);
        let hi = Self::karatsuba2([a[2], a[3]], [b[2], b[3]]);
        let li = Self::karatsuba2([a[0] + a[2], a[1] + a[3]], [b[0] + b[2], b[1] + b[3]]);

        [
            lo[0],
            lo[1],
            lo[2] + li[0] - lo[0] - hi[0],
            li[1] - lo[1] - hi[1],
            li[2] - lo[2] - hi[2] + hi[0],
            hi[1],
            hi[2],
        ]
    }

    // We use the real FFT to avoid redundant computations. See https://www.mdpi.com/2076-3417/12/9/4700
    #[inline(always)]
    fn fft2_real(x: [i64; 2]) -> [i64; 2] {
        [(x[0] + x[1]), (x[0] - x[1])]
    }

    // #[inline(always)]
    // fn ifft2_real(y: [i64; 2]) -> [i64; 2] {
    //     // We avoid divisions by 2 by appropriately scaling the MDS matrix constants.
    //     [(y[0] + y[1]), (y[0] - y[1])]
    // }

    #[inline(always)]
    fn fft4_real(x: [i64; 4]) -> (i64, (i64, i64), i64) {
        let [z0, z2] = Self::fft2_real([x[0], x[2]]);
        let [z1, z3] = Self::fft2_real([x[1], x[3]]);
        let y0 = z0 + z1;
        let y1 = (z2, -z3);
        let y2 = z0 - z1;
        (y0, y1, y2)
    }

    #[inline(always)]
    fn ifft4_real(y: (i64, (i64, i64), i64)) -> [i64; 4] {
        // In calculating 'z0' and 'z1', division by 2 is avoided by appropriately scaling
        // the MDS matrix constants.
        let z0 = y.0 + y.2;
        let z1 = y.0 - y.2;
        let z2 = y.1 .0;
        let z3 = -y.1 .1;

        let [x0, x2] = Self::fft2_real([z0, z2]);
        let [x1, x3] = Self::fft2_real([z1, z3]);

        [x0, x1, x2, x3]
    }

    #[inline(always)]
    fn block1(x: [i64; 4], y: [i64; 4]) -> [i64; 4] {
        let [x0, x1, x2, x3] = x;
        let [y0, y1, y2, y3] = y;
        let z0 = x0 * y0 + x1 * y3 + x2 * y2 + x3 * y1;
        let z1 = x0 * y1 + x1 * y0 + x2 * y3 + x3 * y2;
        let z2 = x0 * y2 + x1 * y1 + x2 * y0 + x3 * y3;
        let z3 = x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0;

        [z0, z1, z2, z3]
    }

    #[inline(always)]
    #[allow(clippy::shadow_unrelated)]
    fn block2(x: [(i64, i64); 4], y: [(i64, i64); 4]) -> [(i64, i64); 4] {
        let [(x0r, x0i), (x1r, x1i), (x2r, x2i), (x3r, x3i)] = x;
        let [(y0r, y0i), (y1r, y1i), (y2r, y2i), (y3r, y3i)] = y;
        let x0s = x0r + x0i;
        let x1s = x1r + x1i;
        let x2s = x2r + x2i;
        let x3s = x3r + x3i;
        let y0s = y0r + y0i;
        let y1s = y1r + y1i;
        let y2s = y2r + y2i;
        let y3s = y3r + y3i;

        // Compute x0​y0​−ix1​y3​−ix2​y2​−ix3​y1​ using Karatsuba
        let m0 = (x0r * y0r, x0i * y0i);
        let m1 = (x1r * y3r, x1i * y3i);
        let m2 = (x2r * y2r, x2i * y2i);
        let m3 = (x3r * y1r, x3i * y1i);
        let z0r = (m0.0 - m0.1)
            + (x1s * y3s - m1.0 - m1.1)
            + (x2s * y2s - m2.0 - m2.1)
            + (x3s * y1s - m3.0 - m3.1);
        let z0i = (x0s * y0s - m0.0 - m0.1) + (-m1.0 + m1.1) + (-m2.0 + m2.1) + (-m3.0 + m3.1);
        let z0 = (z0r, z0i);

        // Compute x0​y1​+x1​y0​−ix2​y3​−ix3​y2​ using Karatsuba
        let m0 = (x0r * y1r, x0i * y1i);
        let m1 = (x1r * y0r, x1i * y0i);
        let m2 = (x2r * y3r, x2i * y3i);
        let m3 = (x3r * y2r, x3i * y2i);
        let z1r =
            (m0.0 - m0.1) + (m1.0 - m1.1) + (x2s * y3s - m2.0 - m2.1) + (x3s * y2s - m3.0 - m3.1);
        let z1i =
            (x0s * y1s - m0.0 - m0.1) + (x1s * y0s - m1.0 - m1.1) + (-m2.0 + m2.1) + (-m3.0 + m3.1);
        let z1 = (z1r, z1i);

        // Compute x0​y2​+x1​y1​+x2​y0​−ix3​y3​​ using Karatsuba
        let m0 = (x0r * y2r, x0i * y2i);
        let m1 = (x1r * y1r, x1i * y1i);
        let m2 = (x2r * y0r, x2i * y0i);
        let m3 = (x3r * y3r, x3i * y3i);
        let z2r = (m0.0 - m0.1) + (m1.0 - m1.1) + (m2.0 - m2.1) + (x3s * y3s - m3.0 - m3.1);
        let z2i = (x0s * y2s - m0.0 - m0.1)
            + (x1s * y1s - m1.0 - m1.1)
            + (x2s * y0s - m2.0 - m2.1)
            + (-m3.0 + m3.1);
        let z2 = (z2r, z2i);

        // Compute x0​y3​+x1​y2​+x2​y1​+x3​y0​​​ using Karatsuba
        let m0 = (x0r * y3r, x0i * y3i);
        let m1 = (x1r * y2r, x1i * y2i);
        let m2 = (x2r * y1r, x2i * y1i);
        let m3 = (x3r * y0r, x3i * y0i);
        let z3r = (m0.0 - m0.1) + (m1.0 - m1.1) + (m2.0 - m2.1) + (m3.0 - m3.1);
        let z3i = (x0s * y3s - m0.0 - m0.1)
            + (x1s * y2s - m1.0 - m1.1)
            + (x2s * y1s - m2.0 - m2.1)
            + (x3s * y0s - m3.0 - m3.1);
        let z3 = (z3r, z3i);

        [z0, z1, z2, z3]
    }

    #[inline(always)]
    fn block3(x: [i64; 4], y: [i64; 4]) -> [i64; 4] {
        let [x0, x1, x2, x3] = x;
        let [y0, y1, y2, y3] = y;

        let z0 = x0 * y0 - x1 * y3 - x2 * y2 - x3 * y1;
        let z1 = x0 * y1 + x1 * y0 - x2 * y3 - x3 * y2;
        let z2 = x0 * y2 + x1 * y1 + x2 * y0 - x3 * y3;
        let z3 = x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0;

        [z0, z1, z2, z3]
    }

    #[inline(always)]
    fn mds_split(state: &mut [BFieldElement; STATE_SIZE]) {
        let mut result = [BFieldElement::zero(); STATE_SIZE];

        let mut lo: [i64; STATE_SIZE] = [0; STATE_SIZE];
        let mut hi: [i64; STATE_SIZE] = [0; STATE_SIZE];
        for (i, b) in state.iter().enumerate() {
            hi[i] = (b.raw_u64() >> 32) as i64;
            lo[i] = (b.raw_u64() as u32) as i64;
        }

        lo = Self::mds_multiply_freq(lo);
        hi = Self::mds_multiply_freq(hi);

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

        return hh;
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

        return hh;
    }

    #[inline(always)]
    fn complex_negacyclomul2(f: [i64; 2], g: [i64; 2]) -> [i64; 2] {
        const N: usize = 1;

        let f0 = (f[0], -f[1]);
        let g0 = (g[0], -g[1]);

        let h0 = Self::complex_product(f0, g0);

        [h0.0, -h0.1]
    }

    #[inline(always)]
    fn fast_negacyclomul8(f: [i64; 8], g: [i64; 8]) -> [i64; 8] {
        const N: usize = 4;
        let mut ff_lo = [0i64; N];
        let mut gg_lo = [0i64; N];
        let mut ff_hi = [0i64; N];
        let mut gg_hi = [0i64; N];
        let mut ff_li = [0i64; N];
        let mut gg_li = [0i64; N];

        for i in 0..N {
            ff_lo[i] = f[i];
            ff_hi[i] = f[i + N];
            ff_li[i] = f[i] + f[i + N];
            gg_lo[i] = g[i];
            gg_hi[i] = g[i + N];
            gg_li[i] = g[i] + g[i + N];
        }

        let hh_lo = Self::karatsuba4(ff_lo, gg_lo);
        let hh_hi = Self::karatsuba4(ff_hi, gg_hi);
        let hh_li = Self::karatsuba4(ff_li, gg_li);

        let mut hh = [0i64; 4 * N - 1];
        for i in 0..(2 * N - 1) {
            hh[i] += hh_lo[i];
            hh[i + N] += hh_li[i] - hh_lo[i] - hh_hi[i];
            hh[i + N * 2] += hh_hi[i];
        }

        let mut result = [0i64; 2 * N];
        for i in 0..2 * N - 1 {
            result[i] = hh[i] - hh[2 * N + i];
        }
        result[2 * N - 1] = hh[2 * N - 1];

        result
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
        let hh_hi = Self::fast_negacyclomul4(ff_hi, gg_hi);

        let mut hh = [0i64; 2 * N];
        for i in 0..N {
            hh[i] = (hh_lo[i] + hh_hi[i]) >> 1;
            hh[i + N] = (hh_lo[i] - hh_hi[i]) >> 1;
        }

        hh
    }

    #[inline(always)]
    fn fast_negacyclomul4(f: [i64; 4], g: [i64; 4]) -> [i64; 4] {
        const N: usize = 2;
        let mut ff_lo = [0i64; N];
        let mut gg_lo = [0i64; N];
        let mut ff_hi = [0i64; N];
        let mut gg_hi = [0i64; N];
        let mut ff_li = [0i64; N];
        let mut gg_li = [0i64; N];

        for i in 0..N {
            ff_lo[i] = f[i];
            ff_hi[i] = f[i + N];
            ff_li[i] = f[i] + f[i + N];
            gg_lo[i] = g[i];
            gg_hi[i] = g[i + N];
            gg_li[i] = g[i] + g[i + N];
        }

        let hh_lo = Self::karatsuba2(ff_lo, gg_lo);
        let hh_hi = Self::karatsuba2(ff_hi, gg_hi);
        let hh_li = Self::karatsuba2(ff_li, gg_li);

        [
            hh_lo[0] - (hh_li[2] - hh_lo[2] - hh_hi[2] + hh_hi[0]),
            hh_lo[1] - hh_hi[1],
            hh_lo[2] + hh_li[0] - hh_lo[0] - hh_hi[0] - hh_hi[2],
            hh_li[1] - hh_lo[1] - hh_hi[1],
        ]
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
        let hh_hi = Self::fast_negacyclomul2(ff_hi, gg_hi);

        let mut hh = [0i64; 2 * N];
        for i in 0..N {
            hh[i] = (hh_lo[i] + hh_hi[i]) >> 1;
            hh[i + N] = (hh_lo[i] - hh_hi[i]) >> 1;
        }

        hh
    }

    #[inline(always)]
    fn fast_negacyclomul2(f: [i64; 2], g: [i64; 2]) -> [i64; 2] {
        let ff_lo = f[0];
        let ff_hi = f[1];
        let ff_li = f[0] + f[1];
        let gg_lo = g[0];
        let gg_hi = g[1];
        let gg_li = g[0] + g[1];

        let hh_lo = ff_lo * gg_lo;
        let hh_hi = ff_hi * gg_hi;
        let hh_li = ff_li * gg_li;

        [hh_lo - hh_hi, hh_li - hh_lo - hh_hi]
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
    fn mds_cyclomul(state: &mut [BFieldElement; STATE_SIZE]) {
        // constants: Sha256("Tip5")
        // const CONSTANTS: [i64; STATE_SIZE] = [
        //     61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901,
        //     12021, 59689, 26798, 17845,
        // ];
        // const CONSTANTS: [i64; STATE_SIZE] = [
        //     -2, 7, -3, -11, -11, 9, -2, -6, -6, 5, -1, -7, -13, 11, 2, -8,
        // ];
        const CONSTANTS: [i64; STATE_SIZE] = [
            61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901,
            12021, 59689, 26798, 17845,
        ];
        let mut result = [BFieldElement::zero(); STATE_SIZE];

        let mut lo: [i64; STATE_SIZE] = [0; STATE_SIZE];
        let mut hi: [i64; STATE_SIZE] = [0; STATE_SIZE];
        for (i, b) in state.iter().enumerate() {
            hi[i] = (b.raw_u64() >> 32) as i64;
            lo[i] = (b.raw_u64() as u32) as i64;
        }

        // hardcoded right-hand side (2x):
        // [
        //     61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901,
        //     12021, 59689, 26798, 17845,
        // ]
        lo = Self::fast_cyclomul16(lo, CONSTANTS);
        hi = Self::fast_cyclomul16(hi, CONSTANTS);

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

    #[inline]
    fn sbox_layer(state: &mut [BFieldElement; STATE_SIZE]) {
        // lookup
        state.iter_mut().take(NUM_SPLIT_AND_LOOKUP).for_each(|s| {
            Self::split_and_lookup(s);
        });

        // power
        for st in state.iter_mut().skip(NUM_SPLIT_AND_LOOKUP) {
            let sq = *st * *st;
            let qu = sq * sq;
            *st *= sq * qu;
        }
    }

    #[inline]
    fn round(sponge: &mut Tip5State, round_index: usize) {
        Self::sbox_layer(&mut sponge.state);

        // Self::mds_noswap(&mut sponge.state);
        // Self::mds_split(&mut sponge.state);
        Self::mds_cyclomul(&mut sponge.state);

        for i in 0..STATE_SIZE {
            sponge.state[i] += ROUND_CONSTANTS[round_index * STATE_SIZE + i];
        }
    }

    // permutation
    #[inline]
    fn permutation(sponge: &mut Tip5State) {
        for i in 0..NUM_ROUNDS {
            Self::round(sponge, i);
        }
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
}

impl SpongeHasher for Tip5 {
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
    use crate::shared_math::ntt;
    use crate::shared_math::ntt::ntt;
    use crate::shared_math::other::random_elements;
    use crate::shared_math::rescue_prime_digest::DIGEST_LENGTH;
    use crate::shared_math::tip5::Tip5;
    use crate::shared_math::tip5::LOOKUP_TABLE;
    use crate::shared_math::tip5::NUM_ROUNDS;
    use crate::shared_math::tip5::ROUND_CONSTANTS;
    use crate::shared_math::tip5::STATE_SIZE;
    use crate::util_types::algebraic_hasher::AlgebraicHasher;

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
                    .into_iter()
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
                    .into_iter()
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
    fn test_mds_test_vector() {
        let mut x = [
            5735458159267578080,
            11079291868388879320,
            7126936809174926852,
            13782161578414002790,
            164785954911215634,
            3118898034727063217,
            6737535956326810438,
            5144821635942763745,
            16200832071427728225,
            8640629006986782903,
            11570592580608458034,
            2895124598773988749,
            3420957867360511946,
            5796711531533733319,
            5282341612640982074,
            7026199320889950703,
        ]
        .map(BFieldElement::new);
        let y = [
            4104170903924047333,
            6387491404022818542,
            14981184993811752484,
            16496996924371698202,
            5837420782411553495,
            4264374326976985633,
            5211883823040202320,
            11836807491772316903,
            8162670480249154941,
            5581482934627657894,
            9403344895570333937,
            8567874241156119862,
            15302967789437559413,
            13072768661755417248,
            18135835343258257325,
            9011523754984921044,
        ]
        .map(BFieldElement::new);
        Tip5::mds_noswap(&mut x);
        assert_eq!(x, y);
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
            13893101008570085259,
            12868275854371526426,
            14048396406819570654,
            5452119458724574669,
            6240471940533869882,
        ]
        .map(BFieldElement::new);
        assert_eq!(digest, final_digest);
    }

    #[test]
    fn hash_varlen_test_vectors() {
        let mut digest_sum = [BFieldElement::zero(); DIGEST_LENGTH];
        for i in 0..20 {
            let preimage = (0..i).into_iter().map(BFieldElement::new).collect_vec();
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
            2239969078742881976,
            15666586250577462559,
            9635526344449925727,
            16655273469642706699,
            5222591716361630589,
        ]
        .map(BFieldElement::new);
        assert_eq!(expected_sum, digest_sum);
    }

    #[test]
    fn test_linearity_of_mds() {
        let mds_procedure = Tip5::mds_split;
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

        // let mds_procedure = Tip5::mds_split;
        // let mds_procedure = Tip5::mds_noswap;
        let mds_procedure = Tip5::mds_cyclomul;

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
    fn test_ntt_noswap() {
        let a: Vec<BFieldElement> = random_elements(16);
        let mut b = a.clone();
        Tip5::ntt_noswap(&mut b);
        let mut c = a.clone();
        ntt::<BFieldElement>(&mut c, BFieldElement::new(1 << 12), 4);
        for i in 0..16 {
            if i < ntt::bitreverse(i, 4) {
                c.swap(i as usize, ntt::bitreverse(i, 4) as usize);
            }
        }

        assert_eq!(c, b);

        Tip5::intt_noswap(&mut b);
        b.iter_mut()
            .for_each(|bb| *bb = *bb / BFieldElement::new(16));
        assert_eq!(a, b);
    }

    #[test]
    fn test_negacyclomul() {
        let mut rng = thread_rng();
        const N: usize = 8;
        let mut f = [0i64; N];
        let mut g = [0i64; N];
        for i in 0..N {
            f[i] = (rng.next_u32() % 65536) as i64;
            g[i] = (rng.next_u32() % 65536) as i64;
        }

        let h0 = Tip5::fast_negacyclomul8(f, g);
        let h1 = Tip5::complex_negacyclomul8(f, g);

        assert_eq!(h0, h1);
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
}
