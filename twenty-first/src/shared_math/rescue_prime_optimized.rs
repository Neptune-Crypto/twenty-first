use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::traits::FiniteField;
use crate::util_types::algebraic_hasher::{AlgebraicHasher, INPUT_LENGTH, OUTPUT_LENGTH};

use super::b_field_element::{BFIELD_ONE, BFIELD_ZERO};
use super::rescue_prime_digest::{Digest, DIGEST_LENGTH};

pub const STATE_SIZE: usize = 16;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;
pub const NUM_ROUNDS: usize = 8;

pub const ALPHA: u64 = 7;
pub const ALPHA_INV: u64 = 10540996611094048183;

/// These constants are still the same as in Rescue-Prime.
/// TODO: change to Rescue-Prime Optimized constants.
pub const ROUND_CONSTANTS: [u64; NUM_ROUNDS * STATE_SIZE * 2] = [
    3006656781416918236,
    4369161505641058227,
    6684374425476535479,
    15779820574306927140,
    9604497860052635077,
    6451419160553310210,
    16926195364602274076,
    6738541355147603274,
    13653823767463659393,
    16331310420018519380,
    10921208506902903237,
    5856388654420905056,
    180518533287168595,
    6394055120127805757,
    4624620449883041133,
    4245779370310492662,
    11436753067664141475,
    9565904130524743243,
    1795462928700216574,
    6069083569854718822,
    16847768509740167846,
    4958030292488314453,
    6638656158077421079,
    7387994719600814898,
    1380138540257684527,
    2756275326704598308,
    6162254851582803897,
    4357202747710082448,
    12150731779910470904,
    3121517886069239079,
    14951334357190345445,
    11174705360936334066,
    17619090104023680035,
    9879300494565649603,
    6833140673689496042,
    8026685634318089317,
    6481786893261067369,
    15148392398843394510,
    11231860157121869734,
    2645253741394956018,
    15345701758979398253,
    1715545688795694261,
    3419893440622363282,
    12314745080283886274,
    16173382637268011204,
    2012426895438224656,
    6886681868854518019,
    9323151312904004776,
    14061124303940833928,
    14720644192628944300,
    3643016909963520634,
    15164487940674916922,
    18095609311840631082,
    17450128049477479068,
    13770238146408051799,
    959547712344137104,
    12896174981045071755,
    15673600445734665670,
    5421724936277706559,
    15147580014608980436,
    10475549030802107253,
    9781768648599053415,
    12208559126136453589,
    14883846462224929329,
    4104889747365723917,
    748723978556009523,
    1227256388689532469,
    5479813539795083611,
    8771502115864637772,
    16732275956403307541,
    4416407293527364014,
    828170020209737786,
    12657110237330569793,
    6054985640939410036,
    4339925773473390539,
    12523290846763939879,
    6515670251745069817,
    3304839395869669984,
    13139364704983394567,
    7310284340158351735,
    10864373318031796808,
    17752126773383161797,
    1934077736434853411,
    12181011551355087129,
    16512655861290250275,
    17788869165454339633,
    12226346139665475316,
    521307319751404755,
    18194723210928015140,
    11017703779172233841,
    15109417014344088693,
    16118100307150379696,
    16104548432406078622,
    10637262801060241057,
    10146828954247700859,
    14927431817078997000,
    8849391379213793752,
    14873391436448856814,
    15301636286727658488,
    14600930856978269524,
    14900320206081752612,
    9439125422122803926,
    17731778886181971775,
    11364016993846997841,
    11610707911054206249,
    16438527050768899002,
    1230592087960588528,
    11390503834342845303,
    10608561066917009324,
    5454068995870010477,
    13783920070953012756,
    10807833173700567220,
    8597517374132535250,
    17631206339728520236,
    8083932512125088346,
    10460229397140806011,
    16904442127403184100,
    15806582425540851960,
    8002674967888750145,
    7088508235236416142,
    2774873684607752403,
    11519427263507311324,
    14949623981479468161,
    18169367272402768616,
    13279771425489376175,
    3437101568566296039,
    11820510872362664493,
    13649520728248893918,
    13432595021904865723,
    12153175375751103391,
    16459175915481931891,
    14698099486055505377,
    14962427686967561007,
    10825731681832829214,
    12562849212348892143,
    18054851842681741827,
    16866664833727482321,
    10485994783891875256,
    8074668712578030015,
    7502837771635714611,
    8326381174040960025,
    1299216707593490898,
    12092900834113479279,
    10147133736028577997,
    12103660182675227350,
    16088613802080804964,
    10323305955081440356,
    12814564542614394316,
    9653856919559060601,
    10390420172371317530,
    7831993942325060892,
    9568326819852151217,
    6299791178740935792,
    12692828392357621723,
    10331476541693143830,
    3115340436782501075,
    17456578083689713056,
    12924575652913558388,
    14365487216177868031,
    7211834371191912632,
    17610068359394967554,
    646302646073569086,
    12437378932700222679,
    2758591586601041336,
    10952396165876183059,
    8827205511644136726,
    17572216767879446421,
    12516044823385174395,
    6380048472179557105,
    1959389938825200414,
    257915527015303758,
    4942451629986849727,
    1698530521870297461,
    1802136667015215029,
    6353258543636931941,
    13791525219506237119,
    7093082295632492630,
    15409842367405634814,
    2090232819855225051,
    13926160661036606054,
    389467431021126699,
    4736917413147385608,
    6217341363393311211,
    4366302820407593918,
    12748238635329332117,
    7671680179984682360,
    17998193362025085453,
    432899318054332645,
    1973816396170253277,
    607886411884636526,
    15080416519109365682,
    13607062276466651973,
    2458254972975404730,
    15323169029557757131,
    10953434699543086460,
    13995946730291266219,
    12803971247555868632,
    3974568790603251423,
    10629169239281589943,
    2058261494620094806,
    15905212873859894286,
    11221574225004694137,
    15430295276730781380,
    10448646831319611878,
    7559293484620816204,
    15679753002507105741,
    6043747003590355195,
    3404573815097301491,
    13392826344874185313,
    6464466389567159772,
    8932733991045074013,
    6565970376680631168,
    7050411859293315754,
    9763347751680159247,
    3140014248604700259,
    5621238883761074228,
    12664766603293629079,
    6533276137502482405,
    914829860407409680,
    14599697497440353734,
    16400390478099648992,
    1619185634767959932,
    16420198681440130663,
    1331388886719756999,
    1430143015191336857,
    14618841684410509097,
    1870494251298489312,
    3783117677312763499,
    16164771504475705474,
    6996935044500625689,
    4356994160244918010,
    13579982029281680908,
    8835524728424198741,
    13281017722683773148,
    2669924686363521592,
    15020410046647566094,
    9534143832529454683,
    156263138519279564,
    17421879327900831752,
    9524879102847422379,
    5120021146470638642,
    9588770058331935449,
    1501841070476096181,
    5687728871183511192,
    16091855309800405887,
    17307425956518746505,
    1162636238106302518,
    8756478993690213481,
    6898084027896327288,
    8485261637658061794,
    4169208979833913382,
    7776158701576840241,
    13861841831073878156,
    4896983281306117497,
    6056805506026814259,
    15706891000994288769,
];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RescuePrimeOptimizedState {
    pub state: [BFieldElement; STATE_SIZE],
}

impl RescuePrimeOptimizedState {
    fn new() -> RescuePrimeOptimizedState {
        RescuePrimeOptimizedState {
            state: [BFIELD_ZERO; STATE_SIZE],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct RescuePrimeOptimized {}

impl RescuePrimeOptimized {
    #[inline]
    fn batch_square(array: &mut [BFieldElement; STATE_SIZE]) {
        for a in array.iter_mut() {
            *a = a.square();
        }
    }

    #[inline]
    fn batch_square_n<const N: usize>(array: &mut [BFieldElement; STATE_SIZE]) {
        for _ in 0..N {
            Self::batch_square(array);
        }
    }

    #[inline]
    fn batch_mul_into(
        array: &mut [BFieldElement; STATE_SIZE],
        operand: [BFieldElement; STATE_SIZE],
    ) {
        for (a, b) in array.iter_mut().zip_eq(operand.iter()) {
            *a *= *b;
        }
    }

    #[inline]
    fn batch_mod_pow_alpha_inv(array: [BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        // alpha^-1 = 0b1001001001001001001001001001000110110110110110110110110110110111

        // credit to Winterfell for this decomposition into 72 multiplications
        // (1) base^10
        // (2) base^100 = (1) << 1
        // (3) base^100100 = (2) << 3 + (2)
        // (4) base^100100100100 = (3) << 6 + (3)
        // (5) base^100100100100100100100100 = (4) << 12 + (4)
        // (6) base^100100100100100100100100100100 = (5) << 6 + (3)
        // (7) base^1001001001001001001001001001000100100100100100100100100100100
        //     = (6) << 31 + (6)
        // (r) base^1001001001001001001001001001000110110110110110110110110110110111
        //     = ((7) << 1 + (6)) << 2 + (2)  +  (1)  +  1

        let mut p1 = array;
        Self::batch_square(&mut p1);

        let mut p2 = p1;
        Self::batch_square(&mut p2);

        let mut p3 = p2;
        Self::batch_square_n::<3>(&mut p3);
        Self::batch_mul_into(&mut p3, p2);

        let mut p4 = p3;
        Self::batch_square_n::<6>(&mut p4);
        Self::batch_mul_into(&mut p4, p3);

        let mut p5 = p4;
        Self::batch_square_n::<12>(&mut p5);
        Self::batch_mul_into(&mut p5, p4);

        let mut p6 = p5;
        Self::batch_square_n::<6>(&mut p6);
        Self::batch_mul_into(&mut p6, p3);

        let mut p7 = p6;
        Self::batch_square_n::<31>(&mut p7);
        Self::batch_mul_into(&mut p7, p6);

        let mut result = p7;
        Self::batch_square(&mut result);
        Self::batch_mul_into(&mut result, p6);
        Self::batch_square_n::<2>(&mut result);
        Self::batch_mul_into(&mut result, p2);
        Self::batch_mul_into(&mut result, p1);
        Self::batch_mul_into(&mut result, array);
        result
    }

    #[inline]
    fn batch_mod_pow_alpha(array: [BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        let mut result = array;
        Self::batch_square(&mut result);
        Self::batch_mul_into(&mut result, array);
        Self::batch_square(&mut result);
        Self::batch_mul_into(&mut result, array);
        result
    }

    #[allow(dead_code)]
    fn batch_mod_pow(
        array: [BFieldElement; STATE_SIZE],
        power: u64,
    ) -> [BFieldElement; STATE_SIZE] {
        let mut acc = [BFIELD_ONE; STATE_SIZE];
        for i in (0..64).rev() {
            if i != 63 {
                Self::batch_square(&mut acc);
            }
            if power & (1 << i) != 0 {
                Self::batch_mul_into(&mut acc, array);
            }
        }

        acc
    }

    #[allow(clippy::many_single_char_names)]
    fn ntt_noswap(x: &mut [BFieldElement]) {
        const POWERS_OF_OMEGA_BITREVERSED: [BFieldElement; 8] = [
            BFieldElement::new(1),
            BFieldElement::new(281474976710656),
            BFieldElement::new(18446744069397807105),
            BFieldElement::new(18446742969902956801),
            BFieldElement::new(17293822564807737345),
            BFieldElement::new(4096),
            BFieldElement::new(4503599626321920),
            BFieldElement::new(18446744000695107585),
        ];

        // outer loop iteration 1
        for j in 0..8 {
            let u = x[j];
            let v = x[j + 8] * BFIELD_ONE;
            x[j] = u + v;
            x[j + 8] = u - v;
        }

        // outer loop iteration 2
        for (i, zeta) in POWERS_OF_OMEGA_BITREVERSED.iter().enumerate().take(2) {
            let s = i * 8;
            for j in s..(s + 4) {
                let u = x[j];
                let v = x[j + 4] * *zeta;
                x[j] = u + v;
                x[j + 4] = u - v;
            }
        }

        // outer loop iteration 3
        for (i, zeta) in POWERS_OF_OMEGA_BITREVERSED.iter().enumerate().take(4) {
            let s = i * 4;
            for j in s..(s + 2) {
                let u = x[j];
                let v = x[j + 2] * *zeta;
                x[j] = u + v;
                x[j + 2] = u - v;
            }
        }

        // outer loop iteration 4
        for (i, zeta) in POWERS_OF_OMEGA_BITREVERSED.iter().enumerate().take(8) {
            let s = i * 2;
            let u = x[s];
            let v = x[s + 1] * *zeta;
            x[s] = u + v;
            x[s + 1] = u - v;
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn intt_noswap(x: &mut [BFieldElement]) {
        const POWERS_OF_OMEGA_INVERSE: [BFieldElement; 8] = [
            BFieldElement::new(1),
            BFieldElement::new(68719476736),
            BFieldElement::new(1099511627520),
            BFieldElement::new(18446744069414580225),
            BFieldElement::new(18446462594437873665),
            BFieldElement::new(18442240469788262401),
            BFieldElement::new(16777216),
            BFieldElement::new(1152921504606846976),
        ];

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
                    let zeta = POWERS_OF_OMEGA_INVERSE[4 * j];
                    {
                        let u = x[j + 2] * zeta;
                        let v = x[j];
                        x[j + 2] = v - u;
                        x[j] = v + u;
                    }
                    // inner loop iteration 2
                    {
                        let u = x[4 + j + 2] * zeta;
                        let v = x[4 + j];
                        x[4 + j + 2] = v - u;
                        x[4 + j] = v + u;
                    }
                    // inner loop iteration 3
                    {
                        let u = x[8 + j + 2] * zeta;
                        let v = x[8 + j];
                        x[8 + j + 2] = v - u;
                        x[8 + j] = v + u;
                    }
                    // inner loop iteration 4
                    {
                        let u = x[12 + j + 2] * zeta;
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
                    let zeta = POWERS_OF_OMEGA_INVERSE[2 * j];
                    // inner loop iteration 1
                    {
                        let u = x[j + 4] * zeta;
                        let v = x[j];
                        x[j + 4] = v - u;
                        x[j] = v + u;
                    }
                    // inner loop iteration 2
                    {
                        let u = x[8 + j + 4] * zeta;
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
                let zeta = POWERS_OF_OMEGA_INVERSE[j];
                let u = x[j + 8] * zeta;
                let v = x[j];
                x[j + 8] = v - u;
                x[j] = v + u;
            }
        }
    }

    #[inline]
    pub fn mds_noswap(state: &mut [BFieldElement; STATE_SIZE]) {
        let mds: [BFieldElement; STATE_SIZE] = [
            BFieldElement::new(1363685766),
            BFieldElement::new(818401426),
            BFieldElement::new(2843477530982740278),
            BFieldElement::new(15603266536318963895),
            BFieldElement::new(4617387998068915967),
            BFieldElement::new(13834281883405632256),
            BFieldElement::new(18438678032804473072),
            BFieldElement::new(3140224485136655),
            BFieldElement::new(3747273207304324287),
            BFieldElement::new(14700029414217449666),
            BFieldElement::new(9286765195715607938),
            BFieldElement::new(9160541823450023167),
            BFieldElement::new(18392355339471673798),
            BFieldElement::new(89869970136635963),
            BFieldElement::new(16012825548870059521),
            BFieldElement::new(2397315778488370688),
        ];
        Self::ntt_noswap(state);

        for (i, m) in mds.iter().enumerate() {
            state[i] *= *m;
        }

        Self::intt_noswap(state);

        // STATE_SIZE^{-1} mod p
        const NINV: BFieldElement = BFieldElement::new(17293822565076172801);
        state.iter_mut().for_each(|s| *s *= NINV);
    }

    /// xlix_round
    /// Apply one round of the XLIX permutation.
    fn xlix_round(sponge: &mut RescuePrimeOptimizedState, round_index: usize) {
        debug_assert!(
            round_index < NUM_ROUNDS,
            "Cannot apply {}th round; only have {} in total.",
            round_index,
            NUM_ROUNDS
        );

        // MDS matrix
        // let mut v: [BFieldElement; STATE_SIZE] = [BFieldElement::from(0u64); STATE_SIZE];
        // for i in 0..STATE_SIZE {
        //     for j in 0..STATE_SIZE {
        //         v[i] += BFieldElement::from(MDS[i * STATE_SIZE + j]) * sponge.state[j];
        //     }
        // }
        // sponge.state = v;
        Self::mds_noswap(&mut sponge.state);

        // round constants A
        for i in 0..STATE_SIZE {
            sponge.state[i] +=
                BFieldElement::from(ROUND_CONSTANTS[round_index * STATE_SIZE * 2 + i]);
        }

        // S-box
        // for i in 0..STATE_SIZE {
        //     self.state[i] = self.state[i].mod_pow_u64(ALPHA);
        // }
        //
        sponge.state = Self::batch_mod_pow_alpha(sponge.state);

        // MDS matrix
        // for i in 0..STATE_SIZE {
        //     v[i] = BFIELD_ZERO;
        //     for j in 0..STATE_SIZE {
        //         v[i] += BFieldElement::from(MDS[i * STATE_SIZE + j]) * sponge.state[j];
        //     }
        // }
        // sponge.state = v;
        Self::mds_noswap(&mut sponge.state);

        // round constants B
        for i in 0..STATE_SIZE {
            sponge.state[i] +=
                BFieldElement::from(ROUND_CONSTANTS[round_index * STATE_SIZE * 2 + STATE_SIZE + i]);
        }

        // Inverse S-box
        // for i in 0..STATE_SIZE {
        //     self.state[i] = self.state[i].mod_pow_u64(ALPHA_INV);
        // }
        //
        // self.state = Self::batch_mod_pow(self.state, ALPHA_INV);
        sponge.state = Self::batch_mod_pow_alpha_inv(sponge.state);
    }

    /// xlix
    /// XLIX is the permutation defined by Rescue-Prime Optimized. This
    /// function applies XLIX to the state of a sponge.
    fn xlix(sponge: &mut RescuePrimeOptimizedState) {
        for i in 0..NUM_ROUNDS {
            Self::xlix_round(sponge, i);
        }
    }

    /// hash_10
    /// Hash 10 elements, or two digests. There is no padding because
    /// the input length is equal to the rate.
    pub fn hash_10(input: &[BFieldElement; INPUT_LENGTH]) -> [BFieldElement; OUTPUT_LENGTH] {
        let mut sponge = RescuePrimeOptimizedState::new();

        // absorb once
        sponge.state[..INPUT_LENGTH].copy_from_slice(input);

        // apply domain separation for fixed-length input
        sponge.state[INPUT_LENGTH] = BFIELD_ONE;

        // apply xlix
        Self::xlix(&mut sponge);

        // squeeze once
        sponge.state[..OUTPUT_LENGTH].try_into().unwrap()
    }

    /// hash_varlen hashes an arbitrary number of field elements.
    ///
    /// Takes care of padding by applying the Rescue Prime Optimized
    /// padding rule: if the input length is not a multiple of the
    /// rate, then:
    ///  - append a single 1 ∈ Fp and as many 0 ∈ Fp elements as
    ///    required to make the number of input elements a multiple
    ///    of the rate
    ///  - set the first capacity element (now indexed with 0) to 1
    pub fn hash_varlen(input: &[BFieldElement]) -> [BFieldElement; 5] {
        let mut sponge = RescuePrimeOptimizedState::new();

        // need padding?
        let mut padded_input = if input.len() % RATE != 0 {
            let mut padded_input = input.to_vec();
            padded_input.push(BFIELD_ONE);
            while padded_input.len() % RATE != 0 {
                padded_input.push(BFIELD_ZERO);
            }
            sponge.state[0] = BFIELD_ONE;
            padded_input
        } else {
            input.to_vec()
        };

        // absorb
        while !padded_input.is_empty() {
            for (sponge_state_element, input_element) in sponge.state[CAPACITY..]
                .iter_mut()
                .take(RATE)
                .zip_eq(padded_input.iter().take(RATE))
            {
                // absorb by overwriting as per Rescue-Prime Optimized
                *sponge_state_element = input_element.to_owned();
            }
            padded_input.drain(..RATE);
            Self::xlix(&mut sponge);
        }

        // squeeze once
        sponge.state[CAPACITY..(CAPACITY + DIGEST_LENGTH)]
            .try_into()
            .unwrap()
    }

    /// trace
    /// Produces the execution trace for one invocation of XLIX
    pub fn trace(
        input: &[BFieldElement; INPUT_LENGTH],
    ) -> [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS] {
        let mut trace = [[BFIELD_ZERO; STATE_SIZE]; 1 + NUM_ROUNDS];
        let mut sponge = RescuePrimeOptimizedState::new();

        // absorb
        sponge.state[0..RATE].copy_from_slice(input);

        // domain separation
        sponge.state[RATE] = BFieldElement::new(1);

        // record trace
        trace[0] = sponge.state;

        // apply N rounds
        for round_index in 0..NUM_ROUNDS {
            // apply round function to state
            Self::xlix_round(&mut sponge, round_index);

            // record trace
            trace[1 + round_index] = sponge.state;
        }

        trace
    }

    /// hash_10_with_trace
    /// Computes the fixed-length hash digest and returns the trace
    /// along with it.
    pub fn hash_10_with_trace(
        input: &[BFieldElement; INPUT_LENGTH],
    ) -> (
        [BFieldElement; OUTPUT_LENGTH],
        [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS],
    ) {
        let trace: [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS] = Self::trace(input);
        let output: [BFieldElement; OUTPUT_LENGTH] =
            trace[NUM_ROUNDS][..OUTPUT_LENGTH].try_into().unwrap();

        (output, trace)
    }
}

impl AlgebraicHasher for RescuePrimeOptimized {
    fn hash_op(input: &[BFieldElement; INPUT_LENGTH]) -> [BFieldElement; OUTPUT_LENGTH] {
        RescuePrimeOptimized::hash_10(input)
    }

    fn hash_slice(elements: &[BFieldElement]) -> Digest {
        Digest::new(RescuePrimeOptimized::hash_varlen(elements))
    }
}

#[cfg(test)]
mod rescue_prime_optimized_tests {
    use crate::shared_math::other::random_elements_array;

    use super::*;

    #[test]
    fn trace_consistent_test() {
        for _ in 0..10 {
            let input: [BFieldElement; INPUT_LENGTH] = random_elements_array();
            let (output_a, _) = RescuePrimeOptimized::hash_10_with_trace(&input);
            let output_b = RescuePrimeOptimized::hash_10(&input);
            assert_eq!(output_a, output_b);
        }
    }
}
