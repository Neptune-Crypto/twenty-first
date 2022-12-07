use itertools::Itertools;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

use super::b_field_element::BFieldElement;

pub const DIGEST_LENGTH: usize = 5;
pub const STATE_SIZE: usize = 16;
pub const LOG2_STATE_SIZE: usize = 4;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;
pub const NUM_ROUNDS: usize = 7;

/// constants come from Rescue-Prime Regular
/// TODO: generate own constants
pub const ROUND_CONSTANTS: [u64; NUM_ROUNDS * STATE_SIZE] = [
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
];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Tip5State {
    pub state: [BFieldElement; STATE_SIZE],
}

impl Tip5State {
    fn new() -> Tip5State {
        Tip5State {
            state: [BFieldElement::zero(); STATE_SIZE],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tip5 {}

impl Tip5 {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {}
    }

    #[inline]
    fn fermat_cube_map(x: u32) -> u32 {
        let x2 = x * x;
        let x2hi = x2 >> 16;
        let x2lo = x2 & 0xffff;
        let x2p = x2lo + u32::from(x2lo < x2hi) * 65537 - x2hi;
        let x3 = x2p * x;
        let x3hi = x3 >> 16;
        let x3lo = x3 & 0xffff;
        x3lo + u32::from(x3lo < x3hi) * 65537 - x3hi
    }

    #[inline]
    fn inverted_fermat_cube_map(x: u32) -> u32 {
        65536 - Self::fermat_cube_map(65535 - x)
    }

    #[inline]
    fn split_and_lookup(&self, element: &mut BFieldElement) -> BFieldElement {
        let value = element.value();

        let a: u32 = (value >> 48).try_into().unwrap();
        let b: u32 = ((value >> 32) & 0xffff).try_into().unwrap();
        let c: u32 = ((value >> 16) & 0xffff).try_into().unwrap();
        let d: u32 = (value & 0xffff).try_into().unwrap();

        // let a_ = 65535 - self.lookup_table[(65535 - a) as usize];
        // let b_ = 65535 - self.lookup_table[(65535 - b) as usize];
        // let c_ = self.lookup_table[c as usize];
        // let d_ = self.lookup_table[d as usize];

        let a_ = Self::inverted_fermat_cube_map(a);
        let b_ = Self::inverted_fermat_cube_map(b);
        let c_ = Self::fermat_cube_map(c);
        let d_ = Self::fermat_cube_map(d);

        BFieldElement::new(
            ((a_ as u64) << 48) | ((b_ as u64) << 32) | ((c_ as u64) << 16) | (d_ as u64),
        )
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
            let v = x[j + 8] * BFieldElement::one();
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

    pub fn mul_state(state: &mut [BFieldElement; STATE_SIZE], arg: BFieldElement) {
        state.iter_mut().for_each(|s| *s *= arg);
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
    }

    #[inline]
    fn sbox_layer(&self, state: &mut [BFieldElement; STATE_SIZE]) {
        // lookup
        state.iter_mut().take(STATE_SIZE / 2).for_each(|s| {
            self.split_and_lookup(s);
        });

        // power
        for st in state.iter_mut().take(STATE_SIZE).skip(STATE_SIZE / 2) {
            let sq = *st * *st;
            let qu = sq * sq;
            *st *= sq * qu;
        }
    }

    #[inline]
    fn round(&self, sponge: &mut Tip5State, round_index: usize) {
        self.sbox_layer(&mut sponge.state);

        Self::mds_noswap(&mut sponge.state);

        for i in 0..STATE_SIZE {
            sponge.state[i] += BFieldElement::from(ROUND_CONSTANTS[round_index * STATE_SIZE + i]);
        }
    }

    // permutation
    fn permutation(&self, sponge: &mut Tip5State) {
        for i in 0..NUM_ROUNDS {
            self.round(sponge, i);
        }
    }

    /// hash_10
    /// Hash 10 elements, or two digests. There is no padding because
    /// the input length is fixed.
    pub fn hash_10(&self, input: &[BFieldElement; 10]) -> [BFieldElement; 5] {
        let mut sponge = Tip5State::new();

        // absorb once
        sponge.state[..10].copy_from_slice(input);

        // apply domain separation for fixed-length input
        sponge.state[10] = BFieldElement::one();

        // apply permutation
        self.permutation(&mut sponge);

        // squeeze once
        sponge.state[..5].try_into().unwrap()
    }

    /// hash_varlen hashes an arbitrary number of field elements.
    ///
    /// Takes care of padding by applying the padding rule: append a single 1 ∈ Fp
    /// and as many 0 ∈ Fp elements as required to make the number of input elements
    /// a multiple of `RATE`.
    pub fn hash_varlen(&self, input: &[BFieldElement]) -> [BFieldElement; 5] {
        let mut sponge = Tip5State::new();

        // pad input
        let mut padded_input = input.to_vec();
        padded_input.push(BFieldElement::one());
        while padded_input.len() % RATE != 0 {
            padded_input.push(BFieldElement::zero());
        }

        // absorb
        while !padded_input.is_empty() {
            for (sponge_state_element, input_element) in sponge
                .state
                .iter_mut()
                .take(RATE)
                .zip_eq(padded_input.iter().take(RATE))
            {
                *sponge_state_element += input_element.to_owned();
            }
            padded_input.drain(..RATE);
            self.permutation(&mut sponge);
        }

        // squeeze once
        sponge.state[..5].try_into().unwrap()
    }
}

#[cfg(test)]
mod tip5_tests {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    #[inline]
    fn fermat_cube_map(x: u32) -> u32 {
        let x2 = x * x;
        let x2hi = x2 >> 16;
        let x2lo = x2 & 0xffff;
        let x2p = x2lo + u32::from(x2lo < x2hi) * 65537 - x2hi;
        let x3 = x2p * x;
        let x3hi = x3 >> 16;
        let x3lo = x3 & 0xffff;
        x3lo + u32::from(x3lo < x3hi) * 65537 - x3hi
    }

    #[inline]
    fn inverted_fermat_cube_map(x: u32) -> u32 {
        65535 - fermat_cube_map(65535 - x)
    }

    #[test]
    #[ignore = "used for calculating parameters"]
    fn test_fermat_cube_map_is_permutation() {
        let mut touched = [false; 65536];
        for i in 0..65536 {
            touched[fermat_cube_map(i) as usize] = true;
        }
        assert!(touched.iter().all(|t| *t));
        assert_eq!(fermat_cube_map(0), 0);
    }

    #[test]
    #[ignore = "used for calculating parameters"]
    fn test_inverted_fermat_cube_map_is_permutation() {
        let mut touched = [false; 65536];
        for i in 0..65536 {
            touched[inverted_fermat_cube_map(i) as usize] = true;
        }
        assert!(touched.iter().all(|t| *t));
        assert_eq!(inverted_fermat_cube_map(65535), 65535);
    }

    #[test]
    #[ignore = "used for calculating parameters"]
    fn calculate_differential_uniformity() {
        // cargo test calculate_differential_uniformity -- --include-ignored --nocapture
        let count_satisfiers_fermat = |a, b| {
            (0..(1 << 16))
                .map(|x| {
                    u32::from(
                        (0xffff + fermat_cube_map((x + a) & 0xffff) - fermat_cube_map(x)) & 0xffff
                            == b,
                    )
                })
                .sum()
        };
        let du_fermat: u32 = (1..65536)
            .into_par_iter()
            .map(|a| {
                (1..65536)
                    .into_iter()
                    .map(|b| count_satisfiers_fermat(a, b))
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        println!("differential uniformity for fermat cube map: {}", du_fermat);

        let count_satisfiers_inverted = |a, b| {
            (0..(1 << 16))
                .map(|x| {
                    u32::from(
                        (0xffff + inverted_fermat_cube_map((x + a) & 0xffff)
                            - inverted_fermat_cube_map(x))
                            & 0xffff
                            == b,
                    )
                })
                .sum()
        };
        let du_inverted: u32 = (1..65536)
            .into_par_iter()
            .map(|a| {
                (1..65536)
                    .into_iter()
                    .map(|b| count_satisfiers_inverted(a, b))
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        println!(
            "differential uniformity for fermat cube map: {}",
            du_inverted
        );
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
        println!("agreement with low-degree function: {}", equal_count);
    }
}
