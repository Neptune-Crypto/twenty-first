use itertools::Itertools;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

use super::b_field_element::BFieldElement;

pub const DIGEST_LENGTH: usize = 5;
pub const STATE_SIZE: usize = 16;
pub const NUM_SPLIT_AND_LOOKUP: usize = 4;
pub const LOG2_STATE_SIZE: usize = 4;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;
pub const NUM_ROUNDS: usize = 5;

/// constants come from Rescue-Prime Regular
/// TODO: generate own constants
/// NOTE: some constants in here are duplicated
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
    fn fermat_cube_map(x: u16) -> u16 {
        let xx: u64 = x.into();
        let xxx = xx * xx * xx;
        (xxx % 257) as u16
    }

    #[inline]
    fn inverted_fermat_cube_map(x: u16) -> u16 {
        257 - Self::fermat_cube_map(257 - x)
    }

    #[inline]
    fn split_and_lookup(&self, element: &mut BFieldElement) -> BFieldElement {
        // let value = element.value();
        let mut bytes = element.raw_bytes();

        #[allow(clippy::needless_range_loop)] // faster like so
        for i in 4..8 {
            bytes[i] = Self::inverted_fermat_cube_map(bytes[i].into()) as u8;
        }
        // bytes[7] = Self::inverted_fermat_cube_map(bytes[7].into()) as u8;
        // bytes[6] = Self::inverted_fermat_cube_map(bytes[6].into()) as u8;
        // bytes[5] = Self::inverted_fermat_cube_map(bytes[5].into()) as u8;
        // bytes[4] = Self::inverted_fermat_cube_map(bytes[4].into()) as u8;

        #[allow(clippy::needless_range_loop)] // faster like so
        for i in 0..4 {
            bytes[i] = Self::fermat_cube_map(bytes[i].into()) as u8;
        }
        // bytes[3] = Self::fermat_cube_map(bytes[3].into()) as u8;
        // bytes[2] = Self::fermat_cube_map(bytes[2].into()) as u8;
        // bytes[1] = Self::fermat_cube_map(bytes[1].into()) as u8;
        // bytes[0] = Self::fermat_cube_map(bytes[0].into()) as u8;

        BFieldElement::from_raw_bytes(&bytes)
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
        state.iter_mut().take(NUM_SPLIT_AND_LOOKUP).for_each(|s| {
            self.split_and_lookup(s);
        });

        // power
        for st in state.iter_mut().skip(NUM_SPLIT_AND_LOOKUP) {
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
