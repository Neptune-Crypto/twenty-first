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
pub struct Tip5 {
    table: [u8; 256],
    round_constants: [BFieldElement; NUM_ROUNDS * STATE_SIZE],
}

impl Tip5 {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let table: [u8; 256] = (0..256)
            .into_iter()
            .map(|t| Self::offset_fermat_cube_map(t as u16) as u8)
            .collect_vec()
            .try_into()
            .unwrap();
        let to_int = |bytes: &[u8]| {
            bytes
                .iter()
                .take(16)
                .enumerate()
                .map(|(i, b)| (*b as u128) << (8 * i))
                .sum::<u128>()
        };
        let p = (1u128 << 64) - (1u128 << 32) + 1u128;
        let round_constants = (0..NUM_ROUNDS * STATE_SIZE)
            .map(|i| ["Tip5".to_string().as_bytes(), &[(i as u8)]].concat())
            .map(|bytes| blake3::hash(&bytes))
            .map(|hash| *hash.as_bytes())
            .map(|bytes| to_int(&bytes))
            .map(|i| (i % p) as u64)
            .map(BFieldElement::from_raw_u64)
            .collect_vec()
            .try_into()
            .unwrap();

        Self {
            table,
            round_constants,
        }
    }

    #[inline]
    fn offset_fermat_cube_map(x: u16) -> u16 {
        let xx: u64 = (x + 1).into();
        let xxx = xx * xx * xx;
        ((xxx + 256) % 257) as u16
    }

    #[inline]
    fn split_and_lookup(&self, element: &mut BFieldElement) -> BFieldElement {
        // let value = element.value();
        let mut bytes = element.raw_bytes();

        #[allow(clippy::needless_range_loop)] // faster like so
        for i in 0..8 {
            // bytes[i] = Self::offset_fermat_cube_map(bytes[i].into()) as u8;
            bytes[i] = self.table[bytes[i] as usize];
        }

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

    #[inline]
    pub fn mds_noswap(state: &mut [BFieldElement; STATE_SIZE]) {
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
            sponge.state[i] += self.round_constants[round_index * STATE_SIZE + i];
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

    /// Hash an arbitrary number of field elements.
    ///
    /// This function takes care of padding by applying the padding
    /// rule: append a single 1 ∈ Fp and as many 0 ∈ Fp elements as
    /// are required to make the number of input elements a multiple
    /// of `RATE`.
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
    use itertools::Itertools;
    use num_traits::Zero;
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    use crate::shared_math::{
        b_field_element::BFieldElement, rescue_prime_optimized::DIGEST_LENGTH, tip5::Tip5,
    };

    use super::RATE;

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
        println!(
            "additive differential uniformity for fermat cube map: {}",
            du_fermat
        );

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
                    println!("a: {}, b: {} -> 256 satisfiers", a, b);
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
        println!(
            "bitwise differential uniformity for fermat cube map: {}",
            du_fermat_bitwise
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
        let mut digest = [BFieldElement::zero(); DIGEST_LENGTH];
        let tip5 = Tip5::new();
        for i in 0..6 {
            digest = tip5.hash_10(&preimage);
            println!(
                "{:?} -> {:?}",
                preimage.iter().map(|b| b.value()).collect_vec(),
                digest.iter().map(|b| b.value()).collect_vec()
            );
            preimage[i..DIGEST_LENGTH + i].copy_from_slice(&digest);
        }
        digest = tip5.hash_10(&preimage);
        println!(
            "{:?} -> {:?}",
            preimage.iter().map(|b| b.value()).collect_vec(),
            digest.iter().map(|b| b.value()).collect_vec()
        );
        let final_digest = [
            14558289001666338382,
            8910286450360777215,
            8687235873380904976,
            9731988339297305717,
            14852227464718284881,
        ]
        .map(BFieldElement::new);
        assert_eq!(digest, final_digest);
    }

    #[test]
    fn hash_varlen_test_vectors() {
        let mut digest_sum = [BFieldElement::zero(); DIGEST_LENGTH];
        let tip5 = Tip5::new();
        for i in 0..20 {
            let preimage = (0..i).into_iter().map(BFieldElement::new).collect_vec();
            let digest = tip5.hash_varlen(&preimage);
            println!(
                "{:?} -> {:?}",
                preimage.iter().map(|b| b.value()).collect_vec(),
                digest.iter().map(|b| b.value()).collect_vec()
            );
            digest_sum
                .iter_mut()
                .zip(digest.iter())
                .for_each(|(s, d)| *s += *d);
        }
        println!(
            "sum of digests: {:?}",
            digest_sum.iter().map(|b| b.value()).collect_vec()
        );
        let expected_sum = [
            8476864380936389014,
            7923359605828412643,
            2436794214779586248,
            12117847227056347517,
            7400965751243819750,
        ]
        .map(BFieldElement::new);
        assert_eq!(expected_sum, digest_sum);
    }
}
