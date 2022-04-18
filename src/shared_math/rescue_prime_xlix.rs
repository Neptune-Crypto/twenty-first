use super::b_field_element::BFieldElement;
use super::rescue_prime_params;
use crate::shared_math::traits::ModPowU64;

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct RescuePrimeXlix<const M: usize> {
    pub capacity: usize,
    pub n: usize,
    pub alpha: u64,
    pub alpha_inv: u64,
    pub mds: Vec<Vec<BWord>>,
    pub mds_inv: Vec<Vec<BWord>>,
    pub round_constants: Vec<BWord>,
}

#[allow(clippy::needless_range_loop)]
impl<const M: usize> RescuePrimeXlix<M> {
    /// Hash an input of arbitrary length; when the input's length isn't
    /// a multiple of `rate`, apply padding. This makes the input divide
    /// among a whole number of iterations.
    pub fn hash_wrapper(&self, input: &[BWord], output_len: usize) -> Vec<BFieldElement> {
        let rate = self.rate();

        if input.len() % rate == 0 {
            self.hash_padded(input, output_len)
        } else {
            // FIXME: Avoid cloning by first looping `inputs`, then looping the padding.
            let mut padded_inputs: Vec<BWord> = input.to_vec();
            padded_inputs.push(1.into());

            while padded_inputs.len() % rate != 0 {
                padded_inputs.push(0.into());
            }

            self.hash_padded(&padded_inputs, output_len)
        }
    }

    pub fn hash_padded(&self, input: &[BWord], output_len: usize) -> Vec<BFieldElement> {
        assert_eq!(0, input.len() % self.rate());
        assert!(output_len < self.rate());

        // Initialize state to all zeros
        let mut state: [BWord; M] = [0.into(); M];

        // Absorbing
        let mut absorb_index: usize = 0;
        while absorb_index < input.len() {
            for i in 0..self.rate() {
                state[i] = input[absorb_index];
                absorb_index += 1;
            }
            self.rescue_xlix_permutation(&mut state);
        }

        // Squeezing
        (&state[0..output_len]).to_vec()
    }

    /// The `rate` of Rescue-Prime's arithmetic sponge is the number of field
    /// elements that are absorbed between invocations of the Rescue-XLIX
    /// permutation.
    ///
    /// The relationship between `rate` and `capacity` is:
    ///
    /// $$
    /// M = rate + capacity
    /// $$
    #[inline]
    pub fn rate(&self) -> usize {
        M - self.capacity
    }

    /// The Rescue-XLIX permutation
    pub fn rescue_xlix_permutation(&self, state: &mut [BWord]) {
        debug_assert_eq!(M, state.len());

        for round in 0..self.n {
            self.rescue_xlix_round(round, state);
        }
    }

    #[inline]
    fn rescue_xlix_round(&self, i: usize, state: &mut [BWord]) {
        // S-box
        for j in 0..M {
            state[j] = state[j].mod_pow_u64(self.alpha);
        }

        // MDS
        for k in 0..M {
            for j in 0..M {
                state[k] += self.mds[k][j] * state[j];
            }
        }

        // Round constants
        for j in 0..M {
            state[j] += self.round_constants[i * 2 * M + j];
        }

        // Inverse S-box
        for j in 0..M {
            state[j] = state[j].mod_pow_u64(self.alpha_inv);
        }

        // MDS
        for i in 0..M {
            for j in 0..M {
                state[i] += self.mds[i][j] * state[j];
            }
        }

        // Round constants
        for j in 0..M {
            state[j] += self.round_constants[i * 2 * M + M + j];
        }
    }
}

pub fn neptune_params() -> RescuePrimeXlix<16> {
    let params = rescue_prime_params::rescue_prime_params_bfield_0();

    let capacity = 12;
    let n = params.steps_count;
    let alpha = params.alpha;
    let alpha_inv = params.alpha_inv;
    let mds = params.mds;
    let mds_inv = params.mds_inv;
    let round_constants = params.round_constants;

    RescuePrimeXlix {
        capacity,
        n,
        alpha,
        alpha_inv,
        mds,
        mds_inv,
        round_constants,
    }
}
