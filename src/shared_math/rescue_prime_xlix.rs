use super::b_field_element::BFieldElement;
use super::rescue_prime_params;
use crate::shared_math::traits::ModPowU64;

type Word = BFieldElement;

pub const RP_DEFAULT_WIDTH: usize = 16;

#[derive(Debug, Clone)]
pub struct RescuePrimeXlix<const M: usize> {
    pub capacity: usize,
    pub n: usize,
    pub alpha: u64,
    pub alpha_inv: u64,
    pub mds: Vec<Vec<Word>>,
    pub mds_inv: Vec<Vec<Word>>,
    pub round_constants: Vec<Word>,
}

#[allow(clippy::needless_range_loop)]
impl<const M: usize> RescuePrimeXlix<M> {
    /// Hash an arbitrary-length slice of words into a vector of at most `M`
    /// words. Do this by feeding `rate` amount of words at a time to the
    /// `rescue_xlix_permutation`, leaving `capacity` number of words between
    /// iterations.
    pub fn hash(&self, input: &[Word], output_len: usize) -> Vec<Word> {
        assert!(output_len <= M, "Output at most {} words.", M);

        // Calculate width of state for input chunks
        let rate = M - self.capacity;

        // Calculate number of rounds that don't require padding
        let full_iterations = input.len() / rate;

        // Calculate number of elements that require padding
        let last_iteration_elements = input.len() % rate;

        // Initialize state to all zeros
        let mut state: [Word; M] = [0.into(); M];

        // Absorbing (full rounds)
        for iteration in 0..full_iterations {
            let start = iteration * rate;
            let end = start + rate;
            let input_slice = &input[start..end];
            state[0..rate].copy_from_slice(input_slice);

            self.rescue_xlix_permutation(&mut state);
        }

        // Absorbing (last round with padding)
        if last_iteration_elements > 0 {
            let start = input.len() - last_iteration_elements;
            let input_slice = &input[start..];
            state[0..last_iteration_elements].copy_from_slice(input_slice);

            // Padding
            state[last_iteration_elements] = 1.into();
            state[last_iteration_elements + 1..rate].fill(0.into());

            self.rescue_xlix_permutation(&mut state);
        }

        // Squeezing
        (&state[0..output_len]).to_vec()
    }

    /// The Rescue-XLIX permutation
    pub fn rescue_xlix_permutation(&self, state: &mut [Word]) {
        debug_assert_eq!(M, state.len());

        for round in 0..self.n {
            self.rescue_xlix_round(round, state);
        }
    }

    #[inline]
    fn rescue_xlix_round(&self, i: usize, state: &mut [Word]) {
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

pub fn neptune_params() -> RescuePrimeXlix<RP_DEFAULT_WIDTH> {
    let params = rescue_prime_params::rescue_prime_params_bfield_0();

    let capacity = 4;
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

#[cfg(test)]
mod rescue_prime_xlix_tests {
    // use super::*;
    // use crate::shared_math::traits::GetRandomElements;
}
