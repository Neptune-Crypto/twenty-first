use twenty_first::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
use twenty_first::shared_math::rescue_prime_regular::{NUM_ROUNDS, RATE, STATE_SIZE};

pub mod stark_constraints;
pub mod stark_rp;

/// trace
/// Produces the execution trace for one invocation of XLIX
///
/// `RescuePrimeRegular::trace()` function was extended to include the full state
/// width for purposes that are unrelated to tracing the tutorial Rescue-Prime function.
/// This function serves as a legacy wrapper
pub fn rescue_prime_trace(
    input: &[BFieldElement; 10],
) -> [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS] {
    let mut state: [BFieldElement; STATE_SIZE] = [BFIELD_ZERO; STATE_SIZE];

    // absorb
    state[..10].copy_from_slice(input);

    // domain separation
    state[RATE] = BFIELD_ONE;

    RescuePrimeRegular::trace(state)
}
