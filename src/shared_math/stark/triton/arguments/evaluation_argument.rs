use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::triton::table::challenges_endpoints::{AllChallenges, AllEndpoints};
use crate::shared_math::x_field_element::XFieldElement;

pub const PROGRAM_EVALUATION_CHALLENGE_INDICES_COUNT: usize = 5;

pub fn all_evaluation_arguments(
    input_symbols: &[BFieldElement],
    output_symbols: &[BFieldElement],
    all_challenges: &AllChallenges,
    all_terminals: &AllEndpoints,
) -> bool {
    vec![
        processor_input_eval_arg(input_symbols, all_challenges, all_terminals),
        processor_output_eval_arg(output_symbols, all_challenges, all_terminals),
    ]
    .into_iter()
    .all(|is_valid| is_valid)
}

// 1. ProcessorTable -> InputTable
pub fn processor_input_eval_arg(
    symbols: &[BFieldElement],
    all_challenges: &AllChallenges,
    all_terminals: &AllEndpoints,
) -> bool {
    let challenge = all_challenges
        .processor_table_challenges
        .input_table_eval_row_weight;

    let terminal = all_terminals.processor_table_endpoints.input_table_eval_sum;

    verify_evaluation_argument(symbols, challenge, terminal)
}

// 2. ProcessorTable -> OutputTable
pub fn processor_output_eval_arg(
    symbols: &[BFieldElement],
    all_challenges: &AllChallenges,
    all_terminals: &AllEndpoints,
) -> bool {
    let challenge = all_challenges
        .processor_table_challenges
        .output_table_eval_row_weight;

    let terminal = all_terminals
        .processor_table_endpoints
        .output_table_eval_sum;

    verify_evaluation_argument(symbols, challenge, terminal)
}

// 3. ProcessorTable -> HashTable
// 4. ProcessorTable <- HashTable
// 5. InstructionTable -> ProgramTable

pub fn verify_evaluation_argument(
    symbols: &[BFieldElement],
    challenge: XFieldElement,
    expected_terminal: XFieldElement,
) -> bool {
    compute_terminal(symbols, challenge) == expected_terminal
}

fn compute_terminal(symbols: &[BFieldElement], challenge: XFieldElement) -> XFieldElement {
    let mut acc = XFieldElement::ring_zero();

    for s in symbols.iter() {
        acc = challenge * acc + s.lift();
    }

    acc
}
