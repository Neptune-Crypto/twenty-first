use crate::shared_math::b_field_element::BFieldElement;

// TODO: Share symbols with reference
pub struct EvaluationArgument {
    pub challenge_index: usize,
    pub terminal_index: usize,
    pub symbols: Vec<BFieldElement>,
}

impl EvaluationArgument {
    pub fn new(challenge_index: usize, terminal_index: usize, symbols: Vec<BFieldElement>) -> Self {
        Self {
            challenge_index,
            terminal_index,
            symbols,
        }
    }
}

pub struct ProgramEvaluationArgument {
    pub challenge_indices: Vec<usize>,
    pub terminal_index: usize,
    pub program: Vec<BFieldElement>,
}

impl ProgramEvaluationArgument {
    pub fn new(
        challenge_indices: Vec<usize>,
        terminal_index: usize,
        program: Vec<BFieldElement>,
    ) -> Self {
        Self {
            challenge_indices,
            terminal_index,
            program,
        }
    }
}
