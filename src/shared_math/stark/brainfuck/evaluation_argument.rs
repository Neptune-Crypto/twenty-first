use crate::shared_math::{b_field_element::BFieldElement, x_field_element::XFieldElement};

use super::stark::{EXTENSION_CHALLENGE_COUNT, TERMINAL_COUNT};

pub const PROGRAM_EVALUATION_CHALLENGE_INDICES_COUNT: usize = 4;

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

    pub fn select_terminal(&self, terminals: [XFieldElement; TERMINAL_COUNT]) -> XFieldElement {
        terminals[self.terminal_index]
    }

    pub fn compute_terminal(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
    ) -> XFieldElement {
        let iota = challenges[self.challenge_index];
        let mut acc = XFieldElement::ring_zero();
        for s in self.symbols.iter() {
            acc = iota * acc + s.lift();
        }

        acc
    }
}

pub struct ProgramEvaluationArgument {
    pub challenge_indices: [usize; PROGRAM_EVALUATION_CHALLENGE_INDICES_COUNT],
    pub terminal_index: usize,
    pub program: Vec<BFieldElement>,
}

impl ProgramEvaluationArgument {
    pub fn new(
        challenge_indices: [usize; PROGRAM_EVALUATION_CHALLENGE_INDICES_COUNT],
        terminal_index: usize,
        program: Vec<BFieldElement>,
    ) -> Self {
        Self {
            challenge_indices,
            terminal_index,
            program,
        }
    }

    pub fn select_terminal(&self, terminals: [XFieldElement; TERMINAL_COUNT]) -> XFieldElement {
        terminals[self.terminal_index]
    }

    pub fn compute_terminal(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
    ) -> XFieldElement {
        let [a, b, c, eta] = [
            challenges[self.challenge_indices[0]],
            challenges[self.challenge_indices[1]],
            challenges[self.challenge_indices[2]],
            challenges[self.challenge_indices[3]],
        ];

        let mut running_sum = XFieldElement::ring_zero();
        let mut previous_address = -XFieldElement::ring_one();
        let mut padded_program: Vec<XFieldElement> =
            self.program.iter().map(|p| p.lift()).collect();
        padded_program.push(XFieldElement::ring_zero());
        for i in 0..padded_program.len() - 1 {
            let address: XFieldElement = BFieldElement::new(i as u64).lift();
            let current_instruction = padded_program[i];
            let next_instruction = padded_program[i + 1];
            if previous_address != address {
                running_sum = running_sum * eta
                    + a * address
                    + b * current_instruction
                    + c * next_instruction;
                previous_address = address;
            }
        }

        let index = padded_program.len() - 1;
        let address: XFieldElement = BFieldElement::new(index as u64).lift();
        let current_instruction = padded_program[index];
        let next_instruction = XFieldElement::ring_zero();

        running_sum * eta + a * address + b * current_instruction + c * next_instruction
    }
}
