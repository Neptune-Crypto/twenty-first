use crate::shared_math::b_field_element::BFieldElement;

use super::table::base_matrix::BaseMatrices;
use super::table::base_table::{BaseTable, HasBaseTable};
use super::table::processor_table::{self, ProcessorTable};
use super::vm::Program;

pub const EXTENSION_CHALLENGE_COUNT: usize = 0;
pub const PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const TERMINAL_COUNT: usize = 0;

type BWord = BFieldElement;

pub struct Stark {
    program: Program,
}

impl Stark {
    pub fn prove(&self, base_matrices: BaseMatrices) {
        // 1. Create base tables based on base matrices
        let processor_base_table = BaseTable::<BWord, { processor_table::BASE_WIDTH }>::new("processor_table", ...);
        let processor_table = ProcessorTable::new(processor_base_table);

        // 2. Create base codeword tables based on those
    }
}
