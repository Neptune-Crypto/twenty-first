use crate::shared_math::b_field_element::BFieldElement;

use super::{instruction_table, io_table, processor_table};

#[derive(Debug, Clone, Default)]
pub struct BaseMatrices {
    pub processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
    pub instruction_matrix: Vec<[BFieldElement; instruction_table::BASE_WIDTH]>,
    pub input_matrix: Vec<[BFieldElement; io_table::BASE_WIDTH]>,
    pub output_matrix: Vec<[BFieldElement; io_table::BASE_WIDTH]>,
}

impl BaseMatrices {
    pub fn sort_instruction_matrix(&mut self) {
        self.instruction_matrix.sort_by_key(|row| row[0].value());
    }
}
