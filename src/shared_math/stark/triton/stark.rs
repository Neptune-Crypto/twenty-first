use super::table::base_matrix::BaseMatrices;
use super::table::base_table::{HasBaseTable, Table};
use super::table::processor_table::ProcessorTable;
use super::vm::Program;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::traits::GetPrimitiveRootOfUnity;

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

        // From brainfuck
        let num_randomizers = 1;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();
        let unpadded_height = base_matrices.processor_matrix.len();

        let mut processor_table = ProcessorTable::new(
            unpadded_height,
            num_randomizers,
            smooth_generator,
            order,
            base_matrices.processor_matrix,
        );

        // 2. Pad matrix
        processor_table.pad();

        // 3. Create base codeword tables based on those
        let coded_processor_table = processor_table.codewords();
    }
}
