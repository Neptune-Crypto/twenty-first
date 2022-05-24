use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::traits::GetPrimitiveRootOfUnity;

use super::table::base_matrix::BaseMatrices;
use super::table::base_table::{derive_omicron, BaseTable, HasBaseTable, Table};
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

        // From brainfuck
        let num_randomizers = 1;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();
        let unpadded_height = base_matrices.processor_matrix.len();
        // this is probably wrong
        let omicron = derive_omicron(unpadded_height);

        let processor_base_table = BaseTable::<BWord, { processor_table::BASE_WIDTH }>::new(
            "processor_table",
            unpadded_height,
            num_randomizers,
            omicron,
            smooth_generator,
            order,
            base_matrices.processor_matrix,
        );
        let _processor_table = ProcessorTable::new(processor_base_table);

        // 2. Create base codeword tables based on those
    }
}
