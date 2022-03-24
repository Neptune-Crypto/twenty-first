use super::instruction_table::InstructionTable;
use super::io_table::IOTable;
use super::memory_table::MemoryTable;
use super::processor_table::ProcessorTable;
use super::table::TableTrait;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::{Degree, MPolynomial};

pub struct TableCollection {
    pub processor_table: ProcessorTable,
    pub instruction_table: InstructionTable,
    pub memory_table: MemoryTable,
    pub input_table: IOTable,
    pub output_table: IOTable,
}

impl TableCollection {
    pub fn new(
        processor_table: ProcessorTable,
        instruction_table: InstructionTable,
        memory_table: MemoryTable,
        input_table: IOTable,
        output_table: IOTable,
    ) -> Self {
        Self {
            processor_table,
            instruction_table,
            memory_table,
            input_table,
            output_table,
        }
    }

    pub fn get_max_degree(&self) -> u64 {
        // TODO: Comment these in when the max_degree is calculated correctly and max_degree test passes.
        [
            self.processor_table.max_degree(),
            // self.instruction_table.max_degree(),
            // self.memory_table.max_degree(),
            // self.input_table.max_degree(),
            // self.output_table.max_degree(),
        ]
        .iter()
        .max()
        .unwrap_or(&1)
        .to_owned() as u64

        // for air in self.processor_table.base_transition_constraints() {
        //     let degree_bounds: Vec<i64> = vec![
        //         self.processor_table.0.interpolant_degree() as i64;
        //         self.processor_table.0.base_width * 2
        //     ];
        //     let degree = air.symbolic_degree_bound(&degree_bounds)
        //         - (self.processor_table.0.height - 1) as i64;
        //     max_degree = std::cmp::max(degree, max_degree);
        // }
    }
}

#[cfg(test)]
mod brainfuck_table_collection_tests {
    use super::*;
    use crate::shared_math::{
        b_field_element::BFieldElement,
        stark::brainfuck::{
            self,
            vm::{BaseMatrices, Register},
        },
        traits::{GetPrimitiveRootOfUnity, IdentityValues},
    };

    static PRINT_EXCLAMATION_MARKS: &str = ">++++++++++[>+++><<-]>+++><<>.................";
    // EXPECTED:
    // max_degree = 1153
    // max_degree = 2047
    // fri_domain_length = 8192

    #[test]
    fn max_degree_test() {
        let actual_program = brainfuck::vm::compile(PRINT_EXCLAMATION_MARKS).unwrap();
        let base_matrices: BaseMatrices =
            brainfuck::vm::simulate(actual_program.clone(), vec![]).unwrap();
        let number_of_randomizers = 1;
        let order = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order)
            .0
            .unwrap();
        let processor_table = ProcessorTable::new(
            base_matrices.processor_matrix.len(),
            number_of_randomizers,
            smooth_generator,
            order as usize,
        );
        let instruction_table = InstructionTable::new(
            base_matrices.instruction_matrix.len(),
            number_of_randomizers,
            smooth_generator,
            order as usize,
        );
        let memory_table = MemoryTable::new(
            base_matrices.processor_matrix.len(),
            number_of_randomizers,
            smooth_generator,
            order as usize,
        );
        let input_table = IOTable::new_input_table(
            base_matrices.input_matrix.len(),
            smooth_generator,
            order as usize,
        );
        let output_table = IOTable::new_output_table(
            base_matrices.input_matrix.len(),
            smooth_generator,
            order as usize,
        );

        let table_collection = TableCollection::new(
            processor_table,
            instruction_table,
            memory_table,
            input_table,
            output_table,
        );

        assert_eq!(1153, table_collection.get_max_degree());
    }
}
