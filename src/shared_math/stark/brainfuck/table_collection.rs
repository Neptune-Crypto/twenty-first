use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::fri::FriDomain;
use crate::shared_math::mpolynomial::Degree;

use super::instruction_table::InstructionTable;
use super::io_table::IOTable;
use super::memory_table::MemoryTable;
use super::processor_table::ProcessorTable;
use super::table::TableTrait;

#[derive(Debug, Clone)]
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
            self.instruction_table.max_degree(),
            self.memory_table.max_degree(),
            self.input_table.max_degree(),
            self.output_table.max_degree(),
        ]
        .iter()
        .max()
        .unwrap_or(&1)
        .to_owned() as u64
    }

    pub fn get_all_base_degree_bounds(&self) -> Vec<Degree> {
        [
            vec![self.processor_table.interpolant_degree(); self.processor_table.base_width()],
            vec![self.instruction_table.interpolant_degree(); self.instruction_table.base_width()],
            vec![self.memory_table.interpolant_degree(); self.memory_table.base_width()],
            vec![self.input_table.interpolant_degree(); self.input_table.base_width()],
            vec![self.output_table.interpolant_degree(); self.output_table.base_width()],
        ]
        .concat()
    }

    // TODO: Add small test of this function
    pub fn get_and_set_all_base_codewords(
        &mut self,
        fri_domain: &FriDomain<BFieldElement>,
    ) -> Vec<Vec<BFieldElement>> {
        [
            self.processor_table.0.lde(&fri_domain),
            self.instruction_table.0.lde(&fri_domain),
            self.memory_table.0.lde(&fri_domain),
            self.input_table.0.lde(&fri_domain),
            self.output_table.0.lde(&fri_domain),
        ]
        .concat()
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
    static HELLO_WORLD: &str = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
    static PRINT_17_CHARS: &str = ",.................";
    // EXPECTED:
    // max_degree = 1153
    // max_degree = 2047
    // fri_domain_length = 8192

    #[test]
    fn max_degree_test() {
        let actual_program = brainfuck::vm::compile(PRINT_EXCLAMATION_MARKS).unwrap();
        let input_data = vec![];
        let table_collection = create_table_collection(&actual_program, &input_data);

        // 1153 is derived from running Python Brainfuck Stark
        assert_eq!(1153, table_collection.get_max_degree());
    }

    #[test]
    fn base_degree_bounds_test() {
        let program_small = brainfuck::vm::compile("++++").unwrap();
        let table_collection_small = create_table_collection(&program_small, &[]);
        // observed from Python BF STARK engine with program `++++`:
        // [8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 8, 8, 8, -1, -1]
        assert_eq!(
            vec![8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 8, 8, 8, -1, -1],
            table_collection_small.get_all_base_degree_bounds()
        );

        // observed from Python BF STARK engine with program `,.................`:
        // [32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 32, 32, 32, 0, 31]
        let program_bigger = brainfuck::vm::compile(PRINT_17_CHARS).unwrap();
        let table_collection_bigger =
            create_table_collection(&program_bigger, &[BFieldElement::new(33)]);
        assert_eq!(
            vec![32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 32, 32, 32, 0, 31],
            table_collection_bigger.get_all_base_degree_bounds()
        );
    }

    fn create_table_collection(
        program: &[BFieldElement],
        input_data: &[BFieldElement],
    ) -> TableCollection {
        let base_matrices: BaseMatrices = brainfuck::vm::simulate(program, input_data).unwrap();
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
            base_matrices.output_matrix.len(),
            smooth_generator,
            order as usize,
        );

        TableCollection::new(
            processor_table,
            instruction_table,
            memory_table,
            input_table,
            output_table,
        )
    }
}
