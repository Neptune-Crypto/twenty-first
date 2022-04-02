use super::instruction_table::InstructionTable;
use super::io_table::IOTable;
use super::memory_table::MemoryTable;
use super::processor_table::ProcessorTable;
use super::stark::{EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT};
use super::table::TableTrait;
use super::vm::{InstructionMatrixBaseRow, Register};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::fri::FriDomain;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::x_field_element::XFieldElement;

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

    pub fn set_matrices(
        &mut self,
        processor_matrix: Vec<Register>,
        instruction_matrix: Vec<InstructionMatrixBaseRow>,
        input_matrix: Vec<BFieldElement>,
        output_matrix: Vec<BFieldElement>,
    ) {
        self.processor_table.0.matrix = processor_matrix.into_iter().map(|x| x.into()).collect();
        self.instruction_table.0.matrix =
            instruction_matrix.into_iter().map(|x| x.into()).collect();
        self.input_table.0.matrix = input_matrix.into_iter().map(|x| vec![x]).collect();
        self.output_table.0.matrix = output_matrix.into_iter().map(|x| vec![x]).collect();
    }

    pub fn pad(&mut self) {
        self.processor_table.pad();
        self.instruction_table.pad();
        self.input_table.pad();
        self.output_table.pad();
    }

    /// Calculate all codewords on the table objects, and return those codewords as a list of codewords
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

    pub fn extend(
        &mut self,
        all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
        all_initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT],
    ) {
        self.processor_table.extend(all_challenges, all_initials);
        self.instruction_table.extend(all_challenges, all_initials);
        // self.memory_table.extend(all_challenges, all_initials);
        self.input_table.extend(all_challenges, all_initials);
        self.output_table.extend(all_challenges, all_initials);
    }
}

#[cfg(test)]
mod brainfuck_table_collection_tests {
    use super::*;
    use crate::shared_math::stark::brainfuck::vm::sample_programs;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::stark::brainfuck;
    use crate::shared_math::stark::brainfuck::vm::BaseMatrices;
    use crate::shared_math::traits::GetPrimitiveRootOfUnity;
    use std::cell::RefCell;
    use std::rc::Rc;

    // EXPECTED:
    // max_degree = 1153
    // max_degree = 2047
    // fri_domain_length = 8192

    #[test]
    fn max_degree_test() {
        let actual_program = brainfuck::vm::compile(sample_programs::PRINT_EXCLAMATION_MARKS).unwrap();
        let input_data = vec![];
        let table_collection = create_table_collection(&actual_program, &input_data);

        // 1153 is derived from running Python Brainfuck Stark
        assert_eq!(1153, table_collection.get_max_degree());
    }

    #[test]
    fn base_degree_bounds_test() {
        let program_small = brainfuck::vm::compile(sample_programs::VERY_SIMPLE_PROGRAM).unwrap();
        let table_collection_small = create_table_collection(&program_small, &[]);
        // observed from Python BF STARK engine with program `++++`:
        // [8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 8, 8, 8, -1, -1]
        assert_eq!(
            vec![8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 8, 8, 8, -1, -1],
            table_collection_small.get_all_base_degree_bounds()
        );

        // observed from Python BF STARK engine with program `,.................`:
        // [32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 32, 32, 32, 0, 31]
        let program_bigger = brainfuck::vm::compile(sample_programs::PRINT_17_CHARS).unwrap();
        let table_collection_bigger =
            create_table_collection(&program_bigger, &[BFieldElement::new(33)]);
        assert_eq!(
            vec![32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 32, 32, 32, 0, 31],
            table_collection_bigger.get_all_base_degree_bounds()
        );

        // observed from Python BF STARK engine with program
        // `++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.`:
        // [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, -1, 15]
        let program_hello_world = brainfuck::vm::compile(sample_programs::HELLO_WORLD).unwrap();
        let table_collection_bigger =
            create_table_collection(&program_hello_world, &[BFieldElement::new(33)]);
        assert_eq!(
            vec![
                1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, -1,
                15,
            ],
            table_collection_bigger.get_all_base_degree_bounds()
        );
    }

    #[test]
    fn get_and_set_all_base_codewords_test() {
        let program_small = brainfuck::vm::compile(sample_programs::VERY_SIMPLE_PROGRAM).unwrap();
        let matrices: BaseMatrices = brainfuck::vm::simulate(&program_small, &[]).unwrap();
        let table_collection: TableCollection = create_table_collection(&program_small, &[]);
        let tc_ref = Rc::new(RefCell::new(table_collection));
        tc_ref.borrow_mut().set_matrices(
            matrices.processor_matrix,
            matrices.instruction_matrix,
            matrices.input_matrix,
            matrices.output_matrix,
        );
        tc_ref.borrow_mut().pad();

        // Instantiate the memory table object
        let processor_matrix_clone = tc_ref.borrow().processor_table.0.matrix.clone();
        tc_ref.borrow_mut().memory_table.0.matrix =
            MemoryTable::derive_matrix(processor_matrix_clone);

        let mock_fri_domain_length = 256;
        let fri_domain = FriDomain {
            length: mock_fri_domain_length,
            offset: BFieldElement::new(7),
            omega: BFieldElement::ring_zero()
                .get_primitive_root_of_unity(mock_fri_domain_length as u128)
                .0
                .unwrap(),
        };

        assert_eq!(
            15,
            tc_ref
                .borrow_mut()
                .get_and_set_all_base_codewords(&fri_domain)
                .len(),
            "Number of base tables must match that from Python tutorial"
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
