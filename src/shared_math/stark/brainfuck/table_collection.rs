use super::instruction_table::InstructionTable;
use super::io_table::IOTable;
use super::memory_table::MemoryTable;
use super::processor_table::ProcessorTable;
use super::stark::{EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT, TERMINAL_COUNT};
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

    pub fn get_all_extension_degree_bounds(&self) -> Vec<Degree> {
        [
            vec![
                self.processor_table.interpolant_degree();
                self.processor_table.full_width() - self.processor_table.base_width()
            ],
            vec![
                self.instruction_table.interpolant_degree();
                self.instruction_table.full_width() - self.instruction_table.base_width()
            ],
            vec![
                self.memory_table.interpolant_degree();
                self.memory_table.full_width() - self.memory_table.base_width()
            ],
            vec![
                self.input_table.interpolant_degree();
                self.input_table.full_width() - self.input_table.base_width()
            ],
            vec![
                self.output_table.interpolant_degree();
                self.output_table.full_width() - self.output_table.base_width()
            ],
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

    pub fn get_and_set_all_extension_codewords(
        &mut self,
        fri_domain: &FriDomain<BFieldElement>,
    ) -> Vec<Vec<XFieldElement>> {
        [
            self.processor_table.0.ldex(fri_domain),
            self.instruction_table.0.ldex(fri_domain),
            self.memory_table.0.ldex(fri_domain),
            self.input_table.0.ldex(fri_domain),
            self.output_table.0.ldex(fri_domain),
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
        self.memory_table.extend(all_challenges, all_initials);
        self.input_table.extend(all_challenges, all_initials);
        self.output_table.extend(all_challenges, all_initials);
    }

    pub fn get_terminals(&self) -> [XFieldElement; TERMINAL_COUNT] {
        [
            self.processor_table.0.more.instruction_permutation_terminal,
            self.processor_table.0.more.memory_permutation_terminal,
            self.processor_table.0.more.input_evaluation_terminal,
            self.processor_table.0.more.output_evaluation_terminal,
            self.instruction_table.0.more.evaluation_terminal,
        ]
    }

    pub fn all_quotients(
        &self,
        fri_domain: &FriDomain<BFieldElement>,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
        terminals: [XFieldElement; TERMINAL_COUNT],
    ) -> Vec<Vec<XFieldElement>> {
        let pt = self.processor_table.all_quotients(
            fri_domain,
            &self.processor_table.0.extended_codewords,
            challenges,
            terminals,
        );
        let instt = self.instruction_table.all_quotients(
            fri_domain,
            &self.instruction_table.0.extended_codewords,
            challenges,
            terminals,
        );
        let mt = self.memory_table.all_quotients(
            fri_domain,
            &self.memory_table.0.extended_codewords,
            challenges,
            terminals,
        );
        let inpt = self.input_table.all_quotients(
            fri_domain,
            &self.input_table.0.extended_codewords,
            challenges,
            terminals,
        );
        let ot = self.output_table.all_quotients(
            fri_domain,
            &self.output_table.0.extended_codewords,
            challenges,
            terminals,
        );

        vec![pt, instt, mt, inpt, ot].concat()
    }

    pub fn all_quotient_degree_bounds(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
        terminals: [XFieldElement; TERMINAL_COUNT],
    ) -> Vec<Degree> {
        let mut i = 0;
        let pt = self
            .processor_table
            .all_quotient_degree_bounds(challenges, terminals);
        let instt = self
            .instruction_table
            .all_quotient_degree_bounds(challenges, terminals);
        let mt = self
            .memory_table
            .all_quotient_degree_bounds(challenges, terminals);
        let inptt = self
            .input_table
            .all_quotient_degree_bounds(challenges, terminals);
        let ot = self
            .output_table
            .all_quotient_degree_bounds(challenges, terminals);
        vec![pt, instt, mt, inptt, ot].concat()
    }
}

#[cfg(test)]
mod brainfuck_table_collection_tests {
    use rand::thread_rng;

    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::polynomial::Polynomial;
    use crate::shared_math::stark::brainfuck;
    use crate::shared_math::stark::brainfuck::vm::sample_programs;
    use crate::shared_math::stark::brainfuck::vm::BaseMatrices;
    use crate::shared_math::traits::GetPrimitiveRootOfUnity;
    use crate::shared_math::traits::GetRandomElements;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::convert::TryInto;
    use std::rc::Rc;

    // EXPECTED:
    // max_degree = 1153
    // max_degree = 2047
    // fri_domain_length = 8192

    #[test]
    fn max_degree_test() {
        let actual_program =
            brainfuck::vm::compile(sample_programs::PRINT_EXCLAMATION_MARKS).unwrap();
        let input_data = vec![];
        let table_collection = create_table_collection(&actual_program, &input_data);

        // 1153 is derived from running Python Brainfuck Stark
        assert_eq!(1153, table_collection.get_max_degree());
    }

    #[test]
    fn degree_bounds_test() {
        let mut expected_bounds: HashMap<&str, (Vec<Degree>, Vec<Degree>, Vec<Degree>)> =
            HashMap::new();

        // The expected values have been found from the Python STARK BF tutorial
        expected_bounds.insert(
            sample_programs::VERY_SIMPLE_PROGRAM,
            (
                vec![8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 8, 8, 8, -1, -1],
                vec![8, 8, 8, 8, 16, 16, 8, -1, -1],
                vec![
                    7, 7, 7, 7, 7, 7, 7, 73, 65, 65, 1, 9, 9, 17, 9, 65, 65, 7, 15, 7, 7, 15, 15,
                    17, 17, 17, 33, 17, 47, 15, 7, 7, 7, 9, 17, 9, 9, 15, -2, 0, -1, -2, 0, -1,
                ],
            ),
        );
        expected_bounds.insert(
            sample_programs::PRINT_17_CHARS,
            (
                vec![32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 32, 32, 32, 0, 31],
                vec![32, 32, 32, 32, 64, 64, 32, 0, 31],
                vec![
                    31, 31, 31, 31, 31, 31, 31, 289, 257, 257, 1, 33, 33, 65, 33, 257, 257, 31, 63,
                    31, 31, 63, 63, 65, 65, 65, 129, 65, 191, 63, 31, 31, 31, 33, 65, 33, 33, 63,
                    -1, 0, -1, 30, 0, 30,
                ],
            ),
        );
        expected_bounds.insert(
            sample_programs::HELLO_WORLD,
            (
                vec![
                    1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                    -1, 15,
                ],
                vec![1024, 1024, 1024, 1024, 1024, 1024, 1024, -1, 15],
                vec![
                    1023, 1023, 1023, 1023, 1023, 1023, 1023, 9217, 8193, 8193, 1, 1025, 1025,
                    2049, 1025, 8193, 8193, 1023, 2047, 1023, 1023, 1023, 1023, 1025, 1025, 1025,
                    2049, 1025, 3071, 1023, 1023, 1023, 1023, 1025, 2049, 1025, 1025, 2047, -2, 0,
                    -1, 14, 0, 14,
                ],
            ),
        );
        expected_bounds.insert(
            sample_programs::ROT13,
            (
                vec![64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 64, 64, 64, 3, 3],
                vec![64, 64, 64, 64, 128, 128, 64, 3, 3],
                vec![
                    63, 63, 63, 63, 63, 63, 63, 577, 513, 513, 1, 65, 65, 129, 65, 513, 513, 63,
                    127, 63, 63, 127, 127, 129, 129, 129, 257, 129, 383, 127, 63, 63, 63, 65, 129,
                    65, 65, 127, 2, 0, 2, 2, 0, 2,
                ],
            ),
        );

        // Verify that `get_all_base_degree_bounds` and `get_all_extension_degree_bounds` return
        // the expected values
        for (code, (expected_base_bounds, expected_extension_bounds, expected_quotient_bounds)) in
            expected_bounds.into_iter()
        {
            let program = brainfuck::vm::compile(code).unwrap();
            let table_collection = create_table_collection(
                &program,
                &[
                    BFieldElement::new(33),
                    BFieldElement::new(34),
                    BFieldElement::new(35),
                ],
            );
            assert_eq!(
                expected_base_bounds,
                table_collection.get_all_base_degree_bounds(),
                "base degree bounds must match expected value from Python BF-STARK tutorial for code {}", code
            );
            assert_eq!(
                expected_extension_bounds,
                table_collection.get_all_extension_degree_bounds(),
                "extension degree bounds must match expected value from Python BF-STARK tutorial for code {}", code
            );
            let mut rng = thread_rng();
            let challenges: [XFieldElement; 11] = XFieldElement::random_elements(11, &mut rng)
                .try_into()
                .unwrap();
            let terminals: [XFieldElement; 5] = XFieldElement::random_elements(5, &mut rng)
                .try_into()
                .unwrap();
            assert_eq!(
                expected_quotient_bounds,
                table_collection.all_quotient_degree_bounds(challenges, terminals),
                "extension degree bounds must match expected value from Python BF-STARK tutorial for code {}", code
            );
        }
    }

    #[test]
    fn get_and_set_all_codewords_test() {
        let program_small = brainfuck::vm::compile(sample_programs::VERY_SIMPLE_PROGRAM).unwrap();
        let matrices: BaseMatrices = brainfuck::vm::simulate(&program_small, &[]).unwrap();
        let table_collection: TableCollection = create_table_collection(&program_small, &[]);
        let tc_ref = Rc::new(RefCell::new(table_collection));
        tc_ref.borrow_mut().set_matrices(
            matrices.processor_matrix.clone(),
            matrices.instruction_matrix.clone(),
            matrices.input_matrix.clone(),
            matrices.output_matrix.clone(),
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

        let base_codewords = tc_ref
            .borrow_mut()
            .get_and_set_all_base_codewords(&fri_domain);
        assert_eq!(
            15,
            base_codewords.len(),
            "Number of base tables must match that from Python tutorial"
        );
        let interpolants: Vec<Polynomial<BFieldElement>> = base_codewords
            .iter()
            .map(|bc| fri_domain.interpolate(bc))
            .collect();

        // Verify that the FRI-domain evaluations derived from the matrix values correspond with
        // the matrix values
        let mut i = 0;
        for column_index in 0..tc_ref.borrow().processor_table.0.base_width {
            for (j, registers) in matrices.processor_matrix.clone().into_iter().enumerate() {
                let register_values: Vec<BFieldElement> = registers.into();
                assert_eq!(
                    register_values[column_index],
                    interpolants[i]
                        .evaluate(&tc_ref.borrow().processor_table.omicron().mod_pow(j as u64)),
                    "The interpolation of the FRI-domain evaluations must agree with the execution trace from processor table"
                )
            }
            i += 1;
        }
        for column_index in 0..tc_ref.borrow().instruction_table.0.base_width {
            for (j, row) in matrices.instruction_matrix.clone().into_iter().enumerate() {
                let row_values: Vec<BFieldElement> = row.into();
                assert_eq!(
                    row_values[column_index],
                    interpolants[i]
                        .evaluate(&tc_ref.borrow().instruction_table.omicron().mod_pow(j as u64)),
                    "The interpolation of the FRI-domain evaluations must agree with the execution trace from instruction table"
                )
            }
            i += 1;
        }
        for column_index in 0..tc_ref.borrow().input_table.0.base_width {
            for (j, element) in matrices.input_matrix.clone().into_iter().enumerate() {
                let row = vec![element];
                assert_eq!(
                    row[column_index],
                    interpolants[i]
                        .evaluate(&tc_ref.borrow().input_table.omicron().mod_pow(j as u64)),
                    "The interpolation of the FRI-domain evaluations must agree with the execution trace from input table"
                )
            }
            i += 1;
        }
        for column_index in 0..tc_ref.borrow().output_table.0.base_width {
            for (j, element) in matrices.output_matrix.clone().into_iter().enumerate() {
                let row = vec![element];
                assert_eq!(
                    row[column_index],
                    interpolants[i]
                        .evaluate(&tc_ref.borrow().output_table.omicron().mod_pow(j as u64)),
                    "The interpolation of the FRI-domain evaluations must agree with the execution trace from output table"
                )
            }
            i += 1;
        }

        assert!(
            base_codewords
                .iter()
                .all(|ec| ec.len() == fri_domain.length),
            "All base codewords must be evaluated over the Ω domain",
        );

        // Extend and verify that extension codewords are also calculated correctly
        let mut rng = thread_rng();
        let all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize] =
            XFieldElement::random_elements(EXTENSION_CHALLENGE_COUNT as usize, &mut rng)
                .try_into()
                .unwrap();
        let all_initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT as usize] =
            XFieldElement::random_elements(PERMUTATION_ARGUMENTS_COUNT as usize, &mut rng)
                .try_into()
                .unwrap();

        tc_ref.borrow_mut().extend(all_challenges, all_initials);
        let extended_codewords: Vec<Vec<XFieldElement>> = tc_ref
            .borrow_mut()
            .get_and_set_all_extension_codewords(&fri_domain);
        assert_eq!(
            9,
            extended_codewords.len(),
            "Number of extension tables must match that from Python tutorial"
        );

        assert!(
            extended_codewords
                .iter()
                .all(|ec| ec.len() == fri_domain.length),
            "All extension codewords must be evaluated over the Ω domain",
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
