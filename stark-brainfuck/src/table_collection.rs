use itertools::Itertools;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::fri::FriDomain;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::instruction_table::InstructionTable;
use crate::io_table::IOTable;
use crate::memory_table::MemoryTable;
use crate::processor_table::ProcessorTable;
use crate::stark::{EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT, TERMINAL_COUNT};
use crate::table::TableTrait;
use crate::vm::{InstructionMatrixBaseRow, Register};

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

    pub fn get_table_codeword_by_index(&self, index: usize) -> &Vec<Vec<XFieldElement>> {
        match index {
            0 => &self.processor_table.0.extended_codewords,
            1 => &self.instruction_table.0.extended_codewords,
            2 => &self.memory_table.0.extended_codewords,
            3 => &self.input_table.0.extended_codewords,
            4 => &self.output_table.0.extended_codewords,
            _ => panic!("Unrecognized index. Got: {}", index),
        }
    }

    pub fn get_table_interpolant_degree_by_index(&self, index: usize) -> Degree {
        match index {
            0 => self.processor_table.interpolant_degree(),
            1 => self.instruction_table.interpolant_degree(),
            2 => self.memory_table.interpolant_degree(),
            3 => self.input_table.interpolant_degree(),
            4 => self.output_table.interpolant_degree(),
            _ => panic!("Unrecognized index. Got: {}", index),
        }
    }

    pub fn get_max_degree(&self) -> u64 {
        self.into_iter()
            .map(|table| table.max_degree())
            .max()
            .unwrap_or(1) as u64
    }

    pub fn get_all_base_degree_bounds(&self) -> Vec<Degree> {
        self.into_iter()
            .map(|table| vec![table.interpolant_degree(); table.base_width()])
            .concat()
    }

    pub fn get_all_extension_degree_bounds(&self) -> Vec<Degree> {
        self.into_iter()
            .map(|table| {
                let extension_width = table.full_width() - table.base_width();
                vec![table.interpolant_degree(); extension_width]
            })
            .concat()
    }

    pub fn set_constraints(&mut self, challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT]) {
        let processor_table_constraints =
            self.processor_table.transition_constraints_ext(challenges);
        self.processor_table.0.more.transition_constraints_ext = Some(processor_table_constraints);
    }

    pub fn set_matrices(
        &mut self,
        processor_matrix: Vec<Register>,
        instruction_matrix: Vec<InstructionMatrixBaseRow>,
        memory_matrix: Vec<Vec<BFieldElement>>,
        input_matrix: Vec<BFieldElement>,
        output_matrix: Vec<BFieldElement>,
    ) {
        self.processor_table.0.matrix = processor_matrix.into_iter().map(|x| x.into()).collect();
        self.instruction_table.0.matrix =
            instruction_matrix.into_iter().map(|x| x.into()).collect();
        self.input_table.0.matrix = input_matrix.into_iter().map(|x| vec![x]).collect();
        self.output_table.0.matrix = output_matrix.into_iter().map(|x| vec![x]).collect();
        self.memory_table.0.matrix = memory_matrix;
    }

    pub fn pad(&mut self) {
        self.processor_table.pad();
        self.instruction_table.pad();
        self.memory_table.pad();
        self.input_table.pad();
        self.output_table.pad();
    }

    /// Calculate all codewords on the table objects, and return those codewords as a list of codewords
    pub fn get_and_set_all_base_codewords(
        &mut self,
        fri_domain: &FriDomain<XFieldElement>,
    ) -> Vec<Vec<BFieldElement>> {
        [
            self.processor_table.0.lde(fri_domain).0,
            self.instruction_table.0.lde(fri_domain).0,
            self.memory_table.0.lde(fri_domain).0,
            self.input_table.0.lde(fri_domain).0,
            self.output_table.0.lde(fri_domain).0,
        ]
        .concat()
    }

    pub fn get_and_set_all_extension_codewords(
        &mut self,
        fri_domain: &FriDomain<XFieldElement>,
    ) -> Vec<Vec<XFieldElement>> {
        [
            self.processor_table.0.ldex(fri_domain).0,
            self.instruction_table.0.ldex(fri_domain).0,
            self.memory_table.0.ldex(fri_domain).0,
            self.input_table.0.ldex(fri_domain).0,
            self.output_table.0.ldex(fri_domain).0,
        ]
        .concat()
    }

    pub fn extend(
        &mut self,
        all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
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

    // TODO: Replace this with an `.iter().map(|table| ...)` when `extended_codewords` lives on the trait.
    pub fn all_quotients(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
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
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
        terminals: [XFieldElement; TERMINAL_COUNT],
    ) -> Vec<Degree> {
        self.into_iter()
            .map(|table| table.all_quotient_degree_bounds(challenges, terminals))
            .concat()
    }
}

/// Make TableCollection iterable in such a way that each table is seen as a `TableTrait` value.
impl<'a> IntoIterator for &'a TableCollection {
    type Item = &'a dyn TableTrait;

    type IntoIter = std::array::IntoIter<&'a dyn TableTrait, 5>;

    fn into_iter(self) -> Self::IntoIter {
        [
            &self.processor_table as &'a dyn TableTrait,
            &self.instruction_table as &'a dyn TableTrait,
            &self.memory_table as &'a dyn TableTrait,
            &self.input_table as &'a dyn TableTrait,
            &self.output_table as &'a dyn TableTrait,
        ]
        .into_iter()
    }
}

#[cfg(test)]
mod brainfuck_table_collection_tests {
    use super::*;

    use rand::thread_rng;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::convert::TryInto;
    use std::rc::Rc;

    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::polynomial::Polynomial;
    use twenty_first::shared_math::traits::GetRandomElements;
    use twenty_first::shared_math::traits::PrimitiveRootOfUnity;

    use crate as brainfuck;
    use crate::vm::sample_programs;
    use crate::vm::BaseMatrices;

    #[test]
    fn max_degree_test() {
        let actual_program =
            brainfuck::vm::compile(sample_programs::PRINT_EXCLAMATION_MARKS).unwrap();
        let input_data = vec![];
        let table_collection = create_table_collection(&actual_program, &input_data, 1);

        assert_eq!(2049, table_collection.get_max_degree());
    }

    #[test]
    fn degree_bounds_test() {
        let mut expected_bounds: HashMap<&str, (Vec<Degree>, Vec<Degree>, Vec<Degree>)> =
            HashMap::new();

        // The expected values have been found from the Python STARK BF tutorial
        expected_bounds.insert(
            sample_programs::HELLO_WORLD,
            (
                vec![
                    1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 8192, 8192, 8192,
                    8192, -1, 15,
                ],
                vec![1024, 1024, 1024, 1024, 1024, 1024, 8192, -1, 15],
                vec![
                    1023, 1023, 1023, 1023, 1023, 1023, 1023, 10241, 8193, 8193, 1, 2049, 2049,
                    8193, 8193, 8193, 8193, 1023, 9215, 1023, 1023, 1023, 1023, 1025, 1025, 1025,
                    8193, 1025, 1023, 1023, 8191, 8191, 8191, 8193, 8193, 8193, 8193, 8193, 8193,
                    16385, 24575, -2, 0, -1, 14, 0, 14,
                ],
            ),
        );
        expected_bounds.insert(
            sample_programs::ROT13,
            (
                vec![
                    64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 64, 64, 64, 64, 3, 3,
                ],
                vec![64, 64, 64, 64, 128, 128, 64, 3, 3],
                vec![
                    63, 63, 63, 63, 63, 63, 63, 641, 513, 513, 1, 129, 129, 513, 513, 513, 513, 63,
                    575, 63, 63, 127, 127, 129, 129, 129, 1025, 129, 127, 127, 63, 63, 63, 65, 65,
                    65, 65, 65, 65, 129, 191, 2, 0, 2, 2, 0, 2,
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
                1,
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
    fn invariance_of_interpolation_polynomials_argument_multiplied_by_omicron_cheap_test() {
        invariance_of_interpolation_polynomials_argument_multiplied_by_omicron(
            sample_programs::VERY_SIMPLE_PROGRAM,
            512,
        );
    }

    #[test]
    #[ignore]
    fn invariance_of_interpolation_polynomials_argument_multiplied_by_omicron_expensive_test() {
        invariance_of_interpolation_polynomials_argument_multiplied_by_omicron(
            sample_programs::HELLO_WORLD,
            1 << 18,
        );
    }

    fn invariance_of_interpolation_polynomials_argument_multiplied_by_omicron(
        source_code: &str,
        fri_domain_length: usize,
    ) {
        let program = brainfuck::vm::compile(source_code).unwrap();

        // create table collections with different number of randomizers
        let matrices: BaseMatrices = brainfuck::vm::simulate(&program, &[]).unwrap();
        let mut t_collect_0_rand: TableCollection = create_table_collection(&program, &[], 0);
        let mut t_collect_1_rand: TableCollection = create_table_collection(&program, &[], 1);
        let mut t_collect_2_rand: TableCollection = create_table_collection(&program, &[], 2);
        t_collect_0_rand.set_matrices(
            matrices.processor_matrix.clone(),
            matrices.instruction_matrix.clone(),
            matrices.memory_matrix.clone(),
            matrices.input_matrix.clone(),
            matrices.output_matrix.clone(),
        );
        t_collect_1_rand.set_matrices(
            matrices.processor_matrix.clone(),
            matrices.instruction_matrix.clone(),
            matrices.memory_matrix.clone(),
            matrices.input_matrix.clone(),
            matrices.output_matrix.clone(),
        );
        t_collect_2_rand.set_matrices(
            matrices.processor_matrix.clone(),
            matrices.instruction_matrix,
            matrices.memory_matrix.clone(),
            matrices.input_matrix,
            matrices.output_matrix,
        );
        t_collect_0_rand.pad();
        t_collect_1_rand.pad();
        t_collect_2_rand.pad();

        // set up FRI domain
        let fri_domain = FriDomain {
            length: fri_domain_length,
            offset: BFieldElement::new(7).lift(),
            omega: XFieldElement::primitive_root_of_unity(fri_domain_length as u64).unwrap(),
        };

        println!("Starting LDE stuff…");
        // get all interpolation polynomials `ip_x` with different number of randomizers
        let interpolants_with_0_randomizers = [
            t_collect_0_rand.processor_table.0.lde(&fri_domain).1,
            t_collect_0_rand.instruction_table.0.lde(&fri_domain).1,
        ]
        .concat();
        println!("Successfully interpolated table collection with 0 random values");
        let interpolants_with_1_randomizer = [
            t_collect_1_rand.processor_table.0.lde(&fri_domain).1,
            t_collect_1_rand.instruction_table.0.lde(&fri_domain).1,
        ]
        .concat();
        println!("Successfully interpolated table collection with 1 random value");
        let interpolants_with_2_randomizers = [
            t_collect_2_rand.processor_table.0.lde(&fri_domain).1,
            t_collect_2_rand.instruction_table.0.lde(&fri_domain).1,
        ]
        .concat();
        println!("Successfully interpolated table collection with 2 random values");

        // get random parts r_1 = ip_1 - ip_0 and r_2 = ip_2 - ip_0
        let diffs_due_to_1_randomizer = interpolants_with_1_randomizer
            .into_iter()
            .zip(interpolants_with_0_randomizers.clone())
            .map(|(p_1, p_0)| p_1 - p_0)
            .collect::<Vec<_>>();
        println!(
            "The r_1 for the very first column is: {}",
            diffs_due_to_1_randomizer[0]
        );
        let diffs_due_to_2_randomizers = interpolants_with_2_randomizers
            .clone()
            .into_iter()
            .zip(interpolants_with_0_randomizers)
            .map(|(p_2, p_0)| p_2 - p_0)
            .collect::<Vec<_>>();
        println!(
            "The r_2 for the very first column is: {}",
            diffs_due_to_2_randomizers[0]
        );

        // sample random evaluation points. They are re-used across property tests for r_0 and r_1
        let omicron: BFieldElement = t_collect_0_rand.processor_table.omicron();
        println!("omicron = {}", omicron);
        let mut rng = rand::thread_rng();
        let eval_points = &BFieldElement::random_elements(100, &mut rng);

        // check the expected invariants
        for (r_1, x) in diffs_due_to_1_randomizer.iter().zip(eval_points) {
            assert_eq!(
                r_1.evaluate(x),
                r_1.evaluate(&(omicron * *x)),
                "polynomial r_1 must be invariant to its argument being multiplied by ο.
                r_1 was sampled in {} and {}",
                x,
                omicron * *x
            );
        }
        println!("Passed sanity check: all r_1s are invariant to argument being multiplied by ο.");

        // for brainfuck VM, input and output are both public. The corresponding interpolation
        // polynomials are not randomized. Thus, this test does not make sense for them.
        let non_io_columns =
            ProcessorTable::BASE_WIDTH + InstructionTable::BASE_WIDTH + MemoryTable::BASE_WIDTH;
        for (col_id, ((r_2, x), ip_2)) in diffs_due_to_2_randomizers
            .iter()
            .zip(eval_points)
            .zip(interpolants_with_2_randomizers)
            .enumerate()
        {
            if col_id < non_io_columns {
                if r_2.is_zero() {
                    assert!(
                        ip_2.is_zero(),
                        "in column {}, r_2 was zero but ip_2 was not:\n\tip_2 = {}\n\tr_2 = {}",
                        col_id,
                        ip_2,
                        r_2
                    );
                } else {
                    assert_ne!(
                        r_2.evaluate(x),
                        r_2.evaluate(&(omicron * *x)),
                        "polynomial r_2 must not be invariant to its argument being multiplied by ο.
                    column: {}
                    r_2 = {}
                    ip_2 = {}
                    r_2 was sampled in {} and {}",
                        col_id,
                        r_2,
                        ip_2,
                        x,
                        omicron * *x
                    );
                }
            }
        }
    }

    #[test]
    fn get_and_set_all_codewords_test() {
        let program_small = brainfuck::vm::compile(sample_programs::VERY_SIMPLE_PROGRAM).unwrap();
        let matrices: BaseMatrices = brainfuck::vm::simulate(&program_small, &[]).unwrap();
        let table_collection: TableCollection = create_table_collection(&program_small, &[], 1);
        let tc_ref = Rc::new(RefCell::new(table_collection));
        tc_ref.borrow_mut().set_matrices(
            matrices.processor_matrix.clone(),
            matrices.instruction_matrix.clone(),
            matrices.memory_matrix.clone(),
            matrices.input_matrix.clone(),
            matrices.output_matrix.clone(),
        );
        tc_ref.borrow_mut().pad();

        let mock_fri_domain_length = 512;
        let fri_domain = FriDomain {
            length: mock_fri_domain_length,
            offset: BFieldElement::new(7).lift(),
            omega: XFieldElement::primitive_root_of_unity(mock_fri_domain_length as u64).unwrap(),
        };

        let base_codewords = tc_ref
            .borrow_mut()
            .get_and_set_all_base_codewords(&fri_domain);
        assert_eq!(
            16,
            base_codewords.len(),
            "Number of base tables must match that from Python tutorial"
        );
        let interpolants: Vec<Polynomial<BFieldElement>> = base_codewords
            .iter()
            .map(|bc| fri_domain.b_interpolate(bc))
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
        let all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT] =
            XFieldElement::random_elements(EXTENSION_CHALLENGE_COUNT, &mut rng)
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
            "Number of extension columns must match that from Python tutorial"
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
        number_of_randomizers: usize,
    ) -> TableCollection {
        let base_matrices: BaseMatrices = brainfuck::vm::simulate(program, input_data).unwrap();
        let order = 1 << 32;
        let smooth_generator = BFieldElement::primitive_root_of_unity(order).unwrap();

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
            base_matrices.memory_matrix.len(),
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
