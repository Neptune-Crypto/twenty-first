use crate::shared_math::{
    b_field_element::BFieldElement, mpolynomial::MPolynomial, other, x_field_element::XFieldElement,
};

use super::{
    stark::{EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT},
    table::{Table, TableMoreTrait, TableTrait},
};

#[derive(Debug, Clone)]
pub struct IOTable(pub Table<IOTableMore>);

#[derive(Debug, Clone)]
pub struct IOTableMore {
    pub challenge_index: usize,
    pub terminal_index: usize,
    pub evaluation_terminal: XFieldElement,
}

impl TableMoreTrait for IOTableMore {
    fn new_more() -> Self {
        IOTableMore {
            challenge_index: 0,
            terminal_index: 0,
            evaluation_terminal: XFieldElement::ring_zero(),
        }
    }
}

impl IOTable {
    // named indices for base columns
    pub const COLUMN: usize = 0;

    // named indices for extension columns
    pub const EVALUATION: usize = 1;

    pub const BASE_WIDTH: usize = 1;
    pub const FULL_WIDTH: usize = 2;

    // TODO: Refactor to avoid duplicate code.
    pub fn new_input_table(length: usize, generator: BFieldElement, order: usize) -> Self {
        let num_randomizers = 0;

        let mut table = Table::<IOTableMore>::new(
            Self::BASE_WIDTH,
            Self::FULL_WIDTH,
            length,
            num_randomizers,
            generator,
            order,
        );

        table.more.challenge_index = 8;
        table.more.terminal_index = 2;

        Self(table)
    }

    pub fn new_output_table(length: usize, generator: BFieldElement, order: usize) -> Self {
        let num_randomizers = 0;
        let base_width = 1;
        let full_width = 2;

        let mut table = Table::<IOTableMore>::new(
            base_width,
            full_width,
            length,
            num_randomizers,
            generator,
            order,
        );

        table.more.challenge_index = 9;
        table.more.terminal_index = 3;

        Self(table)
    }

    pub fn pad(&mut self) {
        // TODO: The current python code does something else here
        while self.0.matrix.len() != 0 && !other::is_power_of_two(self.0.matrix.len()) {
            let padding: Vec<BFieldElement> = vec![BFieldElement::ring_zero()];
            self.0.matrix.push(padding);
        }
    }

    pub fn challenge_index(&self) -> usize {
        self.0.more.challenge_index
    }

    pub fn terminal_index(&self) -> usize {
        self.0.more.terminal_index
    }
}

impl TableTrait for IOTable {
    fn base_width(&self) -> usize {
        self.0.base_width
    }

    fn full_width(&self) -> usize {
        self.0.full_width
    }

    fn length(&self) -> usize {
        self.0.length
    }

    fn num_randomizers(&self) -> usize {
        self.0.num_randomizers
    }

    fn height(&self) -> usize {
        self.0.height
    }

    fn omicron(&self) -> BFieldElement {
        self.0.omicron
    }

    fn generator(&self) -> BFieldElement {
        self.0.generator
    }

    fn order(&self) -> usize {
        self.0.order
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        vec![]
    }

    fn base_boundary_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        vec![]
    }

    fn extend(
        &mut self,
        all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
        _all_initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT],
    ) {
        // `iota` is called `gamma` or `delta` in the other `extend()`s.
        let iota = all_challenges[self.0.more.challenge_index];
        let zero = XFieldElement::ring_zero();

        // prepare loop
        let mut extended_matrix: Vec<Vec<XFieldElement>> =
            vec![Vec::with_capacity(self.full_width()); self.0.matrix.len()];
        let mut io_running_evaluation = zero;
        let mut evaluation_terminal = zero;

        // loop over all rows of table
        for (i, row) in self.0.matrix.iter().enumerate() {
            // TODO: Avoid re-allocating each row a second time; `extended_matrix` is pre-allocated.

            // first, copy over existing row
            // new_row = [xfield.lift(nr) for nr in row]
            let mut new_row: Vec<XFieldElement> = row.into_iter().map(|bfe| bfe.lift()).collect();

            // io_running_evaluation = io_running_evaluation * iota + new_row[IOTable.column]
            io_running_evaluation = io_running_evaluation * iota + new_row[IOTable::COLUMN];

            // new_row += [io_running_evaluation]
            new_row.push(io_running_evaluation);

            // if i == self.length - 1:
            //     evaluation_terminal = io_running_evaluation
            if i == self.length() - 1 {
                evaluation_terminal = io_running_evaluation;
            }

            // extended_matrix += [new_row]
            extended_matrix[i] = new_row;
        }

        assert!(
            self.height() == 0 || other::is_power_of_two(self.height()),
            "height of io_table must be 2^k, or 0"
        );

        // self.matrix = extended_matrix
        self.0.extended_matrix = extended_matrix;

        // self.codewords = [[xfield.lift(c) for c in cdwd]
        //                   for cdwd in self.codewords]
        self.0.extended_codewords = self
            .0
            .codewords
            .iter()
            .map(|row| row.iter().map(|elem| elem.lift()).collect())
            .collect();

        // self.evaluation_terminal = evaluation_terminal
        self.0.more.evaluation_terminal = evaluation_terminal;
    }

    fn transition_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
    ) -> Vec<MPolynomial<XFieldElement>> {
        let variable_count = Self::FULL_WIDTH * 2;
        let vars =
            MPolynomial::<XFieldElement>::variables(variable_count, XFieldElement::ring_one());
        let iota = MPolynomial::from_constant(challenges[self.challenge_index()], variable_count);

        // let _input = vars[0];
        let evaluation = vars[1].clone();
        let input_next = vars[2].clone();
        let evaluation_next = vars[3].clone();

        // polynomials = []
        // polynomials += [evaluation * iota + input_next - evaluation_next]
        // return polynomials
        vec![evaluation * iota + input_next - evaluation_next]
    }

    fn boundary_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
    ) -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }

    fn terminal_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
        terminals: [XFieldElement; super::stark::TERMINAL_COUNT as usize],
    ) -> Vec<MPolynomial<XFieldElement>> {
        todo!()
    }
}

#[cfg(test)]
mod io_table_tests {
    use super::*;
    use crate::shared_math::stark::brainfuck;
    use crate::shared_math::stark::brainfuck::vm::sample_programs;
    use crate::shared_math::stark::brainfuck::vm::BaseMatrices;
    use crate::shared_math::traits::GetRandomElements;
    use crate::shared_math::traits::{GetPrimitiveRootOfUnity, IdentityValues};
    use rand::thread_rng;
    use std::cmp::max;
    use std::convert::TryInto;

    // When we simulate a program, this generates a collection of matrices that contain
    // "abstract" execution traces. When we evaluate the base transition constraints on
    // the rows (points) from the InstructionTable matrix, these should evaluate to zero.
    #[test]
    fn io_table_constraints_evaluate_to_zero_on_test() {
        let mut rng = thread_rng();

        for source_code in sample_programs::get_all_sample_programs().iter() {
            // Run program
            let actual_program = brainfuck::vm::compile(source_code).unwrap();
            let input_data = vec![
                BFieldElement::new(76),
                BFieldElement::new(79),
                BFieldElement::new(76),
            ];
            let base_matrices: BaseMatrices =
                brainfuck::vm::simulate(&actual_program, &input_data).unwrap();

            let _number_of_randomizers = 2;
            let order = 1 << 32;
            let smooth_generator = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();

            // instantiate table objects
            let input_table_: IOTable = IOTable::new_input_table(
                base_matrices.input_matrix.len(),
                smooth_generator,
                order as usize,
            );

            let output_table_: IOTable = IOTable::new_output_table(
                base_matrices.output_matrix.len(),
                smooth_generator,
                order as usize,
            );

            // Prepare test cases
            let mut cases = [
                (input_table_, base_matrices.input_matrix),
                (output_table_, base_matrices.output_matrix),
            ];

            for (io_table, io_matrix) in cases.iter_mut() {
                // Test base transition constraints
                let io_air_constraints = io_table.base_transition_constraints();

                assert_eq!(
                    0,
                    io_air_constraints.len(),
                    "There are exactly 0 base AIR constraints for IOTable"
                );

                let matrix_len = io_table.0.matrix.len() as isize;
                let base_steps = max(0, matrix_len - 1) as usize;
                for step in 0..base_steps {
                    let row: BFieldElement = io_matrix[step].clone();
                    let next_row: BFieldElement = io_matrix[step + 1].clone();
                    let point: Vec<BFieldElement> = vec![row, next_row];

                    // Since there are no base AIR constraints on either IOTables,
                    // the following for loop is never entered. This is a trivial
                    // test, but it should still hold.
                    for air_constraint in io_air_constraints.iter() {
                        assert!(air_constraint.evaluate(&point).is_zero());
                    }
                }

                // Test base transition constraints after padding
                io_table.0.matrix = io_matrix.into_iter().map(|x| vec![x.clone()]).collect();
                io_table.pad();

                let padded_matrix_len = io_table.0.matrix.len() as isize;
                assert!(
                    padded_matrix_len == 0 || other::is_power_of_two(padded_matrix_len),
                    "Matrix length must be power of 2 after padding"
                );

                let padded_steps = max(0, padded_matrix_len - 1) as usize;
                for step in 0..padded_steps {
                    let row = io_table.0.matrix[step].clone();
                    let next_row = io_table.0.matrix[step + 1].clone();
                    let point: Vec<BFieldElement> = vec![row, next_row].concat();

                    // Since there are no base AIR constraints on either IOTables,
                    // the following for loop is never entered. This is a trivial
                    // test, but it should still hold.
                    for air_constraint in io_air_constraints.iter() {
                        assert!(air_constraint.evaluate(&point).is_zero());
                    }
                }

                // Test transition constraints on extension table
                let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize] =
                    XFieldElement::random_elements(EXTENSION_CHALLENGE_COUNT as usize, &mut rng)
                        .try_into()
                        .unwrap();

                let initials =
                    XFieldElement::random_elements(PERMUTATION_ARGUMENTS_COUNT, &mut rng)
                        .try_into()
                        .unwrap();

                io_table.extend(challenges, initials);

                // Get transition constraints for extension table instead
                let io_air_constraints_ext = io_table.transition_constraints_ext(challenges);

                assert_eq!(
                    1,
                    io_air_constraints_ext.len(),
                    "There is exactly 1 extension AIR constraint for IOTable"
                );

                let extended_matrix_len = io_table.0.extended_matrix.len() as isize;
                let extended_steps = max(0, extended_matrix_len - 1) as usize;
                for step in 0..extended_steps {
                    let row = io_table.0.extended_matrix[step].clone();
                    let next_row = io_table.0.extended_matrix[step + 1].clone();
                    let xpoint: Vec<XFieldElement> = vec![row, next_row].concat();

                    for air_constraint_ext in io_air_constraints_ext.iter() {
                        assert!(air_constraint_ext.evaluate(&xpoint).is_zero());
                    }
                }
            }
        }
    }
}
