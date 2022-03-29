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
        all_initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT],
    ) {
        let iota = all_challenges[self.0.more.challenge_index];

        // algebra stuff
        let zero = XFieldElement::ring_zero();
        let one = XFieldElement::ring_one();

        // prepare loop
        let mut extended_matrix: Vec<Vec<XFieldElement>> =
            vec![Vec::with_capacity(self.full_width()); self.0.matrix.len()];
        let mut io_running_evaluation = zero;
        let mut evaluation_terminal = zero;

        // loop over all rows of table
        for (i, row) in self.0.matrix.iter().enumerate() {
            // first, copy over existing row
            // new_row = [xfield.lift(nr) for nr in row]
            let mut new_row: Vec<XFieldElement> = row
                .into_iter()
                .map(|&bfe| XFieldElement::new_const(bfe))
                .collect();

            // io_running_evaluation = io_running_evaluation * iota + new_row[IOTable.column]
            let io_running_evaluation = io_running_evaluation * iota + new_row[IOTable::COLUMN];

            // new_row += [io_running_evaluation]
            new_row.push(io_running_evaluation);

            // if i == self.length - 1:
            //     evaluation_terminal = io_running_evaluation
            if i == self.length() - 1 {
                evaluation_terminal = io_running_evaluation;
            }

            extended_matrix[i] = new_row;
        }

        assert!(
            other::is_power_of_two(self.height()),
            "height of io_table must be 2^k"
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
}

#[cfg(test)]
mod io_table_tests {
    use std::convert::TryInto;

    use rand::thread_rng;

    use super::*;
    use crate::shared_math::stark::brainfuck::vm::{InstructionMatrixBaseRow, MemoryMatrixBaseRow};
    use crate::shared_math::traits::GetRandomElements;
    use crate::shared_math::{
        stark::brainfuck::{
            self,
            vm::{BaseMatrices, Register},
        },
        traits::{GetPrimitiveRootOfUnity, IdentityValues},
    };

    static VERY_SIMPLE_PROGRAM: &str = "++++";
    static TWO_BY_TWO_THEN_OUTPUT: &str = "++[>++<-],>[<.>-]";
    static HELLO_WORLD: &str = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";

    // This test is added to ensure that padding is not the identity operator on *all* tests,
    // that at least *one* call to `pad` mutates both the input and the output table.
    static ODD_INPUT_OUTPUT_SIZES: &str = ",.,.,.";

    // When we simulate a program, this generates a collection of matrices that contain
    // "abstract" execution traces. When we evaluate the base transition constraints on
    // the rows (points) from the InstructionTable matrix, these should evaluate to zero.
    #[test]
    fn io_table_constraints_evaluate_to_zero_on_test() {
        let mut rng = thread_rng();

        for source_code in [
            VERY_SIMPLE_PROGRAM,
            TWO_BY_TWO_THEN_OUTPUT,
            HELLO_WORLD,
            ODD_INPUT_OUTPUT_SIZES,
        ] {
            let actual_program = brainfuck::vm::compile(source_code).unwrap();
            let input_data = vec![
                BFieldElement::new(76),
                BFieldElement::new(79),
                BFieldElement::new(76),
            ];
            let base_matrices: BaseMatrices =
                brainfuck::vm::simulate(&actual_program, &input_data).unwrap();

            let input_matrix = base_matrices.input_matrix;
            let output_matrix = base_matrices.output_matrix;

            let number_of_randomizers = 2;
            let order = 1 << 32;
            let smooth_generator = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();

            // instantiate table objects
            let mut input_table: IOTable =
                IOTable::new_input_table(input_matrix.len(), smooth_generator, order as usize);
            let mut output_table: IOTable =
                IOTable::new_output_table(output_matrix.len(), smooth_generator, order as usize);

            let input_air_constraints = input_table.base_transition_constraints();
            let output_air_constraints = output_table.base_transition_constraints();

            let mut step_count = std::cmp::max(0, input_matrix.len() as isize - 1) as usize;
            for step in 0..step_count {
                let input_row: BFieldElement = input_matrix[step].clone();
                let input_next_row: BFieldElement = input_matrix[step + 1].clone();
                let input_point: Vec<BFieldElement> = vec![input_row, input_next_row];

                // Since there are no base air constraints on either IOTables,
                // We're evaluating that the zero polynomial in a set of points
                // is zero. This is a trivial test, but it should still hold.
                for air_constraint in input_air_constraints.iter() {
                    assert!(air_constraint.evaluate(&input_point).is_zero());
                }
            }

            step_count = std::cmp::max(0, output_matrix.len() as isize - 1) as usize;
            for step in 0..step_count {
                let output_row: BFieldElement = output_matrix[step].clone();
                let output_next_row: BFieldElement = output_matrix[step + 1].clone();
                let output_point: Vec<BFieldElement> = vec![output_row, output_next_row];

                // Since there are no base air constraints on either IOTables,
                // We're evaluating that the zero polynomial in a set of points
                // is zero. This is a trivial test, but it should still hold.
                for air_constraint in output_air_constraints.iter() {
                    assert!(air_constraint.evaluate(&output_point).is_zero());
                }
            }

            // Test air constraints after padding as well
            input_table.0.matrix = input_matrix.into_iter().map(|x| vec![x]).collect();
            input_table.pad();
            output_table.0.matrix = output_matrix.into_iter().map(|x| vec![x]).collect();
            output_table.pad();

            assert!(
                input_table.0.matrix.len() == 0
                    || other::is_power_of_two(input_table.0.matrix.len()),
                "Matrix length must be power of 2 after padding"
            );
            assert!(
                output_table.0.matrix.len() == 0
                    || other::is_power_of_two(output_table.0.matrix.len()),
                "Matrix length must be power of 2 after padding"
            );

            step_count = std::cmp::max(0, input_table.0.matrix.len() as isize - 1) as usize;
            for step in 0..step_count {
                let register = input_table.0.matrix[step].clone();
                let next_register = input_table.0.matrix[step + 1].clone();
                let point: Vec<BFieldElement> = vec![register, next_register].concat();

                for air_constraint in input_air_constraints.iter() {
                    assert!(air_constraint.evaluate(&point).is_zero());
                }
            }

            step_count = std::cmp::max(0, output_table.0.matrix.len() as isize - 1) as usize;
            for step in 0..step_count {
                let register = output_table.0.matrix[step].clone();
                let next_register = output_table.0.matrix[step + 1].clone();
                let point: Vec<BFieldElement> = vec![register, next_register].concat();

                for air_constraint in output_air_constraints.iter() {
                    assert!(air_constraint.evaluate(&point).is_zero());
                }
            }

            // Test the same for the extended matrix on both input_table and output_table

            let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize] =
                XFieldElement::random_elements(EXTENSION_CHALLENGE_COUNT as usize, &mut rng)
                    .try_into()
                    .unwrap();

            input_table.extend(
                challenges,
                XFieldElement::random_elements(2, &mut rng)
                    .try_into()
                    .unwrap(),
            );

            let air_constraints = input_table.transition_constraints_ext(challenges);
            for step in 0..input_table.0.extended_matrix.len() - 1 {
                let register = input_table.0.extended_matrix[step].clone();
                let next_register = input_table.0.extended_matrix[step + 1].clone();
                let xpoint: Vec<XFieldElement> = vec![register, next_register].concat();

                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&xpoint).is_zero());
                }
            }

            output_table.extend(
                challenges,
                XFieldElement::random_elements(2, &mut rng)
                    .try_into()
                    .unwrap(),
            );

            let air_constraints = output_table.transition_constraints_ext(challenges);
            for step in 0..output_table.0.extended_matrix.len() - 1 {
                let register = output_table.0.extended_matrix[step].clone();
                let next_register = output_table.0.extended_matrix[step + 1].clone();
                let xpoint: Vec<XFieldElement> = vec![register, next_register].concat();

                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&xpoint).is_zero());
                }
            }
        }
    }
}
