use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial, other};

use super::table::{Table, TableMoreTrait, TableTrait};

#[derive(Debug, Clone)]
pub struct IOTable(pub Table<IOTableMore>);

#[derive(Debug, Clone)]
pub struct IOTableMore {
    pub challenge_index: usize,
    pub terminal_index: usize,
}

impl TableMoreTrait for IOTableMore {
    fn new_more() -> Self {
        IOTableMore {
            challenge_index: 0,
            terminal_index: 0,
        }
    }
}

impl IOTable {
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
        while !other::is_power_of_two(self.0.matrix.len()) {
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
}

#[cfg(test)]
mod io_table_tests {
    use super::*;
    use crate::shared_math::stark::brainfuck::vm::{InstructionMatrixBaseRow, MemoryMatrixBaseRow};
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

    // When we simulate a program, this generates a collection of matrices that contain
    // "abstract" execution traces. When we evaluate the base transition constraints on
    // the rows (points) from the InstructionTable matrix, these should evaluate to zero.
    #[test]
    fn io_base_table_evaluate_to_zero_on_execution_trace_test() {
        for source_code in [VERY_SIMPLE_PROGRAM, TWO_BY_TWO_THEN_OUTPUT, HELLO_WORLD] {
            let actual_program = brainfuck::vm::compile(source_code).unwrap();
            let input_data = vec![BFieldElement::new(97)];
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
            let input_table: IOTable =
                IOTable::new_input_table(input_matrix.len(), smooth_generator, order as usize);
            let output_table: IOTable =
                IOTable::new_output_table(output_matrix.len(), smooth_generator, order as usize);

            let input_air_constraints = input_table.base_transition_constraints();
            let output_air_constraints = output_table.base_transition_constraints();

            let step_count = std::cmp::max(0, input_matrix.len() as isize - 1) as usize;
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
        }
    }
}
