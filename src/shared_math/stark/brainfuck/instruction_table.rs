use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial, other};

use super::{
    table::{Table, TableMoreTrait, TableTrait},
    vm::InstructionMatrixBaseRow,
};

#[derive(Debug, Clone)]
pub struct InstructionTable(pub Table<InstructionTableMore>);

#[derive(Debug, Clone)]
pub struct InstructionTableMore(());

impl TableMoreTrait for InstructionTableMore {
    fn new_more() -> Self {
        InstructionTableMore(())
    }
}

impl InstructionTable {
    // named indices for base columns
    pub const ADDRESS: usize = 0;
    pub const CURRENT_INSTRUCTION: usize = 1;
    pub const NEXT_INSTRUCTION: usize = 2;

    // named indices for extension columns
    pub const PERMUTATION: usize = 3;
    pub const EVALUATION: usize = 4;

    // base and extension table width
    pub const BASE_WIDTH: usize = 3;
    pub const FULL_WIDTH: usize = 5;

    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let table = Table::<InstructionTableMore>::new(
            Self::BASE_WIDTH,
            Self::FULL_WIDTH,
            length,
            num_randomizers,
            generator,
            order,
        );

        Self(table)
    }

    pub fn pad(&mut self) {
        while !other::is_power_of_two(self.0.matrix.len()) {
            let last = self.0.matrix.last().unwrap();
            let padding = InstructionMatrixBaseRow {
                instruction_pointer: last[Self::ADDRESS],
                current_instruction: BFieldElement::ring_zero(),
                next_instruction: BFieldElement::ring_zero(),
            };
            self.0.matrix.push(padding.into());
        }
    }

    fn transition_constraints_afo_named_variables(
        address: MPolynomial<BFieldElement>,
        current_instruction: MPolynomial<BFieldElement>,
        next_instruction: MPolynomial<BFieldElement>,
        address_next: MPolynomial<BFieldElement>,
        current_instruction_next: MPolynomial<BFieldElement>,
        next_instruction_next: MPolynomial<BFieldElement>,
    ) -> Vec<MPolynomial<BFieldElement>> {
        let mut polynomials: Vec<MPolynomial<BFieldElement>> = vec![];

        let variable_count = Self::BASE_WIDTH * 2;
        let one =
            MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), variable_count);

        // instruction pointer increases by 0 or 1
        polynomials.push(
            (address_next.clone() - address.clone() - one.clone())
                * (address_next.clone() - address.clone()),
        );

        // if address is the same, then current instruction is also
        polynomials.push(
            (address_next.clone() - address.clone() - one.clone())
                * (current_instruction_next - current_instruction),
        );

        // if address is the same, then next instruction is also
        polynomials
            .push((address_next - address - one) * (next_instruction_next - next_instruction));

        polynomials
    }
}

impl TableTrait for InstructionTable {
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
        let variable_count = Self::BASE_WIDTH * 2;
        let vars =
            MPolynomial::<BFieldElement>::variables(variable_count, BFieldElement::ring_one());

        let address = vars[0].clone();
        let current_instruction = vars[1].clone();
        let next_instruction = vars[2].clone();
        let address_next = vars[3].clone();
        let current_instruction_next = vars[4].clone();
        let next_instruction_next = vars[5].clone();

        InstructionTable::transition_constraints_afo_named_variables(
            address,
            current_instruction,
            next_instruction,
            address_next,
            current_instruction_next,
            next_instruction_next,
        )
    }

    // def base_boundary_constraints(self):
    //     # format: mpolynomial
    //     x = MPolynomial.variables(self.width, self.field)
    //     zero = MPolynomial.zero()
    //     return [x[InstructionTable.address]-zero]
    fn base_boundary_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        let x = MPolynomial::<BFieldElement>::variables(
            InstructionTable::FULL_WIDTH,
            BFieldElement::ring_one(),
        );

        // Why create 'x' and then throw all but 'address' away?
        let address = x[InstructionTable::ADDRESS].clone();
        let zero = MPolynomial::<BFieldElement>::zero(InstructionTable::FULL_WIDTH);

        vec![address - zero]
    }
}

#[cfg(test)]
mod instruction_table_tests {
    use super::*;
    use crate::shared_math::stark::brainfuck::vm::InstructionMatrixBaseRow;
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
    fn instruction_base_table_evaluate_to_zero_on_execution_trace_test() {
        for source_code in [VERY_SIMPLE_PROGRAM, TWO_BY_TWO_THEN_OUTPUT, HELLO_WORLD] {
            let actual_program = brainfuck::vm::compile(source_code).unwrap();
            let input_data = vec![BFieldElement::new(97)];
            let base_matrices: BaseMatrices =
                brainfuck::vm::simulate(&actual_program, &input_data).unwrap();

            let instruction_matrix = base_matrices.instruction_matrix;

            let number_of_randomizers = 2;
            let order = 1 << 32;
            let smooth_generator = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();

            // instantiate table objects
            let instruction_table: InstructionTable = InstructionTable::new(
                instruction_matrix.len(),
                number_of_randomizers,
                smooth_generator,
                order as usize,
            );

            let air_constraints = instruction_table.base_transition_constraints();

            for step in 0..instruction_matrix.len() - 1 {
                let row: InstructionMatrixBaseRow = instruction_matrix[step].clone();
                let next_row: InstructionMatrixBaseRow = instruction_matrix[step + 1].clone();

                let point: Vec<BFieldElement> = vec![
                    row.instruction_pointer,
                    row.current_instruction,
                    row.next_instruction,
                    next_row.instruction_pointer,
                    next_row.current_instruction,
                    next_row.next_instruction,
                ];

                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&point).is_zero());
                }
            }
        }
    }
}
