use crate::shared_math::b_field_element as bfe;
use crate::shared_math::stark::brainfuck::stark::TERMINAL_COUNT;
use crate::shared_math::traits::IdentityValues;
use crate::shared_math::{
    b_field_element::BFieldElement, mpolynomial::MPolynomial, other, x_field_element::XFieldElement,
};
use std::convert::TryInto;

use super::{
    stark::{EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT},
    table::{Table, TableMoreTrait, TableTrait},
    vm::InstructionMatrixBaseRow,
};

#[derive(Debug, Clone)]
pub struct InstructionTable(pub Table<InstructionTableMore>);

#[derive(Debug, Clone)]
pub struct InstructionTableMore {
    pub permutation_terminal: XFieldElement,
    pub evaluation_terminal: XFieldElement,
}

impl TableMoreTrait for InstructionTableMore {
    fn new_more() -> Self {
        InstructionTableMore {
            permutation_terminal: XFieldElement::ring_zero(),
            evaluation_terminal: XFieldElement::ring_zero(),
        }
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
            "Instruction table".to_string(),
        );

        Self(table)
    }

    pub fn pad(&mut self) {
        while !self.0.matrix.is_empty() && !other::is_power_of_two(self.0.matrix.len()) {
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
        // (i.e., program memory is read-only)
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

    fn name(&self) -> &str {
        &self.0.name
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

    fn extend(
        &mut self,
        all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
        all_initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT],
    ) {
        let [a, b, c, _d, _e, _f, alpha, _beta, _gamma, _delta, eta] = all_challenges;
        let [processor_instruction_permutation_initial, _processor_memory_permutation_initial] =
            all_initials;

        // Preallocate memory for the extended matrix
        let mut extended_matrix: Vec<Vec<XFieldElement>> =
            vec![Vec::with_capacity(self.full_width()); self.0.matrix.len()];

        let mut permutation_running_product: XFieldElement =
            processor_instruction_permutation_initial;

        let mut evaluation_running_sum = XFieldElement::ring_zero();

        let mut previous_address = -XFieldElement::ring_one();

        for (i, row) in self.0.matrix.iter().enumerate() {
            // first, copy over existing row
            let mut new_row: Vec<XFieldElement> = row
                .iter()
                .map(|&bfe| XFieldElement::new_const(bfe))
                .collect();

            new_row.push(permutation_running_product);
            if !new_row[InstructionTable::CURRENT_INSTRUCTION].is_zero() {
                permutation_running_product *= alpha
                    - a * new_row[InstructionTable::ADDRESS]
                    - b * new_row[InstructionTable::CURRENT_INSTRUCTION]
                    - c * new_row[InstructionTable::NEXT_INSTRUCTION];
            }

            if new_row[InstructionTable::ADDRESS] != previous_address {
                evaluation_running_sum = eta * evaluation_running_sum
                    + a * new_row[InstructionTable::ADDRESS]
                    + b * new_row[InstructionTable::CURRENT_INSTRUCTION]
                    + c * new_row[InstructionTable::NEXT_INSTRUCTION]
            }

            new_row.push(evaluation_running_sum);

            previous_address = new_row[InstructionTable::ADDRESS];

            extended_matrix[i] = new_row;
        }

        self.0.extended_matrix = extended_matrix;

        self.0.extended_codewords = self
            .0
            .codewords
            .iter()
            .map(|row| row.iter().map(|elem| elem.lift()).collect())
            .collect();

        self.0.more.permutation_terminal = permutation_running_product;

        self.0.more.evaluation_terminal = evaluation_running_sum;
    }

    fn transition_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
    ) -> Vec<MPolynomial<XFieldElement>> {
        let [a, b, c, _d, _e, _f, alpha, _beta, _gamma, _delta, eta]: [MPolynomial<XFieldElement>;
            EXTENSION_CHALLENGE_COUNT] = challenges
            .iter()
            .map(|challenge| MPolynomial::from_constant(*challenge, 2 * Self::FULL_WIDTH))
            .collect::<Vec<MPolynomial<XFieldElement>>>()
            .try_into()
            .unwrap();

        let vars = MPolynomial::<BFieldElement>::variables(
            2 * Self::FULL_WIDTH,
            BFieldElement::ring_one(),
        );

        let address = vars[0].clone();
        let current_instruction = vars[1].clone();
        let next_instruction = vars[2].clone();
        let permutation = vars[3].clone();
        let evaluation = vars[4].clone();
        let address_next = vars[5].clone();
        let current_instruction_next = vars[6].clone();
        let next_instruction_next = vars[7].clone();
        let permutation_next = vars[8].clone();
        let evaluation_next = vars[9].clone();

        let mut polynomials: Vec<MPolynomial<XFieldElement>> =
            InstructionTable::transition_constraints_afo_named_variables(
                address.clone(),
                current_instruction.clone(),
                next_instruction.clone(),
                address_next.clone(),
                current_instruction_next.clone(),
                next_instruction_next.clone(),
            )
            .iter()
            .map(bfe::lift_coefficients_to_xfield)
            .collect();

        assert_eq!(
            3,
            polynomials.len(),
            "expect to inherit 3 polynomials from ancestor"
        );

        let address_lifted = bfe::lift_coefficients_to_xfield(&address);
        let _current_instruction_lifted = bfe::lift_coefficients_to_xfield(&current_instruction);
        let next_instruction_lifted = bfe::lift_coefficients_to_xfield(&next_instruction);

        let address_next_lifted = bfe::lift_coefficients_to_xfield(&address_next);
        let current_instruction_next_lifted =
            bfe::lift_coefficients_to_xfield(&current_instruction_next);
        let next_instruction_next_lifted = bfe::lift_coefficients_to_xfield(&next_instruction_next);

        let permutation_lifted = bfe::lift_coefficients_to_xfield(&permutation);
        let permutation_next_lifted = bfe::lift_coefficients_to_xfield(&permutation_next);
        let current_instruction_lifted = bfe::lift_coefficients_to_xfield(&current_instruction);

        let evaluation_lifted = bfe::lift_coefficients_to_xfield(&evaluation);
        let evaluation_next_lifted = bfe::lift_coefficients_to_xfield(&evaluation_next);

        // TODO: Explain what this polynomial does:
        polynomials.push(
            (permutation_lifted
                * (alpha
                    - a.clone() * address_lifted.clone()
                    - b.clone() * current_instruction_lifted.clone()
                    - c.clone() * next_instruction_lifted)
                - permutation_next_lifted)
                * current_instruction_lifted,
        );

        let ifnewaddress: MPolynomial<XFieldElement> =
            address_next_lifted.clone() - address_lifted.clone();
        let ifoldaddress: MPolynomial<XFieldElement> = address_next_lifted.clone()
            - address_lifted
            - MPolynomial::from_constant(XFieldElement::ring_one(), 2 * Self::FULL_WIDTH);

        polynomials.push(
            ifnewaddress
                * (evaluation_lifted.clone() * eta
                    + a * address_next_lifted
                    + b * current_instruction_next_lifted
                    + c * next_instruction_next_lifted
                    - evaluation_next_lifted.clone())
                + ifoldaddress * (evaluation_lifted - evaluation_next_lifted),
        );

        polynomials
    }

    fn boundary_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
    ) -> Vec<MPolynomial<XFieldElement>> {
        let [a, b, c, _d, _e, _f, _alpha, _beta, _gamma, _delta, _eta]: [MPolynomial<XFieldElement>;
            EXTENSION_CHALLENGE_COUNT] = challenges
            .iter()
            .map(|challenge| MPolynomial::from_constant(*challenge, Self::FULL_WIDTH))
            .collect::<Vec<MPolynomial<XFieldElement>>>()
            .try_into()
            .unwrap();

        let x: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(self.full_width(), XFieldElement::ring_one());

        vec![
            x[Self::ADDRESS].clone(),
            x[Self::EVALUATION].clone()
                - a * x[Self::ADDRESS].clone()
                - b * x[Self::CURRENT_INSTRUCTION].clone()
                - c * x[Self::NEXT_INSTRUCTION].clone(),
        ]
    }

    fn terminal_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
        terminals: [XFieldElement; TERMINAL_COUNT],
    ) -> Vec<MPolynomial<XFieldElement>> {
        let [a, b, c, _d, _e, _f, alpha, _beta, _gamma, _delta, _eta]: [MPolynomial<XFieldElement>;
            EXTENSION_CHALLENGE_COUNT] = challenges
            .iter()
            .map(|challenge| MPolynomial::from_constant(*challenge, Self::FULL_WIDTH))
            .collect::<Vec<MPolynomial<XFieldElement>>>()
            .try_into()
            .unwrap();
        let [processor_instruction_permutation_terminal, _processor_memory_permutation_terminal, _processor_input_evaluation_terminal, _processor_output_evaluation_terminal, instruction_evaluation_terminal]: [MPolynomial<XFieldElement>;
            TERMINAL_COUNT] = terminals
            .iter()
            .map(|terminal| MPolynomial::from_constant(*terminal, Self::FULL_WIDTH))
            .collect::<Vec<MPolynomial<XFieldElement>>>()
            .try_into()
            .unwrap();

        let x: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(self.full_width(), XFieldElement::ring_one());

        vec![
            (x[Self::PERMUTATION].clone()
                * (alpha
                    - a * x[Self::ADDRESS].clone()
                    - b * x[Self::CURRENT_INSTRUCTION].clone()
                    - c * x[Self::NEXT_INSTRUCTION].clone())
                - processor_instruction_permutation_terminal)
                * x[Self::CURRENT_INSTRUCTION].clone(),
            x[Self::EVALUATION].clone() - instruction_evaluation_terminal,
        ]
    }
}

#[cfg(test)]
mod instruction_table_tests {
    use super::*;
    use crate::shared_math::stark::brainfuck::vm::sample_programs;
    use crate::shared_math::stark::brainfuck::vm::BaseMatrices;
    use crate::shared_math::stark::brainfuck::vm::InstructionMatrixBaseRow;
    use crate::shared_math::stark::brainfuck::{self};
    use crate::shared_math::traits::GetRandomElements;
    use crate::shared_math::traits::{GetPrimitiveRootOfUnity, IdentityValues};
    use rand::thread_rng;

    // When we simulate a program, this generates a collection of matrices that contain
    // "abstract" execution traces. When we evaluate the base transition constraints on
    // the rows (points) from the InstructionTable matrix, these should evaluate to zero.
    #[test]
    fn instruction_table_constraints_evaluate_to_zero_test() {
        let mut rng = thread_rng();

        for source_code in sample_programs::get_all_sample_programs().iter() {
            let actual_program = brainfuck::vm::compile(source_code).unwrap();
            let input_data = vec![
                BFieldElement::new(97),
                BFieldElement::new(98),
                BFieldElement::new(99),
            ];
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
            let mut instruction_table: InstructionTable = InstructionTable::new(
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

            // Test air constraints after padding as well
            instruction_table.0.matrix = instruction_matrix.into_iter().map(|x| x.into()).collect();
            instruction_table.pad();

            assert!(
                other::is_power_of_two(instruction_table.0.matrix.len()),
                "Matrix length must be power of 2 after padding"
            );

            let air_constraints = instruction_table.base_transition_constraints();
            for step in 0..instruction_table.0.matrix.len() - 1 {
                let register = instruction_table.0.matrix[step].clone();
                let next_register = instruction_table.0.matrix[step + 1].clone();
                let point: Vec<BFieldElement> = vec![register, next_register].concat();

                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&point).is_zero());
                }
            }

            // Test the same for the extended matrix
            let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT] =
                XFieldElement::random_elements(EXTENSION_CHALLENGE_COUNT, &mut rng)
                    .try_into()
                    .unwrap();
            instruction_table.extend(
                challenges,
                XFieldElement::random_elements(2, &mut rng)
                    .try_into()
                    .unwrap(),
            );

            let air_constraints = instruction_table.transition_constraints_ext(challenges);
            for step in 0..instruction_table.0.extended_matrix.len() - 1 {
                let register = instruction_table.0.extended_matrix[step].clone();
                let next_register = instruction_table.0.extended_matrix[step + 1].clone();
                let xpoint: Vec<XFieldElement> = vec![register, next_register].concat();

                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&xpoint).is_zero());
                }
            }
        }
    }
}
