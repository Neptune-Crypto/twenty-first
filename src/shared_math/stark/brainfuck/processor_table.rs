use std::convert::TryInto;

use super::stark::{EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT};
use super::table::{Table, TableMoreTrait, TableTrait};
use super::vm::{Register, INSTRUCTIONS};
use crate::shared_math::other;
use crate::shared_math::traits::{IdentityValues, PrimeField};
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial};

impl TableMoreTrait for ProcessorTableMore {
    fn new_more() -> Self {
        ProcessorTableMore {
            instruction_permutation_terminal: XFieldElement::ring_zero(),
            memory_permutation_terminal: XFieldElement::ring_zero(),
            input_evaluation_terminal: XFieldElement::ring_zero(),
            output_evaluation_terminal: XFieldElement::ring_zero(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessorTableMore {
    pub instruction_permutation_terminal: XFieldElement,
    pub memory_permutation_terminal: XFieldElement,
    pub input_evaluation_terminal: XFieldElement,
    pub output_evaluation_terminal: XFieldElement,
}

impl ProcessorTableMore {}

#[derive(Debug, Clone)]
pub struct ProcessorTable(pub Table<ProcessorTableMore>);

impl ProcessorTable {
    // named indices for base columns (=register)
    pub const CYCLE: usize = 0;
    pub const INSTRUCTION_POINTER: usize = 1;
    pub const CURRENT_INSTRUCTION: usize = 2;
    pub const NEXT_INSTRUCTION: usize = 3;
    pub const MEMORY_POINTER: usize = 4;
    pub const MEMORY_VALUE: usize = 5;
    pub const IS_ZERO: usize = 6;

    // named indices for extension columns
    pub const INSTRUCTION_PERMUTATION: usize = 7;
    pub const MEMORY_PERMUTATION: usize = 8;
    pub const INPUT_EVALUATION: usize = 9;
    pub const OUTPUT_EVALUATION: usize = 10;

    // base and extension table width
    pub const BASE_WIDTH: usize = 7;
    pub const FULL_WIDTH: usize = 11;

    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let table = Table::<ProcessorTableMore>::new(
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
        while self.0.matrix.len() != 0 && !other::is_power_of_two(self.0.matrix.len()) {
            let last = self.0.matrix.last().unwrap();
            let padding = Register {
                cycle: last[ProcessorTable::CYCLE] + BFieldElement::ring_one(),
                instruction_pointer: last[ProcessorTable::INSTRUCTION_POINTER],
                current_instruction: BFieldElement::ring_zero(),
                next_instruction: BFieldElement::ring_zero(),
                memory_pointer: last[ProcessorTable::MEMORY_POINTER],
                memory_value: last[ProcessorTable::MEMORY_VALUE],
                is_zero: last[ProcessorTable::IS_ZERO],
            };
            self.0.matrix.push(padding.into());
        }
    }

    fn transition_constraints_afo_named_variables(
        cycle: MPolynomial<BFieldElement>,
        instruction_pointer: MPolynomial<BFieldElement>,
        current_instruction: MPolynomial<BFieldElement>,
        next_instruction: MPolynomial<BFieldElement>,
        memory_pointer: MPolynomial<BFieldElement>,
        memory_value: MPolynomial<BFieldElement>,
        is_zero: MPolynomial<BFieldElement>,
        cycle_next: MPolynomial<BFieldElement>,
        instruction_pointer_next: MPolynomial<BFieldElement>,
        current_instruction_next: MPolynomial<BFieldElement>,
        next_instruction_next: MPolynomial<BFieldElement>,
        memory_pointer_next: MPolynomial<BFieldElement>,
        memory_value_next: MPolynomial<BFieldElement>,
        is_zero_next: MPolynomial<BFieldElement>,
    ) -> [MPolynomial<BFieldElement>; 6] {
        let elem = MPolynomial::<BFieldElement>::zero(14);
        let mut polynomials: [MPolynomial<BFieldElement>; 6] = [
            elem.clone(),
            elem.clone(),
            elem.clone(),
            elem.clone(),
            elem.clone(),
            elem,
        ];

        for c in INSTRUCTIONS.iter() {
            // Max degree: 3
            let instrs: [MPolynomial<BFieldElement>; 3] = Self::instruction_polynomials(
                *c,
                &cycle,
                &instruction_pointer,
                &current_instruction,
                &next_instruction,
                &memory_pointer,
                &memory_value,
                &is_zero,
                &cycle_next,
                &instruction_pointer_next,
                &current_instruction_next,
                &next_instruction_next,
                &memory_pointer_next,
                &memory_value_next,
                &is_zero_next,
            );

            // Max degree: 7
            let deselector: MPolynomial<BFieldElement> =
                Self::ifnot_instruction(*c, &current_instruction, BFieldElement::ring_one());

            for (i, instr) in instrs.iter().enumerate() {
                polynomials[i] += deselector.to_owned() * instr.to_owned();
            }
        }

        // Instruction independent polynomials
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), 14);
        polynomials[3] = cycle_next - cycle - one.clone();
        polynomials[4] = is_zero.clone() * memory_value;
        polynomials[5] = is_zero.clone() * (one - is_zero);

        polynomials
    }

    fn instruction_polynomials(
        instruction: char,
        _cycle: &MPolynomial<BFieldElement>,
        instruction_pointer: &MPolynomial<BFieldElement>,
        current_instruction: &MPolynomial<BFieldElement>,
        next_instruction: &MPolynomial<BFieldElement>,
        memory_pointer: &MPolynomial<BFieldElement>,
        memory_value: &MPolynomial<BFieldElement>,
        is_zero: &MPolynomial<BFieldElement>,
        _cycle_next: &MPolynomial<BFieldElement>,
        instruction_pointer_next: &MPolynomial<BFieldElement>,
        _current_instruction_next: &MPolynomial<BFieldElement>,
        _next_instruction_next: &MPolynomial<BFieldElement>,
        memory_pointer_next: &MPolynomial<BFieldElement>,
        memory_value_next: &MPolynomial<BFieldElement>,
        _is_zero_next: &MPolynomial<BFieldElement>,
    ) -> [MPolynomial<BFieldElement>; 3] {
        let zero = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_zero(), 14);
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), 14);
        let two = MPolynomial::<BFieldElement>::from_constant(BFieldElement::new(2), 14);
        let mut polynomials: [MPolynomial<BFieldElement>; 3] =
            [zero.clone(), zero.clone(), zero.clone()];

        // # account for padding:
        // # deactivate all polynomials if current instruction is zero
        // for i in range(len(polynomials)):
        //     polynomials[i] *= current_instruction
        match instruction {
            '[' => {
                //     if instr == '[':
                //     polynomials[ProcessorTable.cycle] = memory_value * (instruction_pointer_next - instruction_pointer - two) + \
                //         is_zero * (instruction_pointer_next - next_instruction)
                //     polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
                //     polynomials[ProcessorTable.current_instruction] = memory_value_next - memory_value
                polynomials[0] = memory_value.to_owned()
                    * (instruction_pointer_next.to_owned()
                        - instruction_pointer.to_owned()
                        - two.clone())
                    + is_zero.to_owned()
                        * (instruction_pointer_next.to_owned() - next_instruction.to_owned());
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                polynomials[2] = memory_value_next.to_owned() - memory_value.to_owned();
            }
            ']' => {
                // elif instr == ']':
                //     polynomials[ProcessorTable.cycle] = is_zero * (instruction_pointer_next - instruction_pointer - two) + \
                //         memory_value * (instruction_pointer_next - next_instruction)
                //     polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
                //     polynomials[ProcessorTable.current_instruction] = memory_value_next - memory_value
                polynomials[0] = is_zero.to_owned()
                    * (instruction_pointer_next.to_owned()
                        - instruction_pointer.to_owned()
                        - two.clone())
                    + memory_value.to_owned()
                        * (instruction_pointer_next.to_owned() - next_instruction.to_owned());
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                polynomials[2] = memory_value_next.to_owned() - memory_value.to_owned();
            }
            '<' => {
                // elif instr == '<':
                //     polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                //         instruction_pointer - one
                //     polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - \
                //         memory_pointer + one
                //     # memory value, satisfied by permutation argument
                //     polynomials[ProcessorTable.current_instruction] = zero
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] =
                    memory_pointer_next.to_owned() - memory_pointer.to_owned() + one.clone();
                // Memory value constraint cannot be calculated in processor table for this command. So we don't set it.
                polynomials[2] = zero.clone();
            }
            '>' => {
                // elif instr == '>':
                //     polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                //         instruction_pointer - one
                //     polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - \
                //         memory_pointer - one
                //     # memory value, satisfied by permutation argument
                //     polynomials[ProcessorTable.current_instruction] = zero
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] =
                    memory_pointer_next.to_owned() - memory_pointer.to_owned() - one.clone();
                // Memory value constraint cannot be calculated in processor table for this command. So we don't set it.
                polynomials[2] = zero.clone();
            }
            '+' => {
                // elif instr == '+':
                //     polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                //         instruction_pointer - one
                //     polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
                //     polynomials[ProcessorTable.current_instruction] = memory_value_next - \
                //         memory_value - one
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                polynomials[2] =
                    memory_value_next.to_owned() - memory_value.to_owned() - one.clone();
            }
            '-' => {
                // elif instr == '-':
                //     polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                //         instruction_pointer - one
                //     polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
                //     polynomials[ProcessorTable.current_instruction] = memory_value_next - \
                //         memory_value + one
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                polynomials[2] =
                    memory_value_next.to_owned() - memory_value.to_owned() + one.clone();
            }
            ',' => {
                // elif instr == ',':
                //     polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                //         instruction_pointer - one
                //     polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
                //     # memory value, set by evaluation argument
                //     polynomials[ProcessorTable.current_instruction] = zero

                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                // Memory value constraint cannot be calculated in processor table for this command. So we don't set it.
                polynomials[2] = zero.clone();
            }
            '.' => {
                // elif instr == '.':
                //     polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                //         instruction_pointer - one
                //     polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
                //     polynomials[ProcessorTable.current_instruction] = memory_value_next - memory_value
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                // Memory value is unchanged
                polynomials[2] = memory_value_next.to_owned() - memory_value.to_owned();
            }
            _ => {
                panic!("Unrecognized instruction. Got: {}", instruction);
            }
        }

        // Account for padding by deactivating all polynomials if the instruction is zero
        for poly in polynomials.iter_mut() {
            *poly = poly.to_owned() * current_instruction.to_owned();
        }

        polynomials
    }

    fn if_instruction<PF: PrimeField>(
        instruction: char,
        indeterminate: &MPolynomial<PF>,
        one: PF,
    ) -> MPolynomial<PF> {
        assert!(one.is_one(), "one must be one");
        // TODO: I think `FULL_WIDTH` is correct here, as this is only used on the extension table
        MPolynomial::from_constant(
            one.new_from_usize(instruction as usize),
            2 * Self::FULL_WIDTH,
        ) - indeterminate.to_owned()
    }

    fn ifnot_instruction<PF: PrimeField>(
        instruction: char,
        indeterminate: &MPolynomial<PF>,
        one: PF,
    ) -> MPolynomial<PF> {
        assert!(one.is_one(), "one must be one");
        // TODO: Should this 14 (variable count) be 22?
        let mpol_one = MPolynomial::<PF>::from_constant(one, 14);
        let mut acc = mpol_one;
        for c in INSTRUCTIONS.iter() {
            if *c != instruction {
                acc = acc
                    * (indeterminate.to_owned()
                        - MPolynomial::from_constant(one.new_from_usize(*c as usize), 14));
            }
        }

        acc
    }
}

impl TableTrait for ProcessorTable {
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
        let mut variables = MPolynomial::<BFieldElement>::variables(14, BFieldElement::ring_one());
        variables.reverse();
        let cycle = variables.pop().unwrap();
        let instruction_pointer = variables.pop().unwrap();
        let current_instruction = variables.pop().unwrap();
        let next_instruction = variables.pop().unwrap();
        let memory_pointer = variables.pop().unwrap();
        let memory_value = variables.pop().unwrap();
        let is_zero = variables.pop().unwrap();
        let cycle_next = variables.pop().unwrap();
        let instruction_pointer_next = variables.pop().unwrap();
        let current_instruction_next = variables.pop().unwrap();
        let next_instruction_next = variables.pop().unwrap();
        let memory_pointer_next = variables.pop().unwrap();
        let memory_value_next = variables.pop().unwrap();
        let is_zero_next = variables.pop().unwrap();
        assert!(
            variables.is_empty(),
            "Variables must be empty after destructuring into named variables"
        );

        Self::transition_constraints_afo_named_variables(
            cycle,
            instruction_pointer,
            current_instruction,
            next_instruction,
            memory_pointer,
            memory_value,
            is_zero,
            cycle_next,
            instruction_pointer_next,
            current_instruction_next,
            next_instruction_next,
            memory_pointer_next,
            memory_value_next,
            is_zero_next,
        )
        .into()
    }

    fn base_boundary_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }

    fn transition_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
    ) -> Vec<MPolynomial<XFieldElement>> {
        let [a, b, c, d, e, f, alpha, beta, gamma, delta, _eta]: [MPolynomial<XFieldElement>;
            EXTENSION_CHALLENGE_COUNT as usize] = challenges
            .iter()
            .map(|challenge| MPolynomial::from_constant(*challenge, 2 * Self::FULL_WIDTH))
            .collect::<Vec<MPolynomial<XFieldElement>>>()
            .try_into()
            .unwrap();
        let b_field_variables: [MPolynomial<BFieldElement>; 2 * Self::FULL_WIDTH] =
            MPolynomial::variables(2 * Self::FULL_WIDTH, BFieldElement::ring_one())
                .try_into()
                .unwrap();
        let [
            // row n+1
            b_field_cycle,
            b_field_instruction_pointer,
            b_field_current_instruction,
            b_field_next_instruction,
            b_field_memory_pointer,
            b_field_memory_value,
            b_field_is_zero,

            // row n
            _b_field_instruction_permutation,
            _b_field_memory_permutation,
            _b_field_input_evaluation,
            _b_field_output_evaluation,

            // row n+1
            b_field_cycle_next,
            b_field_instruction_pointer_next,
            b_field_current_instruction_next,
            b_field_next_instruction_next,
            b_field_memory_pointer_next,
            b_field_memory_value_next,
            b_field_is_zero_next,
            
            // row n+1
            _b_field_instruction_permutation_next,
            _b_field_memory_permutation_next,
            _b_field_input_evaluation_next,
            _b_field_output_evaluation_next] = b_field_variables;

        let b_field_polynomials = Self::transition_constraints_afo_named_variables(
            b_field_cycle,
            b_field_instruction_pointer,
            b_field_current_instruction,
            b_field_next_instruction,
            b_field_memory_pointer,
            b_field_memory_value,
            b_field_is_zero,
            b_field_cycle_next,
            b_field_instruction_pointer_next,
            b_field_current_instruction_next,
            b_field_next_instruction_next,
            b_field_memory_pointer_next,
            b_field_memory_value_next,
            b_field_is_zero_next,
        );
        assert_eq!(
            6,
            b_field_polynomials.len(),
            "processor base table is expected to have 6 transition constraint polynomials"
        );

        let x_field_variables: [MPolynomial<XFieldElement>; 2 * Self::FULL_WIDTH] =
            MPolynomial::variables(2 * Self::FULL_WIDTH, XFieldElement::ring_one())
                .try_into()
                .unwrap();
        let [
            // row n
            cycle,
            instruction_pointer,
            current_instruction,
            next_instruction,
            memory_pointer,
            memory_value,
            _is_zero,
            // row n
            instruction_permutation,
            memory_permutation,
            input_evaluation,
            output_evaluation,
            // row n+1
            _cycle_next,
            _instruction_pointer_next,
            _current_instruction_next,
            _next_instruction_next,
            _memory_pointer_next,
            memory_value_next,
            _is_zero_next,
            // row n+1
            instruction_permutation_next,
            memory_permutation_next,
            input_evaluation_next,
            output_evaluation_next] = x_field_variables;

        let mut polynomials: Vec<MPolynomial<XFieldElement>> = b_field_polynomials
            .iter()
            .map(|mpol| mpol.lift_coefficients_to_xfield())
            .collect();

        // extension AIR polynomials
        // running product for instruction permutation
        polynomials.push(
            (instruction_permutation
                * (alpha
                    - a * instruction_pointer.clone()
                    - b * current_instruction.clone()
                    - c * next_instruction.clone())
                - instruction_permutation_next.clone())
                * current_instruction.clone(),
        );

        // running product for memory permutation
        polynomials.push(
            memory_permutation * (beta - d * cycle - e * memory_pointer - f * memory_value.clone())
                - memory_permutation_next,
        );

        // running evaluation for input
        polynomials.push(
            (input_evaluation_next.clone() - input_evaluation.clone() * gamma - memory_value_next)
                * Self::ifnot_instruction(',', &current_instruction, XFieldElement::ring_one())
                * current_instruction.clone()
                + (input_evaluation_next.clone() - input_evaluation.clone())
                    * Self::if_instruction(
                        ',',
                        &current_instruction.clone(),
                        XFieldElement::ring_one(),
                    ),
        );

        // running evaluation for output
        polynomials.push(
            (output_evaluation_next.clone()
                - output_evaluation.clone() * delta
                - memory_value.clone())
                * Self::ifnot_instruction('.', &current_instruction, XFieldElement::ring_one())
                * current_instruction.clone()
                + (output_evaluation_next.clone() - output_evaluation.clone())
                    * Self::if_instruction(
                        '.',
                        &current_instruction.clone(),
                        XFieldElement::ring_one(),
                    ),
        );

        assert_eq!(
            10,
            polynomials.len(),
            "Number of transition constraints for extension table must be 10."
        );

        polynomials
    }

    fn extend(
        &mut self,
        all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
        all_initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT],
    ) {
        let [a, b, c, d, e, f, alpha, beta, gamma, delta, _eta] = all_challenges;
        let [processor_instruction_permutation_initial, processor_memory_permutation_initial] =
            all_initials;

        // Prepare for loop
        let mut instruction_permutation_running_product: XFieldElement =
            processor_instruction_permutation_initial;
        let mut memory_permutation_running_product: XFieldElement =
            processor_memory_permutation_initial;
        let mut input_evaluation_running_evaluation = XFieldElement::ring_zero();
        let mut output_evaluation_running_evaluation = XFieldElement::ring_zero();

        // Preallocate memory for the extended matrix
        let mut extended_matrix: Vec<Vec<XFieldElement>> =
            vec![Vec::with_capacity(self.full_width()); self.0.matrix.len()]; // Vec::with_capacity(self.0.matrix.len());
        for (i, row) in self.0.matrix.iter().enumerate() {
            // First, copy over existing row
            for j in 0..self.base_width() {
                extended_matrix[i].push(row[j].lift());
            }

            // 1. running product for instruction permutation
            extended_matrix[i].push(instruction_permutation_running_product);
            if !extended_matrix[i][Self::CURRENT_INSTRUCTION].is_zero() {
                instruction_permutation_running_product *= alpha
                    - a * extended_matrix[i][Self::INSTRUCTION_POINTER]
                    - b * extended_matrix[i][Self::CURRENT_INSTRUCTION]
                    - c * extended_matrix[i][Self::NEXT_INSTRUCTION];
            }

            // 2. running product for memory access
            extended_matrix[i].push(memory_permutation_running_product);
            memory_permutation_running_product *= beta
                - d * extended_matrix[i][Self::CYCLE]
                - e * extended_matrix[i][Self::MEMORY_POINTER]
                - f * extended_matrix[i][Self::MEMORY_VALUE];

            // 3. evaluation for input
            extended_matrix[i].push(input_evaluation_running_evaluation);
            if row[Self::CURRENT_INSTRUCTION] == BFieldElement::new(',' as u128) {
                input_evaluation_running_evaluation = input_evaluation_running_evaluation * gamma
                    + self.0.matrix[i + 1][Self::MEMORY_VALUE].lift();
                // the memory-value register only assumes the input value after the instruction has been performed
                // TODO: Is that a fair assumption?
            }

            // 4. evaluation for output
            extended_matrix[i].push(output_evaluation_running_evaluation);
            if row[Self::CURRENT_INSTRUCTION] == BFieldElement::new('.' as u128) {
                output_evaluation_running_evaluation = output_evaluation_running_evaluation * delta
                    + extended_matrix[i][Self::MEMORY_VALUE];
            }
        }

        self.0.extended_matrix = extended_matrix;
        self.0.extended_codewords = self
            .0
            .codewords
            .iter()
            .map(|row| row.iter().map(|elem| elem.lift()).collect())
            .collect();

        self.0.more.instruction_permutation_terminal = instruction_permutation_running_product;
        self.0.more.memory_permutation_terminal = memory_permutation_running_product;
        self.0.more.input_evaluation_terminal = input_evaluation_running_evaluation;
        self.0.more.output_evaluation_terminal = output_evaluation_running_evaluation;
    }

    fn boundary_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize],
    ) -> Vec<MPolynomial<BFieldElement>> {
        // field = challenges[0].field
        // # format: mpolynomial
        // x = MPolynomial.variables(self.full_width, field)
        let x = MPolynomial::<BFieldElement>::variables(Self::FULL_WIDTH, BFieldElement::ring_one());

        // one = MPolynomial.constant(field.one())
        // zero = MPolynomial.zero()
        let zero = MPolynomial::<BFieldElement>::zero(Self::FULL_WIDTH);
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), Self::FULL_WIDTH);

        vec![
            x[ProcessorTable::CYCLE] - zero,
            x[ProcessorTable::INSTRUCTION_POINTER] - zero,
            // x[Self::CURRENT_INSTRUCTION] - ??),
            // x[Self::NEXT_INSTRUCTION] - ??),
            x[Self::MEMORY_POINTER] - zero,
            x[Self::MEMORY_VALUE] - zero,
            x[Self::IS_ZERO] - one,
            // x[Self::INSTRUCTION_PERMUTATION] - one,
            // x[Self::MEMORY_PERMUTATION] - one,
            x[Self::INPUT_EVALUATION] - zero,
            x[Self::OUTPUT_EVALUATION] - zero,
        ]
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
mod processor_table_tests {
    use rand::thread_rng;

    use super::*;
    use crate::shared_math::{
        stark::brainfuck::{self, vm::BaseMatrices},
        traits::{GetPrimitiveRootOfUnity, GetRandomElements, IdentityValues},
    };
    use crate::shared_math::stark::brainfuck::vm::sample_programs;

    #[test]
    fn processor_table_constraints_evaluate_to_zero_test() {
        let mut rng = thread_rng();

        for source_code in sample_programs::get_all_sample_programs().iter() {
            let actual_program = brainfuck::vm::compile(source_code).unwrap();
            let input_data = vec![
                BFieldElement::new(97),
                BFieldElement::new(98),
                BFieldElement::new(100),
            ];
            let base_matrices: BaseMatrices =
                brainfuck::vm::simulate(&actual_program, &input_data).unwrap();
            let processor_matrix = base_matrices.processor_matrix;
            let number_of_randomizers = 2;
            let order = 1 << 32;
            let smooth_generator = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();

            // instantiate table objects
            let mut processor_table: ProcessorTable = ProcessorTable::new(
                processor_matrix.len(),
                number_of_randomizers,
                smooth_generator,
                order as usize,
            );

            let air_constraints = processor_table.base_transition_constraints();
            for step in 0..processor_matrix.len() - 1 {
                let register: Vec<BFieldElement> = processor_matrix[step].clone().into();
                let next_register: Vec<BFieldElement> = processor_matrix[step + 1].clone().into();
                let point: Vec<BFieldElement> = vec![register, next_register].concat();

                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&point).is_zero());
                }
            }

            // Test air constraints after padding as well
            processor_table.0.matrix = processor_matrix.into_iter().map(|x| x.into()).collect();
            processor_table.pad();

            assert!(
                other::is_power_of_two(processor_table.0.matrix.len()),
                "Matrix length must be power of 2 after padding"
            );

            let air_constraints = processor_table.base_transition_constraints();
            for step in 0..processor_table.0.matrix.len() - 1 {
                let register = processor_table.0.matrix[step].clone();
                let next_register = processor_table.0.matrix[step + 1].clone();
                let point: Vec<BFieldElement> = vec![register, next_register].concat();

                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&point).is_zero());
                }
            }

            // Test the same for the extended matrix
            let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT as usize] =
                XFieldElement::random_elements(EXTENSION_CHALLENGE_COUNT as usize, &mut rng)
                    .try_into()
                    .unwrap();
            processor_table.extend(
                challenges,
                XFieldElement::random_elements(2, &mut rng)
                    .try_into()
                    .unwrap(),
            );
            let x_air_constraints = processor_table.transition_constraints_ext(challenges);
            for step in 0..processor_table.0.matrix.len() - 1 {
                let row = processor_table.0.extended_matrix[step].clone();
                let next_register = processor_table.0.extended_matrix[step + 1].clone();
                let xpoint: Vec<XFieldElement> = vec![row.clone(), next_register.clone()].concat();

                for x_air_constraint in x_air_constraints.iter() {
                    assert!(x_air_constraint.evaluate(&xpoint).is_zero());
                }

                // TODO: Can we add a negative test here?
            }
        }
    }
}
