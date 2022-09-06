use num_traits::{One, Zero};

use super::stark::{EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT};
use super::table::{Table, TableMoreTrait, TableTrait};
use super::vm::{Register, INSTRUCTIONS};
use crate::shared_math::b_field_element as bfe;
use crate::shared_math::other;
use crate::shared_math::stark::brainfuck::vm::instruction_zerofier;
use crate::shared_math::traits::FiniteField;
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial};
use std::convert::TryInto;

impl TableMoreTrait for ProcessorTableMore {
    fn new_more() -> Self {
        ProcessorTableMore {
            instruction_permutation_terminal: XFieldElement::zero(),
            memory_permutation_terminal: XFieldElement::zero(),
            input_evaluation_terminal: XFieldElement::zero(),
            output_evaluation_terminal: XFieldElement::zero(),
            transition_constraints_ext: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessorTableMore {
    pub instruction_permutation_terminal: XFieldElement,
    pub memory_permutation_terminal: XFieldElement,
    pub input_evaluation_terminal: XFieldElement,
    pub output_evaluation_terminal: XFieldElement,
    pub transition_constraints_ext: Option<Vec<MPolynomial<XFieldElement>>>,
}

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
    pub const MEMORY_VALUE_INVERSE: usize = 6;

    // named indices for extension columns
    pub const INSTRUCTION_PERMUTATION: usize = 7;
    pub const MEMORY_PERMUTATION: usize = 8;
    pub const INPUT_EVALUATION: usize = 9;
    pub const OUTPUT_EVALUATION: usize = 10;

    // base and extension table width
    pub const BASE_WIDTH: usize = 7;
    pub const FULL_WIDTH: usize = 11;

    pub fn pad_matrix(matrix: Vec<Vec<BFieldElement>>) -> Vec<Vec<BFieldElement>> {
        let mut ret = matrix;
        while !ret.is_empty() && !other::is_power_of_two(ret.len()) {
            let last: Vec<BFieldElement> = ret.last().unwrap().to_owned();
            let padding = Self::get_padding_row(&last);
            ret.push(padding.into());
        }

        ret
    }

    pub fn get_padding_row(last: &[BFieldElement]) -> Register {
        Register {
            cycle: last[ProcessorTable::CYCLE] + BFieldElement::one(),
            instruction_pointer: last[ProcessorTable::INSTRUCTION_POINTER],
            current_instruction: BFieldElement::zero(),
            next_instruction: BFieldElement::zero(),
            memory_pointer: last[ProcessorTable::MEMORY_POINTER],
            memory_value: last[ProcessorTable::MEMORY_VALUE],
            memory_value_inverse: last[ProcessorTable::MEMORY_VALUE_INVERSE],
        }
    }

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
            "Processor table".to_string(),
        );

        Self(table)
    }

    pub fn pad(&mut self) {
        self.0.matrix = Self::pad_matrix(self.0.matrix.clone())
    }

    /// We *could* consider fixing this, I guess...
    ///
    /// This function produces six transition constraint polynomials:
    ///
    /// - 3 instruction-specific polynomials
    /// - 3 instruction-independent polynomials
    #[allow(clippy::too_many_arguments)]
    fn transition_constraints_afo_named_variables(
        cycle: MPolynomial<BFieldElement>,
        instruction_pointer: MPolynomial<BFieldElement>,
        current_instruction: MPolynomial<BFieldElement>,
        next_instruction: MPolynomial<BFieldElement>,
        memory_pointer: MPolynomial<BFieldElement>,
        memory_value: MPolynomial<BFieldElement>,
        memory_value_inverse: MPolynomial<BFieldElement>,
        cycle_next: MPolynomial<BFieldElement>,
        instruction_pointer_next: MPolynomial<BFieldElement>,
        current_instruction_next: MPolynomial<BFieldElement>,
        next_instruction_next: MPolynomial<BFieldElement>,
        memory_pointer_next: MPolynomial<BFieldElement>,
        memory_value_next: MPolynomial<BFieldElement>,
        memory_value_inverse_next: MPolynomial<BFieldElement>,
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
            // Max degree: 4
            let instrs: [MPolynomial<BFieldElement>; 3] = Self::instruction_polynomials(
                *c,
                &cycle,
                &instruction_pointer,
                &current_instruction,
                &next_instruction,
                &memory_pointer,
                &memory_value,
                &memory_value_inverse,
                &cycle_next,
                &instruction_pointer_next,
                &current_instruction_next,
                &next_instruction_next,
                &memory_pointer_next,
                &memory_value_next,
                &memory_value_inverse_next,
            );

            // Max degree: 7
            let deselector: MPolynomial<BFieldElement> =
                Self::ifnot_instruction(*c, &current_instruction, BFieldElement::one());

            for (i, instr) in instrs.iter().enumerate() {
                polynomials[i] += deselector.to_owned() * instr.to_owned();
            }
        }

        // Instruction independent polynomials

        // p(x1,...,x14) = 1
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::one(), 14);

        // p(cycle,cycle_next,...) = cycle_next - cycle - 1;
        //
        // The cycle counter increases by one.
        //
        // p(a,a+1,...) = (a+1) - a - 1 = a + 1 - a - 1 = a - a + 1 - 1 = 0
        polynomials[3] = cycle_next - cycle - one.clone();

        let memory_value_is_zero = memory_value.clone() * memory_value_inverse.clone() - one;
        polynomials[4] = memory_value * memory_value_is_zero.clone();
        polynomials[5] = memory_value_inverse * memory_value_is_zero;

        // max degree: 11
        polynomials
    }

    #[allow(clippy::too_many_arguments)]
    fn instruction_polynomials(
        instruction: char,
        _cycle: &MPolynomial<BFieldElement>,
        instruction_pointer: &MPolynomial<BFieldElement>,
        current_instruction: &MPolynomial<BFieldElement>,
        next_instruction: &MPolynomial<BFieldElement>,
        memory_pointer: &MPolynomial<BFieldElement>,
        memory_value: &MPolynomial<BFieldElement>,
        memory_value_inverse: &MPolynomial<BFieldElement>,
        _cycle_next: &MPolynomial<BFieldElement>,
        instruction_pointer_next: &MPolynomial<BFieldElement>,
        _current_instruction_next: &MPolynomial<BFieldElement>,
        _next_instruction_next: &MPolynomial<BFieldElement>,
        memory_pointer_next: &MPolynomial<BFieldElement>,
        memory_value_next: &MPolynomial<BFieldElement>,
        _memory_value_inverse_next: &MPolynomial<BFieldElement>,
    ) -> [MPolynomial<BFieldElement>; 3] {
        let zero = MPolynomial::<BFieldElement>::from_constant(BFieldElement::zero(), 14);
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::one(), 14);
        let two = MPolynomial::<BFieldElement>::from_constant(BFieldElement::new(2), 14);
        let mut polynomials: [MPolynomial<BFieldElement>; 3] =
            [zero.clone(), zero.clone(), zero.clone()];

        let memory_value_is_zero =
            memory_value.clone() * memory_value_inverse.clone() - one.clone();
        match instruction {
            '[' => {
                polynomials[0] = memory_value.to_owned()
                    * (instruction_pointer_next.to_owned() - instruction_pointer.to_owned() - two)
                    + memory_value_is_zero
                        * (instruction_pointer_next.to_owned() - next_instruction.to_owned());
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                polynomials[2] = memory_value_next.to_owned() - memory_value.to_owned();
            }
            ']' => {
                polynomials[0] = memory_value_is_zero
                    * (instruction_pointer_next.to_owned() - instruction_pointer.to_owned() - two)
                    + memory_value.to_owned()
                        * (instruction_pointer_next.to_owned() - next_instruction.to_owned());
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                polynomials[2] = memory_value_next.to_owned() - memory_value.to_owned();
            }
            '<' => {
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned() + one;
                // Memory value constraint cannot be calculated in processor table for this command. So we don't set it.
                polynomials[2] = zero;
            }
            '>' => {
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned() - one;
                // Memory value constraint cannot be calculated in processor table for this command. So we don't set it.
                polynomials[2] = zero;
            }
            '+' => {
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                polynomials[2] = memory_value_next.to_owned() - memory_value.to_owned() - one;
            }
            '-' => {
                polynomials[0] = instruction_pointer_next.to_owned()
                    - instruction_pointer.to_owned()
                    - one.clone();
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                polynomials[2] = memory_value_next.to_owned() - memory_value.to_owned() + one;
            }
            ',' => {
                polynomials[0] =
                    instruction_pointer_next.to_owned() - instruction_pointer.to_owned() - one;
                polynomials[1] = memory_pointer_next.to_owned() - memory_pointer.to_owned();
                // Memory value constraint cannot be calculated in processor table for this command. So we don't set it.
                polynomials[2] = zero;
            }
            '.' => {
                polynomials[0] =
                    instruction_pointer_next.to_owned() - instruction_pointer.to_owned() - one;
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

        // max degree: 4
        polynomials
    }

    /// Returns a multivariate polynomial that evaluates to 0 for the given instruction
    fn if_instruction<PF: FiniteField>(
        instruction: char,
        indeterminate: &MPolynomial<PF>,
        one: PF,
    ) -> MPolynomial<PF> {
        assert!(one.is_one(), "one must be one");
        MPolynomial::from_constant(
            one.new_from_usize(instruction as usize),
            2 * Self::FULL_WIDTH,
        ) - indeterminate.to_owned()
    }

    fn ifnot_instruction<PF: FiniteField>(
        instruction: char,
        indeterminate: &MPolynomial<PF>,
        one: PF,
    ) -> MPolynomial<PF> {
        assert!(one.is_one(), "one must be one");
        let mpol_one = MPolynomial::<PF>::from_constant(one, 14);
        let mut acc = mpol_one;
        for c in INSTRUCTIONS.iter() {
            if *c != instruction {
                acc *= indeterminate.to_owned()
                    - MPolynomial::from_constant(one.new_from_usize(*c as usize), 14);
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
        let mut variables = MPolynomial::<BFieldElement>::variables(14, BFieldElement::one());

        variables.reverse();
        let cycle = variables.pop().unwrap();
        let instruction_pointer = variables.pop().unwrap();
        let current_instruction = variables.pop().unwrap();
        let next_instruction = variables.pop().unwrap();
        let memory_pointer = variables.pop().unwrap();
        let memory_value = variables.pop().unwrap();
        let memory_value_inverse = variables.pop().unwrap();
        let cycle_next = variables.pop().unwrap();
        let instruction_pointer_next = variables.pop().unwrap();
        let current_instruction_next = variables.pop().unwrap();
        let next_instruction_next = variables.pop().unwrap();
        let memory_pointer_next = variables.pop().unwrap();
        let memory_value_next = variables.pop().unwrap();
        let memory_value_inverse_next = variables.pop().unwrap();
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
            memory_value_inverse,
            cycle_next,
            instruction_pointer_next,
            current_instruction_next,
            next_instruction_next,
            memory_pointer_next,
            memory_value_next,
            memory_value_inverse_next,
        )
        .into()
    }

    fn base_boundary_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }

    fn transition_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
    ) -> Vec<MPolynomial<XFieldElement>> {
        // Avoid having to recalculate these if they've already been calculated once
        if self.0.more.transition_constraints_ext.is_some() {
            return self
                .0
                .more
                .transition_constraints_ext
                .as_ref()
                .unwrap()
                .clone();
        }
        let [a, b, c, d, e, f, alpha, beta, gamma, delta, _eta]: [MPolynomial<XFieldElement>;
            EXTENSION_CHALLENGE_COUNT] = challenges
            .iter()
            .map(|challenge| MPolynomial::from_constant(*challenge, 2 * Self::FULL_WIDTH))
            .collect::<Vec<MPolynomial<XFieldElement>>>()
            .try_into()
            .unwrap();
        let b_field_variables: [MPolynomial<BFieldElement>; 2 * Self::FULL_WIDTH] =
            MPolynomial::variables(2 * Self::FULL_WIDTH, BFieldElement::one())
                .try_into()
                .unwrap();
        let [
        // row
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
            MPolynomial::variables(2 * Self::FULL_WIDTH, XFieldElement::one())
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
            .map(bfe::lift_coefficients_to_xfield)
            .collect();

        // extension AIR polynomials
        // running product for instruction permutation
        polynomials.push(
            (instruction_permutation.clone()
                * (alpha
                    - a * instruction_pointer
                    - b * current_instruction.clone()
                    - c * next_instruction)
                - instruction_permutation_next.clone())
                * current_instruction.clone()
                + instruction_zerofier(&current_instruction, 2 * self.full_width())
                    * (instruction_permutation - instruction_permutation_next),
        );

        // running product for memory permutation
        polynomials.push(
            (memory_permutation.clone()
                * (beta - d * cycle - e * memory_pointer - f * memory_value.clone())
                - memory_permutation_next.clone())
                * current_instruction.clone()
                + (memory_permutation - memory_permutation_next)
                    * instruction_zerofier(&current_instruction, 2 * self.full_width()),
        );

        // running evaluation for input
        polynomials.push(
            (input_evaluation_next.clone() - input_evaluation.clone() * gamma - memory_value_next)
                * Self::ifnot_instruction(',', &current_instruction, XFieldElement::one())
                * current_instruction.clone()
                + (input_evaluation_next - input_evaluation)
                    * Self::if_instruction(',', &current_instruction, XFieldElement::one()),
        );

        // running evaluation for output
        polynomials.push(
            (output_evaluation_next.clone() - output_evaluation.clone() * delta - memory_value)
                * Self::ifnot_instruction('.', &current_instruction, XFieldElement::one())
                * current_instruction.clone()
                + (output_evaluation_next - output_evaluation)
                    * Self::if_instruction('.', &current_instruction, XFieldElement::one()),
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
        all_challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
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
        let mut input_evaluation_running_evaluation = XFieldElement::zero();
        let mut output_evaluation_running_evaluation = XFieldElement::zero();

        // Preallocate memory for the extended matrix
        let mut extended_matrix: Vec<Vec<XFieldElement>> =
            vec![Vec::with_capacity(self.full_width()); self.0.matrix.len()];
        for (i, row) in self.0.matrix.iter().enumerate() {
            // First, copy over existing row
            extended_matrix[i]
                .append(&mut row[0..self.base_width()].iter().map(|x| x.lift()).collect());

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

            // If not padding
            if !extended_matrix[i][Self::CURRENT_INSTRUCTION].is_zero() {
                memory_permutation_running_product *= beta
                    - d * extended_matrix[i][Self::CYCLE]
                    - e * extended_matrix[i][Self::MEMORY_POINTER]
                    - f * extended_matrix[i][Self::MEMORY_VALUE];
            }

            // 3. evaluation for input
            extended_matrix[i].push(input_evaluation_running_evaluation);
            if row[Self::CURRENT_INSTRUCTION] == BFieldElement::new(',' as u64) {
                input_evaluation_running_evaluation = input_evaluation_running_evaluation * gamma
                    + self.0.matrix[i + 1][Self::MEMORY_VALUE].lift();
                // the memory-value register only assumes the input value after the instruction has been performed
                // TODO: Is that a fair assumption?
            }

            // 4. evaluation for output
            extended_matrix[i].push(output_evaluation_running_evaluation);
            if row[Self::CURRENT_INSTRUCTION] == BFieldElement::new('.' as u64) {
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
        // TODO: Is `challenges` really not needed here?
        _challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
    ) -> Vec<MPolynomial<XFieldElement>> {
        let x = MPolynomial::<XFieldElement>::variables(Self::FULL_WIDTH, XFieldElement::one());

        let zero = MPolynomial::<XFieldElement>::zero(Self::FULL_WIDTH);
        let cycle = x[ProcessorTable::CYCLE].clone();
        let instruction_pointer = x[ProcessorTable::INSTRUCTION_POINTER].clone();
        let memory_pointer = x[Self::MEMORY_POINTER].clone();
        let memory_value = x[Self::MEMORY_VALUE].clone();
        let memory_value_inverse = x[Self::MEMORY_VALUE_INVERSE].clone();
        let input_evaluation = x[Self::INPUT_EVALUATION].clone();
        let output_evaluation = x[Self::OUTPUT_EVALUATION].clone();

        vec![
            cycle - zero.clone(),
            instruction_pointer - zero.clone(),
            // x[Self::CURRENT_INSTRUCTION] - ??),
            // x[Self::NEXT_INSTRUCTION] - ??),
            memory_pointer - zero.clone(),
            memory_value - zero.clone(),
            memory_value_inverse - zero.clone(),
            // x[Self::INSTRUCTION_PERMUTATION] - one,
            // x[Self::MEMORY_PERMUTATION] - one,
            input_evaluation - zero.clone(),
            output_evaluation - zero,
        ]
    }

    fn terminal_constraints_ext(
        &self,
        challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT],
        terminals: [XFieldElement; super::stark::TERMINAL_COUNT],
    ) -> Vec<MPolynomial<XFieldElement>> {
        let [_a, _b, _c, d, e, f, _alpha, beta, _gamma, _delta, _eta]: [MPolynomial<XFieldElement>;
            EXTENSION_CHALLENGE_COUNT] = challenges
            .iter()
            .map(|challenge| MPolynomial::from_constant(*challenge, Self::FULL_WIDTH))
            .collect::<Vec<MPolynomial<XFieldElement>>>()
            .try_into()
            .unwrap();

        let x = MPolynomial::<XFieldElement>::variables(Self::FULL_WIDTH, XFieldElement::one());

        // FIXME: These anonymous constant offsets into `terminals` are not very clear!
        let processor_instruction_permutation_terminal =
            MPolynomial::<XFieldElement>::from_constant(terminals[0], Self::FULL_WIDTH);
        let processor_memory_permutation_terminal =
            MPolynomial::<XFieldElement>::from_constant(terminals[1], Self::FULL_WIDTH);
        let processor_input_terminal =
            MPolynomial::<XFieldElement>::from_constant(terminals[2], Self::FULL_WIDTH);
        let processor_output_terminal =
            MPolynomial::<XFieldElement>::from_constant(terminals[3], Self::FULL_WIDTH);

        let instruction_permutation = x[Self::INSTRUCTION_PERMUTATION].clone();
        let current_instruction = x[Self::CURRENT_INSTRUCTION].clone();
        let memory_permutation = x[ProcessorTable::MEMORY_PERMUTATION].clone();
        let cycle = x[ProcessorTable::CYCLE].clone();
        let memory_pointer = x[ProcessorTable::MEMORY_POINTER].clone();
        let memory_value = x[ProcessorTable::MEMORY_VALUE].clone();

        let input_evaluation = x[ProcessorTable::INPUT_EVALUATION].clone();
        let output_evaluation = x[ProcessorTable::OUTPUT_EVALUATION].clone();

        vec![
            processor_instruction_permutation_terminal - instruction_permutation,
            (processor_memory_permutation_terminal.clone()
                - memory_permutation.clone()
                    * (beta - d * cycle - e * memory_pointer - f * memory_value))
                * current_instruction.clone()
                + (processor_memory_permutation_terminal - memory_permutation)
                    * instruction_zerofier(&current_instruction, self.full_width()),
            processor_input_terminal - input_evaluation,
            processor_output_terminal - output_evaluation,
        ]
    }
}

#[cfg(test)]
mod processor_table_tests {
    use rand::thread_rng;

    use super::*;
    use crate::shared_math::stark::brainfuck::vm::sample_programs;
    use crate::shared_math::{
        stark::brainfuck::{self, vm::BaseMatrices},
        traits::{GetPrimitiveRootOfUnity, GetRandomElements},
    };

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
            let smooth_generator = BFieldElement::zero()
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
            let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT] =
                XFieldElement::random_elements(EXTENSION_CHALLENGE_COUNT, &mut rng)
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
