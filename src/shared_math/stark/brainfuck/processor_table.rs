use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial};

use super::{
    table::{Table, TableMoreTrait},
    vm::INSTRUCTIONS,
};

impl ProcessorTableMore {
    fn instruction_polynomials(
        instruction: char,
        cycle: &MPolynomial<BFieldElement>,
        instruction_pointer: &MPolynomial<BFieldElement>,
        current_instruction: &MPolynomial<BFieldElement>,
        next_instruction: &MPolynomial<BFieldElement>,
        memory_pointer: &MPolynomial<BFieldElement>,
        memory_value: &MPolynomial<BFieldElement>,
        is_zero: &MPolynomial<BFieldElement>,
        cycle_next: &MPolynomial<BFieldElement>,
        instruction_pointer_next: &MPolynomial<BFieldElement>,
        current_instruction_next: &MPolynomial<BFieldElement>,
        next_instruction_next: &MPolynomial<BFieldElement>,
        memory_pointer_next: &MPolynomial<BFieldElement>,
        memory_value_next: &MPolynomial<BFieldElement>,
        is_zero_next: &MPolynomial<BFieldElement>,
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

    fn ifnot_instruction(
        instruction: char,
        indeterminate: &MPolynomial<BFieldElement>,
    ) -> MPolynomial<BFieldElement> {
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), 14);
        let mut acc = one;
        for c in INSTRUCTIONS.iter() {
            if *c != instruction {
                acc = acc
                    * (indeterminate.to_owned()
                        - MPolynomial::from_constant(BFieldElement::new(*c as u128), 14));
            }
        }

        acc
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
        // TODO: Is variable count = 14 here?
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
            let deselector = Self::ifnot_instruction(*c, &current_instruction);

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
}

struct ProcessorTableMore {
    codewords: Vec<Vec<BFieldElement>>,
}

impl TableMoreTrait for ProcessorTableMore {
    fn new_more() -> Self {
        ProcessorTableMore { codewords: vec![] }
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
}

pub struct ProcessorTable(Table<ProcessorTableMore>);

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

    pub fn base_transition_constraints(&self) -> Vec<MPolynomial<BFieldElement>> {
        self.0.more.base_transition_constraints()
    }
}

#[cfg(test)]
mod rescue_prime_start_test {
    use super::*;
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

    #[test]
    fn processor_base_table_evaluate_to_zero_on_execution_trace_test() {
        for source_code in [VERY_SIMPLE_PROGRAM, TWO_BY_TWO_THEN_OUTPUT, HELLO_WORLD] {
            let actual_program = brainfuck::vm::compile(source_code).unwrap();
            let base_matrices: BaseMatrices =
                brainfuck::vm::simulate(actual_program.clone(), vec![BFieldElement::new(97)])
                    .unwrap();
            let processor_matrix = base_matrices.processor_matrix;
            let number_of_randomizers = 2;
            let order = 1 << 32;
            let smooth_generator = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();
            // instantiate table objects
            let processor_table: ProcessorTable = ProcessorTable::new(
                processor_matrix.len(),
                number_of_randomizers,
                smooth_generator,
                order as usize,
            );
            let air_constraints = processor_table.base_transition_constraints();
            for step in 0..processor_matrix.len() - 1 {
                let register: Register = processor_matrix[step].clone();
                let next_register: Register = processor_matrix[step + 1].clone();
                register.current_instruction;
                let point: Vec<BFieldElement> = vec![
                    register.cycle,
                    register.instruction_pointer,
                    register.current_instruction,
                    register.next_instruction,
                    register.memory_pointer,
                    register.memory_value,
                    register.is_zero,
                    next_register.cycle,
                    next_register.instruction_pointer,
                    next_register.current_instruction,
                    next_register.next_instruction,
                    next_register.memory_pointer,
                    next_register.memory_value,
                    next_register.is_zero,
                ];
                for air_constraint in air_constraints.iter() {
                    assert!(air_constraint.evaluate(&point).is_zero());
                }
            }
        }
    }
}
