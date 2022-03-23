use crate::shared_math::{
    b_field_element::BFieldElement, mpolynomial::MPolynomial, other, x_field_element::XFieldElement,
};

use super::vm::INSTRUCTIONS;

pub const PROCESSOR_TABLE: usize = 0;
pub const INSTRUCTION_TABLE: usize = 1;
pub const MEMORY_TABLE: usize = 2;

pub struct Table<T> {
    base_width: usize,
    full_width: usize,
    length: usize,
    num_randomizers: usize,
    height: usize,
    omicron: BFieldElement,
    generator: BFieldElement,
    order: usize,
    matrix: Vec<Vec<BFieldElement>>,
    more: T,
}

impl<T: TableMoreTrait> Table<T> {
    pub fn new(
        base_width: usize,
        full_width: usize,
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let height = other::roundup_npo2(length as u64) as usize;
        let omicron = Self::derive_omicron(generator, order, height);
        let matrix = vec![];
        let more = T::new_more();

        Self {
            base_width,
            full_width,
            length,
            num_randomizers,
            height,
            omicron,
            generator,
            order,
            matrix,
            more,
        }
    }

    fn derive_omicron(generator: BFieldElement, order: usize, height: usize) -> BFieldElement {
        todo!()
    }
}

pub trait TableMoreTrait {
    fn new_more() -> Self;
    fn base_transition_constraints() -> Vec<MPolynomial<BFieldElement>>;
    fn base_boundary_constraints() -> Vec<MPolynomial<BFieldElement>>;
}

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
        let zero = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), 14);
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), 14);
        let two = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), 14);
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
                polynomials[0] =
                    instruction_pointer_next.to_owned() - memory_pointer.to_owned() - one.clone();
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

    fn base_transition_constraints() -> Vec<MPolynomial<BFieldElement>> {
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

    fn base_boundary_constraints() -> Vec<MPolynomial<BFieldElement>> {
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
}

pub struct InstructionTable(Table<InstructionTableMore>);

pub struct InstructionTableMore(());

impl TableMoreTrait for InstructionTableMore {
    fn new_more() -> Self {
        InstructionTableMore(())
    }

    fn base_transition_constraints() -> Vec<MPolynomial<BFieldElement>> {
        let vars = MPolynomial::<BFieldElement>::variables(6, BFieldElement::ring_one());
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
    fn base_boundary_constraints() -> Vec<MPolynomial<BFieldElement>> {
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

    fn transition_constraints_afo_named_variables(
        address: MPolynomial<BFieldElement>,
        current_instruction: MPolynomial<BFieldElement>,
        next_instruction: MPolynomial<BFieldElement>,
        address_next: MPolynomial<BFieldElement>,
        current_instruction_next: MPolynomial<BFieldElement>,
        next_instruction_next: MPolynomial<BFieldElement>,
    ) -> Vec<MPolynomial<BFieldElement>> {
        let mut polynomials: Vec<MPolynomial<BFieldElement>> = vec![];
        let one = MPolynomial::<BFieldElement>::from_constant(BFieldElement::ring_one(), 14);

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

pub struct MemoryTable(Table<MemoryTableMore>);

pub struct MemoryTableMore(());

impl TableMoreTrait for MemoryTableMore {
    fn new_more() -> Self {
        MemoryTableMore(())
    }

    fn base_transition_constraints() -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }

    fn base_boundary_constraints() -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }
}

impl MemoryTable {
    // named indices for base columns
    pub const CYCLE: usize = 0;
    pub const MEMORY_POINTER: usize = 1;
    pub const MEMORY_VALUE: usize = 2;

    // named indices for extension columns
    pub const PERMUTATION: usize = 3;

    // base and extension table width
    pub const BASE_WIDTH: usize = 3;
    pub const FULL_WIDTH: usize = 4;

    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let table = Table::<MemoryTableMore>::new(
            Self::BASE_WIDTH,
            Self::FULL_WIDTH,
            length,
            num_randomizers,
            generator,
            order,
        );

        Self(table)
    }
}

pub struct IOTable(Table<IOTableMore>);

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

    fn base_transition_constraints() -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }

    fn base_boundary_constraints() -> Vec<MPolynomial<BFieldElement>> {
        todo!()
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

    pub fn challenge_index(&self) -> usize {
        self.0.more.challenge_index
    }

    pub fn terminal_index(&self) -> usize {
        self.0.more.terminal_index
    }
}

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
}
