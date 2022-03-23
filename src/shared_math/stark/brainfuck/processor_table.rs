use crate::shared_math::{
    b_field_element::BFieldElement, mpolynomial::MPolynomial, other, x_field_element::XFieldElement,
};

use super::vm::INSTRUCTIONS;

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
    pub const PROCESSOR_TABLE: usize = 0;
    pub const INSTRUCTION_TABLE: usize = 1;
    pub const MEMORY_TABLE: usize = 2;

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
}

impl ProcessorTableMore {
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
    ) {
        for c in INSTRUCTIONS.iter() {
            // Max degree: 3
            let instr = Self::instruction_polynomials(
                c,
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
            );

            // Max degree: 7
            let deselector = Self::if_not_instruction(c, current_instruction);
        }
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
        let cycle = variables.pop();
        let instruction_pointer = variables.pop();
        let current_instruction = variables.pop();
        let next_instruction = variables.pop();
        let memory_pointer = variables.pop();
        let memory_value = variables.pop();
        let is_zero = variables.pop();
        let cycle_next = variables.pop();
        let instruction_pointer_next = variables.pop();
        let current_instruction_next = variables.pop();
        let next_instruction_next = variables.pop();
        let memory_pointer_next = variables.pop();
        let memory_value_next = variables.pop();
        let is_zero_next = variables.pop();
        assert!(
            variables.is_empty(),
            "Variables must be empty after destructuring into named variables"
        );

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

    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let base_width = 7;
        let full_width = 11;

        let table = Table::<ProcessorTableMore>::new(
            base_width,
            full_width,
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
        todo!()
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

    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let base_width = 3;
        let full_width = 5;

        let table = Table::<InstructionTableMore>::new(
            base_width,
            full_width,
            length,
            num_randomizers,
            generator,
            order,
        );

        Self(table)
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
}

impl MemoryTable {
    // named indices for base columns
    pub const CYCLE: usize = 0;
    pub const MEMORY_POINTER: usize = 1;
    pub const MEMORY_VALUE: usize = 2;

    // named indices for extension columns
    pub const PERMUTATION: usize = 3;

    pub fn new(
        length: usize,
        num_randomizers: usize,
        generator: BFieldElement,
        order: usize,
    ) -> Self {
        let base_width = 3;
        let full_width = 4;

        let table = Table::<MemoryTableMore>::new(
            base_width,
            full_width,
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
}

impl IOTable {
    // TODO: Refactor to avoid duplicate code.
    pub fn new_input_table(length: usize, generator: BFieldElement, order: usize) -> Self {
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
