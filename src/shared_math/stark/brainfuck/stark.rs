use std::fs::Permissions;

use crate::shared_math::stark::brainfuck::evaluation_argument::{
    EvaluationArgument, ProgramEvaluationArgument,
};
use crate::shared_math::stark::brainfuck::permutation_argument::PermutationArgument;
use crate::shared_math::stark::brainfuck::processor_table::{
    IOTable, InstructionTable, MemoryTable, Table, TableCollection,
};
use crate::shared_math::{
    b_field_element::BFieldElement, fri::Fri, other::is_power_of_two,
    stark::brainfuck::processor_table::ProcessorTable, traits::GetPrimitiveRootOfUnity,
    x_field_element::XFieldElement,
};

pub struct Stark {
    trace_length: usize,
    program: Vec<BFieldElement>,
    input_symbols: Vec<BFieldElement>,
    output_symbols: Vec<BFieldElement>,
    expansion_factor: usize,
    security_level: usize,
    num_colinearity_checks: usize,
    num_randomizers: usize,
    // base_tables: [BaseTable; 5],
    // permutation_arguments: [PermuationArgument; 2],
    // evaluation_arguments: [EvaluationArgument; 3],
    max_degree: usize,
    fri: Fri<XFieldElement, blake3::Hasher>,
}

impl Stark {
    pub fn new(
        trace_length: usize,
        program: Vec<BFieldElement>,
        input_symbols: Vec<BFieldElement>,
        output_symbols: Vec<BFieldElement>,
    ) -> Self {
        let log_expansion_factor = 2; // TODO: For speed
        let expansion_factor: usize = 1 << log_expansion_factor;
        let security_level = 2; // TODO: Consider increasing this
        let num_colinearity_checks = security_level / log_expansion_factor;
        assert!(
            num_colinearity_checks > 0,
            "At least one colinearity check is required"
        );
        assert!(
            is_power_of_two(expansion_factor),
            "expansion factor must be power of two."
        );
        assert!(
            expansion_factor >= 4,
            "expansion factor must be at least 4."
        );

        let num_randomizers = 1;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u128)
            .0
            .unwrap();

        // instantiate table objects
        let processor_table =
            ProcessorTable::new(trace_length, num_randomizers, smooth_generator, order);

        let instruction_table = InstructionTable::new(
            trace_length + program.len(),
            num_randomizers,
            smooth_generator,
            order,
        );

        let memory_table = MemoryTable::new(trace_length, num_randomizers, smooth_generator, order);
        let input_table = IOTable::new_input_table(input_symbols.len(), smooth_generator, order);
        let output_table = IOTable::new_output_table(input_symbols.len(), smooth_generator, order);

        let tables = TableCollection::new(
            processor_table,
            instruction_table,
            memory_table,
            input_table,
            output_table,
        );

        // instantiate permutation objects
        let processor_instruction_lhs = (
            Table::<()>::PROCESSOR_TABLE,
            ProcessorTable::INSTRUCTION_PERMUTATION,
        );
        let processor_instruction_rhs = (
            Table::<()>::INSTRUCTION_TABLE,
            InstructionTable::PERMUTATION,
        );
        let processor_instruction_permutation = PermutationArgument::new(
            &tables,
            processor_instruction_lhs,
            processor_instruction_rhs,
        );

        let processor_memory_lhs = (
            Table::<()>::PROCESSOR_TABLE,
            ProcessorTable::MEMORY_PERMUTATION,
        );
        let processor_memory_rhs = (Table::<()>::MEMORY_TABLE, MemoryTable::PERMUTATION);
        let processor_memory_permutation =
            PermutationArgument::new(&tables, processor_memory_lhs, processor_memory_rhs);

        let permutation_arguments: Vec<PermutationArgument> = vec![
            processor_instruction_permutation,
            processor_memory_permutation,
        ];

        // input_evaluation = EvaluationArgument(
        //     8, 2, [BaseFieldElement(ord(i), self.field) for i in input_symbols])
        let input_evaluation = EvaluationArgument::new(
            tables.input_table.challenge_index(),
            tables.input_table.terminal_index(),
            input_symbols,
        );

        // output_evaluation = EvaluationArgument(
        //     9, 3, [BaseFieldElement(ord(o), self.field) for o in output_symbols])
        let output_evaluation = EvaluationArgument::new(
            tables.output_table.challenge_index(),
            tables.output_table.terminal_index(),
            output_symbols,
        );

        // program_evaluation = ProgramEvaluationArgument(
        //     [0, 1, 2, 6], 4, program)
        let program_challenge_indices = vec![0, 1, 2, 6];
        let program_terminal_index = 4;
        let program_evaluation = ProgramEvaluationArgument::new(
            program_challenge_indices,
            program_terminal_index,
            program,
        );

        // TODO: Save these three:
        // self.evaluation_arguments = [
        //     input_evaluation, output_evaluation, program_evaluation]

        // TODO: compute fri domain length
        // TODO: instantiate self.fri object

        todo!()
    }
}
