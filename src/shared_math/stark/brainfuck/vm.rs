use console::Term;
use std::collections::HashMap;

use crate::shared_math::{b_field_element::BFieldElement, traits::IdentityValues};

pub const INSTRUCTIONS: [char; 8] = ['[', ']', '<', '>', '+', '-', ',', '.'];

#[derive(Debug, Clone)]
pub struct Register {
    pub cycle: BFieldElement,
    pub instruction_pointer: BFieldElement,
    pub current_instruction: BFieldElement,
    pub next_instruction: BFieldElement,
    pub memory_pointer: BFieldElement,
    pub memory_value: BFieldElement,
    pub is_zero: BFieldElement,
}

impl Register {
    pub fn default() -> Self {
        Self {
            cycle: BFieldElement::ring_zero(),
            instruction_pointer: BFieldElement::ring_zero(),
            current_instruction: BFieldElement::ring_zero(),
            next_instruction: BFieldElement::ring_zero(),
            memory_pointer: BFieldElement::ring_zero(),
            memory_value: BFieldElement::ring_zero(),
            is_zero: BFieldElement::ring_zero(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BaseMatrices {
    pub processor_matrix: Vec<Register>,
    pub instruction_matrix: Vec<InstructionMatrixBaseRow>,
    pub input_matrix: Vec<BFieldElement>,
    pub output_matrix: Vec<BFieldElement>,
}

impl BaseMatrices {
    pub fn default() -> Self {
        Self {
            processor_matrix: vec![],
            instruction_matrix: vec![],
            input_matrix: vec![],
            output_matrix: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstructionMatrixBaseRow {
    pub instruction_pointer: BFieldElement,
    pub current_instruction: BFieldElement,
    pub next_instruction: BFieldElement,
}

pub fn compile(source_code: &str) -> Option<Vec<BFieldElement>> {
    let mut program: Vec<BFieldElement> = vec![];
    let mut stack: Vec<usize> = vec![];

    for symbol in source_code.chars() {
        program.push(BFieldElement::new(symbol as u128));
        if symbol == '[' {
            program.push(BFieldElement::ring_zero());
            stack.push(program.len() - 1);
        }
        if symbol == ']' {
            let bracket_index: usize = stack.pop()?;
            program.push(BFieldElement::new((bracket_index + 1) as u128));
            program[bracket_index] = BFieldElement::new(program.len() as u128);
        }
    }

    if stack.is_empty() {
        Some(program)
    } else {
        None
    }
}

/// Run program, returns (trace_length, input_data, output_data)
pub fn run(
    program: Vec<BFieldElement>,
    input_data: Vec<BFieldElement>,
) -> Option<(usize, Vec<BFieldElement>, Vec<BFieldElement>)> {
    let mut instruction_pointer: usize = 0;
    let mut memory_pointer: BFieldElement = BFieldElement::ring_zero();
    let mut memory: HashMap<BFieldElement, BFieldElement> = HashMap::new();
    let mut output_data: Vec<BFieldElement> = vec![];
    let mut input_counter: usize = 0;
    let mut input_data_mut = input_data;
    let zero = BFieldElement::ring_zero();
    let one = BFieldElement::ring_one();
    let term = Term::stdout();

    let mut trace_length = 1;
    while instruction_pointer < program.len() {
        let instruction = program[instruction_pointer];
        if instruction == BFieldElement::new('[' as u128) {
            if memory.get(&memory_pointer)?.is_zero() {
                instruction_pointer = program[instruction_pointer + 1].value() as usize;
            } else {
                instruction_pointer += 2;
            }
        } else if instruction == BFieldElement::new(']' as u128) {
            if *memory.get(&memory_pointer).unwrap_or(&zero) != zero {
                instruction_pointer = program[instruction_pointer + 1].value() as usize;
            } else {
                instruction_pointer += 2;
            }
        } else if program[instruction_pointer] == BFieldElement::new('<' as u128) {
            instruction_pointer += 1;
            memory_pointer.decrement();
        } else if program[instruction_pointer] == BFieldElement::new('>' as u128) {
            instruction_pointer += 1;
            memory_pointer.increment();
        } else if program[instruction_pointer] == BFieldElement::new('+' as u128) {
            instruction_pointer += 1;
            memory.insert(
                memory_pointer,
                *memory.get(&memory_pointer).unwrap_or(&zero) + one,
            );
        } else if program[instruction_pointer] == BFieldElement::new('-' as u128) {
            instruction_pointer += 1;
            memory.insert(
                memory_pointer,
                *memory.get(&memory_pointer).unwrap_or(&zero) - one,
            );
        } else if program[instruction_pointer] == BFieldElement::new('.' as u128) {
            instruction_pointer += 1;
            output_data.push(*memory.get(&memory_pointer).unwrap_or(&zero));
        } else if program[instruction_pointer] == BFieldElement::new(',' as u128) {
            instruction_pointer += 1;
            let char: BFieldElement;
            if input_counter < input_data_mut.len() {
                char = input_data_mut[input_counter];
            } else {
                char = BFieldElement::new(term.read_char().unwrap() as u128);
                input_data_mut.push(char);
            }
            input_counter += 1;
            memory.insert(memory_pointer, char);
        } else {
            panic!("Unknown instruction");
        }

        trace_length += 1;
    }

    Some((trace_length, input_data_mut, output_data))
}

pub fn simulate(program: &[BFieldElement], input_data: &[BFieldElement]) -> Option<BaseMatrices> {
    let zero = BFieldElement::ring_zero();
    let one = BFieldElement::ring_one();
    let two = BFieldElement::new(2);
    let mut register = Register::default();
    register.current_instruction = program[0];
    if program.len() < 2 {
        register.next_instruction = zero;
    } else {
        register.next_instruction = program[1];
    }

    let mut memory: HashMap<BFieldElement, BFieldElement> = HashMap::new();
    let mut input_counter: usize = 0;

    // Prepare tables. For '++[>++<-]' this would give:
    // 0 + +
    // 1 + [
    // 2 [ >
    // 3 > +
    // ...
    let mut base_matrices = BaseMatrices::default();
    for i in 0..program.len() - 1 {
        base_matrices
            .instruction_matrix
            .push(InstructionMatrixBaseRow {
                instruction_pointer: BFieldElement::new(i as u128),
                current_instruction: program[i],
                next_instruction: program[i + 1],
            });
    }
    base_matrices
        .instruction_matrix
        .push(InstructionMatrixBaseRow {
            instruction_pointer: BFieldElement::new((program.len() - 1) as u128),
            current_instruction: *program.last().unwrap(),
            next_instruction: zero,
        });

    // base_matrices.input_matrix.append(&mut input_data.clone());

    // main loop
    while (register.instruction_pointer.value() as usize) < program.len() {
        // collect values to add new rows in execution matrices
        base_matrices.processor_matrix.push(register.clone());
        base_matrices
            .instruction_matrix
            .push(InstructionMatrixBaseRow {
                instruction_pointer: register.instruction_pointer,
                current_instruction: register.current_instruction,
                next_instruction: register.next_instruction,
            });

        // update pointer registers according to instruction
        if register.current_instruction == BFieldElement::new('[' as u128) {
            if register.memory_value.is_zero() {
                register.instruction_pointer =
                    program[register.instruction_pointer.value() as usize + 1];
            } else {
                register.instruction_pointer += two;
            }
        } else if register.current_instruction == BFieldElement::new(']' as u128) {
            if !register.memory_value.is_zero() {
                register.instruction_pointer =
                    program[register.instruction_pointer.value() as usize + 1];
            } else {
                register.instruction_pointer += two;
            }
        } else if register.current_instruction == BFieldElement::new('<' as u128) {
            register.instruction_pointer += one;
            register.memory_pointer -= one;
        } else if register.current_instruction == BFieldElement::new('>' as u128) {
            register.instruction_pointer += one;
            register.memory_pointer += one;
        } else if register.current_instruction == BFieldElement::new('+' as u128) {
            register.instruction_pointer += one;
            memory.insert(
                register.memory_pointer,
                *memory.get(&register.memory_pointer).unwrap_or(&zero) + one,
            );
        } else if register.current_instruction == BFieldElement::new('-' as u128) {
            register.instruction_pointer += one;
            memory.insert(
                register.memory_pointer,
                *memory.get(&register.memory_pointer).unwrap_or(&zero) - one,
            );
        } else if register.current_instruction == BFieldElement::new('.' as u128) {
            register.instruction_pointer += one;
            base_matrices
                .output_matrix
                .push(*memory.get(&register.memory_pointer).unwrap_or(&zero));
        } else if register.current_instruction == BFieldElement::new(',' as u128) {
            register.instruction_pointer += one;
            let input_char = input_data[input_counter];
            input_counter += 1;
            memory.insert(register.memory_pointer, input_char);
            base_matrices.input_matrix.push(input_char);
        } else {
            return None;
        }

        // update non-pointer registers
        register.cycle += one;

        if (register.instruction_pointer.value() as usize) < program.len() {
            register.current_instruction = program[register.instruction_pointer.value() as usize];
        } else {
            register.current_instruction = zero;
        }

        if (register.instruction_pointer.value() as usize) < program.len() - 1 {
            register.next_instruction =
                program[(register.instruction_pointer.value() as usize) + 1];
        } else {
            register.next_instruction = zero;
        }

        register.memory_value = *memory.get(&register.memory_pointer).unwrap_or(&zero);
        register.is_zero = if register.memory_value.is_zero() {
            one
        } else {
            zero
        };
    }

    base_matrices.processor_matrix.push(register.clone());
    base_matrices
        .instruction_matrix
        .push(InstructionMatrixBaseRow {
            instruction_pointer: register.instruction_pointer,
            current_instruction: register.current_instruction,
            next_instruction: register.next_instruction,
        });

    // post-process context tables
    // sort by instruction address
    base_matrices
        .instruction_matrix
        .sort_by_key(|row| row.instruction_pointer.value());

    Some(base_matrices)
}

#[cfg(test)]
mod stark_bf_tests {
    use super::*;

    static VERY_SIMPLE_PROGRAM: &str = "++++";
    static TWO_BY_TWO_THEN_OUTPUT: &str = "++[>++<-],>[<.>-]";

    #[test]
    fn runtime_test_simple() {
        let actual = compile(VERY_SIMPLE_PROGRAM);
        let (trace_length, inputs, outputs) = run(actual.unwrap(), vec![]).unwrap();
        assert!(inputs.is_empty());
        assert!(outputs.is_empty());
        assert_eq!(5, trace_length);
    }

    #[test]
    fn compile_two_by_two_test() {
        let actual_program = compile(TWO_BY_TWO_THEN_OUTPUT);
        assert!(actual_program.is_some());
        assert_eq!(
            TWO_BY_TWO_THEN_OUTPUT.len() + 4,
            actual_program.unwrap().len()
        );
    }

    #[test]
    fn run_two_by_two_test() {
        let actual_program = compile(TWO_BY_TWO_THEN_OUTPUT);
        let (_trace_length, inputs, outputs) =
            run(actual_program.unwrap(), vec![BFieldElement::new(97)]).unwrap();
        assert_eq!(
            vec![
                BFieldElement::new(97),
                BFieldElement::new(97),
                BFieldElement::new(97),
                BFieldElement::new(97)
            ],
            outputs
        );
        assert_eq!(vec![BFieldElement::new(97),], inputs);
    }

    #[test]
    fn simulate_two_by_two_test() {
        let actual_program = compile(TWO_BY_TWO_THEN_OUTPUT).unwrap();
        let input_data = vec![BFieldElement::new(97)];
        let base_matrices: BaseMatrices = simulate(&actual_program, &input_data).unwrap();
        let (trace_length, input_data, output_data) =
            run(actual_program, vec![BFieldElement::new(97)]).unwrap();
        assert_eq!(trace_length, base_matrices.processor_matrix.len(), "Number of rows in processor matrix from simulate must match trace length returned from 'run'.");
        assert_eq!(
            input_data.len(),
            base_matrices.input_matrix.len(),
            "Number of rows in input matrix must match length of input data"
        );
        assert_eq!(
            output_data.len(),
            base_matrices.output_matrix.len(),
            "Number of rows in output matrix must match length of output data"
        );
        assert_eq!(
            vec![
                BFieldElement::new(97),
                BFieldElement::new(97),
                BFieldElement::new(97),
                BFieldElement::new(97)
            ],
            base_matrices.output_matrix,
            "Output matrix must match output data"
        );
        assert_eq!(
            vec![BFieldElement::new(97),],
            base_matrices.input_matrix,
            "Input matrix must match input data"
        );
    }
}
