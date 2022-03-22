use console::Term;
use std::collections::HashMap;

use crate::shared_math::{b_field_element::BFieldElement, traits::IdentityValues};

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

#[cfg(test)]
mod stark_bf_tests {
    use super::*;

    #[test]
    fn runtime_test_simple() {
        let program = "++++";
        let actual = compile(program);
        let (trace_length, inputs, outputs) = run(actual.unwrap(), vec![]).unwrap();
        assert!(inputs.is_empty());
        assert!(outputs.is_empty());
        assert_eq!(5, trace_length);
    }

    #[test]
    fn compile_two_by_two_test() {
        let program = "++[>++<-],>[<.>-]";
        let actual = compile(program);
        assert!(actual.is_some());
        assert_eq!(program.len() + 4, actual.unwrap().len());
    }

    #[test]
    fn run_two_by_two_test() {
        let program = "++[>++<-],>[<.>-]";
        let actual = compile(program);
        let (_trace_length, inputs, outputs) =
            run(actual.unwrap(), vec![BFieldElement::new(97)]).unwrap();
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
}
