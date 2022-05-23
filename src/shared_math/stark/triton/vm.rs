use rand::Rng;

use super::instruction::{parse, Instruction};
use super::state::{VMState, AUX_REGISTER_COUNT};
use super::stdio::{InputStream, OutputStream, VecStream};
use super::table::base_matrix::BaseMatrices;
use super::table::processor_table;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_xlix::{neptune_params, RescuePrimeXlix};
use std::error::Error;
use std::fmt::Display;
use std::io::Cursor;

type BWord = BFieldElement;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Program {
    pub instructions: Vec<Instruction>,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut stream = self.instructions.iter();
        while let Some(instruction) = stream.next() {
            writeln!(f, "{}", instruction)?;

            // Skip duplicate placeholder used for aligning instructions and instruction_pointer in VM.
            for _ in 1..instruction.size() {
                stream.next();
            }
        }
        Ok(())
    }
}

pub struct SkippyIter {
    cursor: Cursor<Vec<Instruction>>,
}

impl Iterator for SkippyIter {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.cursor.position();
        let instructions = self.cursor.get_ref();
        let instruction = instructions[pos as usize];
        self.cursor.set_position(pos + instruction.size() as u64);

        Some(instruction)
    }
}

impl IntoIterator for Program {
    type Item = Instruction;

    type IntoIter = SkippyIter;

    fn into_iter(self) -> Self::IntoIter {
        let cursor = Cursor::new(self.instructions);
        SkippyIter { cursor }
    }
}

/// A `Program` is a `Vec<Instruction>` that contains duplicate elements for
/// instructions with a size of 2. This means that the index in the vector
/// corresponds to the VM's `instruction_pointer`. These duplicate values
/// should most often be skipped/ignored, e.g. when pretty-printing.
impl Program {
    pub fn new(input: &[Instruction]) -> Self {
        let mut instructions = vec![];
        for instr in input {
            instructions.append(&mut vec![*instr; instr.size()]);
        }
        Program { instructions }
    }

    pub fn from_code(code: &str) -> Result<Self, Box<dyn Error>> {
        let instructions = parse(code)?;
        Ok(Program::new(&instructions))
    }

    pub fn simulate<R, In, Out>(
        &self,
        rng: &mut R,
        stdin: &mut In,
        stdout: &mut Out,
        rescue_prime: &RescuePrimeXlix<AUX_REGISTER_COUNT>,
    ) -> (BaseMatrices, Option<Box<dyn Error>>)
    where
        R: Rng,
        In: InputStream,
        Out: OutputStream,
    {
        let mut current_state = VMState::new(self);
        let mut base_matrices = BaseMatrices::default();

        self.initialize_instruction_matrix(&mut base_matrices);

        loop {
            let written_word = match current_state.step_mut(rng, stdin, rescue_prime) {
                Err(err) => return (base_matrices, Some(err)),
                Ok(written_word) => written_word,
            };

            match current_state.to_processor_arr() {
                Err(err) => return (base_matrices, Some(err)),
                Ok(row) => base_matrices.processor_matrix.push(row),
            }

            match current_state.to_instruction_arr() {
                Err(err) => return (base_matrices, Some(err)),
                Ok(row) => base_matrices.instruction_matrix.push(row),
            }

            if let Ok(Some(word)) = current_state.read_word() {
                base_matrices.input_matrix.push([word])
            }

            if let Some(word) = written_word {
                base_matrices.output_matrix.push([word]);
            }

            if current_state.is_final() {
                break;
            }
        }

        base_matrices.sort_instruction_matrix();

        (base_matrices, None)
    }

    pub fn run<R, In, Out>(
        &self,
        rng: &mut R,
        stdin: &mut In,
        stdout: &mut Out,
        rescue_prime: &RescuePrimeXlix<AUX_REGISTER_COUNT>,
    ) -> (Vec<VMState>, Option<Box<dyn Error>>)
    where
        R: Rng,
        In: InputStream,
        Out: OutputStream,
    {
        let mut processor_trace = vec![VMState::new(self)];
        let mut current_state = processor_trace.last().unwrap();

        while !current_state.is_final() {
            let next_state = current_state.step(rng, stdin, rescue_prime);
            let (next_state, written_word) = match next_state {
                Err(err) => return (processor_trace, Some(err)),
                Ok(result) => result,
            };

            if let Some(word) = written_word {
                let _written = stdout.write_elem(word);
            }

            processor_trace.push(next_state);
            current_state = processor_trace.last().unwrap();
        }

        (processor_trace, None)
    }

    pub fn run_with_input(&self, input: &[u8]) -> (Vec<VMState>, Vec<u8>, Option<Box<dyn Error>>) {
        let mut rng = rand::thread_rng();
        let mut stdin = VecStream::new(input);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        let (trace, err) = self.run(&mut rng, &mut stdin, &mut stdout, &rescue_prime);

        (trace, stdout.to_vec(), err)
    }

    fn initialize_instruction_matrix(&self, base_matrices: &mut BaseMatrices) {
        // Fixme:  Change `into_iter()` to `iter()`.
        let mut iter = self.clone().into_iter();
        let mut current_instruction = iter.next().unwrap();
        let mut program_index: BWord = BWord::ring_zero();

        while let Some(next_instruction) = iter.next() {
            let current_opcode: BFieldElement = current_instruction.opcode().into();

            if let Some(instruction_arg) = current_instruction.arg() {
                base_matrices.instruction_matrix.push([
                    program_index,
                    current_opcode,
                    instruction_arg,
                ]);
                program_index.increment();

                let next_opcode: BFieldElement = next_instruction.opcode().into();
                base_matrices.instruction_matrix.push([
                    program_index,
                    instruction_arg,
                    next_opcode,
                ]);
                program_index.increment();
            } else {
                let next_opcode: BFieldElement = next_instruction.opcode().into();
                base_matrices
                    .instruction_matrix
                    .push([program_index, current_opcode, next_opcode]);
                program_index.increment();
            }

            current_instruction = next_instruction;
        }
    }

    fn to_base_matrices(
        program: Program,
        trace: &[[BFieldElement; processor_table::BASE_WIDTH]],
    ) -> BaseMatrices {
        let mut base_matrices = BaseMatrices::default();

        // 1. Fill instruction_matrix with program
        let mut program_index: BWord = BWord::ring_zero();
        let mut iter = program.into_iter();
        let mut current_instruction = iter.next().unwrap();

        while let Some(next) = iter.next() {
            let current_opcode: BFieldElement = current_instruction.opcode().into();

            if let Some(instruction_arg) = current_instruction.arg() {
                base_matrices.instruction_matrix.push([
                    program_index,
                    current_opcode,
                    instruction_arg,
                ]);
                program_index.increment();

                let next_opcode: BFieldElement = next.opcode().into();
                base_matrices.instruction_matrix.push([
                    program_index,
                    instruction_arg,
                    next_opcode,
                ]);
                program_index.increment();
            } else {
                let next_opcode: BFieldElement = next.opcode().into();
                base_matrices
                    .instruction_matrix
                    .push([program_index, current_opcode, next_opcode]);
                program_index.increment();
            }

            current_instruction = next;
        }

        // 1.a. Add a terminating zero line
        let last_instruction_pointer = program_index;
        let last_instruction = current_instruction.opcode().into();
        let zero = BWord::ring_zero();
        base_matrices
            .instruction_matrix
            .push([last_instruction_pointer, last_instruction, zero]);

        // 2. Populate all tables with execution trace
        for row in trace {
            // 2.a. processor_matrix.push(vmstate.to_arr())
            base_matrices.processor_matrix.push(*row);
        }

        base_matrices
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

#[cfg(test)]
mod triton_vm_tests {
    use super::*;
    use crate::shared_math::stark::triton::instruction::sample_programs;

    #[test]
    fn initialise_table_test() {
        // 1. Execute program
        let code = sample_programs::GCD_42_56;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, _out, _err) = program.run_with_input(&[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        // 2. Convert trace to base matrices

        // 3. Extract constraints
        // 4. Check constraints
    }
}
