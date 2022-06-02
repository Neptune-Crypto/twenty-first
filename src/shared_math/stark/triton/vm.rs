use itertools::Itertools;
use rand::Rng;

use super::instruction::{parse, Instruction};
use super::state::{VMState, AUX_REGISTER_COUNT};
use super::stdio::{InputStream, OutputStream, VecStream};
use super::table::base_matrix::BaseMatrices;
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
        let pos = self.cursor.position() as usize;
        let instructions = self.cursor.get_ref();
        let instruction = *instructions.get(pos)?;
        self.cursor.set_position((pos + instruction.size()) as u64);

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
    /// Create a `Program` from a slice of `Instruction`.
    ///
    /// All valid programs terminate with `Halt`.
    ///
    /// `new()` will append `Halt` if not present.
    pub fn new(input: &[Instruction]) -> Self {
        let mut instructions = vec![];
        for instr in input {
            instructions.append(&mut vec![*instr; instr.size()]);
        }

        if instructions.last() != Some(&Instruction::Halt) {
            instructions.push(Instruction::Halt)
        }

        Program { instructions }
    }

    /// Create a `Program` by parsing source code.
    ///
    /// All valid programs terminate with `Halt`.
    ///
    /// `from_code()` will append `Halt` if not present.
    pub fn from_code(code: &str) -> Result<Self, Box<dyn Error>> {
        let instructions = parse(code)?;
        Ok(Program::new(&instructions))
    }

    /// Convert a `Program` to a `Vec<BWord>`.
    ///
    /// Every single-word instruction is converted to a single word.
    ///
    /// Every double-word instruction is converted to two words.
    pub fn to_bwords(&self) -> Vec<BWord> {
        self.clone()
            .into_iter()
            .map(|instruction| {
                let opcode = instruction.opcode_b();
                if let Some(arg) = instruction.arg() {
                    vec![opcode, arg]
                } else {
                    vec![opcode]
                }
            })
            .concat()
    }

    /// Simulate (execute) a `Program` and record every state transition.
    ///
    /// Returns a `BaseMatrices` that records the execution.
    ///
    /// Optionally returns errors on premature termination, but returns a
    /// `BaseMatrices` for the execution up to the point of failure.
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
        let mut base_matrices = BaseMatrices::default();
        base_matrices.initialize(self);

        // FIXME: Get rid of .unwrap() x 3 in favor of safe error handling.
        let mut state = VMState::new(self);
        base_matrices.append(&state, None, state.current_instruction().unwrap());

        while !state.is_complete() {
            let written_word = match state.step_mut(rng, stdin, rescue_prime) {
                Err(err) => return (base_matrices, Some(err)),
                Ok(word) => word,
            };

            base_matrices.append(&state, written_word, state.current_instruction().unwrap());

            if let Some(word) = written_word {
                let _written = stdout.write_elem(word);
            }
        }

        base_matrices.append(&state, None, state.current_instruction().unwrap());
        base_matrices.sort_instruction_matrix();
        base_matrices.sort_jump_stack_matrix();

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

        while !current_state.is_complete() {
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
    use crate::shared_math::stark::triton::{
        instruction::sample_programs,
        table::{base_matrix::ProcessorMatrixRow, base_table},
    };

    #[test]
    fn initialise_table_test() {
        // 1. Parse program
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut rng = rand::thread_rng();
        let mut stdin = VecStream::new_b(&[42.into(), 56.into()]);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        // 2. Execute program, convert to base matrices
        let (base_matrices, err) =
            program.simulate(&mut rng, &mut stdin, &mut stdout, &rescue_prime);

        println!("Err: {:?}", err);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }

        println!("{:?}", base_matrices.output_matrix);

        // 3. Extract constraints
        // 4. Check constraints
    }

    #[test]
    fn initialise_table_42_test() {
        // 1. Execute program
        let code = sample_programs::SUBTRACT;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut rng = rand::thread_rng();
        let mut stdin = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut rng, &mut stdin, &mut stdout, &rescue_prime);

        println!("{:?}", err);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    #[test]
    fn simulate_gcd_test() {
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut rng = rand::thread_rng();
        let mut stdin = VecStream::new_b(&[42.into(), 56.into()]);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut rng, &mut stdin, &mut stdout, &rescue_prime);

        assert!(err.is_none());
        let expected = BWord::new(14);
        let actual = base_matrices.output_matrix[0][0];

        assert_eq!(expected, actual);
    }
}
