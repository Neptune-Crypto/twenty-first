use rand::Rng;

use super::instruction::{parse, Instruction};
use super::state::{VMState, AUX_REGISTER_COUNT};
use super::stdio::{InputStream, OutputStream, VecStream};
use crate::shared_math::rescue_prime_xlix::{self, RescuePrimeXlix};
use std::error::Error;
use std::fmt::Display;

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
            if matches!(instruction, Instruction::Push(_)) {
                stream.next();
            }
        }
        Ok(())
    }
}

/// A `Program` is a `Vec<Instruction>` that contains duplicate elements for
/// instructions with a size of 2. This means that the index in the vector
/// corresponds to the VM's `instruction_pointer`. These duplicate values
/// should most often be skipped/ignored, e.g. when pretty-printing.
impl Program {
    pub fn from_code(code: &str) -> Result<Self, Box<dyn Error>> {
        let instructions = parse(code)?;
        Ok(Program::from_instr(&instructions))
    }

    pub fn from_instr(input: &[Instruction]) -> Self {
        let mut instructions = vec![];
        for instr in input {
            instructions.append(&mut vec![*instr; instr.size()]);
        }
        Program { instructions }
    }

    pub fn run_stdio(&self) -> (Vec<VMState>, Option<Box<dyn Error>>) {
        let mut rng = rand::thread_rng();
        let mut stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        let rescue_prime = rescue_prime_xlix::neptune_params();

        self.run(&mut rng, &mut stdin, &mut stdout, &rescue_prime)
    }

    pub fn run_with_input(&self, input: &[u8]) -> (Vec<VMState>, Vec<u8>, Option<Box<dyn Error>>) {
        let mut rng = rand::thread_rng();
        let mut stdin = VecStream::new(input);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = rescue_prime_xlix::neptune_params();

        let (trace, err) = self.run(&mut rng, &mut stdin, &mut stdout, &rescue_prime);

        (trace, stdout.to_vec(), err)
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
        let mut trace = vec![VMState::new(self)];
        let mut prev_state = trace.last().unwrap();

        while !prev_state.is_final() {
            let next_state = prev_state.step(rng, stdin, stdout, rescue_prime);
            if let Err(err) = next_state {
                return (trace, Some(err));
            }
            trace.push(next_state.unwrap());
            prev_state = trace.last().unwrap();
        }

        (trace, None)
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
    use super::Instruction::*;
    use super::*;

    #[test]
    fn vm_run_test() {
        let instructions = vec![Push(2.into()), Push(2.into()), Add];
        let program = Program::from_instr(&instructions);
        let _bla = program.run_stdio();
    }
}
