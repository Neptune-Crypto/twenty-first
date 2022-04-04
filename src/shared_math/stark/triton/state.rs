use super::error::{vm_fail, InstructionError::*};
use super::instruction::{Instruction, Ord4::*, DUP0, DUP1, DUP2, DUP3};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::triton::error::vm_err;
use crate::shared_math::traits::{GetRandomElements, IdentityValues};
use rand::prelude::ThreadRng;
use rand::Rng;
use std::error::Error;
use Instruction::*;

type BWord = BFieldElement;

#[derive(Debug, Default, Clone)]
pub struct VMLiterally {
    // Registers
    pub clk: BWord, // Number of cycles the program has been running for
    pub ip: u32,    // Instruction pointer
    pub ci: BWord,  // Current instruction
    pub ni: BWord,  // Next instruction

    pub ib: [BWord; 6],   // Contains the i'th bit of the instruction
    pub if_: [BWord; 10], // Instruction flags
    pub jsp: BWord,       // Jump stack pointer
    pub jsv: BWord,       // Return address
    pub st: [BWord; 4],   // Operational stack registers
    pub iszero: BWord,    // Top of stack zero indicator
    pub osp: BWord,       // Operational stack pointer
    pub osv: BWord,       // Operational stack value
    pub hv: [BWord; 5],   // Helper variable registers
    pub ramp: BWord,      // RAM pointer
    pub ramv: BWord,      // RAM value
    pub aux: [BWord; 16], // Auxiliary registers (for hashing)

    // Stack, memory
    pub stack: Vec<BWord>,
    pub memory: Vec<BWord>,
}

// An experiment in expressing a reduced VM state (only fields that are not directly derived from others)
#[derive(Debug, Default, Clone)]
pub struct VMState<'pgm> {
    program: &'pgm [Instruction],
    op_stack: Vec<BWord>,
    jump_stack: Vec<usize>,
    memory: Vec<BWord>,

    // Registers
    pub cycle_count: u32, // Number of cycles the program has been running for
    pub instruction_pointer: usize, // hmm

    pub ib: [BWord; 6],   // Contains the i'th bit of the instruction
    pub if_: [BWord; 10], // Instruction flags
    pub jsp: BWord,       // Jump stack pointer
    pub jsv: BWord,       // Return address
    pub iszero: BWord,    // Top of stack zero indicator
    pub osp: BWord,       // Operational stack pointer
    pub osv: BWord,       // Operational stack value
    pub hv: [BWord; 5],   // Helper variable registers
    pub ramp: BWord,      // RAM pointer
    pub ramv: BWord,      // RAM value
    pub aux: [BWord; 16], // Auxiliary registers (for hashing)
}

impl<'pgm> VMState<'pgm> {
    /// Create initial `VMState` for a given `program`
    ///
    /// Since `program` is read-only and transcends any individual state,
    /// it is included as an immutable reference that exceeds the lifetime
    /// of a single state.
    pub fn new(program: &'pgm [Instruction]) -> Self {
        let mut initial_state = VMState::default();
        initial_state.program = program;
        initial_state
    }

    /// Determine if this is a final state.
    pub fn is_final(&self) -> bool {
        self.instruction_pointer == self.program.len()
    }

    /// Given a state, compute the next state.
    pub fn step(&self, rng: &mut ThreadRng) -> Result<VMState<'pgm>, Box<dyn Error>> {
        let mut next_state = self.clone();
        next_state.step_mut(rng)?;
        Ok(next_state)
    }

    /// Perform the state transition as a mutable operation on `self`.
    ///
    /// This function is called from `step`.
    fn step_mut(&mut self, rng: &mut ThreadRng) -> Result<(), Box<dyn Error>> {
        // All instructions increase the cycle count
        self.cycle_count += 1;

        let instruction = self.current_instruction()?;
        match instruction {
            Pop => {
                self.assert_op_stack_height(1)?;
                self.op_stack.pop();
                self.instruction_pointer += 1;
            }

            Push(arg) => {
                self.op_stack.push(arg);
                self.instruction_pointer += 1;
            }

            Pad => {
                let elem = BWord::random_elements(1, rng)[0];
                self.op_stack.push(elem);
                self.instruction_pointer += 1;
            }

            Dup(n) => {
                let n: usize = n.into();
                self.assert_op_stack_height(n + 1)?;
                self.op_stack.push(self.op_stack[n]);
                self.instruction_pointer += 1;
            }

            Swap => {
                // a b -> b a
                self.assert_op_stack_height(2)?;
                self.op_stack.swap(0, 1);
                self.instruction_pointer += 1;
            }

            Pull2 => {
                // a b c -> b a c -> b c a
                self.assert_op_stack_height(3)?;
                self.op_stack.swap(0, 1);
                self.op_stack.swap(1, 2);
                self.instruction_pointer += 1;
            }

            Pull3 => {
                // a b c d -> b a c d -> b c a d -> b c d a
                self.assert_op_stack_height(4)?;
                self.op_stack.swap(0, 1);
                self.op_stack.swap(1, 2);
                self.op_stack.swap(2, 3);
                self.instruction_pointer += 1;
            }

            Nop => {
                self.instruction_pointer += 1;
            }

            Skiz => {
                self.assert_op_stack_height(1)?;
                let elem = self.op_stack.pop().unwrap();
                self.instruction_pointer += if elem.is_zero() { 2 } else { 1 }
            }

            Call(addr) => {
                self.jump_stack.push(self.instruction_pointer + 2);
                self.instruction_pointer = addr.value() as usize;
            }

            Return => {}

            Recurse => {}

            Assert => todo!(),
            Halt => todo!(),
            Load => todo!(),
            LoadInc => todo!(),
            LoadDec => todo!(),
            Save => todo!(),
            SaveInc => todo!(),
            SaveDec => todo!(),
            SetRamp => todo!(),
            GetRamp => todo!(),
            Xlix => todo!(),
            Ntt => todo!(),
            Intt => todo!(),
            ClearAll => todo!(),
            Squeeze(_) => todo!(),
            Absorb(_) => todo!(),
            Clear(_) => todo!(),
            Rotate(_) => todo!(),
            Add => todo!(),
            Neg => todo!(),
            Mul => todo!(),
            Inv => todo!(),
            Lnot => todo!(),
            Split => todo!(),
            Eq => todo!(),
            Lt => todo!(),
            And => todo!(),
            Or => todo!(),
            Xor => todo!(),
            Reverse => todo!(),
            Div => todo!(),
        }

        Ok(())
    }

    fn assert_op_stack_height(&self, n: usize) -> Result<(), Box<dyn Error>> {
        if n > self.op_stack.len() {
            Err(vm_fail(OpStackTooShallow))
        } else {
            Ok(())
        }
    }

    fn assert_jump_stack_height(&self, n: usize) -> Result<(), Box<dyn Error>> {
        if n > self.jump_stack.len() {
            Err(vm_fail(JumpStackTooShallow))
        } else {
            Ok(())
        }
    }

    pub fn current_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .map(|&instruction| instruction)
    }

    pub fn next_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        match self.current_instruction()? {
            // Skip next instruction if top of stack is zero.
            Skiz => todo!(),
            _ => todo!(),
        }
    }

    // Registers that really just wrap another part of the machine state
    pub fn stack_0(&self) -> Result<BFieldElement, Box<dyn Error>> {
        Ok(*self.op_stack.get(0).unwrap_or(&BFieldElement::ring_zero()))
    }

    pub fn stack_1(&self) -> Result<BFieldElement, Box<dyn Error>> {
        Ok(*self.op_stack.get(1).unwrap_or(&BFieldElement::ring_zero()))
    }

    pub fn stack_2(&self) -> Result<BFieldElement, Box<dyn Error>> {
        Ok(*self.op_stack.get(2).unwrap_or(&BFieldElement::ring_zero()))
    }

    pub fn stack_3(&self) -> Result<BFieldElement, Box<dyn Error>> {
        Ok(*self.op_stack.get(3).unwrap_or(&BFieldElement::ring_zero()))
    }
}

#[cfg(test)]
mod vm_state_tests {
    // Property: All instructions increase the cycle count by 1.
    // Property: Most instructions increase the instruction pointer by 1.
}
