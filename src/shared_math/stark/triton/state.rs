use super::error::{vm_fail, InstructionError::*};
use super::instruction::Instruction;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::stark::triton::error::vm_err;
use crate::shared_math::traits::{GetRandomElements, IdentityValues, Inverse};
use rand::Rng;
use std::collections::HashMap;
use std::convert::TryInto;
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
    memory: HashMap<BWord, BWord>,

    // Registers
    pub cycle_count: u32, // Number of cycles the program has been running for
    pub instruction_pointer: usize, // Current instruction's address in program memory

    pub ib: [BWord; 6],   // Contains the i'th bit of the instruction
    pub if_: [BWord; 10], // Instruction flags
    pub jsp: BWord,       // Jump stack pointer
    pub jsv: BWord,       // Return address
    pub iszero: BWord,    // Top of stack zero indicator
    pub osp: BWord,       // Operational stack pointer
    pub osv: BWord,       // Operational stack value
    pub hv: [BWord; 5],   // Helper variable registers

    pub ram_pointer: BWord, // RAM pointer
    pub ram_value: BWord,   // RAM value
    pub aux: [BWord; 16],   // Auxiliary registers (for hashing)
}

impl<'pgm> VMState<'pgm> {
    /// Create initial `VMState` for a given `program`
    ///
    /// Since `program` is read-only and transcends any individual state,
    /// it is included as an immutable reference that exceeds the lifetime
    /// of a single state.
    pub fn new(program: &'pgm [Instruction]) -> Self {
        Self {
            program,
            ..VMState::default()
        }
    }

    /// Determine if this is a final state.
    pub fn is_final(&self) -> bool {
        self.instruction_pointer == self.program.len()
    }

    /// Given a state, compute the next state purely.
    pub fn step<R: Rng>(&self, rng: &mut R) -> Result<VMState<'pgm>, Box<dyn Error>> {
        let mut next_state = self.clone();
        next_state.step_mut(rng)?;
        Ok(next_state)
    }

    /// Perform the state transition as a mutable operation on `self`.
    ///
    /// This function is called from `step`.
    fn step_mut<R: Rng>(&mut self, rng: &mut R) -> Result<(), Box<dyn Error>> {
        // All instructions increase the cycle count
        self.cycle_count += 1;

        let instruction = self.current_instruction()?;
        match instruction {
            Pop => {
                self.op_stack_pop()?;
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

            Dup(arg) => {
                let n: usize = arg.into();
                let elem = self.op_stack_peek(n)?;
                self.op_stack.push(elem);
                self.instruction_pointer += 1;
            }

            Swap => {
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
                let elem = self.op_stack_pop()?;
                self.instruction_pointer += if elem.is_zero() { 2 } else { 1 }
            }

            Call(addr) => {
                self.jump_stack.push(self.instruction_pointer + 2);
                self.instruction_pointer = addr.value() as usize;
            }

            Return => {
                let addr = self.jump_stack_pop()?;
                self.instruction_pointer = addr;
            }

            Recurse => {
                let dest_addr = self.jump_stack_get(1)?;
                self.instruction_pointer = dest_addr;
            }

            Assert => {
                self.assert_op_stack_height(1)?;
                let elem = self.op_stack.pop().unwrap();
                if !elem.is_one() {
                    return Err(vm_fail(AssertionFailed));
                }
                self.instruction_pointer += 1;
            }

            Halt => {
                self.instruction_pointer = self.program.len();
            }

            Load => {
                self.load()?;
                self.instruction_pointer += 1;
            }

            LoadInc => {
                self.load()?;
                self.ram_pointer.increment();
                self.instruction_pointer += 1;
            }

            LoadDec => {
                self.load()?;
                self.ram_pointer.decrement();
                self.instruction_pointer += 1;
            }

            Save => {
                self.save()?;
                self.instruction_pointer += 1;
            }

            SaveInc => {
                self.save()?;
                self.ram_pointer.increment();
                self.instruction_pointer += 1;
            }

            SaveDec => {
                self.save()?;
                self.ram_pointer.decrement();
                self.instruction_pointer += 1;
            }

            SetRamp => {
                let new_ramp = self.op_stack_pop()?;
                self.ram_pointer = new_ramp;
                self.instruction_pointer += 1;
            }

            GetRamp => {
                self.op_stack.push(self.ram_pointer);
                self.instruction_pointer += 1;
            }

            Xlix => todo!(),
            Ntt => todo!(),
            Intt => todo!(),

            ClearAll => {
                self.aux = [BWord::ring_zero(); 16];
                self.instruction_pointer += 1;
            }

            Squeeze(arg) => {
                let n: usize = arg.into();
                self.op_stack.push(self.aux[n]);
                self.instruction_pointer += 1;
            }

            Absorb(arg) => {
                let n: usize = arg.into();
                let elem = self.op_stack_pop()?;
                self.aux[n] = elem;
                self.instruction_pointer += 1;
            }

            Clear(arg) => {
                let n: usize = arg.into();
                self.aux[n] = BWord::ring_zero();
                self.instruction_pointer += 1;
            }

            Rotate(arg) => {
                let n: usize = arg.into();
                self.aux.rotate_right(n);
                self.instruction_pointer += 1;
            }

            Add => {
                let a = self.op_stack_pop()?;
                let b = self.op_stack_pop()?;
                self.op_stack.push(a + b);
                self.instruction_pointer += 1;
            }

            Neg => {
                let elem = self.op_stack_pop()?;
                self.op_stack.push(-elem);
                self.instruction_pointer += 1;
            }

            Mul => {
                let a = self.op_stack_pop()?;
                let b = self.op_stack_pop()?;
                self.op_stack.push(a * b);
                self.instruction_pointer += 1;
            }

            Inv => {
                let elem = self.op_stack_pop()?;
                self.op_stack.push(elem.inverse());
                self.instruction_pointer += 1;
            }

            Lnot => {
                let elem = self.op_stack_pop()?;
                if elem.is_zero() {
                    self.op_stack.push(1.into());
                } else if elem.is_one() {
                    self.op_stack.push(0.into());
                } else {
                    return vm_err(LnotNonBinaryInput);
                }
                self.instruction_pointer += 1;
            }

            Split => {
                let elem = self.op_stack_pop()?;
                let n: u64 = elem.value();
                let lo = n & 0xffff_ffff;
                let hi = n >> 32;
                self.op_stack.push(BWord::new(lo as u128));
                self.op_stack.push(BWord::new(hi as u128));
                self.instruction_pointer += 1;
            }

            Eq => {
                let a = self.op_stack_pop()?;
                let b = self.op_stack_pop()?;
                if a == b {
                    self.op_stack.push(BWord::ring_one());
                } else {
                    self.op_stack.push(BWord::ring_zero());
                };
                self.instruction_pointer += 1;
            }

            Lt => {
                let a: u32 = self.op_stack_pop()?.try_into()?;
                let b: u32 = self.op_stack_pop()?.try_into()?;
                if a < b {
                    self.op_stack.push(BWord::ring_one());
                } else {
                    self.op_stack.push(BWord::ring_zero());
                };
                self.instruction_pointer += 1;
            }

            And => {
                let a: u32 = self.op_stack_pop()?.try_into()?;
                let b: u32 = self.op_stack_pop()?.try_into()?;
                self.op_stack.push((a & b).into());
                self.instruction_pointer += 1;
            }

            Or => {
                let a: u32 = self.op_stack_pop()?.try_into()?;
                let b: u32 = self.op_stack_pop()?.try_into()?;
                self.op_stack.push((a | b).into());
                self.instruction_pointer += 1;
            }

            Xor => {
                let a: u32 = self.op_stack_pop()?.try_into()?;
                let b: u32 = self.op_stack_pop()?.try_into()?;
                self.op_stack.push((a ^ b).into());
                self.instruction_pointer += 1;
            }

            Reverse => {
                let elem: u32 = self.op_stack_pop()?.try_into()?;
                self.op_stack.push(elem.reverse_bits().into());
                self.instruction_pointer += 1;
            }

            Div => {
                let a: u32 = self.op_stack_pop()?.try_into()?;
                let b: u32 = self.op_stack_pop()?.try_into()?;
                let (quot, rem) = other::div_rem(a, b);
                self.op_stack.push(quot.into());
                self.op_stack.push(rem.into());
                self.instruction_pointer += 1;
            }
        }

        Ok(())
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

    // Internal helper functions

    fn current_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .map(|&instruction| instruction)
    }

    fn _next_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        match self.current_instruction()? {
            // Skip next instruction if top of stack is zero.
            Skiz => todo!(),
            _ => todo!(),
        }
    }

    fn assert_op_stack_height(&self, n: usize) -> Result<(), Box<dyn Error>> {
        if n > self.op_stack.len() {
            Err(vm_fail(OpStackTooShallow))
        } else {
            Ok(())
        }
    }

    fn op_stack_pop(&mut self) -> Result<BFieldElement, Box<dyn Error>> {
        self.op_stack
            .pop()
            .ok_or_else(|| vm_fail(OpStackTooShallow))
    }

    fn op_stack_peek(&mut self, n: usize) -> Result<BFieldElement, Box<dyn Error>> {
        self.op_stack
            .get(n)
            .copied()
            .ok_or_else(|| vm_fail(OpStackTooShallow))
    }

    fn jump_stack_pop(&mut self) -> Result<usize, Box<dyn Error>> {
        self.jump_stack
            .pop()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn jump_stack_get(&mut self, n: usize) -> Result<usize, Box<dyn Error>> {
        self.jump_stack
            .get(n)
            .copied()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn memory_get(&self, mem_addr: &BFieldElement) -> Result<BFieldElement, Box<dyn Error>> {
        self.memory
            .get(mem_addr)
            .copied()
            .ok_or_else(|| vm_fail(MemoryAddressNotFound))
    }

    fn load(&mut self) -> Result<(), Box<dyn Error>> {
        let mem_addr = self.op_stack_peek(0)?;
        let mem_val = self.memory_get(&mem_addr)?;
        self.op_stack.push(mem_val);
        Ok(())
    }

    fn save(&mut self) -> Result<(), Box<dyn Error>> {
        let mem_value = self.op_stack_pop()?;
        let mem_addr = self.op_stack_peek(0)?;
        self.memory.insert(mem_addr, mem_value);
        Ok(())
    }
}

#[cfg(test)]
mod vm_state_tests {
    // Property: All instructions increase the cycle count by 1.
    // Property: Most instructions increase the instruction pointer by 1.
}
