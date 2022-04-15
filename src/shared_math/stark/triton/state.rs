use super::error::{vm_fail, InstructionError::*};
use super::instruction::{Instruction, Instruction::*};
use super::op_stack::OpStack;
use super::ord_n::{Ord4::*, Ord6};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::stark::triton::error::vm_err;
use crate::shared_math::traits::{GetRandomElements, IdentityValues, Inverse};
use rand::Rng;
use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;

type BWord = BFieldElement;

pub const AUX_REGISTER_COUNT: usize = 16;

// `VMLiterally` describes the registers, but in a way that requires a lot of
// manual updating.
#[derive(Debug, Clone)]
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
    pub inv: BWord,       // Top of op-stack inverse, or 0
    pub osp: BWord,       // Operational stack pointer
    pub osv: BWord,       // Operational stack value
    pub hv: [BWord; 5],   // Helper variable registers
    pub ramp: BWord,      // RAM pointer
    pub ramv: BWord,      // RAM value
    pub aux: [BWord; 16], // Auxiliary registers (for hashing)
}

#[derive(Debug, Default, Clone)]
pub struct VMState<'pgm> {
    ///
    /// Triton VM's four kinds of memory:
    ///
    /// 1. **Program memory**, from which the VM reads instructions
    program: &'pgm [Instruction],

    /// 2. **Random-access memory**, to which the VM can read and write field elements
    ram: HashMap<BWord, BWord>,

    /// 3. **Op-stack memory**, which stores the part of the operational stack
    ///    that is not represented explicitly by the operational stack registers
    ///
    ///    *(An implementation detail: We keep the entire stack in one `Vec<>`.)*
    op_stack: OpStack,

    /// 4. Jump-stack memory, which stores the entire jump stack
    jump_stack: Vec<usize>,

    ///
    /// Registers
    ///
    /// Number of cycles the program has been running for
    pub cycle_count: u32,

    /// Current instruction's address in program memory
    pub instruction_pointer: usize,

    /// Instruction flags
    pub ifl: [BWord; 10],

    /// RAM pointer
    pub ramp: BWord,

    /// RAM value
    pub ramv: BWord,

    /// Auxiliary registers
    pub aux: [BWord; AUX_REGISTER_COUNT],
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
                self.op_stack.pop()?;
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
                let elem = self.op_stack.safe_peek(n);
                self.op_stack.push(elem);
                self.instruction_pointer += 1;
            }

            Swap => {
                // a b -> b a
                self.op_stack.safe_swap(N0, N1);
                self.instruction_pointer += 1;
            }

            Pull2 => {
                // a b c -> b a c -> b c a
                self.op_stack.safe_swap(N0, N1);
                self.op_stack.safe_swap(N1, N2);
                self.instruction_pointer += 1;
            }

            Pull3 => {
                // a b c d -> b a c d -> b c a d -> b c d a
                self.op_stack.safe_swap(N0, N1);
                self.op_stack.safe_swap(N1, N2);
                self.op_stack.safe_swap(N2, N3);
                self.instruction_pointer += 1;
            }

            Nop => {
                self.instruction_pointer += 1;
            }

            Skiz => {
                let elem = self.op_stack.pop()?;
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
                let elem = self.op_stack.pop()?;
                if !elem.is_one() {
                    return vm_err(AssertionFailed);
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
                self.ramp.increment();
                self.instruction_pointer += 1;
            }

            LoadDec => {
                self.load()?;
                self.ramp.decrement();
                self.instruction_pointer += 1;
            }

            Save => {
                self.save()?;
                self.instruction_pointer += 1;
            }

            SaveInc => {
                self.save()?;
                self.ramp.increment();
                self.instruction_pointer += 1;
            }

            SaveDec => {
                self.save()?;
                self.ramp.decrement();
                self.instruction_pointer += 1;
            }

            SetRamp => {
                let new_ramp = self.op_stack.pop()?;
                self.ramp = new_ramp;
                self.instruction_pointer += 1;
            }

            GetRamp => {
                self.op_stack.push(self.ramp);
                self.instruction_pointer += 1;
            }

            Xlix => todo!(),
            Ntt => todo!(),
            Intt => todo!(),

            ClearAll => {
                self.aux.fill(0.into());
                self.instruction_pointer += 1;
            }

            Squeeze(arg) => {
                let n: usize = arg.into();
                self.op_stack.push(self.aux[n]);
                self.instruction_pointer += 1;
            }

            Absorb(arg) => {
                let n: usize = arg.into();
                let elem = self.op_stack.pop()?;
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
                let a = self.op_stack.pop()?;
                let b = self.op_stack.pop()?;
                self.op_stack.push(a + b);
                self.instruction_pointer += 1;
            }

            Neg => {
                let elem = self.op_stack.pop()?;
                self.op_stack.push(-elem);
                self.instruction_pointer += 1;
            }

            Mul => {
                let a = self.op_stack.pop()?;
                let b = self.op_stack.pop()?;
                self.op_stack.push(a * b);
                self.instruction_pointer += 1;
            }

            Inv => {
                let elem = self.op_stack.pop()?;
                self.op_stack.push(elem.inverse());
                self.instruction_pointer += 1;
            }

            Lnot => {
                let elem = self.op_stack.pop()?;
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
                let elem = self.op_stack.pop()?;
                let n: u64 = elem.value();
                let lo = n & 0xffff_ffff;
                let hi = n >> 32;
                self.op_stack.push(BWord::new(lo as u128));
                self.op_stack.push(BWord::new(hi as u128));
                self.instruction_pointer += 1;
            }

            Eq => {
                let a = self.op_stack.pop()?;
                let b = self.op_stack.pop()?;
                if a == b {
                    self.op_stack.push(BWord::ring_one());
                } else {
                    self.op_stack.push(BWord::ring_zero());
                };
                self.instruction_pointer += 1;
            }

            Lt => {
                let a: u32 = self.op_stack.pop()?.try_into()?;
                let b: u32 = self.op_stack.pop()?.try_into()?;
                if a < b {
                    self.op_stack.push(BWord::ring_one());
                } else {
                    self.op_stack.push(BWord::ring_zero());
                };
                self.instruction_pointer += 1;
            }

            And => {
                let a: u32 = self.op_stack.pop()?.try_into()?;
                let b: u32 = self.op_stack.pop()?.try_into()?;
                self.op_stack.push((a & b).into());
                self.instruction_pointer += 1;
            }

            Or => {
                let a: u32 = self.op_stack.pop()?.try_into()?;
                let b: u32 = self.op_stack.pop()?.try_into()?;
                self.op_stack.push((a | b).into());
                self.instruction_pointer += 1;
            }

            Xor => {
                let a: u32 = self.op_stack.pop()?.try_into()?;
                let b: u32 = self.op_stack.pop()?.try_into()?;
                self.op_stack.push((a ^ b).into());
                self.instruction_pointer += 1;
            }

            Reverse => {
                let elem: u32 = self.op_stack.pop()?.try_into()?;
                self.op_stack.push(elem.reverse_bits().into());
                self.instruction_pointer += 1;
            }

            Div => {
                let a: u32 = self.op_stack.pop()?.try_into()?;
                let b: u32 = self.op_stack.pop()?.try_into()?;
                let (quot, rem) = other::div_rem(a, b);
                self.op_stack.push(quot.into());
                self.op_stack.push(rem.into());
                self.instruction_pointer += 1;
            }
        }

        if self.op_stack.is_too_shallow() {
            return vm_err(OpStackTooShallow);
        }

        Ok(())
    }

    /// Get the i'th instruction bit
    pub fn ib(&self, arg: Ord6) -> Result<BFieldElement, Box<dyn Error>> {
        let instruction = self.current_instruction()?;
        let value = instruction.value();
        let bit_number: usize = arg.into();
        let bit_mask: u32 = 1 << bit_number;

        Ok((value & bit_mask).into())
    }

    /// Jump-stack pointer
    pub fn jsp(&self) -> BWord {
        let height = self.jump_stack.len();
        if height == 0 {
            0.into()
        } else {
            BWord::new((height - 1) as u128)
        }
    }

    /// Jump-stack value
    pub fn jsv(&self) -> BWord {
        self.jump_stack
            .last()
            .map(|&v| BWord::new(v as u128))
            .unwrap_or_else(|| 0.into())
    }

    /// Internal helper functions

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
        self.ram
            .get(mem_addr)
            .copied()
            .ok_or_else(|| vm_fail(MemoryAddressNotFound))
    }

    fn load(&mut self) -> Result<(), Box<dyn Error>> {
        let mem_addr = self.op_stack.safe_peek(N0);
        let mem_val = self.memory_get(&mem_addr)?;
        self.op_stack.push(mem_val);
        Ok(())
    }

    fn save(&mut self) -> Result<(), Box<dyn Error>> {
        let mem_value = self.op_stack.pop()?;
        let mem_addr = self.op_stack.safe_peek(N0);
        self.ram.insert(mem_addr, mem_value);
        Ok(())
    }
}

#[cfg(test)]
mod vm_state_tests {
    // Property: All instructions increase the cycle count by 1.
    // Property: Most instructions increase the instruction pointer by 1.
}
