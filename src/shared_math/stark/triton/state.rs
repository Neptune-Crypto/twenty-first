use super::error::{vm_fail, InstructionError::*};
use super::instruction::{Instruction, Instruction::*};
use super::op_stack::OpStack;
use super::ord_n::{Ord6, Ord8::*};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::rescue_prime_xlix::RescuePrimeXlix;
use crate::shared_math::stark::triton::error::vm_err;
use crate::shared_math::traits::{GetRandomElements, IdentityValues, Inverse};
use crate::shared_math::x_field_element::XFieldElement;
use byteorder::{BigEndian, ReadBytesExt};
use rand::Rng;
use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::io::{Stdin, Stdout, Write};

type BWord = BFieldElement;
type XWord = XFieldElement;

/// The number of `BFieldElement`s in a Rescue-Prime digest for Triton VM.
pub const DIGEST_LEN: usize = 6;

/// The number of auxiliary registers for hashing-specific instructions.
pub const AUX_REGISTER_COUNT: usize = 16;

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
    pub fn step<R: Rng>(
        &self,
        rng: &mut R,
        rescue_prime: &RescuePrimeXlix<AUX_REGISTER_COUNT>,
        stdin: &mut Stdin,
        stdout: &mut Stdout,
    ) -> Result<VMState<'pgm>, Box<dyn Error>> {
        let mut next_state = self.clone();
        next_state.step_mut(rng, rescue_prime, stdin, stdout)?;
        Ok(next_state)
    }

    /// Perform the state transition as a mutable operation on `self`.
    ///
    /// This function is called from `step`.
    fn step_mut<R: Rng>(
        &mut self,
        rng: &mut R,
        rescue_prime: &RescuePrimeXlix<AUX_REGISTER_COUNT>,
        stdin: &mut Stdin,
        stdout: &mut Stdout,
    ) -> Result<(), Box<dyn Error>> {
        // All instructions increase the cycle count
        self.cycle_count += 1;

        let instruction = self.current_instruction()?;
        match instruction {
            Pop => {
                self.op_stack.pop()?;
                self.instruction_pointer += 1;
            }

            Push => {
                // FIXME: Consider what type of error this gives for invalid programs.
                if let PushArg(arg) = self.current_instruction_arg()? {
                    self.op_stack.push(arg);
                    self.instruction_pointer += 2;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
            }

            Pad => {
                let elem = BWord::random_elements(1, rng)[0];
                self.op_stack.push(elem);
                self.instruction_pointer += 1;
            }

            Dup => {
                // FIXME: Consider what type of error this gives for invalid programs.
                if let DupArg(arg) = self.current_instruction_arg()? {
                    let elem = self.op_stack.safe_peek(arg.into());
                    self.op_stack.push(elem);
                    self.instruction_pointer += 2;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
            }

            Swap => {
                // st[0] ... st[n] -> st[n] ... st[0]
                // FIXME: Consider what type of error this gives for invalid programs.
                if let SwapArg(arg) = self.current_instruction_arg()? {
                    self.op_stack.safe_swap(arg.into());
                    self.instruction_pointer += 2;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
            }

            Skiz => {
                let next_instruction = self.next_sequential_instruction()?;
                if !vec![Call, Recurse, Return].contains(&next_instruction) {
                    return vm_err(IllegalInstructionAfterSkiz);
                }
                let elem = self.op_stack.pop()?;
                self.instruction_pointer += if elem.is_zero() {
                    1 + next_instruction.size()
                } else {
                    1
                }
            }

            Call => {
                // FIXME: Consider what type of error this gives for invalid programs.
                if let CallArg(addr) = self.current_instruction_arg()? {
                    self.jump_stack.push(self.instruction_pointer + 2);
                    self.instruction_pointer = addr.value() as usize;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
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

            Save => {
                self.save()?;
                self.instruction_pointer += 1;
            }

            Xlix => {
                rescue_prime.rescue_xlix_permutation(&mut self.aux);
                self.instruction_pointer += 1;
            }

            ClearAll => {
                self.aux.fill(0.into());
                self.instruction_pointer += 1;
            }

            Squeeze => {
                // FIXME: Consider what type of error this gives for invalid programs.
                if let SqueezeArg(arg) = self.current_instruction_arg()? {
                    let n: usize = arg.into();
                    self.op_stack.push(self.aux[n]);
                    self.instruction_pointer += 1;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
            }

            Absorb => {
                // FIXME: Consider what type of error this gives for invalid programs.
                if let AbsorbArg(arg) = self.current_instruction_arg()? {
                    let n: usize = arg.into();
                    let elem = self.op_stack.pop()?;
                    self.aux[n] = elem;
                    self.instruction_pointer += 1;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
            }

            MerkleLeft => todo!(),
            MerkleRight => todo!(),

            CmpDigest => {
                let cmp_bword = if self.cmp_digest() {
                    BWord::ring_one()
                } else {
                    BWord::ring_zero()
                };
                self.op_stack.push(cmp_bword);
                self.instruction_pointer += 1;
            }

            Add => {
                let a = self.op_stack.pop()?;
                let b = self.op_stack.pop()?;
                self.op_stack.push(a + b);
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
                if elem.is_zero() {
                    return vm_err(InverseOfZero);
                }
                self.op_stack.push(elem.inverse());
                self.instruction_pointer += 1;
            }

            Split => {
                let elem = self.op_stack.pop()?;
                let n: u64 = elem.value();
                let lo = n & 0xffff_ffff;
                let hi = n >> 32;
                self.op_stack.push(BWord::new(lo as u64));
                self.op_stack.push(BWord::new(hi as u64));
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

            XxAdd => {
                let a: XWord = self.op_stack.popx()?;
                let b: XWord = self.op_stack.popx()?;
                self.op_stack.pushx(a + b);
                self.instruction_pointer += 1;
            }

            XxMul => {
                let a: XWord = self.op_stack.popx()?;
                let b: XWord = self.op_stack.popx()?;
                self.op_stack.pushx(a + b);
                self.instruction_pointer += 1;
            }

            XInv => {
                let a: XWord = self.op_stack.popx()?;
                self.op_stack.pushx(a.inverse());
                self.instruction_pointer += 1;
            }

            XbMul => {
                let x: XWord = self.op_stack.popx()?;
                let b: BWord = self.op_stack.pop()?;
                self.op_stack.pushx(x * XWord::new_const(b));
                self.instruction_pointer += 1;
            }

            Print => {
                let codepoint: u32 = self.op_stack.pop()?.try_into()?;
                let out_char = codepoint.to_be_bytes();
                let _written = stdout.write(&out_char)?;
                self.instruction_pointer += 1;
            }

            Scan => {
                let in_char: u32 = stdin.read_u32::<BigEndian>()?;
                self.op_stack.push(in_char.into());
                self.instruction_pointer += 1;
            }

            PushArg(_) => return vm_err(RunawayInstructionArg),
            DupArg(_) => return vm_err(RunawayInstructionArg),
            SwapArg(_) => return vm_err(RunawayInstructionArg),
            CallArg(_) => return vm_err(RunawayInstructionArg),
            SqueezeArg(_) => return vm_err(RunawayInstructionArg),
            AbsorbArg(_) => return vm_err(RunawayInstructionArg),
        }

        // Check that no instruction left the OpStack with too few elements
        if self.op_stack.is_too_shallow() {
            return vm_err(OpStackTooShallow);
        }

        Ok(())
    }

    /// Get the i'th instruction bit
    pub fn ib(&self, arg: Ord6) -> Result<BFieldElement, Box<dyn Error>> {
        let instruction = self.current_instruction()?;
        let opcode = instruction
            .opcode()
            .ok_or_else(|| vm_fail(RunawayInstructionArg))?;

        let bit_number: usize = arg.into();
        let bit_mask: u32 = 1 << bit_number;

        Ok((opcode & bit_mask).into())
    }

    /// Jump-stack pointer
    pub fn jsp(&self) -> BWord {
        let height = self.jump_stack.len();
        if height == 0 {
            0.into()
        } else {
            BWord::new((height - 1) as u64)
        }
    }

    /// Jump-stack value
    pub fn jsv(&self) -> BWord {
        self.jump_stack
            .last()
            .map(|&v| BWord::new(v as u64))
            .unwrap_or_else(|| 0.into())
    }

    /// Internal helper functions

    fn current_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .copied()
    }

    // FIXME: Maybe give a more specific error here.
    fn current_instruction_arg(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer + 1)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .copied()
    }

    // Return the next instruction on the tape, skipping arguments
    //
    // Note that this is not necessarily the next instruction to execute,
    // since the current instruction could be a jump.
    fn next_sequential_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        let ci = self.current_instruction()?;
        let ci_size = ci.size();
        self.program
            .get(self.instruction_pointer + ci_size)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .copied()
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
        let mem_addr = self.op_stack.safe_peek(ST0);
        let mem_val = self.memory_get(&mem_addr)?;
        self.op_stack.push(mem_val);
        Ok(())
    }

    fn save(&mut self) -> Result<(), Box<dyn Error>> {
        let mem_value = self.op_stack.pop()?;
        let mem_addr = self.op_stack.safe_peek(ST0);
        self.ram.insert(mem_addr, mem_value);
        Ok(())
    }

    fn cmp_digest(&self) -> bool {
        for i in 0..DIGEST_LEN {
            // Safe as long as DIGEST_LEN <= OP_STACK_REG_COUNT
            if self.aux[i] != self.op_stack.safe_peek(i.try_into().unwrap()) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod vm_state_tests {
    use super::super::op_stack::OP_STACK_REG_COUNT;
    use super::*;
    use crate::shared_math::rescue_prime_xlix;
    use crate::shared_math::stark::triton;
    use crate::shared_math::stark::triton::instruction;
    use crate::shared_math::stark::triton::instruction::sample_programs;

    // Property: All instructions increase the cycle count by 1.
    // Property: Most instructions increase the instruction pointer by 1.

    #[test]
    fn op_stack_big_enough_test() {
        assert!(
            DIGEST_LEN <= OP_STACK_REG_COUNT,
            "The OpStack must be large enough to hold a single Rescue-Prime digest"
        );
    }

    fn step_with_program<'pgm>(prev_state: &'pgm VMState) -> Result<VMState<'pgm>, Box<dyn Error>> {
        let mut rng = rand::thread_rng();
        let rescue_prime = rescue_prime_xlix::neptune_params();
        let mut stdin = std::io::stdin();
        let mut stdout = std::io::stdout();

        prev_state.step(&mut rng, &rescue_prime, &mut stdin, &mut stdout)
    }

    #[test]
    fn run_parse_pop_p() {
        let pgm = sample_programs::push_push_add_pop_p();
        let trace = triton::vm::run(&pgm).unwrap();

        for state in trace {
            println!("{:?}", state);
        }
    }

    #[test]
    fn run_hello_world_1() {
        let code = sample_programs::HELLO_WORLD_1;
        let program = instruction::parse(code).unwrap();
        let trace = triton::vm::run(&program).unwrap();

        // for state in trace {
        //     println!("{:?}", state);
        // }

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::ring_zero(), last_state.op_stack.safe_peek(ST0));

        println!("{:#?}", last_state);
    }
}
