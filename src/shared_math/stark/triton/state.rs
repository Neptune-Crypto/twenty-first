use super::error::{vm_fail, InstructionError::*};
use super::instruction::{Instruction, Instruction::*};
use super::op_stack::OpStack;
use super::ord_n::{Ord4, Ord6, Ord8::*};
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
use std::fmt::Display;
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

    ramp: BWord,
    ramv: BWord,

    /// 3. **Op-stack memory**, which stores the part of the operational stack
    ///    that is not represented explicitly by the operational stack registers
    ///
    ///    *(An implementation detail: We keep the entire stack in one `Vec<>`.)*
    op_stack: OpStack,

    /// 4. Jump-stack memory, which stores the entire jump stack
    jump_stack: Vec<(BWord, BWord)>,

    ///
    /// Registers
    ///
    /// Number of cycles the program has been running for
    pub cycle_count: u32,

    /// Current instruction's address in program memory
    pub instruction_pointer: usize,

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
                if let PushArg(arg) = self.ci_plus_1()? {
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
                if let DupArg(arg) = self.ci_plus_1()? {
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
                if let SwapArg(arg) = self.ci_plus_1()? {
                    self.op_stack.safe_swap(arg.into());
                    self.instruction_pointer += 2;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
            }

            Skiz => {
                let next_instruction = self.next_instruction()?;
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
                if let CallArg(addr) = self.ci_plus_1()? {
                    let o_plus_2 = self.instruction_pointer as u32 + 2;
                    let pair = (o_plus_2.into(), addr);
                    self.jump_stack.push(pair);
                    self.instruction_pointer = addr.value() as usize;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
            }

            Return => {
                let (orig_addr, _dest_addr) = self.jump_stack_pop()?;
                self.instruction_pointer = orig_addr.value() as usize;
            }

            Recurse => {
                let (_orig_addr, dest_addr) = self.jump_stack_peek()?;
                self.instruction_pointer = dest_addr.value() as usize;
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

            ReadMem => {
                self.ramp = self.op_stack.safe_peek(ST0);
                self.ramv = self.memory_get(&self.ramp)?;
                self.op_stack.push(self.ramv);
                self.instruction_pointer += 1;
            }

            WriteMem => {
                self.ramv = self.op_stack.pop()?;
                self.ramp = self.op_stack.safe_peek(ST0);
                self.ram.insert(self.ramp, self.ramv);
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
                if let SqueezeArg(arg) = self.ci_plus_1()? {
                    let n: usize = arg.into();
                    self.op_stack.push(self.aux[n]);
                    self.instruction_pointer += 1;
                } else {
                    return vm_err(RunawayInstructionArg);
                }
            }

            Absorb => {
                // FIXME: Consider what type of error this gives for invalid programs.
                if let AbsorbArg(arg) = self.ci_plus_1()? {
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

            WriteIo => {
                let codepoint: u32 = self.op_stack.pop()?.try_into()?;
                let out_char = codepoint.to_be_bytes();
                let _written = stdout.write(&out_char)?;
                self.instruction_pointer += 1;
            }

            ReadIo => {
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

    /// Jump-stack origin
    pub fn jso(&self) -> BWord {
        self.jump_stack
            .last()
            .map(|(o, _d)| *o)
            .unwrap_or_else(|| 0.into())
    }

    /// Jump-stack destination
    pub fn jsd(&self) -> BWord {
        self.jump_stack
            .last()
            .map(|(_o, d)| *d)
            .unwrap_or_else(|| 0.into())
    }

    /// Internal helper functions

    fn current_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .copied()
    }

    // Return the next instruction on the tape, skipping arguments
    //
    // Note that this is not necessarily the next instruction to execute,
    // since the current instruction could be a jump, but it is either
    // program[ip + 1] or program[ip + 2] depending on whether the current
    // instruction takes an argument or not.
    fn next_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        let ci = self.current_instruction()?;
        let ci_size = ci.size();
        self.program
            .get(self.instruction_pointer + ci_size)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .copied()
    }

    fn next_next_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        let cur_size = self.current_instruction()?.size();
        let next_size = self.next_instruction()?.size();
        self.program
            .get(self.instruction_pointer + cur_size + next_size)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .copied()
    }

    fn ci_plus_1(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer + 1)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .copied()
    }

    fn ci_plus_2(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer + 2)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow))
            .copied()
    }

    fn jump_stack_pop(&mut self) -> Result<(BWord, BWord), Box<dyn Error>> {
        self.jump_stack
            .pop()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn jump_stack_peek(&mut self) -> Result<(BWord, BWord), Box<dyn Error>> {
        self.jump_stack
            .last()
            .copied()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn memory_get(&self, mem_addr: &BFieldElement) -> Result<BFieldElement, Box<dyn Error>> {
        self.ram
            .get(mem_addr)
            .copied()
            .ok_or_else(|| vm_fail(MemoryAddressNotFound))
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

impl<'pgm> Display for VMState<'pgm> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        horizontal_bar(f)?;

        let ci = self
            .current_instruction()
            .map(|ni| ni.to_string())
            .unwrap_or("eof".to_string());

        let ni = self
            .next_instruction()
            .map(|w| w.to_string())
            .unwrap_or("none".to_string());

        let nni = self
            .next_next_instruction()
            .map(|w| w.to_string())
            .unwrap_or("none".to_string());

        let ci_plus_1 = self
            .ci_plus_1()
            .map(|ni| ni.to_string())
            .unwrap_or("none".to_string());

        column(
            f,
            format!(
                "clk: {}   ip: {}",
                self.cycle_count, self.instruction_pointer
            ),
        )?;

        column(f, format!("ci: {}   ci+1: {}", ci, ci_plus_1))?;
        column(f, format!("ni: {}   nni: {}", ni, nni))?;
        column(f, format!("ramp: {}   ramv: {}", self.ramp, self.ramv,))?;
        column(
            f,
            format!(
                "jsp: {}   jso: {}   jsd: {}",
                self.jsp(),
                self.jso(),
                self.jsd(),
            ),
        )?;
        column(
            f,
            format!(
                "inv: {}   osp: {}   osv: {}",
                self.op_stack.inv(),
                self.op_stack.osp(),
                self.op_stack.osv(),
            ),
        )?;
        column(
            f,
            format!(
                "st0: {} st1: {} st2: {} st3: {} st4: {} st5: {} st6: {} st7: {}",
                self.op_stack.st(ST0),
                self.op_stack.st(ST1),
                self.op_stack.st(ST2),
                self.op_stack.st(ST3),
                self.op_stack.st(ST4),
                self.op_stack.st(ST5),
                self.op_stack.st(ST6),
                self.op_stack.st(ST7),
            ),
        )?;

        horizontal_bar(f)?;

        Ok(())
    }
}

fn horizontal_bar(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(
        f,
        "+--------------------------------------------------------------------------+"
    )?;

    Ok(())
}

fn column(f: &mut std::fmt::Formatter<'_>, s: String) -> std::fmt::Result {
    writeln!(f, "| {: <72} |", s)
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
            println!("{}", state);
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

        println!("{}", last_state);
    }

    #[test]
    fn run_countdown_from_10_test() {
        let code = sample_programs::COUNTDOWN_FROM_10;
        let program = instruction::parse(code).unwrap();
        let trace = triton::vm::run(&program).unwrap();

        // println!("{}", program);
        // for state in trace {
        //     println!("{}", state);
        // }

        let last_state = trace.last().unwrap();

        assert_eq!(BWord::ring_zero(), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_fib() {
        let code = sample_programs::FIBONACCI;
        let program = instruction::parse(code).unwrap();
        let trace = triton::vm::run(&program).unwrap();

        let t = trace.clone();
        println!("{}", program);
        for state in trace {
            println!("{}", state);
        }

        let last_state = t.last().unwrap();

        assert_eq!(BWord::ring_zero(), last_state.op_stack.st(ST0));
    }
}
