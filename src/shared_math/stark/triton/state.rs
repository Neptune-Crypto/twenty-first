use super::error::{vm_fail, InstructionError::*};
use super::instruction::{Instruction, Instruction::*};
use super::op_stack::OpStack;
use super::ord_n::{Ord4::*, Ord6::*, Ord8::*};
use super::stdio::{InputStream, OutputStream};
use super::table::instruction_table;
use super::table::processor_table;
use super::vm::Program;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::rescue_prime_xlix::RescuePrimeXlix;
use crate::shared_math::stark::triton::error::vm_err;
use crate::shared_math::traits::{GetRandomElements, IdentityValues, Inverse};
use crate::shared_math::x_field_element::XFieldElement;
use rand::Rng;
use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::fmt::Display;

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
    /// Since `program` is read-only across individual states, and multiple
    /// inner helper functions refer to it, a read-only reference is kept in
    /// the struct.
    pub fn new(program: &'pgm Program) -> Self {
        let program = &program.instructions;
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
    pub fn step<R, In>(
        &self,
        rng: &mut R,
        stdin: &mut In,
        rescue_prime: &RescuePrimeXlix<AUX_REGISTER_COUNT>,
    ) -> Result<(VMState<'pgm>, Option<BFieldElement>), Box<dyn Error>>
    where
        R: Rng,
        In: InputStream,
    {
        let mut next_state = self.clone();
        let written_word = next_state.step_mut(rng, stdin, rescue_prime)?;
        Ok((next_state, written_word))
    }

    /// Perform the state transition as a mutable operation on `self`.
    ///
    /// This function is called from `step`.
    pub fn step_mut<R, In>(
        &mut self,
        rng: &mut R,
        stdin: &mut In,
        rescue_prime: &RescuePrimeXlix<AUX_REGISTER_COUNT>,
    ) -> Result<Option<BFieldElement>, Box<dyn Error>>
    where
        R: Rng,
        In: InputStream,
    {
        // All instructions increase the cycle count
        self.cycle_count += 1;
        let mut written_word = None;

        let instruction = self.current_instruction()?;
        match instruction {
            Pop => {
                self.op_stack.pop()?;
                self.instruction_pointer += 1;
            }

            Push(arg) => {
                self.op_stack.push(arg);
                self.instruction_pointer += 2;
            }

            Pad => {
                let elem = BWord::random_elements(1, rng)[0];
                self.op_stack.push(elem);
                self.instruction_pointer += 1;
            }

            Dup(arg) => {
                let elem = self.op_stack.safe_peek(arg.into());
                self.op_stack.push(elem);
                self.instruction_pointer += 2;
            }

            Swap(arg) => {
                // st[0] ... st[n] -> st[n] ... st[0]
                self.op_stack.safe_swap(arg.into());
                self.instruction_pointer += 2;
            }

            Skiz => {
                let elem = self.op_stack.pop()?;
                self.instruction_pointer += if elem.is_zero() {
                    let next_instruction = self.next_instruction()?;
                    1 + next_instruction.size()
                } else {
                    1
                }
            }

            Call(addr) => {
                let o_plus_2 = self.instruction_pointer as u32 + 2;
                let pair = (o_plus_2.into(), addr);
                self.jump_stack.push(pair);
                self.instruction_pointer = addr.value() as usize;
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
                if b < a {
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
                let d: u32 = self.op_stack.pop()?.try_into()?;
                let n: u32 = self.op_stack.pop()?.try_into()?;
                let (quot, rem) = other::div_rem(n, d);
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
                written_word = Some(self.op_stack.pop()?);
                self.instruction_pointer += 1;
            }

            ReadIo => {
                let in_char: u32 = stdin.read_u32_be()?;
                self.op_stack.push(in_char.into());
                self.instruction_pointer += 1;
            }
        }

        // Check that no instruction left the OpStack with too few elements
        if self.op_stack.is_too_shallow() {
            return vm_err(OpStackTooShallow);
        }

        Ok(written_word)
    }

    pub fn to_instruction_arr(
        &self,
    ) -> Result<[BFieldElement; instruction_table::BASE_WIDTH], Box<dyn Error>> {
        let current_instruction = self.current_instruction()?;

        let ip = (self.instruction_pointer as u32).try_into().unwrap();
        let ci = current_instruction.opcode_b();
        let nia = current_instruction
            .arg()
            .unwrap_or(self.next_instruction()?.opcode_b());

        Ok([ip, ci, nia])
    }

    pub fn to_processor_arr(
        &self,
    ) -> Result<[BFieldElement; processor_table::BASE_WIDTH], Box<dyn Error>> {
        let current_instruction = self.current_instruction()?;

        let clk = self.cycle_count.into();
        let ip = (self.instruction_pointer as u32).try_into().unwrap();
        let ci = current_instruction.opcode_b();
        let nia = current_instruction
            .arg()
            .unwrap_or(self.next_instruction()?.opcode_b());
        let ib0 = current_instruction.ib(IB0);
        let ib1 = current_instruction.ib(IB1);
        let ib2 = current_instruction.ib(IB2);
        let ib3 = current_instruction.ib(IB3);
        let ib4 = current_instruction.ib(IB4);
        let ib5 = current_instruction.ib(IB5);
        let st0 = self.op_stack.st(ST0);
        let st1 = self.op_stack.st(ST1);
        let st2 = self.op_stack.st(ST2);
        let st3 = self.op_stack.st(ST3);
        let st4 = self.op_stack.st(ST4);
        let st5 = self.op_stack.st(ST5);
        let st6 = self.op_stack.st(ST6);
        let st7 = self.op_stack.st(ST7);
        let inv = self.op_stack.inv();
        let osp = self.op_stack.osp();
        let osv = self.op_stack.osv();
        let hv0 = current_instruction.hv(N0);
        let hv1 = current_instruction.hv(N1);
        let hv2 = current_instruction.hv(N2);
        let hv3 = current_instruction.hv(N3);

        Ok([
            clk,
            ip,
            ci,
            nia,
            ib0,
            ib1,
            ib2,
            ib3,
            ib4,
            ib5,
            self.jsp(),
            self.jso(),
            self.jsd(),
            st0,
            st1,
            st2,
            st3,
            st4,
            st5,
            st6,
            st7,
            inv,
            osp,
            osv,
            hv0,
            hv1,
            hv2,
            hv3,
            self.ramp,
            self.ramv,
            self.aux[0],
            self.aux[1],
            self.aux[2],
            self.aux[3],
            self.aux[4],
            self.aux[5],
            self.aux[6],
            self.aux[7],
            self.aux[8],
            self.aux[9],
            self.aux[10],
            self.aux[11],
            self.aux[12],
            self.aux[13],
            self.aux[14],
            self.aux[15],
        ])
    }

    /// Jump-stack pointer
    fn jsp(&self) -> BWord {
        let height = self.jump_stack.len();
        if height == 0 {
            0.into()
        } else {
            BWord::new((height - 1) as u64)
        }
    }

    /// Jump-stack origin
    fn jso(&self) -> BWord {
        self.jump_stack
            .last()
            .map(|(o, _d)| *o)
            .unwrap_or_else(|| 0.into())
    }

    /// Jump-stack destination
    fn jsd(&self) -> BWord {
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

    // FIXME: Use instruction.arg() instead; see .to_arr()'s `let nia = ...`.
    fn ci_plus_1(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer + 1)
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

    pub fn get_readio_arg(&self) -> Result<Option<BFieldElement>, Box<dyn Error>> {
        let current_instruction = self.current_instruction()?;
        if matches!(current_instruction, ReadIo) {
            Ok(Some(self.op_stack.safe_peek(ST0)))
        } else {
            Ok(None)
        }
    }
}

impl<'pgm> Display for VMState<'pgm> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        horizontal_bar(f)?;

        let ci = self
            .current_instruction()
            .map(|ni| ni.to_string())
            .unwrap_or_else(|_| "eof".to_string());

        let ni = self
            .next_instruction()
            .map(|w| w.to_string())
            .unwrap_or_else(|_| "none".to_string());

        let nni = self
            .next_next_instruction()
            .map(|w| w.to_string())
            .unwrap_or_else(|_| "none".to_string());

        let ci_plus_1 = self
            .ci_plus_1()
            .map(|ni| ni.to_string())
            .unwrap_or_else(|_| "none".to_string());

        let width = 15;
        column(
            f,
            format!(
                "clk:  {:>width$} | ip:   {:>width$} |",
                self.cycle_count, self.instruction_pointer
            ),
        )?;

        column(
            f,
            format!("ci:   {:>width$} | ci+1: {:>width$} |", ci, ci_plus_1),
        )?;
        column(f, format!("ni:   {:>width$} | nni:  {:>width$} |", ni, nni))?;
        column(
            f,
            format!(
                "ramp: {:>width$} | ramv: {:>width$} |",
                self.ramp.value(),
                self.ramv.value()
            ),
        )?;
        let width_inv = width + 4;
        column(
            f,
            format!(
                "jsp:  {:>width$} | jso:  {:>width$} | jsd: {:>width_inv$}",
                self.jsp().value(),
                self.jso().value(),
                self.jsd().value()
            ),
        )?;
        column(
            f,
            format!(
                "osp:  {:>width$} | osv:  {:>width$} | inv: {:>width_inv$}",
                self.op_stack.osp().value(),
                self.op_stack.osv().value(),
                self.op_stack.inv().value()
            ),
        )?;

        let width_st = 5;
        column(
            f,
            format!(
                "st7-0: [ {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} ]",
                self.op_stack.st(ST7).value(),
                self.op_stack.st(ST6).value(),
                self.op_stack.st(ST5).value(),
                self.op_stack.st(ST4).value(),
                self.op_stack.st(ST3).value(),
                self.op_stack.st(ST2).value(),
                self.op_stack.st(ST1).value(),
                self.op_stack.st(ST0).value(),
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

    #[test]
    fn run_parse_pop_p() {
        let program = sample_programs::push_push_add_pop_p();
        let (trace, _out, _err) = program.run_with_input(&[]);

        for state in trace.iter() {
            println!("{}", state);
        }
    }

    #[test]
    fn run_hello_world_1() {
        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[]);

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::ring_zero(), last_state.op_stack.safe_peek(ST0));

        println!("{}", last_state);
    }

    #[test]
    fn run_countdown_from_10_test() {
        let code = sample_programs::COUNTDOWN_FROM_10;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::ring_zero(), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_fibonacci_vit() {
        let code = sample_programs::FIBONACCI_VIT;
        let program = Program::from_code(code).unwrap();

        let (trace, _out, _err) = program.run_with_input(&[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::new(21), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_fibonacci_lq() {
        let code = sample_programs::FIBONACCI_LT;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::new(21), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_gcd() {
        let code = sample_programs::GCD_42_56;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, _out, _err) = program.run_with_input(&[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::new(14), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_xgcd() {
        let code = sample_programs::XGCD;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, _out, _err) = program.run_with_input(&[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        //assert_eq!(BWord::new(14), last_state.op_stack.st(ST0));
    }
}
