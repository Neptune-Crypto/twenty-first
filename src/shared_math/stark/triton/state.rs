use super::error::{vm_fail, InstructionError::*};
use super::instruction::{AnInstruction::*, Instruction};
use super::op_stack::OpStack;
use super::ord_n::{Ord6::*, Ord8::*};
use super::stdio::InputStream;
use super::table::{aux_table, instruction_table, jump_stack_table, op_stack_table, u32_op_table};
use super::table::{processor_table, ram_table};
use super::vm::Program;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::rescue_prime_xlix::RescuePrimeXlix;
use crate::shared_math::stark::triton::error::vm_err;
use crate::shared_math::stark::triton::table::base_matrix::ProcessorMatrixRow;
use crate::shared_math::traits::{IdentityValues, Inverse};
use crate::shared_math::x_field_element::XFieldElement;
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

/// The number of helper variable registers
pub const HV_REGISTER_COUNT: usize = 5;

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

    /// Helper Variable Registers
    pub hv: [BWord; HV_REGISTER_COUNT],
}

#[derive(Debug, PartialEq)]
pub enum VMOutput {
    /// Trace output from `write_io`
    WriteIoTrace(BWord),

    /// Trace of auxiliary registers for hash coprocessor table
    ///
    /// One row per round in the XLIX permutation
    XlixTrace(Vec<[BWord; aux_table::BASE_WIDTH]>),

    /// Trace of u32 operations for u32 op table
    ///
    /// One row per defined bit
    U32OpTrace(Vec<[BWord; u32_op_table::BASE_WIDTH]>),
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
    pub fn is_complete(&self) -> bool {
        self.program.len() <= self.instruction_pointer
    }

    /// Given a state, compute `(next_state, vm_output)`.
    pub fn step<In>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
        rescue_prime: &RescuePrimeXlix<AUX_REGISTER_COUNT>,
    ) -> Result<(VMState<'pgm>, Option<VMOutput>), Box<dyn Error>>
    where
        In: InputStream,
    {
        let mut next_state = self.clone();
        next_state
            .step_mut(stdin, secret_in, rescue_prime)
            .map(|vm_output| (next_state, vm_output))
    }

    /// Perform the state transition as a mutable operation on `self`.
    pub fn step_mut<In>(
        &mut self,
        stdin: &mut In,
        secret_in: &mut In,
        rescue_prime: &RescuePrimeXlix<AUX_REGISTER_COUNT>,
    ) -> Result<Option<VMOutput>, Box<dyn Error>>
    where
        In: InputStream,
    {
        // All instructions increase the cycle count
        self.cycle_count += 1;
        let mut vm_output = None;

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

            Divine => {
                let elem = secret_in.read_elem()?;
                self.op_stack.push(elem);
                self.instruction_pointer += 1;
            }

            Dup(arg) => {
                let elem = self.op_stack.safe_peek(arg);
                self.op_stack.push(elem);
                self.instruction_pointer += 2;
            }

            Swap(arg) => {
                // st[0] ... st[n] -> st[n] ... st[0]
                self.op_stack.safe_swap(arg);
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
                let aux_trace = rescue_prime.rescue_xlix_permutation_trace(&mut self.aux);
                vm_output = Some(VMOutput::XlixTrace(aux_trace));
                self.instruction_pointer += 1;
            }

            ClearAll => {
                self.aux.fill(0.into());
                self.instruction_pointer += 1;
            }

            Squeeze(arg) => {
                let n: usize = arg.into();
                self.op_stack.push(self.aux[n]);
                self.instruction_pointer += 2;
            }

            Absorb(arg) => {
                let n: usize = arg.into();
                let elem = self.op_stack.pop()?;
                self.aux[n] = elem;
                self.instruction_pointer += 2;
            }

            DivineSibling => {
                let node_index: u32 = self.op_stack.pop()?.try_into()?;
                let new_node_index = self.divine_sibling::<In>(node_index, secret_in)?;
                self.op_stack.push(new_node_index);
                self.instruction_pointer += 1;
            }

            AssertDigest => {
                let cmp_bword = self.assert_digest();
                if !cmp_bword.is_one() {
                    return vm_err(AssertionFailed);
                }
                self.instruction_pointer += 1;
            }

            Add => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(lhs + rhs);
                self.instruction_pointer += 1;
            }

            Mul => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(lhs * rhs);
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
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(Self::eq(lhs, rhs));
                self.instruction_pointer += 1;
            }

            Lt => {
                let lhs: u32 = self.op_stack.pop()?.try_into()?;
                let rhs: u32 = self.op_stack.pop()?.try_into()?;
                self.op_stack.push(Self::lt(lhs, rhs));
                let trace = self.u32_op_trace(lhs, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            And => {
                let lhs: u32 = self.op_stack.pop()?.try_into()?;
                let rhs: u32 = self.op_stack.pop()?.try_into()?;
                self.op_stack.push((lhs & rhs).into());
                let trace = self.u32_op_trace(lhs, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Xor => {
                let lhs: u32 = self.op_stack.pop()?.try_into()?;
                let rhs: u32 = self.op_stack.pop()?.try_into()?;
                self.op_stack.push((lhs ^ rhs).into());
                let trace = self.u32_op_trace(lhs, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Reverse => {
                let elem: u32 = self.op_stack.pop()?.try_into()?;

                // `rev` needs to constrain that the second-top-most element
                // fits within u32, since otherwise the u32 op table constraint
                // polynomials cannot account for the 'rhs' column going to 0 in
                // at most 32 steps (rows).
                //
                // So while `rev` is a unary instruction (does not have a RHS),
                // it still has a rule about the second-top-most element.
                let rhs: u32 = self.op_stack.safe_peek(ST0).try_into()?;

                self.op_stack.push(elem.reverse_bits().into());
                let trace = self.u32_op_trace(elem, rhs);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            Div => {
                let denom: u32 = self.op_stack.pop()?.try_into()?;
                let numerator: u32 = self.op_stack.pop()?.try_into()?;
                let (quot, rem) = other::div_rem(numerator, denom);
                self.op_stack.push(quot.into());
                self.op_stack.push(rem.into());
                let trace = self.u32_op_trace(denom, numerator);
                vm_output = Some(VMOutput::U32OpTrace(trace));
                self.instruction_pointer += 1;
            }

            XxAdd => {
                let lhs: XWord = self.op_stack.popx()?;
                let rhs: XWord = self.op_stack.popx()?;
                self.op_stack.pushx(lhs + rhs);
                self.instruction_pointer += 1;
            }

            XxMul => {
                let lhs: XWord = self.op_stack.popx()?;
                let rhs: XWord = self.op_stack.popx()?;
                self.op_stack.pushx(lhs + rhs);
                self.instruction_pointer += 1;
            }

            XInv => {
                let elem: XWord = self.op_stack.popx()?;
                self.op_stack.pushx(elem.inverse());
                self.instruction_pointer += 1;
            }

            XbMul => {
                let lhs: XWord = self.op_stack.popx()?;
                let rhs: BWord = self.op_stack.pop()?;
                self.op_stack.pushx(lhs * rhs.lift());
                self.instruction_pointer += 1;
            }

            WriteIo => {
                vm_output = Some(VMOutput::WriteIoTrace(self.op_stack.pop()?));
                self.instruction_pointer += 1;
            }

            ReadIo => {
                let in_elem = stdin.read_elem()?;
                self.op_stack.push(in_elem);
                self.instruction_pointer += 1;
            }
        }

        // Check that no instruction left the OpStack with too few elements
        if self.op_stack.is_too_shallow() {
            return vm_err(OpStackTooShallow);
        }

        Ok(vm_output)
    }

    pub fn to_instruction_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; instruction_table::BASE_WIDTH] {
        let ip = (self.instruction_pointer as u32).try_into().unwrap();
        let ci = current_instruction.opcode_b();
        let nia = self.nia();

        [ip, ci, nia]
    }

    pub fn to_processor_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; processor_table::BASE_WIDTH] {
        let clk = self.cycle_count.into();
        let ip = (self.instruction_pointer as u32).try_into().unwrap();
        let ci = current_instruction.opcode_b();
        let nia = self.nia();
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

        [
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
            self.hv[0],
            self.hv[1],
            self.hv[2],
            self.hv[3],
            self.hv[4],
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
        ]
    }

    pub fn to_op_stack_row(
        &self,
        current_instruction: Instruction,
    ) -> [BFieldElement; op_stack_table::BASE_WIDTH] {
        let clk = self.cycle_count.into();
        let ci = current_instruction.opcode_b();
        let osp = self.op_stack.osp();
        let osv = self.op_stack.osv();

        // clk, ci, osv, osp
        [clk, ci, osv, osp]
    }

    pub fn to_ram_row(&self) -> [BFieldElement; ram_table::BASE_WIDTH] {
        let clk = self.cycle_count.into();

        [clk, self.ramp, self.ramv]
    }

    pub fn to_jump_stack_row(&self) -> [BFieldElement; jump_stack_table::BASE_WIDTH] {
        let clk = self.cycle_count.into();

        [clk, self.jsp(), self.jso(), self.jsd()]
    }

    pub fn u32_op_trace(
        &self,
        mut lhs: u32,
        mut rhs: u32,
    ) -> Vec<[BFieldElement; u32_op_table::BASE_WIDTH]> {
        let mut rows = vec![];
        let mut idc = 1.into();
        let zero = 0.into();

        while lhs > 0 || rhs > 0 {
            let row = [
                idc,
                lhs.into(),
                rhs.into(),
                Self::lt(lhs, rhs),
                (lhs & rhs).into(),
                (lhs ^ rhs).into(),
                lhs.reverse_bits().into(),
            ];
            rows.push(row);
            lhs >>= 1;
            rhs >>= 1;
            idc = zero;
        }

        rows
    }

    fn lt(lhs: u32, rhs: u32) -> BWord {
        if rhs < lhs {
            1.into()
        } else {
            0.into()
        }
    }

    fn eq(lhs: BWord, rhs: BWord) -> BWord {
        if lhs == rhs {
            1.into()
        } else {
            0.into()
        }
    }

    fn nia(&self) -> BWord {
        self.current_instruction()
            .map(|instr| {
                instr.arg().unwrap_or_else(|| {
                    self.next_instruction()
                        .map(|instr| instr.opcode_b())
                        .unwrap_or_else(|_| 0.into())
                })
            })
            .unwrap_or_else(|_| 0.into())
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

    pub fn current_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        self.program
            .get(self.instruction_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(self.instruction_pointer)))
            .copied()
    }

    // Return the next instruction on the tape, skipping arguments
    //
    // Note that this is not necessarily the next instruction to execute,
    // since the current instruction could be a jump, but it is either
    // program[ip + 1] or program[ip + 2] depending on whether the current
    // instruction takes an argument or not.
    pub fn next_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        let ci = self.current_instruction()?;
        let ci_size = ci.size();
        let ni_pointer = self.instruction_pointer + ci_size;
        println!("ci: {}, ci_size: {}, ni_ptr: {}", ci, ci_size, ni_pointer);
        self.program
            .get(ni_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(ni_pointer)))
            .copied()
    }

    fn _next_next_instruction(&self) -> Result<Instruction, Box<dyn Error>> {
        let cur_size = self.current_instruction()?.size();
        let next_size = self.next_instruction()?.size();
        self.program
            .get(self.instruction_pointer + cur_size + next_size)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(self.instruction_pointer)))
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

    fn assert_digest(&self) -> BWord {
        for i in 0..DIGEST_LEN {
            // Safe as long as DIGEST_LEN <= OP_STACK_REG_COUNT
            if self.aux[i] != self.op_stack.safe_peek(i.try_into().unwrap()) {
                return 0.into();
            }
        }
        1.into()
    }

    pub fn read_word(&self) -> Result<Option<BFieldElement>, Box<dyn Error>> {
        let current_instruction = self.current_instruction()?;
        if matches!(current_instruction, ReadIo) {
            Ok(Some(self.op_stack.safe_peek(ST0)))
        } else {
            Ok(None)
        }
    }
    fn divine_sibling<In: InputStream>(
        &mut self,
        node_index: u32,
        secret_in: &mut In,
    ) -> Result<BWord, Box<dyn Error>> {
        let known_digest: [BWord; 6] = self.aux[0..6].try_into().unwrap();
        let sibling_digest = [
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
            secret_in.read_elem()?,
        ];
        // lsb = least significant bit
        let node_index_lsb = node_index % 2;
        let node_index_msbs = node_index / 2;

        let is_left_node = node_index_lsb == 0;
        if is_left_node {
            // no need to change lhs
            // move sibling digest to rhs
            self.aux[6..12].copy_from_slice(&sibling_digest);
        } else {
            // move sibling digest to lhs
            self.aux[0..6].copy_from_slice(&sibling_digest);
            // move lhs to rhs
            self.aux[6..12].copy_from_slice(&known_digest);
        }
        // set capacity to 0
        self.aux[12..16].copy_from_slice(&[0.into(); 4]);

        // set hv registers to correct decomposition of node_index
        self.hv[0] = node_index_lsb.into();
        self.hv[1] = node_index_msbs.into();

        Ok(node_index_msbs.into())
    }
}

impl<'pgm> Display for VMState<'pgm> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = self.current_instruction().map(|instruction| {
            write!(
                f,
                "{}",
                ProcessorMatrixRow {
                    row: self.to_processor_row(instruction)
                }
            )
        });
        res.unwrap_or_else(|_| write!(f, "END-OF-FILE"))
    }
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
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
    }

    #[test]
    fn run_hello_world_1() {
        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::ring_zero(), last_state.op_stack.safe_peek(ST0));

        println!("{}", last_state);
    }

    #[test]
    fn run_mt_ap_verify_test() {
        let code = sample_programs::MT_AP_VERIFY;
        let program = Program::from_code(code).unwrap();
        println!("Successfully parsed the program.");
        let (trace, _out, err) = program.run_with_input(
            &[
                // Merkle root
                BWord::new(17520959918188528334),
                BWord::new(15041448862488693843),
                BWord::new(298116966047567369),
                BWord::new(15868284389682039986),
                BWord::new(10576343946532188879),
                BWord::new(9153918202384051329),
                // leaf's index
                BWord::new(92),
                // leaf's value
                BWord::new(45),
                BWord::new(50),
                BWord::new(47),
            ],
            &[
                // Merkle Authentication Path Element 1
                BWord::new(6083984126818143390),
                BWord::new(3068114068019721586),
                BWord::new(3759135683318231675),
                BWord::new(12661293681732607010),
                BWord::new(6748738404164062279),
                BWord::new(9498241828820249207),
                // Merkle Authentication Path Element 2
                BWord::new(8915238764673447613),
                BWord::new(942439158432159996),
                BWord::new(312764170689326023),
                BWord::new(10945419959481343904),
                BWord::new(5750200734225507788),
                BWord::new(3793111105236268823),
                // Merkle Authentication Path Element 3
                BWord::new(14019449591007199492),
                BWord::new(18357587908965733025),
                BWord::new(12465922695385012477),
                BWord::new(1517140721628804219),
                BWord::new(9611960197719015309),
                BWord::new(9776705825929245273),
                // Merkle Authentication Path Element 4
                BWord::new(7309515647084889872),
                BWord::new(9712533577403755189),
                BWord::new(4921865008647832542),
                BWord::new(4769370959411372374),
                BWord::new(14537750035888652552),
                BWord::new(13532396896348551998),
                // Merkle Authentication Path Element 5
                BWord::new(16902405264740278807),
                BWord::new(14918340102437258285),
                BWord::new(979815985758098826),
                BWord::new(17118084172379918870),
                BWord::new(12824459547005533540),
                BWord::new(16968722063561851448),
                // Merkle Authentication Path Element 6
                BWord::new(699976577639234824),
                BWord::new(13059558942272293990),
                BWord::new(15739587478100963457),
                BWord::new(11329100596238735474),
                BWord::new(11433851170242101939),
                BWord::new(648656172379535759),
            ],
        );

        for state in trace.iter() {
            println!("{}", state);
        }
        if let Some(e) = err {
            println!("Error: {}", e);
        }

        let last_state = trace.last().unwrap();
        // todo check that VM terminated gracefully
        // assert_eq!(last_state.current_instruction().unwrap(), Halt);

        println!("{}", last_state);
    }

    #[test]
    fn run_countdown_from_10_test() {
        let code = sample_programs::COUNTDOWN_FROM_10;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

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

        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::new(21), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_fibonacci_lt() {
        let code = sample_programs::FIBONACCI_LT;
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let last_state = trace.last().unwrap();
        assert_eq!(BWord::new(21), last_state.op_stack.st(ST0));
    }

    #[test]
    fn run_gcd() {
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, out, _err) = program.run_with_input(&[42.into(), 56.into()], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let expected = BFieldElement::new(14);
        let actual = *out.last().unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn run_xgcd() {
        // The XGCD program is work in progress.
        let code = sample_programs::XGCD;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        println!("{}", program);
        for state in trace.iter() {
            println!("{}", state);
        }

        let _last_state = trace.last().unwrap();

        let _expected = BFieldElement::new(14);
        let _actual = _last_state.op_stack.st(ST0);

        //assert_eq!(expected, actual);
    }

    #[test]
    fn swap_test() {
        let code = "push 1 push 2 swap1 halt";
        let program = Program::from_code(code).unwrap();
        let (trace, _out, _err) = program.run_with_input(&[], &[]);

        for state in trace.iter() {
            println!("{}", state);
        }
    }
}
