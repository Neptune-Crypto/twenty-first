use super::{
    hash_table, instruction_table, io_table, jump_stack_table, op_stack_table, processor_table,
    program_table, ram_table, u32_op_table,
};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::triton::instruction::Instruction;
use crate::shared_math::stark::triton::state::{VMOutput, VMState};
use crate::shared_math::stark::triton::table::base_matrix::ProcessorTableColumn::*;
use crate::shared_math::stark::triton::vm::Program;
use std::fmt::Display;

#[derive(Debug, Clone, Default)]
pub struct BaseMatrices {
    pub program_matrix: Vec<[BFieldElement; program_table::BASE_WIDTH]>,
    pub processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
    pub instruction_matrix: Vec<[BFieldElement; instruction_table::BASE_WIDTH]>,
    pub input_matrix: Vec<[BFieldElement; io_table::BASE_WIDTH]>,
    pub output_matrix: Vec<[BFieldElement; io_table::BASE_WIDTH]>,
    pub op_stack_matrix: Vec<[BFieldElement; op_stack_table::BASE_WIDTH]>,
    pub ram_matrix: Vec<[BFieldElement; ram_table::BASE_WIDTH]>,
    pub jump_stack_matrix: Vec<[BFieldElement; jump_stack_table::BASE_WIDTH]>,
    pub hash_matrix: Vec<[BFieldElement; hash_table::BASE_WIDTH]>,
    pub u32_op_matrix: Vec<[BFieldElement; u32_op_table::BASE_WIDTH]>,
}

impl BaseMatrices {
    /// Initialize `program_matrix` and `instruction_matrix` so that both contain one row per word
    /// in the program. Note that this does not mean “one row per instruction:” instructions that
    /// take two words (e.g. `push N`) add two rows.
    pub fn initialize(&mut self, program: &Program) {
        let words = program.to_bwords().into_iter();
        let mut words_with_0 = program.to_bwords();
        words_with_0.push(0.into());
        let next_words = words_with_0.into_iter().skip(1);
        debug_assert_eq!(words.len(), next_words.len());

        for (i, (word, next_word)) in words.zip(next_words).enumerate() {
            let index = (i as u32).into();
            self.program_matrix.push([index, word]);
            self.instruction_matrix.push([index, word, next_word]);
        }
    }

    pub fn sort_instruction_matrix(&mut self) {
        self.instruction_matrix
            .sort_by_key(|row| row[InstructionTableColumn::Address as usize].value());
    }

    pub fn sort_op_stack_matrix(&mut self) {
        self.op_stack_matrix.sort_by_key(|row| {
            (
                row[OpStackTableColumn::OSP as usize].value(),
                row[OpStackTableColumn::CLK as usize].value(),
            )
        })
    }

    pub fn sort_ram_matrix(&mut self) {
        self.ram_matrix.sort_by_key(|row| {
            (
                row[RAMTableColumn::RAMP as usize].value(),
                row[RAMTableColumn::CLK as usize].value(),
            )
        })
    }

    pub fn sort_jump_stack_matrix(&mut self) {
        self.jump_stack_matrix.sort_by_key(|row| {
            (
                row[JumpStackTableColumn::JSP as usize].value(),
                row[JumpStackTableColumn::CLK as usize].value(),
            )
        })
    }

    pub fn append(
        &mut self,
        state: &VMState,
        vm_output: Option<VMOutput>,
        current_instruction: Instruction,
    ) {
        self.processor_matrix
            .push(state.to_processor_row(current_instruction));

        self.instruction_matrix
            .push(state.to_instruction_row(current_instruction));

        self.op_stack_matrix
            .push(state.to_op_stack_row(current_instruction));

        self.ram_matrix.push(state.to_ram_row());

        self.jump_stack_matrix
            .push(state.to_jump_stack_row(current_instruction));

        if let Ok(Some(word)) = state.read_word() {
            self.input_matrix.push([word])
        }

        match vm_output {
            Some(VMOutput::WriteIoTrace(written_word)) => self.output_matrix.push([written_word]),
            Some(VMOutput::XlixTrace(mut aux_trace)) => self.hash_matrix.append(&mut aux_trace),
            Some(VMOutput::U32OpTrace(mut trace)) => self.u32_op_matrix.append(&mut trace),
            None => (),
        }
    }
}

pub struct ProcessorMatrix {
    pub row: [BFieldElement; processor_table::BASE_WIDTH],
}

pub enum ProcessorTableColumn {
    CLK,
    IP,
    CI,
    NIA,
    IB0,
    IB1,
    IB2,
    IB3,
    IB4,
    IB5,
    JSP,
    JSO,
    JSD,
    ST0,
    ST1,
    ST2,
    ST3,
    ST4,
    ST5,
    ST6,
    ST7,
    ST8,
    ST9,
    ST10,
    ST11,
    ST12,
    ST13,
    ST14,
    ST15,
    INV,
    OSP,
    OSV,
    HV0,
    HV1,
    HV2,
    HV3,
    HV4,
    RAMP,
    RAMV,
}

impl From<ProcessorTableColumn> for usize {
    fn from(c: ProcessorTableColumn) -> Self {
        match c {
            ProcessorTableColumn::CLK => 0,
            ProcessorTableColumn::IP => 1,
            ProcessorTableColumn::CI => 2,
            ProcessorTableColumn::NIA => 3,
            ProcessorTableColumn::IB0 => 4,
            ProcessorTableColumn::IB1 => 5,
            ProcessorTableColumn::IB2 => 6,
            ProcessorTableColumn::IB3 => 7,
            ProcessorTableColumn::IB4 => 8,
            ProcessorTableColumn::IB5 => 9,
            ProcessorTableColumn::JSP => 10,
            ProcessorTableColumn::JSO => 11,
            ProcessorTableColumn::JSD => 12,
            ProcessorTableColumn::ST0 => 13,
            ProcessorTableColumn::ST1 => 14,
            ProcessorTableColumn::ST2 => 15,
            ProcessorTableColumn::ST3 => 16,
            ProcessorTableColumn::ST4 => 17,
            ProcessorTableColumn::ST5 => 18,
            ProcessorTableColumn::ST6 => 19,
            ProcessorTableColumn::ST7 => 20,
            ProcessorTableColumn::ST8 => 21,
            ProcessorTableColumn::ST9 => 22,
            ProcessorTableColumn::ST10 => 23,
            ProcessorTableColumn::ST11 => 24,
            ProcessorTableColumn::ST12 => 25,
            ProcessorTableColumn::ST13 => 26,
            ProcessorTableColumn::ST14 => 27,
            ProcessorTableColumn::ST15 => 28,
            ProcessorTableColumn::INV => 29,
            ProcessorTableColumn::OSP => 30,
            ProcessorTableColumn::OSV => 31,
            ProcessorTableColumn::HV0 => 32,
            ProcessorTableColumn::HV1 => 33,
            ProcessorTableColumn::HV2 => 34,
            ProcessorTableColumn::HV3 => 35,
            ProcessorTableColumn::HV4 => 36,
            ProcessorTableColumn::RAMP => 37,
            ProcessorTableColumn::RAMV => 38,
        }
    }
}

pub enum IOTableColumn {
    IOSymbol,
}

impl From<IOTableColumn> for usize {
    fn from(c: IOTableColumn) -> Self {
        match c {
            IOTableColumn::IOSymbol => 0,
        }
    }
}

pub enum ProgramTableColumn {
    Address,
    Instruction,
}

impl From<ProgramTableColumn> for usize {
    fn from(c: ProgramTableColumn) -> Self {
        match c {
            ProgramTableColumn::Address => 0,
            ProgramTableColumn::Instruction => 1,
        }
    }
}

pub enum InstructionTableColumn {
    Address,
    CI,
    NIA,
}

impl From<InstructionTableColumn> for usize {
    fn from(c: InstructionTableColumn) -> Self {
        match c {
            InstructionTableColumn::Address => 0,
            InstructionTableColumn::CI => 1,
            InstructionTableColumn::NIA => 2,
        }
    }
}

pub enum OpStackTableColumn {
    CLK,
    CI,
    OSV,
    OSP,
}

impl From<OpStackTableColumn> for usize {
    fn from(c: OpStackTableColumn) -> Self {
        match c {
            OpStackTableColumn::CLK => 0,
            OpStackTableColumn::CI => 1,
            OpStackTableColumn::OSV => 2,
            OpStackTableColumn::OSP => 3,
        }
    }
}

pub enum RAMTableColumn {
    CLK,
    RAMP,
    RAMV,
}

impl From<RAMTableColumn> for usize {
    fn from(c: RAMTableColumn) -> Self {
        match c {
            RAMTableColumn::CLK => 0,
            RAMTableColumn::RAMP => 1,
            RAMTableColumn::RAMV => 2,
        }
    }
}

pub enum JumpStackTableColumn {
    CLK,
    CI,
    JSP,
    JSO,
    JSD,
}

impl From<JumpStackTableColumn> for usize {
    fn from(c: JumpStackTableColumn) -> Self {
        match c {
            JumpStackTableColumn::CLK => 0,
            JumpStackTableColumn::CI => 1,
            JumpStackTableColumn::JSP => 2,
            JumpStackTableColumn::JSO => 3,
            JumpStackTableColumn::JSD => 4,
        }
    }
}

pub enum HashTableColumn {
    RoundNumber,
    AUX0,
    AUX1,
    AUX2,
    AUX3,
    AUX4,
    AUX5,
    AUX6,
    AUX7,
    AUX8,
    AUX9,
    AUX10,
    AUX11,
    AUX12,
    AUX13,
    AUX14,
    AUX15,
}

impl From<HashTableColumn> for usize {
    fn from(c: HashTableColumn) -> Self {
        match c {
            HashTableColumn::RoundNumber => 0,
            HashTableColumn::AUX0 => 1,
            HashTableColumn::AUX1 => 2,
            HashTableColumn::AUX2 => 3,
            HashTableColumn::AUX3 => 4,
            HashTableColumn::AUX4 => 5,
            HashTableColumn::AUX5 => 6,
            HashTableColumn::AUX6 => 7,
            HashTableColumn::AUX7 => 8,
            HashTableColumn::AUX8 => 9,
            HashTableColumn::AUX9 => 10,
            HashTableColumn::AUX10 => 11,
            HashTableColumn::AUX11 => 12,
            HashTableColumn::AUX12 => 13,
            HashTableColumn::AUX13 => 14,
            HashTableColumn::AUX14 => 15,
            HashTableColumn::AUX15 => 16,
        }
    }
}

pub enum U32OpTableColumn {
    IDC,
    LHS,
    RHS,
    LT,
    AND,
    XOR,
    REV,
}

impl From<U32OpTableColumn> for usize {
    fn from(c: U32OpTableColumn) -> Self {
        match c {
            U32OpTableColumn::IDC => 0,
            U32OpTableColumn::LHS => 1,
            U32OpTableColumn::RHS => 2,
            U32OpTableColumn::LT => 3,
            U32OpTableColumn::AND => 4,
            U32OpTableColumn::XOR => 5,
            U32OpTableColumn::REV => 6,
        }
    }
}

pub struct ProcessorMatrixRow {
    pub row: [BFieldElement; processor_table::BASE_WIDTH],
}

impl Display for ProcessorMatrixRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn row(f: &mut std::fmt::Formatter<'_>, s: String) -> std::fmt::Result {
            writeln!(f, "│ {: <103} │", s)
        }

        fn row_blank(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            row(f, "".into())
        }

        writeln!(
            f,
            "╭┴────────────────┴───────────────────────────────────────────────\
            ────────────────────┬───────────────────╮"
        )?;

        let width = 20;
        row(
            f,
            format!(
                "ip:   {:>width$} ╷ ci:   {:>width$} ╷ nia: {:>width$} │ {:>17}",
                self.row[IP as usize].value(),
                self.row[CI as usize].value(),
                self.row[NIA as usize].value(),
                self.row[CLK as usize].value(),
            ),
        )?;

        writeln!(
            f,
            "│ jsp:  {:>width$} │ jso:  {:>width$} │ jsd: {:>width$} ╰───────────────────┤",
            self.row[JSP as usize].value(),
            self.row[JSO as usize].value(),
            self.row[JSD as usize].value(),
        )?;
        row(
            f,
            format!(
                "ramp: {:>width$} │ ramv: {:>width$} │",
                self.row[RAMP as usize].value(),
                self.row[RAMV as usize].value(),
            ),
        )?;
        writeln!(
            f,
            "│ osp:  {:>width$} ╵ osv:  {:>width$} ╵                 \
            ╭─────────────────────────────┤",
            self.row[OSP as usize].value(),
            self.row[OSV as usize].value(),
        )?;
        writeln!(
            f,
            "│ {:>72} ╶╯ inv: {:>width$}   │",
            " ",
            self.row[INV as usize].value()
        )?;
        row(
            f,
            format!(
                "st3-0:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST3 as usize].value(),
                self.row[ST2 as usize].value(),
                self.row[ST1 as usize].value(),
                self.row[ST0 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st7-4:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST7 as usize].value(),
                self.row[ST6 as usize].value(),
                self.row[ST5 as usize].value(),
                self.row[ST4 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st11-8:   [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST11 as usize].value(),
                self.row[ST10 as usize].value(),
                self.row[ST9 as usize].value(),
                self.row[ST8 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st15-12:  [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST15 as usize].value(),
                self.row[ST14 as usize].value(),
                self.row[ST13 as usize].value(),
                self.row[ST12 as usize].value(),
            ),
        )?;

        row_blank(f)?;

        row(f, "".into())?;

        row(
            f,
            format!(
                "hv4-0: [ {:>16} | {:>16} | {:>16} | {:>16} | {:>16} ]",
                self.row[HV4 as usize].value(),
                self.row[HV3 as usize].value(),
                self.row[HV2 as usize].value(),
                self.row[HV1 as usize].value(),
                self.row[HV0 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "ib5-0: [ {:>12} | {:>13} | {:>13} | {:>13} | {:>13} | {:>13} ]",
                self.row[IB5 as usize].value(),
                self.row[IB4 as usize].value(),
                self.row[IB3 as usize].value(),
                self.row[IB2 as usize].value(),
                self.row[IB1 as usize].value(),
                self.row[IB0 as usize].value(),
            ),
        )?;
        writeln!(
            f,
            "╰─────────────────────────────────────────────────────────────────\
            ────────────────────────────────────────╯"
        )?;
        Ok(())
    }
}
