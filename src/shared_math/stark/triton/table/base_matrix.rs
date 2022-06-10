use super::{
    aux_table, instruction_table, io_table, jump_stack_table, op_stack_table, processor_table,
    program_table, ram_table, u32_op_table,
};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::triton::instruction::Instruction;
use crate::shared_math::stark::triton::state::{VMOutput, VMState};
use crate::shared_math::stark::triton::table::base_matrix::ProcessorTableColumn::*;
use crate::shared_math::stark::triton::vm::Program;
use std::fmt::Display;

type BWord = BFieldElement;

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
    pub aux_matrix: Vec<[BFieldElement; aux_table::BASE_WIDTH]>,
    pub u32_op_matrix: Vec<[BFieldElement; u32_op_table::BASE_WIDTH]>,
}

impl BaseMatrices {
    /// Initialize `program_matrix` and `instruction_matrix` so that both
    /// contain one row per word in the program. Instructions that take two
    /// words (e.g. `push N`) add two rows.
    pub fn initialize(&mut self, program: &Program) {
        let mut words = program.to_bwords().into_iter().enumerate();
        let (i, mut current_word) = words.next().unwrap();

        let index: BWord = (i as u32).into();
        self.program_matrix.push([index, current_word]);

        for (i, next_word) in words {
            let index: BWord = (i as u32).into();
            self.program_matrix.push([index, current_word]);
            self.instruction_matrix
                .push([index, current_word, next_word]);
            current_word = next_word;
        }
    }

    // FIXME: We have to name the row's fields here.
    //
    // row[0] corresponds to clk
    pub fn sort_instruction_matrix(&mut self) {
        self.instruction_matrix.sort_by_key(|row| row[0].value());
    }

    // FIXME: We have to name the row's fields here.
    //
    // row[1] corresponds to jsp
    // row[0] corresponds to clk
    pub fn sort_jump_stack_matrix(&mut self) {
        self.jump_stack_matrix
            .sort_by_key(|row| (row[1].value(), row[0].value()))
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

        self.ram_matrix.push(state.to_ram_row(current_instruction));

        self.jump_stack_matrix.push(state.to_jump_stack_row());

        if let Ok(Some(word)) = state.read_word() {
            self.input_matrix.push([word])
        }

        match vm_output {
            Some(VMOutput::WriteIoTrace(written_word)) => self.output_matrix.push([written_word]),
            Some(VMOutput::XlixTrace(mut aux_trace)) => self.aux_matrix.append(&mut aux_trace),
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
            ProcessorTableColumn::INV => 21,
            ProcessorTableColumn::OSP => 22,
            ProcessorTableColumn::OSV => 23,
            ProcessorTableColumn::HV0 => 24,
            ProcessorTableColumn::HV1 => 25,
            ProcessorTableColumn::HV2 => 26,
            ProcessorTableColumn::HV3 => 27,
            ProcessorTableColumn::HV4 => 28,
            ProcessorTableColumn::RAMP => 29,
            ProcessorTableColumn::RAMV => 30,
            ProcessorTableColumn::AUX0 => 31,
            ProcessorTableColumn::AUX1 => 32,
            ProcessorTableColumn::AUX2 => 33,
            ProcessorTableColumn::AUX3 => 34,
            ProcessorTableColumn::AUX4 => 35,
            ProcessorTableColumn::AUX5 => 36,
            ProcessorTableColumn::AUX6 => 37,
            ProcessorTableColumn::AUX7 => 38,
            ProcessorTableColumn::AUX8 => 39,
            ProcessorTableColumn::AUX9 => 40,
            ProcessorTableColumn::AUX10 => 41,
            ProcessorTableColumn::AUX11 => 42,
            ProcessorTableColumn::AUX12 => 43,
            ProcessorTableColumn::AUX13 => 44,
            ProcessorTableColumn::AUX14 => 45,
            ProcessorTableColumn::AUX15 => 46,
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
        row(f, "".into())?;
        row(
            f,
            format!(
                "aux3-0:   [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[AUX3 as usize].value(),
                self.row[AUX2 as usize].value(),
                self.row[AUX1 as usize].value(),
                self.row[AUX0 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "aux7-4:   [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[AUX7 as usize].value(),
                self.row[AUX6 as usize].value(),
                self.row[AUX5 as usize].value(),
                self.row[AUX4 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "aux11-8:  [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[AUX11 as usize].value(),
                self.row[AUX10 as usize].value(),
                self.row[AUX9 as usize].value(),
                self.row[AUX8 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "aux15-12: [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[AUX15 as usize].value(),
                self.row[AUX14 as usize].value(),
                self.row[AUX13 as usize].value(),
                self.row[AUX12 as usize].value(),
            ),
        )?;
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
