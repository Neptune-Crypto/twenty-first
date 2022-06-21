use super::table_column::{
    InstructionTableColumn, JumpStackTableColumn, OpStackTableColumn, ProcessorTableColumn::*,
    RamTableColumn,
};
use super::{
    hash_table, instruction_table, io_table, jump_stack_table, op_stack_table, processor_table,
    program_table, ram_table, u32_op_table,
};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::triton::instruction::Instruction;
use crate::shared_math::stark::triton::state::{VMOutput, VMState};
use crate::shared_math::stark::triton::vm::Program;
use itertools::Itertools;
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
        let mut words_with_0 = program.to_bwords();
        words_with_0.push(0.into());

        for (i, (word, next_word)) in words_with_0.into_iter().tuple_windows().enumerate() {
            let index = (i as u32).into();
            self.program_matrix.push([index, word]);
            self.instruction_matrix.push([index, word, next_word]);
        }

        debug_assert_eq!(program.len(), self.instruction_matrix.len());
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
                row[RamTableColumn::RAMP as usize].value(),
                row[RamTableColumn::CLK as usize].value(),
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
