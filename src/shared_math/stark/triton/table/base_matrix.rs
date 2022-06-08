use super::{
    aux_table, instruction_table, io_table, jump_stack_table, op_stack_table, processor_table,
    program_table, ram_table, u32_op_table,
};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::triton::instruction::Instruction;
use crate::shared_math::stark::triton::state::{VMOutput, VMState};
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

pub struct ProcessorMatrixRow {
    pub row: [BFieldElement; processor_table::BASE_WIDTH],
}

impl ProcessorMatrixRow {
    pub fn clk(&self) -> BFieldElement {
        self.row[0]
    }
    pub fn ip(&self) -> BFieldElement {
        self.row[1]
    }
    pub fn ci(&self) -> BFieldElement {
        self.row[2]
    }
    pub fn nia(&self) -> BFieldElement {
        self.row[3]
    }
    pub fn ib0(&self) -> BFieldElement {
        self.row[4]
    }
    pub fn ib1(&self) -> BFieldElement {
        self.row[5]
    }
    pub fn ib2(&self) -> BFieldElement {
        self.row[6]
    }
    pub fn ib3(&self) -> BFieldElement {
        self.row[7]
    }
    pub fn ib4(&self) -> BFieldElement {
        self.row[8]
    }
    pub fn ib5(&self) -> BFieldElement {
        self.row[9]
    }
    pub fn jsp(&self) -> BFieldElement {
        self.row[10]
    }
    pub fn jso(&self) -> BFieldElement {
        self.row[11]
    }
    pub fn jsd(&self) -> BFieldElement {
        self.row[12]
    }
    pub fn st0(&self) -> BFieldElement {
        self.row[13]
    }
    pub fn st1(&self) -> BFieldElement {
        self.row[14]
    }
    pub fn st2(&self) -> BFieldElement {
        self.row[15]
    }
    pub fn st3(&self) -> BFieldElement {
        self.row[16]
    }
    pub fn st4(&self) -> BFieldElement {
        self.row[17]
    }
    pub fn st5(&self) -> BFieldElement {
        self.row[18]
    }
    pub fn st6(&self) -> BFieldElement {
        self.row[19]
    }
    pub fn st7(&self) -> BFieldElement {
        self.row[20]
    }
    pub fn inv(&self) -> BFieldElement {
        self.row[21]
    }
    pub fn osp(&self) -> BFieldElement {
        self.row[22]
    }
    pub fn osv(&self) -> BFieldElement {
        self.row[23]
    }
    pub fn hv0(&self) -> BFieldElement {
        self.row[24]
    }
    pub fn hv1(&self) -> BFieldElement {
        self.row[25]
    }
    pub fn hv2(&self) -> BFieldElement {
        self.row[26]
    }
    pub fn hv3(&self) -> BFieldElement {
        self.row[27]
    }
    pub fn ramp(&self) -> BFieldElement {
        self.row[28]
    }
    pub fn ramv(&self) -> BFieldElement {
        self.row[29]
    }
    pub fn aux0(&self) -> BFieldElement {
        self.row[30]
    }
    pub fn aux1(&self) -> BFieldElement {
        self.row[31]
    }
    pub fn aux2(&self) -> BFieldElement {
        self.row[32]
    }
    pub fn aux3(&self) -> BFieldElement {
        self.row[33]
    }
    pub fn aux4(&self) -> BFieldElement {
        self.row[34]
    }
    pub fn aux5(&self) -> BFieldElement {
        self.row[35]
    }
    pub fn aux6(&self) -> BFieldElement {
        self.row[36]
    }
    pub fn aux7(&self) -> BFieldElement {
        self.row[37]
    }
    pub fn aux8(&self) -> BFieldElement {
        self.row[38]
    }
    pub fn aux9(&self) -> BFieldElement {
        self.row[39]
    }
    pub fn aux10(&self) -> BFieldElement {
        self.row[40]
    }
    pub fn aux11(&self) -> BFieldElement {
        self.row[41]
    }
    pub fn aux12(&self) -> BFieldElement {
        self.row[42]
    }
    pub fn aux13(&self) -> BFieldElement {
        self.row[43]
    }
    pub fn aux14(&self) -> BFieldElement {
        self.row[44]
    }
    pub fn aux15(&self) -> BFieldElement {
        self.row[45]
    }
}

impl Display for ProcessorMatrixRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        horizontal_bar(f)?;

        let width = 15;
        column(
            f,
            format!(
                "clk:  {:>width$} | ip:   {:>width$} |",
                self.clk(),
                self.ip()
            ),
        )?;

        column(
            f,
            format!(
                "ci:   {:>width$} | nia: {:>width$} |",
                self.ci(),
                self.nia()
            ),
        )?;

        // FIXME: ib0-ib5
        // FIXME: hv0-hv3

        column(
            f,
            format!(
                "ramp: {:>width$} | ramv: {:>width$} |",
                self.ramp(),
                self.ramv()
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
                self.osp(),
                self.osv(),
                self.inv()
            ),
        )?;

        let width_st = 5;
        column(
            f,
            format!(
                "st7-0: [ {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} | {:^width_st$} ]",
                self.st7(),
                self.st6(),
                self.st5(),
                self.st4(),
                self.st3(),
                self.st2(),
                self.st1(),
                self.st0(),
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
