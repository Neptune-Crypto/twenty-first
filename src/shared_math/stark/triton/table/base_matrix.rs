use std::fmt::Display;

use crate::shared_math::b_field_element::BFieldElement;

use super::{instruction_table, io_table, processor_table};

#[derive(Debug, Clone, Default)]
pub struct BaseMatrices {
    pub processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
    pub instruction_matrix: Vec<[BFieldElement; instruction_table::BASE_WIDTH]>,
    pub input_matrix: Vec<[BFieldElement; io_table::BASE_WIDTH]>,
    pub output_matrix: Vec<[BFieldElement; io_table::BASE_WIDTH]>,
}

impl BaseMatrices {
    pub fn sort_instruction_matrix(&mut self) {
        self.instruction_matrix.sort_by_key(|row| row[0].value());
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
