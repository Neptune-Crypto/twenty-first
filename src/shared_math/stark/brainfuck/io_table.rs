use crate::shared_math::{b_field_element::BFieldElement, mpolynomial::MPolynomial};

use super::table::{Table, TableMoreTrait};

pub struct IOTable(Table<IOTableMore>);

pub struct IOTableMore {
    pub challenge_index: usize,
    pub terminal_index: usize,
}

impl TableMoreTrait for IOTableMore {
    fn new_more() -> Self {
        IOTableMore {
            challenge_index: 0,
            terminal_index: 0,
        }
    }

    fn base_transition_constraints() -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }

    fn base_boundary_constraints() -> Vec<MPolynomial<BFieldElement>> {
        todo!()
    }
}

impl IOTable {
    pub const BASE_WIDTH: usize = 1;
    pub const FULL_WIDTH: usize = 2;

    // TODO: Refactor to avoid duplicate code.
    pub fn new_input_table(length: usize, generator: BFieldElement, order: usize) -> Self {
        let num_randomizers = 0;

        let mut table = Table::<IOTableMore>::new(
            Self::BASE_WIDTH,
            Self::FULL_WIDTH,
            length,
            num_randomizers,
            generator,
            order,
        );

        table.more.challenge_index = 8;
        table.more.terminal_index = 2;

        Self(table)
    }

    pub fn new_output_table(length: usize, generator: BFieldElement, order: usize) -> Self {
        let num_randomizers = 0;
        let base_width = 1;
        let full_width = 2;

        let mut table = Table::<IOTableMore>::new(
            base_width,
            full_width,
            length,
            num_randomizers,
            generator,
            order,
        );

        table.more.challenge_index = 9;
        table.more.terminal_index = 3;

        Self(table)
    }

    pub fn challenge_index(&self) -> usize {
        self.0.more.challenge_index
    }

    pub fn terminal_index(&self) -> usize {
        self.0.more.terminal_index
    }
}
