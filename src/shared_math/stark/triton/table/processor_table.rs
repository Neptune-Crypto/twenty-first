use super::base_table::{BaseTable, HasBaseTable, Table};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::x_field_element::XFieldElement;

pub const BASE_WIDTH: usize = 7;
pub const FULL_WIDTH: usize = 11;

pub struct ProcessorTable {
    base: BaseTable<BFieldElement, BASE_WIDTH>,
}

impl HasBaseTable<BFieldElement, BASE_WIDTH> for ProcessorTable {
    fn new(base: BaseTable<BFieldElement, BASE_WIDTH>) -> Self {
        Self { base }
    }

    fn base(&self) -> &BaseTable<BFieldElement, BASE_WIDTH> {
        &self.base
    }
}

pub struct ExtProcessorTable {
    base: BaseTable<XFieldElement, FULL_WIDTH>,
}

impl HasBaseTable<XFieldElement, FULL_WIDTH> for ExtProcessorTable {
    fn new(base: BaseTable<XFieldElement, FULL_WIDTH>) -> Self {
        Self { base }
    }

    fn base(&self) -> &BaseTable<XFieldElement, FULL_WIDTH> {
        &self.base
    }
}

impl Table<BFieldElement, BASE_WIDTH> for ProcessorTable {
    fn pad(_matrix: &mut Vec<[BFieldElement; BASE_WIDTH]>) {
        todo!()
    }

    fn boundary_constraints(
        &self,
        _challenges: &[BFieldElement],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BFieldElement>> {
        todo!()
    }

    fn transition_constraints(
        &self,
        _challenges: &[BFieldElement],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BFieldElement>> {
        todo!()
    }

    fn terminal_constraints(
        &self,
        _challenges: &[BFieldElement],
        _terminals: &[BFieldElement],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BFieldElement>> {
        todo!()
    }
}

impl Table<XFieldElement, FULL_WIDTH> for ExtProcessorTable {
    fn pad(_matrix: &mut Vec<[XFieldElement; FULL_WIDTH]>) {
        panic!("Extension tables don't get padded");
    }

    fn boundary_constraints(
        &self,
        _challenges: &[XFieldElement],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<XFieldElement>> {
        todo!()
    }

    fn transition_constraints(
        &self,
        _challenges: &[XFieldElement],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<XFieldElement>> {
        todo!()
    }

    fn terminal_constraints(
        &self,
        _challenges: &[XFieldElement],
        _terminals: &[XFieldElement],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<XFieldElement>> {
        todo!()
    }
}

impl ExtensionTable<FULL_WIDTH> for ExtProcessorTable {}

#[cfg(test)]
mod processor_table_tests {
    use super::*;

    #[test]
    fn initialise_table_test() {
        // 1. Execute program

        // 2. Convert to table
        // 3. Extract constraints
        // 4. Check constraints
    }
}
