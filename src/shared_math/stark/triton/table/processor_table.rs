use super::base_table::{BaseTable, HasBaseTable, Table};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::x_field_element::XFieldElement;

pub const BASE_WIDTH: usize = 46;
pub const FULL_WIDTH: usize = 11; // FIXME: Should of course be >BASE_WIDTH

pub struct ProcessorTable {
    base: BaseTable<BFieldElement, BASE_WIDTH>,
}

impl HasBaseTable<BFieldElement, BASE_WIDTH> for ProcessorTable {
    fn from_base(base: BaseTable<BFieldElement, BASE_WIDTH>) -> Self {
        Self { base }
    }

    fn to_base(&self) -> &BaseTable<BFieldElement, BASE_WIDTH> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BFieldElement, BASE_WIDTH> {
        &mut self.base
    }
}

pub struct ExtProcessorTable {
    base: BaseTable<XFieldElement, FULL_WIDTH>,
}

impl HasBaseTable<XFieldElement, FULL_WIDTH> for ExtProcessorTable {
    fn from_base(base: BaseTable<XFieldElement, FULL_WIDTH>) -> Self {
        Self { base }
    }

    fn to_base(&self) -> &BaseTable<XFieldElement, FULL_WIDTH> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement, FULL_WIDTH> {
        &mut self.base
    }
}

impl Table<BFieldElement, BASE_WIDTH> for ProcessorTable {
    fn name(&self) -> String {
        "ProcessorTable".to_string()
    }

    // FIXME: Apply correct padding, not just 0s.
    fn pad(&mut self) {
        let data = self.data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let _last = data.last().unwrap();
            let padding = [0.into(); BASE_WIDTH];
            data.push(padding);
        }
    }

    fn codewords(&self) -> Self {
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
    fn name(&self) -> String {
        "ExtProcessorTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }

    fn codewords(&self) -> Self {
        todo!()
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
