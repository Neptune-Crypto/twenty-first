use super::base_table::{BaseTable, HasBaseTable, Table};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::x_field_element::XFieldElement;

pub const BASE_WIDTH: usize = 1; // FIXME: Find right width
pub const FULL_WIDTH: usize = 0; // FIXME: Should of course be >BASE_WIDTH

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct U32OpTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for U32OpTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

pub struct ExtU32OpTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtU32OpTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for U32OpTable {
    fn name(&self) -> String {
        "U32OpTable".to_string()
    }

    // FIXME: Apply correct padding, not just 0s.
    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let _last = data.last().unwrap();
            let padding = vec![0.into(); BASE_WIDTH];
            data.push(padding);
        }
    }

    fn boundary_constraints(
        &self,
        _challenges: &[BWord],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BWord>> {
        vec![]
    }

    fn transition_constraints(
        &self,
        _challenges: &[BWord],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BWord>> {
        vec![]
    }

    fn terminal_constraints(
        &self,
        _challenges: &[BWord],
        _terminals: &[BWord],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BWord>> {
        vec![]
    }
}

impl Table<XFieldElement> for ExtU32OpTable {
    fn name(&self) -> String {
        "ExtU32OpTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }

    fn boundary_constraints(
        &self,
        _challenges: &[XFieldElement],
    ) -> Vec<MPolynomial<XFieldElement>> {
        vec![]
    }

    fn transition_constraints(
        &self,
        _challenges: &[XFieldElement],
    ) -> Vec<MPolynomial<XFieldElement>> {
        vec![]
    }

    fn terminal_constraints(
        &self,
        _challenges: &[XFieldElement],
        _terminals: &[XFieldElement],
    ) -> Vec<MPolynomial<XFieldElement>> {
        vec![]
    }
}

impl ExtensionTable for ExtU32OpTable {}
