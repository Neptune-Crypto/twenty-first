use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::base_table::{BaseTable, HasBaseTable, Table};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::x_field_element::XFieldElement;

pub const BASE_WIDTH: usize = 46;
pub const FULL_WIDTH: usize = 11; // FIXME: Should of course be >BASE_WIDTH

type BWord = BFieldElement;

pub struct ProcessorTable {
    base: BaseTable<BWord, BASE_WIDTH>,
}

impl HasBaseTable<BWord, BASE_WIDTH> for ProcessorTable {
    fn from_base(base: BaseTable<BWord, BASE_WIDTH>) -> Self {
        Self { base }
    }

    fn to_base(&self) -> &BaseTable<BWord, BASE_WIDTH> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord, BASE_WIDTH> {
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

impl Table<BWord, BASE_WIDTH> for ProcessorTable {
    fn name(&self) -> String {
        "ProcessorTable".to_string()
    }

    // FIXME: Apply correct padding, not just 0s.
    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let _last = data.last().unwrap();
            let padding = [0.into(); BASE_WIDTH];
            data.push(padding);
        }
    }

    fn boundary_constraints(
        &self,
        _challenges: &[BWord],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BWord>> {
        todo!()
    }

    fn transition_constraints(
        &self,
        _challenges: &[BWord],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BWord>> {
        todo!()
    }

    fn terminal_constraints(
        &self,
        _challenges: &[BWord],
        _terminals: &[BWord],
    ) -> Vec<crate::shared_math::mpolynomial::MPolynomial<BWord>> {
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

    fn boundary_constraints(
        &self,
        _challenges: &[XFieldElement],
    ) -> Vec<MPolynomial<XFieldElement>> {
        todo!()
    }

    fn transition_constraints(
        &self,
        _challenges: &[XFieldElement],
    ) -> Vec<MPolynomial<XFieldElement>> {
        todo!()
    }

    fn terminal_constraints(
        &self,
        _challenges: &[XFieldElement],
        _terminals: &[XFieldElement],
    ) -> Vec<MPolynomial<XFieldElement>> {
        todo!()
    }
}

impl ExtensionTable<FULL_WIDTH> for ExtProcessorTable {}
