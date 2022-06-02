use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::x_field_element::XFieldElement;

pub const BASE_WIDTH: usize = 46;
pub const FULL_WIDTH: usize = 0; // FIXME: Should of course be >BASE_WIDTH

type BWord = BFieldElement;

#[derive(Debug, Clone)]
pub struct ProcessorTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for ProcessorTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

impl ProcessorTable {
    pub fn new(
        unpadded_height: usize,
        num_randomizers: usize,
        generator: BWord,
        order: usize,
        matrix: Vec<Vec<BWord>>,
    ) -> Self {
        let dummy = generator;
        let omicron = base_table::derive_omicron(unpadded_height as u64, dummy);
        let base = BaseTable::new(
            BASE_WIDTH,
            unpadded_height,
            num_randomizers,
            omicron,
            generator,
            order,
            matrix,
        );

        Self { base }
    }

    pub fn codewords(&self, fri_domain: &FriDomain<BWord>) -> Self {
        let codewords = self.low_degree_extension(fri_domain);
        Self::new(
            self.unpadded_height(),
            self.num_randomizers(),
            self.generator(),
            self.order(),
            codewords,
        )
    }
}

pub struct ExtProcessorTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtProcessorTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for ProcessorTable {
    fn name(&self) -> String {
        "ProcessorTable".to_string()
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

impl Table<XFieldElement> for ExtProcessorTable {
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

impl ExtensionTable for ExtProcessorTable {}
