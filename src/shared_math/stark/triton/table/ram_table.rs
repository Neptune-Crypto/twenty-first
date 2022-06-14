use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::table::base_matrix::RAMTableColumn;
use crate::shared_math::x_field_element::XFieldElement;

pub const RAM_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const RAM_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const RAM_TABLE_INITIALS_COUNT: usize =
    RAM_TABLE_PERMUTATION_ARGUMENTS_COUNT + RAM_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 3 because it combines: clk, ramv, ramp
pub const RAM_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 3;

pub const BASE_WIDTH: usize = 3;
pub const FULL_WIDTH: usize = 5; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct RAMTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for RAMTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtRAMTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtRAMTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl RAMTable {
    pub fn new_verifier(
        generator: BWord,
        order: usize,
        num_randomizers: usize,
        padded_height: usize,
    ) -> Self {
        let matrix: Vec<Vec<BWord>> = vec![];

        let dummy = generator;
        let omicron = base_table::derive_omicron(padded_height as u64, dummy);
        let base = BaseTable::new(
            BASE_WIDTH,
            padded_height,
            num_randomizers,
            omicron,
            generator,
            order,
            matrix,
        );

        Self { base }
    }

    pub fn new_prover(
        generator: BWord,
        order: usize,
        num_randomizers: usize,
        matrix: Vec<Vec<BWord>>,
    ) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::pad_height(unpadded_height);

        let dummy = generator;
        let omicron = base_table::derive_omicron(padded_height as u64, dummy);
        let base = BaseTable::new(
            BASE_WIDTH,
            padded_height,
            num_randomizers,
            omicron,
            generator,
            order,
            matrix,
        );

        Self { base }
    }

    pub fn extend(
        &self,
        challenges: &RamTableChallenges,
        initials: &RamTableEndpoints,
    ) -> (ExtRAMTable, RamTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product = initials.processor_perm_product;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            let (clk, ramp, ramv) = (
                extension_row[RAMTableColumn::CLK as usize],
                extension_row[RAMTableColumn::RAMP as usize],
                extension_row[RAMTableColumn::RAMV as usize],
            );

            let (clk_w, ramp_w, ramv_w) = (
                challenges.clk_weight,
                challenges.ramp_weight,
                challenges.ramv_weight,
            );

            // 1. Compress multiple values within one row so they become one value.
            let compressed_row_for_permutation_argument =
                clk * clk_w + ramp * ramp_w + ramv * ramv_w;

            extension_row.push(compressed_row_for_permutation_argument);

            // 2. Compute the running *product* of the compressed column (permutation value)
            extension_row.push(running_product);
            running_product = running_product
                * (challenges.processor_perm_row_weight - compressed_row_for_permutation_argument);

            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtRAMTable { base };
        let terminals = RamTableEndpoints {
            processor_perm_product: running_product,
        };

        (table, terminals)
    }
}

impl ExtRAMTable {
    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain);
        let base = self.base.with_data(ext_codewords);

        ExtRAMTable { base }
    }
}

impl Table<BWord> for RAMTable {
    fn name(&self) -> String {
        "RAMTable".to_string()
    }

    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let mut padding_row = data.last().unwrap().clone();
            // add same clk padding as in processor table
            padding_row[RAMTableColumn::CLK as usize] = ((data.len() - 1) as u32).into();
            data.push(padding_row);
        }
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BWord>> {
        vec![]
    }
}

impl Table<XFieldElement> for ExtRAMTable {
    fn name(&self) -> String {
        "ExtRAMTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl ExtensionTable for ExtRAMTable {
    fn ext_boundary_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_transition_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_terminal_constraints(
        &self,
        _challenges: &AllChallenges,
        _terminals: &AllEndpoints,
    ) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct RamTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the op-stack table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ramv_weight: XFieldElement,
    pub ramp_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct RamTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_product: XFieldElement,
}
