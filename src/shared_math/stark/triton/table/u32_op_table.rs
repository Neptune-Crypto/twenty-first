use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_initials::{AllChallenges, AllInitials};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::x_field_element::XFieldElement;

pub const U32_OP_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 5;
pub const U32_OP_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const U32_OP_TABLE_INITIALS_COUNT: usize =
    U32_OP_TABLE_PERMUTATION_ARGUMENTS_COUNT + U32_OP_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 15 because it combines: (lt, and, xor, reverse, div) x (lhs, rhs, result)
pub const U32_OP_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 15;

pub const BASE_WIDTH: usize = 7;
pub const FULL_WIDTH: usize = 12; // BASE + INITIALS

type BWord = BFieldElement;
type XWord = XFieldElement;

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

#[derive(Debug, Clone)]
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

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BWord>> {
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

    fn base_transition_constraints(&self) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl ExtensionTable for ExtU32OpTable {
    fn ext_boundary_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_transition_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_terminal_constraints(
        &self,
        _challenges: &AllChallenges,
        _terminals: &AllInitials,
    ) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl U32OpTable {
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
        challenges: &U32OpTableChallenges,
        initials: &U32OpTableInitials,
    ) -> ExtU32OpTable {
        todo!()
    }
}

pub struct U32OpTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the u32 op table.
    pub processor_lt_perm_row_weight: XFieldElement,
    pub processor_and_perm_row_weight: XFieldElement,
    pub processor_xor_perm_row_weight: XFieldElement,
    pub processor_reverse_perm_row_weight: XFieldElement,
    pub processor_div_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub lt_lhs_weight: XFieldElement,
    pub lt_rhs_weight: XFieldElement,
    pub lt_result_weight: XFieldElement,

    pub and_lhs_weight: XFieldElement,
    pub and_rhs_weight: XFieldElement,
    pub and_result_weight: XFieldElement,

    pub xor_lhs_weight: XFieldElement,
    pub xor_rhs_weight: XFieldElement,
    pub xor_result_weight: XFieldElement,

    pub reverse_lhs_weight: XFieldElement,
    pub reverse_rhs_weight: XFieldElement,
    pub reverse_result_weight: XFieldElement,

    pub div_lhs_weight: XFieldElement,
    pub div_rhs_weight: XFieldElement,
    pub div_result_weight: XFieldElement,
}

pub struct U32OpTableInitials {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_lt_perm_initial: XFieldElement,
    pub processor_and_perm_initial: XFieldElement,
    pub processor_xor_perm_initial: XFieldElement,
    pub processor_reverse_perm_initial: XFieldElement,
    pub processor_div_perm_initial: XFieldElement,
}
