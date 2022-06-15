use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_initials::{AllChallenges, AllInitials};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::table::base_matrix::U32OpTableColumn;
use crate::shared_math::x_field_element::XFieldElement;

pub const U32_OP_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 5;
pub const U32_OP_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const U32_OP_TABLE_INITIALS_COUNT: usize =
    U32_OP_TABLE_PERMUTATION_ARGUMENTS_COUNT + U32_OP_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 14 because it combines: (lt, and, xor, div) x (lhs, rhs, result) + (rev) x (lhs, result)
pub const U32_OP_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 14;

pub const BASE_WIDTH: usize = 7;
pub const FULL_WIDTH: usize = 17; // BASE_WIDTH + 2 * INITIALS_COUNT

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

    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let padding_row = vec![0.into(); BASE_WIDTH];
            data.push(padding_row);
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

    pub fn extend(&self, challenges: &AllChallenges, initials: &AllInitials) -> ExtU32OpTable {
        let mut lt_running_product = initials.u32_op_table_initials.processor_lt_perm_initial;
        let mut and_running_product = initials.u32_op_table_initials.processor_and_perm_initial;
        let mut xor_running_product = initials.u32_op_table_initials.processor_xor_perm_initial;
        let mut reverse_running_product = initials
            .u32_op_table_initials
            .processor_reverse_perm_initial;
        let mut div_running_product = initials.u32_op_table_initials.processor_div_perm_initial;

        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // lhs and rhs are needed for _all_ of U32Table's Permutation Arguments
            let lhs = extension_row[U32OpTableColumn::LHS as usize];
            let rhs = extension_row[U32OpTableColumn::RHS as usize];

            // Compress (lhs, rhs, lt) into single value
            let lt = extension_row[U32OpTableColumn::LT as usize];
            let compressed_row_for_lt = lhs * challenges.u32_op_table_challenges.lt_lhs_weight
                + rhs * challenges.u32_op_table_challenges.lt_rhs_weight
                + lt * challenges.u32_op_table_challenges.lt_result_weight;
            extension_row.push(compressed_row_for_lt);

            // Multiply compressed value into running product for lt
            lt_running_product = lt_running_product
                * (challenges
                    .u32_op_table_challenges
                    .processor_lt_perm_row_weight
                    - compressed_row_for_lt);
            extension_row.push(lt_running_product);

            // Compress (lhs, rhs, and) into single value
            let and = extension_row[U32OpTableColumn::AND as usize];
            let compressed_row_for_and = lhs * challenges.u32_op_table_challenges.and_lhs_weight
                + rhs * challenges.u32_op_table_challenges.and_rhs_weight
                + and * challenges.u32_op_table_challenges.and_result_weight;
            extension_row.push(compressed_row_for_and);

            // Multiply compressed value into running product for and
            and_running_product = and_running_product
                * (challenges
                    .u32_op_table_challenges
                    .processor_and_perm_row_weight
                    - compressed_row_for_and);
            extension_row.push(and_running_product);

            // Compress (lhs, rhs, xor) into single value
            let xor = extension_row[U32OpTableColumn::XOR as usize];
            let compressed_row_for_xor = lhs * challenges.u32_op_table_challenges.xor_lhs_weight
                + rhs * challenges.u32_op_table_challenges.xor_rhs_weight
                + xor * challenges.u32_op_table_challenges.xor_result_weight;
            extension_row.push(compressed_row_for_xor);

            // Multiply compressed value into running product for xor
            xor_running_product = xor_running_product
                * (challenges
                    .u32_op_table_challenges
                    .processor_xor_perm_row_weight
                    - compressed_row_for_xor);
            extension_row.push(xor_running_product);

            // Compress (lhs, reverse) into single value
            let reverse = extension_row[U32OpTableColumn::REV as usize];
            let compressed_row_for_reverse = lhs
                * challenges.u32_op_table_challenges.reverse_lhs_weight
                + reverse * challenges.u32_op_table_challenges.reverse_result_weight;
            extension_row.push(compressed_row_for_reverse);

            // Multiply compressed value into running product for reverse
            reverse_running_product = reverse_running_product
                * (challenges
                    .u32_op_table_challenges
                    .processor_reverse_perm_row_weight
                    - compressed_row_for_reverse);
            extension_row.push(reverse_running_product);

            // Compress (lhs, rhs, lt) into single value for div
            let lt_for_div = extension_row[U32OpTableColumn::LT as usize];
            let compressed_row_for_div = lhs
                * challenges.u32_op_table_challenges.div_divisor_weight
                + rhs * challenges.u32_op_table_challenges.div_remainder_weight
                + lt_for_div * challenges.u32_op_table_challenges.div_result_weight;
            extension_row.push(compressed_row_for_div);

            // Multiply compressed value into running product for div
            div_running_product = div_running_product
                * (challenges
                    .u32_op_table_challenges
                    .processor_div_perm_row_weight
                    - compressed_row_for_div);
            extension_row.push(div_running_product);

            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        ExtU32OpTable { base }
    }
}

impl ExtU32OpTable {
    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain);
        let base = self.base.with_data(ext_codewords);

        ExtU32OpTable { base }
    }
}

#[derive(Debug, Clone)]
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
    pub reverse_result_weight: XFieldElement,

    pub div_divisor_weight: XFieldElement,
    pub div_remainder_weight: XFieldElement,
    pub div_result_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct U32OpTableInitials {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_lt_perm_initial: XFieldElement,
    pub processor_and_perm_initial: XFieldElement,
    pub processor_xor_perm_initial: XFieldElement,
    pub processor_reverse_perm_initial: XFieldElement,
    pub processor_div_perm_initial: XFieldElement,
}
