use super::base_matrix::ProcessorTableColumn;
use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_initials::{AllChallenges, AllInitials};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::x_field_element::XFieldElement;

pub const PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 9;
pub const PROCESSOR_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 4;
pub const PROCESSOR_TABLE_INITIALS_COUNT: usize =
    PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT + PROCESSOR_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 48 because it combines all other tables (except program).
pub const PROCESSOR_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 48;

pub const BASE_WIDTH: usize = 39;
/// BASE_WIDTH + 2 * INITIALS_COUNT - 2 (because IOSymbols don't need compression)
pub const FULL_WIDTH: usize = 63;

type BWord = BFieldElement;
type XWord = XFieldElement;

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

    pub fn extend(&self, challenges: &AllChallenges, initials: &AllInitials) -> ExtProcessorTable {
        todo!()
    }
}

impl ExtProcessorTable {
    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain);
        let base = self.base.with_data(ext_codewords);

        ExtProcessorTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessorTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the processor table.
    pub input_table_eval_row_weight: XFieldElement,
    pub output_table_eval_row_weight: XFieldElement,
    pub to_hash_table_eval_row_weight: XFieldElement,
    pub from_hash_table_eval_row_weight: XFieldElement,

    pub instruction_perm_row_weight: XFieldElement,
    pub op_stack_perm_row_weight: XFieldElement,
    pub ram_perm_row_weight: XFieldElement,
    pub jump_stack_perm_row_weight: XFieldElement,

    pub u32_lt_perm_row_weight: XFieldElement,
    pub u32_and_perm_row_weight: XFieldElement,
    pub u32_xor_perm_row_weight: XFieldElement,
    pub u32_reverse_perm_row_weight: XFieldElement,
    pub u32_div_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub instruction_table_ip_weight: XFieldElement,
    pub instruction_table_ci_processor_weight: XFieldElement,
    pub instruction_table_nia_weight: XFieldElement,

    pub op_stack_table_clk_weight: XFieldElement,
    pub op_stack_table_ci_weight: XFieldElement,
    pub op_stack_table_osv_weight: XFieldElement,
    pub op_stack_table_osp_weight: XFieldElement,

    pub ram_table_clk_weight: XFieldElement,
    pub ram_table_ramv_weight: XFieldElement,
    pub ram_table_ramp_weight: XFieldElement,

    pub jump_stack_table_clk_weight: XFieldElement,
    pub jump_stack_table_ci_weight: XFieldElement,
    pub jump_stack_table_jsp_weight: XFieldElement,
    pub jump_stack_table_jso_weight: XFieldElement,
    pub jump_stack_table_jsd_weight: XFieldElement,

    pub hash_table_stack_input_weights: [XFieldElement; 2 * DIGEST_LEN],
    pub hash_table_digest_output_weights: [XFieldElement; DIGEST_LEN],

    pub u32_op_table_lt_lhs_weight: XFieldElement,
    pub u32_op_table_lt_rhs_weight: XFieldElement,
    pub u32_op_table_lt_result_weight: XFieldElement,

    pub u32_op_table_and_lhs_weight: XFieldElement,
    pub u32_op_table_and_rhs_weight: XFieldElement,
    pub u32_op_table_and_result_weight: XFieldElement,

    pub u32_op_table_xor_lhs_weight: XFieldElement,
    pub u32_op_table_xor_rhs_weight: XFieldElement,
    pub u32_op_table_xor_result_weight: XFieldElement,

    pub u32_op_table_reverse_lhs_weight: XFieldElement,
    pub u32_op_table_reverse_rhs_weight: XFieldElement,
    pub u32_op_table_reverse_result_weight: XFieldElement,

    pub u32_op_table_div_lhs_weight: XFieldElement,
    pub u32_op_table_div_rhs_weight: XFieldElement,
    pub u32_op_table_div_result_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct ProcessorTableInitials {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_initial: XFieldElement,
}

#[derive(Debug, Clone)]
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

    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let mut padding_row = data.last().unwrap().clone();
            padding_row[ProcessorTableColumn::CLK as usize] =
                padding_row[ProcessorTableColumn::CLK as usize] + 1.into();
            data.push(padding_row);
        }
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BWord>> {
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

    fn base_transition_constraints(&self) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl ExtensionTable for ExtProcessorTable {
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

/// TODO: Move the constraint polynomial boilerplate to a separate section of the code (e.g. a subdirectory for polynomials)
///
/// Working title. This needs to have a good name.
///
/// 2 * BASE_WIDTH because
pub struct ProcessorTableTransitionConstraintPolynomialFactory {
    variables: [MPolynomial<BWord>; 2 * BASE_WIDTH],
}

impl Default for ProcessorTableTransitionConstraintPolynomialFactory {
    fn default() -> Self {
        let variables = MPolynomial::<BWord>::variables(2 * BASE_WIDTH, 1.into())
            .try_into()
            .expect("Dynamic conversion of polynomials for each variable to known-width set of variables");

        Self { variables }
    }
}

impl ProcessorTableTransitionConstraintPolynomialFactory {
    /// ## The cycle counter (`clk`) always increases by one
    ///
    /// $$
    /// p(..., clk, clk_next, ...) = clk_next - clk - 1
    /// $$
    ///
    /// In general, for all $clk = a$, and $clk_next = a + 1$,
    ///
    /// $$
    /// p(..., a, a+1, ...) = (a+1) - a - 1 = a + 1 - a - 1 = a - a + 1 - 1 = 0
    /// $$
    ///
    /// So the `clk_increase_by_one` base transition constraint polynomial holds exactly
    /// when every `clk` register $a$ is one less than `clk` register $a + 1$.
    pub fn clk_always_increases_by_one(&self) -> MPolynomial<BWord> {
        let one = self.one();
        let clk = self.clk();
        let clk_next = self.clk_next();

        clk_next - clk - one
    }

    // FIXME: Consider caching this on first run (caching computed getter)
    pub fn one(&self) -> MPolynomial<BWord> {
        MPolynomial::from_constant(1.into(), 2 * BASE_WIDTH)
    }

    pub fn clk(&self) -> MPolynomial<BWord> {
        self.variables[usize::from(ProcessorTableColumn::CLK)].clone()
    }

    // Property: All polynomial variables that contain '_next' have the same
    // variable position / value as the one without '_next', +/- BASE_WIDTH.
    pub fn clk_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + usize::from(ProcessorTableColumn::CLK)].clone()
    }
}
