use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_initials::{AllChallenges, AllInitials};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
<<<<<<< Updated upstream
use crate::shared_math::stark::triton::fri_domain::FriDomain;
||||||| constructed merge base
=======
use crate::shared_math::stark::triton::table::base_matrix::IOTableColumn::*;
>>>>>>> Stashed changes
use crate::shared_math::x_field_element::XFieldElement;

/// Determine how many random values we get from the verifier
pub const IOTABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const IOTABLE_EVALUATION_ARGUMENT_COUNT: usize = 1;
pub const IOTABLE_INITIALS_COUNT: usize =
    IOTABLE_PERMUTATION_ARGUMENTS_COUNT + IOTABLE_EVALUATION_ARGUMENT_COUNT;

/// There are no extension challenges because there is only one column,
/// and we use extension challenges to condense
pub const IOTABLE_EXTENSION_CHALLENGE_COUNT: usize = 0;

pub const BASE_WIDTH: usize = 1;
pub const FULL_WIDTH: usize = 2; // BASE + INITIALS

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct IOTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for IOTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtIOTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtIOTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for IOTable {
    fn name(&self) -> String {
        "IOTable".to_string()
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

impl Table<XFieldElement> for ExtIOTable {
    fn name(&self) -> String {
        "ExtIOTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl ExtensionTable for ExtIOTable {
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

impl IOTable {
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

    pub fn extend(&self, challenges: &IOTableChallenges, initials: &IOTableInitials) -> ExtIOTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());

        let mut running_sum = initials.processor_eval_initial;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(row.len() + 1);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // 1. No compression needed for a single column
            let iosymbol = extension_row[IOSymbol as usize];
            // 2. Not applicable
            // 3. Not applicable

            // 4. In the case of the evalutation arguement we need to compute the running *sum*.
            running_sum = running_sum * challenges.processor_eval_row_weight + iosymbol;
            extension_row.push(running_sum);
        }

        let base = self.base.with_lifted_data(extension_matrix);

        ExtIOTable { base }
    }
}

impl ExtIOTable {
    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain);
        let base = self.base.with_data(ext_codewords);

        ExtIOTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct IOTableChallenges {
    /// The weight that combines two consecutive rows in the evaluation column of the input table.
    pub processor_eval_row_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct IOTableInitials {
    /// A value randomly generated by the prover for zero-knowledge.
    pub processor_eval_initial: XFieldElement,
}
