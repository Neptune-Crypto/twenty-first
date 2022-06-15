use itertools::Itertools;

use super::base_matrix::InstructionTableColumn::*;
use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_initials::{AllChallenges, AllInitials};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::table::base_matrix::InstructionTableColumn;
use crate::shared_math::x_field_element::XFieldElement;

pub const INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_INITIALS_COUNT: usize =
    INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT + INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 5 because it combines: (ip, ci, nia) and (addr, instruction).
pub const INSTRUCTION_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 5;

pub const BASE_WIDTH: usize = 3;
pub const FULL_WIDTH: usize = 7; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct InstructionTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for InstructionTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtInstructionTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtInstructionTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for InstructionTable {
    fn name(&self) -> String {
        "InstructionTable".to_string()
    }

    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let mut padding_row = data.last().unwrap().clone();
            // address keeps increasing
            padding_row[InstructionTableColumn::Address as usize] =
                padding_row[InstructionTableColumn::Address as usize] + 1.into();
            data.push(padding_row);
        }
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BWord>> {
        vec![]
    }
}

impl Table<XFieldElement> for ExtInstructionTable {
    fn name(&self) -> String {
        "ExtInstructionTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl ExtensionTable for ExtInstructionTable {
    fn ext_boundary_constraints(&self, challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_transition_constraints(&self, challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
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

impl InstructionTable {
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
        all_challenges: &AllChallenges,
        all_initials: &AllInitials,
    ) -> ExtInstructionTable {
        let challenges = &all_challenges.instruction_table_challenges;
        let initials = &all_initials.instruction_table_initials;

        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());

        let mut running_product = initials.processor_perm_initial;
        let mut running_sum = initials.program_eval_initial;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // 1. Compress multiple values within one row so they become one value.
            let ip = row[Address as usize].lift();
            let ci = row[CI as usize].lift();
            let nia = row[NIA as usize].lift();

            let (ip_w, ci_w, nia_w) = (
                challenges.ip_weight,
                challenges.ci_processor_weight,
                challenges.nia_weight,
            );
            let compressed_row_for_permutation_argument = ip * ip_w + ci * ci_w + nia * nia_w;
            extension_row.push(compressed_row_for_permutation_argument);

            // 2. In the case of the permutation value we need to compute the running *product* of the compressed column.
            extension_row.push(running_product);
            running_product = running_product
                * (challenges.processor_perm_row_weight - compressed_row_for_permutation_argument);

            // 3. Since we are in the instruction table we compress multiple values for the evaluation arguement.
            let address = row[Address as usize].lift();
            let instruction = row[CI as usize].lift();

            let (address_w, instruction_w) =
                (challenges.addr_weight, challenges.instruction_weight);
            let compressed_row_for_evaluation_arguement =
                address * address_w + instruction * instruction_w;
            extension_row.push(compressed_row_for_evaluation_arguement);

            // 4. In the case of the evalutation arguement we need to compute the running *sum*.
            extension_row.push(running_sum);
            running_sum = running_sum * challenges.program_eval_row_weight
                + compressed_row_for_evaluation_arguement;

            // Build the extension matrix
            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);

        ExtInstructionTable { base }
    }
}

impl ExtInstructionTable {
    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain);
        let base = self.base.with_data(ext_codewords);

        ExtInstructionTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct InstructionTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the instruction table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub ip_weight: XFieldElement,
    pub ci_processor_weight: XFieldElement,
    pub nia_weight: XFieldElement,

    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the instruction table.
    pub program_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to program table.)
    pub addr_weight: XFieldElement,
    pub instruction_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct InstructionTableInitials {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_initial: XFieldElement,
    pub program_eval_initial: XFieldElement,
}
