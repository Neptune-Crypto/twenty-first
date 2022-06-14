use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_initials::{AllChallenges, AllInitials};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::rescue_prime_xlix::neptune_params;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::state::{AUX_REGISTER_COUNT, DIGEST_LEN};
use crate::shared_math::stark::triton::table::base_matrix::HashTableColumn;
use crate::shared_math::traits::IdentityValues;
use crate::shared_math::x_field_element::XFieldElement;
use itertools::Itertools;

pub const HASH_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const HASH_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 2;
pub const HASH_TABLE_INITIALS_COUNT: usize =
    HASH_TABLE_PERMUTATION_ARGUMENTS_COUNT + HASH_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 18 because it combines: 12 stack_input_weights and 6 digest_output_weights.
pub const HASH_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 18;

pub const BASE_WIDTH: usize = 17;
pub const FULL_WIDTH: usize = 21; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct HashTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for HashTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtHashTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtHashTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for HashTable {
    fn name(&self) -> String {
        "HashTable".to_string()
    }

    fn pad(&mut self) {
        let data = self.mut_data();
        let rescue_prime = neptune_params();
        let mut aux = [BFieldElement::ring_zero(); AUX_REGISTER_COUNT];
        let hash_trace = rescue_prime.rescue_xlix_permutation_trace(&mut aux);
        let padding = &mut hash_trace.iter().map(|row| row.to_vec()).collect();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            data.append(padding);
        }
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BWord>> {
        vec![]
    }
}

impl Table<XFieldElement> for ExtHashTable {
    fn name(&self) -> String {
        "ExtHashTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl ExtensionTable for ExtHashTable {
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

impl HashTable {
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

    pub fn extend(&self, challenges: &AllChallenges, initials: &AllInitials) -> ExtHashTable {
        let mut from_processor_running_sum =
            initials.hash_table_initials.from_processor_eval_initial;
        let mut to_processor_running_sum = initials.hash_table_initials.to_processor_eval_initial;

        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // Compress input values into single value (independent of round index)
            let aux_for_input = [
                extension_row[HashTableColumn::AUX0 as usize],
                extension_row[HashTableColumn::AUX1 as usize],
                extension_row[HashTableColumn::AUX2 as usize],
                extension_row[HashTableColumn::AUX3 as usize],
                extension_row[HashTableColumn::AUX4 as usize],
                extension_row[HashTableColumn::AUX5 as usize],
                extension_row[HashTableColumn::AUX6 as usize],
                extension_row[HashTableColumn::AUX7 as usize],
                extension_row[HashTableColumn::AUX8 as usize],
                extension_row[HashTableColumn::AUX9 as usize],
                extension_row[HashTableColumn::AUX10 as usize],
                extension_row[HashTableColumn::AUX11 as usize],
            ];
            let compressed_aux_for_input = aux_for_input
                .iter()
                .zip(challenges.hash_table_challenges.stack_input_weights.iter())
                .map(|(aux, weight)| *weight * *aux)
                .fold(XWord::ring_zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_aux_for_input);

            // Add compressed input to running sum if round index is 0
            if row[HashTableColumn::RoundNumber as usize].is_zero() {
                from_processor_running_sum = from_processor_running_sum
                    * challenges
                        .hash_table_challenges
                        .from_processor_eval_row_weight
                    + compressed_aux_for_input;
            }
            extension_row.push(from_processor_running_sum);

            // Compress digest values into single value (independent of round index)
            let aux_for_output = [
                extension_row[HashTableColumn::AUX0 as usize],
                extension_row[HashTableColumn::AUX1 as usize],
                extension_row[HashTableColumn::AUX2 as usize],
                extension_row[HashTableColumn::AUX3 as usize],
                extension_row[HashTableColumn::AUX4 as usize],
                extension_row[HashTableColumn::AUX5 as usize],
            ];
            let compressed_aux_for_output = aux_for_output
                .iter()
                .zip(
                    challenges
                        .hash_table_challenges
                        .digest_output_weights
                        .iter(),
                )
                .map(|(aux, weight)| *weight * *aux)
                .fold(XWord::ring_zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_aux_for_output);

            // Add compressed digest to running sum if round index is 7
            if row[HashTableColumn::RoundNumber as usize].value() == 7 {
                to_processor_running_sum = to_processor_running_sum
                    * challenges
                        .hash_table_challenges
                        .to_processor_eval_row_weight
                    + compressed_aux_for_output;
            }
            extension_row.push(to_processor_running_sum);

            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        ExtHashTable { base }
    }
}

impl ExtHashTable {
    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain);
        let base = self.base.with_data(ext_codewords);

        ExtHashTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct HashTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the hash table.
    pub from_processor_eval_row_weight: XFieldElement,
    pub to_processor_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub stack_input_weights: [XFieldElement; 2 * DIGEST_LEN],
    pub digest_output_weights: [XFieldElement; DIGEST_LEN],
}

#[derive(Debug, Clone)]
pub struct HashTableInitials {
    /// Values randomly generated by the prover for zero-knowledge.
    pub from_processor_eval_initial: XFieldElement,
    pub to_processor_eval_initial: XFieldElement,
}
