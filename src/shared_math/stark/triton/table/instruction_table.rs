use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_initials::{AllChallenges, AllInitials};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::x_field_element::XFieldElement;

pub const INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_INITIALS_COUNT: usize =
    INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT + INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 3 because it combines: ip, ci, nia
pub const INSTRUCTION_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 5;

pub const BASE_WIDTH: usize = 3;
pub const FULL_WIDTH: usize = 5; // BASE + INITIALS

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
        challenges: &InstructionTableChallenges,
        initials: &InstructionTableInitials,
    ) -> ExtInstructionTable {
        todo!()
    }
}

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

pub struct InstructionTableInitials {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_initial: XFieldElement,
    pub program_eval_initial: XFieldElement,
}
