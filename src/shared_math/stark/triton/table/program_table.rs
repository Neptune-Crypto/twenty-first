use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::table::base_matrix::ProgramTableColumn;
use crate::shared_math::x_field_element::XFieldElement;

pub const PROGRAM_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const PROGRAM_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 1;
pub const PROGRAM_TABLE_INITIALS_COUNT: usize =
    PROGRAM_TABLE_PERMUTATION_ARGUMENTS_COUNT + PROGRAM_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 2 because it combines: addr, instruction
pub const PROGRAM_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 2;

pub const BASE_WIDTH: usize = 2;
pub const FULL_WIDTH: usize = 4; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct ProgramTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for ProgramTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtProgramTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtProgramTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for ProgramTable {
    fn name(&self) -> String {
        "ProgramTable".to_string()
    }

    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let mut padding_row = data.last().unwrap().clone();
            // address keeps increasing
            padding_row[ProgramTableColumn::Address as usize] += 1.into();
            data.push(padding_row);
        }
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<BWord>> {
        vec![]
    }
}

impl Table<XFieldElement> for ExtProgramTable {
    fn name(&self) -> String {
        "ExtProgramTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }

    fn base_transition_constraints(&self) -> Vec<MPolynomial<XWord>> {
        vec![]
    }
}

impl ExtensionTable for ExtProgramTable {
    fn base_width(&self) -> usize {
        BASE_WIDTH
    }

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

impl ProgramTable {
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
        challenges: &ProgramTableChallenges,
        initials: &ProgramTableEndpoints,
    ) -> (ExtProgramTable, ProgramTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_sum = initials.instruction_eval_sum;

        for row in self.data().iter() {
            let (address, instruction) = (
                row[ProgramTableColumn::Address as usize].lift(),
                row[ProgramTableColumn::Instruction as usize].lift(),
            );

            let (address_w, instruction_w) =
                (challenges.address_weight, challenges.instruction_weight);

            // 1. Compress multiple values within one row so they become one value.
            let compressed_row_for_evaluation_argument =
                address * address_w + instruction * instruction_w;

            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));
            extension_row.push(compressed_row_for_evaluation_argument);

            // 2. Compute the running *product* of the compressed column (permutation value)
            extension_row.push(running_sum);
            running_sum = running_sum * challenges.instruction_eval_row_weight
                + compressed_row_for_evaluation_argument;

            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtProgramTable { base };
        let terminals = ProgramTableEndpoints {
            instruction_eval_sum: running_sum,
        };

        (table, terminals)
    }
}

impl ExtProgramTable {
    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain);
        let base = self.base.with_data(ext_codewords);

        ExtProgramTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct ProgramTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the program table.
    pub instruction_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to program table.)
    pub address_weight: XFieldElement,
    pub instruction_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct ProgramTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub instruction_eval_sum: XFieldElement,
}
