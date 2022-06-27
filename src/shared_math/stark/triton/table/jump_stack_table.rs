use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use super::table_column::JumpStackTableColumn::*;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::instruction::Instruction;
use crate::shared_math::x_field_element::XFieldElement;

pub const JUMP_STACK_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const JUMP_STACK_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const JUMP_STACK_TABLE_INITIALS_COUNT: usize =
    JUMP_STACK_TABLE_PERMUTATION_ARGUMENTS_COUNT + JUMP_STACK_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 5 because it combines: clk, ci, jsp, jso, jsd,
pub const JUMP_STACK_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 5;

pub const BASE_WIDTH: usize = 5;
pub const FULL_WIDTH: usize = 7; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct JumpStackTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for JumpStackTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtJumpStackTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtJumpStackTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for JumpStackTable {
    fn name(&self) -> String {
        "JumpStackTable".to_string()
    }

    // FIXME: Apply correct padding, not just 0s.
    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let mut padding_row = data.last().unwrap().clone();
            // add same clk padding as in processor table
            padding_row[CLK as usize] = ((data.len() - 1) as u32).into();
            data.push(padding_row);
        }
    }
}

impl Table<XFieldElement> for ExtJumpStackTable {
    fn name(&self) -> String {
        "ExtJumpStackTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }
}

impl ExtensionTable for ExtJumpStackTable {
    fn base_width(&self) -> usize {
        BASE_WIDTH
    }

    fn ext_boundary_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        vec![]
    }

    fn ext_consistency_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
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

impl JumpStackTable {
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
        challenges: &JumpStackTableChallenges,
        initials: &JumpStackTableEndpoints,
    ) -> (ExtJumpStackTable, JumpStackTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product = initials.processor_perm_product;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            let (clk, ci, jsp, jso, jsd) = (
                extension_row[CLK as usize],
                extension_row[CI as usize],
                extension_row[JSP as usize],
                extension_row[JSO as usize],
                extension_row[JSD as usize],
            );

            let (clk_w, ci_w, jsp_w, jso_w, jsd_w) = (
                challenges.clk_weight,
                challenges.ci_weight,
                challenges.jsp_weight,
                challenges.jso_weight,
                challenges.jsd_weight,
            );

            // 1. Compress multiple values within one row so they become one value.
            let compressed_row_for_permutation_argument =
                clk * clk_w + ci * ci_w + jsp * jsp_w + jso * jso_w + jsd * jsd_w;

            extension_row.push(compressed_row_for_permutation_argument);

            // 2. Compute the running *product* of the compressed column (permutation value)
            extension_row.push(running_product);
            running_product *=
                challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;

            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtJumpStackTable { base };
        let terminals = JumpStackTableEndpoints {
            processor_perm_product: running_product,
        };

        (table, terminals)
    }
}

impl ExtJumpStackTable {
    pub fn with_padded_height(
        generator: XWord,
        order: usize,
        num_randomizers: usize,
        padded_height: usize,
    ) -> Self {
        let matrix: Vec<Vec<XWord>> = vec![];

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

    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain);
        let base = self.base.with_data(ext_codewords);

        ExtJumpStackTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct JumpStackTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the op-stack table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ci_weight: XFieldElement,
    pub jsp_weight: XFieldElement,
    pub jso_weight: XFieldElement,
    pub jsd_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct JumpStackTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_product: XFieldElement,
}

pub struct JumpStackConstraintPolynomialFactory {
    variables: [MPolynomial<BWord>; 2 * BASE_WIDTH],
}

impl Default for JumpStackConstraintPolynomialFactory {
    fn default() -> Self {
        let variables = MPolynomial::<BWord>::variables(2 * BASE_WIDTH, 1.into())
            .try_into()
            .expect("Dynamic conversion of polynomials for each variable to known-width set of variables");

        Self { variables }
    }
}

impl JumpStackConstraintPolynomialFactory {
    // Boundary Constraint(s)
    pub fn all_registers_initially_zero(&self) -> MPolynomial<BWord> {
        let clk = self.clk();
        let ci = self.ci();
        let jsp = self.jsp();
        let jso = self.jso();
        let jsd = self.jsd();

        clk.square() + ci.square() + jsp.square() + jso.square() + jsd.square()
    }

    // Transition Constraints
    pub fn transition_constraints(&self) -> MPolynomial<BWord> {
        let case_1 = self.jsp_next() - self.jsp() - self.one();

        let case_2 = (self.clk_next() - self.clk() - self.one())
            + (self.jsp() - self.jsp_next()).square()
            + (self.jso() - self.jso_next()).square()
            + (self.jsd() - self.jsd_next()).square();

        let call_opcode = Instruction::Call(BFieldElement::ring_zero()).opcode_b();
        debug_assert_eq!(call_opcode, 11.into());

        let call_mpol = MPolynomial::from_constant(call_opcode, 2 * BASE_WIDTH);
        let case_3 = (self.ci() - call_mpol).square()
            + (self.jsp() - self.jsp_next()).square()
            + (self.jso() - self.jso_next()).square()
            + (self.jsd() - self.jsd_next()).square();

        let return_opcode = Instruction::Return.opcode_b();
        debug_assert_eq!(return_opcode, 12.into());

        let return_mpol = MPolynomial::from_constant(return_opcode, 2 * BASE_WIDTH);
        let case_4 = (self.jsp() - self.jsp_next()).square() + (self.ci() - return_mpol).square();

        case_1 * case_2 * case_3 * case_4
    }

    pub fn one(&self) -> MPolynomial<BWord> {
        MPolynomial::from_constant(1.into(), 2 * BASE_WIDTH)
    }

    pub fn clk(&self) -> MPolynomial<BWord> {
        self.variables[CLK as usize].clone()
    }

    fn ci(&self) -> MPolynomial<BWord> {
        self.variables[CI as usize].clone()
    }

    fn jsp(&self) -> MPolynomial<BWord> {
        self.variables[JSP as usize].clone()
    }

    fn jsd(&self) -> MPolynomial<BWord> {
        self.variables[JSD as usize].clone()
    }

    fn jso(&self) -> MPolynomial<BWord> {
        self.variables[JSO as usize].clone()
    }

    pub fn clk_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + CLK as usize].clone()
    }

    pub fn ci_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + CI as usize].clone()
    }

    pub fn jsp_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + JSP as usize].clone()
    }

    pub fn jso_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + JSO as usize].clone()
    }

    pub fn jsd_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + JSD as usize].clone()
    }
}
