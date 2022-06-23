use std::collections::HashMap;

use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use super::table_column::ProcessorTableColumn::{self, *};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::instruction::{
    all_instructions, AnInstruction::*, Instruction,
};
use crate::shared_math::stark::triton::ord_n::Ord16;
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::x_field_element::XFieldElement;

pub const PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 9;
pub const PROCESSOR_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 4;
pub const PROCESSOR_TABLE_INITIALS_COUNT: usize =
    PROCESSOR_TABLE_PERMUTATION_ARGUMENTS_COUNT + PROCESSOR_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 47 because it combines all other tables (except program).
pub const PROCESSOR_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 47;

pub const BASE_WIDTH: usize = 38;
/// BASE_WIDTH + 2 * INITIALS_COUNT - 2 (because IOSymbols don't need compression)
pub const FULL_WIDTH: usize = 62;

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
        challenges: &ProcessorTableChallenges,
        initials: &ProcessorTableEndpoints,
    ) -> (ExtProcessorTable, ProcessorTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());

        let mut input_table_running_sum = initials.input_table_eval_sum;
        let mut output_table_running_sum = initials.output_table_eval_sum;
        let mut instruction_table_running_product = initials.instruction_table_perm_product;
        let mut opstack_table_running_product = initials.opstack_table_perm_product;
        let mut ram_table_running_product = initials.ram_table_perm_product;
        let mut jump_stack_running_product = initials.jump_stack_perm_product;
        let mut to_hash_table_running_sum = initials.to_hash_table_eval_sum;
        let mut from_hash_table_running_sum = initials.from_hash_table_eval_sum;
        let mut u32_table_lt_running_product = initials.u32_table_lt_perm_product;
        let mut u32_table_and_running_product = initials.u32_table_and_perm_product;
        let mut u32_table_xor_running_product = initials.u32_table_xor_perm_product;
        let mut u32_table_reverse_running_product = initials.u32_table_reverse_perm_product;
        let mut u32_table_div_running_product = initials.u32_table_div_perm_product;

        let mut previous_row: Option<Vec<BFieldElement>> = None;
        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // Input table
            extension_row.push(input_table_running_sum);
            if let Some(prow) = previous_row.clone() {
                if prow[CI as usize] == Instruction::ReadIo.opcode_b() {
                    let input_symbol = extension_row[ST0 as usize];
                    input_table_running_sum = input_table_running_sum
                        * challenges.input_table_eval_row_weight
                        + input_symbol;
                }
            }

            // Output table
            extension_row.push(output_table_running_sum);
            if row[CI as usize] == Instruction::WriteIo.opcode_b() {
                let output_symbol = extension_row[ST0 as usize];
                output_table_running_sum = output_table_running_sum
                    * challenges.output_table_eval_row_weight
                    + output_symbol;
            }

            // Instruction table
            let ip = extension_row[IP as usize];
            let ci = extension_row[CI as usize];
            let nia = extension_row[NIA as usize];

            let ip_w = challenges.instruction_table_ip_weight;
            let ci_w = challenges.instruction_table_ci_processor_weight;
            let nia_w = challenges.instruction_table_nia_weight;

            let compressed_row_for_instruction_table_permutation_argument =
                ip * ip_w + ci * ci_w + nia * nia_w;
            extension_row.push(compressed_row_for_instruction_table_permutation_argument);

            extension_row.push(instruction_table_running_product);
            instruction_table_running_product *= challenges.instruction_perm_row_weight
                - compressed_row_for_instruction_table_permutation_argument;

            // OpStack table
            let clk = extension_row[CLK as usize];
            let osv = extension_row[OSV as usize];
            let osp = extension_row[OSP as usize];

            let compressed_row_for_op_stack_table_permutation_argument = clk
                * challenges.op_stack_table_clk_weight
                + ci * challenges.op_stack_table_ci_weight
                + osv * challenges.op_stack_table_osv_weight
                + osp * challenges.op_stack_table_osp_weight;
            extension_row.push(compressed_row_for_op_stack_table_permutation_argument);

            extension_row.push(opstack_table_running_product);
            opstack_table_running_product *= challenges.op_stack_perm_row_weight
                - compressed_row_for_op_stack_table_permutation_argument;

            // RAM Table
            let ramv = extension_row[RAMV as usize];
            let ramp = extension_row[ST1 as usize];

            let compressed_row_for_ram_table_permutation_argument = clk
                * challenges.ram_table_clk_weight
                + ramv * challenges.ram_table_ramv_weight
                + ramp * challenges.ram_table_ramp_weight;
            extension_row.push(compressed_row_for_ram_table_permutation_argument);

            extension_row.push(ram_table_running_product);
            ram_table_running_product *=
                challenges.ram_perm_row_weight - compressed_row_for_ram_table_permutation_argument;

            // JumpStack Table
            let jsp = extension_row[JSP as usize];
            let jso = extension_row[JSO as usize];
            let jsd = extension_row[JSD as usize];
            let compressed_row_for_jump_stack_table = clk * challenges.jump_stack_table_clk_weight
                + ci * challenges.jump_stack_table_ci_weight
                + jsp * challenges.jump_stack_table_jsp_weight
                + jso * challenges.jump_stack_table_jso_weight
                + jsd * challenges.jump_stack_table_jsd_weight;
            extension_row.push(compressed_row_for_jump_stack_table);

            extension_row.push(jump_stack_running_product);
            jump_stack_running_product *=
                challenges.jump_stack_perm_row_weight - compressed_row_for_jump_stack_table;

            // Hash Table – Hash's input from Processor to Hash Coprocessor
            let st_0_through_11 = [
                extension_row[ST0 as usize],
                extension_row[ST1 as usize],
                extension_row[ST2 as usize],
                extension_row[ST3 as usize],
                extension_row[ST4 as usize],
                extension_row[ST5 as usize],
                extension_row[ST6 as usize],
                extension_row[ST7 as usize],
                extension_row[ST8 as usize],
                extension_row[ST9 as usize],
                extension_row[ST10 as usize],
                extension_row[ST11 as usize],
            ];
            let compressed_row_for_hash_input = st_0_through_11
                .iter()
                .zip(challenges.hash_table_stack_input_weights.iter())
                .map(|(st, weight)| *weight * *st)
                .fold(XWord::ring_zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_row_for_hash_input);

            extension_row.push(to_hash_table_running_sum);
            if row[CI as usize] == Instruction::Hash.opcode_b() {
                to_hash_table_running_sum = to_hash_table_running_sum
                    * challenges.to_hash_table_eval_row_weight
                    + compressed_row_for_hash_input;
            }

            // Hash Table – Hash's output from Hash Coprocessor to Processor
            let st_0_through_5 = [
                extension_row[ST0 as usize],
                extension_row[ST1 as usize],
                extension_row[ST2 as usize],
                extension_row[ST3 as usize],
                extension_row[ST4 as usize],
                extension_row[ST5 as usize],
            ];
            let compressed_row_for_hash_digest = st_0_through_5
                .iter()
                .zip(challenges.hash_table_digest_output_weights.iter())
                .map(|(st, weight)| *weight * *st)
                .fold(XWord::ring_zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_row_for_hash_digest);

            extension_row.push(from_hash_table_running_sum);
            if let Some(prow) = previous_row.clone() {
                if prow[CI as usize] == Instruction::Hash.opcode_b() {
                    from_hash_table_running_sum = from_hash_table_running_sum
                        * challenges.to_hash_table_eval_row_weight
                        + compressed_row_for_hash_digest;
                }
            }

            // U32 Table
            if let Some(prow) = previous_row {
                let lhs = prow[ST0 as usize].lift();
                let rhs = prow[ST1 as usize].lift();
                let u32_op_result = extension_row[ST0 as usize];

                // less than
                let compressed_row_for_u32_lt = lhs * challenges.u32_op_table_lt_lhs_weight
                    + rhs * challenges.u32_op_table_lt_rhs_weight
                    + u32_op_result * challenges.u32_op_table_lt_result_weight;
                extension_row.push(compressed_row_for_u32_lt);

                extension_row.push(u32_table_lt_running_product);
                if prow[CI as usize] == Instruction::Lt.opcode_b() {
                    u32_table_lt_running_product *=
                        challenges.u32_lt_perm_row_weight - compressed_row_for_u32_lt;
                }

                // and
                let compressed_row_for_u32_and = lhs * challenges.u32_op_table_and_lhs_weight
                    + rhs * challenges.u32_op_table_and_rhs_weight
                    + u32_op_result * challenges.u32_op_table_and_result_weight;
                extension_row.push(compressed_row_for_u32_and);

                extension_row.push(u32_table_and_running_product);
                if prow[CI as usize] == Instruction::And.opcode_b() {
                    u32_table_and_running_product *=
                        challenges.u32_and_perm_row_weight - compressed_row_for_u32_and;
                }

                // xor
                let compressed_row_for_u32_xor = lhs * challenges.u32_op_table_xor_lhs_weight
                    + rhs * challenges.u32_op_table_xor_rhs_weight
                    + u32_op_result * challenges.u32_op_table_xor_result_weight;
                extension_row.push(compressed_row_for_u32_xor);

                extension_row.push(u32_table_xor_running_product);
                if prow[CI as usize] == Instruction::Xor.opcode_b() {
                    u32_table_xor_running_product *=
                        challenges.u32_xor_perm_row_weight - compressed_row_for_u32_xor;
                }

                // reverse
                let compressed_row_for_u32_reverse = lhs
                    * challenges.u32_op_table_reverse_lhs_weight
                    + u32_op_result * challenges.u32_op_table_reverse_result_weight;
                extension_row.push(compressed_row_for_u32_reverse);

                extension_row.push(u32_table_reverse_running_product);
                if prow[CI as usize] == Instruction::Reverse.opcode_b() {
                    u32_table_reverse_running_product *=
                        challenges.u32_reverse_perm_row_weight - compressed_row_for_u32_reverse;
                }

                // div
                let divisor = prow[ST0 as usize].lift();
                let remainder = extension_row[ST0 as usize];
                let lt_for_div_result = extension_row[HV0 as usize];
                let compressed_row_for_u32_div = divisor
                    * challenges.u32_op_table_div_divisor_weight
                    + remainder * challenges.u32_op_table_div_remainder_weight
                    + lt_for_div_result * challenges.u32_op_table_div_result_weight;
                extension_row.push(compressed_row_for_u32_div);

                extension_row.push(u32_table_div_running_product);
                if prow[CI as usize] == Instruction::Div.opcode_b() {
                    u32_table_div_running_product *=
                        challenges.u32_lt_perm_row_weight - compressed_row_for_u32_div;
                }
            } else {
                // If there is no previous row, none of the u32 operations make sense. The extension
                // columns must still be filled in. All stack registers are initialized to 0, and
                // the stack in the non-existing previous row can be safely assumed to be all 0.
                // Thus, all the compressed_row-values are 0 for the very first extension_row.
                // The running products can be used as-are, amounting to pushing the initials.
                let zero = XFieldElement::ring_zero();
                extension_row.push(zero);
                extension_row.push(u32_table_lt_running_product);
                extension_row.push(zero);
                extension_row.push(u32_table_and_running_product);
                extension_row.push(zero);
                extension_row.push(u32_table_xor_running_product);
                extension_row.push(zero);
                extension_row.push(u32_table_reverse_running_product);
                extension_row.push(zero);
                extension_row.push(u32_table_div_running_product);
            }

            debug_assert_eq!(
                FULL_WIDTH,
                extension_row.len(),
                "After extending, the row must match the table's full width."
            );
            previous_row = Some(row.clone());
            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtProcessorTable { base };
        let terminals = ProcessorTableEndpoints {
            input_table_eval_sum: input_table_running_sum,
            output_table_eval_sum: output_table_running_sum,
            instruction_table_perm_product: instruction_table_running_product,
            opstack_table_perm_product: opstack_table_running_product,
            ram_table_perm_product: ram_table_running_product,
            jump_stack_perm_product: jump_stack_running_product,
            to_hash_table_eval_sum: to_hash_table_running_sum,
            from_hash_table_eval_sum: from_hash_table_running_sum,
            u32_table_lt_perm_product: u32_table_lt_running_product,
            u32_table_and_perm_product: u32_table_and_running_product,
            u32_table_xor_perm_product: u32_table_xor_running_product,
            u32_table_reverse_perm_product: u32_table_reverse_running_product,
            u32_table_div_perm_product: u32_table_div_running_product,
        };

        (table, terminals)
    }
}

impl ExtProcessorTable {
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
    pub u32_op_table_reverse_result_weight: XFieldElement,

    pub u32_op_table_div_divisor_weight: XFieldElement,
    pub u32_op_table_div_remainder_weight: XFieldElement,
    pub u32_op_table_div_result_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct ProcessorTableEndpoints {
    pub input_table_eval_sum: XFieldElement,
    pub output_table_eval_sum: XFieldElement,

    pub instruction_table_perm_product: XFieldElement,
    pub opstack_table_perm_product: XFieldElement,
    pub ram_table_perm_product: XFieldElement,
    pub jump_stack_perm_product: XFieldElement,

    pub to_hash_table_eval_sum: XFieldElement,
    pub from_hash_table_eval_sum: XFieldElement,

    pub u32_table_lt_perm_product: XFieldElement,
    pub u32_table_and_perm_product: XFieldElement,
    pub u32_table_xor_perm_product: XFieldElement,
    pub u32_table_reverse_perm_product: XFieldElement,
    pub u32_table_div_perm_product: XFieldElement,
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
            padding_row[ProcessorTableColumn::CLK as usize] += 1.into();
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

pub struct ProcessorConstraintPolynomialFactory {
    variables: [MPolynomial<BWord>; 2 * BASE_WIDTH],
    deselectors: HashMap<Instruction, MPolynomial<BWord>>,
}

impl Default for ProcessorConstraintPolynomialFactory {
    fn default() -> Self {
        let variables = MPolynomial::<BWord>::variables(2 * BASE_WIDTH, 1.into())
            .try_into()
            .expect("Dynamic conversion of polynomials for each variable to known-width set of variables");

        let deselectors = Self::all_instruction_deselectors();

        Self {
            variables,
            deselectors,
        }
    }
}

impl ProcessorConstraintPolynomialFactory {
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

    pub fn indicator_polynomial(&self, i: usize) -> MPolynomial<BWord> {
        let hv0 = self.hv0();
        let hv1 = self.hv1();
        let hv2 = self.hv2();
        let hv3 = self.hv3();

        match i {
            0 => (self.one() - hv3) * (self.one() - hv2) * (self.one() - hv1) * (self.one() - hv0),
            1 => (self.one() - hv3) * (self.one() - hv2) * (self.one() - hv1) * hv0,
            2 => (self.one() - hv3) * (self.one() - hv2) * hv1 * (self.one() - hv0),
            3 => (self.one() - hv3) * (self.one() - hv2) * hv1 * hv0,
            4 => (self.one() - hv3) * hv2 * (self.one() - hv1) * (self.one() - hv0),
            5 => (self.one() - hv3) * hv2 * (self.one() - hv1) * hv0,
            6 => (self.one() - hv3) * hv2 * hv1 * (self.one() - hv0),
            7 => (self.one() - hv3) * hv2 * hv1 * hv0,
            8 => hv3 * (self.one() - hv2) * (self.one() - hv1) * (self.one() - hv0),
            9 => hv3 * (self.one() - hv2) * (self.one() - hv1) * hv0,
            10 => hv3 * (self.one() - hv2) * hv1 * (self.one() - hv0),
            11 => hv3 * (self.one() - hv2) * hv1 * hv0,
            12 => hv3 * hv2 * (self.one() - hv1) * (self.one() - hv0),
            13 => hv3 * hv2 * (self.one() - hv1) * hv0,
            14 => hv3 * hv2 * hv1 * (self.one() - hv0),
            15 => hv3 * hv2 * hv1 * hv0,
            _ => panic!(
                "No indicator polynomial with index {} exists: there are only 16.",
                i
            ),
        }
    }

    pub fn instruction_pop(&self) -> Vec<MPolynomial<BWord>> {
        // no further constraints
        vec![]
    }

    pub fn instruction_push(&self) -> Vec<MPolynomial<BWord>> {
        let st0_next = self.st0_next();
        let nia = self.nia();

        // push'es argument should be on the stack after execution
        // st0_next == nia  =>  st0_next - nia == 0
        vec![st0_next - nia]
    }

    pub fn instruction_divine(&self) -> Vec<MPolynomial<BWord>> {
        // no further constraints
        vec![]
    }

    pub fn instruction_dup(&self) -> Vec<MPolynomial<BWord>> {
        vec![
            self.indicator_polynomial(0) * (self.st0_next() - self.st0()),
            self.indicator_polynomial(1) * (self.st0_next() - self.st1()),
            self.indicator_polynomial(2) * (self.st0_next() - self.st2()),
            self.indicator_polynomial(3) * (self.st0_next() - self.st3()),
            self.indicator_polynomial(4) * (self.st0_next() - self.st4()),
            self.indicator_polynomial(5) * (self.st0_next() - self.st5()),
            self.indicator_polynomial(6) * (self.st0_next() - self.st6()),
            self.indicator_polynomial(7) * (self.st0_next() - self.st7()),
            self.indicator_polynomial(8) * (self.st0_next() - self.st8()),
            self.indicator_polynomial(9) * (self.st0_next() - self.st9()),
            self.indicator_polynomial(10) * (self.st0_next() - self.st10()),
            self.indicator_polynomial(11) * (self.st0_next() - self.st11()),
            self.indicator_polynomial(12) * (self.st0_next() - self.st12()),
            self.indicator_polynomial(13) * (self.st0_next() - self.st13()),
            self.indicator_polynomial(14) * (self.st0_next() - self.st14()),
            self.indicator_polynomial(15) * (self.st0_next() - self.st15()),
        ]
    }

    pub fn instruction_swap(&self) -> Vec<MPolynomial<BWord>> {
        vec![
            self.one() - self.indicator_polynomial(0),
            self.indicator_polynomial(1) * (self.st1_next() - self.st0()),
            self.indicator_polynomial(2) * (self.st2_next() - self.st0()),
            self.indicator_polynomial(3) * (self.st3_next() - self.st0()),
            self.indicator_polynomial(4) * (self.st4_next() - self.st0()),
            self.indicator_polynomial(5) * (self.st5_next() - self.st0()),
            self.indicator_polynomial(6) * (self.st6_next() - self.st0()),
            self.indicator_polynomial(7) * (self.st7_next() - self.st0()),
            self.indicator_polynomial(8) * (self.st8_next() - self.st0()),
            self.indicator_polynomial(9) * (self.st9_next() - self.st0()),
            self.indicator_polynomial(10) * (self.st10_next() - self.st0()),
            self.indicator_polynomial(11) * (self.st11_next() - self.st0()),
            self.indicator_polynomial(12) * (self.st12_next() - self.st0()),
            self.indicator_polynomial(13) * (self.st13_next() - self.st0()),
            self.indicator_polynomial(14) * (self.st14_next() - self.st0()),
            self.indicator_polynomial(15) * (self.st15_next() - self.st0()),
            self.indicator_polynomial(1) * (self.st0_next() - self.st1()),
            self.indicator_polynomial(2) * (self.st0_next() - self.st2()),
            self.indicator_polynomial(3) * (self.st0_next() - self.st3()),
            self.indicator_polynomial(4) * (self.st0_next() - self.st4()),
            self.indicator_polynomial(5) * (self.st0_next() - self.st5()),
            self.indicator_polynomial(6) * (self.st0_next() - self.st6()),
            self.indicator_polynomial(7) * (self.st0_next() - self.st7()),
            self.indicator_polynomial(8) * (self.st0_next() - self.st8()),
            self.indicator_polynomial(9) * (self.st0_next() - self.st9()),
            self.indicator_polynomial(10) * (self.st0_next() - self.st10()),
            self.indicator_polynomial(11) * (self.st0_next() - self.st11()),
            self.indicator_polynomial(12) * (self.st0_next() - self.st12()),
            self.indicator_polynomial(13) * (self.st0_next() - self.st13()),
            self.indicator_polynomial(14) * (self.st0_next() - self.st14()),
            self.indicator_polynomial(15) * (self.st0_next() - self.st15()),
            (self.one() - self.indicator_polynomial(1)) * (self.st1_next() - self.st1()),
            (self.one() - self.indicator_polynomial(2)) * (self.st2_next() - self.st2()),
            (self.one() - self.indicator_polynomial(3)) * (self.st3_next() - self.st3()),
            (self.one() - self.indicator_polynomial(4)) * (self.st4_next() - self.st4()),
            (self.one() - self.indicator_polynomial(5)) * (self.st5_next() - self.st5()),
            (self.one() - self.indicator_polynomial(6)) * (self.st6_next() - self.st6()),
            (self.one() - self.indicator_polynomial(7)) * (self.st7_next() - self.st7()),
            (self.one() - self.indicator_polynomial(8)) * (self.st8_next() - self.st8()),
            (self.one() - self.indicator_polynomial(9)) * (self.st9_next() - self.st9()),
            (self.one() - self.indicator_polynomial(10)) * (self.st10_next() - self.st10()),
            (self.one() - self.indicator_polynomial(11)) * (self.st11_next() - self.st11()),
            (self.one() - self.indicator_polynomial(12)) * (self.st12_next() - self.st12()),
            (self.one() - self.indicator_polynomial(13)) * (self.st13_next() - self.st13()),
            (self.one() - self.indicator_polynomial(14)) * (self.st14_next() - self.st14()),
            (self.one() - self.indicator_polynomial(15)) * (self.st15_next() - self.st15()),
        ]
    }

    pub fn instruction_nop(&self) -> Vec<MPolynomial<BWord>> {
        // no further constraints
        vec![]
    }

    pub fn instruction_skiz(&self) -> Vec<MPolynomial<BWord>> {
        // The jump stack pointer jsp does not change.
        let jsp_does_not_change = self.jsp_next() - self.jsp();

        // The last jump's origin jso does not change.
        let jso_does_not_change = self.jso_next() - self.jso();

        // The last jump's destination jsd does not change.
        let jsd_does_not_change = self.jsd_next() - self.jsd();

        // The next instruction nia is decomposed into helper variables hv.
        let nia_decomposes_to_hvs = self.nia() - (self.hv0() + self.two() * self.hv1());

        // The relevant helper variable hv0 is either 0 or 1. Here, hv0 == 1 means that nia takes an argument.
        let hv0_is_0_or_1 = self.hv0() * (self.hv0() - self.one());

        // Register ip increments by (1 if st0 is non-zero else (2 if nia takes no argument else 3)).
        let ip_incr_by_1_or_2_or_3 = self.ip_next()
            - (self.ip() + self.one() + self.st0() * self.inv() * (self.one() + self.hv0()));

        let two = self.one() + self.one();
        vec![
            jsp_does_not_change,
            jso_does_not_change,
            jsd_does_not_change,
            nia_decomposes_to_hvs,
            hv0_is_0_or_1,
            ip_incr_by_1_or_2_or_3,
        ]
    }

    // 1. Create stubs for all instruction polynomials
    pub fn instruction_call(&self) -> Vec<MPolynomial<BWord>> {
        // The jump stack pointer jsp is incremented by 1.
        let jsp_incr_1 = self.jsp_next() - (self.jsp() + self.one());

        // The jump's origin jso is set to the current instruction pointer ip plus 2.
        let jso_becomes_ip_plus_2 = self.jso_next() - (self.ip() + self.two());

        // The jump's destination jsd is set to the instruction's argument.
        let jsd_becomes_nia = self.jsd_next() - self.nia();

        // The instruction pointer ip is set to the instruction's argument.
        let ip_becomes_nia = self.ip_next() - self.nia();

        vec![
            jsp_incr_1,
            jso_becomes_ip_plus_2,
            jsd_becomes_nia,
            ip_becomes_nia,
        ]
    }

    pub fn instruction_return(&self) -> Vec<MPolynomial<BWord>> {
        // The jump stack pointer jsp is decremented by 1.
        let jsp_incr_1 = self.jsp_next() - (self.jsp() - self.one());

        // The instruction pointer ip is set to the last call's origin jso.
        let ip_becomes_jso = self.ip_next() - self.jso();

        vec![jsp_incr_1, ip_becomes_jso]
    }

    pub fn instruction_recurse(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_assert(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_halt(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_read_mem(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_write_mem(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_hash(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_divine_sibling(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_assert_vector(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_add(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_mul(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_invert(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_split(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_eq(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_lt(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_and(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_xor(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_reverse(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_div(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_xxadd(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_xxmul(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_xinv(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_xbmul(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_read_io(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn instruction_write_io(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    // 2. Create stubs for all instruction group polynomials
    // 3. Find out where to combine deselectors, instruction polynomials and instruction group polynomials

    // FIXME: Consider caching this on first run (caching computed getter)
    pub fn one(&self) -> MPolynomial<BWord> {
        MPolynomial::from_constant(1.into(), 2 * BASE_WIDTH)
    }

    // FIXME: Consider caching this on first run (caching computed getter)
    pub fn two(&self) -> MPolynomial<BWord> {
        MPolynomial::from_constant(2.into(), 2 * BASE_WIDTH)
    }

    pub fn constant(&self, constant: u32) -> MPolynomial<BWord> {
        MPolynomial::from_constant(constant.into(), 2 * BASE_WIDTH)
    }

    pub fn clk(&self) -> MPolynomial<BWord> {
        self.variables[CLK as usize].clone()
    }
    fn ip(&self) -> MPolynomial<BWord> {
        self.variables[IP as usize].clone()
    }
    fn ci(&self) -> MPolynomial<BWord> {
        self.variables[CI as usize].clone()
    }
    fn nia(&self) -> MPolynomial<BWord> {
        self.variables[NIA as usize].clone()
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
    pub fn st0(&self) -> MPolynomial<BWord> {
        self.variables[ST0 as usize].clone()
    }
    pub fn st1(&self) -> MPolynomial<BWord> {
        self.variables[ST1 as usize].clone()
    }
    pub fn st2(&self) -> MPolynomial<BWord> {
        self.variables[ST2 as usize].clone()
    }
    pub fn st3(&self) -> MPolynomial<BWord> {
        self.variables[ST3 as usize].clone()
    }
    pub fn st4(&self) -> MPolynomial<BWord> {
        self.variables[ST4 as usize].clone()
    }
    pub fn st5(&self) -> MPolynomial<BWord> {
        self.variables[ST5 as usize].clone()
    }
    pub fn st6(&self) -> MPolynomial<BWord> {
        self.variables[ST6 as usize].clone()
    }
    pub fn st7(&self) -> MPolynomial<BWord> {
        self.variables[ST7 as usize].clone()
    }
    pub fn st8(&self) -> MPolynomial<BWord> {
        self.variables[ST8 as usize].clone()
    }
    pub fn st9(&self) -> MPolynomial<BWord> {
        self.variables[ST9 as usize].clone()
    }
    pub fn st10(&self) -> MPolynomial<BWord> {
        self.variables[ST10 as usize].clone()
    }
    pub fn st11(&self) -> MPolynomial<BWord> {
        self.variables[ST11 as usize].clone()
    }
    pub fn st12(&self) -> MPolynomial<BWord> {
        self.variables[ST12 as usize].clone()
    }
    pub fn st13(&self) -> MPolynomial<BWord> {
        self.variables[ST13 as usize].clone()
    }
    pub fn st14(&self) -> MPolynomial<BWord> {
        self.variables[ST14 as usize].clone()
    }
    pub fn st15(&self) -> MPolynomial<BWord> {
        self.variables[ST15 as usize].clone()
    }
    pub fn inv(&self) -> MPolynomial<BWord> {
        self.variables[INV as usize].clone()
    }
    pub fn osp(&self) -> MPolynomial<BWord> {
        self.variables[OSP as usize].clone()
    }
    pub fn osv(&self) -> MPolynomial<BWord> {
        self.variables[OSV as usize].clone()
    }
    fn hv0(&self) -> MPolynomial<BWord> {
        self.variables[HV0 as usize].clone()
    }
    fn hv1(&self) -> MPolynomial<BWord> {
        self.variables[HV1 as usize].clone()
    }
    fn hv2(&self) -> MPolynomial<BWord> {
        self.variables[HV2 as usize].clone()
    }
    fn hv3(&self) -> MPolynomial<BWord> {
        self.variables[HV3 as usize].clone()
    }
    fn hv4(&self) -> MPolynomial<BWord> {
        self.variables[HV4 as usize].clone()
    }
    fn ramv(&self) -> MPolynomial<BWord> {
        self.variables[RAMV as usize].clone()
    }

    // Property: All polynomial variables that contain '_next' have the same
    // variable position / value as the one without '_next', +/- BASE_WIDTH.
    pub fn clk_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + CLK as usize].clone()
    }
    fn ip_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + IP as usize].clone()
    }
    fn _ci_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + CI as usize].clone()
    }
    fn jsp_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + JSP as usize].clone()
    }
    fn jsd_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + JSD as usize].clone()
    }
    fn jso_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + JSO as usize].clone()
    }
    pub fn st0_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST0 as usize].clone()
    }
    pub fn st1_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST1 as usize].clone()
    }
    pub fn st2_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST2 as usize].clone()
    }
    pub fn st3_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST3 as usize].clone()
    }
    pub fn st4_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST4 as usize].clone()
    }
    pub fn st5_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST5 as usize].clone()
    }
    pub fn st6_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST6 as usize].clone()
    }
    pub fn st7_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST7 as usize].clone()
    }
    pub fn st8_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST8 as usize].clone()
    }
    pub fn st9_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST9 as usize].clone()
    }
    pub fn st10_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST10 as usize].clone()
    }
    pub fn st11_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST11 as usize].clone()
    }
    pub fn st12_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST12 as usize].clone()
    }
    pub fn st13_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST13 as usize].clone()
    }
    pub fn st14_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST14 as usize].clone()
    }
    pub fn st15_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + ST15 as usize].clone()
    }
    pub fn osp_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + OSP as usize].clone()
    }
    pub fn osv_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + OSV as usize].clone()
    }
    fn ramv_next(&self) -> MPolynomial<BWord> {
        self.variables[BASE_WIDTH + RAMV as usize].clone()
    }

    pub fn decompose_arg(&self) -> MPolynomial<BWord> {
        todo!()
    }

    pub fn step_1(&self) -> Vec<MPolynomial<BWord>> {
        let one = self.one();
        let ip = self.ip();
        let ip_next = self.ip_next();

        vec![ip_next - ip - one]
    }

    pub fn step_2(&self) -> Vec<MPolynomial<BWord>> {
        let one = self.one();
        let ip = self.ip();
        let ip_next = self.ip_next();

        vec![ip_next - ip - (one.clone() + one)]
    }

    pub fn u32_op(&self) -> Vec<MPolynomial<BWord>> {
        // This group has no constraints. It is used for the Permutation Argument with the uint32 table.
        vec![]
    }

    pub fn grow_stack(&self) -> Vec<MPolynomial<BWord>> {
        vec![
            // The stack element in st0 is moved into st1.
            self.st1_next() - self.st0(),
            // The stack element in st1 is moved into st2.
            self.st2_next() - self.st1(),
            // etc
            self.st3_next() - self.st2(),
            self.st4_next() - self.st3(),
            self.st5_next() - self.st4(),
            self.st6_next() - self.st5(),
            self.st7_next() - self.st6(),
            self.st8_next() - self.st7(),
            self.st9_next() - self.st8(),
            self.st10_next() - self.st9(),
            self.st11_next() - self.st10(),
            self.st12_next() - self.st11(),
            self.st13_next() - self.st12(),
            self.st14_next() - self.st13(),
            self.st15_next() - self.st14(),
            // The stack element in st15 is moved to the top of OpStack underflow, i.e., osv.
            self.osv_next() - self.st15(),
            // The OpStack pointer is incremented by 1.
            self.osp_next() - (self.osp_next() + self.one()),
        ]
    }

    pub fn keep_stack(&self) -> Vec<MPolynomial<BWord>> {
        vec![
            self.st0_next() - self.st0(),
            self.st1_next() - self.st1(),
            self.st2_next() - self.st2(),
            self.st3_next() - self.st3(),
            self.st4_next() - self.st4(),
            self.st5_next() - self.st5(),
            self.st6_next() - self.st6(),
            self.st7_next() - self.st7(),
            self.st8_next() - self.st8(),
            self.st9_next() - self.st9(),
            self.st10_next() - self.st10(),
            self.st11_next() - self.st11(),
            self.st12_next() - self.st12(),
            self.st13_next() - self.st13(),
            self.st14_next() - self.st14(),
            self.st15_next() - self.st15(),
            self.osv_next() - self.osv(),
            self.osp_next() - self.osp(),
            self.ramv_next() - self.ramv(),
        ]
    }

    pub fn shrink_stack(&self) -> Vec<MPolynomial<BWord>> {
        vec![
            self.st0_next() - self.st1(),
            self.st1_next() - self.st2(),
            self.st2_next() - self.st3(),
            self.st3_next() - self.st4(),
            self.st4_next() - self.st5(),
            self.st5_next() - self.st6(),
            self.st6_next() - self.st7(),
            self.st7_next() - self.st8(),
            self.st8_next() - self.st9(),
            self.st9_next() - self.st10(),
            self.st10_next() - self.st11(),
            self.st11_next() - self.st12(),
            self.st12_next() - self.st13(),
            self.st13_next() - self.st14(),
            self.st14_next() - self.st15(),
            self.st15_next() - self.osv(),
            self.osp_next() - (self.osp() - self.one()),
            (self.osp_next() - self.constant(15)) * self.hv4() - self.one(),
        ]
    }

    pub fn unop(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }

    pub fn binop(&self) -> Vec<MPolynomial<BWord>> {
        todo!()
    }
}

impl ProcessorConstraintPolynomialFactory {
    pub fn instruction_deselector(&self, instruction: Instruction) -> MPolynomial<BWord> {
        self.deselectors[&instruction].clone()
    }

    fn all_instruction_deselectors() -> HashMap<Instruction, MPolynomial<BWord>> {
        let instructions_as_mpoly = Self::all_instructions_as_mpoly();
        let mut deselectors = HashMap::<Instruction, MPolynomial<BFieldElement>>::new();
        let one = MPolynomial::from_constant(1.into(), 2 * BASE_WIDTH);

        for deselected_instruction in all_instructions().into_iter() {
            let deselector = all_instructions()
                .into_iter()
                .filter(|instruction| *instruction != deselected_instruction)
                .map(|instruction| instructions_as_mpoly[&instruction].clone())
                .fold(one.clone(), |a, b| a + b);

            deselectors.insert(deselected_instruction, deselector);
        }

        deselectors
    }

    fn all_instructions_selector(&self) -> MPolynomial<BWord> {
        all_instructions()
            .into_iter()
            .map(|instruction| self.instruction_selector(instruction))
            .fold(self.one(), |a, b| a + b)
    }

    fn instruction_selector(&self, instruction: Instruction) -> MPolynomial<BWord> {
        self.ci() - Self::instruction_as_mpoly(instruction)
    }

    fn instruction_as_mpoly(instruction: Instruction) -> MPolynomial<BWord> {
        MPolynomial::from_constant(instruction.opcode_b(), 2 * BASE_WIDTH)
    }

    fn all_instructions_as_mpoly() -> HashMap<Instruction, MPolynomial<BWord>> {
        all_instructions()
            .into_iter()
            .map(|instruction| (instruction, Self::instruction_as_mpoly(instruction)))
            .collect()
    }
}
