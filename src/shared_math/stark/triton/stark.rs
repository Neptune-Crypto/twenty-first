use super::super::triton;
use super::table::base_matrix::BaseMatrices;
use super::table::base_table::Table;
use super::table::processor_table::ProcessorTable;
use super::vm::Program;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other::roundup_npo2;
use crate::shared_math::rescue_prime_xlix::{RescuePrimeXlix, RP_DEFAULT_WIDTH};
use crate::shared_math::stark::triton::instruction::sample_programs;
use crate::shared_math::stark::triton::table::table_collection::BaseTableCollection;
use crate::shared_math::traits::GetPrimitiveRootOfUnity;
use crate::shared_math::{other, xfri};

pub const PERMUTATION_ARGUMENTS_COUNT: usize = 10;
pub const EXTENSION_CHALLENGE_COUNT: usize = 2;
pub const TERMINAL_COUNT: usize = 0;

type BWord = BFieldElement;
type StarkHasher = RescuePrimeXlix<RP_DEFAULT_WIDTH>;

// We use a type-parameterised FriDomain to avoid duplicate `b_*()` and `x_*()` methods.
pub struct Stark {
    _padded_height: usize,
    _log_expansion_factor: usize,
    _security_level: usize,
    _fri_domain: triton::fri_domain::FriDomain<BWord>,
    _fri: xfri::Fri<StarkHasher>,
}

impl Stark {
    pub fn new(_padded_height: usize, log_expansion_factor: usize, security_level: usize) -> Self {
        assert_eq!(
            0,
            security_level % log_expansion_factor,
            "security_level/log_expansion_factor must be a positive integer"
        );

        let expansion_factor: u64 = 1 << log_expansion_factor;
        let colinearity_checks: usize = security_level / log_expansion_factor;

        assert!(
            colinearity_checks > 0,
            "At least one colinearity check is required"
        );

        assert!(
            expansion_factor >= 4,
            "expansion factor must be at least 4."
        );

        let num_randomizers = 2;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();

        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();

        let (base_matrices, _err) = program.simulate_with_input(&[], &[]);

        let base_table_collection = BaseTableCollection::from_base_matrices(
            smooth_generator,
            order,
            num_randomizers,
            &base_matrices,
        );

        let max_degree = other::roundup_npo2(base_table_collection.max_degree()) - 1;
        let fri_domain_length = ((max_degree + 1) * expansion_factor) as usize;

        let offset = BWord::generator();
        let omega = BWord::ring_zero()
            .get_primitive_root_of_unity(fri_domain_length as u64)
            .0
            .unwrap();

        let _fri_domain = triton::fri_domain::FriDomain {
            offset,
            omega,
            length: fri_domain_length as usize,
        };

        todo!()
    }

    pub fn prove(&self, base_matrices: BaseMatrices) {
        // 1. Create base tables based on base matrices

        // From brainfuck
        let num_randomizers = 1;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();
        let unpadded_height = base_matrices.processor_matrix.len();
        let _padded_height = roundup_npo2(unpadded_height as u64);

        let mut processor_table = ProcessorTable::new_prover(
            smooth_generator,
            order,
            num_randomizers,
            base_matrices
                .processor_matrix
                .iter()
                .map(|row| row.to_vec())
                .collect(),
        );

        // 2. Pad matrix
        processor_table.pad();

        // 3. Create base codeword tables based on those
        //let coded_processor_table = processor_table.codewords(&self.fri_domain);
        todo!()
    }

    fn sample_weights(
        hasher: &StarkHasher,
        seed: &StarkDigest,
        count: usize,
    ) -> Vec<XFieldElement> {
        hasher
            .get_n_hash_rounds(seed, count)
            .iter()
            .flat_map(|digest| {
                vec![
                    XFieldElement::new([digest[0], digest[1], digest[2]]),
                    XFieldElement::new([digest[3], digest[4], digest[5]]),
                ]
            })
            .collect()
    }
}
