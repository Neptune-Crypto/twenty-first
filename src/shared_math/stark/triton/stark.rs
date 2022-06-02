use super::super::triton;
use super::table::base_matrix::BaseMatrices;
use super::table::base_table::{HasBaseTable, Table};
use super::table::processor_table::{self, ProcessorTable};
use super::vm::Program;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_xlix::{RescuePrimeXlix, RP_DEFAULT_WIDTH};
use crate::shared_math::traits::GetPrimitiveRootOfUnity;
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::{other, xfri};

pub const EXTENSION_CHALLENGE_COUNT: usize = 0;
pub const PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const TERMINAL_COUNT: usize = 0;

type BWord = BFieldElement;
type XWord = XFieldElement;
type StarkHasher = RescuePrimeXlix<RP_DEFAULT_WIDTH>;

// We use a type-parameterised FriDomain to avoid duplicate `b_*()` and `x_*()` methods.
pub struct Stark {
    padded_height: usize,
    log_expansion_factor: usize,
    security_level: usize,
    fri_domain: triton::fri_domain::FriDomain<BWord>,
    fri: xfri::Fri<StarkHasher>,
}

impl Stark {
    pub fn new(padded_height: usize, log_expansion_factor: usize, security_level: usize) -> Self {
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

        let num_randomizers = 1;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();

        // TODO: Create empty table collection to derive max degree.

        let mut max_degree = todo!(); // rc_base_tables.borrow().get_max_degree();
        max_degree = other::roundup_npo2(max_degree) - 1;
        let fri_domain_length = (max_degree + 1) * expansion_factor;

        let offset = BWord::generator();
        let omega = BWord::ring_zero()
            .get_primitive_root_of_unity(fri_domain_length as u64)
            .0
            .unwrap();

        let fri_domain = triton::fri_domain::FriDomain {
            offset,
            omega,
            length: fri_domain_length,
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

        let mut processor_table = ProcessorTable::new(
            unpadded_height,
            num_randomizers,
            smooth_generator,
            order,
            base_matrices
                .processor_matrix
                .iter()
                .map(|row| row.to_vec())
                .collect(),
        );

        // 2. Pad matrix
        processor_table.pad();

        // 3. Create base codeword tables based on those
        // FIXME: Create triton::fri_domain::FriDomain<BWord> object
        let coded_processor_table = processor_table.codewords(&self.fri_domain);
    }
}
