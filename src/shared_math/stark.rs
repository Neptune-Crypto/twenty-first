use rand::RngCore;

use crate::shared_math::fri::ValidationError;
use crate::shared_math::traits::{
    CyclicGroupGenerator, FieldBatchInversion, GetGeneratorDomain, GetRandomElements,
    IdentityValues, ModPowU32, PrimeFieldElement,
};
use crate::shared_math::x_field_element::XFieldElement;
use crate::util_types::merkle_tree::{MerkleTree, PartialAuthenticationPath};
use crate::util_types::proof_stream::ProofStream;
use crate::utils::{blake3_digest, get_index_from_bytes};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use super::b_field_element::BFieldElement;
use super::mpolynomial::MPolynomial;
use super::other::log_2_ceil;
use super::polynomial::Polynomial;
use super::x_field_fri::Fri;
use crate::shared_math::ntt::intt;

pub const DOCUMENT_HASH_LENGTH: usize = 32usize;
pub const MERKLE_ROOT_HASH_LENGTH: usize = 32usize;

// TODO: Consider <B: PrimeFieldElement, X: PrimeFieldElement>
// This requires a trait a la Lift<X> to generalise XFE::new_const().
#[derive(Clone, Debug)]
pub struct Stark {
    expansion_factor: u32,
    fri: Fri<XFieldElement>,
    field_generator: BFieldElement,
    randomizer_count: u32,
    omega: BFieldElement,
    pub omicron: BFieldElement, // omicron = omega^expansion_factor
    omicron_domain: Vec<BFieldElement>,
    omicron_domain_length: u32,
    original_trace_length: u32,
    randomized_trace_length: u32,
    register_count: u32,
}

// TODO: Consider impl<B: PrimeFieldElement, X: PrimeFieldElement>
impl Stark {
    pub fn new(
        expansion_factor: u32,
        colinearity_check_count: u32,
        register_count: u32,
        cycle_count: u32,
        transition_constraints_degree: u32,
        field_generator: BFieldElement,
    ) -> Self {
        let num_randomizers = 4 * colinearity_check_count;
        let original_trace_length = cycle_count;
        let randomized_trace_length = original_trace_length + num_randomizers;

        // FIXME: Fix fri_domain_length according to this comment:
        //
        // The FRI domain needs to expansion_factor times larger than the degree of the
        // transition quotients, which is AIR_degree * trace_length - omicron_domain_length

        let omicron_domain_length = 1u32 << log_2_ceil(randomized_trace_length as u64);
        let fri_domain_length =
            omicron_domain_length * expansion_factor * transition_constraints_degree;
        let omega = BFieldElement::get_primitive_root_of_unity(fri_domain_length as u128)
            .0
            .unwrap();
        let omicron = omega.mod_pow(expansion_factor.into());

        // Verify omega and omicron values
        assert!(
            omicron.mod_pow(omicron_domain_length.into()).is_one(),
            "omicron must have correct order"
        );
        assert!(
            !omicron.mod_pow((omicron_domain_length / 2).into()).is_one(),
            "omicron must have correct primitive order"
        );
        assert!(
            omega.mod_pow(fri_domain_length.into()).is_one(),
            "omicron must have correct order"
        );
        assert!(
            !omega.mod_pow((fri_domain_length / 2).into()).is_one(),
            "omicron must have correct primitive order"
        );

        let omicron_domain = omicron.get_cyclic_group();

        let fri = Fri::new(
            XFieldElement::new_const(field_generator),
            XFieldElement::new_const(omega.clone()),
            fri_domain_length as usize,
            expansion_factor as usize,
            colinearity_check_count as usize,
        );

        Self {
            expansion_factor,
            field_generator,
            randomizer_count: num_randomizers,
            omega,
            omicron,
            omicron_domain,
            omicron_domain_length,
            original_trace_length,
            randomized_trace_length,
            register_count,
            fri,
        }
    }
}

// TODO: Make this work for XFieldElement via trait.
#[derive(Clone, Debug)]
pub struct BoundaryConstraint {
    pub cycle: usize,
    pub register: usize,
    pub value: BFieldElement,
}

// A hashmap from register value to (x, y) value of boundary constraint
pub type BoundaryConstraintsMap = HashMap<usize, (BFieldElement, BFieldElement)>;

#[derive(Clone, Debug)]
pub struct StarkPreprocessedValuesProver {
    transition_zerofier: Polynomial<BFieldElement>,
    transition_zerofier_mt: MerkleTree<BFieldElement>,
}

#[derive(Clone, Debug)]
pub struct StarkPreprocessedValues {
    transition_zerofier_mt_root: [u8; MERKLE_ROOT_HASH_LENGTH],
    prover: Option<StarkPreprocessedValuesProver>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StarkProofError {
    InputOutputMismatch,
    HighDegreeExtendedComputationalTrace,
    HighDegreeBoundaryQuotient,
    HighDegreeTransitionQuotient,
    HighDegreeLinearCombination,
    MissingPreprocessedValues,
    NonZeroBoundaryRemainder,
    NonZeroTransitionRemainder,
}

impl Error for StarkProofError {}

impl fmt::Display for StarkProofError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MerkleProofError {
    BoundaryQuotientError(usize),
    RandomizerError,
    TransitionZerofierError,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StarkVerifyError {
    BadAirPaths,
    BadNextAirPaths,
    BadAirBoundaryIndentity(usize),
    BadAirTransitionIdentity(usize),
    BadBoundaryConditionAuthenticationPaths,
    BadMerkleProof(MerkleProofError),
    LinearCombinationAuthenticationPath,
    LinearCombinationMismatch(usize), // integer refers to first index where a mismatch is found
    LinearCombinationTupleMismatch(usize), // integer refers to first index where a mismatch is found
    InputOutputMismatch,
    HighDegreeExtendedComputationalTrace,
    HighDegreeBoundaryQuotient,
    HighDegreeTransitionQuotient,
    HighDegreeLinearCombination,
    MissingPreprocessedValues,
    NonZeroBoundaryRemainder,
    NonZeroTransitionRemainder,
}

impl Error for StarkVerifyError {}

impl fmt::Display for StarkVerifyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
