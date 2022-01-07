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

impl Stark {
    pub fn prove(
        &self,
        // Trace is indexed as trace[cycle][register]
        trace: Vec<Vec<BFieldElement>>,
        transition_constraints: Vec<MPolynomial<BFieldElement>>,
        boundary_constraints: Vec<BoundaryConstraint>,
        proof_stream: &mut ProofStream,
    ) -> Result<(), Box<dyn Error>> {
        // Extend execution trace with `randomizer_count` rows of randomizers.
        let mut rng = rand::thread_rng();
        let mut randomized_trace: Vec<Vec<BFieldElement>> = trace;
        for _ in 0..self.randomizer_count {
            randomized_trace.push(BFieldElement::random_elements(
                self.register_count as usize,
                &mut rng,
            ));
        }

        // Interpolate the trace to get a polynomial going through all trace values.
        let randomized_trace_domain: Vec<BFieldElement> = self.omicron.get_generator_domain();
        // self.get_generator_values(&self.omicron, randomized_trace.len());

        let mut trace_polynomials = vec![];
        for r in 0..self.register_count as usize {
            let values = &randomized_trace
                .iter()
                .map(|t| t[r].clone())
                .collect::<Vec<BFieldElement>>();

            trace_polynomials.push(Polynomial::fast_interpolate(
                &randomized_trace_domain,
                &values,
                &self.omicron,
                self.omicron_domain_length as usize,
            ));
        }

        // Subtract boundary interpolants and divide out boundary zerofiers
        let bcs_formatted = self.format_boundary_constraints(boundary_constraints);
        let boundary_interpolants: Vec<Polynomial<BFieldElement>> =
            get_boundary_interpolants(bcs_formatted.clone());
        let boundary_zerofiers: Vec<Polynomial<BFieldElement>> =
            get_boundary_zerofiers(bcs_formatted.clone());
        let mut boundary_quotients: Vec<Polynomial<BFieldElement>> =
            vec![Polynomial::ring_zero(); self.register_count as usize];

        for r in 0..self.register_count as usize {
            let div_res = (trace_polynomials[r].clone() - boundary_interpolants[r].clone())
                .divide(boundary_zerofiers[r].clone());
            assert!(
                div_res.1.is_zero(),
                "Remainder must be zero when dividing out boundary zerofier"
            );
            boundary_quotients[r] = div_res.0;
        }

        // Commit to boundary quotients
        let mut boundary_quotient_merkle_trees: Vec<MerkleTree<BFieldElement>> = vec![];
        for bq in boundary_quotients.iter() {
            let boundary_quotient_codeword: Vec<BFieldElement> = bq
                .fast_coset_evaluate(&self.field_generator, &self.omega, self.fri.domain_length)
                .iter()
                .map(|x| x.clone())
                .collect();
            let bq_merkle_tree = MerkleTree::from_vec(&boundary_quotient_codeword);
            proof_stream.enqueue(&bq_merkle_tree.get_root())?;
            boundary_quotient_merkle_trees.push(bq_merkle_tree);
        }

        // Symbolically evaluate transition constraints
        let x = Polynomial {
            coefficients: vec![self.omega.ring_zero(), self.omega.ring_one()],
        };
        let mut point: Vec<Polynomial<BFieldElement>> = vec![x.clone()];

        // add polynomial representing trace[x_i] and trace[x_{i+1}]
        point.append(&mut trace_polynomials.clone());
        point.append(
            &mut trace_polynomials
                .clone() // TODO: REMOVE
                .into_iter()
                .map(|tp| tp.scale(&self.omicron))
                .collect(),
        );
        let transition_polynomials: Vec<Polynomial<BFieldElement>> = transition_constraints
            .iter()
            .map(|x| x.evaluate_symbolic(&point))
            .collect();

        // divide out transition zerofier
        let transition_quotients: Vec<Polynomial<BFieldElement>> = transition_polynomials
            .iter()
            .map(|tp| {
                Polynomial::fast_coset_divide(
                    tp,
                    &transition_zerofier,
                    &self.field_generator,
                    &self.omicron,
                    self.omicron_domain.len(),
                )
            })
            .collect();

        // Commit to randomizer polynomial
        let max_degree = self.max_degree(&transition_constraints);
        let mut randomizer_polynomial_coefficients: Vec<BFieldElement> = vec![];
        for _ in 0..max_degree + 1 {
            let mut rand_bytes = [0u8; 32];
            rng.fill_bytes(&mut rand_bytes);
            randomizer_polynomial_coefficients.push(self.field.from_bytes(&rand_bytes));
        }

        let randomizer_polynomial = Polynomial {
            coefficients: randomizer_polynomial_coefficients,
        };

        let randomizer_codeword: Vec<BigInt> = randomizer_polynomial
            .fast_coset_evaluate(&self.field_generator, &self.omega, self.fri.domain_length)
            .iter()
            .map(|x| x.value.clone())
            .collect();
        let randomizer_mt = MerkleTree::from_vec(&randomizer_codeword);
        proof_stream.enqueue(&randomizer_mt.get_root())?;

        // Sanity check, should probably be removed
        let expected_tq_degrees = self.transition_quotient_degree_bounds(&transition_constraints);
        for r in 0..self.register_count {
            assert_eq!(
                expected_tq_degrees[r] as isize,
                transition_quotients[r].degree(),
                "Transition quotient degree must match expected value"
            );
        }

        // Compute terms of nonlinear combination polynomial
        let boundary_degrees = self.boundary_quotient_degree_bounds(&boundary_zerofiers);
        let mut terms: Vec<Polynomial<BFieldElement>> = vec![randomizer_polynomial];
        for (tq, tq_degree) in transition_quotients.iter().zip(expected_tq_degrees.iter()) {
            terms.push(tq.to_owned());
            let shift = max_degree - tq_degree;
            let shifted = tq.shift_coefficients(shift, self.omega.ring_zero());
            assert_eq!(max_degree as isize, shifted.degree()); // TODO: Can be removed
            terms.push(shifted);
        }
        for (bq, bq_degree) in boundary_quotients.iter().zip(boundary_degrees.iter()) {
            terms.push(bq.to_owned());
            let shift = max_degree - bq_degree;
            let shifted = bq.shift_coefficients(shift, self.omega.ring_zero());
            assert_eq!(max_degree as isize, shifted.degree()); // TODO: Can be removed
            terms.push(shifted);
        }

        // Take weighted sum
        // # get weights for nonlinear combination
        // #  - 1 randomizer
        // #  - 2 for every transition quotient
        // #  - 2 for every boundary quotient
        let fiat_shamir_hash: Vec<u8> = proof_stream.prover_fiat_shamir();
        let weights = self.sample_weights(
            &fiat_shamir_hash,
            1 + 2 * transition_quotients.len() + 2 * boundary_quotients.len(),
        );
        assert_eq!(
            weights.len(),
            terms.len(),
            "weights and terms length must match"
        );
        let combination = weights
            .iter()
            .zip(terms.iter())
            .fold(Polynomial::ring_zero(), |sum, (weight, pol)| {
                sum + pol.scalar_mul(weight.to_owned())
            });

        let combined_codeword = combination.fast_coset_evaluate(
            &self.field_generator,
            &self.omega,
            self.fri.domain_length,
        );

        // Prove low degree of combination polynomial, and collect indices
        let indices: Vec<usize> = self.fri.prove(
            &combined_codeword
                .iter()
                .map(|x| x.value.clone())
                .collect::<Vec<BigInt>>(),
            proof_stream,
        )?;

        // Process indices
        let mut duplicated_indices = indices.clone();
        duplicated_indices.append(
            &mut indices
                .into_iter()
                .map(|i| (i + self.expansion_factor) % self.fri.domain_length)
                .collect(),
        );
        let mut quadrupled_indices = duplicated_indices.clone();
        quadrupled_indices.append(
            &mut duplicated_indices
                .into_iter()
                .map(|i| (i + self.fri.domain_length / 2) % self.fri.domain_length)
                .collect(),
        );
        quadrupled_indices.sort_unstable();

        // Open indicated positions in the boundary quotient codewords
        for bq_mt in boundary_quotient_merkle_trees {
            proof_stream.enqueue_length_prepended(&bq_mt.get_multi_proof(&quadrupled_indices))?;
        }

        // Open indicated positions in the randomizer
        proof_stream
            .enqueue_length_prepended(&randomizer_mt.get_multi_proof(&quadrupled_indices))?;

        // Open indicated positions in the zerofier
        proof_stream.enqueue_length_prepended(
            &transition_zerofier_mt.get_multi_proof(&quadrupled_indices),
        )?;

        Ok(())
    }

    // Convert boundary constraints into a vector of boundary
    // constraints indexed by register.
    fn format_boundary_constraints(
        &self,
        boundary_constraints: Vec<BoundaryConstraint>,
    ) -> Vec<Vec<(BFieldElement, BFieldElement)>> {
        let mut bcs: Vec<Vec<(BFieldElement, BFieldElement)>> =
            vec![vec![]; self.register_count as usize];

        for bc in boundary_constraints {
            // XXX: Should bc.cycle have type usize or u32?
            bcs[bc.register].push((self.omicron.mod_pow_u32(bc.cycle as u32), bc.value));
        }

        bcs
    }
}

// Return the interpolants for the provided points. This is the `L(x)` in the equation
// to derive the boundary quotient: `Q_B(x) = (ECT(x) - L(x)) / Z_B(x)`.
// input is indexed with bcs[register][cycle]
fn get_boundary_interpolants(
    bcs: Vec<Vec<(BFieldElement, BFieldElement)>>,
) -> Vec<Polynomial<BFieldElement>> {
    bcs.iter()
        .map(|points| Polynomial::slow_lagrange_interpolation(points))
        .collect()
}

fn get_boundary_zerofiers(
    bcs: Vec<Vec<(BFieldElement, BFieldElement)>>,
) -> Vec<Polynomial<BFieldElement>> {
    let roots: Vec<Vec<BFieldElement>> = bcs
        .iter()
        .map(|points| points.iter().map(|(x, _y)| x.to_owned()).collect())
        .collect();
    roots
        .iter()
        .map(|points| Polynomial::get_polynomial_with_roots(points))
        .collect()
}
