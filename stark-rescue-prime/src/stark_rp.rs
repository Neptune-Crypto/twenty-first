use num_traits::One;
use num_traits::Zero;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::iter::zip;
use twenty_first::shared_math::rescue_prime_digest::DIGEST_LENGTH;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::fri::Fri;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::ntt::intt;
use twenty_first::shared_math::other::log_2_ceil;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::other::random_elements_array;
use twenty_first::shared_math::other::roundup_npo2;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::shared_math::rescue_prime_regular::*;
use twenty_first::shared_math::traits::CyclicGroupGenerator;
use twenty_first::shared_math::traits::PrimitiveRootOfUnity;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::timing_reporter::TimingReporter;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::CpuParallel;
use twenty_first::util_types::merkle_tree::{
    MerkleTree, PartialAuthenticationPath, SaltedMerkleTree,
};
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;
use twenty_first::util_types::proof_stream::ProofStream;

use crate::stark_constraints::BoundaryConstraint;

type Degree = i64;

pub const DOCUMENT_HASH_LENGTH: usize = 32usize;
pub const MERKLE_ROOT_HASH_LENGTH: usize = 32usize;
pub const B_FIELD_ELEMENT_SALTS_PER_VALUE: usize = 3usize;

// TODO: Consider <B: PrimeFieldElement, X: PrimeFieldElement>
// This requires a trait a la Lift<X> to generalise XFE::new_const().
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarkRp {
    expansion_factor: u32,
    field_generator: BFieldElement,
    colinearity_check_count: u32,
    min_num_randomizers: u32,
    num_registers: u32,
}

// TODO: Consider impl<B: PrimeFieldElement, X: PrimeFieldElement>
impl StarkRp {
    pub fn new(
        expansion_factor: u32,
        colinearity_check_count: u32,
        num_registers: u32,
        field_generator: BFieldElement,
    ) -> Self {
        let min_num_randomizers = 4 * colinearity_check_count;

        Self {
            expansion_factor,
            field_generator,
            colinearity_check_count,
            min_num_randomizers,
            num_registers,
        }
    }
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
    UnexpectedFriValue,
}

impl Error for StarkVerifyError {}

impl fmt::Display for StarkVerifyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

type StarkHasher = blake3::Hasher;
type SaltedMt = SaltedMerkleTree<StarkHasher>;
type Maker = CpuParallel;
type XFieldMt = MerkleTree<StarkHasher, Maker>;
type XFieldFri = Fri<StarkHasher>;

impl StarkRp {
    pub fn lift_b_x(b_poly: &Polynomial<BFieldElement>) -> Polynomial<XFieldElement> {
        let x_field_coefficients = b_poly.coefficients.iter().map(|b| b.lift()).collect();
        Polynomial::new(x_field_coefficients)
    }

    pub fn prove(
        &self,
        // Trace is indexed as trace[cycle][register]
        trace: &[[BFieldElement; STATE_SIZE]],
        transition_constraints: &[MPolynomial<BFieldElement>],
        boundary_constraints: &[BoundaryConstraint],
        proof_stream: &mut ProofStream,
        input_omicron: BFieldElement,
    ) -> Result<(u32, BFieldElement), Box<dyn Error>> {
        let mut timer = TimingReporter::start();

        // infer details about computation
        let original_trace_length = trace.len() as i64;
        let rounded_trace_length =
            roundup_npo2((original_trace_length + self.min_num_randomizers as i64) as u64);
        let num_randomizers = rounded_trace_length - original_trace_length as u64;
        let tp_bounds =
            self.transition_degree_bounds(transition_constraints, rounded_trace_length as usize);
        let tp_degree: &Degree = tp_bounds.iter().max().unwrap();
        let tq_degree_max: Degree = tp_degree - (original_trace_length - 1);
        let max_degree: Degree = roundup_npo2((tq_degree_max + 1) as u64) as i64 - 1; // The max degree bound provable by FRI
        let fri_domain_length = (max_degree + 1) * self.expansion_factor as i64;
        let blowup_factor_new = fri_domain_length / (rounded_trace_length as i64);

        timer.elapsed("calculate initial details");

        // compute generators
        let omega = BFieldElement::primitive_root_of_unity(fri_domain_length as u64).unwrap();

        timer.elapsed("calculate omega");

        let omicron_domain_length = rounded_trace_length;
        let omicron = BFieldElement::primitive_root_of_unity(omicron_domain_length).unwrap();

        timer.elapsed("calculate omicron");

        assert_eq!(
            input_omicron, omicron,
            "Calculated omicron must match input omicron"
        );

        // add randomizers to trace for zero-knowledge
        let mut randomized_trace = trace.to_owned();
        self.randomize_trace(&mut randomized_trace, num_randomizers);

        timer.elapsed("calculate and add randomizers");

        // Use `intt' (interpolation) to convert trace codewords into trace polynomials
        let mut trace_interpolants = vec![];
        for r in 0..self.num_registers as usize {
            // `trace_interpolant' starts as a codeword, meaning a column in the trace...
            let mut trace_interpolant: Vec<BFieldElement> = randomized_trace
                .iter()
                .map(|t| t[r])
                .collect::<Vec<BFieldElement>>();

            // ...and is subsequently transformed into polynomial coefficients via intt:
            intt::<BFieldElement>(
                &mut trace_interpolant,
                omicron,
                log_2_ceil(omicron_domain_length as u128) as u32,
            );

            trace_interpolants.push(Polynomial {
                coefficients: trace_interpolant,
            });

            // Sanity checks; consider moving into unit tests.
            // for (i, item) in trace.iter().enumerate() {
            //     assert_eq!(
            //         item[r],
            //         trace_interpolant.evaluate(&omicron.mod_pow(i as u64)),
            //         "Trace interpolant evaluates back to trace element"
            //     );
            // }
            // assert_eq!(
            //     (randomized_trace.len() - 1) as isize,
            //     trace_interpolant.degree(),
            //     "Trace interpolant has one degree lower than the points"
            // );
        }

        timer.elapsed("calculate intt for each column in trace");

        // Sanity check
        // for ti in trace_interpolants.iter() {
        //     assert!(ti.degree() == rounded_trace_length as isize - 1);
        // }

        //// bq(x) = (ti(x) - bi(x)) / bz(x)
        //
        // where
        //   bq: boundary quotients
        //   ti: boundary interpolants
        //   bz: boundary zerofiers
        //
        // In the case where there are no boundary conditions this formula reduces to:
        // bq(x) = (ti(x) - 0) / 1 = ti(x)

        // Generate boundary quotients
        // Subtract boundary interpolants and divide out boundary zerofiers
        let bcs_formatted = self.format_boundary_constraints(omicron, boundary_constraints);
        let boundary_interpolants: Vec<Polynomial<BFieldElement>> =
            self.get_boundary_interpolants(bcs_formatted.clone());
        let boundary_zerofiers: Vec<Polynomial<BFieldElement>> =
            self.get_boundary_zerofiers(bcs_formatted);
        let mut boundary_quotients: Vec<Polynomial<BFieldElement>> =
            vec![Polynomial::zero(); self.num_registers as usize];

        timer.elapsed("calculate intt for each column in trace");

        // TODO: Use coset_divide here
        for r in 0..self.num_registers as usize {
            let div_res = (trace_interpolants[r].clone() - boundary_interpolants[r].clone())
                .divide(boundary_zerofiers[r].clone());
            assert_eq!(
                Polynomial::<BFieldElement>::zero(),
                div_res.1,
                "Remainder must be zero when dividing out boundary zerofier"
            );
            boundary_quotients[r] = div_res.0;
        }

        timer.elapsed("calculate boundary quotients");

        // Commit to boundary quotients
        let mut boundary_quotient_merkle_trees: Vec<SaltedMt> = vec![];
        let mut boundary_quotient_codewords: Vec<Vec<BFieldElement>> = vec![];

        for bq in boundary_quotients.iter() {
            let boundary_quotient_codeword: Vec<BFieldElement> =
                bq.fast_coset_evaluate(&self.field_generator, omega, fri_domain_length as usize);

            boundary_quotient_codewords.push(boundary_quotient_codeword.clone());
            // Build MT
            let boundary_quotient_codeword_digests: Vec<_> = boundary_quotient_codeword
                .iter()
                .map(StarkHasher::hash)
                .collect();

            // FIXME: Replace with randomly generating digests directly
            let salts_per_leaf = 5;
            let boundary_quotient_codeword_salts: Vec<Digest> =
                random_elements(boundary_quotient_codeword.len() * salts_per_leaf);

            let bq_merkle_tree = SaltedMt::from_digests(
                &boundary_quotient_codeword_digests,
                &boundary_quotient_codeword_salts,
            );
            proof_stream.enqueue(&bq_merkle_tree.get_root())?;
            boundary_quotient_merkle_trees.push(bq_merkle_tree);
        }

        timer.elapsed("calculate boundary and commit quotient codewords to proof stream");

        //// tq(x) = tp(x) / tz(x)
        //
        // where
        //   tq(x) = transition quotients
        //   tp(x) = transition polynomials
        //   tz(x) = transition zerofiers

        // Symbolically evaluate transition constraints
        let x = Polynomial {
            coefficients: vec![BFieldElement::zero(), BFieldElement::one()],
        };
        let mut point: Vec<Polynomial<BFieldElement>> = vec![x];
        // add polynomial representing trace[x_i] and trace[x_{i+1}]
        point.append(&mut trace_interpolants.clone());
        point.append(
            &mut trace_interpolants
                .clone() // FIXME: Remove again
                .into_iter()
                .map(|ti| ti.scale(&omicron))
                .collect(),
        );

        timer.elapsed("scale trace interpolants");

        let mut exponents_memoization: HashMap<Vec<u8>, Polynomial<BFieldElement>> = HashMap::new();

        MPolynomial::precalculate_symbolic_exponents(
            // A slight speedup can be achieved here by only sending the 1st
            // transition_constraints element to the precalculation function. I didn't
            // do it though, as it feels like cheating which is an optimization I don't
            // understand, for this, use: &transition_constraints[0..1] as 1st argument
            transition_constraints,
            &point,
            &mut exponents_memoization,
        )?;
        timer.elapsed("Precalculate intermediate results");

        // Precalculate `point` exponentiations for faster symbolic evaluation
        // TODO: I'm a bit unsure about the upper limit of the outer loop.
        // Getting this number right will just mean slightly faster code. It shouldn't
        // lead to errors if the number is too high or too low.
        // let mut point_exponents = point.clone();
        // for i in 2..tp_degree / rounded_trace_length + 2 {
        //     for j in 0..point.len() {
        //         point_exponents[j] = point_exponents[j].clone() * point[j].clone();
        //         mod_pow_memoization.insert((j, i), point_exponents[j].clone());
        //     }
        // }
        // timer.elapsed("Precalculate mod_pow values");

        let mut transition_polynomials: Vec<Polynomial<BFieldElement>> = vec![];
        for constraint in transition_constraints {
            transition_polynomials.push(
                constraint.evaluate_symbolic_with_memoization_precalculated(
                    &point,
                    &mut exponents_memoization,
                ),
            );
        }

        timer.elapsed("symbolically evaluate transition constraints");

        // TODO: Sanity check REMOVE
        // for tp in transition_polynomials.iter() {
        //     assert_eq!(*tp_degree, tp.degree() as u64);
        //     for i in 0..original_trace_length - 1 {
        //         let x = omicron.mod_pow(i);
        //         assert!(tp.evaluate(&x).is_zero());
        //     }
        // }

        // TODO: Calculate the transition_zerofier faster than this using group theory.
        let transition_zerofier: Polynomial<BFieldElement> = self.get_transition_zerofier(
            omicron,
            omicron_domain_length as usize,
            original_trace_length as usize,
        );

        timer.elapsed("get transition zerofiers");

        // divide out transition zerofier
        let transition_quotients: Vec<Polynomial<BFieldElement>> = transition_polynomials
            .iter()
            .map(|tp| {
                // Sanity check, remove
                // let (_quot, rem) = tp.divide(transition_zerofier.clone());
                // assert_eq!(
                //     Polynomial::zero(),
                //     rem,
                //     "Remainder must be zero when calculating transition quotient"
                // );

                Polynomial::fast_coset_divide(
                    tp,
                    &transition_zerofier,
                    self.field_generator,
                    omega,
                    fri_domain_length as usize,
                )
            })
            .collect();

        timer.elapsed("fast_coset_divide each transition polynomial");

        // Commit to randomizer polynomial
        let randomizer_polynomial: Polynomial<XFieldElement> = Polynomial {
            coefficients: random_elements(max_degree as usize + 1),
        };

        let randomizer_codeword: Vec<XFieldElement> = randomizer_polynomial.fast_coset_evaluate(
            &self.field_generator,
            omega,
            fri_domain_length as usize,
        );

        let randomizer_codeword_digests: Vec<Digest> =
            randomizer_codeword.iter().map(StarkHasher::hash).collect();
        let randomizer_mt: XFieldMt = Maker::from_digests(&randomizer_codeword_digests);

        let randomizer_mt_root = randomizer_mt.get_root();
        proof_stream.enqueue(&randomizer_mt_root)?;

        timer.elapsed("fast_coset_evaluate and commit randomizer codeword to proof stream");

        let expected_tq_degrees = self.transition_quotient_degree_bounds(
            transition_constraints,
            original_trace_length as usize,
            rounded_trace_length as usize,
        );

        timer.elapsed("calculate transition_quotient_degree_bounds");

        // Sanity check; Consider moving to unit test,
        for r in 0..self.num_registers as usize {
            assert_eq!(
                expected_tq_degrees[r] as isize,
                transition_quotients[r].degree(),
                "Transition quotient degree must match expected value"
            );
        }

        // Compute terms of nonlinear combination polynomial
        let boundary_degrees = self
            .boundary_quotient_degree_bounds(&boundary_zerofiers, rounded_trace_length as usize);

        timer.elapsed("calculate boundary_quotient_degree_bounds");

        let mut terms: Vec<Polynomial<XFieldElement>> = vec![randomizer_polynomial];
        for (tq, tq_degree) in transition_quotients.iter().zip(expected_tq_degrees.iter()) {
            let tq_x: Polynomial<XFieldElement> = Self::lift_b_x(tq);
            terms.push(tq_x.clone());
            let shift = max_degree - *tq_degree;

            // Make new polynomial with max_degree degree by shifting all terms up
            let shifted = tq_x.shift_coefficients(shift as usize);
            assert_eq!(max_degree as isize, shifted.degree());
            terms.push(shifted);
        }
        for (bq, bq_degree) in boundary_quotients.iter().zip(boundary_degrees.iter()) {
            let bq_x: Polynomial<XFieldElement> = Self::lift_b_x(bq);
            terms.push(bq_x.clone());
            let shift = max_degree as usize - bq_degree;

            // Make new polynomial with max_degree degree by shifting all terms up
            let shifted = bq_x.shift_coefficients(shift);
            assert_eq!(max_degree as isize, shifted.degree());
            terms.push(shifted);
        }

        timer.elapsed("calculate terms");

        // Take weighted sum
        // # get weights for nonlinear combination
        // #  - 1 randomizer
        // #  - 2 for every transition quotient
        // #  - 2 for every boundary quotient
        let challenge: Digest = proof_stream.prover_fiat_shamir();
        let weights: Vec<XFieldElement> = Self::sample_weights(
            &challenge,
            1 + 2 * transition_quotients.len() + 2 * boundary_quotients.len(),
        );
        assert_eq!(
            weights.len(),
            terms.len(),
            "weights and terms length must match"
        );

        timer.elapsed("calculate prover_fiat_shamir");

        let combination: Polynomial<XFieldElement> = weights
            .iter()
            .zip(terms.iter())
            .fold(Polynomial::zero(), |sum, (&weight, pol)| {
                sum + pol.scalar_mul(weight)
            });

        timer.elapsed("calculate sum of combination polynomial");

        let combined_codeword: Vec<XFieldElement> = combination.fast_coset_evaluate(
            &self.field_generator,
            omega,
            fri_domain_length as usize,
        );

        timer.elapsed("calculate fast_coset_evaluate of combination polynomial");

        // Prove low degree of combination polynomial, and collect indices
        let fri = XFieldFri::new(
            self.field_generator,
            omega,
            fri_domain_length as usize,
            self.expansion_factor as usize,
            self.colinearity_check_count as usize,
        );

        let indices: Vec<usize> = fri.prove(&combined_codeword, proof_stream)?;

        timer.elapsed("calculate fri.prove()");

        // Process indices
        let mut duplicated_indices = indices.clone();
        let fri_domain_length_usize = fri_domain_length as usize;
        duplicated_indices.append(
            &mut indices
                .into_iter()
                .map(|i| (i + blowup_factor_new as usize) % fri_domain_length_usize)
                .collect(),
        );

        let mut quadrupled_indices = duplicated_indices.clone();
        quadrupled_indices.append(
            &mut duplicated_indices
                .into_iter()
                .map(|i| (i + fri_domain_length_usize / 2) % fri_domain_length_usize)
                .collect(),
        );
        quadrupled_indices.sort_unstable();

        timer.elapsed("sort quadrupled indices");

        // Open indicated positions in the boundary quotient codewords
        for (j, bq_mt) in boundary_quotient_merkle_trees.into_iter().enumerate() {
            let authetication_paths_and_salts: Vec<(
                PartialAuthenticationPath<Digest>,
                Digest, // salts
            )> = bq_mt.get_authentication_structure_and_salt(&quadrupled_indices);
            proof_stream.enqueue_length_prepended(&authetication_paths_and_salts)?;
            let values: Vec<BFieldElement> = quadrupled_indices
                .iter()
                .map(|i| boundary_quotient_codewords[j][*i])
                .collect();

            proof_stream.enqueue_length_prepended(&values)?;
        }

        timer.elapsed("calculate bq_mt.get_multi_proof(quadrupled_indices) for all boundary quotient merkle trees");

        // Open indicated positions in the randomizer
        let randomizer_auth_paths: Vec<PartialAuthenticationPath<Digest>> =
            randomizer_mt.get_authentication_structure(&quadrupled_indices);
        proof_stream.enqueue_length_prepended(&randomizer_auth_paths)?;

        let randomizer_values: Vec<XFieldElement> = quadrupled_indices
            .iter()
            .map(|&i| randomizer_codeword[i])
            .collect();
        proof_stream.enqueue_length_prepended(&randomizer_values)?;

        timer.elapsed("calculate bq_mt.get_multi_proof(quadrupled_indices) for randomizer");
        let report = timer.finish();
        println!("{}", report);

        Ok((fri_domain_length as u32, omega))
    }

    pub fn verify(
        &self,
        proof_stream: &mut ProofStream,
        transition_constraints: &[MPolynomial<BFieldElement>],
        boundary_constraints: &[BoundaryConstraint],
        fri_domain_length: u32,
        omega: BFieldElement,
        original_trace_length: u32,
    ) -> Result<(), Box<dyn Error>> {
        let mut timer = TimingReporter::start();
        // assert!(omega.mod_pow(fri_domain_length as u64).is_one());
        // assert!(!omega.mod_pow((fri_domain_length / 2) as u64).is_one());

        // Get Merkle root of boundary quotient codewords
        let mut boundary_quotient_mt_roots: Vec<Digest> = vec![];
        for _ in 0..self.num_registers {
            let bq_mt_root = proof_stream.dequeue(32)?;
            boundary_quotient_mt_roots.push(bq_mt_root);
        }
        timer.elapsed("get BQ merkle roots from proof stream");

        let randomizer_mt_root: Digest = proof_stream.dequeue(32)?;
        timer.elapsed("get randomizer_mt_root from proof stream");

        // Get weights for nonlinear combination
        // 1 weight element for randomizer
        // 2 for every transition quotient
        // 2 for every boundary quotient
        let fiat_shamir_hash: Digest = proof_stream.verifier_fiat_shamir();
        let weights: Vec<XFieldElement> = Self::sample_weights(
            &fiat_shamir_hash,
            1 + 2 * boundary_quotient_mt_roots.len() + 2 * transition_constraints.len(),
        );
        timer.elapsed("Calculate weights challenge");

        // Verify low degree of combination polynomial, and collect indices
        // Note that FRI verifier verifies number of samples, so we don't have
        // to check that number here
        let fri = XFieldFri::new(
            self.field_generator,
            omega,
            fri_domain_length as usize,
            self.expansion_factor as usize,
            self.colinearity_check_count as usize,
        );

        let combination_values: Vec<(usize, XFieldElement)> = fri.verify(proof_stream)?;
        timer.elapsed("Run FRI verifier");

        let (indices, values): (Vec<usize>, Vec<XFieldElement>) =
            combination_values.into_iter().unzip();

        // Because fri_domain_length = (max_degree + 1) * expansion_factor...
        let max_degree = (fri_domain_length / self.expansion_factor) - 1;
        let rounded_trace_length =
            roundup_npo2((original_trace_length + self.min_num_randomizers) as u64);
        let blowup_factor_new = fri_domain_length / rounded_trace_length as u32;
        let omicron_domain_length = rounded_trace_length;

        let mut duplicated_indices = indices.clone();
        duplicated_indices.append(
            &mut indices
                .iter()
                .map(|i| (*i + blowup_factor_new as usize) % fri_domain_length as usize)
                .collect(),
        );
        duplicated_indices.sort_unstable();
        timer.elapsed("Calculate indices");

        let duplicated_indices_local = duplicated_indices.clone();
        // Read and verify boundary quotient leafs
        // revealed boundary quotient codeword values, indexed by (register, codeword index)
        let mut boundary_quotients: Vec<HashMap<usize, BFieldElement>> = vec![];
        for (i, bq_root) in boundary_quotient_mt_roots.into_iter().enumerate() {
            boundary_quotients.push(HashMap::new());
            let authentication_paths_and_salt: Vec<(
                PartialAuthenticationPath<Digest>,
                Digest, // sals
            )> = proof_stream.dequeue_length_prepended()?;

            let bq_values: Vec<BFieldElement> = proof_stream.dequeue_length_prepended()?;

            let unsalted_leaves: Vec<Digest> = bq_values.iter().map(StarkHasher::hash).collect();

            let valid = SaltedMt::verify_authentication_structure(
                bq_root,
                &duplicated_indices_local,
                &unsalted_leaves,
                &authentication_paths_and_salt,
            );
            if !valid {
                return Err(Box::new(StarkVerifyError::BadMerkleProof(
                    MerkleProofError::BoundaryQuotientError(i),
                )));
            }

            assert_eq!(
                duplicated_indices_local.len(),
                bq_values.len(),
                "Index set and values from prover should correspond."
            );
            for (index, value) in zip(duplicated_indices_local.clone(), bq_values) {
                boundary_quotients[i].insert(index, value);
            }
        }
        timer.elapsed("Verify boundary quotient Merkle paths");

        // Read and verify randomizer leafs
        let randomizer_auth_paths: Vec<PartialAuthenticationPath<Digest>> =
            proof_stream.dequeue_length_prepended()?;
        let revealed_randomizer_values: Vec<XFieldElement> =
            proof_stream.dequeue_length_prepended()?;
        let randomizer_values_digests: Vec<Digest> = revealed_randomizer_values
            .iter()
            .map(StarkHasher::hash)
            .collect();

        let auth_pairs: Vec<_> = zip(randomizer_auth_paths, randomizer_values_digests).collect();

        let valid = XFieldMt::verify_authentication_structure(
            randomizer_mt_root,
            &duplicated_indices,
            &auth_pairs,
        );
        if !valid {
            return Err(Box::new(StarkVerifyError::BadMerkleProof(
                MerkleProofError::RandomizerError,
            )));
        }
        timer.elapsed("Verify randomizer Merkle paths");

        // Insert randomizer values in HashMap
        let mut randomizer_values: HashMap<usize, XFieldElement> = HashMap::new();
        for (index, value) in zip(duplicated_indices, revealed_randomizer_values) {
            randomizer_values.insert(index, value);
        }

        let omicron = BFieldElement::primitive_root_of_unity(omicron_domain_length).unwrap();
        timer.elapsed("Insert randomizer values in HashMap");

        // Verify leafs of combination polynomial
        let formatted_bcs: Vec<Vec<(BFieldElement, BFieldElement)>> =
            self.format_boundary_constraints(omicron, boundary_constraints);
        let boundary_zerofiers: Vec<Polynomial<BFieldElement>> =
            self.get_boundary_zerofiers(formatted_bcs.clone());
        let boundary_interpolants: Vec<Polynomial<BFieldElement>> =
            self.get_boundary_interpolants(formatted_bcs);
        let boundary_degrees: Vec<usize> = self
            .boundary_quotient_degree_bounds(&boundary_zerofiers, rounded_trace_length as usize);
        timer.elapsed("Calculate boundary zerofiers and interpolants");

        let expected_tq_degrees: Vec<i64> = self.transition_quotient_degree_bounds(
            transition_constraints,
            original_trace_length as usize,
            rounded_trace_length as usize,
        );
        let max_exponent: u8 = transition_constraints
            .iter()
            .map(|mpol| mpol.max_exponent())
            .max()
            .unwrap();
        timer.elapsed("Calculate expected TQ degrees");

        // TODO: Calculate the transition_zerofier faster than this using group theory.
        let transition_zerofier: Polynomial<BFieldElement> = self.get_transition_zerofier(
            omicron,
            omicron_domain_length as usize,
            original_trace_length as usize,
        );
        timer.elapsed("Calculate transition zerofier");

        let exponents_list: Vec<Vec<u8>> =
            MPolynomial::extract_exponents_list(transition_constraints)?;
        timer.elapsed("Calculate exponents list");
        for (i, current_index) in indices.into_iter().enumerate() {
            let current_x: BFieldElement =
                self.field_generator * omega.mod_pow(current_index as u64);
            timer.elapsed(&format!("current_x {}", i));
            let next_index: usize =
                (current_index + blowup_factor_new as usize) % fri_domain_length as usize;
            let next_x: BFieldElement = self.field_generator * omega.mod_pow(next_index as u64);
            timer.elapsed(&format!("next_x {}", i));
            let mut current_trace: Vec<BFieldElement> = (0..self.num_registers as usize)
                .map(|r| {
                    boundary_quotients[r][&current_index]
                        * boundary_zerofiers[r].evaluate(&current_x)
                        + boundary_interpolants[r].evaluate(&current_x)
                })
                .collect();
            timer.elapsed(&format!("current_trace {}", i));
            let mut next_trace: Vec<BFieldElement> = (0..self.num_registers as usize)
                .map(|r| {
                    boundary_quotients[r][&next_index] * boundary_zerofiers[r].evaluate(&next_x)
                        + boundary_interpolants[r].evaluate(&next_x)
                })
                .collect();
            timer.elapsed(&format!("next_trace {}", i));

            let mut point: Vec<BFieldElement> = vec![current_x];
            point.append(&mut current_trace);
            point.append(&mut next_trace);
            timer.elapsed(&format!("generate \"point\" {}", i));

            // println!("point length: {}", point.len());
            // println!(
            //     "transition_constraints length: {}",
            //     transition_constraints.len()
            // );
            // let tc_coefficient_counts: Vec<usize> = transition_constraints
            //     .iter()
            //     .map(|x| x.coefficients.len())
            //     .collect();
            // println!(
            //     "transition_constraints coefficient count: {:?}",
            //     tc_coefficient_counts
            // );
            // let tc_degrees: Vec<u64> = transition_constraints.iter().map(|x| x.degree()).collect();
            // println!("transition_constraints degrees: {:?}", tc_degrees);

            // TODO: For some reason this mod pow precalculation is super slow
            let precalculated_mod_pows: HashMap<(usize, u8), BFieldElement> =
                MPolynomial::<BFieldElement>::precalculate_scalar_mod_pows(max_exponent, &point);
            timer.elapsed(&format!("precalculate mod_pows {}", i));
            let intermediate_results: HashMap<Vec<u8>, BFieldElement> =
                MPolynomial::<BFieldElement>::precalculate_scalar_exponents(
                    &point,
                    &precalculated_mod_pows,
                    &exponents_list,
                )?;
            timer.elapsed(&format!(
                "precalculate transition_constraint_values intermediate results {}",
                i
            ));
            let transition_constraint_values: Vec<BFieldElement> = transition_constraints
                .iter()
                .map(|tc| tc.evaluate_with_precalculation(&point, &intermediate_results))
                .collect();
            timer.elapsed(&format!("transition_constraint_values {}", i));

            let current_transition_zerofier_value: BFieldElement =
                transition_zerofier.evaluate(&current_x);
            timer.elapsed(&format!("current_transition_zerofier_value {}", i));

            // Get combination polynomial evaluation value
            // Loop over all registers for transition quotient values, and for boundary quotient values
            let mut terms: Vec<XFieldElement> = vec![randomizer_values[&current_index]];
            for (tcv, tq_degree) in transition_constraint_values
                .iter()
                .zip(expected_tq_degrees.iter())
            {
                let transition_quotient = *tcv / current_transition_zerofier_value;
                terms.push(transition_quotient.lift());
                let shift = max_degree as i64 - *tq_degree;
                terms.push(
                    (transition_quotient * current_x.mod_pow(shift.try_into().unwrap())).lift(),
                );
            }
            for (bqvs, bq_degree) in boundary_quotients.iter().zip(boundary_degrees.iter()) {
                terms.push(bqvs[&current_index].lift());
                let shift = max_degree as u64 - *bq_degree as u64;
                terms.push((bqvs[&current_index] * current_x.mod_pow(shift)).lift());
            }

            assert_eq!(
                weights.len(),
                terms.len(),
                "weights and terms length must match in verifier"
            );
            let combination: XFieldElement = weights
                .iter()
                .zip(terms.iter())
                .fold(XFieldElement::zero(), |sum, (&weight, &term)| {
                    sum + term * weight
                });
            timer.elapsed(&format!("combination {}", i));

            if values[i] != combination {
                return Err(Box::new(StarkVerifyError::LinearCombinationMismatch(
                    current_index,
                )));
            }
        }
        timer.elapsed("Verify revealed values from combination polynomial");
        let report = timer.finish();
        println!("{}", report);
        Ok(())
    }

    fn randomize_trace(&self, trace: &mut Vec<[BFieldElement; STATE_SIZE]>, num_randomizers: u64) {
        let mut randomizers: Vec<[BFieldElement; STATE_SIZE]> = (0..num_randomizers)
            .map(|_| random_elements_array::<BFieldElement, STATE_SIZE>())
            .collect();

        trace.append(&mut randomizers);
    }

    // Convert boundary constraints into a vector of boundary constraints indexed by register.
    fn format_boundary_constraints(
        &self,
        omicron: BFieldElement,
        boundary_constraints: &[BoundaryConstraint],
    ) -> Vec<Vec<(BFieldElement, BFieldElement)>> {
        let mut bcs: Vec<Vec<(BFieldElement, BFieldElement)>> =
            vec![vec![]; self.num_registers as usize];

        for bc in boundary_constraints {
            // (x, y)
            bcs[bc.register].push((omicron.mod_pow(bc.cycle as u64), bc.value));
        }

        bcs
    }

    // Return the interpolants for the provided points. This is the `L(x)` in the equation
    // to derive the boundary quotient: `Q_B(x) = (ECT(x) - L(x)) / Z_B(x)`.
    // input is indexed with bcs[register][cycle]. Returns the zero-polynomial in case `points`
    // is the empty list.
    fn get_boundary_interpolants(
        &self,
        bcs: Vec<Vec<(BFieldElement, BFieldElement)>>,
    ) -> Vec<Polynomial<BFieldElement>> {
        bcs.iter()
            .map(|points| {
                if !points.is_empty() {
                    Polynomial::lagrange_interpolate_zipped(points)
                } else {
                    Polynomial::zero()
                }
            })
            .collect()
    }

    fn get_boundary_zerofiers(
        &self,
        bcs: Vec<Vec<(BFieldElement, BFieldElement)>>,
    ) -> Vec<Polynomial<BFieldElement>> {
        let roots: Vec<Vec<BFieldElement>> = bcs
            .iter()
            .map(|points| points.iter().map(|(x, _y)| x.to_owned()).collect())
            .collect();
        roots
            .iter()
            .map(|points| Polynomial::zerofier(points))
            .collect()
    }

    // Return the degrees of the boundary quotients
    fn boundary_quotient_degree_bounds(
        &self,
        boundary_zerofiers: &[Polynomial<BFieldElement>],
        rounded_trace_length: usize,
    ) -> Vec<usize> {
        let transition_degree = rounded_trace_length - 1;
        boundary_zerofiers
            .iter()
            .map(|x| transition_degree - x.degree() as usize)
            .collect()
    }

    /// Return the max degree for all transition quotients
    /// This is the degree of the execution trace interpolations
    /// divided by the transition zerofier polynomial
    fn transition_quotient_degree_bounds(
        &self,
        transition_constraints: &[MPolynomial<BFieldElement>],
        original_trace_length: usize,
        rounded_trace_length: usize,
    ) -> Vec<i64> {
        // The degree is the degree of the trace plus the randomizers
        // minus the original trace length minus 1.
        self.transition_degree_bounds(transition_constraints, rounded_trace_length)
            .iter()
            .map(|d| d - (original_trace_length as i64 - 1))
            .collect()
    }

    // Return the max degree for all interpolations of the execution trace
    fn transition_degree_bounds(
        &self,
        transition_constraints: &[MPolynomial<BFieldElement>],
        randomized_trace_length: usize,
    ) -> Vec<i64> {
        let mut point_degrees: Vec<i64> = vec![0; 1 + 2 * self.num_registers as usize];
        point_degrees[0] = 1;
        for r in 0..self.num_registers as usize {
            point_degrees[1 + r] = randomized_trace_length as i64 - 1;
            point_degrees[1 + r + self.num_registers as usize] = randomized_trace_length as i64 - 1;
        }

        // This could also be achieved with symbolic evaluation of the
        // input `transition_constraints` and then taking the degree of
        // the resulting polynomials.
        transition_constraints
            .iter()
            .map(|tc| tc.symbolic_degree_bound(&point_degrees))
            .collect()
    }

    fn get_transition_zerofier(
        &self,
        omicron: BFieldElement,
        omicron_domain_length: usize,
        original_trace_length: usize,
    ) -> Polynomial<BFieldElement> {
        // omicron_trace_elements is a large enough subset of the omicron domain
        let omicron_trace_elements =
            omicron.get_cyclic_group_elements(Some(original_trace_length - 1));
        Polynomial::fast_zerofier(&omicron_trace_elements, &omicron, omicron_domain_length)
    }

    fn sample_weights(randomness: &Digest, count: usize) -> Vec<XFieldElement> {
        StarkHasher::get_n_hash_rounds(randomness, count)
            .iter()
            .map(XFieldElement::sample)
            .collect()
    }

    /// Return a pair of a list of polynomials, first element in the pair,
    /// (first_round_constants[register], second_round_constants[register])
    pub fn get_round_constant_polynomials(
        omicron: BFieldElement,
    ) -> (
        Vec<MPolynomial<BFieldElement>>,
        Vec<MPolynomial<BFieldElement>>,
    ) {
        let domain = omicron.get_cyclic_group_elements(None);
        let variable_count = 2 * STATE_SIZE + 1;
        let mut first_round_constants: Vec<MPolynomial<BFieldElement>> = vec![];
        for i in 0..STATE_SIZE {
            let values: Vec<BFieldElement> = ROUND_CONSTANTS
                .into_iter()
                .skip(i)
                .step_by(2 * STATE_SIZE)
                .map(BFieldElement::from)
                .collect();
            // let coefficients = intt(&values, omicron);
            let points: Vec<(BFieldElement, BFieldElement)> = domain
                .clone()
                .iter()
                .zip(values.iter())
                .map(|(x, y)| (x.to_owned(), y.to_owned()))
                .collect();
            let coefficients =
                Polynomial::<BFieldElement>::lagrange_interpolate_zipped(&points).coefficients;
            first_round_constants.push(MPolynomial::lift(
                Polynomial { coefficients },
                0,
                variable_count,
            ));
        }

        let mut second_round_constants: Vec<MPolynomial<BFieldElement>> = vec![];
        for i in 0..STATE_SIZE {
            let values: Vec<BFieldElement> = ROUND_CONSTANTS
                .into_iter()
                .skip(i + STATE_SIZE)
                .step_by(2 * STATE_SIZE)
                .map(BFieldElement::from)
                .collect();
            // let coefficients = intt(&values, omicron);
            let points: Vec<(BFieldElement, BFieldElement)> = domain
                .clone()
                .iter()
                .zip(values.iter())
                .map(|(x, y)| (x.to_owned(), y.to_owned()))
                .collect();
            let coefficients =
                Polynomial::<BFieldElement>::lagrange_interpolate_zipped(&points).coefficients;
            second_round_constants.push(MPolynomial::lift(
                Polynomial { coefficients },
                0,
                variable_count,
            ));
        }

        (first_round_constants, second_round_constants)
    }

    // Returns the multivariate polynomial which takes the triplet (domain, trace, next_trace) and
    // returns composition polynomial, which is the evaluation of the air for a specific trace.
    //
    // AIR: [F_p x F_p^m x F_p^m] --> F_p^m
    //
    // The composition polynomial values are low-degree polynomial combinations
    // (as opposed to linear combinations) of the values:
    // `domain` (scalar), `trace` (vector), `next_trace` (vector).
    pub fn get_air_constraints(omicron: BFieldElement) -> Vec<MPolynomial<BFieldElement>> {
        let (first_step_constants, second_step_constants) =
            Self::get_round_constant_polynomials(omicron);

        let variable_count = 1 + 2 * STATE_SIZE;
        let variables = MPolynomial::variables(1 + 2 * STATE_SIZE);
        let previous_state = &variables[1..(STATE_SIZE + 1)];
        let next_state = &variables[(STATE_SIZE + 1)..(2 * STATE_SIZE + 1)];

        let previous_state_pow_alpha = previous_state
            .iter()
            .map(|poly| poly.pow(ALPHA as u8))
            .collect::<Vec<MPolynomial<BFieldElement>>>();

        // TODO: Consider refactoring MPolynomial<BFieldElement>
        // ::mod_pow(exp: BigInt, one: BFieldElement) into
        // ::mod_pow_u64(exp: u64)
        // let air: Vec<MPolynomial<BFieldElement>> = MDS
        //     .par_iter()
        //     .zip(MDS_INV.par_iter())
        //     .zip(first_step_constants.into_par_iter())
        //     .map(|(_, fsc)| {
        //         let mut lhs = MPolynomial::from_constant(BFieldElement::zero(), variable_count);
        //         for k in 0..STATE_SIZE {
        //             lhs += previous_state_pow_alpha[k].scalar_mul(BFieldElement::from(MDS[k]));
        //         }
        //         lhs += fsc;

        //         let mut rhs = MPolynomial::from_constant(BFieldElement::zero(), variable_count);
        //         for k in 0..STATE_SIZE {
        //             rhs += (next_state[k].clone() - second_step_constants[k].clone())
        //                 .scalar_mul(BFieldElement::from(MDS_INV[k]));
        //         }
        //         rhs = rhs.mod_pow(ALPHA.into(), one);

        //         lhs - rhs
        //     })
        //     .collect();
        // air

        let mut forward_airs = vec![MPolynomial::zero(variable_count); STATE_SIZE];
        (0..STATE_SIZE)
            .into_par_iter()
            .map(|i| {
                MDS[i * STATE_SIZE..(i + 1) * STATE_SIZE]
                    .iter()
                    .zip(previous_state_pow_alpha.iter())
                    .map(|(m, a)| MPolynomial::from_constant(*m, STATE_SIZE) * a.clone())
                    .fold(MPolynomial::zero(variable_count), |x, y| x + y)
                    + first_step_constants[i].to_owned()
            })
            .collect_into_vec(&mut forward_airs);

        let mut backward_airs = vec![MPolynomial::zero(variable_count); STATE_SIZE];
        (0..STATE_SIZE)
            .into_par_iter()
            .map(|i| {
                MDS_INV[i * STATE_SIZE..(i + 1) * STATE_SIZE]
                    .iter()
                    .zip(next_state.iter().enumerate())
                    .map(|(m, (j, a))| {
                        MPolynomial::from_constant(*m, variable_count)
                            * (a.clone() - second_step_constants[j].to_owned())
                    })
                    .fold(MPolynomial::zero(variable_count), |x, y| x + y)
                    .pow(ALPHA as u8)
            })
            .collect_into_vec(&mut backward_airs);

        forward_airs
            .into_par_iter()
            .zip_eq(backward_airs.into_par_iter())
            .map(|(lhs, rhs)| lhs - rhs)
            .collect()
    }

    pub fn get_boundary_constraints(output_elements: &[BFieldElement]) -> Vec<BoundaryConstraint> {
        let mut bcs = vec![];

        // We are hashing a fixed-length input, so there is no padding.
        // However, there is domain separation, which requires that the
        // rate-th element be initialized to 1.
        bcs.push(BoundaryConstraint {
            cycle: 0,
            register: 10,
            value: BFieldElement::one(),
        });
        for i in 11..STATE_SIZE {
            let bc = BoundaryConstraint {
                cycle: 0,
                register: i,
                value: BFieldElement::zero(),
            };
            bcs.push(bc);
        }

        // The output of the hash function puts a constraint on the trace
        // in the form of boundary conditions
        #[allow(clippy::needless_range_loop)]
        for i in 0..DIGEST_LENGTH {
            let end_constraint = BoundaryConstraint {
                cycle: NUM_ROUNDS,
                register: i,
                value: output_elements[i].to_owned(),
            };
            bcs.push(end_constraint);
        }

        bcs
    }
}

#[cfg(test)]
pub mod test_stark {
    use super::*;

    use rand::Rng;
    use serde_json;

    use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
    use twenty_first::timing_reporter::TimingReporter;

    fn gen_polynomial() -> Polynomial<BFieldElement> {
        let mut rng = rand::thread_rng();
        let coefficient_count: usize = rng.gen_range(0..40);

        Polynomial {
            coefficients: random_elements(coefficient_count),
        }
    }

    #[test]
    fn lift_b_x_test() {
        for _ in 0..5 {
            let pol = gen_polynomial();
            let lifted_pol: Polynomial<XFieldElement> = StarkRp::lift_b_x(&pol);
            for (coefficient, lifted_coefficient) in
                pol.coefficients.iter().zip(lifted_pol.coefficients.iter())
            {
                assert_eq!(Some(*coefficient), lifted_coefficient.unlift());
            }
        }
    }

    #[test]
    #[ignore = "too slow"]
    fn air_is_zero_on_execution_trace_test() {
        // let rp = params::rescue_prime_medium_test_params();
        // let rp = params::rescue_prime_params_bfield_0();

        // rescue prime test vector 1
        let omicron_res = BFieldElement::primitive_root_of_unity(1 << 5);
        let omicron = omicron_res.unwrap();

        // Verify that the round constants polynomials are correct
        let (fst_rc_pol, snd_rc_pol) = StarkRp::get_round_constant_polynomials(omicron);
        for step in 0..NUM_ROUNDS {
            let mut point = vec![omicron.mod_pow(step as u64)];
            point.append(&mut vec![BFieldElement::zero(); 2 * STATE_SIZE]);

            for (register, item) in fst_rc_pol.iter().enumerate().take(STATE_SIZE) {
                let fst_eval = item.evaluate(&point);
                assert_eq!(
                    Into::<BFieldElement>::into(ROUND_CONSTANTS[2 * step * STATE_SIZE + register]),
                    fst_eval
                );
            }
            for (register, item) in snd_rc_pol.iter().enumerate().take(STATE_SIZE) {
                let snd_eval = item.evaluate(&point);
                assert_eq!(
                    Into::<BFieldElement>::into(
                        ROUND_CONSTANTS[2 * step * STATE_SIZE + STATE_SIZE + register]
                    ),
                    snd_eval
                );
            }
        }

        // Verify that the AIR constraints evaluation over the trace is zero along the trace
        let input_2 = [BFieldElement::new(42); 10];
        let trace = RescuePrimeRegular::trace(&input_2);
        println!("Computing get_air_constraints(omicron)...");
        let now = std::time::Instant::now();
        let air_constraints = StarkRp::get_air_constraints(omicron);
        let elapsed = now.elapsed();
        println!("Completed get_air_constraints(omicron) in {:?}!", elapsed);

        for round in 0..NUM_ROUNDS - 1 {
            println!("Round {}", round);
            for air_constraint in air_constraints.iter() {
                let mut point = vec![omicron.mod_pow(round as u64)];
                for i in 0..STATE_SIZE {
                    point.push(trace[round][i]);
                }
                for i in 0..STATE_SIZE {
                    point.push(trace[round + 1][i]);
                }
                let eval = air_constraint.evaluate(&point);
                assert!(eval.is_zero());
            }
        }
    }

    #[test]
    #[ignore = "too slow"]
    fn prove_and_verify_small_stark_test() {
        let num_colinearity_checks = 2;
        let num_randomizers = 4 * num_colinearity_checks as usize;
        let stark: StarkRp = StarkRp::new(
            16,
            num_colinearity_checks,
            STATE_SIZE as u32,
            BFieldElement::new(7),
        );

        let mut input = [BFieldElement::zero(); 10];
        input[0] = BFieldElement::one();
        let (output, trace) = RescuePrimeRegular::hash_10_with_trace(&input);
        assert_eq!(9, trace.len());

        let mut npo2 = trace.len() + num_randomizers;
        if npo2 & (npo2 - 1) != 0 {
            npo2 <<= 1;
            while npo2 & (npo2 - 1) != 0 {
                npo2 &= npo2 - 1;
            }
        }

        let omicron = BFieldElement::primitive_root_of_unity(npo2 as u64).unwrap();
        let air_constraints = StarkRp::get_air_constraints(omicron);
        let boundary_constraints = StarkRp::get_boundary_constraints(&output);
        let mut proof_stream = ProofStream::default();

        let prove_result = stark.prove(
            &trace,
            &air_constraints,
            &boundary_constraints,
            &mut proof_stream,
            omicron,
        );

        match prove_result {
            Ok(_) => (),
            Err(e) => panic!("{}", e),
        };

        let (fri_domain_length, omega): (u32, BFieldElement) = prove_result.unwrap();

        let verify_result = stark.verify(
            &mut proof_stream,
            &air_constraints,
            &boundary_constraints,
            fri_domain_length,
            omega,
            trace.len() as u32,
        );

        match verify_result {
            Ok(_) => (),
            Err(e) => panic!("{}", e),
        };
    }

    #[test]
    #[ignore = "too slow"]
    fn prove_and_verify_medium_stark_test() {
        // let rp: RescuePrime = params::rescue_prime_params_bfield_0();
        let num_colinearity_checks = 2;
        let num_randomizers = num_colinearity_checks * 4;
        let stark: StarkRp = StarkRp::new(
            16,
            num_colinearity_checks,
            STATE_SIZE as u32,
            BFieldElement::new(7),
        );

        let mut input = [BFieldElement::zero(); 10];
        input[0] = BFieldElement::one();
        let (output, trace) = RescuePrimeRegular::hash_10_with_trace(&input);

        let mut npo2 = trace.len() + num_randomizers as usize;
        if npo2 & (npo2 - 1) != 0 {
            npo2 <<= 1;
            while npo2 & (npo2 - 1) != 0 {
                npo2 &= npo2 - 1;
            }
        }
        let omicron = BFieldElement::primitive_root_of_unity(npo2 as u64).unwrap();

        let mut timer = TimingReporter::start();
        let air_constraints = StarkRp::get_air_constraints(omicron);
        timer.elapsed("get_air_constraints(omicron)");
        let boundary_constraints = StarkRp::get_boundary_constraints(&output);
        timer.elapsed("get_boundary_constraints(output)");
        let report = timer.finish();
        println!("{}", report);

        let mut proof_stream = ProofStream::default();

        let prove_result = stark.prove(
            &trace,
            &air_constraints,
            &boundary_constraints,
            &mut proof_stream,
            omicron,
        );

        assert!(prove_result.is_ok());

        let (fri_domain_length, omega): (u32, BFieldElement) = prove_result.unwrap();

        let verify_result = stark.verify(
            &mut proof_stream,
            &air_constraints,
            &boundary_constraints,
            fri_domain_length,
            omega,
            trace.len() as u32,
        );

        println!("stark params: {}", serde_json::to_string(&stark).unwrap());
        println!("proof_stream: {} bytes", proof_stream.len());

        assert!(verify_result.is_ok());
    }

    #[test]
    #[ignore = "too slow"]
    fn stark_with_registers_without_boundary_conditions_test() {
        // Use the small test parameter set but with an input length of 2
        // and an output length of 1. This leaves register 1 (execution trace
        // has two registers: `0` and `1`) without any boundary condition.
        let num_colinearity_checks = 2;
        let num_randomizers = num_colinearity_checks * 4;
        let stark: StarkRp = StarkRp::new(
            16,
            num_colinearity_checks,
            STATE_SIZE as u32,
            BFieldElement::new(7),
        );
        let mut input = [BFieldElement::zero(); 10];
        input[0] = BFieldElement::one();
        let (output, trace) = RescuePrimeRegular::hash_10_with_trace(&input);
        assert_eq!(9, trace.len());

        let mut npo2 = trace.len() + num_randomizers as usize;
        if npo2 & (npo2 - 1) != 0 {
            npo2 <<= 1;
            while npo2 & (npo2 - 1) != 0 {
                npo2 &= npo2 - 1;
            }
        }

        let omicron = BFieldElement::primitive_root_of_unity(npo2 as u64).unwrap();
        let air_constraints = StarkRp::get_air_constraints(omicron);
        let boundary_constraints = StarkRp::get_boundary_constraints(&output);
        let mut proof_stream = ProofStream::default();

        let prove_result = stark.prove(
            &trace,
            &air_constraints,
            &boundary_constraints,
            &mut proof_stream,
            omicron,
        );

        let (fri_domain_length, omega) = match prove_result {
            Ok(res) => res,
            Err(e) => panic!("Error while generating proof: {}", e),
        };

        let verify_result = stark.verify(
            &mut proof_stream,
            &air_constraints,
            &boundary_constraints,
            fri_domain_length,
            omega,
            trace.len() as u32,
        );

        assert!(verify_result.is_ok());
    }
}
