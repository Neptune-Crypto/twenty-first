use rand::prelude::ThreadRng;

use crate::shared_math::ntt::intt;
// use crate::shared_math::fri::ValidationError;
// use crate::shared_math::traits::{
//     FieldBatchInversion, GetGeneratorDomain, GetRandomElements,
//     ModPowU32, PrimeFieldElement,
// };
// use crate::util_types::proof_stream::ProofStream;
// use crate::utils::{blake3_digest, get_index_from_bytes};
// use super::mpolynomial::MPolynomial;
// use crate::shared_math::ntt::intt;
use crate::shared_math::traits::{CyclicGroupGenerator, IdentityValues};
use crate::shared_math::x_field_element::XFieldElement;
use crate::util_types::merkle_tree::{MerkleTree, PartialAuthenticationPath};
use crate::util_types::proof_stream::ProofStream;
use crate::utils;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use super::b_field_element::BFieldElement;
use super::mpolynomial::MPolynomial;
use super::other::roundup_npo2;
use super::polynomial::Polynomial;
use super::traits::GetRandomElements;
use super::x_field_fri::Fri;

pub const DOCUMENT_HASH_LENGTH: usize = 32usize;
pub const MERKLE_ROOT_HASH_LENGTH: usize = 32usize;

// TODO: Consider <B: PrimeFieldElement, X: PrimeFieldElement>
// This requires a trait a la Lift<X> to generalise XFE::new_const().
#[derive(Clone, Debug)]
pub struct Stark {
    expansion_factor: u32,
    field_generator: BFieldElement,
    colinearity_check_count: u32,
    min_num_randomizers: u32,
    num_registers: u32,
}

// TODO: Consider impl<B: PrimeFieldElement, X: PrimeFieldElement>
impl Stark {
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

#[derive(Clone, Debug)]
pub struct BoundaryConstraint {
    pub cycle: usize,
    pub register: usize,
    pub value: BFieldElement,
}

// A hashmap from register value to (x, y) value of boundary constraint
pub type BoundaryConstraintsMap = HashMap<usize, (BFieldElement, BFieldElement)>;

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

impl Stark {
    pub fn prove(
        &self,
        // Trace is indexed as trace[cycle][register]
        trace: &Vec<Vec<BFieldElement>>,
        transition_constraints: &Vec<MPolynomial<BFieldElement>>,
        boundary_constraints: &Vec<BoundaryConstraint>,
        proof_stream: &mut ProofStream,
        input_omicron: BFieldElement,
    ) -> Result<(u32, BFieldElement), Box<dyn Error>> {
        // infer details about computation
        let original_trace_length = trace.len() as u64;
        let rounded_trace_length =
            roundup_npo2(original_trace_length + self.min_num_randomizers as u64);
        let num_randomizers = rounded_trace_length - original_trace_length;

        // let tp_degree = air_degree * (rounded_trace_length - 1);
        let tp_bounds =
            self.transition_degree_bounds(&transition_constraints, rounded_trace_length as usize);
        let tp_degree = tp_bounds.iter().max().unwrap();

        let tq_degree = tp_degree - (original_trace_length - 1);
        let max_degree = roundup_npo2(tq_degree + 1) - 1; // The max degree bound provable by FRI
        let fri_domain_length = (max_degree + 1) * self.expansion_factor as u64;
        let blowup_factor_new = fri_domain_length / rounded_trace_length;

        // compute generators
        let omega = BFieldElement::get_primitive_root_of_unity(fri_domain_length as u128)
            .0
            .unwrap();

        let omicron_domain_length = rounded_trace_length;
        let omicron = BFieldElement::get_primitive_root_of_unity(omicron_domain_length as u128)
            .0
            .unwrap();

        assert_eq!(
            input_omicron, omicron,
            "Calculated omicron must match input omicron"
        );

        // ...
        let mut rng = rand::thread_rng();
        let mut randomized_trace = trace.clone();
        self.randomize_trace(&mut rng, &mut randomized_trace, num_randomizers);

        // ...
        let mut trace_interpolants = vec![];
        for r in 0..self.num_registers as usize {
            let trace_column = &randomized_trace
                .iter()
                .map(|t| t[r])
                .collect::<Vec<BFieldElement>>();

            let trace_interpolant_coefficients = intt(trace_column, &omicron);
            let trace_interpolant = Polynomial {
                coefficients: trace_interpolant_coefficients,
            };

            // Sanity checks; consider moving into unit tests.
            for (i, item) in trace.iter().enumerate() {
                assert_eq!(
                    item[r],
                    trace_interpolant.evaluate(&omicron.mod_pow(i as u64)),
                    "Trace interpolant evaluates back to trace element"
                );
            }
            assert_eq!(
                (randomized_trace.len() - 1) as isize,
                trace_interpolant.degree(),
                "Trace interpolant has one degree lower than the points"
            );

            trace_interpolants.push(trace_interpolant);
        }

        // Sanity check
        for ti in trace_interpolants.iter() {
            assert!(ti.degree() == rounded_trace_length as isize - 1);
        }

        //// bq(x) = (ti(x) - bi(x)) / bz(x)
        //
        // where
        //   bq: boundary quotients
        //   ti: boundary interpolants
        //   bz: boundary zerofiers

        // Subtract boundary interpolants and divide out boundary zerofiers
        let bcs_formatted = self.format_boundary_constraints(omicron, boundary_constraints);
        let boundary_interpolants: Vec<Polynomial<BFieldElement>> =
            self.get_boundary_interpolants(bcs_formatted.clone());
        let boundary_zerofiers: Vec<Polynomial<BFieldElement>> =
            self.get_boundary_zerofiers(bcs_formatted.clone());
        let mut boundary_quotients: Vec<Polynomial<BFieldElement>> =
            vec![Polynomial::ring_zero(); self.num_registers as usize];

        for r in 0..self.num_registers as usize {
            let div_res = (trace_interpolants[r].clone() - boundary_interpolants[r].clone())
                .divide(boundary_zerofiers[r].clone());
            assert_eq!(
                Polynomial::<BFieldElement>::ring_zero(),
                div_res.1,
                "Remainder must be zero when dividing out boundary zerofier"
            );
            boundary_quotients[r] = div_res.0;
        }

        // Commit to boundary quotients
        // TODO: Consider salted Merkle trees here.
        let mut boundary_quotient_merkle_trees: Vec<MerkleTree<BFieldElement>> = vec![];
        for bq in boundary_quotients.iter() {
            let boundary_quotient_codeword: Vec<BFieldElement> =
                bq.fast_coset_evaluate(&self.field_generator, &omega, fri_domain_length as usize);
            let bq_merkle_tree = MerkleTree::from_vec(&boundary_quotient_codeword);
            proof_stream.enqueue(&bq_merkle_tree.get_root())?;
            boundary_quotient_merkle_trees.push(bq_merkle_tree);
        }

        //// tq(x) = tp(x) / tz(x)
        //
        // where
        //   tq(x) = transition quotients
        //   tp(x) = transition polynomials
        //   tz(x) = transition zerofiers

        // Symbolically evaluate transition constraints
        let x = Polynomial {
            coefficients: vec![BFieldElement::ring_zero(), BFieldElement::ring_one()],
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
        let transition_polynomials: Vec<Polynomial<BFieldElement>> = transition_constraints
            .iter()
            .map(|x| x.evaluate_symbolic(&point))
            .collect();

        // TODO: Sanity check REMOVE
        for tp in transition_polynomials.iter() {
            assert_eq!(*tp_degree, tp.degree() as u64);
            for i in 0..original_trace_length - 1 {
                let x = omicron.mod_pow(i);
                assert!(tp.evaluate(&x).is_zero());
            }
        }

        let transition_zerofier: Polynomial<BFieldElement> = self.get_transition_zerofier(
            omicron,
            omicron_domain_length as usize,
            original_trace_length as usize,
        );

        // FIXME: Use this in combination with LeaflessPartialAuthenticationPaths.
        let _transition_zerofier_mt: MerkleTree<BFieldElement> = self.get_transition_zerofier_mt(
            &transition_zerofier,
            omega,
            fri_domain_length as usize,
        );

        // divide out transition zerofier
        let transition_quotients: Vec<Polynomial<BFieldElement>> = transition_polynomials
            .iter()
            .map(|tp| {
                // Sanity check, remove
                let (_quot, rem) = tp.divide(transition_zerofier.clone());
                assert_eq!(
                    Polynomial::ring_zero(),
                    rem,
                    "Remainder must be zero when calculating transition quotient"
                );

                Polynomial::fast_coset_divide(
                    tp,
                    &transition_zerofier,
                    &self.field_generator,
                    &omega,
                    fri_domain_length as usize,
                )
            })
            .collect();

        // Commit to randomizer polynomial
        let randomizer_polynomial = Polynomial {
            coefficients: BFieldElement::random_elements(max_degree as usize + 1, &mut rng),
        };

        let randomizer_codeword: Vec<BFieldElement> = randomizer_polynomial.fast_coset_evaluate(
            &self.field_generator,
            &omega,
            fri_domain_length as usize,
        );
        let randomizer_mt = MerkleTree::from_vec(&randomizer_codeword);
        proof_stream.enqueue(&randomizer_mt.get_root())?;

        let expected_tq_degrees = self.transition_quotient_degree_bounds(
            &transition_constraints,
            original_trace_length as usize,
            rounded_trace_length as usize,
        );

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

        let mut terms: Vec<Polynomial<BFieldElement>> = vec![randomizer_polynomial];
        for (tq, tq_degree) in transition_quotients.iter().zip(expected_tq_degrees.iter()) {
            terms.push(tq.to_owned());
            let shift = max_degree - tq_degree;

            // Make new polynomial with max_degree degree by shifting all terms up
            let shifted = tq.shift_coefficients(shift as usize, BFieldElement::ring_zero());
            assert_eq!(max_degree as isize, shifted.degree()); // TODO: Can be removed
            terms.push(shifted);
        }
        for (bq, bq_degree) in boundary_quotients.iter().zip(boundary_degrees.iter()) {
            terms.push(bq.to_owned());
            let shift = max_degree as usize - bq_degree;

            // Make new polynomial with max_degree degree by shifting all terms up
            let shifted = bq.shift_coefficients(shift, BFieldElement::ring_zero());
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
            &omega,
            fri_domain_length as usize,
        );

        // Prove low degree of combination polynomial, and collect indices
        let fri = Fri::<XFieldElement>::new(
            XFieldElement::new_const(self.field_generator),
            XFieldElement::new_const(omega),
            fri_domain_length as usize,
            self.expansion_factor as usize,
            self.colinearity_check_count as usize,
        );

        // Since we're working in the extension field...
        let lifted_combined_codeword: Vec<XFieldElement> = combined_codeword
            .iter()
            .map(|x| XFieldElement::new_const(*x))
            .collect();

        let indices: Vec<usize> = fri.prove(&lifted_combined_codeword, proof_stream)?;

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

        // FIXME: Use newer leafless_* MT functions below. They don't currently export the subset of values.
        // Vec<LeaflessPartialAuthenticationPath> -> Vec<(LeaflessPartialAuthenticationPath, U)>

        // Open indicated positions in the boundary quotient codewords
        for bq_mt in boundary_quotient_merkle_trees {
            let authentication_paths: Vec<PartialAuthenticationPath<BFieldElement>> =
                bq_mt.get_multi_proof(&quadrupled_indices);
            proof_stream.enqueue_length_prepended(&authentication_paths)?;
        }

        // Open indicated positions in the randomizer
        let randomizer_auth_path = randomizer_mt.get_multi_proof(&quadrupled_indices);
        proof_stream.enqueue_length_prepended(&randomizer_auth_path)?;

        Ok((fri_domain_length as u32, omega))
    }

    pub fn verify(
        &self,
        proof_stream: &mut ProofStream,
        transition_constraints: &Vec<MPolynomial<BFieldElement>>,
        boundary_constraints: &Vec<BoundaryConstraint>,
        fri_domain_length: u32,
        omega: BFieldElement,
        original_trace_length: u32,
    ) -> Result<(), Box<dyn Error>> {
        assert!(omega.mod_pow(fri_domain_length as u64).is_one());
        assert!(!omega.mod_pow((fri_domain_length / 2) as u64).is_one());

        // Get Merkle root of boundary quotient codewords
        let mut boundary_quotient_mt_roots: Vec<[u8; 32]> = vec![];
        for _ in 0..self.num_registers {
            boundary_quotient_mt_roots.push(proof_stream.dequeue(32)?);
        }

        let randomizer_mt_root: [u8; 32] = proof_stream.dequeue(32)?;

        // Get weights for nonlinear combination
        // 1 weight element for randomizer
        // 2 for every transition quotient
        // 2 for every boundary quotient
        let fiat_shamir_hash: Vec<u8> = proof_stream.verifier_fiat_shamir();
        let weights = self.sample_weights(
            &fiat_shamir_hash,
            1 + 2 * boundary_quotient_mt_roots.len() + 2 * transition_constraints.len(),
        );

        // Verify low degree of combination polynomial, and collect indices
        // Note that FRI verifier verifies number of samples, so we don't have
        // to check that number here
        let fri = Fri::<XFieldElement>::new(
            XFieldElement::new_const(self.field_generator),
            XFieldElement::new_const(omega),
            fri_domain_length as usize,
            self.expansion_factor as usize,
            self.colinearity_check_count as usize,
        );

        let polynomial_values = fri.verify(proof_stream)?;

        let (indices, x_field_values): (Vec<usize>, Vec<XFieldElement>) =
            polynomial_values.into_iter().unzip();

        let unlifted_values: Option<Vec<BFieldElement>> =
            x_field_values.iter().map(|xv| xv.unlift()).collect();

        let values: Vec<BFieldElement> =
            unlifted_values.ok_or(StarkVerifyError::UnexpectedFriValue)?;

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

        // Read and verify boundary quotient leafs
        // revealed boundary quotient codeword values, indexed by (register, codeword index)
        let mut boundary_quotients: Vec<HashMap<usize, BFieldElement>> = vec![];
        for (i, bq_root) in boundary_quotient_mt_roots.into_iter().enumerate() {
            boundary_quotients.push(HashMap::new());
            let authentication_paths: Vec<PartialAuthenticationPath<BFieldElement>> =
                proof_stream.dequeue_length_prepended()?;
            let valid =
                MerkleTree::verify_multi_proof(bq_root, &duplicated_indices, &authentication_paths);
            if !valid {
                return Err(Box::new(StarkVerifyError::BadMerkleProof(
                    MerkleProofError::BoundaryQuotientError(i),
                )));
            }

            duplicated_indices
                .iter()
                .zip(authentication_paths.iter())
                .for_each(|(index, authentication_path)| {
                    boundary_quotients[i].insert(*index, authentication_path.get_value());
                });
        }

        // Read and verify randomizer leafs
        let randomizer_authentication_paths: Vec<PartialAuthenticationPath<BFieldElement>> =
            proof_stream.dequeue_length_prepended()?;
        let valid = MerkleTree::verify_multi_proof(
            randomizer_mt_root,
            &duplicated_indices,
            &randomizer_authentication_paths,
        );
        if !valid {
            return Err(Box::new(StarkVerifyError::BadMerkleProof(
                MerkleProofError::RandomizerError,
            )));
        }

        let mut randomizer_values: HashMap<usize, BFieldElement> = HashMap::new();
        duplicated_indices
            .iter()
            .zip(randomizer_authentication_paths.iter())
            .for_each(|(index, authentication_path)| {
                randomizer_values.insert(*index, authentication_path.get_value());
            });

        // FIXME: Can we calculate this faster using omega?
        let omicron = BFieldElement::get_primitive_root_of_unity(omicron_domain_length as u128)
            .0
            .unwrap();

        // Verify leafs of combination polynomial
        let formatted_bcs = self.format_boundary_constraints(omicron, boundary_constraints);
        let boundary_zerofiers = self.get_boundary_zerofiers(formatted_bcs.clone());
        let boundary_interpolants = self.get_boundary_interpolants(formatted_bcs);
        let boundary_degrees = self
            .boundary_quotient_degree_bounds(&boundary_zerofiers, rounded_trace_length as usize);
        let expected_tq_degrees = self.transition_quotient_degree_bounds(
            &transition_constraints,
            original_trace_length as usize,
            rounded_trace_length as usize,
        );

        // TODO: Find a way to calculate the transition_zerofier faster than this.
        let transition_zerofier: Polynomial<BFieldElement> = self.get_transition_zerofier(
            omicron,
            omicron_domain_length as usize,
            original_trace_length as usize,
        );

        for (i, current_index) in indices.into_iter().enumerate() {
            let current_x: BFieldElement =
                self.field_generator.clone() * omega.mod_pow(current_index as u64);
            let next_index: usize =
                (current_index + blowup_factor_new as usize) % fri_domain_length as usize;
            let next_x: BFieldElement =
                self.field_generator.clone() * omega.mod_pow(next_index as u64);
            let mut current_trace: Vec<BFieldElement> = (0..self.num_registers as usize)
                .map(|r| {
                    boundary_quotients[r][&current_index].clone()
                        * boundary_zerofiers[r].evaluate(&current_x)
                        + boundary_interpolants[r].evaluate(&current_x)
                })
                .collect();
            let mut next_trace: Vec<BFieldElement> = (0..self.num_registers as usize)
                .map(|r| {
                    boundary_quotients[r][&next_index].clone()
                        * boundary_zerofiers[r].evaluate(&next_x)
                        + boundary_interpolants[r].evaluate(&next_x)
                })
                .collect();

            let mut point: Vec<BFieldElement> = vec![current_x.clone()];
            point.append(&mut current_trace);
            point.append(&mut next_trace);

            let transition_constraint_values: Vec<BFieldElement> = transition_constraints
                .iter()
                .map(|tc| tc.evaluate(&point))
                .collect();

            let current_transition_zerofier_value: BFieldElement =
                transition_zerofier.evaluate(&current_x);

            // Get combination polynomial evaluation value
            // Loop over all registers for transition quotient values, and for boundary quotient values
            let mut terms: Vec<BFieldElement> = vec![randomizer_values[&current_index]];
            for (tcv, tq_degree) in transition_constraint_values
                .iter()
                .zip(expected_tq_degrees.iter())
            {
                let transition_quotient = *tcv / current_transition_zerofier_value;
                terms.push(transition_quotient);
                let shift = max_degree as u64 - tq_degree;
                terms.push(transition_quotient * current_x.mod_pow(shift));
            }
            for (bqvs, bq_degree) in boundary_quotients.iter().zip(boundary_degrees.iter()) {
                terms.push(bqvs[&current_index]);
                let shift = max_degree as u64 - *bq_degree as u64;
                terms.push(bqvs[&current_index] * current_x.mod_pow(shift));
            }

            assert_eq!(
                weights.len(),
                terms.len(),
                "weights and terms length must match in verifier"
            );
            let combination = weights
                .iter()
                .zip(terms.iter())
                .fold(BFieldElement::ring_zero(), |sum, (weight, term)| {
                    sum + term.to_owned() * weight.to_owned()
                });

            if values[i] != combination {
                return Err(Box::new(StarkVerifyError::LinearCombinationMismatch(
                    current_index,
                )));
            }
        }

        Ok(())
    }

    fn randomize_trace(
        &self,
        mut rng: &mut ThreadRng,
        trace: &mut Vec<Vec<BFieldElement>>,
        num_randomizers: u64,
    ) {
        let mut randomizer_coset: Vec<Vec<BFieldElement>> = (0..num_randomizers)
            .map(|_| BFieldElement::random_elements(self.num_registers as usize, &mut rng))
            .collect();

        trace.append(&mut randomizer_coset);
    }

    // Convert boundary constraints into a vector of boundary constraints indexed by register.
    fn format_boundary_constraints(
        &self,
        omicron: BFieldElement,
        boundary_constraints: &Vec<BoundaryConstraint>,
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
    // input is indexed with bcs[register][cycle]
    fn get_boundary_interpolants(
        &self,
        bcs: Vec<Vec<(BFieldElement, BFieldElement)>>,
    ) -> Vec<Polynomial<BFieldElement>> {
        bcs.iter()
            .map(|points| Polynomial::slow_lagrange_interpolation(points))
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
            .map(|points| Polynomial::get_polynomial_with_roots(points))
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
    ) -> Vec<u64> {
        // The degree is the degree of the trace plus the randomizers
        // minus the original trace length minus 1.
        self.transition_degree_bounds(transition_constraints, rounded_trace_length)
            .iter()
            .map(|d| d - (original_trace_length as u64 - 1))
            .collect()
    }

    // Return the max degree for all interpolations of the execution trace
    fn transition_degree_bounds(
        &self,
        transition_constraints: &[MPolynomial<BFieldElement>],
        randomized_trace_length: usize,
    ) -> Vec<u64> {
        let mut point_degrees: Vec<u64> = vec![0; 1 + 2 * self.num_registers as usize];
        point_degrees[0] = 1;
        for r in 0..self.num_registers as usize {
            point_degrees[1 + r] = randomized_trace_length as u64 - 1;
            point_degrees[1 + r + self.num_registers as usize] = randomized_trace_length as u64 - 1;
        }

        // This could also be achieved with symbolic evaluation of the
        // input `transition_constraints` and then taking the degree of
        // the resulting polynomials.
        let mut res: Vec<u64> = vec![];
        for a in transition_constraints {
            let mut max_degree = 0;
            for (k, _) in a.coefficients.iter() {
                let mut acc = 0;
                for (r, l) in point_degrees.iter().zip(k.iter()) {
                    acc += *r * (*l);
                }
                if acc > max_degree {
                    max_degree = acc;
                }
            }
            res.push(max_degree);
        }

        res
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

    // TODO: Consider naming this something with "evaluate"
    fn get_transition_zerofier_mt(
        &self,
        transition_zerofier: &Polynomial<BFieldElement>,
        omega: BFieldElement,
        fri_domain_length: usize,
    ) -> MerkleTree<BFieldElement> {
        let transition_zerofier_codeword: Vec<BFieldElement> = transition_zerofier
            .fast_coset_evaluate(&self.field_generator, &omega, fri_domain_length);

        MerkleTree::from_vec(&transition_zerofier_codeword)
    }

    fn sample_weights(&self, randomness: &[u8], number: usize) -> Vec<BFieldElement> {
        let k_seeds = utils::get_n_hash_rounds(randomness, number as u32);

        // TODO: BFieldElement::from assumes something about the hash size.
        // Make sure we change this when changing the hash function.
        k_seeds
            .iter()
            .map(|seed| BFieldElement::from(seed.to_vec()))
            .collect::<Vec<BFieldElement>>()
    }
}

#[cfg(test)]
pub mod test_stark {
    use super::*;
    use crate::shared_math::rescue_prime::RescuePrime;
    use crate::shared_math::rescue_prime_params as params;

    #[test]
    fn prove_something_test() {
        let rp: RescuePrime = params::rescue_prime_small_test_params();
        let stark: Stark = Stark::new(16, 2, 2, BFieldElement::new(7));

        let one = BFieldElement::ring_one();
        let (output, trace) = rp.eval_and_trace(&one);
        assert_eq!(4, trace.len());

        let omicron = BFieldElement::get_primitive_root_of_unity(16).0.unwrap();
        let air_constraints = rp.get_air_constraints(omicron);
        let boundary_constraints = rp.get_boundary_constraints(output);
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

        assert!(verify_result.is_ok());
    }

    #[test]
    fn prove_and_verify_medium_stark_test() {
        let rp: RescuePrime = params::rescue_prime_medium_test_params();
        let stark: Stark = Stark::new(16, 2, rp.m as u32, BFieldElement::new(7));

        let one = BFieldElement::ring_one();
        let (output, trace) = rp.eval_and_trace(&one);
        assert_eq!(6, trace.len());

        let omicron = BFieldElement::get_primitive_root_of_unity(16).0.unwrap();
        let air_constraints = rp.get_air_constraints(omicron);
        let boundary_constraints = rp.get_boundary_constraints(output);
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

        assert!(verify_result.is_ok());
    }
}
