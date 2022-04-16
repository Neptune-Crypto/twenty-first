use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::b_field_element::BFieldElement;
use super::other::{log_2_ceil, log_2_floor};
use super::polynomial::Polynomial;
use super::stark::brainfuck::stark_proof_stream::{Item, StarkProofStream};
use super::traits::{CyclicGroupGenerator, ModPowU32};
use super::x_field_element::XFieldElement;
use crate::shared_math::ntt::{intt, ntt};
use crate::shared_math::traits::{IdentityValues, PrimeField};
use crate::util_types::merkle_tree::{LeaflessPartialAuthenticationPath, MerkleTree};
use crate::util_types::simple_hasher::{Hasher, ToDigest};
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

impl Error for ValidationError {}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Deserialization error for LowDegreeProof: {:?}", self)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum ValidationError {
    BadMerkleProof,
    BadSizedProof,
    NonPostiveRoundCount,
    NotColinear(usize),
    LastIterationTooHighDegree,
    BadMerkleRootForLastCodeword,
}

#[derive(Debug, Clone)]
pub struct FriDomain {
    pub offset: XFieldElement,
    pub omega: XFieldElement,
    pub length: usize,
}

impl FriDomain {
    pub fn x_evaluate(&self, polynomial: &Polynomial<XFieldElement>) -> Vec<XFieldElement> {
        polynomial.fast_coset_evaluate(&self.offset, self.omega, self.length as usize)
    }

    pub fn x_interpolate(&self, values: &[XFieldElement]) -> Polynomial<XFieldElement> {
        Polynomial::<XFieldElement>::fast_coset_interpolate(&self.offset, self.omega, values)
    }

    pub fn b_domain_value(&self, index: u32) -> BFieldElement {
        self.omega.unlift().unwrap().mod_pow_u32(index) * self.offset.unlift().unwrap()
    }

    pub fn b_domain_values(&self) -> Vec<BFieldElement> {
        (0..self.length)
            .map(|i| {
                self.omega.unlift().unwrap().mod_pow_u32(i as u32) * self.offset.unlift().unwrap()
            })
            .collect()
    }

    pub fn b_evaluate(
        &self,
        polynomial: &Polynomial<BFieldElement>,
        zero: BFieldElement,
    ) -> Vec<BFieldElement> {
        assert!(zero.is_zero(), "zero must be zero");
        let mut polynomial_representation: Vec<BFieldElement> = polynomial
            .scale(&self.offset.unlift().unwrap())
            .coefficients;
        polynomial_representation.resize(self.length as usize, zero);
        ntt(
            &mut polynomial_representation,
            self.omega.unlift().unwrap(),
            log_2_ceil(self.length as u64) as u32,
        );

        polynomial_representation
    }

    pub fn b_interpolate(&self, values: &[BFieldElement]) -> Polynomial<BFieldElement> {
        Polynomial::<BFieldElement>::fast_coset_interpolate(
            &self.offset.unlift().unwrap(),
            self.omega.unlift().unwrap(),
            values,
        )
    }
}

#[derive(Debug, Clone)]
pub struct Fri<H> {
    pub expansion_factor: usize,         // = domain_length / trace_length
    pub colinearity_checks_count: usize, // number of colinearity checks in each round
    pub domain: FriDomain,
    _hasher: PhantomData<H>,
}

type CodewordEvaluation<T> = (usize, T);

impl<H> Fri<H>
where
    H: Hasher<Digest = Vec<BFieldElement>> + std::marker::Sync,
{
    pub fn new(
        offset: XFieldElement,
        omega: XFieldElement,
        domain_length: usize,
        expansion_factor: usize,
        colinearity_checks_count: usize,
    ) -> Self {
        let domain = FriDomain {
            offset,
            omega,
            length: domain_length,
        };
        let _hasher = PhantomData;
        Self {
            domain,
            expansion_factor,
            colinearity_checks_count,
            _hasher,
        }
    }

    pub fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut StarkProofStream,
    ) -> Result<Vec<usize>, Box<dyn Error>> {
        assert_eq!(
            self.domain.length,
            codeword.len(),
            "Initial codeword length must match that set in FRI object"
        );

        // Commit phase
        #[allow(clippy::type_complexity)]
        let values_and_merkle_trees: Vec<(
            Vec<XFieldElement>,
            MerkleTree<Vec<BFieldElement>, H>,
        )> = self.commit(codeword, proof_stream)?;
        let codewords: Vec<Vec<XFieldElement>> = values_and_merkle_trees
            .iter()
            .map(|x| x.0.clone())
            .collect();

        // fiat-shamir phase (get indices)
        let top_level_indices: Vec<usize> = self.sample_indices(&proof_stream.prover_fiat_shamir());

        // query phase
        let mut c_indices = top_level_indices.clone();
        for i in 0..values_and_merkle_trees.len() - 1 {
            c_indices = c_indices
                .clone()
                .iter()
                .map(|x| x % (codewords[i].len() / 2))
                .collect();

            self.query(
                values_and_merkle_trees[i].clone(),
                values_and_merkle_trees[i + 1].clone(),
                &c_indices,
                proof_stream,
            )?;
        }

        Ok(top_level_indices)
    }

    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut StarkProofStream,
    ) -> Result<Vec<(Vec<XFieldElement>, MerkleTree<Vec<BFieldElement>, H>)>, Box<dyn Error>> {
        let mut generator = self.domain.omega;
        let mut offset = self.domain.offset;
        let mut codeword_local = codeword.to_vec();

        let zero: XFieldElement = generator.ring_zero();
        let mt_dummy_value: Vec<BFieldElement> = vec![BFieldElement::ring_zero()];
        let one: XFieldElement = generator.ring_one();
        let two: XFieldElement = one + one;
        let two_inv = one / two;

        // Compute and send Merkle root
        let hasher = H::new();
        let mut digests: Vec<Vec<BFieldElement>> = Vec::with_capacity(codeword_local.len());
        codeword_local
            .clone()
            .into_par_iter()
            .map(|xfe| {
                let digest: Vec<BFieldElement> = xfe.coefficients.into();
                hasher.hash(&digest)
            })
            .collect_into_vec(&mut digests);
        let mut mt: MerkleTree<Vec<BFieldElement>, H> =
            MerkleTree::from_digests(&digests, &mt_dummy_value);
        let mt_root: &<H as Hasher>::Digest = mt.get_root();
        // proof_stream.enqueue_length_prepended(mt_root)?;
        proof_stream.enqueue(&Item::MerkleRoot(mt_root.to_owned()));
        let mut values_and_merkle_trees = vec![(codeword_local.clone(), mt)];

        let (num_rounds, _) = self.num_rounds();
        for _ in 0..num_rounds {
            let n = codeword_local.len();

            // Sanity check to verify that generator has the right order; requires ModPowU64
            //assert!(generator.inv() == generator.mod_pow((n - 1).into())); // TODO: REMOVE

            // Get challenge, one just acts as *any* element in this field -- the field element
            // is completely determined from the byte stream.
            let _alpha: H::Digest = proof_stream.prover_fiat_shamir();
            let alpha: XFieldElement = XFieldElement::new([_alpha[0], _alpha[1], _alpha[2]]);

            let x_offset: Vec<XFieldElement> = generator
                .get_cyclic_group_elements(None)
                .into_iter()
                .map(|x| x * offset)
                .collect();

            let x_offset_inverses = XFieldElement::batch_inversion(x_offset);
            for i in 0..n / 2 {
                codeword_local[i] = two_inv
                    * ((one + alpha * x_offset_inverses[i]) * codeword_local[i]
                        + (one - alpha * x_offset_inverses[i]) * codeword_local[n / 2 + i]);
            }
            codeword_local.resize(n / 2, zero);

            // Compute and send Merkle root
            codeword_local
                .clone()
                .into_par_iter()
                .map(|xfe| {
                    let digest: Vec<BFieldElement> = xfe.to_digest();
                    hasher.hash(&digest)
                })
                .collect_into_vec(&mut digests);

            mt = MerkleTree::from_digests(&digests, &mt_dummy_value);
            // proof_stream.enqueue_length_prepended(mt.get_root())?;
            let mt_root: &H::Digest = mt.get_root();
            proof_stream.enqueue(&Item::MerkleRoot(mt_root.to_owned()));
            values_and_merkle_trees.push((codeword_local.clone(), mt));

            // Update subgroup generator and offset
            generator = generator * generator;
            offset = offset * offset;
        }

        // Send the last codeword
        let last_codeword: Vec<XFieldElement> = codeword_local;
        // proof_stream.enqueue_length_prepended(&last_codeword)?;
        proof_stream.enqueue(&Item::FriCodeword(last_codeword));

        Ok(values_and_merkle_trees)
    }

    // Return the c-indices for the 1st round of FRI
    fn sample_indices(&self, seed: &H::Digest) -> Vec<usize> {
        let hasher = H::new();

        // This algorithm starts with the inner-most indices to pick up
        // to `last_codeword_length` indices from the codeword in the last round.
        // It then calculates the indices in the subsequent rounds by choosing
        // between the two possible next indices in the next round until we get
        // the c-indices for the first round.
        let num_rounds = self.num_rounds().0;
        let last_codeword_length = self.domain.length >> num_rounds;
        assert!(
            self.colinearity_checks_count <= last_codeword_length,
            "Requested number of indices must not exceed length of last codeword"
        );

        // TODO: FRI's sample_indices can be expressed using simple_hasher's get_n_hash_rounds(),
        // since both use counter mode. But because FRI uses counter mode over two separate loops,
        // enough hash rounds must be made for both loops.
        let _total_hash_rounds = self.colinearity_checks_count * num_rounds as usize;

        let mut last_indices: Vec<usize> = vec![];
        let mut remaining_last_round_exponents: Vec<usize> = (0..last_codeword_length).collect();
        let mut counter = 0u32;
        for _ in 0..self.colinearity_checks_count {
            let digest: H::Digest = hasher.hash_pair(seed, &(counter as u128).to_digest());
            let index: usize =
                hasher.sample_index_not_power_of_two(&digest, remaining_last_round_exponents.len());
            last_indices.push(remaining_last_round_exponents.remove(index));
            counter += 1;
        }

        // Use last indices to derive first c-indices
        let mut indices = last_indices;
        for i in 1..num_rounds {
            let codeword_length = last_codeword_length << i;

            let mut new_indices: Vec<usize> = vec![];
            for index in indices {
                let digest: H::Digest = hasher.hash_pair(seed, &(counter as u128).to_digest());
                let reduce_modulo: bool = hasher.sample_index(&digest, 2) == 0;
                let new_index = if reduce_modulo {
                    index + codeword_length / 2
                } else {
                    index
                };
                new_indices.push(new_index);

                counter += 1;
            }

            indices = new_indices;
        }

        indices
    }

    fn query(
        &self,
        current_values_and_mt: (Vec<XFieldElement>, MerkleTree<Vec<BFieldElement>, H>),
        next_values_and_mt: (Vec<XFieldElement>, MerkleTree<Vec<BFieldElement>, H>),
        c_indices: &[usize],
        proof_stream: &mut StarkProofStream,
    ) -> Result<(), Box<dyn Error>> {
        let a_indices: Vec<usize> = c_indices.to_vec();
        let mut b_indices: Vec<usize> = c_indices
            .iter()
            .map(|x| x + current_values_and_mt.1.get_number_of_leafs() / 2)
            .collect();
        let mut ab_indices = a_indices;
        ab_indices.append(&mut b_indices);

        // Reveal authentication paths
        let current_ap_with_value: Vec<(
            LeaflessPartialAuthenticationPath<H::Digest>,
            XFieldElement,
        )> = current_values_and_mt
            .1
            .get_leafless_multi_proof(&ab_indices)
            .into_iter()
            .zip(ab_indices.iter())
            .map(|(ap, i)| (ap, current_values_and_mt.0[*i]))
            .collect();

        let next_ap: Vec<(LeaflessPartialAuthenticationPath<H::Digest>, XFieldElement)> =
            next_values_and_mt
                .1
                .get_leafless_multi_proof(c_indices)
                .into_iter()
                .zip(c_indices.iter())
                .map(|(ap, i)| (ap, next_values_and_mt.0[*i]))
                .collect();

        proof_stream.enqueue(&Item::FriProof(current_ap_with_value));
        proof_stream.enqueue(&Item::FriProof(next_ap));

        Ok(())
    }

    pub fn verify(
        &self,
        proof_stream: &mut StarkProofStream,
    ) -> Result<Vec<CodewordEvaluation<XFieldElement>>, Box<dyn Error>> {
        let mut omega = self.domain.omega;
        let mut offset = self.domain.offset;
        let (num_rounds, degree_of_last_round) = self.num_rounds();

        // Extract all roots and calculate alpha, the challenges
        let mut roots: Vec<H::Digest> = vec![];
        let mut alphas: Vec<XFieldElement> = vec![];
        // let first_root: H::Digest = proof_stream.dequeue_length_prepended::<H::Digest>()?;
        let first_root: H::Digest = proof_stream.dequeue()?.as_merkle_root()?;
        roots.push(first_root);

        for _ in 0..num_rounds {
            // Get a challenge from the proof stream
            let _alpha: H::Digest = proof_stream.verifier_fiat_shamir();
            let alpha: XFieldElement = XFieldElement::new([_alpha[0], _alpha[1], _alpha[2]]);
            alphas.push(alpha);

            let root: H::Digest = proof_stream.dequeue()?.as_merkle_root()?;
            roots.push(root);
        }

        // Extract last codeword
        let mut last_codeword: Vec<XFieldElement> = proof_stream.dequeue()?.as_fri_codeword()?;

        // Check if last codeword matches the given root
        let zero = omega.ring_zero();
        let last_codeword_mt = MerkleTree::<XFieldElement, H>::from_vec(&last_codeword, &zero);
        let last_root = roots.last().unwrap();
        if last_root != last_codeword_mt.get_root() {
            return Err(Box::new(ValidationError::BadMerkleRootForLastCodeword));
        }

        // Verify that last codeword is of sufficiently low degree
        let mut last_omega = omega;
        let mut last_offset = offset;
        for _ in 0..num_rounds {
            last_omega = last_omega * last_omega;
            last_offset = last_offset * last_offset;
        }

        // Compute interpolant to get the degree of the last codeword
        // Note that we don't have to scale the polynomial back to the
        // trace subgroup since we only check its degree and don't use
        // it further.
        let log_2_of_n = log_2_floor(last_codeword.len() as u64) as u32;
        intt::<XFieldElement>(&mut last_codeword, last_omega, log_2_of_n);
        let last_poly_degree: isize = (Polynomial::<XFieldElement> {
            coefficients: last_codeword,
        })
        .degree();
        if last_poly_degree > degree_of_last_round as isize {
            return Err(Box::new(ValidationError::LastIterationTooHighDegree));
        }

        let seed: H::Digest = proof_stream.verifier_fiat_shamir();
        let top_level_indices: Vec<usize> = self.sample_indices(&seed);

        // for every round, check consistency of subsequent layers
        let mut codeword_evaluations: Vec<CodewordEvaluation<XFieldElement>> = vec![];
        for r in 0..num_rounds as usize {
            // Fold c indices
            let c_indices: Vec<usize> = top_level_indices
                .iter()
                .map(|x| x % (self.domain.length >> (r + 1)))
                .collect();

            // Infer a and b indices
            let a_indices = c_indices.clone();
            let b_indices: Vec<usize> = a_indices
                .iter()
                .map(|x| x + (self.domain.length >> (r + 1)))
                .collect();
            let mut ab_indices: Vec<usize> = a_indices.clone();
            ab_indices.append(&mut b_indices.clone());

            // Read values and check colinearity
            let ab_proof: Vec<(LeaflessPartialAuthenticationPath<H::Digest>, XFieldElement)> =
                proof_stream.dequeue()?.as_fri_proof()?;
            let c_proof: Vec<(LeaflessPartialAuthenticationPath<H::Digest>, XFieldElement)> =
                proof_stream.dequeue()?.as_fri_proof()?;

            // verify Merkle authentication paths
            if !MerkleTree::<XFieldElement, H>::verify_leafless_multi_proof(
                roots[r].clone(),
                &ab_indices,
                &ab_proof,
            ) {
                return Err(Box::new(ValidationError::BadMerkleProof));
            }

            if !MerkleTree::<XFieldElement, H>::verify_leafless_multi_proof(
                roots[r + 1].clone(),
                &c_indices,
                &c_proof,
            ) {
                return Err(Box::new(ValidationError::BadMerkleProof));
            }

            // Verify that the expected number of samples are present
            if ab_proof.len() != 2 * self.colinearity_checks_count
                || c_proof.len() != self.colinearity_checks_count
            {
                return Err(Box::new(ValidationError::BadSizedProof));
            }

            // Colinearity check
            let axs: Vec<XFieldElement> = (0..self.colinearity_checks_count)
                .map(|i| offset * omega.mod_pow_u32(a_indices[i] as u32))
                .collect();
            let bxs: Vec<XFieldElement> = (0..self.colinearity_checks_count)
                .map(|i| offset * omega.mod_pow_u32(b_indices[i] as u32))
                .collect();
            let cx: XFieldElement = alphas[r];
            let ays: Vec<XFieldElement> = (0..self.colinearity_checks_count)
                .map(|i| ab_proof[i].1)
                .collect();
            let bys: Vec<XFieldElement> = (0..self.colinearity_checks_count)
                .map(|i| ab_proof[i + self.colinearity_checks_count].1)
                .collect();
            let cys: Vec<XFieldElement> = (0..self.colinearity_checks_count)
                .map(|i| c_proof[i].1)
                .collect();

            if (0..self.colinearity_checks_count).any(|i| {
                !Polynomial::<XFieldElement>::are_colinear_3(
                    (axs[i], ays[i]),
                    (bxs[i], bys[i]),
                    (cx, cys[i]),
                )
            }) {
                return Err(Box::new(ValidationError::NotColinear(r)));
            }
            // Update subgroup generator and offset
            omega = omega * omega;
            offset = offset * offset;

            // Return top-level values to caller
            if r == 0 {
                for s in 0..self.colinearity_checks_count {
                    codeword_evaluations.push((a_indices[s], ays[s]));
                    codeword_evaluations.push((b_indices[s], bys[s]));
                }
            }
        }

        Ok(codeword_evaluations)
    }

    pub fn get_evaluation_domain(&self) -> Vec<XFieldElement> {
        let omega_domain = self.domain.omega.get_cyclic_group_elements(None);
        omega_domain
            .into_iter()
            .map(|x| x * self.domain.offset)
            .collect()
    }

    fn num_rounds(&self) -> (u8, u32) {
        let max_degree = (self.domain.length / self.expansion_factor) - 1;
        let mut rounds_count = log_2_ceil(max_degree as u64 + 1) as u8;
        let mut max_degree_of_last_round = 0u32;
        if self.expansion_factor < self.colinearity_checks_count {
            let num_missed_rounds = log_2_ceil(
                (self.colinearity_checks_count as f64 / self.expansion_factor as f64).ceil() as u64,
            ) as u8;
            rounds_count -= num_missed_rounds;
            max_degree_of_last_round = 2u32.pow(num_missed_rounds as u32) - 1;
        }

        (rounds_count, max_degree_of_last_round)
    }
}

#[cfg(test)]
mod fri_domain_tests {
    use super::*;
    use crate::shared_math::{
        b_field_element::BFieldElement, traits::GetPrimitiveRootOfUnity,
        x_field_element::XFieldElement,
    };

    #[test]
    fn x_values_test() {
        // pol = x^3
        let x_squared_coefficients = vec![
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_one(),
        ];

        for order in [4, 8, 32] {
            let omega = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();
            let domain = FriDomain {
                offset: BFieldElement::generator().lift(),
                omega: omega.lift(),
                length: order as usize,
            };
            let expected_x_values: Vec<BFieldElement> = (0..order)
                .map(|i| BFieldElement::generator() * omega.mod_pow(i as u64))
                .collect();
            let x_values = domain.b_domain_values();
            assert_eq!(expected_x_values, x_values);

            // Verify that `x_value` also returns expected values
            for i in 0..order {
                assert_eq!(
                    expected_x_values[i as usize],
                    domain.b_domain_value(i as u32)
                );
            }

            let pol = Polynomial::<BFieldElement>::new(x_squared_coefficients.clone());
            let values = domain.b_evaluate(&pol, BFieldElement::ring_zero());
            assert_ne!(values, x_squared_coefficients);
            let interpolant = domain.b_interpolate(&values);
            assert_eq!(pol, interpolant);

            // Verify that batch-evaluated values match a manual evaluation
            for i in 0..order {
                assert_eq!(
                    pol.evaluate(&domain.b_domain_value(i as u32)),
                    values[i as usize]
                );
            }

            let x_squared_coefficients_lifted: Vec<XFieldElement> = x_squared_coefficients
                .clone()
                .into_iter()
                .map(|x| x.lift())
                .collect();
            let xpol = Polynomial::new(x_squared_coefficients_lifted.clone());
            let x_field_x_values = domain.x_evaluate(&xpol);
            assert_ne!(x_field_x_values, x_squared_coefficients_lifted);
            let x_interpolant = domain.x_interpolate(&x_field_x_values);
            assert_eq!(xpol, x_interpolant);
        }
    }
}

#[cfg(test)]
mod xfri_tests {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::traits::GetPrimitiveRootOfUnity;
    use crate::shared_math::traits::{CyclicGroupGenerator, ModPowU32};
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::util_types::simple_hasher::{RescuePrimeProduction, ToDigest};
    use itertools::Itertools;

    #[test]
    fn get_rounds_count_test() {
        type Hasher = RescuePrimeProduction;

        let subgroup_order = 512;
        let expansion_factor = 4;
        let mut fri: Fri<Hasher> =
            get_x_field_fri_test_object::<Hasher>(subgroup_order, expansion_factor, 2);

        assert_eq!((7, 0), fri.num_rounds());
        fri.colinearity_checks_count = 8;
        assert_eq!((6, 1), fri.num_rounds());
        fri.colinearity_checks_count = 10;
        assert_eq!((5, 3), fri.num_rounds());
        fri.colinearity_checks_count = 16;
        assert_eq!((5, 3), fri.num_rounds());
        fri.colinearity_checks_count = 17;
        assert_eq!((4, 7), fri.num_rounds());
        fri.colinearity_checks_count = 18;
        assert_eq!((4, 7), fri.num_rounds());
        fri.colinearity_checks_count = 31;
        assert_eq!((4, 7), fri.num_rounds());
        fri.colinearity_checks_count = 32;
        assert_eq!((4, 7), fri.num_rounds());
        fri.colinearity_checks_count = 33;
        assert_eq!((3, 15), fri.num_rounds());

        fri.domain.length = 256;
        assert_eq!((2, 15), fri.num_rounds());
        fri.colinearity_checks_count = 32;
        assert_eq!((3, 7), fri.num_rounds());

        fri.colinearity_checks_count = 32;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((15, 3), fri.num_rounds());

        fri.colinearity_checks_count = 33;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((14, 7), fri.num_rounds());

        fri.colinearity_checks_count = 63;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((14, 7), fri.num_rounds());

        fri.colinearity_checks_count = 64;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((14, 7), fri.num_rounds());

        fri.colinearity_checks_count = 65;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((13, 15), fri.num_rounds());

        fri.domain.length = 256;
        fri.expansion_factor = 4;
        fri.colinearity_checks_count = 17;
        assert_eq!((3, 7), fri.num_rounds());
    }

    #[test]
    fn fri_on_x_field_test() {
        type Hasher = RescuePrimeProduction;

        let subgroup_order = 1024;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<Hasher> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let mut proof_stream: StarkProofStream = StarkProofStream::default();
        let subgroup = fri.domain.omega.get_cyclic_group_elements(None);

        fri.prove(&subgroup, &mut proof_stream).unwrap();
        let verify_result = fri.verify(&mut proof_stream);
        assert!(verify_result.is_ok());
    }

    #[test]
    fn fri_x_field_limit_test() {
        type Hasher = RescuePrimeProduction;

        let subgroup_order = 128;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<Hasher> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let subgroup = fri.domain.omega.get_cyclic_group_elements(None);

        let mut points: Vec<XFieldElement>;
        for n in [1, 5, 20, 30, 31] {
            points = subgroup.clone().iter().map(|p| p.mod_pow_u32(n)).collect();

            // TODO: Test elsewhere that proof_stream can be re-used for multiple .prove().
            let mut proof_stream: StarkProofStream = StarkProofStream::default();
            fri.prove(&points, &mut proof_stream).unwrap();

            let verify_result = fri.verify(&mut proof_stream);
            if verify_result.is_err() {
                println!(
                    "There are {} points, |<128>^{}| = {}, and verify_result = {:?}",
                    points.len(),
                    n,
                    points.iter().unique().count(),
                    verify_result
                );
            }

            assert!(verify_result.is_ok());

            // TODO: Add negative test with bad Merkle authentication path
            // This probably requires manipulating the proof stream somehow.
        }

        // Negative test with too high degree
        let too_high = subgroup_order as u32 / expansion_factor as u32;
        points = subgroup.iter().map(|p| p.mod_pow_u32(too_high)).collect();
        let mut proof_stream: StarkProofStream = StarkProofStream::default();
        fri.prove(&points, &mut proof_stream).unwrap();
        let verify_result = fri.verify(&mut proof_stream);
        assert!(verify_result.is_err());
    }

    fn get_x_field_fri_test_object<H>(
        subgroup_order: u128,
        expansion_factor: usize,
        colinearity_checks: usize,
    ) -> Fri<H>
    where
        H: Hasher<Digest = Vec<BFieldElement>> + Sized + std::marker::Sync,
        XFieldElement: ToDigest<H::Digest>,
    {
        let (omega, _primes1): (Option<XFieldElement>, Vec<u128>) =
            XFieldElement::ring_zero().get_primitive_root_of_unity(subgroup_order);

        // The following offset was picked arbitrarily by copying the one found in
        // `get_b_field_fri_test_object`. It does not generate the full Z_p\{0}, but
        // we're not sure it needs to, Alan?
        let offset: Option<XFieldElement> = Some(XFieldElement::new_const(BFieldElement::new(7)));

        let fri: Fri<H> = Fri::new(
            offset.unwrap(),
            omega.unwrap(),
            subgroup_order as usize,
            expansion_factor,
            colinearity_checks,
        );
        fri
    }
}
