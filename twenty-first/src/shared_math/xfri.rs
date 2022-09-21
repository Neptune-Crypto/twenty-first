use itertools::Itertools;
use num_traits::{One, Zero};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use super::b_field_element::BFieldElement;
use super::other::{log_2_ceil, log_2_floor};
use super::polynomial::Polynomial;
use super::stark::brainfuck::stark_proof_stream::*;
use super::traits::{CyclicGroupGenerator, ModPowU32};
use super::x_field_element::XFieldElement;
use crate::shared_math::ntt::{intt, ntt};
use crate::shared_math::traits::FiniteField;
use crate::timing_reporter::TimingReporter;
use crate::util_types::merkle_tree::{MerkleTree, PartialAuthenticationPath};
use crate::util_types::simple_hasher::{Hashable, Hasher, SamplableFrom};
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
    MismatchingLastCodeword,
    LastIterationTooHighDegree,
    BadMerkleRootForFirstCodeword,
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

    pub fn b_evaluate(&self, polynomial: &Polynomial<BFieldElement>) -> Vec<BFieldElement> {
        let zero = BFieldElement::zero();
        let mut polynomial_representation: Vec<BFieldElement> = polynomial
            .scale(&self.offset.unlift().unwrap())
            .coefficients;
        polynomial_representation.resize(self.length as usize, zero);
        ntt(
            &mut polynomial_representation,
            self.omega.unlift().unwrap(),
            log_2_ceil(self.length as u128) as u32,
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
    // In STARK, the expansion factor <FRI domain length> / max_degree, where
    // `max_degree` is the max degree of any interpolation rounded up to the
    // nearest power of 2.
    pub expansion_factor: usize,
    pub colinearity_checks_count: usize,
    pub domain: FriDomain,
    _hasher: PhantomData<H>,
}

impl<H> Fri<H>
where
    H: Hasher + std::marker::Sync,
    BFieldElement: Hashable<H::T>,
    usize: Hashable<H::T>,
    XFieldElement: SamplableFrom<H::Digest> + Hashable<H::T>,
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

    /// Build the (deduplicated) Merkle authentication paths for the codeword at the given indices
    /// and enqueue the corresponding values and (partial) authentication paths on the proof stream.
    fn enqueue_auth_pairs(
        indices: &[usize],
        codeword: &[XFieldElement],
        merkle_tree: &MerkleTree<H>,
        proof_stream: &mut StarkProofStream<H>,
    ) {
        let value_ap_pairs: Vec<(PartialAuthenticationPath<H::Digest>, XFieldElement)> =
            merkle_tree
                .get_authentication_structure(indices)
                .into_iter()
                .zip(indices.iter())
                .map(|(ap, i)| (ap, codeword[*i]))
                .collect_vec();
        proof_stream.enqueue(&ProofItem::FriProof(value_ap_pairs))
    }

    /// Given a set of `indices`, a merkle `root`, and the (correctly set) `proof_stream`, verify
    /// whether the values at the `indices` are members of the set committed to by the merkle `root`
    /// and return these values if they are. Fails otherwise.
    fn dequeue_and_authenticate(
        indices: &[usize],
        root: H::Digest,
        proof_stream: &mut StarkProofStream<H>,
    ) -> Result<Vec<XFieldElement>, Box<dyn Error>> {
        let hasher = H::new();
        let (paths, values): (
            Vec<PartialAuthenticationPath<H::Digest>>,
            Vec<XFieldElement>,
        ) = proof_stream.dequeue()?.as_fri_proof()?.into_iter().unzip();
        let digests: Vec<H::Digest> = values
            .par_iter()
            .map(|v| hasher.hash_sequence(&v.to_sequence()))
            .collect();
        let path_digest_pairs = paths.into_iter().zip(digests).collect_vec();
        if MerkleTree::<H>::verify_authentication_structure(root, indices, &path_digest_pairs) {
            Ok(values)
        } else {
            Err(Box::new(ValidationError::BadMerkleProof))
        }
    }

    /// Create a FRI proof and return chosen indices of round 0 and Merkle root of round 0 codeword
    pub fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut StarkProofStream<H>,
    ) -> Result<(Vec<usize>, H::Digest), Box<dyn Error>> {
        debug_assert_eq!(
            self.domain.length,
            codeword.len(),
            "Initial codeword length must match that set in FRI object"
        );

        // Commit phase
        let mut timer = TimingReporter::start();
        let (codewords, merkle_trees): (Vec<Vec<XFieldElement>>, Vec<MerkleTree<H>>) =
            self.commit(codeword, proof_stream)?.into_iter().unzip();
        timer.elapsed("Commit phase");

        // fiat-shamir phase (get indices)
        let top_level_indices: Vec<usize> =
            self.sample_indices(&proof_stream.prover_fiat_shamir().to_sequence());
        timer.elapsed("Sample indices");

        // Query phase
        // query step 0: enqueue authentication paths for all points `A` into proof stream
        let initial_a_indices: Vec<usize> = top_level_indices.clone();
        Self::enqueue_auth_pairs(&initial_a_indices, codeword, &merkle_trees[0], proof_stream);
        // query step 1: loop over FRI rounds, enqueue authentication paths for all points `B`
        let mut current_domain_len = self.domain.length;
        let mut b_indices: Vec<usize> = initial_a_indices;
        // the last codeword is transmitted to the verifier in the clear. Thus, no co-linearity
        // check is needed for the last codeword and we only have to look at the interval given here
        for r in 0..merkle_trees.len() - 1 {
            debug_assert_eq!(
                codewords[r].len(),
                current_domain_len,
                "The current domain length needs to be the same as the length of the \
                current codeword"
            );
            b_indices = b_indices
                .iter()
                .map(|x| (x + current_domain_len / 2) % current_domain_len)
                .collect();
            Self::enqueue_auth_pairs(&b_indices, &codewords[r], &merkle_trees[r], proof_stream);
            current_domain_len /= 2;
            timer.elapsed(&format!("Query phase {}", r));
        }

        println!("FRI-prover, timing report\n{}", timer.finish());

        let merkle_root_of_1st_round: H::Digest = merkle_trees[0].get_root();
        Ok((top_level_indices, merkle_root_of_1st_round))
    }

    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut StarkProofStream<H>,
    ) -> Result<Vec<(Vec<XFieldElement>, MerkleTree<H>)>, Box<dyn Error>> {
        let mut subgroup_generator = self.domain.omega;
        let mut offset = self.domain.offset;
        let mut codeword_local = codeword.to_vec();

        let one: XFieldElement = XFieldElement::one();
        let two: XFieldElement = one + one;
        let two_inv = one / two;

        // Compute and send Merkle root
        let hasher = H::new();
        let mut digests: Vec<H::Digest> = Vec::with_capacity(codeword_local.len());
        codeword_local
            .clone()
            .into_par_iter()
            .map(|xfe| {
                let b_elements: Vec<BFieldElement> = xfe.coefficients.into();
                hasher.hash_sequence(
                    &b_elements
                        .iter()
                        .flat_map(|b| b.to_sequence())
                        .collect_vec(),
                )
            })
            .collect_into_vec(&mut digests);
        let mut mt: MerkleTree<H> = MerkleTree::from_digests(&digests);
        let mut mt_root: <H as Hasher>::Digest = mt.get_root();

        proof_stream.enqueue(&ProofItem::MerkleRoot(mt_root));
        let mut values_and_merkle_trees = vec![(codeword_local.clone(), mt)];

        let (num_rounds, _) = self.num_rounds();
        for _round in 0..num_rounds {
            let n = codeword_local.len();

            // Get challenge
            let alpha = XFieldElement::sample(&proof_stream.prover_fiat_shamir());

            let x_offset: Vec<XFieldElement> = subgroup_generator
                .get_cyclic_group_elements(None)
                .into_par_iter()
                .map(|x| x * offset)
                .collect();

            let x_offset_inverses = XFieldElement::batch_inversion(x_offset);
            codeword_local = (0..n / 2)
                .into_par_iter()
                .map(|i| {
                    two_inv
                        * ((one + alpha * x_offset_inverses[i]) * codeword_local[i]
                            + (one - alpha * x_offset_inverses[i]) * codeword_local[n / 2 + i])
                })
                .collect();

            // Compute and send Merkle root. We have to do that within this loops, since
            // the next round's alpha must be calculated from the previous round's Merkle root.
            codeword_local
                .clone()
                .into_par_iter()
                .map(|xfe| {
                    let b_elements: Vec<BFieldElement> = xfe.coefficients.into();
                    hasher.hash_sequence(
                        &b_elements
                            .iter()
                            .flat_map(|b| b.to_sequence())
                            .collect_vec(),
                    )
                })
                .collect_into_vec(&mut digests);

            mt = MerkleTree::from_digests(&digests);
            mt_root = mt.get_root();
            proof_stream.enqueue(&ProofItem::MerkleRoot(mt_root));
            values_and_merkle_trees.push((codeword_local.clone(), mt));

            // Update subgroup generator and offset
            subgroup_generator = subgroup_generator * subgroup_generator;
            offset = offset * offset;
        }

        // Send the last codeword
        // todo! use coefficient form for last codeword?
        let last_codeword: Vec<XFieldElement> = codeword_local;
        proof_stream.enqueue(&ProofItem::FriCodeword(last_codeword));

        Ok(values_and_merkle_trees)
    }

    // Return the c-indices for the 1st round of FRI
    fn sample_indices(&self, seed: &[H::T]) -> Vec<usize> {
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

        let mut last_indices: Vec<usize> = vec![];
        let mut remaining_last_round_exponents: Vec<usize> = (0..last_codeword_length).collect();
        let mut counter = 0usize;
        for _ in 0..self.colinearity_checks_count {
            let digest: H::Digest =
                hasher.hash_sequence(&[seed, &(counter).to_sequence()].concat());
            let index: usize =
                hasher.sample_index_not_power_of_two(&digest, remaining_last_round_exponents.len());
            last_indices.push(remaining_last_round_exponents.remove(index));
            counter += 1;
        }

        // Use last indices to derive first c-indices
        let mut indices = last_indices;
        for i in 1..num_rounds {
            let codeword_length = last_codeword_length << i;

            indices = indices
                .into_iter()
                .zip((counter..counter + self.colinearity_checks_count as usize).into_iter())
                .map(|(index, count)| {
                    let digest: H::Digest =
                        hasher.hash_sequence(&[seed, &(count as usize).to_sequence()].concat());
                    let reduce_modulo: bool = hasher.sample_index(&digest, 2) == 0;
                    if reduce_modulo {
                        index + codeword_length / 2
                    } else {
                        index
                    }
                })
                .collect();
        }

        indices
    }

    pub fn verify(
        &self,
        proof_stream: &mut StarkProofStream<H>,
        first_codeword_mt_root: &H::Digest,
    ) -> Result<(), Box<dyn Error>> {
        let hasher = H::new();

        let (num_rounds, degree_of_last_round) = self.num_rounds();
        let num_rounds = num_rounds as usize;
        let mut timer = TimingReporter::start();

        // Extract all roots and calculate alpha, the challenges
        let mut roots: Vec<H::Digest> = vec![];
        let mut alphas: Vec<XFieldElement> = vec![];

        let first_root: H::Digest = proof_stream.dequeue()?.as_merkle_root()?;
        if first_root != *first_codeword_mt_root {
            return Err(Box::new(ValidationError::BadMerkleRootForFirstCodeword));
        }

        roots.push(first_root);
        timer.elapsed("Init");

        for _round in 0..num_rounds {
            // Get a challenge from the proof stream
            let alpha = XFieldElement::sample(&proof_stream.verifier_fiat_shamir());
            alphas.push(alpha);

            let root: H::Digest = proof_stream.dequeue()?.as_merkle_root()?;
            roots.push(root);
        }
        timer.elapsed("Roots and alpha");

        // Extract last codeword
        let last_codeword: Vec<XFieldElement> = proof_stream.dequeue()?.as_fri_codeword()?;

        // Check if last codeword matches the given root
        let codeword_digests = last_codeword
            .iter()
            .map(|l| hasher.hash_sequence(&l.to_sequence()))
            .collect_vec();
        let last_codeword_mt = MerkleTree::<H>::from_digests(&codeword_digests);
        let last_root = roots.last().unwrap();
        if *last_root != last_codeword_mt.get_root() {
            return Err(Box::new(ValidationError::BadMerkleRootForLastCodeword));
        }

        // Verify that last codeword is of sufficiently low degree

        // Compute interpolant to get the degree of the last codeword
        // Note that we don't have to scale the polynomial back to the
        // trace subgroup since we only check its degree and don't use
        // it further.
        let log_2_of_n = log_2_floor(last_codeword.len() as u128) as u32;
        let mut last_polynomial = last_codeword.clone();
        let last_omega = self.domain.omega.mod_pow_u32(2u32.pow(num_rounds as u32));
        intt::<XFieldElement>(&mut last_polynomial, last_omega, log_2_of_n);
        let last_poly_degree: isize = (Polynomial::<XFieldElement> {
            coefficients: last_polynomial,
        })
        .degree();
        if last_poly_degree > degree_of_last_round as isize {
            return Err(Box::new(ValidationError::LastIterationTooHighDegree));
        }
        timer.elapsed("Verified last round");

        // Query phase
        // query step 0: get "A" indices and verify set membership of corresponding values.
        let mut a_indices: Vec<usize> =
            self.sample_indices(&proof_stream.verifier_fiat_shamir().to_sequence());
        timer.elapsed("Sample indices");
        let mut a_values =
            Self::dequeue_and_authenticate(&a_indices, roots[0].clone(), proof_stream)?;

        // set up "B" for offsetting inside loop.  Note that "B" and "A" indices
        // can be calcuated from each other.
        let mut b_indices = a_indices.clone();
        let mut current_domain_len = self.domain.length;

        // query step 1:  loop over FRI rounds, verify "B"s, compute values for "C"s
        timer.elapsed("Starting query phase step 1 (loop)");
        for r in 0..num_rounds {
            timer.elapsed(&format!("Round {} started.", r));

            // get "B" indices and verify set membership of corresponding values
            b_indices = b_indices
                .iter()
                .map(|x| (x + current_domain_len / 2) % current_domain_len)
                .collect();
            timer.elapsed(&format!("Get b-indices for current round ({})", r));
            let b_values =
                Self::dequeue_and_authenticate(&b_indices, roots[r].clone(), proof_stream)?;
            timer.elapsed(&format!("Read & verify b-auth paths round {}", r));
            debug_assert_eq!(
                self.colinearity_checks_count,
                a_indices.len(),
                "There must be equally many 'a indices' as there are colinearity checks."
            );
            debug_assert_eq!(
                self.colinearity_checks_count,
                b_indices.len(),
                "There must be equally many 'b indices' as there are colinearity checks."
            );
            debug_assert_eq!(
                self.colinearity_checks_count,
                a_values.len(),
                "There must be equally many 'a values' as there are colinearity checks."
            );
            debug_assert_eq!(
                self.colinearity_checks_count,
                b_values.len(),
                "There must be equally many 'b values' as there are colinearity checks."
            );

            // compute "C" indices and values for next round from "A" and "B`"" of current round
            current_domain_len /= 2;
            let c_indices = a_indices.iter().map(|x| x % current_domain_len).collect();
            timer.elapsed(&format!(
                "Got c-indices for current round equal to a-indices for next round ({})",
                r + 1
            ));
            let c_values = (0..self.colinearity_checks_count)
                .into_par_iter()
                .map(|i| {
                    Polynomial::<XFieldElement>::get_colinear_y(
                        (self.get_evaluation_argument(a_indices[i], r), a_values[i]),
                        (self.get_evaluation_argument(b_indices[i], r), b_values[i]),
                        alphas[r],
                    )
                })
                .collect();
            timer.elapsed(&format!("Computed colinear c-values for current round equal to a-values for next round ({})", r + 1 ));

            // Notice that next rounds "A"s correspond to current rounds "C":
            a_indices = c_indices;
            a_values = c_values;

            timer.elapsed(&format!("Round {} finished.", r));
        }
        timer.elapsed("Stopping query phase step 1 (loop)");

        // Finally compare "C" values (which are named "A" values in this
        // enclosing scope) with last codeword from the proofstream.
        a_indices = a_indices.iter().map(|x| x % current_domain_len).collect();
        if (0..self.colinearity_checks_count).any(|i| last_codeword[a_indices[i]] != a_values[i]) {
            return Err(Box::new(ValidationError::MismatchingLastCodeword));
        }

        timer.elapsed("LastCodeword comparison");
        println!("FRI-verifier Timing Report\n{}", timer.finish());
        Ok(())
    }

    /// Given index `i` of the FRI codeword in round `round`, compute the corresponding value in the
    /// FRI (co-)domain. This corresponds to `ω^i` in `f(ω^i)` from
    /// [STARK-Anatomy](https://neptune.cash/learn/stark-anatomy/fri/#split-and-fold).
    fn get_evaluation_argument(&self, idx: usize, round: usize) -> XFieldElement {
        (self.domain.offset * self.domain.omega.mod_pow_u32(idx as u32))
            .mod_pow_u32(2u32.pow(round as u32))
    }

    fn num_rounds(&self) -> (u8, u32) {
        let max_degree = (self.domain.length / self.expansion_factor) - 1;
        let mut rounds_count = log_2_ceil(max_degree as u128 + 1) as u8;
        let mut max_degree_of_last_round = 0u32;
        if self.expansion_factor < self.colinearity_checks_count {
            let num_missed_rounds = log_2_ceil(
                (self.colinearity_checks_count as f64 / self.expansion_factor as f64).ceil()
                    as u128,
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
        b_field_element::BFieldElement, traits::PrimitiveRootOfUnity,
        x_field_element::XFieldElement,
    };

    #[test]
    fn x_values_test() {
        // pol = x^3
        let x_squared_coefficients = vec![
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::one(),
        ];

        for order in [4, 8, 32] {
            let omega = BFieldElement::primitive_root_of_unity(order).unwrap();
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
            let values = domain.b_evaluate(&pol);
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
    use crate::shared_math::rescue_prime_regular::RescuePrimeRegular;
    use crate::shared_math::traits::PrimitiveRootOfUnity;
    use crate::shared_math::traits::{CyclicGroupGenerator, ModPowU32};
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::utils::has_unique_elements;
    use itertools::Itertools;
    use rand::{thread_rng, RngCore};

    #[test]
    fn sample_indices_test() {
        type Hasher = RescuePrimeRegular;

        let hasher = RescuePrimeRegular::new();
        let mut rng = thread_rng();
        let subgroup_order = 16;
        let expansion_factor = 4;
        let colinearity_checks = 16;
        let fri: Fri<Hasher> = get_x_field_fri_test_object::<Hasher>(
            subgroup_order,
            expansion_factor,
            colinearity_checks,
        );
        let indices = fri.sample_indices(
            &hasher.hash_sequence(&hasher.hash_sequence(&vec![BFieldElement::new(rng.next_u64())])),
        );
        assert!(
            has_unique_elements(indices.iter()),
            "Picked indices must be unique"
        );
    }

    #[test]
    fn get_rounds_count_test() {
        type Hasher = RescuePrimeRegular;

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
        type Hasher = RescuePrimeRegular;

        let subgroup_order = 1024;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<Hasher> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let mut proof_stream: StarkProofStream<Hasher> = StarkProofStream::default();
        let subgroup = fri.domain.omega.get_cyclic_group_elements(None);

        let (_, merkle_root_of_round_0) = fri.prove(&subgroup, &mut proof_stream).unwrap();
        let verdict = fri.verify(&mut proof_stream, &merkle_root_of_round_0);
        if let Err(e) = verdict {
            panic!("Found error: {}", e);
        }
    }

    #[test]
    fn prove_and_verify_low_degree_of_twice_cubing_plus_one() {
        type Hasher = RescuePrimeRegular;

        let subgroup_order = 1024;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<Hasher> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let mut proof_stream: StarkProofStream<Hasher> = StarkProofStream::default();

        let zero = XFieldElement::zero();
        let one = XFieldElement::one();
        let two = one + one;
        let poly = Polynomial::<XFieldElement>::new(vec![one, zero, zero, two]);
        let codeword = fri.domain.x_evaluate(&poly);

        let (_, merkle_root_of_round_0) = fri.prove(&codeword, &mut proof_stream).unwrap();
        let verdict = fri.verify(&mut proof_stream, &merkle_root_of_round_0);
        if let Err(e) = verdict {
            panic!("Found error: {}", e);
        }
    }

    #[test]
    fn fri_x_field_limit_test() {
        type Hasher = RescuePrimeRegular;

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
            let mut proof_stream: StarkProofStream<Hasher> = StarkProofStream::default();
            let (_, mut merkle_root_of_round_0) = fri.prove(&points, &mut proof_stream).unwrap();

            let verify_result = fri.verify(&mut proof_stream, &merkle_root_of_round_0);
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

            // Manipulate Merkle root of 0 and verify failure with expected error message
            proof_stream.reset_for_verifier();
            merkle_root_of_round_0[0].increment();
            let bad_verify_result = fri.verify(&mut proof_stream, &merkle_root_of_round_0);
            assert!(bad_verify_result.is_err());
            println!("bad_verify_result = {:?}", bad_verify_result);

            // TODO: Add negative test with bad Merkle authentication path
            // This probably requires manipulating the proof stream somehow.
        }

        // Negative test with too high degree
        let too_high = subgroup_order as u32 / expansion_factor as u32;
        points = subgroup.iter().map(|p| p.mod_pow_u32(too_high)).collect();
        let mut proof_stream: StarkProofStream<Hasher> = StarkProofStream::default();
        let (_, merkle_root_of_round_0) = fri.prove(&points, &mut proof_stream).unwrap();
        let verify_result = fri.verify(&mut proof_stream, &merkle_root_of_round_0);
        assert!(verify_result.is_err());
    }

    fn get_x_field_fri_test_object<H>(
        subgroup_order: u64,
        expansion_factor: usize,
        colinearity_checks: usize,
    ) -> Fri<H>
    where
        H: Hasher + Sized + std::marker::Sync,
        XFieldElement: Hashable<H::T> + SamplableFrom<H::Digest>,
        BFieldElement: Hashable<H::T>,
        usize: Hashable<H::T>,
    {
        let maybe_omega: Option<XFieldElement> =
            XFieldElement::primitive_root_of_unity(subgroup_order);

        // The element 7 generates all of Zp\{0}
        let offset: Option<XFieldElement> = Some(XFieldElement::new_const(BFieldElement::new(7)));

        let fri: Fri<H> = Fri::<H>::new(
            offset.unwrap(),
            maybe_omega.unwrap(),
            subgroup_order as usize,
            expansion_factor,
            colinearity_checks,
        );
        fri
    }
}
