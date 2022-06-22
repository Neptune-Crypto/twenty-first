use super::super::triton;
use super::table::base_matrix::BaseMatrices;
use super::vm::Program;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::other::roundup_npo2;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::rescue_prime_xlix::{
    neptune_params, RescuePrimeXlix, RP_DEFAULT_OUTPUT_SIZE, RP_DEFAULT_WIDTH,
};
use crate::shared_math::stark::brainfuck::stark_proof_stream::{Item, StarkProofStream};
use crate::shared_math::stark::triton::arguments::permutation_argument::PermArg;
use crate::shared_math::stark::triton::instruction::sample_programs;
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::stark::triton::table::challenges_endpoints::{AllChallenges, AllEndpoints};
use crate::shared_math::stark::triton::table::table_collection::{
    BaseTableCollection, ExtTableCollection,
};
use crate::shared_math::traits::{GetPrimitiveRootOfUnity, GetRandomElements, ModPowU32};
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::{other, xfri};
use crate::timing_reporter::TimingReporter;
use crate::util_types::merkle_tree::{MerkleTree, PartialAuthenticationPath};
use crate::util_types::simple_hasher::{Hasher, ToDigest};
use itertools::Itertools;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

type BWord = BFieldElement;
type XWord = XFieldElement;
type StarkHasher = RescuePrimeXlix<RP_DEFAULT_WIDTH>;
type StarkDigest = Vec<BFieldElement>;

// We use a type-parameterised FriDomain to avoid duplicate `b_*()` and `x_*()` methods.
pub struct Stark {
    _padded_height: usize,
    _log_expansion_factor: usize,
    security_level: usize,
    bfri_domain: triton::fri_domain::FriDomain<BWord>,
    xfri_domain: triton::fri_domain::FriDomain<XWord>,
    fri: xfri::Fri<StarkHasher>,
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

        let bfri_domain = triton::fri_domain::FriDomain {
            offset,
            omega,
            length: fri_domain_length as usize,
        };

        let dummy_xfri_domain = triton::fri_domain::FriDomain::<XFieldElement> {
            offset: offset.lift(),
            omega: omega.lift(),
            length: fri_domain_length as usize,
        };

        let dummy_xfri = xfri::Fri::new(
            offset.lift(),
            omega.lift(),
            fri_domain_length,
            expansion_factor as usize,
            colinearity_checks,
        );

        Stark {
            _padded_height,
            _log_expansion_factor: log_expansion_factor,
            security_level,
            bfri_domain,
            xfri_domain: dummy_xfri_domain,
            fri: dummy_xfri,
        }
    }

    pub fn prove(&self, base_matrices: BaseMatrices) -> StarkProofStream {
        let mut timer = TimingReporter::start();

        let num_randomizers = 1;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();
        let unpadded_height = base_matrices.processor_matrix.len();
        let _padded_height = roundup_npo2(unpadded_height as u64);

        // 1. Create base tables based on base matrices

        let mut base_tables = BaseTableCollection::from_base_matrices(
            smooth_generator,
            order,
            num_randomizers,
            &base_matrices,
        );

        timer.elapsed("assert, set_matrices");

        base_tables.pad();

        timer.elapsed("pad");

        let max_degree = base_tables.max_degree();

        // Randomizer bla bla
        let mut rng = rand::thread_rng();
        let randomizer_coefficients =
            XFieldElement::random_elements(max_degree as usize + 1, &mut rng);
        let randomizer_polynomial = Polynomial::new(randomizer_coefficients);

        let x_randomizer_codeword: Vec<XFieldElement> =
            self.fri.domain.x_evaluate(&randomizer_polynomial);
        let mut b_randomizer_codewords: [Vec<BFieldElement>; 3] = [vec![], vec![], vec![]];
        for x_elem in x_randomizer_codeword.iter() {
            b_randomizer_codewords[0].push(x_elem.coefficients[0]);
            b_randomizer_codewords[1].push(x_elem.coefficients[1]);
            b_randomizer_codewords[2].push(x_elem.coefficients[2]);
        }

        timer.elapsed("randomizer_codewords");

        let base_codewords: Vec<Vec<BFieldElement>> =
            base_tables.all_base_codewords(&self.bfri_domain);

        let all_base_codewords =
            vec![b_randomizer_codewords.into(), base_codewords.clone()].concat();

        timer.elapsed("get_and_set_all_base_codewords");

        let transposed_base_codewords: Vec<Vec<BFieldElement>> = (0..all_base_codewords[0].len())
            .map(|i| {
                all_base_codewords
                    .iter()
                    .map(|inner| inner[i])
                    .collect::<Vec<BFieldElement>>()
            })
            .collect();

        timer.elapsed("transposed_base_codewords");

        let hasher = neptune_params();

        let mut base_codeword_digests_by_index: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(transposed_base_codewords.len());

        transposed_base_codewords
            .par_iter()
            .map(|values| hasher.hash(values, DIGEST_LEN))
            .collect_into_vec(&mut base_codeword_digests_by_index);

        let base_merkle_tree =
            MerkleTree::<StarkHasher>::from_digests(&base_codeword_digests_by_index);

        timer.elapsed("base_merkle_tree");

        // Commit to base codewords

        let mut proof_stream = StarkProofStream::default();
        let base_merkle_tree_root: Vec<BFieldElement> = base_merkle_tree.get_root();
        proof_stream.enqueue(&Item::MerkleRoot(base_merkle_tree_root));

        timer.elapsed("proof_stream.enqueue");

        let seed = proof_stream.prover_fiat_shamir();

        timer.elapsed("prover_fiat_shamir");

        let challenge_weights =
            Self::sample_weights(&hasher, &seed, AllChallenges::TOTAL_CHALLENGES);
        let all_challenges: AllChallenges = AllChallenges::create_challenges(&challenge_weights);

        timer.elapsed("sample_weights");

        let initial_weights = Self::sample_weights(&hasher, &seed, AllEndpoints::TOTAL_ENDPOINTS);
        let all_initials: AllEndpoints = AllEndpoints::create_initials(&initial_weights);

        timer.elapsed("initials");

        let (ext_tables, all_terminals) =
            ExtTableCollection::extend_tables(&base_tables, &all_challenges, &all_initials);
        let ext_codeword_tables = ext_tables.codeword_tables(&self.xfri_domain);
        let all_ext_codewords: Vec<Vec<XWord>> = ext_codeword_tables.concat_table_data();

        timer.elapsed("extend + get_terminals");

        timer.elapsed("get_and_set_all_extension_codewords");

        let transposed_extension_codewords: Vec<Vec<XFieldElement>> = (0..all_ext_codewords[0]
            .len())
            .map(|i| {
                all_ext_codewords
                    .iter()
                    .map(|inner| inner[i])
                    .collect::<Vec<XFieldElement>>()
            })
            .collect();

        let mut extension_codeword_digests_by_index: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(transposed_extension_codewords.len());

        let transposed_extension_codewords_clone = transposed_extension_codewords.clone();
        transposed_extension_codewords_clone
            .into_par_iter()
            .map(|xvalues| {
                let bvalues: Vec<BFieldElement> = xvalues
                    .into_iter()
                    .map(|x| x.coefficients.clone().to_vec())
                    .concat();
                assert_eq!(
                    27,
                    bvalues.len(),
                    "9 X-field elements must become 27 B-field elements"
                );
                hasher.hash(&bvalues, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect_into_vec(&mut extension_codeword_digests_by_index);

        let extension_tree =
            MerkleTree::<StarkHasher>::from_digests(&extension_codeword_digests_by_index);
        proof_stream.enqueue(&Item::MerkleRoot(extension_tree.get_root()));

        timer.elapsed("extension_tree");

        let extension_degree_bounds: Vec<Degree> = ext_tables.get_all_extension_degree_bounds();

        timer.elapsed("get_all_extension_degree_bounds");

        let mut quotient_codewords =
            ext_tables.get_all_quotients(&self.bfri_domain, &all_challenges, &all_terminals);

        timer.elapsed("all_quotients");

        let mut quotient_degree_bounds =
            ext_tables.get_all_quotient_degree_bounds(&all_challenges, &all_terminals);

        timer.elapsed("all_quotient_degree_bounds");

        // Prove equal initial values for the permutation-extension column pairs
        for pa in PermArg::all_permutation_arguments().iter() {
            quotient_codewords.push(pa.quotient(&ext_codeword_tables, &self.fri.domain));
            quotient_degree_bounds.push(pa.quotient_degree_bound(&ext_codeword_tables));
        }

        // Calculate `num_base_polynomials` and `num_extension_polynomials` for asserting
        let num_base_polynomials: usize = base_tables.into_iter().map(|table| table.width()).sum();
        let num_extension_polynomials: usize = ext_tables
            .into_iter()
            .map(|ext_table| ext_table.width() - ext_table.base_width())
            .sum();

        timer.elapsed("num_(base+extension)_polynomials");

        let num_randomizer_polynomials: usize = 1;
        let num_quotient_polynomials: usize = quotient_degree_bounds.len();
        let base_degree_bounds = base_tables.get_all_base_degree_bounds();

        timer.elapsed("get_all_base_degree_bounds");

        // Get weights for nonlinear combination
        let weights_seed: Vec<BFieldElement> = proof_stream.prover_fiat_shamir();

        timer.elapsed("prover_fiat_shamir (again)");

        let weights_count = num_randomizer_polynomials
            + 2 * (num_base_polynomials + num_extension_polynomials + num_quotient_polynomials);
        let weights = Self::sample_weights(&hasher, &weights_seed, weights_count);

        timer.elapsed("sample_weights");

        // let mut terms: Vec<Vec<XFieldElement>> = vec![x_randomizer_codeword];
        let mut weights_counter = 0;
        let mut combination_codeword: Vec<XFieldElement> = x_randomizer_codeword
            .into_iter()
            .map(|elem| elem * weights[weights_counter])
            .collect();
        weights_counter += 1;
        assert_eq!(base_codewords.len(), num_base_polynomials);
        let fri_x_values: Vec<BFieldElement> = self.fri.domain.b_domain_values();

        timer.elapsed("b_domain_values");

        for (i, (bc, bdb)) in base_codewords
            .iter()
            .zip(base_degree_bounds.iter())
            .enumerate()
        {
            let bc_lifted: Vec<XFieldElement> = bc.iter().map(|bfe| bfe.lift()).collect();

            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(bc_lifted.into_par_iter())
                .map(|(c, bcl)| c + bcl * weights[weights_counter])
                .collect();
            weights_counter += 1;
            let shift = (max_degree as Degree - bdb) as u32;
            let bc_shifted: Vec<XFieldElement> = fri_x_values
                .par_iter()
                .zip(bc.par_iter())
                .map(|(x, &bce)| (bce * x.mod_pow_u32(shift)).lift())
                .collect();

            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(&bc_shifted);
                assert!(
                    interpolated.degree() == -1 || interpolated.degree() == max_degree as isize,
                    "The shifted base codeword with index {} must be of maximal degree {}. Got {}.",
                    i,
                    max_degree,
                    interpolated.degree()
                );
            }

            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(bc_shifted.into_par_iter())
                .map(|(c, new_elem)| c + new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
        }

        timer.elapsed("...shift and collect base codewords");

        assert_eq!(all_ext_codewords.len(), num_extension_polynomials);
        for (i, (ec, edb)) in all_ext_codewords
            .iter()
            .zip(extension_degree_bounds.iter())
            .enumerate()
        {
            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(ec.par_iter())
                .map(|(c, new_elem)| c + *new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
            let shift = (max_degree as Degree - edb) as u32;
            let ec_shifted: Vec<XFieldElement> = fri_x_values
                .par_iter()
                .zip(ec.into_par_iter())
                .map(|(x, &ece)| ece * x.mod_pow_u32(shift).lift())
                .collect();

            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(&ec_shifted);
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == max_degree as isize,
                    "The shifted extension codeword with index {} must be of maximal degree {}. Got {}.",
                    i,
                    max_degree,
                    interpolated.degree()
                );
            }

            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(ec_shifted.into_par_iter())
                .map(|(c, new_elem)| c + new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
        }
        timer.elapsed("...shift and collect extension codewords");

        assert_eq!(quotient_codewords.len(), num_quotient_polynomials);
        for (_i, (qc, qdb)) in quotient_codewords
            .iter()
            .zip(quotient_degree_bounds.iter())
            .enumerate()
        {
            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(qc.par_iter())
                .map(|(c, new_elem)| c + *new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
            let shift = (max_degree as Degree - qdb) as u32;
            let qc_shifted: Vec<XFieldElement> = fri_x_values
                .par_iter()
                .zip(qc.into_par_iter())
                .map(|(x, &qce)| qce * x.mod_pow_u32(shift).lift())
                .collect();

            // TODO: Not all the degrees of the shifted quotient codewords are of max degree. Why?
            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(&qc_shifted);
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == max_degree as isize,
                    "The shifted quotient codeword with index {} must be of maximal degree {}. Got {}. Predicted degree of unshifted codeword: {}. . Shift = {}",
                    _i,
                    max_degree,
                    interpolated.degree(),
                    qdb,
                    shift
                );
            }

            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(qc_shifted.into_par_iter())
                .map(|(c, new_elem)| c + new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
        }

        timer.elapsed("...shift and collect quotient codewords");

        let mut combination_codeword_digests: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(combination_codeword.len());
        combination_codeword
            .clone()
            .into_par_iter()
            .map(|xfe| {
                let digest: Vec<BFieldElement> = xfe.to_digest();
                hasher.hash(&digest, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect_into_vec(&mut combination_codeword_digests);
        let combination_tree =
            MerkleTree::<StarkHasher>::from_digests(&combination_codeword_digests);
        let combination_root: Vec<BFieldElement> = combination_tree.get_root();
        proof_stream.enqueue(&Item::MerkleRoot(combination_root.clone()));

        timer.elapsed("combination_tree");

        // TODO: Consider factoring out code to find `unit_distances`, duplicated in verifier
        let mut unit_distances: Vec<usize> = ext_tables // XXX:
            .into_iter()
            .map(|table| table.unit_distance(self.fri.domain.length))
            .collect();
        unit_distances.push(0);
        unit_distances.sort_unstable();
        unit_distances.dedup();

        timer.elapsed("unit_distances");

        // Get indices of leafs to prove nonlinear combination
        let indices_seed: Vec<BFieldElement> = proof_stream.prover_fiat_shamir();
        let indices: Vec<usize> =
            hasher.sample_indices(self.security_level, &indices_seed, self.fri.domain.length);

        timer.elapsed("sample_indices");

        // TODO: I don't like that we're calling FRI right after getting the indices through
        // the Fiat-Shamir public oracle above. The reason I don't like this is that it implies
        // using Fiat-Shamir twice with somewhat similar proof stream content. A cryptographer
        // or mathematician should take a look on this part of the code.
        // prove low degree of combination polynomial
        let (_fri_indices, combination_root_verify) = self
            .fri
            .prove(&combination_codeword, &mut proof_stream)
            .unwrap();
        timer.elapsed("fri.prove");
        assert_eq!(
            combination_root, combination_root_verify,
            "Combination root from STARK and from FRI must agree"
        );

        // Open leafs of zipped codewords at indicated positions
        let mut revealed_indices: Vec<usize> = vec![];
        for index in indices.iter() {
            for unit_distance in unit_distances.iter() {
                let idx: usize = (index + unit_distance) % self.fri.domain.length;
                revealed_indices.push(idx);
            }
        }
        revealed_indices.sort_unstable();
        revealed_indices.dedup();

        let revealed_elements: Vec<Vec<BFieldElement>> = revealed_indices
            .iter()
            .map(|idx| transposed_base_codewords[*idx].clone())
            .collect();
        let auth_paths: Vec<PartialAuthenticationPath<Vec<BFieldElement>>> =
            base_merkle_tree.get_multi_proof(&revealed_indices);

        proof_stream.enqueue(&Item::TransposedBaseElementVectors(revealed_elements));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(auth_paths));

        let revealed_extension_elements: Vec<Vec<XFieldElement>> = revealed_indices
            .iter()
            .map(|idx| transposed_extension_codewords[*idx].clone())
            .collect_vec();
        let extension_auth_paths = extension_tree.get_multi_proof(&revealed_indices);
        proof_stream.enqueue(&Item::TransposedExtensionElementVectors(
            revealed_extension_elements,
        ));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(extension_auth_paths));

        // debug_assert!(
        //     MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
        //         base_merkle_tree.get_root(),
        //         &revealed_indices,
        //         revealed_indices.iter().map()base_codeword_digests_by_index[idx].clone(),
        //         auth_path.clone(),
        //     ),
        //     "authentication path for base tree must be valid"
        // );

        timer.elapsed("open leafs of zipped codewords");

        // open combination codeword at the same positions
        // Notice that we need to loop over `indices` here, not `revealed_indices`
        // as the latter includes adjacent table rows relative to the values in `indices`
        let revealed_combination_elements: Vec<XFieldElement> =
            indices.iter().map(|i| combination_codeword[*i]).collect();
        let revealed_combination_auth_paths = combination_tree.get_multi_proof(&indices);
        proof_stream.enqueue(&Item::RevealedCombinationElements(
            revealed_combination_elements,
        ));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(
            revealed_combination_auth_paths,
        ));

        timer.elapsed("open combination codeword at same positions");

        let report = timer.finish();
        println!("{}", report);
        println!(
            "Created proof containing {} B-field elements",
            proof_stream.transcript_length()
        );

        proof_stream
    }

    // FIXME: This interface leaks abstractions: We want a function that generates a number of weights
    // that doesn't care about the weights-to-digest ratio (we can make two weights per digest).
    fn sample_weights(
        hasher: &StarkHasher,
        seed: &StarkDigest,
        count: usize,
    ) -> Vec<XFieldElement> {
        hasher
            .get_n_hash_rounds(seed, count / 2)
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
