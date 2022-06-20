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
use crate::shared_math::stark::triton::instruction::sample_programs;
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::stark::triton::table::challenges_endpoints::{AllChallenges, AllEndpoints};
use crate::shared_math::stark::triton::table::table_collection::{
    BaseTableCollection, ExtTableCollection,
};
use crate::shared_math::traits::{GetPrimitiveRootOfUnity, GetRandomElements};
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::{other, xfri};
use crate::timing_reporter::TimingReporter;
use crate::util_types::merkle_tree::MerkleTree;
use crate::util_types::simple_hasher::Hasher;
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
    _security_level: usize,
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

        let _fri_domain = triton::fri_domain::FriDomain {
            offset,
            omega,
            length: fri_domain_length as usize,
        };

        todo!()
    }

    pub fn prove(&self, base_matrices: BaseMatrices) {
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

        let all_base_codewords = vec![b_randomizer_codewords.into(), base_codewords].concat();

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

        transposed_extension_codewords
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

        let _extension_degree_bounds: Vec<Degree> = ext_tables.get_all_extension_degree_bounds();

        timer.elapsed("get_all_extension_degree_bounds");

        let mut _quotient_codewords =
            ext_tables.get_all_quotients(&self.bfri_domain, &all_challenges, &all_terminals);

        timer.elapsed("all_quotients");

        let mut _quotient_degree_bounds =
            ext_tables.get_all_quotient_degree_bounds(&all_challenges, &all_terminals);

        timer.elapsed("all_quotient_degree_bounds");

        // Prove equal initial values for the permutation-extension column pairs
        // for pa in self.permutation_arguments.iter() {
        //     quotient_codewords.push(pa.quotient(&self.fri.domain));
        //     quotient_degree_bounds.push(pa.quotient_degree_bound());
        // }

        todo!()
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
