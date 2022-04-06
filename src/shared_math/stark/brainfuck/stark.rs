use super::vm::{InstructionMatrixBaseRow, Register};
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::other::roundup_npo2;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::stark::brainfuck::evaluation_argument::{
    EvaluationArgument, ProgramEvaluationArgument, PROGRAM_EVALUATION_CHALLENGE_INDICES_COUNT,
};
use crate::shared_math::stark::brainfuck::instruction_table::InstructionTable;
use crate::shared_math::stark::brainfuck::io_table::IOTable;
use crate::shared_math::stark::brainfuck::memory_table::MemoryTable;
use crate::shared_math::stark::brainfuck::permutation_argument::PermutationArgument;
use crate::shared_math::stark::brainfuck::table;
use crate::shared_math::stark::brainfuck::table_collection::TableCollection;
use crate::shared_math::stark::stark_verify_error::StarkVerifyError;
use crate::shared_math::traits::{FromVecu8, GetRandomElements, Inverse, ModPowU32};
use crate::shared_math::{
    b_field_element::BFieldElement, fri::Fri, other::is_power_of_two,
    stark::brainfuck::processor_table::ProcessorTable, traits::GetPrimitiveRootOfUnity,
    x_field_element::XFieldElement,
};
use crate::util_types::merkle_tree::MerkleTree;
use crate::util_types::proof_stream::ProofStream;
use crate::util_types::simple_hasher::{Hasher, RescuePrimeProduction, ToDigest};
use itertools::Itertools;
use rand::thread_rng;
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::rc::Rc;

pub const EXTENSION_CHALLENGE_COUNT: usize = 11;
pub const PERMUTATION_ARGUMENTS_COUNT: usize = 2;
pub const TERMINAL_COUNT: usize = 5;

const SIZE_OF_RP_HASH_IN_BYTES: usize = 8 + 5 * 128 / 8;

pub struct Stark {
    trace_length: usize,
    program: Vec<BFieldElement>,
    // TODO: Are all these fields really not needed?
    _input_symbols: Vec<BFieldElement>,
    _output_symbols: Vec<BFieldElement>,
    _expansion_factor: u64,
    security_level: usize,
    _num_randomizers: usize,
    tables: Rc<RefCell<TableCollection>>,
    // TODO: turn max_degree into i64 to match other degrees, which are i64
    max_degree: u64,
    fri: Fri<XFieldElement, blake3::Hasher>,

    permutation_arguments: [PermutationArgument; PERMUTATION_ARGUMENTS_COUNT],
    io_evaluation_arguments: [EvaluationArgument; 2],
    program_evaluation_argument: ProgramEvaluationArgument,
}

impl Stark {
    // TODO: Change this to use Rescue prime instead of Vec<u8>/Blake3
    // TODO: Use simple_hasher's get_n_hash_rounds() instead.
    // TODO: Also: This does prevent repeated indices which I think reduces security
    fn sample_indices(number: usize, seed: Vec<u8>, bound: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = vec![];
        for i in 0..number {
            let mut byte_array: Vec<u8> = seed.clone();
            byte_array.append(&mut i.to_be_bytes().to_vec());
            let digest: Vec<u8> = blake3::hash(&byte_array).as_bytes().to_vec();
            let mut integer: u128 = 0;
            for b in digest.iter().take(16) {
                integer = integer * 256 + *b as u128;
            }
            indices.push((integer % bound as u128) as usize);
        }

        indices
    }

    // TODO: Change this to use Rescue prime instead of Vec<u8>/Blake3
    // TODO: Use simple_hasher's get_n_hash_rounds() instead.
    fn sample_weights(number: u8, seed: Vec<u8>) -> Vec<XFieldElement> {
        let mut challenges: Vec<XFieldElement> = vec![];
        for i in 0..number {
            let mut mutated_challenge_seed = seed.clone();
            mutated_challenge_seed[0] = ((mutated_challenge_seed[0] as u16 + i as u16) % 256) as u8;
            // This is wrong because it doesn't re-hash above value:
            challenges.push(XFieldElement::ring_zero().from_vecu8(mutated_challenge_seed));
        }

        challenges
    }

    pub fn new(
        trace_length: usize,
        program: Vec<BFieldElement>,
        input_symbols: Vec<BFieldElement>,
        output_symbols: Vec<BFieldElement>,
    ) -> Self {
        let log_expansion_factor = 2; // TODO: For speed
        let expansion_factor: u64 = 1 << log_expansion_factor;
        let security_level = 2; // TODO: Consider increasing this
        let colinearity_checks_count = security_level / log_expansion_factor;
        assert!(
            colinearity_checks_count > 0,
            "At least one colinearity check is required"
        );
        assert!(
            is_power_of_two(expansion_factor),
            "expansion factor must be power of two."
        );
        assert!(
            expansion_factor >= 4,
            "expansion factor must be at least 4."
        );

        let num_randomizers = 1;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u128)
            .0
            .unwrap();

        // instantiate table objects
        let processor_table =
            ProcessorTable::new(trace_length, num_randomizers, smooth_generator, order);

        let instruction_table = InstructionTable::new(
            trace_length + program.len(),
            num_randomizers,
            smooth_generator,
            order,
        );

        let memory_table = MemoryTable::new(trace_length, num_randomizers, smooth_generator, order);
        let input_table = IOTable::new_input_table(input_symbols.len(), smooth_generator, order);
        let output_table = IOTable::new_output_table(output_symbols.len(), smooth_generator, order);

        let base_tables = TableCollection::new(
            processor_table,
            instruction_table,
            memory_table,
            input_table,
            output_table,
        );

        // instantiate permutation objects
        let rc_base_tables = Rc::new(RefCell::new(base_tables));

        let processor_instruction_lhs = (
            table::PROCESSOR_TABLE,
            ProcessorTable::INSTRUCTION_PERMUTATION,
        );
        let processor_instruction_rhs = (table::INSTRUCTION_TABLE, InstructionTable::PERMUTATION);
        let processor_instruction_permutation = PermutationArgument::new(
            rc_base_tables.clone(),
            processor_instruction_lhs,
            processor_instruction_rhs,
        );

        let processor_memory_lhs = (table::PROCESSOR_TABLE, ProcessorTable::MEMORY_PERMUTATION);
        let processor_memory_rhs = (table::MEMORY_TABLE, MemoryTable::PERMUTATION);
        let processor_memory_permutation = PermutationArgument::new(
            rc_base_tables.clone(),
            processor_memory_lhs,
            processor_memory_rhs,
        );

        let permutation_arguments: [PermutationArgument; 2] = [
            processor_instruction_permutation,
            processor_memory_permutation,
        ];

        let input_evaluation = EvaluationArgument::new(
            rc_base_tables.borrow().input_table.challenge_index(),
            rc_base_tables.borrow().input_table.terminal_index(),
            input_symbols.clone(),
        );

        let output_evaluation = EvaluationArgument::new(
            rc_base_tables.borrow().output_table.challenge_index(),
            rc_base_tables.borrow().output_table.terminal_index(),
            output_symbols.clone(),
        );
        let io_evaluation_arguments = [input_evaluation, output_evaluation];

        let program_challenge_indices: [usize; PROGRAM_EVALUATION_CHALLENGE_INDICES_COUNT] =
            [0, 1, 2, 10];
        let program_terminal_index = 4;

        let program_evaluation_argument = ProgramEvaluationArgument::new(
            program_challenge_indices,
            program_terminal_index,
            program.clone(),
        );

        // Compute max degree
        let mut max_degree: u64 = rc_base_tables.borrow().get_max_degree();
        max_degree = roundup_npo2(max_degree) - 1;
        let fri_domain_length: u64 = (max_degree + 1) * expansion_factor;

        // Instantiate FRI object
        let b_field_generator = BFieldElement::generator();
        let b_field_omega = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(fri_domain_length as u128)
            .0
            .unwrap();
        let fri: Fri<XFieldElement, blake3::Hasher> = Fri::new(
            b_field_generator.lift(),
            b_field_omega.lift(),
            fri_domain_length as usize,
            expansion_factor as usize,
            colinearity_checks_count,
        );

        Self {
            trace_length,
            program,
            _input_symbols: input_symbols,
            _output_symbols: output_symbols,
            _expansion_factor: expansion_factor,
            security_level,
            _num_randomizers: num_randomizers,
            tables: rc_base_tables,
            max_degree,
            fri,
            permutation_arguments,
            io_evaluation_arguments,
            program_evaluation_argument,
        }
    }

    pub fn prove(
        &mut self,
        processor_matrix: Vec<Register>,
        instruction_matrix: Vec<InstructionMatrixBaseRow>,
        input_matrix: Vec<BFieldElement>,
        output_matrix: Vec<BFieldElement>,
    ) -> Result<ProofStream, Box<dyn Error>> {
        assert_eq!(self.trace_length, processor_matrix.len());
        assert_eq!(
            self.trace_length + self.program.len(),
            instruction_matrix.len(),
            "instruction_matrix must contain both the execution trace and the program"
        );

        self.tables.borrow_mut().set_matrices(
            processor_matrix,
            instruction_matrix,
            input_matrix,
            output_matrix,
        );

        self.tables.borrow_mut().pad();

        // Instantiate the memory table object
        let processor_matrix_clone = self.tables.borrow().processor_table.0.matrix.clone();
        self.tables.borrow_mut().memory_table.0.matrix =
            MemoryTable::derive_matrix(processor_matrix_clone);

        // Generate randomizer codewords for zero-knowledge
        // This generates three B field randomizer codewords, each with the same length as the FRI domain
        let mut rng = thread_rng();
        let randomizer_polynomial = Polynomial::new(XFieldElement::random_elements(
            self.max_degree as usize + 1,
            &mut rng,
        ));
        let x_randomizer_codeword: Vec<XFieldElement> =
            self.fri.domain.x_evaluate(&randomizer_polynomial);
        let mut b_randomizer_codewords: [Vec<BFieldElement>; 3] = [vec![], vec![], vec![]];
        for x_elem in x_randomizer_codeword.iter() {
            b_randomizer_codewords[0].push(x_elem.coefficients[0]);
            b_randomizer_codewords[1].push(x_elem.coefficients[1]);
            b_randomizer_codewords[2].push(x_elem.coefficients[2]);
        }

        let base_codewords: Vec<Vec<BFieldElement>> = self
            .tables
            .borrow_mut()
            .get_and_set_all_base_codewords(&self.fri.domain);
        let all_base_codewords =
            vec![b_randomizer_codewords.into(), base_codewords.clone()].concat();

        // TODO: How do I make a single Merkle tree from many codewords?
        // If the Merkle trees are always opened for all base codewords for a single index, then
        // we *should* be able to make a commitment to *each* index and store that list of commitments
        // in a single Merkle tree. This list of commitments will have length 2^k, so this should be
        // possible, as the MT requires a leaf count that is a power of two.
        let transposed_base_codewords: Vec<Vec<BFieldElement>> = (0..all_base_codewords[0].len())
            .map(|i| {
                all_base_codewords
                    .iter()
                    .map(|inner| inner[i])
                    .collect::<Vec<BFieldElement>>()
            })
            .collect();
        let mut hasher = RescuePrimeProduction::new();

        // Current length of each element in `transposed_base_codewords` is 18 which exceeds
        // max length of RP hash function. So we chop it into elements that will fit into the
        // rescue prime hash function. This is done by chopping the hash function input into
        // chunks of `max_length / 2` and calling `hash_many` on this input. Half the max
        // length is needed since the chunks are hashed two at a time.
        let base_codeword_digests_by_index: Vec<Vec<BFieldElement>> = transposed_base_codewords
            .clone()
            .into_iter()
            .map(|values| {
                let chunks: Vec<Vec<BFieldElement>> = values
                    .chunks(hasher.0.max_input_length / 2)
                    .map(|s| s.to_vec())
                    .collect();
                hasher.hash_many(&chunks)
            })
            .collect();
        let base_merkle_tree =
            MerkleTree::<Vec<BFieldElement>, RescuePrimeProduction>::from_digests(
                &base_codeword_digests_by_index,
                &vec![BFieldElement::ring_zero()],
            );

        // Commit to base codewords
        let mut proof_stream = ProofStream::default();
        let base_merkle_tree_root: &Vec<BFieldElement> = base_merkle_tree.get_root();
        proof_stream.enqueue(base_merkle_tree_root)?;

        // Get coefficients for table extension
        // TODO: REPLACE THIS WITH RescuePrime/B field elements. The type of `challenges`
        // must not change though, it should remain `Vec<XFieldElement>`.
        let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT] = Self::sample_weights(
            EXTENSION_CHALLENGE_COUNT as u8,
            proof_stream.prover_fiat_shamir(),
        )
        .try_into()
        .unwrap();

        let initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT] =
            XFieldElement::random_elements(PERMUTATION_ARGUMENTS_COUNT, &mut rng)
                .try_into()
                .unwrap();

        self.tables.borrow_mut().extend(challenges, initials);

        let extension_codewords = self
            .tables
            .borrow_mut()
            .get_and_set_all_extension_codewords(&self.fri.domain);

        let transposed_extension_codewords: Vec<Vec<XFieldElement>> = (0..extension_codewords[0]
            .len())
            .map(|i| {
                extension_codewords
                    .iter()
                    .map(|inner| inner[i])
                    .collect::<Vec<XFieldElement>>()
            })
            .collect();
        let extension_codeword_digests_by_index: Vec<Vec<BFieldElement>> =
            transposed_extension_codewords
                .clone()
                .into_iter()
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
                    let chunks: Vec<Vec<BFieldElement>> = bvalues
                        .chunks(hasher.0.max_input_length / 2)
                        .map(|s| s.into())
                        .collect();
                    assert_eq!(
                        6,
                        chunks.len(),
                        "27 B-field elements must be divided into 6 chunks for hashing"
                    );
                    hasher.hash_many(&chunks)
                })
                .collect();
        let extension_tree = MerkleTree::<Vec<BFieldElement>, RescuePrimeProduction>::from_digests(
            &extension_codeword_digests_by_index,
            &vec![BFieldElement::ring_zero()],
        );
        proof_stream.enqueue(extension_tree.get_root())?;

        let extension_degree_bounds: Vec<Degree> =
            self.tables.borrow().get_all_extension_degree_bounds();

        let terminals: [XFieldElement; TERMINAL_COUNT] = self.tables.borrow().get_terminals();

        let mut quotient_codewords =
            self.tables
                .borrow()
                .all_quotients(&self.fri.domain, challenges, terminals);
        let mut quotient_degree_bounds = self
            .tables
            .borrow()
            .all_quotient_degree_bounds(challenges, terminals);

        for pa in self.permutation_arguments.iter() {
            quotient_codewords.push(pa.quotient(&self.fri.domain));
            quotient_degree_bounds.push(pa.quotient_degree_bound());
        }

        for t in terminals.iter() {
            proof_stream.enqueue(t)?;
        }

        let num_base_polynomials: usize = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| table.base_width())
            .sum();
        let num_extension_polynomials: usize = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| table.full_width() - table.base_width())
            .sum();
        let num_randomizer_polynomials: usize = 1;
        let num_quotient_polynomials: usize = quotient_degree_bounds.len();
        let base_degree_bounds = self.tables.borrow().get_all_base_degree_bounds();

        let mut terms: Vec<Vec<XFieldElement>> = vec![x_randomizer_codeword];
        assert_eq!(base_codewords.len(), num_base_polynomials);
        let fri_x_values: Vec<BFieldElement> = self.fri.domain.b_domain_values();
        for (i, (bc, bdb)) in base_codewords
            .iter()
            .zip(base_degree_bounds.iter())
            .enumerate()
        {
            let bc_lifted: Vec<XFieldElement> = bc.iter().map(|bfe| bfe.lift()).collect();
            terms.push(bc_lifted);
            let shift = (self.max_degree as Degree - bdb) as u32;
            let bc_shifted: Vec<XFieldElement> = fri_x_values
                .iter()
                .zip(bc.iter())
                .map(|(x, &bce)| (bce * x.mod_pow_u32(shift)).lift())
                .collect();
            terms.push(bc_shifted);

            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(terms.last().unwrap());
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == self.max_degree as isize,
                    "The shifted base codeword with index {} must be of maximal degree {}. Got {}.",
                    i,
                    self.max_degree,
                    interpolated.degree()
                );
            }
        }

        assert_eq!(extension_codewords.len(), num_extension_polynomials);
        for (i, (ec, edb)) in extension_codewords
            .iter()
            .zip(extension_degree_bounds.iter())
            .enumerate()
        {
            terms.push(ec.to_vec());
            let shift = (self.max_degree as Degree - edb) as u32;
            let ec_shifted: Vec<XFieldElement> = fri_x_values
                .iter()
                .zip(ec.iter())
                .map(|(x, &ece)| ece * x.mod_pow_u32(shift).lift())
                .collect();
            terms.push(ec_shifted);

            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(terms.last().unwrap());
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == self.max_degree as isize,
                    "The shifted extension codeword with index {} must be of maximal degree {}. Got {}.",
                    i,
                    self.max_degree,
                    interpolated.degree()
                );
            }
        }

        assert_eq!(quotient_codewords.len(), num_quotient_polynomials);
        for (i, (qc, qdb)) in quotient_codewords
            .iter()
            .zip(quotient_degree_bounds.iter())
            .enumerate()
        {
            terms.push(qc.to_vec());
            let shift = (self.max_degree as Degree - qdb) as u32;
            let qc_shifted: Vec<XFieldElement> = fri_x_values
                .iter()
                .zip(qc.iter())
                .map(|(x, &qce)| qce * x.mod_pow_u32(shift).lift())
                .collect();
            terms.push(qc_shifted);

            // TODO: Not all the degrees of the shifted quotient codewords are of max degree. Why?
            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(terms.last().unwrap());
                let unshifted_degree = self
                    .fri
                    .domain
                    .x_interpolate(&terms[terms.len() - 2])
                    .degree();
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == self.max_degree as isize,
                    "The shifted quotient codeword with index {} must be of maximal degree {}. Got {}. Predicted degree of unshifted codeword: {}. Actual degree of unshifted codeword: {}. Shift = {}",
                    i,
                    self.max_degree,
                    interpolated.degree(),
                    qdb,
                    unshifted_degree,
                    shift
                );
            }
        }

        // Get weights for nonlinear combination
        let weights_seed = proof_stream.prover_fiat_shamir();
        let weights = Self::sample_weights(
            (num_randomizer_polynomials
                + 2 * (num_base_polynomials + num_extension_polynomials + num_quotient_polynomials))
                as u8,
            weights_seed,
        );

        assert_eq!(
            terms.len(),
            weights.len(),
            "Number of terms in non-linear combination must match number of weights"
        );

        // Take weighted sum
        let combination_codeword: Vec<XFieldElement> = weights
            .iter()
            .zip(terms.iter())
            .map(|(w, t)| {
                (0..self.fri.domain.length)
                    .map(|i| *w * t[i])
                    .collect::<Vec<XFieldElement>>()
            })
            .fold(
                vec![XFieldElement::ring_zero(); self.fri.domain.length],
                |acc, weighted_terms| {
                    acc.iter()
                        .zip(weighted_terms.iter())
                        .map(|(a, wt)| *a + *wt)
                        .collect()
                },
            );
        let combination_codeword_digests: Vec<Vec<BFieldElement>> = combination_codeword
            .clone()
            .into_iter()
            .map(|xfe| {
                let digest: Vec<BFieldElement> = xfe.to_digest();
                hasher.hash(&digest)
            })
            .collect();
        let combination_tree =
            MerkleTree::<Vec<BFieldElement>, RescuePrimeProduction>::from_digests(
                &combination_codeword_digests,
                &vec![BFieldElement::ring_zero()],
            );
        let combination_root: &Vec<BFieldElement> = combination_tree.get_root();
        proof_stream.enqueue(combination_root)?;

        // TODO: Consider factoring out code to find `unit_distances`, duplicated in verifier
        let mut unit_distances: Vec<usize> = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| table.unit_distance(self.fri.domain.length))
            .collect();
        unit_distances.push(0);
        unit_distances.sort_unstable();
        unit_distances.dedup();

        // Get indices of leafs to prove nonlinear combination
        let indices_seed = proof_stream.prover_fiat_shamir();
        let indices =
            Self::sample_indices(self.security_level, indices_seed, self.fri.domain.length);

        // Open leafs of zipped codewords at indicated positions
        for index in indices.iter() {
            for unit_distance in unit_distances.iter() {
                let idx: usize = (index + unit_distance) % self.fri.domain.length;

                let elements: Vec<BFieldElement> = transposed_base_codewords[idx].clone();
                let auth_path: Vec<Vec<BFieldElement>> =
                    base_merkle_tree.get_authentication_path(idx);
                proof_stream.enqueue(&elements)?;
                // proof_stream.enqueue(&auth_path)?;
                proof_stream.enqueue_length_prepended(&auth_path)?;

                let leaf_digest = base_codeword_digests_by_index[idx].clone();
                let success = MerkleTree::<Vec<BFieldElement>, RescuePrimeProduction>::verify_authentication_path_from_leaf_hash(
                    base_merkle_tree.get_root().clone(), idx as u32, leaf_digest, auth_path);
                assert!(success, "authentication path for base tree must be valid");

                let extension_elements: Vec<XFieldElement> =
                    transposed_extension_codewords[idx].clone();
                let extension_path: Vec<Vec<BFieldElement>> =
                    extension_tree.get_authentication_path(idx);
                proof_stream.enqueue(&extension_elements)?;
                proof_stream.enqueue_length_prepended(&extension_path)?;
            }
        }

        // open combination codeword at the same positions
        for index in indices {
            let revealed_combination_element = combination_codeword[index];
            let revealed_combination_auth_path = combination_tree.get_authentication_path(index);
            proof_stream.enqueue(&revealed_combination_element)?;
            proof_stream.enqueue_length_prepended(&revealed_combination_auth_path)?;
            assert!(MerkleTree::<Vec<BFieldElement>, RescuePrimeProduction>::
                    verify_authentication_path_from_leaf_hash(
                combination_tree.get_root().to_owned(),
                index as u32,
                combination_codeword_digests[index].clone(),
                revealed_combination_auth_path,
            ), "Combination Merkle Tree authentication path must verify");
        }

        // prove low degree of combination polynomial, and collect indices
        let _indices = self.fri.prove(&combination_codeword, &mut proof_stream)?;

        Ok(proof_stream)
    }

    pub fn verify(&self, proof_stream: &mut ProofStream) -> Result<bool, Box<dyn Error>> {
        let base_merkle_tree_root: Vec<BFieldElement> =
            proof_stream.dequeue(SIZE_OF_RP_HASH_IN_BYTES)?;

        // Get coefficients for table extension
        // TODO: REPLACE THIS WITH RescuePrime/B field elements. The type of `challenges`
        // must not change though, it should remain `Vec<XFieldElement>`.
        let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT] = Self::sample_weights(
            EXTENSION_CHALLENGE_COUNT as u8,
            proof_stream.verifier_fiat_shamir(),
        )
        .try_into()
        .unwrap();

        let extension_tree_merkle_root: Vec<BFieldElement> =
            proof_stream.dequeue(SIZE_OF_RP_HASH_IN_BYTES)?;
        let processor_instruction_permutation_terminal: XFieldElement =
            proof_stream.dequeue(3 * 128 / 8)?;
        let processor_memory_permutation_terminal: XFieldElement =
            proof_stream.dequeue(3 * 128 / 8)?;
        let processor_input_evaluation_terminal: XFieldElement =
            proof_stream.dequeue(3 * 128 / 8)?;
        let processor_output_evaluation_terminal: XFieldElement =
            proof_stream.dequeue(3 * 128 / 8)?;
        let instruction_evaluation_terminal: XFieldElement = proof_stream.dequeue(3 * 128 / 8)?;
        let terminals = [
            processor_instruction_permutation_terminal,
            processor_memory_permutation_terminal,
            processor_input_evaluation_terminal,
            processor_output_evaluation_terminal,
            instruction_evaluation_terminal,
        ];

        let base_degree_bounds: Vec<Degree> = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| vec![table.interpolant_degree(); table.base_width()])
            .concat();

        let extension_degree_bounds: Vec<Degree> = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| vec![table.interpolant_degree(); table.full_width() - table.base_width()])
            .concat();

        // get weights for nonlinear combination
        //  - 1 randomizer
        //  - 2 for every other polynomial (base, extension, quotients)
        let num_base_polynomials = base_degree_bounds.len();
        let num_extension_polynomials = extension_degree_bounds.len();
        let num_randomizer_polynomials = 1;
        let num_quotient_polynomials: usize = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| {
                table
                    .all_quotient_degree_bounds(challenges, terminals)
                    .len()
            })
            .sum();
        let num_difference_quotients = self.permutation_arguments.len();

        let weights_seed: Vec<u8> = proof_stream.verifier_fiat_shamir();
        let weights: Vec<XFieldElement> = Self::sample_weights(
            (num_randomizer_polynomials
                + 2 * num_base_polynomials
                + 2 * num_extension_polynomials
                + 2 * num_quotient_polynomials
                + 2 * num_difference_quotients) as u8,
            weights_seed,
        );

        let combination_root: Vec<BFieldElement> =
            proof_stream.dequeue(SIZE_OF_RP_HASH_IN_BYTES)?;

        let indices_seed: Vec<u8> = proof_stream.verifier_fiat_shamir();
        let indices =
            Self::sample_indices(self.security_level, indices_seed, self.fri.domain.length);

        // TODO: Consider factoring out code to find `unit_distances`, duplicated in prover
        let mut unit_distances: Vec<usize> = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| table.unit_distance(self.fri.domain.length))
            .collect();
        unit_distances.push(0);
        unit_distances.sort_unstable();
        unit_distances.dedup();

        let mut hasher = RescuePrimeProduction::new();
        let mut tuples: HashMap<usize, Vec<XFieldElement>> = HashMap::new();
        // TODO: we can store the elements mushed into "tuples" separately, like in "points" below,
        // to avoid unmushing later
        for index in indices.clone() {
            for unit_distance in unit_distances.iter() {
                let idx = (index + unit_distance) % self.fri.domain.length;
                let elements: Vec<BFieldElement> = proof_stream.dequeue(8 + 18 * 128 / 8)?;
                let auth_path: Vec<Vec<BFieldElement>> = proof_stream.dequeue_length_prepended()?;

                let hash_input: Vec<Vec<BFieldElement>> = elements
                    .chunks(hasher.0.max_input_length / 2)
                    .map(|s| s.to_vec())
                    .collect();
                let leaf_hash: Vec<BFieldElement> = hasher.hash_many(&hash_input);
                let mt_base_success = MerkleTree::<Vec<BFieldElement>, RescuePrimeProduction>::verify_authentication_path_from_leaf_hash(
                    base_merkle_tree_root.clone(), idx as u32, leaf_hash, auth_path);
                if !mt_base_success {
                    // TODO: Replace this by a specific error type, or just return `Ok(false)`
                    panic!("Failed to verify authentication path for base codeword");
                    // return Ok(false);
                }

                let randomizer: XFieldElement =
                    XFieldElement::new([elements[0], elements[1], elements[2]]);
                assert_eq!(
                    1, num_randomizer_polynomials,
                    "For now number of randomizers must be 1"
                );
                let mut values: Vec<XFieldElement> = vec![randomizer];
                values.extend_from_slice(
                    &elements
                        .iter()
                        .skip(3 * num_randomizer_polynomials)
                        .map(|bfe| bfe.lift())
                        .collect::<Vec<XFieldElement>>(),
                );
                tuples.insert(idx, values);

                let extension_elements: Vec<XFieldElement> =
                    proof_stream.dequeue(8 + 9 * 3 * 128 / 8)?;
                let extension_auth_path: Vec<Vec<BFieldElement>> =
                    proof_stream.dequeue_length_prepended()?;

                let extension_element_chunked: Vec<Vec<BFieldElement>> = extension_elements
                    .clone()
                    .into_iter()
                    .map(|xfe| xfe.coefficients.clone().to_vec())
                    .concat()
                    .chunks(hasher.0.max_input_length / 2)
                    .map(|s| s.to_vec())
                    .collect();
                let ext_leaf_hash = hasher.hash_many(&extension_element_chunked);

                let mt_ext_success = MerkleTree::<Vec<BFieldElement>, RescuePrimeProduction>::verify_authentication_path_from_leaf_hash(
                    extension_tree_merkle_root.clone(), idx as u32, ext_leaf_hash, extension_auth_path);
                if !mt_ext_success {
                    // TODO: Replace this by a specific error type, or just return `Ok(false)`
                    panic!("Failed to verify authentication path for extended codeword");
                    // return Ok(false);
                }

                tuples.insert(idx, vec![tuples[&idx].clone(), extension_elements].concat());
            }
        }

        // verify nonlinear combination
        for index in indices {
            // collect terms: randomizer
            let mut terms: Vec<XFieldElement> = (0..num_randomizer_polynomials)
                .map(|i| tuples[&index][i])
                .collect();

            // collect terms: base
            for i in num_randomizer_polynomials..num_randomizer_polynomials + num_base_polynomials {
                terms.push(tuples[&index][i]);
                let shift: u32 = (self.max_degree as i64
                    - base_degree_bounds[i - num_randomizer_polynomials])
                    as u32;
                terms.push(
                    tuples[&index][i]
                        * self
                            .fri
                            .domain
                            .b_domain_value(index as u32)
                            .mod_pow_u32(shift)
                            .lift(),
                );
            }

            // collect terms: extension
            let extension_offset = num_randomizer_polynomials + num_base_polynomials;

            assert_eq!(
                terms.len(),
                2 * extension_offset - num_randomizer_polynomials,
                "number of terms does not match with extension offset"
            );

            // TODO: We don't seem to need a separate loop for the base and extension columns.
            // But merging them would also require concatenating the degree bounds vector.
            for (i, edb) in extension_degree_bounds.iter().enumerate() {
                let extension_element: XFieldElement = tuples[&index][extension_offset + i];
                terms.push(extension_element);
                let shift = (self.max_degree as i64 - edb) as u32;
                terms.push(
                    extension_element
                        * self
                            .fri
                            .domain
                            .b_domain_value(index as u32)
                            .mod_pow_u32(shift)
                            .lift(),
                )
            }

            // collect terms: quotients, quotients need to be computed
            let mut acc_index = num_randomizer_polynomials;
            let mut points: Vec<Vec<XFieldElement>> = vec![];
            for table in self.tables.borrow().into_iter() {
                let table_base_width = table.base_width();
                points.push(tuples[&index][acc_index..acc_index + table_base_width].to_vec());
                acc_index += table_base_width;
            }

            assert_eq!(
                extension_offset, acc_index,
                "Column count in verifier must match until extension columns"
            );

            for (point, table) in points.iter_mut().zip(self.tables.borrow().into_iter()) {
                let step_size = table.full_width() - table.base_width();
                point.extend_from_slice(&tuples[&index][acc_index..acc_index + step_size]);
                acc_index += step_size;
            }

            assert_eq!(
                tuples[&index].len(),
                acc_index,
                "Column count in verifier must match until end"
            );

            let mut base_acc_index = num_randomizer_polynomials;
            let mut ext_acc_index = extension_offset;
            for (point, table) in points.iter().zip(self.tables.borrow().into_iter()) {
                // boundary
                for (constraint, bound) in table
                    .boundary_constraints_ext(challenges)
                    .iter()
                    .zip(table.boundary_quotient_degree_bounds(challenges).iter())
                {
                    let eval = constraint.evaluate(point);
                    let quotient = eval
                        / (self.fri.domain.b_domain_value(index as u32).lift()
                            - XFieldElement::ring_one());
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(
                        quotient
                            * self
                                .fri
                                .domain
                                .b_domain_value(index as u32)
                                .mod_pow_u32(shift)
                                .lift(),
                    );
                }

                // transition
                let unit_distance = table.unit_distance(self.fri.domain.length);
                let next_index = (index + unit_distance) % self.fri.domain.length;
                let mut next_point = tuples[&next_index]
                    [base_acc_index..base_acc_index + table.base_width()]
                    .to_vec();
                next_point.extend_from_slice(
                    &tuples[&next_index]
                        [ext_acc_index..ext_acc_index + table.full_width() - table.base_width()],
                );
                base_acc_index += table.base_width();
                ext_acc_index += table.full_width() - table.base_width();
                for (constraint, bound) in table
                    .transition_constraints_ext(challenges)
                    .iter()
                    .zip(table.transition_quotient_degree_bounds(challenges).iter())
                {
                    let eval =
                        constraint.evaluate(&vec![point.to_owned(), next_point.clone()].concat());
                    // If height == 0, then there is no subgroup where the transition polynomials should be zero.
                    // The fast zerofier (based on group theory) needs a non-empty group.
                    // Forcing it on an empty group generates a division by zero error.
                    let quotient = if table.height() == 0 {
                        XFieldElement::ring_zero()
                    } else {
                        let num = (self.fri.domain.b_domain_value(index as u32)
                            - table.omicron().inverse())
                        .lift();
                        let denom = self
                            .fri
                            .domain
                            .b_domain_value(index as u32)
                            .mod_pow_u32(table.height() as u32)
                            .lift()
                            - XFieldElement::ring_one();
                        eval * num / denom
                    };
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(
                        quotient
                            * self
                                .fri
                                .domain
                                .b_domain_value(index as u32)
                                .mod_pow_u32(shift)
                                .lift(),
                    );
                }

                // terminal
                for (constraint, bound) in table
                    .terminal_constraints_ext(challenges, terminals)
                    .iter()
                    .zip(
                        table
                            .terminal_quotient_degree_bounds(challenges, terminals)
                            .iter(),
                    )
                {
                    let eval = constraint.evaluate(point);
                    let quotient = eval
                        / (self.fri.domain.b_domain_value(index as u32).lift()
                            - table.omicron().inverse().lift());
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(
                        quotient
                            * self
                                .fri
                                .domain
                                .b_domain_value(index as u32)
                                .mod_pow_u32(shift)
                                .lift(),
                    )
                }
            }

            for arg in self.permutation_arguments.iter() {
                let quotient = arg.evaluate_difference(&points)
                    / (self.fri.domain.b_domain_value(index as u32).lift()
                        - XFieldElement::ring_one());
                terms.push(quotient);
                let degree_bound = arg.quotient_degree_bound();
                let shift = (self.max_degree as i64 - degree_bound) as u32;
                terms.push(
                    quotient
                        * self
                            .fri
                            .domain
                            .b_domain_value(index as u32)
                            .mod_pow_u32(shift)
                            .lift(),
                );
            }

            assert_eq!(
                weights.len(),
                terms.len(),
                "length of terms must be equal to length of weights"
            );

            // compute inner product of weights and terms
            // Todo: implement `sum` on XFieldElements
            let inner_product = weights
                .iter()
                .zip(terms.into_iter())
                .map(|(w, t)| *w * t)
                .fold(XFieldElement::ring_zero(), |x, y| x + y);
            // get value of the combination codeword to test the inner product against
            let combination_leaf: XFieldElement = proof_stream.dequeue(3 * 128 / 8)?;
            let combination_path: Vec<Vec<BFieldElement>> =
                proof_stream.dequeue_length_prepended()?;

            assert!(
                MerkleTree::<Vec<BFieldElement>, RescuePrimeProduction>::
                verify_authentication_path_from_leaf_hash(
                    combination_root.clone(),
                    index as u32,
                    hasher.hash(&combination_leaf.coefficients.to_vec()),
                    combination_path,
                ), "The combination root must match with the combination authentication path");

            assert_eq!(
                combination_leaf, inner_product,
                "The combination leaf must equal the inner product"
            );
        }

        // Verify low degree of combination polynomial
        self.fri.verify(proof_stream)?;

        // Verify external terminals
        for (i, iea) in self.io_evaluation_arguments.iter().enumerate() {
            if iea.select_terminal(terminals) != iea.compute_terminal(challenges) {
                return Err(Box::new(StarkVerifyError::EvaluationArgument(i)));
            }
        }

        if self.program_evaluation_argument.select_terminal(terminals)
            != self
                .program_evaluation_argument
                .compute_terminal(challenges)
        {
            return Err(Box::new(StarkVerifyError::ProgramEvaluationArgument));
        }

        Ok(true)
    }
}

#[cfg(test)]
mod brainfuck_stark_tests {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::stark::brainfuck;
    use crate::shared_math::stark::brainfuck::vm::BaseMatrices;

    #[test]
    fn prove_verify_test() {
        let program: Vec<BFieldElement> =
            brainfuck::vm::compile(brainfuck::vm::sample_programs::SHORT_INPUT_AND_OUTPUT).unwrap();
        let (trace_length, input_symbols, output_symbols) = brainfuck::vm::run(
            &program,
            vec![
                BFieldElement::new(97),
                BFieldElement::new(98),
                BFieldElement::new(99),
            ],
        )
        .unwrap();
        let base_matrices: BaseMatrices =
            brainfuck::vm::simulate(&program, &input_symbols).unwrap();
        let mut stark = Stark::new(trace_length, program, input_symbols, output_symbols);

        // TODO: If we set the `DEBUG` environment variable here, we *should* catch a lot of bugs.
        // Do we want to do that?
        let mut proof_stream = stark
            .prove(
                base_matrices.processor_matrix,
                base_matrices.instruction_matrix,
                base_matrices.input_matrix,
                base_matrices.output_matrix,
            )
            .unwrap();

        let verifier_verdict: Result<bool, Box<dyn Error>> = stark.verify(&mut proof_stream);
        match verifier_verdict {
            Ok(_) => (),
            Err(err) => panic!("error in STARK verifier: {}", err),
        };
    }
}
