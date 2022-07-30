use super::stark_proof_stream::StarkProofStream;
use super::vm::BaseMatrices;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::other::roundup_npo2;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::rescue_prime_xlix::{
    neptune_params, RescuePrimeXlix, RP_DEFAULT_OUTPUT_SIZE, RP_DEFAULT_WIDTH,
};
use crate::shared_math::stark::brainfuck::evaluation_argument::{
    EvaluationArgument, ProgramEvaluationArgument, PROGRAM_EVALUATION_CHALLENGE_INDICES_COUNT,
};
use crate::shared_math::stark::brainfuck::instruction_table::InstructionTable;
use crate::shared_math::stark::brainfuck::io_table::IOTable;
use crate::shared_math::stark::brainfuck::memory_table::MemoryTable;
use crate::shared_math::stark::brainfuck::permutation_argument::PermutationArgument;
use crate::shared_math::stark::brainfuck::stark_proof_stream::Item;
use crate::shared_math::stark::brainfuck::table_collection::TableCollection;
use crate::shared_math::stark::brainfuck::{table, vm};
use crate::shared_math::stark::stark_verify_error::StarkVerifyError;
use crate::shared_math::traits::{GetRandomElements, Inverse, ModPowU32};
use crate::shared_math::{
    b_field_element::BFieldElement, other::is_power_of_two,
    stark::brainfuck::processor_table::ProcessorTable, traits::GetPrimitiveRootOfUnity,
    x_field_element::XFieldElement, xfri::Fri,
};
use crate::timing_reporter::TimingReporter;
use crate::util_types::merkle_tree::{MerkleTree, PartialAuthenticationPath};
use crate::util_types::simple_hasher::{Hasher, ToDigest};
use itertools::Itertools;
use rand::thread_rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::rc::Rc;

pub const EXTENSION_CHALLENGE_COUNT: usize = 11;
pub const PERMUTATION_ARGUMENTS_COUNT: usize = 2;
pub const TERMINAL_COUNT: usize = 5;

type StarkHasher = RescuePrimeXlix<RP_DEFAULT_WIDTH>;
type StarkDigest = Vec<BFieldElement>;

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
    fri: Fri<StarkHasher>,

    permutation_arguments: [PermutationArgument; PERMUTATION_ARGUMENTS_COUNT],
    io_evaluation_arguments: [EvaluationArgument; 2],
    program_evaluation_argument: ProgramEvaluationArgument,
}

impl Stark {
    fn sample_weights(
        hasher: &StarkHasher,
        seed: &StarkDigest,
        count: usize,
    ) -> Vec<XFieldElement> {
        // FIXME: Perhaps re-use hasher.
        // FIXME: To make this work for blake3::Hasher, we need .into()s.
        // FIXME: When digests get a length of 6, produce half as many digests as XFieldElements.
        hasher
            .get_n_hash_rounds(seed, count)
            .par_iter()
            .map(|digest| XFieldElement::new([digest[0], digest[1], digest[2]]))
            .collect()
    }

    pub fn new(
        trace_length: usize,
        source_code: String,
        input_symbols: Vec<BFieldElement>,
        output_symbols: Vec<BFieldElement>,
        log_expansion_factor: usize,
        security_level: usize,
    ) -> Self {
        assert_eq!(
            0,
            security_level % log_expansion_factor,
            "security_level/log_expansion_factor must be a positive integer"
        );
        let expansion_factor: u64 = 1 << log_expansion_factor;
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

        // Verify that program compiles
        let program = match vm::compile(&source_code) {
            None => panic!("Failed to compile program received. Got: {}", source_code),
            Some(prog) => prog,
        };

        // Fewer than 2 randomizers means no zero-knowledge for the prover's execution.
        let num_randomizers = 2;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
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

        println!("fri_domain_length = {}", fri_domain_length);

        // Instantiate FRI object
        let b_field_generator = BFieldElement::generator();
        let b_field_omega = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(fri_domain_length as u64)
            .0
            .unwrap();
        let fri: Fri<StarkHasher> = Fri::new(
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
        base_matrices: BaseMatrices,
    ) -> Result<StarkProofStream, Box<dyn Error>> {
        let mut timer = TimingReporter::start();

        assert_eq!(self.trace_length, base_matrices.processor_matrix.len());
        assert_eq!(
            self.trace_length + self.program.len(),
            base_matrices.instruction_matrix.len(),
            "instruction_matrix must contain both the execution trace and the program"
        );

        self.tables.borrow_mut().set_matrices(
            base_matrices.processor_matrix,
            base_matrices.instruction_matrix,
            base_matrices.input_matrix,
            base_matrices.output_matrix,
        );

        timer.elapsed("assert, set_matrices");

        self.tables.borrow_mut().pad();

        timer.elapsed("pad");

        // Instantiate the memory table object
        let processor_matrix_clone = self.tables.borrow().processor_table.0.matrix.clone();
        self.tables.borrow_mut().memory_table.0.matrix =
            MemoryTable::derive_matrix(processor_matrix_clone);

        timer.elapsed("derive_matrix");

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

        timer.elapsed("randomizer_codewords");

        let base_codewords: Vec<Vec<BFieldElement>> = self
            .tables
            .borrow_mut()
            .get_and_set_all_base_codewords(&self.fri.domain);
        let all_base_codewords =
            vec![b_randomizer_codewords.into(), base_codewords.clone()].concat();

        timer.elapsed("get_and_set_all_base_codewords");

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

        timer.elapsed("transposed_base_codewords");

        let hasher = neptune_params();

        // Current length of each element in `transposed_base_codewords` is 18 which exceeds
        // max length of RP hash function. So we chop it into elements that will fit into the
        // rescue prime hash function. This is done by chopping the hash function input into
        // chunks of `max_length / 2` and calling `hash_many` on this input. Half the max
        // length is needed since the chunks are hashed two at a time.
        let mut base_codeword_digests_by_index: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(transposed_base_codewords.len());

        transposed_base_codewords
            .par_iter()
            .map(|values| hasher.hash(values, RP_DEFAULT_OUTPUT_SIZE))
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

        // Get coefficients for table extension
        // TODO: REPLACE THIS WITH RescuePrime/B field elements. The type of `challenges`
        // must not change though, it should remain `Vec<XFieldElement>`.
        let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT] =
            Self::sample_weights(&hasher, &seed, EXTENSION_CHALLENGE_COUNT)
                .try_into()
                .unwrap();

        timer.elapsed("sample_weights");

        let initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT] =
            XFieldElement::random_elements(PERMUTATION_ARGUMENTS_COUNT, &mut rng)
                .try_into()
                .unwrap();

        timer.elapsed("initials");

        self.tables.borrow_mut().extend(challenges, initials);

        timer.elapsed("extend");

        let extension_codewords = self
            .tables
            .borrow_mut()
            .get_and_set_all_extension_codewords(&self.fri.domain);

        timer.elapsed("get_and_set_all_extension_codewords");

        let transposed_extension_codewords: Vec<Vec<XFieldElement>> = (0..extension_codewords[0]
            .len())
            .map(|i| {
                extension_codewords
                    .iter()
                    .map(|inner| inner[i])
                    .collect::<Vec<XFieldElement>>()
            })
            .collect();

        let mut extension_codeword_digests_by_index: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(transposed_extension_codewords.len());
        transposed_extension_codewords
            .clone()
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

        let extension_degree_bounds: Vec<Degree> =
            self.tables.borrow().get_all_extension_degree_bounds();

        timer.elapsed("get_all_extension_degree_bounds");

        let terminals: [XFieldElement; TERMINAL_COUNT] = self.tables.borrow().get_terminals();

        timer.elapsed("get_terminals");

        let mut quotient_codewords =
            self.tables
                .borrow()
                .all_quotients(&self.fri.domain, challenges, terminals);

        timer.elapsed("all_quotients");

        let mut quotient_degree_bounds = self
            .tables
            .borrow()
            .all_quotient_degree_bounds(challenges, terminals);

        proof_stream.enqueue(&Item::Terminals(terminals));

        timer.elapsed("all_quotient_degree_bounds");

        // Prove equal initial values for the permutation-extension column pairs
        for pa in self.permutation_arguments.iter() {
            quotient_codewords.push(pa.quotient(&self.fri.domain));
            quotient_degree_bounds.push(pa.quotient_degree_bound());
        }

        // Calculate `num_base_polynomials` and `num_extension_polynomials` for asserting
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

        timer.elapsed("num_(base+extension)_polynomials");

        let num_randomizer_polynomials: usize = 1;
        let num_quotient_polynomials: usize = quotient_degree_bounds.len();
        let base_degree_bounds = self.tables.borrow().get_all_base_degree_bounds();

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
            let shift = (self.max_degree as Degree - bdb) as u32;
            let bc_shifted: Vec<XFieldElement> = fri_x_values
                .par_iter()
                .zip(bc.par_iter())
                .map(|(x, &bce)| (bce * x.mod_pow_u32(shift)).lift())
                .collect();

            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(&bc_shifted);
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == self.max_degree as isize,
                    "The shifted base codeword with index {} must be of maximal degree {}. Got {}.",
                    i,
                    self.max_degree,
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

        assert_eq!(
            extension_codewords.len(),
            num_extension_polynomials,
            "The number of extension columns is equal to the number extension polynomials"
        );

        for (i, (ec, edb)) in extension_codewords
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
            let shift = (self.max_degree as Degree - edb) as u32;
            let ec_shifted: Vec<XFieldElement> = fri_x_values
                .par_iter()
                .zip(ec.into_par_iter())
                .map(|(x, &ece)| ece * x.mod_pow_u32(shift).lift())
                .collect();

            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(&ec_shifted);
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == self.max_degree as isize,
                    "The shifted extension codeword with index {} must be of maximal degree {}. Got {}.",
                    i,
                    self.max_degree,
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
            let shift = (self.max_degree as Degree - qdb) as u32;
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
                        || interpolated.degree() == self.max_degree as isize,
                    "The shifted quotient codeword with index {} must be of maximal degree {}. Got {}. Predicted degree of unshifted codeword: {}. . Shift = {}",
                    _i,
                    self.max_degree,
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

        // assert_eq!(
        //     terms.len(),
        //     weights.len(),
        //     "Number of terms in non-linear combination must match number of weights"
        // );

        // Take weighted sum
        // TODO: Consider if this would go faster with some form of memoization
        // let combination_codeword: Vec<XFieldElement> = weights
        //     .par_iter()
        //     .zip(terms.par_iter())
        //     .map(|(w, t)| {
        //         (0..self.fri.domain.length)
        //             .map(|i| *w * t[i])
        //             .collect::<Vec<XFieldElement>>()
        //     })
        //     .reduce(
        //         || vec![XFieldElement::ring_zero(); self.fri.domain.length],
        //         |acc, weighted_terms| {
        //             acc.iter()
        //                 .zip(weighted_terms.iter())
        //                 .map(|(a, wt)| *a + *wt)
        //                 .collect()
        //         },
        //     );

        timer.elapsed("combination_codeword");
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
        let mut unit_distances: Vec<usize> = self
            .tables
            .borrow()
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
        let (_fri_indices, combination_root_verify) =
            self.fri.prove(&combination_codeword, &mut proof_stream)?;
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
            .collect();
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

        Ok(proof_stream)
    }

    pub fn verify(&self, proof_stream_: &mut StarkProofStream) -> Result<bool, Box<dyn Error>> {
        let mut timer = TimingReporter::start();
        let hasher = StarkHasher::new();

        // let base_merkle_tree_root: Vec<BFieldElement> =
        //     proof_stream.dequeue(SIZE_OF_RP_HASH_IN_BYTES)?;
        let base_merkle_tree_root: Vec<BFieldElement> =
            proof_stream_.dequeue()?.as_merkle_root()?;

        let seed = proof_stream_.verifier_fiat_shamir();

        timer.elapsed("verifier_fiat_shamir");
        // Get coefficients for table extension
        let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT] =
            Self::sample_weights(&hasher, &seed, EXTENSION_CHALLENGE_COUNT)
                .try_into()
                .unwrap();

        // let extension_tree_merkle_root: Vec<BFieldElement> =
        //     proof_stream.dequeue(SIZE_OF_RP_HASH_IN_BYTES)?;
        let extension_tree_merkle_root: Vec<BFieldElement> =
            proof_stream_.dequeue()?.as_merkle_root()?;

        let terminals = proof_stream_.dequeue()?.as_terminals()?;
        timer.elapsed("Read from proof stream");

        let base_degree_bounds: Vec<Degree> = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| vec![table.interpolant_degree(); table.base_width()])
            .concat();
        timer.elapsed("Calculated base degree bounds");

        let extension_degree_bounds: Vec<Degree> = self
            .tables
            .borrow()
            .into_iter()
            .map(|table| vec![table.interpolant_degree(); table.full_width() - table.base_width()])
            .concat();
        timer.elapsed("Calculated extension degree bounds");

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
        timer.elapsed("Calculated quotient degree bounds");

        let weights_seed: Vec<BFieldElement> = proof_stream_.verifier_fiat_shamir();

        timer.elapsed("verifier_fiat_shamir (again)");
        let weights_count = num_randomizer_polynomials
            + 2 * num_base_polynomials
            + 2 * num_extension_polynomials
            + 2 * num_quotient_polynomials
            + 2 * num_difference_quotients;
        let weights: Vec<XFieldElement> =
            Self::sample_weights(&hasher, &weights_seed, weights_count);
        timer.elapsed("Calculated weights");

        let combination_root: Vec<BFieldElement> = proof_stream_.dequeue()?.as_merkle_root()?;

        let indices_seed: Vec<BFieldElement> = proof_stream_.verifier_fiat_shamir();
        let indices =
            hasher.sample_indices(self.security_level, &indices_seed, self.fri.domain.length);
        timer.elapsed("Got indices");

        // Verify low degree of combination polynomial
        self.fri.verify(proof_stream_, &combination_root)?;
        timer.elapsed("Verified FRI proof");

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
        timer.elapsed("Got unit distances");

        let mut tuples: HashMap<usize, Vec<XFieldElement>> = HashMap::new();
        // TODO: we can store the elements mushed into "tuples" separately, like in "points" below,
        // to avoid unmushing later

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
        timer.elapsed("Calculated revealed indices");

        let revealed_base_elements: Vec<Vec<BFieldElement>> = proof_stream_
            .dequeue()?
            .as_transposed_base_element_vectors()?;
        let auth_paths: Vec<PartialAuthenticationPath<Vec<BFieldElement>>> = proof_stream_
            .dequeue()?
            .as_compressed_authentication_paths()?;
        timer.elapsed("Read base elements and auth paths from proof stream");
        let leaf_digests: Vec<Vec<BFieldElement>> = revealed_base_elements
            .par_iter()
            .map(|re| hasher.hash(re, RP_DEFAULT_OUTPUT_SIZE))
            .collect();
        timer.elapsed(&format!(
            "Calculated {} leaf digests for base elements",
            indices.len()
        ));
        let mt_base_success = MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            base_merkle_tree_root,
            &revealed_indices,
            &leaf_digests,
            &auth_paths,
        );
        if !mt_base_success {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for base codeword");
            // return Ok(false);
        }
        timer.elapsed(&format!(
            "Verified authentication paths for {} base elements",
            indices.len()
        ));

        // Get extension elements
        let revealed_extension_elements: Vec<Vec<XFieldElement>> = proof_stream_
            .dequeue()?
            .as_transposed_extension_element_vectors()?;
        let extension_auth_paths = proof_stream_
            .dequeue()?
            .as_compressed_authentication_paths()?;
        timer.elapsed("Read extension elements and auth paths from proof stream");
        let extension_leaf_digests: Vec<Vec<BFieldElement>> = revealed_extension_elements
            .clone()
            .into_par_iter()
            .map(|xvalues| {
                let bvalues: Vec<BFieldElement> = xvalues
                    .into_iter()
                    .map(|x| x.coefficients.clone().to_vec())
                    .concat();
                debug_assert_eq!(
                    27,
                    bvalues.len(),
                    "9 X-field elements must become 27 B-field elements"
                );
                hasher.hash(&bvalues, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect();
        timer.elapsed(&format!(
            "Calculated {} leaf digests for extension elements",
            indices.len()
        ));
        let mt_extension_success = MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            extension_tree_merkle_root,
            &revealed_indices,
            &extension_leaf_digests,
            &extension_auth_paths,
        );
        if !mt_extension_success {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for extension codeword");
            // return Ok(false);
        }
        timer.elapsed(&format!(
            "Verified authentication paths for {} extension elements",
            indices.len()
        ));

        // Collect values in a hash map
        for (i, &idx) in revealed_indices.iter().enumerate() {
            let randomizer: XFieldElement = XFieldElement::new([
                revealed_base_elements[i][0],
                revealed_base_elements[i][1],
                revealed_base_elements[i][2],
            ]);
            debug_assert_eq!(
                1, num_randomizer_polynomials,
                "For now number of randomizers must be 1"
            );
            let mut values: Vec<XFieldElement> = vec![randomizer];
            values.extend_from_slice(
                &revealed_base_elements[i]
                    .iter()
                    .skip(3 * num_randomizer_polynomials)
                    .map(|bfe| bfe.lift())
                    .collect::<Vec<XFieldElement>>(),
            );
            tuples.insert(idx, values);
            tuples.insert(
                idx,
                vec![tuples[&idx].clone(), revealed_extension_elements[i].clone()].concat(),
            );
        }
        timer.elapsed(&format!(
            "Collected {} values into a hash map",
            indices.len()
        ));

        // Verify Merkle authentication path for combination elements
        let revealed_combination_elements: Vec<XFieldElement> = proof_stream_
            .dequeue()?
            .as_revealed_combination_elements()?;
        let revealed_combination_digests: Vec<Vec<BFieldElement>> = revealed_combination_elements
            .clone()
            .into_par_iter()
            .map(|xfe| {
                let b_elements: Vec<BFieldElement> = xfe.to_digest();
                hasher.hash(&b_elements, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect();
        let revealed_combination_auth_paths: Vec<PartialAuthenticationPath<Vec<BFieldElement>>> =
            proof_stream_
                .dequeue()?
                .as_compressed_authentication_paths()?;
        let mt_combination_success = MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            combination_root.clone(),
            &indices,
            &revealed_combination_digests,
            &revealed_combination_auth_paths,
        );
        if !mt_combination_success {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for combination codeword");
            // return Ok(false);
        }
        timer.elapsed(&format!(
            "Verified combination authentication paths for {} indices",
            indices.len()
        ));

        // verify nonlinear combination
        for (i, &index) in indices.iter().enumerate() {
            let b_domain_value = self.fri.domain.b_domain_value(index as u32);
            // collect terms: randomizer
            let mut terms: Vec<XFieldElement> = (0..num_randomizer_polynomials)
                .map(|j| tuples[&index][j])
                .collect();

            // collect terms: base
            for j in num_randomizer_polynomials..num_randomizer_polynomials + num_base_polynomials {
                terms.push(tuples[&index][j]);
                let shift: u32 = (self.max_degree as i64
                    - base_degree_bounds[j - num_randomizer_polynomials])
                    as u32;
                terms.push(tuples[&index][j] * b_domain_value.mod_pow_u32(shift).lift());
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
            for (j, edb) in extension_degree_bounds.iter().enumerate() {
                let extension_element: XFieldElement = tuples[&index][extension_offset + j];
                terms.push(extension_element);
                let shift = (self.max_degree as i64 - edb) as u32;
                terms.push(extension_element * b_domain_value.mod_pow_u32(shift).lift())
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
                    let quotient = eval / (b_domain_value.lift() - XFieldElement::ring_one());
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(quotient * b_domain_value.mod_pow_u32(shift).lift());
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
                        let num = (b_domain_value - table.omicron().inverse()).lift();
                        let denom = b_domain_value.mod_pow_u32(table.height() as u32).lift()
                            - XFieldElement::ring_one();
                        eval * num / denom
                    };
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(quotient * b_domain_value.mod_pow_u32(shift).lift());
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
                    let quotient =
                        eval / (b_domain_value.lift() - table.omicron().inverse().lift());
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(quotient * b_domain_value.mod_pow_u32(shift).lift())
                }
            }

            for arg in self.permutation_arguments.iter() {
                let quotient = arg.evaluate_difference(&points)
                    / (b_domain_value.lift() - XFieldElement::ring_one());
                terms.push(quotient);
                let degree_bound = arg.quotient_degree_bound();
                let shift = (self.max_degree as i64 - degree_bound) as u32;
                terms.push(quotient * b_domain_value.mod_pow_u32(shift).lift());
            }

            assert_eq!(
                weights.len(),
                terms.len(),
                "length of terms must be equal to length of weights"
            );

            // compute inner product of weights and terms
            // Todo: implement `sum` on XFieldElements
            let inner_product = weights
                .par_iter()
                .zip(terms.par_iter())
                .map(|(w, t)| *w * *t)
                .reduce(XFieldElement::ring_zero, |x, y| x + y);

            assert_eq!(
                revealed_combination_elements[i], inner_product,
                "The combination leaf must equal the inner product"
            );
        }
        timer.elapsed(&format!(
            "Verified {} non-linear combinations",
            indices.len()
        ));

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
        timer.elapsed("Verified terminals");

        let report = timer.finish();
        println!("{}", report);

        Ok(true)
    }
}

#[cfg(test)]
mod brainfuck_stark_tests {
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::stark::brainfuck;
    use crate::shared_math::stark::brainfuck::vm::{
        BaseMatrices, InstructionMatrixBaseRow, Register,
    };
    use crate::shared_math::traits::IdentityValues;

    pub fn new_test_stark(
        trace_length: usize,
        source_code: String,
        input_symbols: Vec<BFieldElement>,
        output_symbols: Vec<BFieldElement>,
    ) -> Stark {
        // These parameters are too low for security, but work for testing correctness
        let log_expansion_factor = 2;
        let security_level = 2;

        Stark::new(
            trace_length,
            source_code,
            input_symbols,
            output_symbols,
            log_expansion_factor,
            security_level,
        )
    }

    fn mallorys_simulate(
        program: &[BFieldElement],
        input_symbols: &[BFieldElement],
    ) -> Option<BaseMatrices> {
        let zero = BFieldElement::ring_zero();
        let one = BFieldElement::ring_one();
        let two = BFieldElement::new(2);
        let mut register = Register::default();
        register.current_instruction = program[0];
        if program.len() < 2 {
            register.next_instruction = zero;
        } else {
            register.next_instruction = program[1];
        }

        let mut memory: HashMap<BFieldElement, BFieldElement> = HashMap::new();
        let mut input_counter: usize = 0;

        // Prepare tables. For '++[>++<-]' this would give:
        // 0 + +
        // 1 + [
        // 2 [ >
        // 3 > +
        // ...
        let mut base_matrices = BaseMatrices::default();
        for i in 0..program.len() - 1 {
            base_matrices
                .instruction_matrix
                .push(InstructionMatrixBaseRow {
                    instruction_pointer: BFieldElement::new(i as u64),
                    current_instruction: program[i],
                    next_instruction: program[i + 1],
                });
        }
        base_matrices
            .instruction_matrix
            .push(InstructionMatrixBaseRow {
                instruction_pointer: BFieldElement::new((program.len() - 1) as u64),
                current_instruction: *program.last().unwrap(),
                next_instruction: zero,
            });

        // main loop
        while (register.instruction_pointer.value() as usize) < program.len() {
            // collect values to add new rows in execution matrices
            base_matrices.processor_matrix.push(register.clone());
            base_matrices
                .instruction_matrix
                .push(InstructionMatrixBaseRow {
                    instruction_pointer: register.instruction_pointer,
                    current_instruction: register.current_instruction,
                    next_instruction: register.next_instruction,
                });

            // update pointer registers according to instruction
            if register.current_instruction == BFieldElement::new('[' as u64) {
                // This is the 1st part of the attack, a loop is *always* entered
                register.instruction_pointer += two;

                // Original version is commented out below
                // if register.memory_value.is_zero() {
                //     register.instruction_pointer =
                //         program[register.instruction_pointer.value() as usize + 1];
                // } else {
                //     register.instruction_pointer += two;
                // }
            } else if register.current_instruction == BFieldElement::new(']' as u64) {
                if !register.memory_value.is_zero() {
                    register.instruction_pointer =
                        program[register.instruction_pointer.value() as usize + 1];
                } else {
                    register.instruction_pointer += two;
                }
            } else if register.current_instruction == BFieldElement::new('<' as u64) {
                register.instruction_pointer += one;
                register.memory_pointer -= one;
            } else if register.current_instruction == BFieldElement::new('>' as u64) {
                register.instruction_pointer += one;
                register.memory_pointer += one;
            } else if register.current_instruction == BFieldElement::new('+' as u64) {
                register.instruction_pointer += one;
                memory.insert(
                    register.memory_pointer,
                    *memory.get(&register.memory_pointer).unwrap_or(&zero) + one,
                );
            } else if register.current_instruction == BFieldElement::new('-' as u64) {
                register.instruction_pointer += one;
                memory.insert(
                    register.memory_pointer,
                    *memory.get(&register.memory_pointer).unwrap_or(&zero) - one,
                );
            } else if register.current_instruction == BFieldElement::new('.' as u64) {
                register.instruction_pointer += one;
                base_matrices
                    .output_matrix
                    .push(*memory.get(&register.memory_pointer).unwrap_or(&zero));
            } else if register.current_instruction == BFieldElement::new(',' as u64) {
                register.instruction_pointer += one;
                let input_char = input_symbols[input_counter];
                input_counter += 1;
                memory.insert(register.memory_pointer, input_char);
                base_matrices.input_matrix.push(input_char);
            } else {
                return None;
            }

            // update non-pointer registers
            register.cycle += one;

            if (register.instruction_pointer.value() as usize) < program.len() {
                register.current_instruction =
                    program[register.instruction_pointer.value() as usize];
            } else {
                register.current_instruction = zero;
            }

            if (register.instruction_pointer.value() as usize) < program.len() - 1 {
                register.next_instruction =
                    program[(register.instruction_pointer.value() as usize) + 1];
            } else {
                register.next_instruction = zero;
            }

            register.memory_value = *memory.get(&register.memory_pointer).unwrap_or(&zero);
            register.memory_value_inverse = if register.memory_value.is_zero() {
                zero
            } else {
                register.memory_value.inverse()
            };

            // This is the 2nd part of the attack
            if register.current_instruction == BFieldElement::new('[' as u64) {
                register.memory_value_inverse = BFieldElement::new(42);
            }
        }

        base_matrices.processor_matrix.push(register.clone());
        base_matrices
            .instruction_matrix
            .push(InstructionMatrixBaseRow {
                instruction_pointer: register.instruction_pointer,
                current_instruction: register.current_instruction,
                next_instruction: register.next_instruction,
            });

        // post-process context tables
        // sort by instruction address
        base_matrices
            .instruction_matrix
            .sort_by_key(|row| row.instruction_pointer.value());

        Some(base_matrices)
    }

    #[test]
    fn set_adversarial_is_zero_value_test() {
        // Expected (honest) output state:
        // `memory_pointer`
        //        |
        //        |
        //        V
        //    [1, 0] with cycle = 3
        // Malory's output state:
        // `memory_pointer`
        //     |
        //     |
        //     V
        //    [0, 2] with cycle = 8

        // Run honest execution, verify that it succeeds in prover/verifier
        let source_code = "+>[++<-]".to_string();
        let program: Vec<BFieldElement> = brainfuck::vm::compile(&source_code).unwrap();
        let input_symbols: Vec<BFieldElement> = vec![];
        let regular_matrices: BaseMatrices =
            brainfuck::vm::simulate(&program, &input_symbols).unwrap();
        let mut regular_stark = new_test_stark(
            regular_matrices.processor_matrix.len(),
            source_code.clone(),
            input_symbols.clone(),
            vec![],
        );
        let mut regular_proof_stream: StarkProofStream =
            regular_stark.prove(regular_matrices).unwrap();
        let regular_verify = regular_stark.verify(&mut regular_proof_stream);
        assert!(regular_verify.unwrap(), "Regular execution must succeed");

        // Run attack, verify that it is caught by the verifier
        let mallorys_matrices: BaseMatrices = mallorys_simulate(&program, &input_symbols).unwrap();
        let mut mallorys_stark = new_test_stark(
            mallorys_matrices.processor_matrix.len(),
            source_code,
            input_symbols,
            vec![],
        );
        let mut mallorys_proof_stream: StarkProofStream =
            mallorys_stark.prove(mallorys_matrices).unwrap();

        let mallorys_verify = mallorys_stark.verify(&mut mallorys_proof_stream);
        match mallorys_verify {
            Ok(true) => {
                panic!("Attack passes the STARK verifier!!")
            }
            _ => (),
        }
    }

    #[test]
    fn bf_prove_verify_test() {
        for source_code in [
            brainfuck::vm::sample_programs::VERY_SIMPLE_PROGRAM,
            brainfuck::vm::sample_programs::TWO_BY_TWO_THEN_OUTPUT,
            brainfuck::vm::sample_programs::SHORT_INPUT_AND_OUTPUT,
            brainfuck::vm::sample_programs::PRINT_17_CHARS,
        ] {
            let program: Vec<BFieldElement> = brainfuck::vm::compile(source_code).unwrap();
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
            let mut stark = new_test_stark(
                trace_length,
                source_code.to_string(),
                input_symbols,
                output_symbols,
            );

            // TODO: If we set the `DEBUG` environment variable here, we *should* catch a lot of bugs.
            // Do we want to do that?
            let mut proof_stream = stark.prove(base_matrices).unwrap();

            let verifier_verdict: Result<bool, Box<dyn Error>> = stark.verify(&mut proof_stream);
            match verifier_verdict {
                Ok(_) => (),
                Err(err) => panic!("error in STARK verifier: {}", err),
            };
        }
    }
}
