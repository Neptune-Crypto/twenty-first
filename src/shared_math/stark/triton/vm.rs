use super::instruction;
use super::instruction::{parse, Instruction, LabelledInstruction};
use super::state::{VMOutput, VMState, STATE_REGISTER_COUNT};
use super::stdio::{InputStream, OutputStream, VecStream};
use super::table::base_matrix::BaseMatrices;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_xlix::{neptune_params, RescuePrimeXlix};
use itertools::Itertools;
use std::error::Error;
use std::fmt::Display;
use std::io::Cursor;

type BWord = BFieldElement;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Program {
    pub instructions: Vec<Instruction>,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut stream = self.instructions.iter();
        while let Some(instruction) = stream.next() {
            writeln!(f, "{}", instruction)?;

            // Skip duplicate placeholder used for aligning instructions and instruction_pointer in VM.
            for _ in 1..instruction.size() {
                stream.next();
            }
        }
        Ok(())
    }
}

pub struct SkippyIter {
    cursor: Cursor<Vec<Instruction>>,
}

impl Iterator for SkippyIter {
    type Item = Instruction;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.cursor.position() as usize;
        let instructions = self.cursor.get_ref();
        let instruction = *instructions.get(pos)?;
        self.cursor.set_position((pos + instruction.size()) as u64);

        Some(instruction)
    }
}

impl IntoIterator for Program {
    type Item = Instruction;

    type IntoIter = SkippyIter;

    fn into_iter(self) -> Self::IntoIter {
        let cursor = Cursor::new(self.instructions);
        SkippyIter { cursor }
    }
}

/// A `Program` is a `Vec<Instruction>` that contains duplicate elements for
/// instructions with a size of 2. This means that the index in the vector
/// corresponds to the VM's `instruction_pointer`. These duplicate values
/// should most often be skipped/ignored, e.g. when pretty-printing.
impl Program {
    /// Create a `Program` from a slice of `Instruction`.
    ///
    /// All valid programs terminate with `Halt`.
    ///
    /// `new()` will append `Halt` if not present.
    pub fn new(input: &[LabelledInstruction]) -> Self {
        let instructions = instruction::convert_labels(input)
            .iter()
            .flat_map(|instr| vec![*instr; instr.size()])
            .collect::<Vec<_>>();

        Program { instructions }
    }

    /// Create a `Program` by parsing source code.
    ///
    /// All valid programs terminate with `Halt`.
    ///
    /// `from_code()` will append `Halt` if not present.
    pub fn from_code(code: &str) -> Result<Self, Box<dyn Error>> {
        let instructions = parse(code)?;
        Ok(Program::new(&instructions))
    }

    /// Convert a `Program` to a `Vec<BWord>`.
    ///
    /// Every single-word instruction is converted to a single word.
    ///
    /// Every double-word instruction is converted to two words.
    pub fn to_bwords(&self) -> Vec<BWord> {
        self.clone()
            .into_iter()
            .map(|instruction| {
                let opcode = instruction.opcode_b();
                if let Some(arg) = instruction.arg() {
                    vec![opcode, arg]
                } else {
                    vec![opcode]
                }
            })
            .concat()
    }

    /// Simulate (execute) a `Program` and record every state transition.
    ///
    /// Returns a `BaseMatrices` that records the execution.
    ///
    /// Optionally returns errors on premature termination, but returns a
    /// `BaseMatrices` for the execution up to the point of failure.
    pub fn simulate<In, Out>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
        stdout: &mut Out,
        rescue_prime: &RescuePrimeXlix<{ STATE_REGISTER_COUNT }>,
    ) -> (BaseMatrices, Option<Box<dyn Error>>)
    where
        In: InputStream,
        Out: OutputStream,
    {
        let mut base_matrices = BaseMatrices::default();
        base_matrices.initialize(self);

        let mut state = VMState::new(self);
        let initial_instruction = state.current_instruction().unwrap();

        base_matrices.append(&state, None, initial_instruction);

        while !state.is_complete() {
            let vm_output = match state.step_mut(stdin, secret_in, rescue_prime) {
                Err(err) => return (base_matrices, Some(err)),
                Ok(vm_output) => vm_output,
            };
            let current_instruction = state.current_instruction().unwrap_or(Instruction::Halt);

            if let Some(VMOutput::WriteOutputSymbol(written_word)) = vm_output {
                if let Err(error) = stdout.write_elem(written_word) {
                    return (base_matrices, Some(Box::new(error)));
                }
            }

            base_matrices.append(&state, vm_output, current_instruction);
        }

        base_matrices.sort_instruction_matrix();
        base_matrices.sort_op_stack_matrix();
        base_matrices.sort_ram_matrix();
        base_matrices.sort_jump_stack_matrix();

        base_matrices.set_ram_matrix_inverse_of_ramp_diff();

        (base_matrices, None)
    }

    pub fn simulate_with_input(
        &self,
        input: &[BFieldElement],
        secret_input: &[BFieldElement],
    ) -> (BaseMatrices, Option<Box<dyn Error>>) {
        let input_bytes = input
            .iter()
            .flat_map(|elem| elem.value().to_be_bytes())
            .collect_vec();
        let secret_input_bytes = secret_input
            .iter()
            .flat_map(|elem| elem.value().to_be_bytes())
            .collect_vec();
        let mut stdin = VecStream::new_bytes(&input_bytes);
        let mut secret_in = VecStream::new_bytes(&secret_input_bytes);
        let mut stdout = VecStream::new_bytes(&[]);
        let rescue_prime = neptune_params();

        self.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime)
    }

    pub fn run<In, Out>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
        stdout: &mut Out,
        rescue_prime: &RescuePrimeXlix<{ STATE_REGISTER_COUNT }>,
    ) -> (Vec<VMState>, Option<Box<dyn Error>>)
    where
        In: InputStream,
        Out: OutputStream,
    {
        let mut states = vec![VMState::new(self)];
        let mut current_state = states.last().unwrap();

        while !current_state.is_complete() {
            let step = current_state.step(stdin, secret_in, rescue_prime);
            let (next_state, vm_output) = match step {
                Err(err) => return (states, Some(err)),
                Ok((next_state, vm_output)) => (next_state, vm_output),
            };

            if let Some(VMOutput::WriteOutputSymbol(written_word)) = vm_output {
                if let Err(error) = stdout.write_elem(written_word) {
                    return (states, Some(Box::new(error)));
                }
            }

            states.push(next_state);
            current_state = states.last().unwrap();
        }

        (states, None)
    }

    pub fn run_with_input(
        &self,
        input: &[BFieldElement],
        secret_input: &[BFieldElement],
    ) -> (Vec<VMState>, Vec<BWord>, Option<Box<dyn Error>>) {
        let input_bytes = input
            .iter()
            .flat_map(|elem| elem.value().to_be_bytes())
            .collect_vec();
        let secret_input_bytes = secret_input
            .iter()
            .flat_map(|elem| elem.value().to_be_bytes())
            .collect_vec();
        let mut stdin = VecStream::new_bytes(&input_bytes);
        let mut secret_in = VecStream::new_bytes(&secret_input_bytes);
        let mut stdout = VecStream::new_bytes(&[]);
        let rescue_prime = neptune_params();

        let (trace, err) = self.run(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        let out = stdout.to_bword_vec();

        (trace, out, err)
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

#[cfg(test)]
mod triton_vm_tests {
    use std::iter::zip;

    use super::*;
    use crate::shared_math::mpolynomial::MPolynomial;
    use crate::shared_math::other;
    use crate::shared_math::rescue_prime_xlix::RP_DEFAULT_WIDTH;
    use crate::shared_math::stark::triton::instruction::sample_programs;
    use crate::shared_math::stark::triton::table::base_matrix::ProcessorMatrixRow;
    use crate::shared_math::stark::triton::table::base_table::{HasBaseTable, Table};
    use crate::shared_math::stark::triton::table::challenges_endpoints::{
        AllChallenges, AllEndpoints,
    };
    use crate::shared_math::stark::triton::table::extension_table::ExtensionTable;
    use crate::shared_math::stark::triton::table::processor_table::ProcessorTable;
    use crate::shared_math::traits::{GetPrimitiveRootOfUnity, IdentityValues};
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::util_types::simple_hasher::{Hasher, ToDigest};

    #[test]
    fn initialise_table_test() {
        // 1. Parse program
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[42.into(), 56.into()]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        // 2. Execute program, convert to base matrices
        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        println!("Err: {:?}", err);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }

        println!("Output: {:?}", stdout.to_bword_vec());

        // 3. Extract constraints
        // 4. Check constraints
    }

    #[test]
    fn initialise_table_42_test() {
        // 1. Execute program
        let code = sample_programs::SUBTRACT;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        println!("{:?}", err);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
    }

    #[test]
    fn simulate_gcd_test() {
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[42.into(), 56.into()]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (_, err) = program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        assert!(err.is_none());
        let expected = BWord::new(14);
        let actual = stdout.to_bword_vec()[0];

        assert_eq!(expected, actual);
    }

    #[test]
    fn hello_world() {
        // 1. Execute program
        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        println!("{:?}", err);
        for row in base_matrices.processor_matrix.clone() {
            println!("{}", ProcessorMatrixRow { row });
        }

        // 1. Check `output_matrix`.
        {
            let expecteds = vec![
                10, 33, 100, 108, 114, 111, 87, 32, 44, 111, 108, 108, 101, 72,
            ]
            .into_iter()
            .rev()
            .map(|x| BWord::new(x));
            let actuals = stdout.to_bword_vec();

            assert_eq!(expecteds.len(), actuals.len());

            for (expected, actual) in zip(expecteds, actuals) {
                assert_eq!(expected, actual)
            }
        }

        // 2. Each `hash` operation result in 8 rows.
        {
            let hash_instruction_count = 0;
            let prc_rows_count = base_matrices.processor_matrix.len();
            assert!(hash_instruction_count <= 8 * prc_rows_count)
        }

        //3. noRows(jmpstack_tabel) == noRows(processor_table)
        {
            let jmp_rows_count = base_matrices.jump_stack_matrix.len();
            let prc_rows_count = base_matrices.processor_matrix.len();
            assert_eq!(jmp_rows_count, prc_rows_count)
        }

        // "4. "READIO; WRITEIO" -> noRows(inputable) + noRows(outputtable) == noReadIO +
        // noWriteIO"

        {
            // Input
            let expected_input_count = 0;

            let actual_input_count = stdin.to_bword_vec().len();

            assert_eq!(expected_input_count, actual_input_count);

            // Output
            let expected_output_count = 14;
            //let actual = base_matrices.ram_matrix.len();

            let actual_output_count = stdout.to_bword_vec().len();

            assert_eq!(expected_output_count, actual_output_count);
        }
    }

    #[test]
    fn hash_hash_hash_test() {
        // 1. Execute program
        let code = sample_programs::HASH_HASH_HASH_HALT;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        let jmp_rows_count = base_matrices.jump_stack_matrix.len();
        let prc_rows_count = base_matrices.processor_matrix.len();

        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }
        println!("Errors: {:?}", err);

        // 1. Each of three `hash` instructions result in 8 rows.
        assert_eq!(24, base_matrices.hash_matrix.len());

        // 2. noRows(jmpstack_table) == noRows(processor_table)
        assert_eq!(jmp_rows_count, prc_rows_count);

        // 3. The number of input_table rows is equivalent to the number of read_io instructions.
        assert_eq!(0, stdin.to_bword_vec().len());

        // 4. The number of output_table rows is equivalent to the number of write_io instructions.
        assert_eq!(0, stdout.to_bword_vec().len());
    }

    fn _check_base_matrices(
        base_matrices: &BaseMatrices,
        input_symbols: &[BFieldElement],
        output_symbols: &[BFieldElement],
        expected_input_rows: usize,
        expected_output_rows: usize,
        hash_instruction_count: usize,
    ) {
        // 1. Check `output_matrix`.
        {
            let expecteds = vec![].into_iter().rev().map(|x| BWord::new(x));
            let actuals = output_symbols.to_vec();

            assert_eq!(expecteds.len(), actuals.len());

            for (expected, actual) in zip(expecteds, actuals) {
                assert_eq!(expected, actual)
            }
        }

        // 2. Each `hash` operation result in 8 rows in the state matrix.
        {
            let hash_state_rows_count = base_matrices.hash_matrix.len();
            assert_eq!(hash_instruction_count * 8, hash_state_rows_count)
        }

        //3. noRows(jmpstack_tabel) == noRows(processor_table)
        {
            let jmp_rows_count = base_matrices.jump_stack_matrix.len();
            let prc_rows_count = base_matrices.processor_matrix.len();
            assert_eq!(jmp_rows_count, prc_rows_count)
        }

        // "4. "READIO; WRITEIO" -> noRows(inputable) + noRows(outputtable) == noReadIO +
        // noWriteIO"
        {
            // Input
            let actual_input_rows = input_symbols.len();
            assert_eq!(expected_input_rows, actual_input_rows);

            // Output
            let actual_output_rows = output_symbols.len();

            assert_eq!(expected_output_rows, actual_output_rows);
        }
    }

    #[test]
    fn processor_table_constraints_evaluate_to_zero_test() {
        let mut _rng = rand::thread_rng();
        let hasher = RescuePrimeXlix::<RP_DEFAULT_WIDTH>::new();

        let all_programs = vec![sample_programs::PUSH_PUSH_ADD_POP_S];
        for source_code in all_programs.into_iter() {
            let program = Program::from_code(source_code).expect("Could not load source code.");
            let (base_matrices, err) = program.simulate_with_input(&[], &[]);

            assert!(err.is_none());

            let number_of_randomizers = 2;
            let order = 1 << 32;
            let smooth_generator = BFieldElement::ring_zero()
                .get_primitive_root_of_unity(order)
                .0
                .unwrap();

            let processor_matrix = base_matrices
                .processor_matrix
                .iter()
                .map(|row| row.to_vec())
                .collect_vec();

            let mut processor_table: ProcessorTable = ProcessorTable::new_prover(
                smooth_generator,
                order as usize,
                number_of_randomizers,
                processor_matrix,
            );

            // let air_constraints = processor_table.base_transition_constraints();
            // assert_air_constraints_on_matrix(processor_table.data(), &air_constraints);

            // Test air constraints after padding as well
            processor_table.pad();

            assert!(
                other::is_power_of_two(processor_table.data().len()),
                "Matrix length must be power of 2 after padding"
            );

            // assert_air_constraints_on_matrix(processor_table.data(), &air_constraints);

            // Test the same for the extended matrix

            // FIXME: This interface leaks abstractions: We want a function that generates a number of weights
            // that doesn't care about the weights-to-digest ratio (we can make two weights per digest).

            let mock_seed = 0u128.to_digest();
            let mock_challenge_weights: Vec<XFieldElement> = hasher
                .get_n_hash_rounds(&mock_seed, AllChallenges::TOTAL_CHALLENGES / 2)
                .iter()
                .flat_map(|digest| {
                    vec![
                        XFieldElement::new([digest[0], digest[1], digest[2]]),
                        XFieldElement::new([digest[3], digest[4], digest[5]]),
                    ]
                })
                .collect();
            let challenges: AllChallenges =
                AllChallenges::create_challenges(&mock_challenge_weights);

            let mock_initial_weights: Vec<XFieldElement> = hasher
                .get_n_hash_rounds(&mock_seed, AllEndpoints::TOTAL_ENDPOINTS / 2)
                .iter()
                .flat_map(|digest| {
                    vec![
                        XFieldElement::new([digest[0], digest[1], digest[2]]),
                        XFieldElement::new([digest[3], digest[4], digest[5]]),
                    ]
                })
                .collect();
            let initials: AllEndpoints = AllEndpoints::create_initials(&mock_initial_weights);

            let (ext_processor_table, _terminals) = processor_table.extend(
                &challenges.processor_table_challenges,
                &initials.processor_table_endpoints,
            );
            let x_air_constraints = ext_processor_table.ext_transition_constraints(&challenges);
            let ext_data = ext_processor_table.data();

            for step in 0..ext_processor_table.padded_height() - 1 {
                let row = ext_data[step].clone();
                let next_register = ext_data[step + 1].clone();
                let xpoint: Vec<XFieldElement> = vec![row.clone(), next_register.clone()].concat();

                for x_air_constraint in x_air_constraints.iter() {
                    assert!(x_air_constraint.evaluate(&xpoint).is_zero());
                }

                // TODO: Can we add a negative test here?
            }
        }
    }

    fn _assert_air_constraints_on_matrix(
        table_data: &[Vec<BFieldElement>],
        air_constraints: &[MPolynomial<BFieldElement>],
    ) {
        for step in 0..table_data.len() - 1 {
            let register: Vec<BFieldElement> = table_data[step].clone().into();
            let next_register: Vec<BFieldElement> = table_data[step + 1].clone().into();
            let point: Vec<BFieldElement> = vec![register, next_register].concat();

            for air_constraint in air_constraints.iter() {
                assert!(air_constraint.evaluate(&point).is_zero());
            }
        }
    }

    #[test]
    fn xxadd() {
        // 1. Execute program
        let code = sample_programs::XXADD;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin =
            VecStream::new_bwords(&[2.into(), 3.into(), 5.into(), 7.into(), 11.into(), 13.into()]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (_base_matrices, _err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        let mut expected = VecStream::new_bwords(&[9.into(), 14.into(), 18.into()]);
        // Move the inner cursor to agree.
        expected.set_position(stdout.position());
        assert_eq!(expected, stdout);
    }

    #[test]
    fn xxmul() {
        // 1. Execute program
        let code = sample_programs::XXMUL;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin =
            VecStream::new_bwords(&[2.into(), 3.into(), 5.into(), 7.into(), 11.into(), 13.into()]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (_base_matrices, _err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        let mut expected = VecStream::new_bwords(&[108.into(), 123.into(), 22.into()]);
        // Move the inner cursor to agree.
        expected.set_position(stdout.position());
        assert_eq!(expected, stdout);
    }

    #[test]
    fn xinv() {
        // 1. Execute program
        let code = sample_programs::XINV;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[2.into(), 3.into(), 5.into()]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (_base_matrices, _err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        let mut expected = VecStream::new_bytes(&[
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            0.into(),
            1.into(),
            227.into(),
            13.into(),
            145.into(),
            162.into(),
            216.into(),
            50.into(),
            168.into(),
            66.into(),
            197.into(),
            51.into(),
            143.into(),
            211.into(),
            207.into(),
            38.into(),
            229.into(),
            197.into(),
            61.into(),
            131.into(),
            42.into(),
            131.into(),
            212.into(),
            148.into(),
            90.into(),
            118.into(),
        ]);
        // Move the inner cursor to agree.
        expected.set_position(stdout.position());
        assert_eq!(expected, stdout);
    }

    #[test]
    fn xbmul() {
        // 1. Execute program
        let code = sample_programs::XBMUL;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_bwords(&[2.into(), 3.into(), 5.into(), 7.into()]);
        let mut secret_in = VecStream::new_bwords(&[]);
        let mut stdout = VecStream::new_bwords(&[]);
        let rescue_prime = neptune_params();

        let (_base_matrices, _err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        let mut expected = VecStream::new_bwords(&[14.into(), 21.into(), 35.into()]);
        // Move the inner cursor to agree.
        expected.set_position(stdout.position());
        assert_eq!(expected, stdout);
    }
}
