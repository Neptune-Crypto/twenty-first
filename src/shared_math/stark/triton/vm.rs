use super::instruction;
use super::instruction::{parse, Instruction, LabelledInstruction};
use super::state::{VMOutput, VMState, AUX_REGISTER_COUNT};
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
        rescue_prime: &RescuePrimeXlix<{ AUX_REGISTER_COUNT }>,
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

            if let Some(VMOutput::WriteIoTrace(written_word)) = &vm_output {
                let _written = stdout.write_elem(*written_word);
            }

            base_matrices.append(&state, vm_output, current_instruction);
        }

        base_matrices.sort_instruction_matrix();
        base_matrices.sort_jump_stack_matrix();

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
        let mut stdin = VecStream::new(&input_bytes);
        let mut secret_in = VecStream::new(&secret_input_bytes);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        self.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime)
    }

    pub fn run<In, Out>(
        &self,
        stdin: &mut In,
        secret_in: &mut In,
        stdout: &mut Out,
        rescue_prime: &RescuePrimeXlix<{ AUX_REGISTER_COUNT }>,
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

            if let Some(VMOutput::WriteIoTrace(written_word)) = vm_output {
                let _written = stdout.write_elem(written_word);
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
        let mut stdin = VecStream::new(&input_bytes);
        let mut secret_in = VecStream::new(&secret_input_bytes);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        let (trace, err) = self.run(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        const U64SIZE: usize = std::mem::size_of::<u64>();
        let out = stdout
            .to_vec()
            .chunks_exact(U64SIZE)
            .map(|chunk: &[u8]| -> &[u8; 8] {
                chunk.try_into().expect("Chunks must have length 8.")
            }) // force compatible type
            .map(|&chunk| BFieldElement::new(u64::from_be_bytes(chunk)))
            .collect_vec();

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
    use crate::shared_math::stark::triton::instruction::sample_programs;
    use crate::shared_math::stark::triton::stark::{
        EXTENSION_CHALLENGE_COUNT, PERMUTATION_ARGUMENTS_COUNT,
    };
    use crate::shared_math::stark::triton::table::base_matrix::ProcessorMatrixRow;
    use crate::shared_math::stark::triton::table::base_table::{HasBaseTable, Table};
    use crate::shared_math::stark::triton::table::extension_table::ExtensionTable;
    use crate::shared_math::stark::triton::table::processor_table::ProcessorTable;
    use crate::shared_math::traits::GetRandomElements;
    use crate::shared_math::traits::{GetPrimitiveRootOfUnity, IdentityValues};
    use crate::shared_math::x_field_element::XFieldElement;

    #[test]
    fn initialise_table_test() {
        // 1. Parse program
        let code = sample_programs::GCD_X_Y;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new_b(&[42.into(), 56.into()]);
        let mut secret_in = VecStream::new_b(&[]);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        // 2. Execute program, convert to base matrices
        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        println!("Err: {:?}", err);
        for row in base_matrices.processor_matrix {
            println!("{}", ProcessorMatrixRow { row });
        }

        println!("{:?}", base_matrices.output_matrix);

        // 3. Extract constraints
        // 4. Check constraints
    }

    #[test]
    fn initialise_table_42_test() {
        // 1. Execute program
        let code = sample_programs::SUBTRACT;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new(&[]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);
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

        let mut stdin = VecStream::new_b(&[42.into(), 56.into()]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        assert!(err.is_none());
        let expected = BWord::new(14);
        let actual = base_matrices.output_matrix[0][0];

        assert_eq!(expected, actual);
    }

    #[test]
    fn hello_world() {
        // 1. Execute program
        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new(&[]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);
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
            let actuals: Vec<BWord> = base_matrices
                .output_matrix
                .iter()
                .map(|&[val]| val)
                .collect_vec();

            assert_eq!(expecteds.len(), actuals.len());

            for (expected, actual) in zip(expecteds, actuals) {
                assert_eq!(expected, actual)
            }
        }

        // 2. Each `xlix` operation result in 8 rows.
        {
            let xlix_instruction_count = 0;
            let prc_rows_count = base_matrices.processor_matrix.len();
            assert!(xlix_instruction_count <= 8 * prc_rows_count)
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

            let actual_input_count = base_matrices.input_matrix.len();

            assert_eq!(expected_input_count, actual_input_count);

            // Output
            let expected_output_count = 14;
            //let actual = base_matrices.ram_matrix.len();

            let actual_output_count = base_matrices.output_matrix.len();

            assert_eq!(expected_output_count, actual_output_count);
        }
    }

    #[test]
    #[ignore = "rewrite this test according to 'hash' instruction"]
    fn hash_hash_hash_test() {
        // 1. Execute program
        let code = sample_programs::HASH_HASH_HASH_HALT;
        let program = Program::from_code(code).unwrap();

        println!("{}", program);

        let mut stdin = VecStream::new(&[]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        println!("{:?}", err);
        for row in base_matrices.processor_matrix.clone() {
            println!("{}", ProcessorMatrixRow { row });
        }

        // 1. Check that `output_matrix` is equivalent to `stdout`
        {
            let expecteds = vec![].into_iter().rev().map(|x| BWord::new(x));
            let actuals: Vec<BWord> = base_matrices
                .output_matrix
                .iter()
                .map(|&[val]| val)
                .collect_vec();

            assert_eq!(expecteds.len(), actuals.len());

            for (expected, actual) in zip(expecteds, actuals) {
                assert_eq!(expected, actual)
            }
        }

        // 2. Each of three `xlix` operations result in 8 rows.
        {
            let expected = 24;
            assert_eq!(expected, base_matrices.aux_matrix.len());
        }

        //3. noRows(jmpstack_tabel) == noRows(processor_table)
        {
            let jmp_rows_count = base_matrices.jump_stack_matrix.len();
            let prc_rows_count = base_matrices.processor_matrix.len();
            assert_eq!(jmp_rows_count, prc_rows_count)
        }

        {
            // 4.1. The number of input_table rows is equivalent to the number of read_io operations.
            let expected_input_count = 0;
            let actual_input_count = base_matrices.input_matrix.len();

            assert_eq!(expected_input_count, actual_input_count);

            // 4.2. The number of output_table rows is equivalent to the number of write_io operations.
            let expected_output_count = 0;
            let actual_output_count = base_matrices.output_matrix.len();

            assert_eq!(expected_output_count, actual_output_count);
        }
    }

    fn _check_base_matrices(
        base_matrices: &BaseMatrices,
        expected_input_rows: usize,
        expected_output_rows: usize,
        xlix_instruction_count: usize,
    ) {
        // 1. Check `output_matrix`.
        {
            let expecteds = vec![].into_iter().rev().map(|x| BWord::new(x));
            let actuals: Vec<BWord> = base_matrices
                .output_matrix
                .iter()
                .map(|&[val]| val)
                .collect_vec();

            assert_eq!(expecteds.len(), actuals.len());

            for (expected, actual) in zip(expecteds, actuals) {
                assert_eq!(expected, actual)
            }
        }

        // 2. Each `xlix` operation result in 8 rows in the aux matrix.
        {
            let aux_rows_count = base_matrices.aux_matrix.len();
            assert_eq!(xlix_instruction_count * 8, aux_rows_count)
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
            let actual_input_rows = base_matrices.input_matrix.len();
            assert_eq!(expected_input_rows, actual_input_rows);

            // Output
            let actual_output_rows = base_matrices.output_matrix.len();

            assert_eq!(expected_output_rows, actual_output_rows);
        }
    }

    fn _check_polynomials_of_program(program: Program) {
        let mut _rng = rand::thread_rng();
        let mut stdin = VecStream::new(&[]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        let (_base_matrices, _err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        // 1. Make table collections so we can extract polynomials.
        // let table_collection = BaseTableCollection::from_base_matrices(base_matrices);

        // 2. Extract polynomials to get vector of MPolynomial<BFieldElement>
    }

    #[test]
    #[ignore = "rewrite this test according to 'hash' instruction"]
    fn processor_table_constraints_evaluate_to_zero_test() {
        let mut rng = rand::thread_rng();

        let all_programs = vec![sample_programs::HASH_HASH_HASH_HALT];
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

            // instantiate table objects
            // unpadded_height: usize,
            // num_randomizers: usize,
            // generator: BWord,
            // order: usize,
            // matrix: Vec<Vec<BWord>>,
            let mut processor_table: ProcessorTable = ProcessorTable::new_prover(
                smooth_generator,
                order as usize,
                number_of_randomizers,
                processor_matrix,
            );

            let air_constraints = processor_table.base_transition_constraints();
            assert_air_constraints_on_matrix(processor_table.data(), &air_constraints);

            // Test air constraints after padding as well
            processor_table.pad();

            assert!(
                other::is_power_of_two(processor_table.data().len()),
                "Matrix length must be power of 2 after padding"
            );

            assert_air_constraints_on_matrix(processor_table.data(), &air_constraints);

            // Test the same for the extended matrix
            let challenges: [XFieldElement; EXTENSION_CHALLENGE_COUNT] =
                XFieldElement::random_elements(EXTENSION_CHALLENGE_COUNT, &mut rng)
                    .try_into()
                    .unwrap();

            let initials: [XFieldElement; PERMUTATION_ARGUMENTS_COUNT] =
                XFieldElement::random_elements(PERMUTATION_ARGUMENTS_COUNT, &mut rng)
                    .try_into()
                    .unwrap();

            let ext_processor_table = processor_table.extend(challenges, initials);
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

    fn assert_air_constraints_on_matrix(
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
}
