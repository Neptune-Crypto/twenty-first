use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::stark::brainfuck;
use twenty_first::shared_math::stark::brainfuck::stark::Stark;
use twenty_first::shared_math::stark::brainfuck::vm::BaseMatrices;
use twenty_first::timing_reporter::TimingReporter;

fn compile_simulate_prove_verify(program_code: &str, input: &[BFieldElement]) {
    let mut timer = TimingReporter::start();

    let program = brainfuck::vm::compile(program_code).unwrap();

    timer.elapsed("compile");

    let (trace_length, input_symbols, output_symbols) =
        brainfuck::vm::run(&program, input.to_vec()).unwrap();
    timer.elapsed("run");

    let base_matrices: BaseMatrices = brainfuck::vm::simulate(&program, &input_symbols).unwrap();
    timer.elapsed("simulate");

    let mut stark = Stark::new(
        trace_length,
        program_code.to_string(),
        input_symbols,
        output_symbols,
    );
    timer.elapsed("new");

    let mut proof_stream = stark.prove(base_matrices).unwrap();
    timer.elapsed("prove");

    let verifier_verdict = stark.verify(&mut proof_stream);
    timer.elapsed("verify");
    let report = timer.finish();

    match verifier_verdict {
        Ok(_) => (),
        Err(err) => panic!("error in STARK verifier: {}", err),
    };
    println!("{}", report);
}

fn two_by_two_then_output() {
    let program_code = brainfuck::vm::sample_programs::TWO_BY_TWO_THEN_OUTPUT;
    let input = [97].map(BFieldElement::new).to_vec();
    compile_simulate_prove_verify(program_code, &input);
}

fn stark_bf(c: &mut Criterion) {
    let mut group = c.benchmark_group("stark_bf");

    group.sample_size(10);

    let two_by_two_then_output_id = BenchmarkId::new("TWO_BY_TWO_THEN_OUTPUT", 0);
    group.bench_function(two_by_two_then_output_id, |bencher| {
        bencher.iter(|| two_by_two_then_output());
    });

    group.finish();
}

criterion_group!(benches, stark_bf);
criterion_main!(benches);
