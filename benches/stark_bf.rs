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

    // Standard high parameters
    let log_expansion_factor = 4;
    let security_level = 160;

    let mut stark = Stark::new(
        trace_length,
        program_code.to_string(),
        input_symbols,
        output_symbols,
        log_expansion_factor,
        security_level,
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

fn _two_by_two_then_output() {
    let program_code = brainfuck::vm::sample_programs::TWO_BY_TWO_THEN_OUTPUT;
    let input = [97].map(BFieldElement::new).to_vec();
    compile_simulate_prove_verify(program_code, &input);
}

fn _hello_world() {
    let program_code = brainfuck::vm::sample_programs::HELLO_WORLD;
    let input = [].map(BFieldElement::new).to_vec();
    compile_simulate_prove_verify(program_code, &input);
}

fn _the_raven() {
    let program_code = brainfuck::vm::sample_programs::THE_RAVEN;
    let input = [].map(BFieldElement::new).to_vec();
    compile_simulate_prove_verify(program_code, &input);
}

fn stark_bf(c: &mut Criterion) {
    let mut group = c.benchmark_group("stark_bf");

    group.sample_size(10);

    let two_by_two_then_output_id = BenchmarkId::new("TWO_BY_TWO_THEN_OUTPUT", 0);
    group.bench_function(two_by_two_then_output_id, |bencher| {
        bencher.iter(|| _two_by_two_then_output());
    });

    // let hello_world_id = BenchmarkId::new("HELLO_WORLD", 0);
    // group.bench_function(hello_world_id, |bencher| {
    //     bencher.iter(|| _hello_world());
    // });

    // let the_raven_id = BenchmarkId::new("THE_RAVEN", 0);
    // group.bench_function(the_raven_id, |bencher| {
    //     bencher.iter(|| _the_raven());
    // });

    group.finish();
}

criterion_group!(benches, stark_bf);
criterion_main!(benches);
