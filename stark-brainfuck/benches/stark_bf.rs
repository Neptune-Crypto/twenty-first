use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::timing_reporter::TimingReporter;

use stark_brainfuck as brainfuck;
use stark_brainfuck::stark::Stark;
use stark_brainfuck::vm::sample_programs;
use stark_brainfuck::vm::BaseMatrices;

fn compile_simulate_prove_verify(program_code: &str, input: &[BFieldElement]) {
    let mut timer = TimingReporter::start();

    let program = brainfuck::vm::compile(program_code).unwrap();

    timer.elapsed("compile");

    let (trace_length, input_symbols, output_symbols) =
        brainfuck::vm::run(&program, input.to_vec()).unwrap();
    timer.elapsed("run");
    println!("run done");

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
        base_matrices.memory_matrix.len(),
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

fn stark_bf(c: &mut Criterion) {
    let mut group = c.benchmark_group("stark_bf");

    group.sample_size(10);

    // Last time I checked this produces a FRI domain length of 2^14
    let two_by_two_then_output_id = BenchmarkId::new("TWO_BY_TWO_THEN_OUTPUT", 97);
    let tbtto_input = [97].map(BFieldElement::new).to_vec();
    group.bench_with_input(
        two_by_two_then_output_id,
        &tbtto_input,
        |bencher, input_symbols| {
            bencher.iter(|| {
                compile_simulate_prove_verify(
                    sample_programs::TWO_BY_TWO_THEN_OUTPUT,
                    input_symbols,
                )
            });
        },
    );

    // Last time I checked this produces a FRI domain length of 2^19
    let hello_world_id = BenchmarkId::new("HELLO_WORLD", "");
    group.bench_function(hello_world_id, |bencher| {
        bencher.iter(|| compile_simulate_prove_verify(sample_programs::HELLO_WORLD, &[]));
    });

    // Last time I checked this produces a FRI domain length of 2^23
    // let the_raven_id = BenchmarkId::new("THE_RAVEN", "");
    // group.bench_function(the_raven_id, |bencher| {
    //     bencher.iter(|| compile_simulate_prove_verify(sample_programs::THE_RAVEN, &[]));
    // });

    group.finish();
}

criterion_group!(benches, stark_bf);
criterion_main!(benches);
