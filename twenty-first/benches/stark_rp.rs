use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime::RescuePrime;
use twenty_first::shared_math::rescue_prime_params as params;
use twenty_first::shared_math::stark::rescue_prime::stark_rp::StarkRp;
use twenty_first::shared_math::traits::GetPrimitiveRootOfUnity;
use twenty_first::timing_reporter::TimingReporter;
use twenty_first::util_types::proof_stream::ProofStream;

fn stark_medium(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("stark_rp");

    let rp: RescuePrime = params::rescue_prime_params_bfield_0();
    let benchmark_id = BenchmarkId::new("large", 7);
    // let rp: RescuePrime = params::rescue_prime_medium_test_params();
    // let benchmark_id = BenchmarkId::new("medium", 5);

    group.sample_size(10);
    group.bench_function(benchmark_id, |bencher| {
        let stark: StarkRp = StarkRp::new(16, 2, rp.m as u32, BFieldElement::new(7));

        let mut timer = TimingReporter::start();

        let mut input = vec![BFieldElement::zero(); rp.max_input_length];
        input[0] = BFieldElement::one();
        let (output, trace) = rp.eval_and_trace(&input);
        timer.elapsed("rp.eval_and_trace(...)");
        let omicron = BFieldElement::zero()
            .get_primitive_root_of_unity(16)
            .0
            .unwrap();
        timer.elapsed("BFieldElement::get_primitive_root_of_unity(16)");
        let air_constraints = rp.get_air_constraints(omicron);
        timer.elapsed("rp.get_air_constraints(omicron)");
        let boundary_constraints = rp.get_boundary_constraints(&output);
        timer.elapsed("rp.get_boundary_constraints(...)");

        let report = timer.finish();
        println!("{}", report);

        bencher.iter(|| {
            let mut proof_stream = ProofStream::default();

            let prove_result = stark.prove(
                &trace,
                &air_constraints,
                &boundary_constraints,
                &mut proof_stream,
                omicron,
            );

            let (fri_domain_length, omega): (u32, BFieldElement) = prove_result.unwrap();

            let verify_result = stark.verify(
                &mut proof_stream,
                &air_constraints,
                &boundary_constraints,
                fri_domain_length,
                omega,
                trace.len() as u32,
            );
            println!("proof_stream: {} bytes", proof_stream.len());
            assert!(verify_result.is_ok());
        });
    });
}

criterion_group!(benches, stark_medium);
criterion_main!(benches);
