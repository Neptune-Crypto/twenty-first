use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime::RescuePrime;
use twenty_first::shared_math::rescue_prime_params as params;
use twenty_first::shared_math::stark::Stark;
use twenty_first::util_types::proof_stream::ProofStream;

fn stark_medium(criterion: &mut Criterion) {
    let rp: RescuePrime = params::rescue_prime_medium_test_params();
    let stark: Stark = Stark::new(16, 2, rp.m as u32, BFieldElement::new(7));

    let mut group = criterion.benchmark_group("rescue_prime_air_constraints");
    group.sample_size(10);
    let benchmark_id = BenchmarkId::new("stark", 5);
    group.bench_function(benchmark_id, |bencher| {
        bencher.iter(|| {
            let one = BFieldElement::ring_one();
            let (output, trace) = rp.eval_and_trace(&one);
            let omicron = BFieldElement::get_primitive_root_of_unity(16).0.unwrap();
            let air_constraints = rp.get_air_constraints(omicron);
            let boundary_constraints = rp.get_boundary_constraints(output);
            let mut proof_stream = ProofStream::default();

            let prove_result = stark.prove(
                &trace,
                &air_constraints,
                &boundary_constraints,
                &mut proof_stream,
                omicron,
            );

            let (fri_domain_length, omega): (u32, BFieldElement) = prove_result.unwrap();

            stark
                .verify(
                    &mut proof_stream,
                    &air_constraints,
                    &boundary_constraints,
                    fri_domain_length,
                    omega,
                    trace.len() as u32,
                )
                .unwrap();
        });
    });
}

criterion_group!(benches, stark_medium);
criterion_main!(benches);
