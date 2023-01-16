use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_traits::{One, Zero};

use stark_shared::proof_stream::ProofStream;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_digest::DIGEST_LENGTH;
use twenty_first::shared_math::rescue_prime_regular::{RescuePrimeRegular, STATE_SIZE};
use twenty_first::shared_math::traits::PrimitiveRootOfUnity;
use twenty_first::timing_reporter::TimingReporter;

use stark_rescue_prime::stark_rp::StarkRp;

fn stark_medium(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("stark_rp");

    let benchmark_id = BenchmarkId::new("large", 7);
    // let rp: RescuePrime = params::rescue_prime_medium_test_params();
    // let benchmark_id = BenchmarkId::new("medium", 5);

    group.sample_size(10);
    group.bench_function(benchmark_id, |bencher| {
        let stark: StarkRp = StarkRp::new(16, 2, STATE_SIZE as u32, BFieldElement::new(7));

        let mut timer = TimingReporter::start();

        let mut input = [BFieldElement::zero(); 10];
        input[0] = BFieldElement::one();
        let trace = RescuePrimeRegular::trace(&input);
        let output = &trace[trace.len() - 1][0..DIGEST_LENGTH];

        timer.elapsed("rp.eval_and_trace(...)");
        let omicron = BFieldElement::primitive_root_of_unity(32).unwrap();
        timer.elapsed("BFieldElement::get_primitive_root_of_unity(32)");
        let air_constraints = StarkRp::get_air_constraints(omicron);
        timer.elapsed("rp.get_air_constraints(omicron)");
        let boundary_constraints = StarkRp::get_boundary_constraints(output);
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
