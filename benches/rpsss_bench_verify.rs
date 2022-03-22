use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use twenty_first::shared_math::prime_field_element_flexible::PrimeFieldElementFlexible;
use twenty_first::shared_math::rescue_prime_pfe_flexible::RescuePrime;
use twenty_first::shared_math::stark::rpsss::{Signature, RPSSS};
use twenty_first::shared_math::stark::stark_pfe_flexible::StarkPrimeFieldElementFlexible;

pub fn get_tutorial_stark<'a>() -> (StarkPrimeFieldElementFlexible, RescuePrime) {
    let expansion_factor = 4;
    let colinearity_checks_count = 2;
    let prime = RescuePrime::prime_from_tutorial();
    let rescue_prime = RescuePrime::from_tutorial();
    let register_count = rescue_prime.m;
    let cycles_count = rescue_prime.steps_count + 1;
    let transition_constraints_degree = 2;
    let generator_value: u128 = 85408008396924667383611388730472331217;
    let generator = PrimeFieldElementFlexible::new(generator_value.into(), prime);

    let stark = StarkPrimeFieldElementFlexible::new(
        expansion_factor,
        colinearity_checks_count,
        register_count,
        cycles_count,
        transition_constraints_degree,
        generator,
    );

    (stark, rescue_prime)
}

fn rpsss_bench_verify(c: &mut Criterion) {
    let (mut stark, rp) = get_tutorial_stark();

    stark.prover_preprocess();
    let rpsss = RPSSS {
        stark: stark.clone(),
        rp,
    };

    let document_string: String = "Hello Neptune!".to_string();
    let document: Vec<u8> = document_string.into_bytes();

    let (sk, pk) = rpsss.keygen();
    let signature: Signature = rpsss.sign(&sk, &document).unwrap();
    let mut group_verify = c.benchmark_group("rpsss_bench_verify");
    group_verify
        .bench_with_input(
            BenchmarkId::from_parameter("rpsss_bench_verify"),
            &1,
            |b, _| {
                b.iter(|| rpsss.verify(&pk, &signature, &document));
            },
        )
        .sample_size(10);
    group_verify.finish();
}
criterion_group!(benches, rpsss_bench_verify);
criterion_main!(benches);
