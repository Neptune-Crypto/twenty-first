use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use primitive_types::U256;
use twenty_first::shared_math::prime_field_element_flexible::PrimeFieldElementFlexible;
use twenty_first::shared_math::rescue_prime_pfe_flexible::RescuePrime;
use twenty_first::shared_math::stark::rpsss::RPSSS;
use twenty_first::shared_math::stark::stark_pfe_flexible::StarkPrimeFieldElementFlexible;

pub fn get_tutorial_stark(prime: U256) -> (StarkPrimeFieldElementFlexible, RescuePrime) {
    let expansion_factor = 4;
    let colinearity_checks_count = 2;
    let rescue_prime = RescuePrime::from_tutorial();
    let register_count = rescue_prime.m;
    let cycles_count = rescue_prime.steps_count + 1;
    let transition_constraints_degree = 2;
    let generator =
        PrimeFieldElementFlexible::new(85408008396924667383611388730472331217u128.into(), prime);

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

fn rpsss_bench_sign(c: &mut Criterion) {
    let prime = (407u128 * (1 << 119) + 1).into();
    let (mut stark, rp) = get_tutorial_stark(prime);
    stark.prover_preprocess();
    let rpsss = RPSSS {
        stark: stark.clone(),
        rp,
    };
    let document_string: String = "Hello Neptune!".to_string();
    let document: Vec<u8> = document_string.into_bytes();

    let (sk, _pk) = rpsss.keygen();
    let mut group_sign = c.benchmark_group("rpsss_bench_sign");
    group_sign
        .bench_with_input(
            BenchmarkId::from_parameter("rpsss_bench_sign"),
            &1,
            |b, _| {
                b.iter(|| rpsss.sign(&sk, &document));
            },
        )
        .sample_size(10);
    group_sign.finish();
}

criterion_group!(benches, rpsss_bench_sign);
criterion_main!(benches);
