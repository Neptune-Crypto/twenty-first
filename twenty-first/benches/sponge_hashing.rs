use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::{random_elements, random_elements_array};
use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
use twenty_first::util_types::algebraic_hasher::{AlgebraicHasher, SpongeHasher, RATE};

fn hash_varlen_bench<H: AlgebraicHasher>(c: &mut Criterion) {
    let mut group = c.benchmark_group("sponge");

    let input_length = 150;

    group.sample_size(1000);
    group.bench_function(BenchmarkId::new("hash_varlen", input_length), |bencher| {
        let input = random_elements(input_length);
        bencher.iter(|| {
            H::hash_varlen(&input);
        });
    });
}

fn sample_indices_bench<H: SpongeHasher>(c: &mut Criterion) {
    let mut group = c.benchmark_group("sponge");

    let num_indices = 50;
    let upper_bound = 1 << 20;

    group.sample_size(1000);
    group.bench_function(BenchmarkId::new("sample_indices", num_indices), |bencher| {
        let seed = random_elements_array::<BFieldElement, RATE>();
        let mut sponge = H::absorb_init(&seed);
        bencher.iter(|| {
            H::sample_indices(&mut sponge, upper_bound, num_indices);
        });
    });
}

fn sample_weights_bench<H: SpongeHasher>(c: &mut Criterion) {
    let mut group = c.benchmark_group("sponge");

    let num_weights = 120;

    group.sample_size(1000);
    group.bench_function(BenchmarkId::new("sample_weights", num_weights), |bencher| {
        let seed = random_elements_array::<BFieldElement, RATE>();
        let mut sponge = H::absorb_init(&seed);
        bencher.iter(|| H::sample_weights(&mut sponge, num_weights));
    });
}

criterion_group!(
    benches,
    hash_varlen_bench<RescuePrimeRegular>,
    sample_indices_bench<RescuePrimeRegular>,
    sample_weights_bench<RescuePrimeRegular>,
);
criterion_main!(benches);
