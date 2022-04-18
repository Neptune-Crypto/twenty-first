use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_xlix::RescuePrimeXlix;
use twenty_first::shared_math::traits::GetRandomElements;
use twenty_first::util_types::simple_hasher::{Hasher, RescuePrimeProduction};

fn rescue_prime_wat(c: &mut Criterion) {
    let mut group = c.benchmark_group("rescue_prime_wat");

    group.sample_size(100);

    let mut hasher_rp = RescuePrimeProduction::new();
    let mut hasher_rp_xlix = RescuePrimeXlix::new();

    let mut rng = rand::thread_rng();
    let elements = BFieldElement::random_elements(2048, &mut rng);
    let element = elements[0];

    // Hashing individual BFieldElements

    group.bench_function(BenchmarkId::new("RescuePrime-one", 0), |bencher| {
        bencher.iter(|| hasher_rp.hash(&element));
    });

    group.bench_function(BenchmarkId::new("RescuePrimeXlix-one", 0), |bencher| {
        bencher.iter(|| hasher_rp_xlix.hash(&element));
    });

    // Hashing many BFieldElements

    group.bench_function(BenchmarkId::new("RescuePrime-many", 0), |bencher| {
        bencher.iter(|| {
            let chunks: Vec<Vec<BFieldElement>> = elements.chunks(5).map(|s| s.to_vec()).collect();
            hasher_rp.hash_many(&chunks);
        });
    });

    group.bench_function(BenchmarkId::new("RescuePrimeXlix-many", 0), |bencher| {
        bencher.iter(|| {
            hasher_rp_xlix.hash_wrapper(&elements, 5);
        });
    });

    // Changing the rate from 12 to 15 doesn't make a big difference.

    // let mut hasher_rp_xlix_wide = RescuePrimeXlix::new();
    // hasher_rp_xlix_wide.capacity = 1;

    // group.bench_function(
    //     BenchmarkId::new("RescuePrimeXlix-many-wide", 0),
    //     |bencher| {
    //         bencher.iter(|| {
    //             hasher_rp_xlix.hash_wrapper(&elements, 5);
    //         });
    //     },
    // );
}

criterion_group!(benches, rescue_prime_wat);
criterion_main!(benches);
