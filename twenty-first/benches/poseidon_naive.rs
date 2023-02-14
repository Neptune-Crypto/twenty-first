use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use itertools::Itertools;
use rand::RngCore;
use rayon::prelude::IntoParallelRefIterator;
use rayon::prelude::ParallelIterator;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::poseidon::Poseidon;
use twenty_first::shared_math::poseidon::STATE_SIZE;

fn bench_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon_naive/hash_10");

    let size = 10;
    group.sample_size(100);

    let mut rng = rand::thread_rng();
    let single_element: [BFieldElement; STATE_SIZE] = (0..STATE_SIZE)
        .into_iter()
        .map(|_| BFieldElement::new(rng.next_u64()))
        .collect_vec()
        .try_into()
        .unwrap();

    group.bench_function(
        BenchmarkId::new("Poseidon Naive / Hash 10", size),
        |bencher| {
            bencher.iter(|| Poseidon::poseidon_naive(single_element));
        },
    );
}

fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon_naive/parallel");

    let size = 65536;
    group.sample_size(50);
    let elements: Vec<[BFieldElement; STATE_SIZE]> = (0..size)
        .map(|_| random_elements(STATE_SIZE).try_into().unwrap())
        .collect();

    group.bench_function(
        BenchmarkId::new("Poseidon Naive / Parallel Hash", size),
        |bencher| {
            bencher.iter(|| {
                elements
                    .par_iter()
                    .map(|&state| Poseidon::poseidon_naive(state))
                    .collect::<Vec<_>>()
            });
        },
    );
}

criterion_group!(benches, bench_10, bench_parallel);
criterion_main!(benches);
