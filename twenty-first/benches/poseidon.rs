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
use twenty_first::util_types::algebraic_hasher::SpongeHasher;

fn bench_10(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon/hash_10");

    let size = 10;
    group.sample_size(100);

    let mut rng = rand::thread_rng();
    let single_element: [BFieldElement; STATE_SIZE] = (0..STATE_SIZE)
        .into_iter()
        .map(|_| BFieldElement::new(rng.next_u64()))
        .collect_vec()
        .try_into()
        .unwrap();

    group.bench_function(BenchmarkId::new("Poseidon / Hash 10", size), |bencher| {
        bencher.iter(|| Poseidon::poseidon(single_element));
    });
}

fn bench_varlen(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon/hash_varlen");

    let size = 16_390;
    group.sample_size(50);
    let elements = random_elements(size);

    group.bench_function(
        BenchmarkId::new("Poseidon / Hash Variable Length", size),
        |bencher| {
            bencher.iter(|| {
                let mut sponge = Poseidon::init();
                Poseidon::absorb_repeatedly(&mut sponge, elements.iter())
            });
        },
    );
}

fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon/parallel");

    let size = 65536;
    group.sample_size(50);
    let elements: Vec<[BFieldElement; STATE_SIZE]> = (0..size)
        .map(|_| random_elements(STATE_SIZE).try_into().unwrap())
        .collect();

    group.bench_function(
        BenchmarkId::new("Poseidon / Parallel Hash", size),
        |bencher| {
            bencher.iter(|| {
                elements
                    .par_iter()
                    .map(|&state| Poseidon::poseidon(state))
                    .collect::<Vec<_>>()
            });
        },
    );
}

criterion_group!(benches, bench_10, bench_varlen, bench_parallel);
criterion_main!(benches);
