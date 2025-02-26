use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand::random;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

#[inline(always)]
fn fast_kernel_mast_hash(
    kernel_auth_path: [Digest; 2],
    header_auth_path: [Digest; 3],
    nonce: Digest,
) -> Digest {
    let header_mast_hash = Tip5::hash_pair(Tip5::hash_varlen(&nonce.encode()), header_auth_path[0]);
    let header_mast_hash = Tip5::hash_pair(header_mast_hash, header_auth_path[1]);
    let header_mast_hash = Tip5::hash_pair(header_auth_path[2], header_mast_hash);

    Tip5::hash_pair(
        Tip5::hash_pair(
            Tip5::hash_varlen(&header_mast_hash.encode()),
            kernel_auth_path[0],
        ),
        kernel_auth_path[1],
    )
}

fn par_fast_kernel_mast_hash_bench(c: &mut Criterion) {
    use rayon::iter::IntoParallelRefIterator;
    use rayon::iter::ParallelIterator;

    let mut group = c.benchmark_group("tip5");
    group.sample_size(10);

    let kernel_auth_path: [Digest; 2] = random();
    let header_auth_path: [Digest; 3] = random();
    let nonces: Vec<Digest> = random_elements(1_000_000);
    group.bench_function(BenchmarkId::new("parallel", "1_000_000"), |bencher| {
        bencher.iter(|| {
            let _a: Vec<_> = nonces
                .par_iter()
                .map(|nonce| fast_kernel_mast_hash(kernel_auth_path, header_auth_path, *nonce))
                .collect();
        });
    });
}

fn seq_fast_kernel_mast_hash_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tip5");
    group.sample_size(10);

    let kernel_auth_path: [Digest; 2] = random();
    let header_auth_path: [Digest; 3] = random();
    let nonces: Vec<Digest> = random_elements(100_000);
    group.bench_function(BenchmarkId::new("sequential", "100_000"), |bencher| {
        bencher.iter(|| {
            let _a: Vec<_> = nonces
                .iter()
                .map(|nonce| fast_kernel_mast_hash(kernel_auth_path, header_auth_path, *nonce))
                .collect();
        });
    });
}

criterion_group!(
    benches,
    seq_fast_kernel_mast_hash_bench,
    par_fast_kernel_mast_hash_bench
);
criterion_main!(benches);
