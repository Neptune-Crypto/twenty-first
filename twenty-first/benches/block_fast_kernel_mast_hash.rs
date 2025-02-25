use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use rand::random;
use twenty_first::prelude::*;

fn fast_kernel_mast_hash_bench(c: &mut Criterion) {
    #[inline(always)]
    fn fast_kernel_mast_hash(
        kernel_auth_path: [Digest; 2],
        header_auth_path: [Digest; 3],
        nonce: Digest,
    ) -> Digest {
        let header_mast_hash =
            Tip5::hash_pair(Tip5::hash_varlen(&nonce.encode()), header_auth_path[0]);
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

    let mut group = c.benchmark_group("tip5");

    group.sample_size(10000);

    let kernel_auth_path: [Digest; 2] = random();
    let header_auth_path: [Digest; 3] = random();
    let nonce: Digest = random();
    group.bench_function(
        BenchmarkId::new("Tip5", "fast kernel MAST hash"),
        |bencher| {
            bencher.iter(|| fast_kernel_mast_hash(kernel_auth_path, header_auth_path, nonce));
        },
    );
}

criterion_group!(benches, fast_kernel_mast_hash_bench);
criterion_main!(benches);
