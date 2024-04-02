use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use itertools::Itertools;

use twenty_first::math::ntt;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = interpolation<{ 1 << 10 }>,
              interpolation<{ 1 << 11 }>,
              interpolation<{ 1 << 12 }>,
              interpolation<{ 1 << 13 }>,
              interpolation<{ 1 << 14 }>,
);

fn interpolation<const SIZE: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE.ilog2();
    let mut group = c.benchmark_group(format!("Various Interpolations in 2^{log2_of_size} Points"));
    group.throughput(Throughput::Elements(u64::try_from(SIZE).unwrap()));

    let xs: Vec<BFieldElement> = random_elements(SIZE);
    let ys: Vec<BFieldElement> = random_elements(SIZE);

    let id = BenchmarkId::new("Lagrange interpolate", log2_of_size);
    let lagrange = || Polynomial::lagrange_interpolate(&xs, &ys);
    group.bench_function(id, |b| b.iter(lagrange));

    let id = BenchmarkId::new("Fast interpolate", log2_of_size);
    group.bench_function(id, |b| b.iter(|| Polynomial::fast_interpolate(&xs, &ys)));

    let id = BenchmarkId::new("Batch fast interpolate (100)", log2_of_size);
    const BATCH_SIZE: usize = 100;
    let root = BFieldElement::primitive_root_of_unity(u64::try_from(SIZE).unwrap()).unwrap();
    let batch_yss = (0..BATCH_SIZE).map(|_| random_elements(SIZE)).collect_vec();
    let batch_interpolate = || Polynomial::batch_fast_interpolate(&xs, &batch_yss, root, SIZE);
    group.bench_function(id, |b| b.iter(batch_interpolate));

    // Note that NTT/iNTT can only handle inputs of length 2^k and the domain has to be a subgroup
    // of order 2^k whereas the other interpolation methods are generic.
    let id = BenchmarkId::new("Regular iNTT", log2_of_size);
    let mut ys = ys;
    group.bench_function(id, |b| b.iter(|| ntt::intt(&mut ys, root, log2_of_size)));

    group.finish();
}
