use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use num_traits::Pow;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::polynomial;
use twenty_first::shared_math::traits::PrimitiveRootOfUnity;

fn lagrange_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lagrange_interpolation");

    let log2_of_sizes: Vec<usize> = vec![3, 4, 7, 10];

    // Benchmarking forward ntt on BFieldElements
    for &log2_of_size in log2_of_sizes.iter() {
        lagrange_interpolate(
            &mut group,
            BenchmarkId::new("lagrange_interpolate", 2.pow(log2_of_size)),
            log2_of_size,
        );
        ntt_based_fast_interpolate(
            &mut group,
            BenchmarkId::new("NTT-Based interpolation", 2.pow(log2_of_size)),
            log2_of_size,
        );
    }

    group.finish();
}

fn lagrange_interpolate(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
) {
    let size: usize = 1 << log2_of_size;
    let xs: Vec<BFieldElement> = random_elements(size);
    let ys: Vec<BFieldElement> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::lagrange_interpolate(&xs, &ys))
    });
    group.sample_size(10);
}

fn ntt_based_fast_interpolate(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
) {
    let size: usize = 1 << log2_of_size;
    let xs: Vec<BFieldElement> = random_elements(size);
    let ys: Vec<BFieldElement> = random_elements(size);
    let omega = BFieldElement::primitive_root_of_unity(size as u64).unwrap();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::fast_interpolate(&xs, &ys, &omega, size))
    });
    group.sample_size(10);
}

criterion_group!(benches, lagrange_interpolation);
criterion_main!(benches);
