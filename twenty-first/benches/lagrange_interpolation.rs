use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use num_traits::Pow;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::traits::PrimitiveRootOfUnity;
use twenty_first::shared_math::{ntt, polynomial};

fn lagrange_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lagrange_interpolation");

    let log2_of_sizes: Vec<usize> = vec![10, 14];

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
        ntt_based_fast_interpolate_batched(
            &mut group,
            BenchmarkId::new("Batch fast interpolate (100)", 2.pow(log2_of_size)),
            log2_of_size,
        );
        regular_intt(
            &mut group,
            BenchmarkId::new("Regular INTT", 2.pow(log2_of_size)),
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

    group.sample_size(10);
    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::lagrange_interpolate(&xs, &ys))
    });
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

    group.sample_size(10);
    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::fast_interpolate(&xs, &ys, &omega, size))
    });
}

fn ntt_based_fast_interpolate_batched(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
) {
    let batch_size = 100;
    let size: usize = 1 << log2_of_size;
    let xs: Vec<BFieldElement> = random_elements(size);
    let ys = (0..batch_size).map(|_| random_elements(size)).collect();
    let omega = BFieldElement::primitive_root_of_unity(size as u64).unwrap();

    group.sample_size(10);
    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::batch_fast_interpolate(&xs, &ys, &omega, size))
    });
}

// Note that NTT/INTT can only handle inputs of length 2^k and the domain has to be a subgroup
// of order 2^k whereas the other interpolation methods are generic.
fn regular_intt(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, log2_of_size: usize) {
    let size: usize = 1 << log2_of_size;
    let mut ys: Vec<BFieldElement> = random_elements(size);
    let omega = BFieldElement::primitive_root_of_unity(size as u64).unwrap();

    group.sample_size(10);
    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| ntt::intt(&mut ys, omega, log2_of_size as u32))
    });
}

criterion_group!(benches, lagrange_interpolation);
criterion_main!(benches);
