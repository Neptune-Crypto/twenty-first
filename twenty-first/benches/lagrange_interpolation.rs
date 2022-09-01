use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use num_traits::Pow;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::polynomial;
use twenty_first::shared_math::traits::{GetPrimitiveRootOfUnity, GetRandomElements};

fn lagrange_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("lagrange_interpolation");

    let log2_of_sizes: Vec<usize> = vec![3, 4, 7, 10];

    // Benchmarking forward ntt on BFieldElements
    for &log2_of_size in log2_of_sizes.iter() {
        fast_lagrange_interpolate(
            &mut group,
            BenchmarkId::new("fast_lagrange_interpolate", 2.pow(log2_of_size)),
            log2_of_size,
        );
        slow_lagrange_interpolation_internal(
            &mut group,
            BenchmarkId::new("slow_lagrange_interpolation_internal", 2.pow(log2_of_size)),
            log2_of_size,
        );
        lagrange_interpolate(
            &mut group,
            BenchmarkId::new("lagrange_interpolate", 2.pow(log2_of_size)),
            log2_of_size,
        );
        ntt_bench(
            &mut group,
            BenchmarkId::new("ntt", 2.pow(log2_of_size)),
            log2_of_size,
        );
    }

    group.finish();
}

fn fast_lagrange_interpolate(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
) {
    let size: usize = 1 << log2_of_size;

    let mut rng = rand::thread_rng();
    let xs = BFieldElement::random_elements(size, &mut rng);
    let ys = BFieldElement::random_elements(size, &mut rng);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::fast_lagrange_interpolate(&xs, &ys))
    });
    group.sample_size(10);
}

fn slow_lagrange_interpolation_internal(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
) {
    let size: usize = 1 << log2_of_size;

    let mut rng = rand::thread_rng();
    let xs = BFieldElement::random_elements(size, &mut rng);
    let ys = BFieldElement::random_elements(size, &mut rng);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::slow_lagrange_interpolation_internal(&xs, &ys))
    });
    group.sample_size(10);
}

fn lagrange_interpolate(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
) {
    let size: usize = 1 << log2_of_size;

    let mut rng = rand::thread_rng();
    let xs = BFieldElement::random_elements(size, &mut rng);
    let ys = BFieldElement::random_elements(size, &mut rng);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::lagrange_interpolate(&xs, &ys))
    });
    group.sample_size(10);
}

fn ntt_bench(group: &mut BenchmarkGroup<WallTime>, bench_id: BenchmarkId, log2_of_size: usize) {
    let size: usize = 1 << log2_of_size;

    let mut rng = rand::thread_rng();
    let xs = BFieldElement::random_elements(size, &mut rng);
    let ys = BFieldElement::random_elements(size, &mut rng);
    let omega = xs[0].get_primitive_root_of_unity(size as u64).0.unwrap();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| polynomial::Polynomial::fast_interpolate(&xs, &ys, &omega, size))
    });
    group.sample_size(10);
}

criterion_group!(benches, lagrange_interpolation);
criterion_main!(benches);
