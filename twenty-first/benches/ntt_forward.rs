use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::ntt::ntt;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::math::x_field_element::XFieldElement;

fn chu_ntt_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("chu_ntt_forward");

    let log2_of_sizes: Vec<usize> = vec![3, 7, 12, 18, 23];

    // Benchmarking forward ntt on BFieldElements
    for &log2_of_size in log2_of_sizes.iter() {
        bfield_benchmark(
            &mut group,
            BenchmarkId::new("bfield", log2_of_size),
            log2_of_size,
        );
    }

    // Benchmarking forward ntt on XFieldElements
    for &log2_of_size in log2_of_sizes.iter() {
        xfield_benchmark(
            &mut group,
            BenchmarkId::new("xfield", log2_of_size),
            log2_of_size,
        );
    }

    group.finish();
}

fn bfield_benchmark(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
) {
    let size: usize = 1 << log2_of_size;
    let mut xs: Vec<BFieldElement> = random_elements(size);
    let omega = BFieldElement::primitive_root_of_unity(size as u64).unwrap();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| ntt::<BFieldElement>(&mut xs, omega, log2_of_size as u32))
    });
    group.sample_size(10);
}

fn xfield_benchmark(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
) {
    let size: usize = 1 << log2_of_size;

    let mut xs: Vec<XFieldElement> = random_elements(size);
    let omega = BFieldElement::primitive_root_of_unity(size as u64).unwrap();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| ntt::<XFieldElement>(&mut xs, omega, log2_of_size as u32))
    });
    group.sample_size(10);
}

criterion_group!(benches, chu_ntt_forward);
criterion_main!(benches);
