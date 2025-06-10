use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::x_field_element::XFieldElement;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bfe_ntt::<{ 1 << 7 }>,
              bfe_ntt::<{ 1 << 18 }>,
              bfe_ntt::<{ 1 << 23 }>,
              xfe_ntt::<{ 1 << 7 }>,
              xfe_ntt::<{ 1 << 18 }>,
              xfe_ntt::<{ 1 << 23 }>,
              bfe_intt::<{ 1 << 7 }>,
              bfe_intt::<{ 1 << 18 }>,
              bfe_intt::<{ 1 << 23 }>,
              xfe_intt::<{ 1 << 7 }>,
              xfe_intt::<{ 1 << 18 }>,
              xfe_intt::<{ 1 << 23 }>,
);

fn bfe_ntt<const LEN: usize>(c: &mut Criterion) {
    let mut xs = random_elements::<BFieldElement>(LEN);
    c.benchmark_group("bfe_ntt")
        .throughput(Throughput::Elements(LEN as u64))
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::ntt(&mut xs))
        });
}

fn xfe_ntt<const LEN: usize>(c: &mut Criterion) {
    let mut xs = random_elements::<XFieldElement>(LEN);
    c.benchmark_group("xfe_ntt")
        .throughput(Throughput::Elements(LEN as u64))
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::ntt(&mut xs))
        });
}

fn bfe_intt<const LEN: usize>(c: &mut Criterion) {
    let mut xs = random_elements::<BFieldElement>(LEN);
    c.benchmark_group("bfe_intt")
        .throughput(Throughput::Elements(LEN as u64))
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::intt(&mut xs))
        });
}

fn xfe_intt<const LEN: usize>(c: &mut Criterion) {
    let mut xs = random_elements::<XFieldElement>(LEN);
    c.benchmark_group("xfe_intt")
        .throughput(Throughput::Elements(LEN as u64))
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::intt(&mut xs))
        });
}
