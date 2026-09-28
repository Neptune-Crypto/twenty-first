use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::math::x_field_element::XFieldElement;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bit_reverse_permutation::<{ 1 << 20 }>,
              bit_reverse_permutation::<{ 1 << 26 }>,
              pre_compute_twiddle_factors::<{ 1 << 20 }>,
              pre_compute_twiddle_factors::<{ 1 << 26 }>,
              bfe_ntt::<{ 1 << 7 }>,
              bfe_ntt::<{ 1 << 18 }>,
              bfe_ntt::<{ 1 << 23 }>,
              xfe_ntt::<{ 1 << 7 }>,
              xfe_ntt::<{ 1 << 18 }>,
              xfe_ntt::<{ 1 << 23 }>,
              bfe_par_ntt::<{ 1 << 18 }>,
              bfe_par_ntt::<{ 1 << 23 }>,
              xfe_par_ntt::<{ 1 << 18 }>,
              xfe_par_ntt::<{ 1 << 23 }>,
              bfe_intt::<{ 1 << 7 }>,
              bfe_intt::<{ 1 << 18 }>,
              bfe_intt::<{ 1 << 23 }>,
              xfe_intt::<{ 1 << 7 }>,
              xfe_intt::<{ 1 << 18 }>,
              xfe_intt::<{ 1 << 23 }>,
);

fn bit_reverse_permutation<const LEN: usize>(c: &mut Criterion) {
    let mut xs = random_elements::<BFieldElement>(LEN);
    c.benchmark_group("bit_reverse_permutation")
        .throughput(Throughput::Elements(LEN as u64))
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::bit_reverse_permutation(&mut xs))
        });
}

fn pre_compute_twiddle_factors<const LEN: u32>(c: &mut Criterion) {
    let root = BFieldElement::primitive_root_of_unity(LEN.into()).unwrap();
    c.benchmark_group("compute_twiddle_factors")
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::twiddle_factors(LEN, root))
        });
}

fn bfe_ntt<const LEN: usize>(c: &mut Criterion) {
    let mut xs = random_elements::<BFieldElement>(LEN);
    c.benchmark_group("bfe_ntt")
        .throughput(Throughput::Elements(LEN as u64))
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::ntt(&mut xs))
        });
}

fn bfe_par_ntt<const LEN: usize>(c: &mut Criterion) {
    let mut xs = random_elements::<BFieldElement>(LEN);
    c.benchmark_group("bfe_par_ntt")
        .throughput(Throughput::Elements(LEN as u64))
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::par_ntt(&mut xs))
        });
}

fn xfe_par_ntt<const LEN: usize>(c: &mut Criterion) {
    let mut xs = random_elements::<XFieldElement>(LEN);
    c.benchmark_group("xfe_par_ntt")
        .throughput(Throughput::Elements(LEN as u64))
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::par_ntt(&mut xs))
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
