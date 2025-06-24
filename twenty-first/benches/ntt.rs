use std::array;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::math::x_field_element::XFieldElement;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = // pre_compute_swap_indices::<{ 1 << 20 }>,
            //   pre_compute_swap_indices::<{ 1 << 26 }>,
            //   pre_compute_twiddle_factors::<{ 1 << 20 }>,
            //   pre_compute_twiddle_factors::<{ 1 << 26 }>,
            //   bfe_ntt::<{ 1 << 7 }>,
              bfe_ntt::<{ 1 << 24 }>,
              par_bfe_ntt<{ 1 << 24 }>,
              xfe_ntt::<{ 1 << 24 }>,
              par_xfe_ntt<{ 1 << 24 }>,
            //   bfe_ntt::<{ 1 << 23 }>,
            //   xfe_ntt::<{ 1 << 7 }>,
            //   xfe_ntt::<{ 1 << 18 }>,
            //   xfe_ntt::<{ 1 << 23 }>,
            //   bfe_intt::<{ 1 << 7 }>,
            //   bfe_intt::<{ 1 << 18 }>,
            //   bfe_intt::<{ 1 << 23 }>,
            //   xfe_intt::<{ 1 << 7 }>,
            //   xfe_intt::<{ 1 << 18 }>,
            //   xfe_intt::<{ 1 << 23 }>,
);

fn pre_compute_swap_indices<const LEN: usize>(c: &mut Criterion) {
    c.benchmark_group("compute_swap_indices")
        .bench_function(BenchmarkId::new("len", LEN.ilog2()), |b| {
            b.iter(|| twenty_first::math::ntt::swap_indices(LEN))
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

fn par_bfe_ntt<const LEN: usize>(c: &mut Criterion) {
    const COLUMNS: usize = 379;
    let mut xs: [Vec<BFieldElement>; COLUMNS] = array::from_fn(|_| random_elements(LEN));
    let id = BenchmarkId::new("len/width: ", format!("{}/{COLUMNS}", LEN.ilog2()));
    c.benchmark_group("par_bfe_ntt")
        .throughput(Throughput::Elements((LEN * COLUMNS) as u64))
        .bench_with_input(id, &LEN, |b, _| {
            b.iter(|| {
                xs.par_iter_mut()
                    .for_each(|x| twenty_first::math::ntt::ntt(x));
            })
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

fn par_xfe_ntt<const LEN: usize>(c: &mut Criterion) {
    const COLUMNS: usize = 88;
    let mut xs: [Vec<XFieldElement>; COLUMNS] = array::from_fn(|_| random_elements(LEN));
    let id = BenchmarkId::new("len/width: ", format!("{}/{COLUMNS}", LEN.ilog2()));
    c.benchmark_group("par_xfe_ntt")
        .throughput(Throughput::Elements((LEN * COLUMNS) as u64))
        .bench_with_input(id, &LEN, |b, _| {
            b.iter(|| {
                xs.par_iter_mut()
                    .for_each(|x| twenty_first::math::ntt::ntt(x));
            })
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
