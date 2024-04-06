use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = poly_div<12,  4>,
              poly_div<12,  6>,
              poly_div<12,  8>,
              poly_div<12, 12>,
              poly_div<13,  4>,
              poly_div<13,  6>,
              poly_div<13,  8>,
              poly_div<13, 12>,
              poly_div<14,  4>,
              poly_div<14,  6>,
              poly_div<14,  8>,
              poly_div<14, 12>,
              poly_div<15,  4>,
              poly_div<15,  6>,
              poly_div<15,  8>,
              poly_div<15, 12>,
              poly_div<16,  4>,
              poly_div<16,  6>,
              poly_div<16,  8>,
              poly_div<16, 12>,
              poly_div<17,  4>,
              poly_div<17,  6>,
              poly_div<17,  8>,
              poly_div<17, 12>,
              poly_div<18,  4>,
              poly_div<18,  6>,
              poly_div<18,  8>,
              poly_div<18, 12>,
);

fn poly_div<const LOG2_NUM_DEG: usize, const LOG2_DEN_DEG: usize>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!(
        "Division of Polynomials – \
         Dividend Degree: 2^{LOG2_NUM_DEG}, Divisor Degree: 2^{LOG2_DEN_DEG}"
    ));
    let bench_param = || format!("{LOG2_NUM_DEG}|{LOG2_DEN_DEG}");

    let num = Polynomial::<BFieldElement>::new(random_elements((1 << LOG2_NUM_DEG) + 1));
    let den = Polynomial::<BFieldElement>::new(random_elements((1 << LOG2_DEN_DEG) + 1));

    let id = BenchmarkId::new("Naïve", bench_param());
    group.bench_function(id, |b| b.iter(|| num.naive_divide(&den)));

    let id = BenchmarkId::new("Fast", bench_param());
    group.bench_function(id, |b| b.iter(|| num.fast_divide(&den)));

    let id = BenchmarkId::new("Faster of the two", bench_param());
    group.bench_function(id, |b| b.iter(|| num.divide(&den)));

    group.finish();
}
