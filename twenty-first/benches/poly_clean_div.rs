use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::other::random_elements;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    // the following seem to outline the boundaries where clean division is faster
    targets = poly_clean_div< 9,  8>,
              poly_clean_div< 9,  9>,

              poly_clean_div<10,  8>,
              poly_clean_div<10,  9>,
              poly_clean_div<10, 10>,

              poly_clean_div<11,  8>,
              poly_clean_div<11,  9>,
              poly_clean_div<11, 10>,
              poly_clean_div<11, 11>,

              poly_clean_div<12,  8>,
              poly_clean_div<12,  9>,
              poly_clean_div<12, 11>,
              poly_clean_div<12, 12>,

              poly_clean_div<14,  8>,
              poly_clean_div<14,  9>,
              poly_clean_div<14, 13>,
              poly_clean_div<14, 14>,
);

fn poly_clean_div<const LOG2_NUM_DEG: usize, const LOG2_DEN_DEG: usize>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!(
        "Clean Division of Polynomials – \
         Dividend Degree: 2^{LOG2_NUM_DEG}, Divisor Degree: 2^{LOG2_DEN_DEG}"
    ));
    let bench_param = || format!("{LOG2_NUM_DEG}|{LOG2_DEN_DEG}");

    let num_roots = random_elements(1 << LOG2_NUM_DEG);
    let den_roots = &num_roots[..1 << LOG2_DEN_DEG];
    let num = Polynomial::<BFieldElement>::zerofier(&num_roots);
    let den = Polynomial::<BFieldElement>::zerofier(den_roots);

    let id = BenchmarkId::new("Long", bench_param());
    group.bench_function(id, |b| b.iter(|| num.divide(&den)));

    let id = BenchmarkId::new("Clean", bench_param());
    group.bench_function(id, |b| b.iter(|| num.clone().clean_divide(den.clone())));

    group.finish();
}
