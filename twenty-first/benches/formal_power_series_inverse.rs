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
    targets = fpsi,
);

fn fpsi(c: &mut Criterion) {
    let mut group = c.benchmark_group("Formal power series ring inverse");

    for log2_degree in 1..11 {
        let degree = (1 << log2_degree) - 1;

        let coefficients: Vec<BFieldElement> = random_elements(1 + degree);
        let polynomial = Polynomial::new(coefficients);

        let id = BenchmarkId::new("fpsi", format!("2^{log2_degree}"));
        group.bench_function(id, |b| {
            b.iter(|| {
                let precision = 1 << (log2_degree + 1);
                let polynomial = polynomial.clone();
                polynomial.formal_power_series_inverse_newton(precision)
            })
        });
    }
    group.finish();
}
