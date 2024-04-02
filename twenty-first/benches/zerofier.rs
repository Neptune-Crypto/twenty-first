use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use itertools::Itertools;

use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::polynomial::Polynomial;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = zerofier<0>,
              zerofier<10>,
              zerofier<100>,
              zerofier<200>,
              zerofier<500>,
              zerofier<700>,
              zerofier<1_000>,
              zerofier<10_000>,
);

fn zerofier<const SIZE: usize>(c: &mut Criterion) {
    let group_name = format!("Various Zerofiers with {SIZE} Roots");
    let mut group = c.benchmark_group(group_name);

    let roots = random_elements::<BFieldElement>(SIZE * 2);
    let roots = roots.into_iter().unique().take(SIZE).collect_vec();
    assert_eq!(SIZE, roots.len());

    let id = BenchmarkId::new("Naïve", SIZE);
    group.bench_function(id, |b| b.iter(|| Polynomial::naive_zerofier(&roots)));

    let id = BenchmarkId::new("Fast", SIZE);
    group.bench_function(id, |b| b.iter(|| Polynomial::fast_zerofier(&roots)));

    let id = BenchmarkId::new("Faster of the two", SIZE);
    group.bench_function(id, |b| b.iter(|| Polynomial::zerofier(&roots)));

    group.finish();
}
