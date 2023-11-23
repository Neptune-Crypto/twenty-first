use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BatchSize;
use criterion::Criterion;
use rand::thread_rng;
use rand::Rng;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::polynomial::Polynomial;

criterion_main!(benches);
criterion_group!(
    benches,
    naive_zerofier<0>,
    naive_zerofier<10>,
    naive_zerofier<100>,
    naive_zerofier<200>,
    naive_zerofier<500>,
    naive_zerofier<700>,
    naive_zerofier<1_000>,
    naive_zerofier<10_000>,
    fast_zerofier<0>,
    fast_zerofier<10>,
    fast_zerofier<100>,
    fast_zerofier<200>,
    fast_zerofier<500>,
    fast_zerofier<700>,
    fast_zerofier<1_000>,
    fast_zerofier<10_000>,
);

fn naive_zerofier<const N: usize>(c: &mut Criterion) {
    let mut rng = thread_rng();
    let id = format!("zerofier {N}");
    c.bench_function(&id, |b| {
        b.iter_batched(
            || rng.gen(),
            |v: [BFieldElement; N]| Polynomial::naive_zerofier(&v),
            BatchSize::SmallInput,
        )
    });
}

fn fast_zerofier<const N: usize>(c: &mut Criterion) {
    let mut rng = thread_rng();
    let id = format!("fast_zerofier {N}");
    c.bench_function(&id, |b| {
        b.iter_batched(
            || rng.gen(),
            |v: [BFieldElement; N]| Polynomial::fast_zerofier(&v),
            BatchSize::SmallInput,
        )
    });
}
