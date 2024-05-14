use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use itertools::Itertools;
use twenty_first::math::ntt::intt;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::FiniteField;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::prelude::*;

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = extrapolation<{ 1 << 18 }, { 1 << 6 }>,
              extrapolation<{ 1 << 18 }, { 1 << 7 }>,
              extrapolation<{ 1 << 18 }, { 1 << 8 }>,
              extrapolation<{ 1 << 19 }, { 1 << 6 }>,
              extrapolation<{ 1 << 19 }, { 1 << 7 }>,
              extrapolation<{ 1 << 19 }, { 1 << 8 }>,
              extrapolation<{ 1 << 20 }, { 1 << 6 }>,
              extrapolation<{ 1 << 20 }, { 1 << 7 }>,
              extrapolation<{ 1 << 20 }, { 1 << 8 }>,
);

fn intt_then_evaluate(codeword: &[BFieldElement], points: &[BFieldElement]) -> Vec<BFieldElement> {
    let omega = BFieldElement::primitive_root_of_unity(codeword.len() as u64).unwrap();
    let log_domain_length = codeword.len().ilog2();
    let mut coefficients = codeword.to_vec();
    intt(&mut coefficients, omega, log_domain_length);
    let polynomial: Polynomial<BFieldElement> = Polynomial::new(coefficients);
    polynomial.batch_evaluate(points)
}

fn iterative_barycentric(
    codeword: &[BFieldElement],
    points: &[BFieldElement],
) -> Vec<BFieldElement> {
    points
        .iter()
        .map(|&p| barycentric_evaluate(codeword, p))
        .collect_vec()
}

/// Lifted from repo triton-vm file fri.rs, which in turn [Credit]s Al Kindi.
///
/// [Credit]: https://github.com/0xPolygonMiden/miden-vm/issues/568
pub fn barycentric_evaluate(
    codeword: &[BFieldElement],
    indeterminate: BFieldElement,
) -> BFieldElement {
    let root_order = codeword.len().try_into().unwrap();
    let generator = BFieldElement::primitive_root_of_unity(root_order).unwrap();
    let domain_iter = (0..root_order)
        .scan(bfe!(1), |acc, _| {
            let to_yield = Some(*acc);
            *acc *= generator;
            to_yield
        })
        .collect_vec();

    let domain_shift = domain_iter.iter().map(|&d| indeterminate - d).collect();
    let domain_shift_inverses = BFieldElement::batch_inversion(domain_shift);
    let domain_over_domain_shift = domain_iter
        .into_iter()
        .zip(domain_shift_inverses)
        .map(|(d, inv)| d * inv);
    let numerator = domain_over_domain_shift
        .clone()
        .zip(codeword)
        .map(|(dsi, &abscis)| dsi * abscis)
        .sum::<BFieldElement>();
    let denominator = domain_over_domain_shift.sum::<BFieldElement>();
    numerator / denominator
}

fn extrapolation<const SIZE: usize, const NUM_POINTS: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE.ilog2();
    let mut group = c.benchmark_group(format!(
        "Extrapolation of length-{SIZE} codeword in {NUM_POINTS} Points"
    ));
    group.sample_size(10);

    let codeword = random_elements(SIZE);
    let offset = BFieldElement::new(7);
    let eval_points: Vec<BFieldElement> = random_elements(NUM_POINTS);

    let id = BenchmarkId::new("INTT-then-Evaluate", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| intt_then_evaluate(&codeword, &eval_points))
    });

    let id = BenchmarkId::new("Iterative Barycentric", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| iterative_barycentric(&codeword, &eval_points))
    });

    let id = BenchmarkId::new("Fast Codeword Extrapolation", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| Polynomial::<BFieldElement>::coset_extrapolation(offset, &codeword, &eval_points))
    });

    group.finish();
}
