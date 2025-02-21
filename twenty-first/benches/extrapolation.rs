use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use twenty_first::math::ntt::intt;
use twenty_first::math::other::random_elements;
use twenty_first::math::zerofier_tree::ZerofierTree;
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

fn intt_then_evaluate(
    codeword: &[BFieldElement],
    offset: BFieldElement,
    zerofier_tree: &ZerofierTree<BFieldElement>,
    shift_coefficients: &[BFieldElement],
    tail_length: usize,
) -> Vec<BFieldElement> {
    let mut coefficients = codeword.to_vec();
    intt(&mut coefficients);
    let polynomial: Polynomial<BFieldElement> = Polynomial::new(coefficients)
        .scale(offset.inverse())
        .reduce_by_ntt_friendly_modulus(shift_coefficients, tail_length);
    polynomial.divide_and_conquer_batch_evaluate(zerofier_tree)
}

fn extrapolation<const SIZE: usize, const NUM_POINTS: usize>(c: &mut Criterion) {
    let log2_of_size = SIZE.ilog2();
    let mut group = c.benchmark_group(format!(
        "Extrapolation of length-{SIZE} codeword in {NUM_POINTS} Points"
    ));

    let codeword = random_elements(SIZE);
    let offset = BFieldElement::new(7);
    let eval_points: Vec<BFieldElement> = random_elements(NUM_POINTS);

    let zerofier_tree = ZerofierTree::new_from_domain(&eval_points);
    let modulus = zerofier_tree.zerofier();
    let preprocessing_data =
        Polynomial::fast_modular_coset_interpolate_preprocess(SIZE, offset, &modulus);

    let id = BenchmarkId::new("INTT-then-Evaluate", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| {
            intt_then_evaluate(
                &codeword,
                offset,
                &zerofier_tree,
                &preprocessing_data.shift_coefficients,
                preprocessing_data.tail_length,
            )
        })
    });

    // We used to have another benchmark here that used barycentric evaluation
    // (from `fri.rs` in repo triton-vm) inside of a loop over all points. It
    // was never close to faster.

    let id = BenchmarkId::new("Fast Codeword Extrapolation", log2_of_size);
    type BfePoly = Polynomial<'static, BFieldElement>;
    group.bench_function(id, |b| {
        b.iter(|| {
            let minimal_interpolant =
                BfePoly::fast_modular_coset_interpolate_with_zerofiers_and_ntt_friendly_multiple(
                    &codeword,
                    offset,
                    &modulus,
                    &preprocessing_data,
                );
            minimal_interpolant.divide_and_conquer_batch_evaluate(&zerofier_tree)
        })
    });

    let id = BenchmarkId::new("Dispatcher (includes preprocessing)", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| Polynomial::coset_extrapolate(offset, &codeword, &eval_points))
    });

    group.finish();
}
