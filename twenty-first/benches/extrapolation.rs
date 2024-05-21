use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use itertools::Itertools;
use num_traits::One;
use twenty_first::math::ntt::intt;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::PrimitiveRootOfUnity;
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
    let omega = BFieldElement::primitive_root_of_unity(codeword.len() as u64).unwrap();
    let log_domain_length = codeword.len().ilog2();
    let mut coefficients = codeword.to_vec();
    intt(&mut coefficients, omega, log_domain_length);
    let polynomial: Polynomial<BFieldElement> = Polynomial::new(coefficients)
        .scale(offset.inverse())
        .reduce_by_ntt_friendly_modulus(shift_coefficients, tail_length);
    polynomial.divide_and_conquer_batch_evaluate(zerofier_tree)
}

struct PreprocessingData {
    pub even_zerofiers_reduced: Vec<Polynomial<BFieldElement>>,
    pub odd_zerofiers_reduced: Vec<Polynomial<BFieldElement>>,
    pub shift_coefficients: Vec<BFieldElement>,
    pub tail_length: usize,
}

fn preprocess(
    codeword_length: usize,
    domain_offset: BFieldElement,
    modulus: &Polynomial<BFieldElement>,
) -> PreprocessingData {
    let n = codeword_length;
    let omega = BFieldElement::primitive_root_of_unity(n as u64).unwrap();
    let modular_squares = (0..n.ilog2())
        .scan(Polynomial::new(bfe_vec![0, 1]), |acc, _| {
            let yld = acc.clone();
            *acc = acc.multiply(acc).reduce(modulus);
            Some(yld)
        })
        .collect_vec();
    let even_zerofier_lcs = (0..n.ilog2()).map(|i| domain_offset.inverse().mod_pow(1 << i));
    let even_zerofiers_reduced = even_zerofier_lcs
        .zip(modular_squares.iter())
        .map(|(lc, sq)| sq.scalar_mul(lc) - Polynomial::one())
        .collect_vec();
    let odd_zerofier_lcs =
        (0..n.ilog2()).map(|i| (domain_offset * omega).inverse().mod_pow(1 << i));
    let odd_zerofiers_reduced = odd_zerofier_lcs
        .zip(modular_squares.iter())
        .map(|(lc, sq)| sq.scalar_mul(lc) - Polynomial::one())
        .collect_vec();

    // precompute NTT-friendly multiple of the modulus
    let (shift_coefficients, tail_length) = modulus.shift_factor_ntt_with_tail_size();

    PreprocessingData {
        even_zerofiers_reduced,
        odd_zerofiers_reduced,
        shift_coefficients,
        tail_length,
    }
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
    let preprocessing_data = preprocess(SIZE, offset, &modulus);

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
    group.bench_function(id, |b| {
        b.iter(|| {
            let minimal_interpolant =
        Polynomial::<BFieldElement>::fast_modular_coset_interpolate_with_zerofiers_and_ntt_friendly_multiple(
            &codeword,
                offset,
                &modulus,
                &preprocessing_data.even_zerofiers_reduced,
                &preprocessing_data.odd_zerofiers_reduced,
                &preprocessing_data.shift_coefficients,
                preprocessing_data.tail_length,
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
