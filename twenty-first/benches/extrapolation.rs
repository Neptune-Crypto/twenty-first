use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use itertools::Itertools;
use num_traits::One;
use twenty_first::math::ntt::intt;
use twenty_first::math::other::random_elements;
use twenty_first::math::traits::FiniteField;
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

#[allow(dead_code)]
fn iterative_barycentric(
    codeword: &[BFieldElement],
    points: &[BFieldElement],
) -> Vec<BFieldElement> {
    points
        .iter()
        .map(|&p| barycentric_evaluate(codeword, p))
        .collect_vec()
}

fn even_and_odd_zerofiers_and_shift_coefficients_with_tail_length(
    n: usize,
    offset: BFieldElement,
    modulus: &Polynomial<BFieldElement>,
) -> (
    Vec<Polynomial<BFieldElement>>,
    Vec<Polynomial<BFieldElement>>,
    Vec<BFieldElement>,
    usize,
) {
    let omega = BFieldElement::primitive_root_of_unity(n as u64).unwrap();
    let modular_squares = (0..n.ilog2())
        .scan(
            Polynomial::<BFieldElement>::new(vec![BFieldElement::from(0), BFieldElement::from(1)]),
            |acc: &mut Polynomial<BFieldElement>, _| {
                let yld = acc.clone();
                *acc = acc.multiply(acc).reduce(modulus);
                Some(yld)
            },
        )
        .collect_vec();
    let even_zerofier_lcs = (0..n.ilog2())
        .map(|i| offset.inverse().mod_pow(1u64 << i))
        .collect_vec();
    let even_zerofiers = even_zerofier_lcs
        .into_iter()
        .zip(modular_squares.iter())
        .map(|(lc, sq)| sq.scalar_mul(BFieldElement::from(lc.value())) - Polynomial::one())
        .collect_vec();
    let odd_zerofier_lcs = (0..n.ilog2())
        .map(|i| (offset * omega).inverse().mod_pow(1u64 << i))
        .collect_vec();
    let odd_zerofiers = odd_zerofier_lcs
        .into_iter()
        .zip(modular_squares.iter())
        .map(|(lc, sq)| sq.scalar_mul(BFieldElement::from(lc.value())) - Polynomial::one())
        .collect_vec();

    // precompute NTT-friendly multiple of the modulus
    let (shift_coefficients, tail_length) = modulus.shift_factor_ntt_with_tail_size();

    (
        even_zerofiers,
        odd_zerofiers,
        shift_coefficients,
        tail_length,
    )
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

pub fn coset_extrapolation(
    domain_offset: BFieldElement,
    codeword: &[BFieldElement],
    zerofier_tree: &ZerofierTree<BFieldElement>,
    even_zerofiers: &[Polynomial<BFieldElement>],
    odd_zerofiers: &[Polynomial<BFieldElement>],
    shift_coefficients: &[BFieldElement],
    tail_length: usize,
) -> Vec<BFieldElement> {
    let minimal_interpolant =
        Polynomial::<BFieldElement>::fast_modular_coset_interpolate_with_zerofiers_and_ntt_friendly_multiple(
            codeword,
            domain_offset,
            &zerofier_tree.zerofier(),
            even_zerofiers,
            odd_zerofiers,
            shift_coefficients,
            tail_length
        );
    minimal_interpolant.divide_and_conquer_batch_evaluate(&zerofier_tree)
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

    let zerofier_tree = ZerofierTree::new_from_domain(&eval_points);
    let modulus = zerofier_tree.zerofier();
    let (even_zerofiers, odd_zerofiers, shift_coefficients, tail_length) =
        even_and_odd_zerofiers_and_shift_coefficients_with_tail_length(SIZE, offset, &modulus);

    let id = BenchmarkId::new("INTT-then-Evaluate", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| {
            intt_then_evaluate(
                &codeword,
                offset,
                &zerofier_tree,
                &shift_coefficients,
                tail_length,
            )
        })
    });

    // Iterative barycentric is never close to faster.
    // let id = BenchmarkId::new("Iterative Barycentric", log2_of_size);
    // group.bench_function(id, |b| {
    //     b.iter(|| iterative_barycentric(&codeword, &eval_points))
    // });

    let id = BenchmarkId::new("Fast Codeword Extrapolation", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| {
            coset_extrapolation(
                offset,
                &codeword,
                &zerofier_tree,
                &even_zerofiers,
                &odd_zerofiers,
                &shift_coefficients,
                tail_length,
            )
        })
    });

    let id = BenchmarkId::new("Dispatcher (includes preprocessing)", log2_of_size);
    group.bench_function(id, |b| {
        b.iter(|| Polynomial::coset_extrapolate(offset, &codeword, &eval_points))
    });

    group.finish();
}
