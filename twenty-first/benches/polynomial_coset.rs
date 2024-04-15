use std::hint::black_box;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::WallTime;
use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::other::random_elements;
use twenty_first::math::polynomial::Polynomial;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::math::x_field_element::XFieldElement;

fn coset_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial coset");

    let sizes = [1 << 10, 1 << 17];

    for size in sizes {
        coset_evaluation_bfe_pol(
            &mut group,
            BenchmarkId::new("coset-evaluate bfe-pol", size),
            size,
        );
    }

    for size in sizes {
        coset_evaluation_xfe_pol(
            &mut group,
            BenchmarkId::new("coset-evaluate xfe-pol", size),
            size,
        );
    }

    for size in sizes {
        coset_interpolation_bfes(
            &mut group,
            BenchmarkId::new("coset-interpolate bfe-pol", size),
            size,
        );
    }

    for size in sizes {
        coset_interpolation_xfes(
            &mut group,
            BenchmarkId::new("coset-interpolate xfe-pol", size),
            size,
        );
    }

    group.finish();
}

fn coset_evaluation_bfe_pol(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    size: usize,
) {
    let bfe_pol: Polynomial<BFieldElement> = Polynomial::new(random_elements(size));
    let offset = BFieldElement::generator();
    let generator = BFieldElement::primitive_root_of_unity(size as u64).unwrap();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let _ = black_box(bfe_pol.fast_coset_evaluate(offset, generator, size));
        })
    });
    group.sample_size(10);
}

fn coset_evaluation_xfe_pol(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    size: usize,
) {
    let xfe_pol: Polynomial<XFieldElement> = Polynomial::new(random_elements(size));
    let offset = BFieldElement::generator();
    let generator = BFieldElement::primitive_root_of_unity(size as u64).unwrap();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let _ = black_box(xfe_pol.fast_coset_evaluate(offset, generator, size));
        })
    });
    group.sample_size(10);
}

fn coset_interpolation_bfes(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    size: usize,
) {
    let offset = BFieldElement::generator();
    let generator = BFieldElement::primitive_root_of_unity(size as u64).unwrap();
    let values: Vec<BFieldElement> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let _ = black_box(Polynomial::fast_coset_interpolate(
                offset, generator, &values,
            ));
        })
    });
    group.sample_size(10);
}

fn coset_interpolation_xfes(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    size: usize,
) {
    let offset = BFieldElement::generator();
    let generator = BFieldElement::primitive_root_of_unity(size as u64).unwrap();
    let values: Vec<XFieldElement> = random_elements(size);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let _ = black_box(Polynomial::fast_coset_interpolate(
                offset, generator, &values,
            ));
        })
    });
    group.sample_size(10);
}

criterion_group!(benches, coset_functions);
criterion_main!(benches);
