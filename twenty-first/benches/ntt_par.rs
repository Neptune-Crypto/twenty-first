use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::WallTime;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use twenty_first::math::b_field_element::BFieldElement;
use twenty_first::math::ntt::ntt;
use twenty_first::math::ntt::ntt_with_precalculated_values;
use twenty_first::math::ntt::precalculate_ntt_values;
use twenty_first::math::other::random_elements;
use twenty_first::math::x_field_element::XFieldElement;

fn ntt_par(c: &mut Criterion) {
    const NUM_COLUMNS: usize = 200;
    let mut group = c.benchmark_group("par_ntt");
    group.sample_size(10);

    let log2_of_sizes: Vec<usize> = vec![18];

    for &log2_of_size in log2_of_sizes.iter() {
        let name_prefix = format!("naive_bfe_par_{NUM_COLUMNS}");
        bfield_naive_par(
            &mut group,
            BenchmarkId::new(&name_prefix, log2_of_size),
            log2_of_size,
            NUM_COLUMNS,
        );
    }

    for &log2_of_size in log2_of_sizes.iter() {
        let name_prefix = format!("bfe_with_precalculation_{NUM_COLUMNS}");
        bfield_with_precalculation(
            &mut group,
            BenchmarkId::new(&name_prefix, log2_of_size),
            log2_of_size,
            NUM_COLUMNS,
        );
    }

    for &log2_of_size in log2_of_sizes.iter() {
        let name_prefix = format!("naive_xfe_par_{NUM_COLUMNS}");
        xfield_naive_par(
            &mut group,
            BenchmarkId::new(&name_prefix, log2_of_size),
            log2_of_size,
            NUM_COLUMNS,
        );
    }

    for &log2_of_size in log2_of_sizes.iter() {
        let name_prefix = format!("xfe_with_precalculation_{NUM_COLUMNS}");
        xfield_with_precalculation(
            &mut group,
            BenchmarkId::new(&name_prefix, log2_of_size),
            log2_of_size,
            NUM_COLUMNS,
        );
    }

    group.finish();
}

fn bfield_naive_par(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
    num_columns: usize,
) {
    let size: usize = 1 << log2_of_size;
    let mut xs: Vec<Vec<BFieldElement>> = vec![random_elements(size); num_columns];

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            xs.par_iter_mut().for_each(|x| ntt(x));
        });
    });
}

fn xfield_naive_par(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
    num_columns: usize,
) {
    let size: usize = 1 << log2_of_size;
    let mut xs: Vec<Vec<XFieldElement>> = vec![random_elements(size); num_columns];

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            xs.par_iter_mut().for_each(|x| ntt(x));
        });
    });
}

fn bfield_with_precalculation(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
    num_columns: usize,
) {
    let size: usize = 1 << log2_of_size;
    let mut xs: Vec<Vec<BFieldElement>> = vec![random_elements(size); num_columns];

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let precalculated = precalculate_ntt_values(log2_of_size as u32);
            xs.par_iter_mut()
                .for_each(|x| ntt_with_precalculated_values(x, &precalculated));
        });
    });
}

fn xfield_with_precalculation(
    group: &mut BenchmarkGroup<WallTime>,
    bench_id: BenchmarkId,
    log2_of_size: usize,
    num_columns: usize,
) {
    let size: usize = 1 << log2_of_size;
    let mut xs: Vec<Vec<XFieldElement>> = vec![random_elements(size); num_columns];

    group.throughput(Throughput::Elements(size as u64));
    group.bench_with_input(bench_id, &size, |b, _| {
        b.iter(|| {
            let precalculated = precalculate_ntt_values(log2_of_size as u32);
            xs.par_iter_mut()
                .for_each(|x| ntt_with_precalculated_values(x, &precalculated));
        });
    });
}

criterion_group!(benches, ntt_par);
criterion_main!(benches);
