use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use itertools::izip;
use twenty_first::shared_math::{
    b_field_element::BFieldElement, other::log_2_floor, polynomial::Polynomial,
    traits::GetRandomElements,
};

fn polynomial_square(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("polynomial_square");
    let sizes: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192];
    let log_of_sizes: Vec<usize> = sizes
        .clone()
        .into_iter()
        .map(|x| log_2_floor(x as u64) as usize)
        .collect();
    let mut rng = rand::thread_rng();
    for (size, log_of_size) in izip!(sizes, log_of_sizes) {
        let polynomial = Polynomial {
            coefficients: BFieldElement::random_elements(size, &mut rng),
        };
        group.throughput(Throughput::Elements(size as u64));
        group
            .bench_with_input(
                BenchmarkId::new("square", log_of_size),
                &log_of_size,
                |b, _| {
                    b.iter(|| polynomial.square());
                },
            )
            .sample_size(10)
            .plot_config(plot_config.clone());
        group
            .bench_with_input(
                BenchmarkId::new("fast_square", log_of_size),
                &log_of_size,
                |b, _| {
                    b.iter(|| polynomial.fast_square());
                },
            )
            .sample_size(10)
            .plot_config(plot_config.clone());
    }
    group.finish();
}

criterion_group!(benches, polynomial_square);
criterion_main!(benches);
