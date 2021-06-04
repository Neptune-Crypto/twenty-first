use criterion::{
    criterion_group, criterion_main, AxisScale, Bencher, Benchmark, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use twenty_first::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
use twenty_first::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
use twenty_first::shared_math::prime_field_polynomial::PrimeFieldPolynomial;

fn generate_prime_field_lagrange_interpolation_input<'a>(
    number_of_points: usize,
    primitive_root_of_unity: &'a PrimeFieldElement<'a>,
) -> Vec<(PrimeFieldElement, PrimeFieldElement)> {
    let mut output: Vec<(PrimeFieldElement, PrimeFieldElement)> =
        Vec::with_capacity(number_of_points);
    for i in 0..number_of_points {
        let x = primitive_root_of_unity.mod_pow(i as i128);
        let y = rand::random::<i128>() % primitive_root_of_unity.field.q;
        output.push((x, PrimeFieldElement::new(y, &primitive_root_of_unity.field)));
    }

    output
}

fn integer_lagrange_interpolation_slow(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    static PRIME: i128 = 167772161;
    let pqr = PolynomialQuotientRing::new(256, PRIME);
    let field = PrimeField::new(PRIME);
    let mut group = c.benchmark_group("integer_lagrange_interpolation");
    for number_of_points in [256, 512, 1024, 1048576].iter() {
        let (primitive_root_of_unity_option, _) =
            field.get_primitive_root_of_unity(*number_of_points);
        let primitive_root_of_unity = primitive_root_of_unity_option.unwrap();
        let input: Vec<(PrimeFieldElement, PrimeFieldElement)> =
            generate_prime_field_lagrange_interpolation_input(
                *number_of_points as usize,
                &primitive_root_of_unity,
            );

        // Get input for the fast interpolation
        // y_values: &[i128],
        // prime: i128,
        // primitive_root_of_unity: i128,
        let y_values_fast: Vec<i128> = input.iter().map(|&x| x.1.value).collect();
        group.throughput(Throughput::Elements(*number_of_points as u64));
        if (*number_of_points < 2048) {
            group
                .bench_with_input(
                    BenchmarkId::new("Slow", number_of_points),
                    number_of_points,
                    |b, _| {
                        b.iter(|| {
                            PrimeFieldPolynomial::finite_field_lagrange_interpolation(&input, &pqr)
                        });
                    },
                )
                .sample_size(10)
                .plot_config(plot_config.clone());
        }
        group.throughput(Throughput::Elements(*number_of_points as u64));
        group
            .bench_with_input(
                BenchmarkId::new("Fast", number_of_points),
                number_of_points,
                |b, _| {
                    b.iter(|| {
                        twenty_first::fft::fast_polynomial_interpolate(
                            &y_values_fast,
                            primitive_root_of_unity.field.q,
                            primitive_root_of_unity.value,
                        )
                    });
                },
            )
            .sample_size(10);
    }
    group.finish();
}

criterion_group!(benches, integer_lagrange_interpolation_slow);
criterion_main!(benches);
