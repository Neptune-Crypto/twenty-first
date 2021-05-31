use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use twenty_first::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;

fn generate_integer_lagrange_interpolation_input(
    number_of_points: usize,
    modulus: i128,
) -> Vec<(i128, i128)> {
    let mut output: Vec<(i128, i128)> = Vec::with_capacity(number_of_points);
    for i in 0..number_of_points {
        let x = -(number_of_points as i128) / 2 + i as i128;
        let y = rand::random::<i128>() % modulus;
        output.push((x, y));
    }

    output
}

fn integer_lagrange_interpolation_slow(c: &mut Criterion) {
    static PRIME: i128 = 7;
    let pqr = PolynomialQuotientRing::new(256, PRIME);
    let mut group = c.benchmark_group("integer_lagrange_interpolation_slow");
    // For non-finite fields, 18 is about as high as we can go without overflowing
    for number_of_points in [4, 5, 6, 8, 10, 12, 14, 16].iter() {
        group.throughput(Throughput::Elements(*number_of_points as u64));
        group
            .bench_with_input(
                BenchmarkId::from_parameter(number_of_points),
                number_of_points,
                |b, &log2_of_size| {
                    let input = generate_integer_lagrange_interpolation_input(log2_of_size as usize, PRIME);
                    b.iter(|| {
                        twenty_first::shared_math::prime_field_polynomial::PrimeFieldPolynomial::integer_lagrange_interpolation(
                            &input,
                            &pqr,
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
