use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use twenty_first::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
use twenty_first::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
use twenty_first::shared_math::prime_field_polynomial::PrimeFieldPolynomial;

fn generate_prime_field_lagrange_interpolation_input(
    number_of_points: usize,
    modulus: i128,
    field: &'_ PrimeField,
) -> Vec<(PrimeFieldElement, PrimeFieldElement)> {
    let mut output: Vec<(PrimeFieldElement, PrimeFieldElement)> =
        Vec::with_capacity(number_of_points);
    for i in 0..number_of_points {
        let x = -(number_of_points as i128) / 2 + i as i128;
        let y = rand::random::<i128>() % modulus;
        output.push((
            PrimeFieldElement::new(x, field),
            PrimeFieldElement::new(y, field),
        ));
    }

    output
}

fn integer_lagrange_interpolation_slow(c: &mut Criterion) {
    static PRIME: i128 = 9999889;
    let pqr = PolynomialQuotientRing::new(256, PRIME);
    let field = PrimeField::new(PRIME);
    let mut group = c.benchmark_group("integer_lagrange_interpolation_slow");
    for number_of_points in [10, 20, 40, 80, 160, 320, 640].iter() {
        group.throughput(Throughput::Elements(*number_of_points as u64));
        group
            .bench_with_input(
                BenchmarkId::from_parameter(number_of_points),
                number_of_points,
                |b, &log2_of_size| {
                    let input = generate_prime_field_lagrange_interpolation_input(
                        log2_of_size as usize,
                        PRIME,
                        &field,
                    );
                    b.iter(|| {
                        PrimeFieldPolynomial::finite_field_lagrange_interpolation(&input, &pqr)
                    });
                },
            )
            .sample_size(10);
    }
    group.finish();
}

criterion_group!(benches, integer_lagrange_interpolation_slow);
criterion_main!(benches);
