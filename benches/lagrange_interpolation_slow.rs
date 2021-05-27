use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashSet;
use std::hash::Hash;
use twenty_first::homomorphic_encryption::polynomial_quotient_ring::PolynomialQuotientRing;

fn generate_lagrange_interpolation_input(
    log2_number_of_points: usize,
    modulus: i128,
) -> Vec<(i128, i128)> {
    fn has_unique_elements<T>(iter: T) -> bool
    where
        T: IntoIterator,
        T::Item: Eq + Hash,
    {
        let mut uniq = HashSet::new();
        iter.into_iter().all(move |x| uniq.insert(x))
    }

    let number_of_points = 2usize.pow(log2_number_of_points as u32);
    let mut output: Vec<(i128, i128)> = Vec::with_capacity(number_of_points);
    for _ in 0..number_of_points {
        let x = rand::random::<i128>() % modulus;
        let y = rand::random::<i128>() % modulus;
        output.push((x, y));
    }

    // disallow repeated x values
    if !has_unique_elements(output.iter().map(|&x| x.0)) {
        panic!("Found repeated x value in: {:?}", output);
    }

    output
}

fn lagrange_interpolation_slow(c: &mut Criterion) {
    static PRIME: i128 = 984445284980888355177813739321;
    let pqr = PolynomialQuotientRing::new(256, PRIME);
    let mut group = c.benchmark_group("lagrange_interpolation_slow");
    for log2_of_size in [3usize, 4, 5, 6, 7, 8, 9].iter() {
        let number_of_points = 2usize.pow(*log2_of_size as u32);
        group.throughput(Throughput::Elements(number_of_points as u64));
        group
            .bench_with_input(
                BenchmarkId::from_parameter(log2_of_size),
                log2_of_size,
                |b, &log2_of_size| {
                    let input = generate_lagrange_interpolation_input(log2_of_size as usize, PRIME);
                    b.iter(|| {
                        twenty_first::homomorphic_encryption::polynomial::Polynomial::lagrange_interpolation(
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

criterion_group!(benches, lagrange_interpolation_slow);
criterion_main!(benches);
