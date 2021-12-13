use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use twenty_first::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};

fn generate_ntt_input<'a>(
    log2_of_size: usize,
    prime: &'a PrimeField,
) -> Vec<twenty_first::shared_math::prime_field_element::PrimeFieldElement<'a>> {
    let size: usize = 2usize.pow(log2_of_size as u32);
    let mut output: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(size);
    for _ in 0..size {
        let rand = rand::random::<u32>() as i128 % prime.q;
        output.push(PrimeFieldElement::new(rand, prime));
    }

    output
}

fn ntt_forward(c: &mut Criterion) {
    static PRIME: i128 = 167772161; // 5 * 2^25 + !
    let prime_field: PrimeField = PrimeField::new(PRIME);
    let mut group = c.benchmark_group("ntt");
    for &log2_of_size in [16usize, 18, 20].iter() {
        let size = 2i128.pow(log2_of_size as u32);
        let (unity_root, _) = prime_field.get_primitive_root_of_unity(size);
        let input = generate_ntt_input(log2_of_size, &prime_field);
        group.throughput(Throughput::Elements(size as u64));
        group
            .bench_with_input(BenchmarkId::from_parameter(log2_of_size), &size, |b, _| {
                b.iter(|| twenty_first::fft::ntt_fft(input.clone(), &unity_root.unwrap()));
            })
            .sample_size(10);
    }
    group.finish();
}

criterion_group!(benches, ntt_forward);
criterion_main!(benches);
