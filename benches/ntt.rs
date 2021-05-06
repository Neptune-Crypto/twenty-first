use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use twenty_first::fft::prime_field_element::{PrimeField, PrimeFieldElement};

fn generate_ntt_input<'a>(
    log2_of_size: usize,
    prime: &'a PrimeField,
) -> Vec<twenty_first::fft::prime_field_element::PrimeFieldElement<'a>> {
    let size: usize = 2usize.pow(log2_of_size as u32);
    let mut output: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(size);
    for _ in 0..size {
        let rand = rand::random::<u32>() as i64 % prime.q;
        output.push(PrimeFieldElement::new(rand, prime));
    }

    output
}

fn ntt_forward(c: &mut Criterion) {
    static PRIME: i64 = 167772161; // 5 * 2^25 + !
    let prime_field: PrimeField = PrimeField::new(PRIME);
    let mut group = c.benchmark_group("ntt_forward");
    for log2_of_size in [18usize, 19, 20, 21, 22].iter() {
        let size = 2i64.pow(*log2_of_size as u32);
        let unity_root = prime_field.get_primitive_root_of_unity(size);
        group.throughput(Throughput::Bytes(*log2_of_size as u64));
        group
            .bench_with_input(
                BenchmarkId::from_parameter(log2_of_size),
                log2_of_size,
                |b, &log2_of_size| {
                    b.iter(|| {
                        twenty_first::fft::ntt_fft(
                            generate_ntt_input(log2_of_size as usize, &prime_field),
                            &unity_root.unwrap(),
                        )
                    });
                },
            )
            .sample_size(10);
    }
    group.finish();
}

criterion_group!(benches, ntt_forward);
criterion_main!(benches);
