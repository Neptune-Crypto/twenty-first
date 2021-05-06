use criterion::{black_box, criterion_group, criterion_main, Criterion};

use twenty_first::fft;
use twenty_first::fft::prime_field_element::{PrimeField, PrimeFieldElement};

fn fft_benchmark(c: &mut Criterion) {
    // Setup
    let range = 1048576; // 8192 ;
    let prime = 167772161; // = 5 * 2^25 + 1
    let mut ntt_input: Vec<PrimeFieldElement> = Vec::with_capacity(range);
    let new_field = PrimeField::new(prime);
    for _ in 0..range {
        ntt_input.push(PrimeFieldElement::new(
            rand::random::<u32>() as i64 % prime,
            &new_field,
        ))
    }
    let omega = fft::get_omega(&ntt_input);

    c.bench_function("fft", |b| {
        b.iter(|| fft::ntt_fft(black_box(ntt_input.clone()), black_box(&omega)))
    });
}

criterion_group!(benches, fft_benchmark);
criterion_main!(benches);
