use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use twenty_first::shared_math::{b_field_element::BFieldElement, traits::GetPrimitiveRootOfUnity};

fn generate_ntt_input<'a>(log2_of_size: usize) -> Vec<BFieldElement> {
    let size: usize = 2usize.pow(log2_of_size as u32);
    let mut output: Vec<BFieldElement> = Vec::with_capacity(size);
    for _ in 0..size {
        let rand = rand::random::<u64>() as u128 % BFieldElement::QUOTIENT;
        output.push(BFieldElement::new(rand));
    }

    output
}

fn ntt_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_bfield");
    for &log2_of_size in [16usize, 18, 20].iter() {
        let size = 2u128.pow(log2_of_size as u32);
        let (unity_root, _) = BFieldElement::ring_zero().get_primitive_root_of_unity(size);
        let input = generate_ntt_input(log2_of_size);
        group.throughput(Throughput::Elements(size as u64));
        group
            .bench_with_input(BenchmarkId::from_parameter(log2_of_size), &size, |b, _| {
                b.iter(|| twenty_first::shared_math::ntt::ntt(&input, &unity_root.unwrap()));
            })
            .sample_size(10);
    }
    group.finish();
}

criterion_group!(benches, ntt_forward);
criterion_main!(benches);
