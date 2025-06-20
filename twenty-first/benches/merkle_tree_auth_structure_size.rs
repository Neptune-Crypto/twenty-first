use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::Measurement;
use criterion::measurement::ValueFormatter;
use itertools::Itertools;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;
use twenty_first::prelude::BFieldCodec;
use twenty_first::prelude::MerkleTree;
use twenty_first::prelude::Tip5;

#[derive(Debug, Clone, Copy)]
struct AuthStructureEncodingLength(f64);

#[derive(Debug, Clone, Copy)]
struct AuthStructureEncodingLengthFormatter;

impl Measurement for AuthStructureEncodingLength {
    type Intermediate = ();
    type Value = Self;

    fn start(&self) -> Self::Intermediate {}

    fn end(&self, _i: Self::Intermediate) -> Self::Value {
        self.to_owned()
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        AuthStructureEncodingLength(v1.0 + v2.0)
    }

    fn zero(&self) -> Self::Value {
        AuthStructureEncodingLength(0.0)
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        value.0
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &AuthStructureEncodingLengthFormatter
    }
}

impl ValueFormatter for AuthStructureEncodingLengthFormatter {
    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "bfe"
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        _throughput: &Throughput,
        _values: &mut [f64],
    ) -> &'static str {
        "bfe/s"
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "bfe"
    }
}

fn auth_structure_len(c: &mut Criterion<AuthStructureEncodingLength>) {
    let mut rng = StdRng::seed_from_u64(0);

    let tree_height = 22;
    let num_leafs = 1 << tree_height;
    let leafs = (0..num_leafs).map(|_| rng.next_u64()).collect_vec();
    let leaf_digests = leafs.iter().map(Tip5::hash).collect_vec();
    let mt = MerkleTree::par_new(&leaf_digests).unwrap();

    let num_opened_indices = 40;
    let mut group = c.benchmark_group("merkle_tree_auth_structure_size");
    group.bench_function(
        BenchmarkId::new("auth_structure_size", num_leafs),
        |bencher| {
            bencher.iter_custom(|iters| {
                let mut total_len = AuthStructureEncodingLength(0.0);
                for _ in 0..iters {
                    let opened_indices = (0..num_opened_indices)
                        .map(|_| rng.random_range(0..num_leafs))
                        .collect_vec();
                    let auth_structure = mt.authentication_structure(&opened_indices).unwrap();
                    let this_len = auth_structure.encode().len();
                    let this_len = AuthStructureEncodingLength(this_len as f64);
                    total_len = total_len.add(&total_len, &this_len);
                }
                total_len
            })
        },
    );
}

fn auth_structure_len_measurements() -> Criterion<AuthStructureEncodingLength> {
    Criterion::default()
        .with_measurement(AuthStructureEncodingLength(0.0))
        .sample_size(100)
}

criterion_group!(
    name = benches;
    config =  auth_structure_len_measurements();
    targets = auth_structure_len
);
criterion_main!(benches);
