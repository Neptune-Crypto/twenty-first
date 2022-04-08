use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::prime_field_element_flexible::PrimeFieldElementFlexible;
use crate::shared_math::rescue_prime::RescuePrime;
use crate::shared_math::x_field_element::XFieldElement;
use crate::shared_math::{other, rescue_prime_params};
use crate::util_types::blake3_wrapper::Blake3Hash;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
/// A simple `Hasher` trait that allows for hashing one, two or many values into one digest.
///
/// The type of digest is determined by the `impl` of a given `Hasher`, and it requires that
/// `Value` has a `ToDigest<Self::Digest>` instance. For hashing hash digests, this `impl`
/// is quite trivial. For non-trivial cases it may include byte-encoding or hashing.
pub trait Hasher: Sized {
    type Digest: ToDigest<Self::Digest>
        + PartialEq
        + Clone
        + std::fmt::Debug
        + Serialize
        + DeserializeOwned
        + Sized;

    fn new() -> Self;
    fn hash<Value: ToDigest<Self::Digest>>(&mut self, input: &Value) -> Self::Digest;
    fn hash_pair(&mut self, left_input: &Self::Digest, right_input: &Self::Digest) -> Self::Digest;
    fn hash_many(&mut self, inputs: &[Self::Digest]) -> Self::Digest;
    // TODO: Consider moving the 'Self::Digest: ToDigest<Self::Digest>' constraint up.
    fn hash_with_salts<Value>(&mut self, mut digest: Self::Digest, salts: &[Value]) -> Self::Digest
    where
        Value: ToDigest<Self::Digest>,
        Self::Digest: ToDigest<Self::Digest>,
    {
        for salt in salts {
            digest = self.hash_pair(&digest, &salt.to_digest());
        }

        digest
    }

    // FIXME: Chunk these more efficiently by making it abstract and moving it into RescuePrimeProduction's impl.
    fn fiat_shamir<Value: ToDigest<Self::Digest>>(&mut self, items: &[Value]) -> Self::Digest {
        let digests: Vec<Self::Digest> = items.iter().map(|item| item.to_digest()).collect();
        self.hash_many(&digests)
    }

    /// Given a uniform random `input` digest and a `max` that is a power of two,
    /// produce a uniform random number in the interval `[0; max)`. The input should
    /// be a Fiat-Shamir digest to ensure a high degree of randomness.
    ///
    /// - `input`: A hash digest
    /// - `max`: The (non-inclusive) upper bound (a power of two)
    fn sample_index(&self, input: &Self::Digest, max: usize) -> usize {
        assert!(other::is_power_of_two(max));

        // FIXME: Default serialization of vectors uses length-prefixing, which means
        // the first 64 bits of the byte-serialization mostly contain zeroes and so are
        // not very random at all.
        let bytes = bincode::serialize(input).unwrap();
        let length_prefix_offset: usize = 8;
        let mut byte_counter: usize = length_prefix_offset;
        let mut max_bits: usize = other::log_2_floor(max as u64) as usize;
        let mut acc: usize = 0;

        while max_bits > 0 {
            let take = std::cmp::min(8, max_bits);
            let add = (bytes[byte_counter] >> (8 - take)) as usize;
            acc = (acc << take) + add;
            max_bits -= take;
            byte_counter += 1;
        }

        acc
    }

    // FIXME: This is not uniform.
    fn sample_index_not_power_of_two(&self, input: &Self::Digest, max: usize) -> usize {
        self.sample_index(input, other::roundup_npo2(max as u64) as usize) % max
    }

    /// Given a uniform random `seed` digest, a `max` that is a power of two,
    /// produce `count` uniform random numbers (sample indices) in the interval
    /// `[0; max)`. The seed should be a Fiat-Shamir digest to ensure a high
    /// degree of randomness.
    ///
    /// - `count`: The number of sample indices
    /// - `seed`: A hash digest
    /// - `max`: The (non-inclusive) upper bound (a power of two)
    fn sample_indices(&mut self, count: usize, seed: &Self::Digest, max: usize) -> Vec<usize>
    where
        u128: ToDigest<Self::Digest>,
    {
        self.get_n_hash_rounds(seed, count)
            .iter()
            .map(|random_input| self.sample_index(random_input, max))
            .collect()
    }

    // FIXME: Consider not using u128 here; we just do it out of convenience because the trait impl existed already.
    fn get_n_hash_rounds(&mut self, seed: &Self::Digest, count: usize) -> Vec<Self::Digest>
    where
        u128: ToDigest<Self::Digest>,
    {
        let mut digests = Vec::with_capacity(count);
        for i in 0..count {
            let digest = self.hash_pair(seed, &(i as u128).to_digest());
            digests.push(digest);
        }
        digests
    }
}

/// In order to hash arbitrary things using a `Hasher`, it must `impl ToDigest<Digest>`
/// where the concrete `Digest` is what's chosen for the `impl Hasher`. For example, in
/// order to
pub trait ToDigest<Digest> {
    fn to_digest(&self) -> Digest;
}

impl ToDigest<Blake3Hash> for PrimeFieldElementFlexible {
    fn to_digest(&self) -> Blake3Hash {
        let bytes = bincode::serialize(&self).unwrap();
        let digest = Blake3Hash(blake3::hash(bytes.as_slice()));

        digest
    }
}

// The specification for MMR from mimblewimble specifies that the
// node count is included in the hash preimage. Representing the
// node count as a u128 makes this possible
impl ToDigest<Blake3Hash> for u128 {
    fn to_digest(&self) -> Blake3Hash {
        (*self).into()
    }
}

impl ToDigest<Vec<BFieldElement>> for u128 {
    fn to_digest(&self) -> Vec<BFieldElement> {
        // Only shifting with 63 *should* prevent collissions for all
        // numbers below u64::MAX
        vec![
            BFieldElement::new((self >> 63) % u64::MAX as u128),
            BFieldElement::new(self % BFieldElement::MAX),
        ]
    }
}

impl ToDigest<Blake3Hash> for Blake3Hash {
    fn to_digest(&self) -> Blake3Hash {
        *self
    }
}

impl ToDigest<Blake3Hash> for BFieldElement {
    fn to_digest(&self) -> Blake3Hash {
        let bytes = bincode::serialize(&self).unwrap();
        let digest = Blake3Hash(blake3::hash(bytes.as_slice()));

        digest
    }
}

impl ToDigest<Blake3Hash> for XFieldElement {
    fn to_digest(&self) -> Blake3Hash {
        let bytes = bincode::serialize(&self).unwrap();
        let digest = blake3::hash(bytes.as_slice());

        digest.into()
    }
}

/// Trivial implementation when hashing `Vec<BFieldElement>` into `Vec<BFieldElement>`s.
impl ToDigest<Vec<BFieldElement>> for Vec<BFieldElement> {
    fn to_digest(&self) -> Vec<BFieldElement> {
        self.to_owned()
    }
}

/// Trivial implementation when hashing `Vec<BFieldElement>` into `BFieldElement`.
impl ToDigest<Vec<BFieldElement>> for BFieldElement {
    fn to_digest(&self) -> Vec<BFieldElement> {
        let mut digest = vec![*self];
        digest.append(&mut vec![BFieldElement::ring_zero(); 4]);
        digest
    }
}

impl ToDigest<Vec<BFieldElement>> for XFieldElement {
    fn to_digest(&self) -> Vec<BFieldElement> {
        self.coefficients.to_vec()
    }
}

// TODO: This 'Blake3Hash' wrapper looks messy, but at least it is contained here. Can we move it to 'blake3_wrapper'?
impl Hasher for blake3::Hasher {
    type Digest = Blake3Hash;

    fn new() -> Self {
        blake3::Hasher::new()
    }

    fn hash<Value: ToDigest<Self::Digest>>(&mut self, input: &Value) -> Self::Digest {
        let Blake3Hash(digest) = input.to_digest();
        self.reset();
        self.update(digest.as_bytes());
        Blake3Hash(self.finalize())
    }

    fn hash_pair(&mut self, left: &Self::Digest, right: &Self::Digest) -> Self::Digest {
        let Blake3Hash(left_digest) = left;
        let Blake3Hash(right_digest) = right;

        self.reset();
        self.update(left_digest.as_bytes());
        self.update(right_digest.as_bytes());
        Blake3Hash(self.finalize())
    }

    // Uses blake3::Hasher's sponge
    fn hash_many(&mut self, input: &[Self::Digest]) -> Self::Digest {
        self.reset();
        for digest in input {
            let Blake3Hash(digest) = digest;
            self.update(digest.as_bytes());
        }
        Blake3Hash(self.finalize())
    }
}

/// Since each struct can only have one `impl Hasher`, and we have many sets
/// of RescuePrime parameters, we create one here to act as the canonical
/// production version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescuePrimeProduction(pub RescuePrime);

impl Hasher for RescuePrimeProduction {
    type Digest = Vec<BFieldElement>;

    fn new() -> Self {
        RescuePrimeProduction(rescue_prime_params::rescue_prime_params_bfield_0())
    }

    fn hash<Value: ToDigest<Self::Digest>>(&mut self, input: &Value) -> Self::Digest {
        self.0.hash(&input.to_digest())
    }

    fn hash_pair(&mut self, left: &Self::Digest, right: &Self::Digest) -> Self::Digest {
        let input: Vec<BFieldElement> = vec![left.to_owned(), right.to_owned()].concat();
        self.0.hash(&input)
    }

    // TODO: Rewrite this when exposing RescuePrime's sponge
    fn hash_many(&mut self, inputs: &[Self::Digest]) -> Self::Digest {
        let mut acc = self.hash(&inputs[0]);
        for input in &inputs[1..] {
            acc = self.hash_pair(&acc, input);
        }
        acc
    }
}

#[cfg(test)]
pub mod test_simple_hasher {

    use rand::Rng;

    use crate::shared_math::traits::GetRandomElements;

    use super::*;

    #[test]
    fn u128_to_digest_test() {
        let one = 1u128;
        let bfields_one: Vec<BFieldElement> = one.to_digest();
        assert_eq!(2, bfields_one.len());
        assert_eq!(BFieldElement::ring_zero(), bfields_one[0]);
        assert_eq!(BFieldElement::ring_one(), bfields_one[1]);

        let beyond_bfield0 = u64::MAX as u128;
        let bfields: Vec<BFieldElement> = beyond_bfield0.to_digest();
        assert_eq!(2, bfields.len());
        assert_eq!(BFieldElement::ring_one(), bfields[0]);
        assert_eq!(BFieldElement::new(4294967295u128), bfields[1]);

        let beyond_bfield1 = BFieldElement::MAX + 1;
        let bfields: Vec<BFieldElement> = beyond_bfield1.to_digest();
        assert_eq!(2, bfields.len());
        assert_eq!(BFieldElement::ring_one(), bfields[0]);
        assert_eq!(BFieldElement::new(1u128), bfields[1]);
    }

    #[test]
    fn blake3_digest_from_u128_test() {
        // Verify that u128 values can be converted into Blake3 hash input digests
        let _128_val: Blake3Hash = 100u128.into();
        let Blake3Hash(inner) = _128_val;
        assert_eq!(
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 100
            ],
            inner.as_bytes()
        );
    }

    #[test]
    fn rescue_prime_equivalence_test() {
        let mut rpp: RescuePrimeProduction = RescuePrimeProduction::new();

        let input0: Vec<BFieldElement> = vec![1u128, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let expected_output0: Vec<BFieldElement> = vec![
            BFieldElement::new(16408223883448864076),
            BFieldElement::new(17937404513354951095),
            BFieldElement::new(17784658070603252681),
            BFieldElement::new(4690418723130302842),
            BFieldElement::new(3079713491308723285),
        ];
        assert_eq!(
            expected_output0,
            rpp.hash(&input0),
            "Hashing a single 1 produces a concrete 5-element output"
        );

        let input2_left: Vec<BFieldElement> = vec![3, 1, 4, 1, 5]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let input2_right: Vec<BFieldElement> = vec![9, 2, 6, 5, 3]
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        let expected_output2: Vec<BFieldElement> = vec![
            8224332136734371881,
            8736343702647113032,
            9660176071866133892,
            575034608412522142,
            13216022346578371396,
        ]
        .into_iter()
        .map(BFieldElement::new)
        .collect();

        assert_eq!(
            expected_output2,
            rpp.hash_pair(&input2_left, &input2_right),
            "Hashing two 5-element inputs produces a concrete 5-element output"
        );

        let inputs_2: Vec<Vec<BFieldElement>> = vec![vec![
            BFieldElement::new(3),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(1),
            BFieldElement::new(5),
        ]];
        assert_ne!(
            inputs_2[0],
            rpp.hash_many(&inputs_2),
            "Hashing many a single digest with hash_many must not return the input"
        );
    }

    #[test]
    fn random_index_test() {
        let mut hasher: RescuePrimeProduction = RescuePrimeProduction::new();

        // TODO: Test that this is uniformly random by samling.

        let mut rng = rand::thread_rng();
        for element in BFieldElement::random_elements(100, &mut rng) {
            let digest = hasher.hash(&vec![element]);
            let power_of_two = rng.gen_range(0..=32) as usize;
            let max = 1 << power_of_two;
            let index = hasher.sample_index(&digest, max);

            assert!(index < max);
        }

        println!("log_2_floor(256) = {}", other::log_2_floor(512));
    }
}
