use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::x_field_element::XFieldElement;
use crate::util_types::blake3_wrapper::Blake3Hash;
use itertools::Itertools;
use num_traits::Zero;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::de::DeserializeOwned;
use serde::Serialize;

pub trait ToVec<T> {
    fn to_vec(&self) -> Vec<T>;
}

/// A simple `Hasher` trait that allows for hashing one, two or many values into one digest.
///
/// The type of digest is determined by the `impl` of a given `Hasher`, and it requires that
/// `Value` has a `ToDigest<Self::Digest>` instance. For hashing hash digests, this `impl`
/// is quite trivial. For non-trivial cases it may include byte-encoding or hashing.
pub trait Hasher: Sized + Send + Sync + Clone {
    type T: Clone;
    type Digest: Hashable<Self::T>
        + PartialEq
        + Clone
        + std::fmt::Debug
        + Serialize
        + DeserializeOwned
        + Sized
        + Sync
        + Send
        + ToVec<Self::T>;

    fn new() -> Self;
    fn hash_sequence(&self, input: &[Self::T]) -> Self::Digest;
    fn hash_pair(&self, left_input: &Self::Digest, right_input: &Self::Digest) -> Self::Digest;

    fn hash_many(&self, inputs: &[Self::Digest]) -> Self::Digest {
        if inputs.is_empty() {
            panic!("Function hash_many has to take nonzero number of digests.")
        }
        let mut acc = inputs[0].clone();
        for inp in inputs[1..].iter() {
            acc = self.hash_pair(&acc, inp);
        }
        acc
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
        let mut max_bits: usize = other::log_2_floor(max as u128) as usize;
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
        self.sample_index(input, (1 << 16) * other::roundup_npo2(max as u64) as usize) % max
    }

    /// Given a uniform random `seed` digest, a `max` that is a power of two,
    /// produce `count` uniform random numbers (sample indices) in the interval
    /// `[0; max)`. The seed should be a Fiat-Shamir digest to ensure a high
    /// degree of randomness.
    ///
    /// - `count`: The number of sample indices
    /// - `seed`: A hash digest
    /// - `max`: The (non-inclusive) upper bound (a power of two)
    fn sample_indices(&self, count: usize, seed: &Self::Digest, max: usize) -> Vec<usize>
    where
        usize: Hashable<Self::T>,
    {
        self.get_n_hash_rounds(seed, count)
            .iter()
            .map(|random_input| self.sample_index(random_input, max))
            .collect()
    }

    // FIXME: Consider not using u128 here; we just do it out of convenience because the trait impl existed already.
    fn get_n_hash_rounds(&self, seed: &Self::Digest, count: usize) -> Vec<Self::Digest>
    where
        usize: Hashable<Self::T>,
    {
        let mut digests = Vec::with_capacity(count);
        (0..count)
            .into_par_iter()
            .map(|i| self.hash_sequence(&[seed.to_sequence(), i.to_sequence()].concat()))
            .collect_into_vec(&mut digests);

        digests
    }
}

/// In order to hash arbitrary things using a `Hasher`, it must `impl ToDigest<Digest>`
/// where the concrete `Digest` is what's chosen for the `impl Hasher`. For example, in
/// order to
pub trait Hashable<D> {
    fn to_sequence(&self) -> Vec<D>;
}

impl Hashable<u8> for usize {
    fn to_sequence(&self) -> Vec<u8> {
        (0..8)
            .map(|i| ((*self >> (8 * i)) & 0xff) as u8)
            .collect::<Vec<u8>>()
    }
}

// The specification for MMR from mimblewimble specifies that the
// node count is included in the hash preimage. Representing the
// node count as a u128 makes this possible

impl Hashable<u8> for u128 {
    fn to_sequence(&self) -> Vec<u8> {
        (0..16)
            .map(|i| ((*self >> (8 * i)) & 0xff) as u8)
            .collect::<Vec<u8>>()
    }
}

impl Hashable<u8> for XFieldElement {
    fn to_sequence(&self) -> Vec<u8> {
        let mut array: Vec<u8> = Vec::with_capacity(3 * 8);
        for i in 0..3 {
            for j in 0..8 {
                array.push(((u64::from(self.coefficients[i]) >> (j * 8)) & 0xff) as u8);
            }
        }
        array
    }
}

impl Hashable<u8> for BFieldElement {
    fn to_sequence(&self) -> Vec<u8> {
        // u64::from( BFieldElement ) -> u64 converts the
        // BFieldElement to canonical representation before casting.
        (0..8)
            .map(|i| ((u64::from(*self) >> (i * 8)) & 0xff) as u8)
            .collect::<Vec<u8>>()
    }
}

impl Hashable<u8> for Vec<BFieldElement> {
    fn to_sequence(&self) -> Vec<u8> {
        self.iter().flat_map(|b| b.to_sequence()).collect_vec()
    }
}

impl Hashable<BFieldElement> for u128 {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        // Only shifting with 63 *should* prevent collissions for all
        // numbers below u64::MAX
        vec![
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::new((self >> 126) as u64),
            BFieldElement::new(((self >> 63) % BFieldElement::MAX as u128) as u64),
            BFieldElement::new((self % BFieldElement::MAX as u128) as u64),
        ]
    }
}

impl Hashable<BFieldElement> for XFieldElement {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        self.coefficients.to_vec()
    }
}

impl Hashable<BFieldElement> for usize {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }
}

impl Hashable<BFieldElement> for BFieldElement {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        vec![*self]
    }
}

impl ToVec<u8> for Blake3Hash {
    fn to_vec(&self) -> Vec<u8> {
        self.0.as_bytes().to_vec()
    }
}

impl Hashable<u8> for Blake3Hash {
    fn to_sequence(&self) -> Vec<u8> {
        self.to_vec()
    }
}

// TODO: This 'Blake3Hash' wrapper looks messy, but at least it is contained here. Can we move it to 'blake3_wrapper'?
impl Hasher for blake3::Hasher {
    type Digest = Blake3Hash;
    type T = u8;

    fn new() -> Self {
        blake3::Hasher::new()
    }

    fn hash_sequence(&self, input: &[u8]) -> Self::Digest {
        let mut hasher = Self::new();
        hasher.update(input);
        Blake3Hash(hasher.finalize())
    }

    fn hash_pair(&self, left: &Self::Digest, right: &Self::Digest) -> Self::Digest {
        let Blake3Hash(left_digest) = left;
        let Blake3Hash(right_digest) = right;

        let mut hasher = Self::new();
        hasher.update(left_digest.as_bytes());
        hasher.update(right_digest.as_bytes());
        Blake3Hash(hasher.finalize())
    }
}

impl ToVec<BFieldElement> for Vec<BFieldElement> {
    fn to_vec(&self) -> Vec<BFieldElement> {
        self.clone()
    }
}

impl Hashable<BFieldElement> for Vec<BFieldElement> {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        self.clone()
    }
}

pub trait SamplableFrom<Digest> {
    fn sample(digest: &Digest) -> Self;
}

/// Sample pseudo-uniform BFieldElement from vector of uniform bytes.
/// Statistical distance from uniform: ~2^{-64}.
impl SamplableFrom<Vec<u8>> for BFieldElement {
    fn sample(digest: &Vec<u8>) -> Self {
        assert!(
            digest.len() >= 16,
            "Cannot sample pseudo-uniform BFieldElements from less than 16 bytes."
        );
        let mut integer = 0u64;
        digest[0..8].iter().for_each(|d| {
            integer *= 256u64;
            integer += *d as u64;
        });
        BFieldElement::new(integer)
    }
}

/// Sample pseudo-uniform XFieldElement from vector of uniform bytes.
/// Statistical distance from uniform: ~2^{-192}.
impl SamplableFrom<Vec<u8>> for XFieldElement {
    fn sample(digest: &Vec<u8>) -> Self {
        assert!(
            digest.len() >= 48,
            "Cannot sample pseudo-uniform XFieldElements from less than 48 bytes."
        );
        XFieldElement {
            coefficients: (0..3)
                .map(|i| BFieldElement::sample(&digest[16 * i..16 * (i + 1)].to_vec()))
                .collect::<Vec<BFieldElement>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl SamplableFrom<Vec<BFieldElement>> for BFieldElement {
    fn sample(digest: &Vec<BFieldElement>) -> Self {
        assert!(
            !digest.is_empty(),
            "Cannot sample BFieldElement uniformly from less than 1 BFieldElement."
        );
        digest[0]
    }
}

impl SamplableFrom<Vec<BFieldElement>> for XFieldElement {
    fn sample(digest: &Vec<BFieldElement>) -> Self {
        assert!(
            digest.len() >= 3,
            "Cannot sample XFieldElement uniformly from less than 3 BFieldElements."
        );
        XFieldElement {
            coefficients: digest[0..3].try_into().unwrap(),
        }
    }
}

#[cfg(test)]
pub mod test_simple_hasher {
    use crate::shared_math::rescue_prime_regular::RescuePrimeRegular;

    use super::*;

    #[test]
    fn blake3_digest_from_u128_test() {
        // Verify that u128 values can be converted into Blake3 hash input digests
        let _128_val: Blake3Hash = 100u128.into();
        let Blake3Hash(inner) = _128_val;
        assert_eq!(
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 100,
            ],
            inner.as_bytes()
        );
    }

    #[test]
    fn rescue_prime_pair_many_equivalence_test()
    where
        usize: Hashable<BFieldElement>,
    {
        type Digest = <RescuePrimeRegular as Hasher>::Digest;
        let rpr: RescuePrimeRegular = RescuePrimeRegular::new();
        let digest1: Digest = rpr.hash_sequence(&42usize.to_sequence());
        let digest2: Digest = rpr.hash_sequence(&((1 << 4 + 42) as usize).to_sequence());
        let digests = vec![digest1.clone(), digest2.clone()].concat();
        let hash_sequence_digest = rpr.hash_sequence(&digests);
        let hash_pair_digest = rpr.hash_pair(&digest1, &digest2);
        let hash_many_digest = rpr.hash_many(&[digest1, digest2]);
        println!("hash_sequence_digest = {:?}", hash_sequence_digest);
        println!("hash_pair_digest = {:?}", hash_pair_digest);
        println!("hash_many_digest = {:?}", hash_many_digest);
        assert_ne!(hash_sequence_digest, hash_many_digest);
        assert_ne!(hash_sequence_digest, hash_pair_digest);
        assert_eq!(hash_many_digest, hash_pair_digest);
    }
}
