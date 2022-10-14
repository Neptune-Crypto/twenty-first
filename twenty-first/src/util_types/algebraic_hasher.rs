use num_traits::Zero;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::other;
use crate::shared_math::rescue_prime_digest::Digest;
use crate::shared_math::x_field_element::XFieldElement;

pub trait AlgebraicHasher: Clone + Send + Sync {
    fn hash_slice(elements: &[BFieldElement]) -> Digest;
    fn hash_pair(left: &Digest, right: &Digest) -> Digest;
    fn hash<T: Hashable>(item: &T) -> Digest {
        Self::hash_slice(&item.to_sequence())
    }

    /// Given a uniform random `input` digest and a `max` that is a power of two,
    /// produce a uniform random number in the interval `[0; max)`. The input should
    /// be a Fiat-Shamir digest to ensure a high degree of randomness.
    ///
    /// - `input`: A hash digest
    /// - `upper_bound`: The (non-inclusive) upper bound (a power of two)
    fn sample_index(seed: &Digest, upper_bound: usize) -> usize {
        assert!(
            other::is_power_of_two(upper_bound),
            "Non-inclusive upper bound {} is a power of two",
            upper_bound
        );

        let bytes = bincode::serialize(&seed.values()).unwrap();
        let length_prefix_offset: usize = 8;
        let mut byte_counter: usize = length_prefix_offset;
        let mut max_bits: usize = other::log_2_floor(upper_bound as u128) as usize;
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
    fn sample_index_not_power_of_two(seed: &Digest, max: usize) -> usize {
        Self::sample_index(seed, (1 << 16) * other::roundup_npo2(max as u64) as usize) % max
    }

    /// Given a uniform random `seed` digest, a `max` that is a power of two,
    /// produce `count` uniform random numbers (sample indices) in the interval
    /// `[0; max)`. The seed should be a Fiat-Shamir digest to ensure a high
    /// degree of randomness.
    ///
    /// - `count`: The number of sample indices
    /// - `seed`: A hash digest
    /// - `max`: The (non-inclusive) upper bound (a power of two)
    fn sample_indices(count: usize, seed: &Digest, max: usize) -> Vec<usize> {
        Self::get_n_hash_rounds(seed, count)
            .iter()
            .map(|random_input| Self::sample_index(random_input, max))
            .collect()
    }

    fn get_n_hash_rounds(seed: &Digest, count: usize) -> Vec<Digest> {
        let mut digests = Vec::with_capacity(count);
        (0..count)
            .into_par_iter()
            .map(|i: usize| Self::hash_slice(&[seed.to_sequence(), i.to_sequence()].concat()))
            .collect_into_vec(&mut digests);

        digests
    }
}

pub trait Hashable {
    fn to_sequence(&self) -> Vec<BFieldElement>;
}

impl Hashable for Digest {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        self.values().to_vec()
    }
}

impl Hashable for u128 {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        // Only shifting with 63 *should* prevent collissions for all numbers below u64::MAX.
        vec![
            BFieldElement::zero(),
            BFieldElement::zero(),
            BFieldElement::new((self >> 126) as u64),
            BFieldElement::new(((self >> 63) % BFieldElement::MAX as u128) as u64),
            BFieldElement::new((self % BFieldElement::MAX as u128) as u64),
        ]
    }
}

impl Hashable for XFieldElement {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        self.coefficients.to_vec()
    }
}

// FIXME: Not safe.
impl Hashable for usize {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }
}

impl Hashable for u32 {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }
}

impl Hashable for BFieldElement {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        vec![*self]
    }
}
