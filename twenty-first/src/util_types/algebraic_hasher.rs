use std::iter;

use itertools::Itertools;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use crate::shared_math::other::{self, roundup_nearest_multiple};
use crate::shared_math::rescue_prime_digest::{Digest, DIGEST_LENGTH};
use crate::shared_math::x_field_element::{XFieldElement, EXTENSION_DEGREE};

pub const RATE: usize = 10;

pub trait SpongeHasher: Clone + Send + Sync {
    type SpongeState;

    fn absorb_init(input: &[BFieldElement]) -> Self::SpongeState;
    fn absorb(state: &mut Self::SpongeState, input: &[BFieldElement]);
    fn squeeze(state: &mut Self::SpongeState) -> [BFieldElement; RATE];

    /// Given a sponge state and an `upper_bound` that is a power of two,
    /// produce `num_indices` uniform random numbers (sample indices) in
    /// the interval `[0; upper_bound)`.
    ///
    /// - `state`: A `Self::SpongeState`
    /// - `upper_bound`: The (non-inclusive) upper bound (a power of two)
    /// - `num_indices`: The number of sample indices
    fn sample_indices(
        state: &mut Self::SpongeState,
        upper_bound: usize,
        num_indices: usize,
    ) -> Vec<usize> {
        assert!(upper_bound <= BFieldElement::MAX as usize);
        let num_squeezes = num_indices / RATE;
        (0..num_squeezes)
            .flat_map(|_| Self::squeeze(state))
            .take(num_indices)
            .map(|elem| elem.value() as usize % upper_bound)
            .collect()
    }

    fn sample_weights(state: &mut Self::SpongeState, num_weights: usize) -> Vec<XFieldElement> {
        let num_squeezes = (num_weights * EXTENSION_DEGREE) / RATE;
        (0..num_squeezes)
            .map(|_| Self::squeeze(state))
            .flat_map(|elems| {
                vec![
                    XFieldElement::new([elems[0], elems[1], elems[2]]),
                    XFieldElement::new([elems[3], elems[4], elems[5]]),
                    XFieldElement::new([elems[6], elems[7], elems[8]]),
                    // spill 1 element, elems[9], per squeeze
                ]
            })
            .collect()
    }
}

pub trait AlgebraicHasherNew: SpongeHasher {
    fn hash_pair(left: &Digest, right: &Digest) -> Digest;

    fn hash_varlen(input: &[BFieldElement]) -> Digest {
        // calculate padded length
        let padded_length = roundup_nearest_multiple(input.len() + 1, RATE);

        // apply padding
        let input_iter = input.iter();
        let padding_iter = [&BFIELD_ONE].into_iter().chain(iter::repeat(&BFIELD_ZERO));
        let padded_input = input_iter
            .chain(padding_iter)
            .take(padded_length)
            .chunks(RATE);
        let mut padded_input_iter = padded_input
            .into_iter()
            .map(|chunk| chunk.into_iter().copied().collect::<Vec<_>>());

        // absorb_init once
        let absorb_init_elems: Vec<BFieldElement> =
            padded_input_iter.next().expect("at least one absorb");
        let mut sponge = Self::absorb_init(&absorb_init_elems);

        // absorb variably
        for absorb_elems in padded_input_iter {
            Self::absorb(&mut sponge, &absorb_elems);
        }

        // squeeze
        let produce: [BFieldElement; RATE] = Self::squeeze(&mut sponge);

        Digest::new((&produce[..DIGEST_LENGTH]).try_into().unwrap())
    }
}

pub trait AlgebraicHasher: Clone + Send + Sync {
    fn hash_slice(elements: &[BFieldElement]) -> Digest;
    fn hash_pair(left: &Digest, right: &Digest) -> Digest;
    fn hash<T: Hashable>(item: &T) -> Digest {
        Self::hash_slice(&item.to_sequence())
    }

    /// Given a uniform random `sample` and an `upper_bound` that is a power of
    /// two, produce a uniform random number in the interval `[0; upper_bound)`.
    ///
    /// The sample should have a high degree of randomness.
    ///
    /// - `sample`: A hash digest
    /// - `upper_bound`: The (non-inclusive) upper bound (a power of two)
    fn sample_index(sample: &Digest, upper_bound: usize) -> usize {
        assert!(
            other::is_power_of_two(upper_bound),
            "Non-inclusive upper bound {} must be a power of two",
            upper_bound
        );

        assert!(
            upper_bound <= 0x1_0000_0000,
            "Non-inclusive upper bound {} must be at most 2^32",
            upper_bound,
        );

        sample.values()[4].value() as usize % upper_bound
    }

    // FIXME: This is not uniform.
    fn sample_index_not_power_of_two(seed: &Digest, upper_bound: usize) -> usize {
        Self::sample_index(
            seed,
            (1 << 16) * other::roundup_npo2(upper_bound as u64) as usize,
        ) % upper_bound
    }

    /// Given a uniform random `seed` digest, an `upper_bound` that is a power of two,
    /// produce `num_indices` uniform random numbers (sample indices) in the interval
    /// `[0; upper_bound)`. The seed should be a Fiat-Shamir digest to ensure a high
    /// degree of randomness.
    ///
    /// - `seed`: A hash `Digest`
    /// - `upper_bound`: The (non-inclusive) upper bound (a power of two)
    /// - `num_indices`: The number of sample indices
    fn sample_indices(seed: &Digest, upper_bound: usize, num_indices: usize) -> Vec<usize> {
        Self::get_n_hash_rounds(seed, num_indices)
            .iter()
            .map(|random_input| Self::sample_index(random_input, upper_bound))
            .collect()
    }

    /// Given a uniform random `seed` digest, produce `num_weights` uniform random
    /// `XFieldElement`s (sample weights). The seed should be a Fiat-Shamir digest
    /// to ensure a high degree of randomness.
    ///
    /// - `seed`: A hash `Digest`
    /// - `num_weights`: The number of sample weights
    fn sample_weights(seed: &Digest, num_weights: usize) -> Vec<XFieldElement> {
        Self::get_n_hash_rounds(seed, num_weights)
            .iter()
            .map(XFieldElement::sample)
            .collect()
    }

    fn get_n_hash_rounds(seed: &Digest, count: usize) -> Vec<Digest> {
        assert!(count <= BFieldElement::MAX as usize);
        let mut digests = Vec::with_capacity(count);
        (0..count)
            .into_par_iter()
            .map(|counter: usize| {
                let counter = Digest::new([
                    BFieldElement::new(counter as u64),
                    BFIELD_ZERO,
                    BFIELD_ZERO,
                    BFIELD_ZERO,
                    BFIELD_ZERO,
                ]);
                Self::hash_pair(&counter, seed)
            })
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
            BFIELD_ZERO,
            BFIELD_ZERO,
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

#[cfg(test)]
mod algebraic_hasher_tests {
    use num_traits::Zero;
    use rand::Rng;

    use crate::shared_math::rescue_prime_digest::DIGEST_LENGTH;
    use crate::shared_math::x_field_element::EXTENSION_DEGREE;

    use super::*;

    #[test]
    fn to_sequence_length_test() {
        let mut rng = rand::thread_rng();
        let bfe_max = BFieldElement::new(BFieldElement::MAX);

        let some_digest: Digest = rng.gen();
        let zero_digest: Digest = Digest::new([BFieldElement::zero(); DIGEST_LENGTH]);
        let max_digest: Digest = Digest::new([bfe_max; DIGEST_LENGTH]);
        for digest in [some_digest, zero_digest, max_digest] {
            assert_eq!(DIGEST_LENGTH, digest.to_sequence().len());
        }

        let some_u128: u128 = rng.gen();
        let zero_u128: u128 = 0;
        let max_u128: u128 = u128::MAX;
        for u128 in [some_u128, zero_u128, max_u128] {
            assert_eq!(DIGEST_LENGTH, u128.to_sequence().len());
        }

        let some_xfe: XFieldElement = rng.gen();
        let zero_xfe: XFieldElement = XFieldElement::zero();
        let max_xfe: XFieldElement = XFieldElement::new([bfe_max; EXTENSION_DEGREE]);
        for xfe in [some_xfe, zero_xfe, max_xfe] {
            assert_eq!(EXTENSION_DEGREE, xfe.to_sequence().len());
        }
    }
}
