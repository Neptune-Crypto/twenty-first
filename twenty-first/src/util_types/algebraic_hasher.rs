use std::iter;

use itertools::Itertools;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use crate::shared_math::other::{self, is_power_of_two, roundup_nearest_multiple};
use crate::shared_math::rescue_prime_digest::{Digest, DIGEST_LENGTH};
use crate::shared_math::x_field_element::{XFieldElement, EXTENSION_DEGREE};

pub const RATE: usize = 10;

#[derive(Debug, PartialEq, Eq)]
pub enum Domain {
    VariableLength,
    FixedLength,
}

pub trait SpongeHasher: Clone + Send + Sync {
    type SpongeState;

    fn absorb_init(input: &[BFieldElement; RATE]) -> Self::SpongeState;
    fn absorb(sponge: &mut Self::SpongeState, input: &[BFieldElement; RATE]);
    fn squeeze(sponge: &mut Self::SpongeState) -> [BFieldElement; RATE];

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
        assert!(is_power_of_two(upper_bound));
        assert!(upper_bound <= BFieldElement::MAX as usize);
        let num_squeezes = roundup_nearest_multiple(num_indices, RATE) / RATE;
        (0..num_squeezes)
            .flat_map(|_| Self::squeeze(state))
            .take(num_indices)
            .map(|elem| elem.value() as usize % upper_bound)
            .collect()
    }

    fn sample_weights(state: &mut Self::SpongeState, num_weights: usize) -> Vec<XFieldElement> {
        let num_squeezes = roundup_nearest_multiple(num_weights * EXTENSION_DEGREE, RATE) / RATE;
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

    fn hash<T: Hashable>(value: &T) -> Digest {
        Self::hash_varlen(&value.to_sequence())
    }

    fn hash_varlen(input: &[BFieldElement]) -> Digest {
        // calculate padded length
        let padded_length = roundup_nearest_multiple(input.len() + 1, RATE);

        // pad input
        let input_iter = input.iter();
        let padding_iter = [&BFIELD_ONE].into_iter().chain(iter::repeat(&BFIELD_ZERO));
        let padded_input = input_iter
            .chain(padding_iter)
            .take(padded_length)
            .chunks(RATE);
        let mut padded_input_iter = padded_input.into_iter().map(|chunk| {
            chunk
                .into_iter()
                .copied()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        });

        // absorb_init
        let absorb_init_elems: [BFieldElement; RATE] =
            padded_input_iter.next().expect("at least one absorb");
        let mut sponge = Self::absorb_init(&absorb_init_elems);

        // absorb repeatedly
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
            "Non-inclusive upper bound {upper_bound} must be a power of two"
        );

        assert!(
            upper_bound <= 0x1_0000_0000,
            "Non-inclusive upper bound {upper_bound} must be at most 2^32",
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
    /// - `num_indices`: The number of indices to sample
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

impl Hashable for bool {
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

impl Hashable for u64 {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        let lo: u64 = self & 0xffff_ffff;
        let hi: u64 = self >> 32;

        vec![BFieldElement::new(lo), BFieldElement::new(hi)]
    }
}

impl Hashable for usize {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        assert_eq!(usize::BITS, u64::BITS);
        (*self as u64).to_sequence()
    }
}

impl Hashable for XFieldElement {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        self.coefficients.to_vec()
    }
}

impl Hashable for Digest {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        self.values().to_vec()
    }
}

impl Hashable for u128 {
    fn to_sequence(&self) -> Vec<BFieldElement> {
        let lo: u64 = (self & 0xffff_ffff) as u64;
        let lo_mid: u64 = ((self >> 32) & 0xffff_ffff) as u64;
        let hi_mid: u64 = ((self >> 64) & 0xffff_ffff) as u64;
        let hi: u64 = ((self >> 96) & 0xffff_ffff) as u64;

        vec![
            BFieldElement::new(lo),
            BFieldElement::new(lo_mid),
            BFieldElement::new(hi_mid),
            BFieldElement::new(hi),
        ]
    }
}

#[cfg(test)]
mod algebraic_hasher_tests {
    use num_traits::Zero;
    use rand::Rng;
    use rand_distr::{Distribution, Standard};

    use crate::shared_math::rescue_prime_digest::DIGEST_LENGTH;
    use crate::shared_math::x_field_element::EXTENSION_DEGREE;

    use super::*;

    fn to_sequence_prop<T>(smallest: T, largest: T)
    where
        T: Eq + Hashable,
        Standard: Distribution<T>,
    {
        let smallest_seq = smallest.to_sequence();
        let largest_seq = largest.to_sequence();
        assert_ne!(smallest_seq, largest_seq);
        assert_eq!(smallest_seq.len(), largest_seq.len());

        let mut rng = rand::thread_rng();
        let random_a: T = rng.gen();
        let random_b: T = rng.gen();

        if random_a != random_b {
            assert_ne!(random_a.to_sequence(), random_b.to_sequence());
        } else {
            assert_eq!(random_a.to_sequence(), random_b.to_sequence());
        }
    }

    #[test]
    fn to_sequence_test() {
        // bool
        to_sequence_prop(false, true);

        // u32
        to_sequence_prop(0u32, u32::MAX);

        // u64
        to_sequence_prop(0u64, u64::MAX);

        // BFieldElement
        let bfe_max = BFieldElement::new(BFieldElement::MAX);
        to_sequence_prop(BFIELD_ZERO, bfe_max);

        // XFieldElement
        let xfe_max = XFieldElement::new([bfe_max; EXTENSION_DEGREE]);
        to_sequence_prop(XFieldElement::zero(), xfe_max);

        // Digest
        let digest_zero = Digest::new([BFIELD_ZERO; DIGEST_LENGTH]);
        let digest_max = Digest::new([bfe_max; DIGEST_LENGTH]);
        to_sequence_prop(digest_zero, digest_max);

        // u128
        to_sequence_prop(0u128, u128::MAX);
    }
}
