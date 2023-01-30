use std::iter;

use itertools::Itertools;

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use crate::shared_math::other::{is_power_of_two, roundup_nearest_multiple};
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

pub trait AlgebraicHasher: SpongeHasher {
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
