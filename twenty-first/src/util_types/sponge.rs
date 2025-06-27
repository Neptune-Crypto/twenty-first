use std::fmt::Debug;

use num_traits::ConstOne;
use num_traits::ConstZero;

use crate::math::b_field_element::BFieldElement;

pub const RATE: usize = 10;

/// The hasher [Domain] differentiates between the modes of hashing.
///
/// The main purpose of declaring the domain is to prevent collisions between different types of
/// hashing by introducing defining differences in the way the hash function's internal state
/// (e.g. a sponge state's capacity) is initialized.
#[derive(Debug, PartialEq, Eq)]
pub enum Domain {
    /// The `VariableLength` domain is used for hashing objects that potentially serialize to more
    /// than [`RATE`] number of field elements.
    VariableLength,

    /// The `FixedLength` domain is used for hashing objects that always fit
    /// within [RATE] number of fields elements, e.g. a pair of
    /// [Digest](crate::prelude::Digest)s.
    FixedLength,
}

/// A [cryptographic sponge][sponge]. Should only be based on a cryptographic permutation, e.g.,
/// [`Tip5`][tip5].
///
/// [sponge]: https://keccak.team/files/CSF-0.1.pdf
/// [tip5]: crate::prelude::Tip5
pub trait Sponge: Clone + Debug + Default + Send + Sync {
    const RATE: usize;

    fn init() -> Self;

    fn absorb(&mut self, input: [BFieldElement; RATE]);

    fn squeeze(&mut self) -> [BFieldElement; RATE];

    fn pad_and_absorb_all(&mut self, input: &[BFieldElement]) {
        let mut chunks = input.chunks_exact(RATE);
        for chunk in chunks.by_ref() {
            // `chunks_exact` yields only chunks of length RATE; unwrap is fine
            self.absorb(chunk.try_into().unwrap());
        }

        // Pad input with [1, 0, 0, …] – padding is at least one element.
        // Since remainder's len is at most `RATE - 1`, the indexing is safe.
        let remainder = chunks.remainder();
        let mut last_chunk = const { [BFieldElement::ZERO; RATE] };
        last_chunk[..remainder.len()].copy_from_slice(remainder);
        last_chunk[remainder.len()] = BFieldElement::ONE;
        self.absorb(last_chunk);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::ops::Mul;

    use rand::Rng;
    use rand::distr::Distribution;
    use rand::distr::StandardUniform;

    use super::*;
    use crate::math::x_field_element::EXTENSION_DEGREE;
    use crate::prelude::BFieldCodec;
    use crate::prelude::XFieldElement;
    use crate::tip5::Digest;
    use crate::tip5::Tip5;

    fn encode_prop<T>(smallest: T, largest: T)
    where
        T: Eq + BFieldCodec,
        StandardUniform: Distribution<T>,
    {
        let smallest_seq = smallest.encode();
        let largest_seq = largest.encode();
        assert_ne!(smallest_seq, largest_seq);
        assert_eq!(smallest_seq.len(), largest_seq.len());

        let mut rng = rand::rng();
        let random_a: T = rng.random();
        let random_b: T = rng.random();

        if random_a != random_b {
            assert_ne!(random_a.encode(), random_b.encode());
        } else {
            assert_eq!(random_a.encode(), random_b.encode());
        }
    }

    #[test]
    fn to_sequence_test() {
        // bool
        encode_prop(false, true);

        // u32
        encode_prop(0u32, u32::MAX);

        // u64
        encode_prop(0u64, u64::MAX);

        // BFieldElement
        let bfe_max = BFieldElement::new(BFieldElement::MAX);
        encode_prop(BFieldElement::ZERO, bfe_max);

        // XFieldElement
        let xfe_max = XFieldElement::new([bfe_max; EXTENSION_DEGREE]);
        encode_prop(XFieldElement::ZERO, xfe_max);

        // Digest
        let digest_max = Digest::new([bfe_max; Digest::LEN]);
        encode_prop(Digest::ALL_ZERO, digest_max);

        // u128
        encode_prop(0u128, u128::MAX);
    }

    fn sample_indices_prop(max: u32, num_indices: usize) {
        let mut sponge = Tip5::randomly_seeded();
        let indices = sponge.sample_indices(max, num_indices);
        assert_eq!(num_indices, indices.len());
        assert!(indices.into_iter().all(|index| index < max));
    }

    #[test]
    fn sample_indices_test() {
        let cases = [
            (2, 0),
            (4, 1),
            (8, 9),
            (16, 10),
            (32, 11),
            (64, 19),
            (128, 20),
            (256, 21),
            (512, 65),
        ];

        for (upper_bound, num_indices) in cases {
            sample_indices_prop(upper_bound, num_indices);
        }
    }

    #[test]
    fn sample_scalars_test() {
        let amounts = [0, 1, 2, 3, 4];
        let mut sponge = Tip5::randomly_seeded();
        let mut product = XFieldElement::ONE;
        for amount in amounts {
            let scalars = sponge.sample_scalars(amount);
            assert_eq!(amount, scalars.len());
            product *= scalars
                .into_iter()
                .fold(XFieldElement::ONE, XFieldElement::mul);
        }
        assert_ne!(product, XFieldElement::ZERO); // false failure with prob ~2^{-192}
    }
}
