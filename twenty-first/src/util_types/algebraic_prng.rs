use itertools::Itertools;

use crate::shared_math::x_field_element::XFieldElement;

use super::algebraic_hasher::SpongeHasher;

trait AlgebraicPrng {
    type PrngState;
    fn sample_xfield(state: &mut Self::PrngState, num_elements: usize) -> Vec<XFieldElement>;
    fn sample_indices(state: &mut Self::PrngState, max: u32, num_elements: usize) -> Vec<u32>;
}

impl<H: SpongeHasher> AlgebraicPrng for H {
    type PrngState = H::SpongeState;

    fn sample_xfield(state: &mut Self::PrngState, num_elements: usize) -> Vec<XFieldElement> {
        let num_squeezes =
            (num_elements * 3 + <H as SpongeHasher>::RATE - 1) / <H as SpongeHasher>::RATE;
        (0..num_squeezes)
            .into_iter()
            .flat_map(|_| <H as SpongeHasher>::squeeze(state))
            .collect_vec()
            .chunks(3)
            .map(|ch| XFieldElement::new(ch.try_into().unwrap()))
            .collect_vec()
    }

    fn sample_indices(state: &mut Self::PrngState, max: u32, num_indices: usize) -> Vec<u32> {
        let num_squeezes =
            (num_indices + <H as SpongeHasher>::RATE - 1) / <H as SpongeHasher>::RATE;
        (0..num_squeezes)
            .into_iter()
            .flat_map(|_| <H as SpongeHasher>::squeeze(state))
            .map(|b| (b.value() % max as u64) as u32)
            .collect()
    }
}
