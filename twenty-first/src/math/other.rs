use rand::Rng;
use rand::distr::Distribution;
use rand::distr::StandardUniform;

/// Generate `n` random elements using [`rand::rng()`].
///
/// For example implementations of the [`Distribution`] trait for
/// [`StandardUniform`], see [`BFieldElement`][bfe] or [`XFieldElement`][xfe].
///
/// [bfe]: crate::prelude::BFieldElement
/// [xfe]: crate::prelude::XFieldElement
pub fn random_elements<T>(n: usize) -> Vec<T>
where
    StandardUniform: Distribution<T>,
{
    rand::rng().sample_iter(StandardUniform).take(n).collect()
}
