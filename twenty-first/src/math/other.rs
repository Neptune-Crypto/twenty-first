use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;

/// Generate `n` random elements using [`rand::thread_rng()`].
///
/// For example implementations of the [`Distribution`] trait for [`Standard`], see
/// [`BFieldElement`][bfe] or [`XFieldElement`][xfe].
///
/// [bfe]: crate::prelude::BFieldElement
/// [xfe]: crate::prelude::XFieldElement
pub fn random_elements<T>(n: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    rand::thread_rng().sample_iter(Standard).take(n).collect()
}
