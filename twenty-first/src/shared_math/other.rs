use std::ops::BitAnd;
use std::ops::Sub;

use num_traits::One;
use num_traits::Zero;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::uniform::SampleUniform;

// Function for creating a bigint from an i128

const fn num_bits<T>() -> u64 {
    std::mem::size_of::<T>() as u64 * 8
}

pub fn log_2_floor(x: u128) -> u64 {
    assert!(x > 0);
    num_bits::<u128>() - x.leading_zeros() as u64 - 1
}

pub fn log_2_ceil(x: u128) -> u64 {
    if is_power_of_two(x) {
        log_2_floor(x)
    } else {
        log_2_floor(x) + 1
    }
}

/// Check if the number is a power of two: { 1,2,4 .. }
/// [Bit Twiddling Hacks]: <https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2>
pub fn is_power_of_two<T: Zero + One + Sub<Output = T> + BitAnd<Output = T> + Copy>(n: T) -> bool {
    !n.is_zero() && (n & (n - T::one())).is_zero()
}

/// Round up to the nearest power of 2
pub fn roundup_npo2(x: u64) -> u64 {
    1 << log_2_ceil(x as u128)
}

pub fn roundup_nearest_multiple(mut x: usize, multiple: usize) -> usize {
    let remainder = x % multiple;
    if remainder != 0 {
        x += multiple - remainder;
    }
    x
}

/// Generate `n` random elements using `rand::thread_rng()`.
///
/// This requires the trait instance `Standard: Distribution<T>`.
///
/// See trait instances for BFieldElement or XFieldElement for examples.
pub fn random_elements<T>(n: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    rand::thread_rng().sample_iter(Standard).take(n).collect()
}

pub fn random_elements_distinct<T>(n: usize) -> Vec<T>
where
    T: PartialEq,
    Standard: Distribution<T>,
{
    let mut sampler = rand::thread_rng().sample_iter(Standard);
    let mut distinct_elements = Vec::with_capacity(n);
    while distinct_elements.len() < n {
        let sample = sampler.next().expect("Random sampler ran out of elements");
        if !distinct_elements.contains(&sample) {
            distinct_elements.push(sample);
        }
    }
    distinct_elements
}

pub fn random_elements_range<T, R>(n: usize, range: R) -> Vec<T>
where
    T: SampleUniform,
    R: Into<Uniform<T>>,
    Standard: Distribution<T>,
{
    let mut rng = rand::thread_rng();
    range.into().sample_iter(&mut rng).take(n).collect()
}

pub fn random_elements_distinct_range<T, R>(n: usize, range: R) -> Vec<T>
where
    T: SampleUniform + PartialEq,
    R: Into<Uniform<T>>,
    Standard: Distribution<T>,
{
    let mut sampler = rand::thread_rng().sample_iter(range.into());
    let mut distinct_elements = Vec::with_capacity(n);
    while distinct_elements.len() < n {
        let sample = sampler.next().expect("Random sampler ran out of elements");
        if !distinct_elements.contains(&sample) {
            distinct_elements.push(sample);
        }
    }
    distinct_elements
}

pub fn random_elements_array<T, const N: usize>() -> [T; N]
where
    Standard: Distribution<T>,
{
    rand::thread_rng().sample::<[T; N], Standard>(Standard)
}

#[cfg(test)]
mod test_other {
    use super::*;

    #[test]
    fn log_2_ceil_test() {
        assert_eq!(4, log_2_floor(16));
        assert_eq!(1, log_2_floor(2));
        assert_eq!(0, log_2_floor(1));
        assert_eq!(40, log_2_floor(2u128.pow(40)));
        assert_eq!(40, log_2_floor(2u128.pow(40) + 1));
        assert_eq!(40, log_2_floor(2u128.pow(40) + 456456));
        assert_eq!(4, log_2_ceil(16));
        assert_eq!(5, log_2_ceil(17));
        assert_eq!(5, log_2_ceil(18));
        assert_eq!(5, log_2_ceil(19));
        assert_eq!(1, log_2_ceil(2));
        assert_eq!(0, log_2_ceil(1));
        assert_eq!(40, log_2_ceil(2u128.pow(40)));
        assert_eq!(41, log_2_ceil(2u128.pow(40) + 1));
        assert_eq!(41, log_2_ceil(2u128.pow(40) + 456456));
    }

    #[test]
    fn is_power_of_two_test() {
        let powers_of_two: Vec<u8> = vec![1, 2, 4, 8, 16, 32, 64, 128];
        for i in 0..u8::MAX {
            if powers_of_two.contains(&i) {
                assert!(is_power_of_two(i));
            } else {
                assert!(!is_power_of_two(i));
            }
        }
    }

    #[test]
    fn binary_tree_properties() {
        for height in 0..4 {
            let leaf_count = 2usize.pow(height);

            assert!(
                is_power_of_two(leaf_count),
                "The leaf count should be a power of two."
            );
        }
    }

    #[test]
    fn roundup_nearest_multiple_test() {
        let cases = [(0, 10, 0), (1, 10, 10), (10, 10, 10), (11, 10, 20)];
        for (x, multiple, expected) in cases {
            let actual = roundup_nearest_multiple(x, multiple);
            assert_eq!(expected, actual);
        }
    }
}
