use std::ops::BitAnd;
use std::ops::Sub;

use num_traits::One;
use num_traits::Zero;
use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;

#[deprecated(since = "0.39.0", note = "use `.ilog2()` instead")]
pub fn log_2_floor(x: u128) -> u64 {
    x.ilog2().into()
}

pub fn log_2_ceil(x: u128) -> u64 {
    let log = x.ilog2().into();
    if x.is_power_of_two() {
        log
    } else {
        log + 1
    }
}

/// Check if the number is a power of two: { 1,2,4 .. }
/// [Bit Twiddling Hacks]: <https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2>
#[deprecated(since = "0.39.0", note = "use `.is_power_of_two()` instead")]
pub fn is_power_of_two<T: Zero + One + Sub<Output = T> + BitAnd<Output = T> + Copy>(n: T) -> bool {
    !n.is_zero() && (n & (n - T::one())).is_zero()
}

/// Round up to the nearest power of 2
#[deprecated(since = "0.39.0", note = "use `.next_power_of_two()` instead")]
pub fn roundup_npo2(x: u64) -> u64 {
    x.next_power_of_two()
}

#[deprecated(since = "0.39.0", note = "use `.next_multiple_of(…)` instead")]
pub fn roundup_nearest_multiple(x: usize, multiple: usize) -> usize {
    x.next_multiple_of(multiple)
}

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

#[cfg(test)]
mod test_other {
    use super::*;

    #[test]
    #[allow(deprecated)] // until removal of `log_2_floor`
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
    #[allow(deprecated)] // until removal of `is_power_of_two`
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
    #[allow(deprecated)] // until removal of `roundup_nearest_multiple`
    fn roundup_nearest_multiple_test() {
        let cases = [(0, 10, 0), (1, 10, 10), (10, 10, 10), (11, 10, 20)];
        for (x, multiple, expected) in cases {
            let actual = roundup_nearest_multiple(x, multiple);
            assert_eq!(expected, actual);
        }
    }
}
