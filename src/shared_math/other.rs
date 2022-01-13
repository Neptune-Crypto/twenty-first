use num_bigint::BigInt;

// Function for creating a bigint from an i128
pub fn bigint(input: i128) -> BigInt {
    Into::<BigInt>::into(input)
}

const fn num_bits<T>() -> u64 {
    std::mem::size_of::<T>() as u64 * 8
}

pub fn log_2_floor(x: u64) -> u64 {
    assert!(x > 0);
    num_bits::<u64>() - x.leading_zeros() as u64 - 1
}

pub fn log_2_ceil(x: u64) -> u64 {
    // if x is a power of 2, return log_2_floor. Otherwise return log_2_floor + 1.
    if x & (x - 1) == 0 {
        log_2_floor(x)
    } else {
        log_2_floor(x) + 1
    }
}

// Round up to the nearest power of 2
pub fn roundup_npo2(x: u64) -> u64 {
    1 << log_2_ceil(x)
}

// pub fn lagrange_interpolation_2

#[cfg(test)]
mod test_other {
    use super::*;

    #[test]
    fn bigint_test() {
        assert_eq!(
            Into::<BigInt>::into(12345678901234567890i128),
            bigint(12345678901234567890i128)
        );
    }

    #[test]
    fn log_2_ceil_test() {
        assert_eq!(4, log_2_floor(16));
        assert_eq!(1, log_2_floor(2));
        assert_eq!(0, log_2_floor(1));
        assert_eq!(40, log_2_floor(2u64.pow(40)));
        assert_eq!(40, log_2_floor(2u64.pow(40) + 1));
        assert_eq!(40, log_2_floor(2u64.pow(40) + 456456));
        assert_eq!(4, log_2_ceil(16));
        assert_eq!(5, log_2_ceil(17));
        assert_eq!(5, log_2_ceil(18));
        assert_eq!(5, log_2_ceil(19));
        assert_eq!(1, log_2_ceil(2));
        assert_eq!(0, log_2_ceil(1));
        assert_eq!(40, log_2_ceil(2u64.pow(40)));
        assert_eq!(41, log_2_ceil(2u64.pow(40) + 1));
        assert_eq!(41, log_2_ceil(2u64.pow(40) + 456456));
    }
}
