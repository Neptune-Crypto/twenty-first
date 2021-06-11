const fn num_bits<T>() -> u64 {
    std::mem::size_of::<T>() as u64 * 8
}

pub fn log_2(x: u64) -> u64 {
    assert!(x > 0);
    num_bits::<u64>() - x.leading_zeros() as u64 - 1
}
