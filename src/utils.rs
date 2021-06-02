use rand::RngCore;
use std::collections::HashSet;
use std::hash::Hash;

pub const FIRST_TEN_PRIMES: &[i128] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29];

pub fn has_unique_elements<T>(iter: T) -> bool
where
    T: IntoIterator,
    T::Item: Eq + Hash,
{
    let mut uniq = HashSet::new();
    iter.into_iter().all(move |x| uniq.insert(x))
}

pub fn generate_random_numbers(size: usize, modulus: i128) -> Vec<i128> {
    let mut prng = rand::thread_rng();
    let mut rand = vec![0u8; size * 32];
    prng.fill_bytes(rand.as_mut_slice());

    // This looks pretty inefficient
    // How is this done with a map instead?
    // TODO: Only generates values up to 256!
    let values: Vec<i128> = rand.iter().map(|&x| x as i128 % modulus).collect();
    values
}

#[cfg(test)]
mod test_utils {
    use super::*;

    #[test]
    fn has_unique_elements_test() {
        let v = vec![10, 20, 30, 10, 50];
        assert!(!has_unique_elements(v));

        let v = vec![10, 20, 30, 40, 50];
        assert!(has_unique_elements(v));
    }
}
