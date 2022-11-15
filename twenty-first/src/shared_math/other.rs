use num_bigint::BigInt;
use num_traits::{One, Zero};
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Standard, Uniform};
use std::fmt::Display;
use std::ops::{BitAnd, Div, Rem, Shl, Sub};

// Function for creating a bigint from an i128
pub fn bigint(input: i128) -> BigInt {
    Into::<BigInt>::into(input)
}

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

/// Convert a number to an array showing which bits are set in its bit representation
pub fn bit_representation(x: u128) -> Vec<u8> {
    // The peak heights in an MMR can be read directly from the bit-decomposition
    // of the leaf count.
    if x == 0 {
        return vec![];
    }

    let bit_count: u64 = log_2_floor(x);
    let mut heights = vec![];
    for i in 0..=bit_count {
        if ((1 << i) & x) != 0 {
            heights.push(i as u8);
        }
    }

    // Reverse order of heights so we get this highest bit first.
    heights.reverse();

    heights
}

/// Check if the number is a power of two: { 1,2,4 .. }
/// [Bit Twiddling Hacks]: https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
pub fn is_power_of_two<T: Zero + One + Sub<Output = T> + BitAnd<Output = T> + Copy>(n: T) -> bool {
    !n.is_zero() && (n & (n - T::one())).is_zero()
}

pub fn powers_of_two_below<T>(max: T, bits: u32) -> impl Iterator<Item = T>
where
    T: One + PartialOrd + Shl<u32, Output = T> + Copy,
{
    (0..bits)
        .map(|i: u32| T::one() << i)
        .take_while(move |&x| x < max)
}

/// Round up to the nearest power of 2
pub fn roundup_npo2(x: u64) -> u64 {
    1 << log_2_ceil(x as u128)
}

#[inline]
pub fn mod_pow_raw(x: u128, exp: u64, quotient: u128) -> u128 {
    // Special case for handling 0^0 = 1
    if exp == 0 {
        return 1;
    }

    let mut acc = 1;

    let bit_length = count_bits(exp);
    for i in 0..bit_length {
        acc = (acc * acc) % quotient;
        if exp & (1 << (bit_length - 1 - i)) != 0 {
            acc = (acc * x) % quotient;
        }
    }

    acc
}

/// Simultaneously perform division and remainder.
///
/// While there is apparently no built-in Rust function for this,
/// the optimizer will still compile this to a single instruction
/// on x86.
pub fn div_rem<T: Div<Output = T> + Rem<Output = T> + Copy>(x: T, y: T) -> (T, T) {
    let quot = x / y;
    let rem = x % y;
    (quot, rem)
}

// TODO: Abstract for multiple unsigned output types.
pub fn primes_lt(bound: u64) -> Vec<u64> {
    let mut primes: Vec<bool> = (0..bound + 1).map(|num| num == 2 || num & 1 != 0).collect();

    let mut num = 3u64;
    while num * num <= bound {
        let mut j = num * num;
        while j <= bound {
            primes[j as usize] = false;
            j += num;
        }
        num += 2;
    }

    primes
        .into_iter()
        .enumerate()
        .skip(2)
        .filter_map(|(i, p)| if p { Some(i as u64) } else { None })
        .collect::<Vec<u64>>()
}

#[inline]
/// Extended Euclid's Algorithm.
pub fn xgcd<
    T: Zero + One + Rem<Output = T> + Div<Output = T> + Sub<Output = T> + Clone + Display,
>(
    mut x: T,
    mut y: T,
) -> (T, T, T) {
    let (mut a_factor, mut a1, mut b_factor, mut b1) = (T::one(), T::zero(), T::zero(), T::one());

    while !y.is_zero() {
        let (quotient, remainder) = (x.clone() / y.clone(), x.clone() % y.clone());
        let (c, d) = (
            a_factor - quotient.clone() * a1.clone(),
            b_factor.clone() - quotient * b1.clone(),
        );

        x = y;
        y = remainder;
        a_factor = a1;
        a1 = c;
        b_factor = b1;
        b1 = d;
    }

    // x is the gcd
    (x, a_factor, b_factor)
}

/// Matrix transpose
///
/// ```py
/// [a b c]
/// [d e f]
/// ```
///
/// returns
///
/// ```py
/// [a d]
/// [b e]
/// [c f]
/// ```
///
/// Assumes that input is regular.
pub fn transpose<P: Copy>(codewords: &[Vec<P>]) -> Vec<Vec<P>> {
    (0..codewords[0].len())
        .map(|col_idx| codewords.iter().map(|row| row[col_idx]).collect())
        .collect()
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

/// Compute the number of nodes (used as capacity) in a complete binary tree
/// from the number of leaves.
pub fn get_node_count_in_complete_binary_tree(leaf_count: usize) -> usize {
    assert!(is_power_of_two(leaf_count));
    2 * leaf_count - 1
}

pub fn get_height_of_complete_binary_tree(leaf_count: usize) -> usize {
    assert!(is_power_of_two(leaf_count));
    // nightly: leaves_count.log2() + 1
    log_2_floor(leaf_count as u128) as usize
}

/// Count bits required to represent an unsigned number. Inspired by and similar to
/// BigInteger's `bits()` method.
#[inline]
pub fn count_bits(input: u64) -> u32 {
    64 - input.leading_zeros()
}

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
    fn powers_of_two_below_test() {
        for i in powers_of_two_below::<u32>(u32::MAX, u32::BITS) {
            assert!(is_power_of_two(i));
        }

        // TODO: Test that it can be empty.
        // TODO: Test that no powers of below max are missing.
    }

    #[test]
    fn transpose_test() {
        let input = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let expected = vec![vec![1, 4], vec![2, 5], vec![3, 6]];
        let actual = transpose(&input);
        assert_eq!(expected, actual);

        let mut rng = rand::thread_rng();
        let n = rng.gen_range(1..10);
        let m = rng.gen_range(1..10);
        let random_matrix: Vec<Vec<u64>> = (0..n)
            .map(|_| (0..m).map(|_| rng.gen()).collect())
            .collect();

        assert_eq!(n, random_matrix.len());
        assert_eq!(m, random_matrix[0].len());

        let transposed = transpose(&random_matrix);
        assert_eq!(m, transposed.len());
        assert_eq!(n, transposed[0].len());

        let transposed_transposed = transpose(&transposed);
        assert_eq!(random_matrix, transposed_transposed);
    }

    #[test]
    fn bit_representation_test() {
        assert_eq!(Vec::<u8>::new(), bit_representation(0));
        assert_eq!(vec![0], bit_representation(1));
        assert_eq!(vec![1], bit_representation(2));
        assert_eq!(vec![1, 0], bit_representation(3));

        // test example made on calculator
        assert_eq!(vec![23, 7, 5, 4, 3, 2, 1, 0], bit_representation(8388799));
    }

    #[test]
    fn binary_tree_properties() {
        for height in 0..4 {
            let leaf_count = 2usize.pow(height);

            assert!(
                is_power_of_two(leaf_count),
                "The leaf count should be a power of two."
            );

            assert_eq!(
                leaf_count,
                2usize.pow(get_height_of_complete_binary_tree(leaf_count) as u32),
                "The leaf counts should agree."
            )
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed: is_power_of_two(leaf_count)")]
    fn small_tree0_height() {
        let leaves0: [i32; 0] = [];
        let leaf_count0 = leaves0.len();
        let height0 = get_height_of_complete_binary_tree(leaf_count0);
        assert_eq!(height0, 42, "Execution should not reach here.");
    }

    #[test]
    #[should_panic(expected = "assertion failed: is_power_of_two(leaf_count)")]
    fn small_tree0_node_count() {
        let leaves0: [i32; 0] = [];
        let leaf_count0 = leaves0.len();
        let node_count0 = get_node_count_in_complete_binary_tree(leaf_count0);
        assert_eq!(node_count0, 42, "Execution should not reach here.");
    }

    #[test]
    fn small_tree1() {
        let leaves1 = [1];
        let leaf_count1 = leaves1.len();
        let height1 = get_height_of_complete_binary_tree(leaf_count1);
        assert_eq!(height1, 0);
        let node_count1 = get_node_count_in_complete_binary_tree(leaf_count1);
        assert_eq!(node_count1, 1, "The root is also the only node.");
    }

    #[test]
    fn small_tree2() {
        let leaves2 = [1, 2];
        let leaf_count2 = leaves2.len();
        let height2 = get_height_of_complete_binary_tree(leaf_count2);
        assert_eq!(height2, 1);
        let node_count2 = get_node_count_in_complete_binary_tree(leaf_count2);
        assert_eq!(node_count2, 3, "The root and the two leaves.");
    }

    #[test]
    fn small_tree4() {
        let leaves4 = [1, 2, 3, 4];
        let leaf_count4 = leaves4.len();
        let height4 = get_height_of_complete_binary_tree(leaf_count4);
        assert_eq!(height4, 2);
        let node_count4 = get_node_count_in_complete_binary_tree(leaf_count4);
        assert_eq!(
            node_count4, 7,
            "The root, its two children and the four leaves."
        );
    }

    #[test]
    fn count_bits_test() {
        assert_eq!(0, count_bits(0));
        assert_eq!(1, count_bits(1));
        assert_eq!(2, count_bits(2));
        assert_eq!(2, count_bits(3));
        assert_eq!(3, count_bits(4));
        assert_eq!(3, count_bits(5));
        assert_eq!(3, count_bits(6));
        assert_eq!(3, count_bits(7));
        assert_eq!(4, count_bits(8));
        assert_eq!(4, count_bits(9));
        assert_eq!(4, count_bits(10));
        assert_eq!(10, count_bits(1023));
        assert_eq!(11, count_bits(1024));
    }
}
