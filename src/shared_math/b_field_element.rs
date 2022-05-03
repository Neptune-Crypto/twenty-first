use super::mpolynomial::MPolynomial;
use super::other;
use super::traits::{FromVecu8, GetPrimitiveRootOfUnity, Inverse};
use super::x_field_element::XFieldElement;
use crate::shared_math::traits::GetRandomElements;
use crate::shared_math::traits::{
    CyclicGroupGenerator, IdentityValues, ModPowU32, ModPowU64, New, PrimeField,
};
use crate::utils::FIRST_THOUSAND_PRIMES;
use num_traits::{One, Zero};
use std::hash::{Hash, Hasher};

use phf::phf_map;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::num::TryFromIntError;
use std::ops::{AddAssign, MulAssign, SubAssign};
use std::{
    fmt::{self},
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

static PRIMITIVE_ROOTS: phf::Map<u64, u64> = phf_map! {
    2u64 => 18446744069414584320,
    4u64 => 281474976710656,
    8u64 => 18446744069397807105,
    16u64 => 17293822564807737345,
    32u64 => 70368744161280,
    64u64 => 549755813888,
    128u64 => 17870292113338400769,
    256u64 => 13797081185216407910,
    512u64 => 1803076106186727246,
    1024u64 => 11353340290879379826,
    2048u64 => 455906449640507599,
    4096u64 => 17492915097719143606,
    8192u64 => 1532612707718625687,
    16384u64 => 16207902636198568418,
    32768u64 => 17776499369601055404,
    65536u64 => 6115771955107415310,
    131072u64 => 12380578893860276750,
    262144u64 => 9306717745644682924,
    524288u64 => 18146160046829613826,
    1048576u64 => 3511170319078647661,
    2097152u64 => 17654865857378133588,
    4194304u64 => 5416168637041100469,
    8388608u64 => 16905767614792059275,
    16777216u64 => 9713644485405565297,
    33554432u64 => 5456943929260765144,
    67108864u64 => 17096174751763063430,
    134217728u64 => 1213594585890690845,
    268435456u64 => 6414415596519834757,
    536870912u64 => 16116352524544190054,
    1073741824u64 => 9123114210336311365,
    2147483648u64 => 4614640910117430873,
    4294967296u64 => 1753635133440165772,
};

// BFieldElement ∈ ℤ_{2^64 - 2^32 + 1}
#[derive(Debug, Copy, Clone, Serialize, Deserialize, Default)]
pub struct BFieldElement(u64);

impl PartialEq for BFieldElement {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
            || Self::canonical_representation(self) == Self::canonical_representation(other)
    }
}

impl Eq for BFieldElement {}

impl Hash for BFieldElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical_representation().hash(state);
    }
}

impl BFieldElement {
    // 2^64 - 2^32 + 1
    pub const QUOTIENT: u64 = 0xffff_ffff_0000_0001u64;
    pub const MAX: u64 = Self::QUOTIENT - 1;
    const LOWER_MASK: u64 = 0xFFFFFFFF;

    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value % Self::QUOTIENT)
    }

    #[inline]
    pub fn value(&self) -> u64 {
        self.canonical_representation()
    }

    #[inline]
    /// Square the base M times and multiply the result by the tail value
    pub fn power_accumulator<const N: usize, const M: usize>(
        base: [Self; N],
        tail: [Self; N],
    ) -> [Self; N] {
        let mut result = base;
        for _ in 0..M {
            result.iter_mut().for_each(|r| *r *= *r);
        }
        result.iter_mut().zip(tail).for_each(|(r, t)| *r *= t);
        result
    }

    /// Get a generator for the entire field
    pub fn generator() -> Self {
        BFieldElement::new(7)
    }

    #[inline]
    pub fn lift(&self) -> XFieldElement {
        XFieldElement::new_const(*self)
    }

    // You should probably only use `next` and `previous` for testing purposes
    pub fn next(&mut self) {
        self.0 = Self::canonical_representation(&(*self + Self::ring_one()));
    }

    // You should probably only use `next` and `previous` for testing purposes
    pub fn previous(&mut self) {
        self.0 = Self::canonical_representation(&(*self - Self::ring_one()));
    }

    // TODO: Currently, IdentityValues has &self as part of its signature, so we hotfix
    // being able to refer to a zero/one element without having an element at hand. This
    // will go away when moving to Zero/One traits.
    #[inline]
    pub const fn ring_zero() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn ring_one() -> Self {
        Self(1)
    }

    #[inline]
    fn canonical_representation(&self) -> u64 {
        if self.0 > Self::MAX {
            self.0 - Self::QUOTIENT
        } else {
            self.0
        }
    }

    #[must_use]
    #[inline]
    pub fn mod_pow(&self, exp: u64) -> Self {
        // Special case for handling 0^0 = 1
        if exp == 0 {
            return BFieldElement::ring_one();
        }

        let mut acc = BFieldElement::ring_one();
        let bit_length = other::count_bits(exp);
        for i in 0..bit_length {
            acc = acc * acc;
            if exp & (1 << (bit_length - 1 - i)) != 0 {
                acc *= *self;
            }
        }

        acc
    }

    pub fn legendre_symbol(&self) -> i8 {
        let elem = self.mod_pow((Self::QUOTIENT - 1) as u64 / 2).0;

        // Ugly hack to force a result in {-1,0,1}
        if elem == Self::QUOTIENT - 1 {
            -1
        } else if elem == 0 {
            0
        } else {
            1
        }
    }

    #[inline(always)]
    fn mod_reduce(x: u128) -> u64 {
        // Copied from (MIT licensed):
        // https://github.com/anonauthorsub/asiaccs_2021_440/blob/5141f349e5915ab750208c0ab1b5f7be95adaeac/math/src/field/f64/mod.rs#L551
        // assume x consists of four 32-bit values: a, b, c, d such that a contains 32 least
        // significant bits and d contains 32 most significant bits. we break x into corresponding
        // values as shown below
        let ab = x as u64;
        let cd = (x >> 64) as u64;
        let c = (cd as u32) as u64;
        let d = cd >> 32;

        // compute ab - d; because d may be greater than ab we need to handle potential underflow
        let (tmp0, under) = ab.overflowing_sub(d);
        let tmp0 = tmp0.wrapping_sub(Self::LOWER_MASK * (under as u64));

        // compute c * 2^32 - c; this is guaranteed not to underflow
        let tmp1 = (c << 32) - c;

        // add temp values and return the result; because each of the temp may be up to 64 bits,
        // we need to handle potential overflow
        let (result, over) = tmp0.overflowing_add(tmp1);
        result.wrapping_add(Self::LOWER_MASK * (over as u64))
    }
}

impl fmt::Display for BFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for BFieldElement {
    fn from(value: u32) -> Self {
        BFieldElement::new(value.into())
    }
}

impl TryFrom<BFieldElement> for u32 {
    type Error = TryFromIntError;

    fn try_from(value: BFieldElement) -> Result<Self, Self::Error> {
        u32::try_from(value.0)
    }
}

impl Inverse for BFieldElement {
    #[must_use]
    #[inline]
    fn inverse(&self) -> Self {
        let (_, _, a) = other::xgcd(Self::QUOTIENT as i128, self.0 as i128);

        Self(
            ((a % Self::QUOTIENT as i128 + Self::QUOTIENT as i128) % Self::QUOTIENT as i128) as u64,
        )
    }
}

impl ModPowU32 for BFieldElement {
    #[inline]
    fn mod_pow_u32(&self, exp: u32) -> Self {
        self.mod_pow(exp as u64)
    }
}

impl CyclicGroupGenerator for BFieldElement {
    fn get_cyclic_group_elements(&self, max: Option<usize>) -> Vec<Self> {
        let mut val = *self;
        let mut ret: Vec<Self> = vec![self.ring_one()];

        loop {
            ret.push(val);
            val *= *self;
            if val.is_one() || max.is_some() && ret.len() >= max.unwrap() {
                break;
            }
        }
        ret
    }
}

impl GetRandomElements for BFieldElement {
    fn random_elements<R: Rng>(length: usize, prng: &mut R) -> Vec<Self> {
        let mut values: Vec<BFieldElement> = Vec::with_capacity(length);
        let max = BFieldElement::MAX as u64;

        while values.len() < length {
            let n = prng.next_u64();

            if n > max {
                continue;
            }

            values.push(BFieldElement::new(n));
        }

        values
    }
}

impl New for BFieldElement {
    fn new_from_usize(&self, value: usize) -> Self {
        Self::new(value as u64)
    }
}

// This is used for: Convert a hash value to a BFieldElement. Consider making From<Blake3Hash> trait
impl FromVecu8 for BFieldElement {
    fn from_vecu8(&self, bytes: Vec<u8>) -> Self {
        // TODO: Right now we only accept if 'bytes' has 8 bytes; while that is true in
        // the single call site this is used, it also seems unnecessarily fragile (when we
        // change from BLAKE3 to Rescue-Prime, the hash length will change and this will be
        // be wrong). We should make this a From<Blake3Hash> to ensure that it has the right
        // length.
        let (eight_bytes, _rest) = bytes.as_slice().split_at(std::mem::size_of::<u64>());
        let coerced: [u8; 8] = eight_bytes.try_into().unwrap();
        let n: u64 = u64::from_ne_bytes(coerced);
        BFieldElement::new(n)
    }
}

impl PrimeField for BFieldElement {}

// TODO: We implement IdentityValues so this module works in existing code, but we want to replace IdentityValues with Zero and One eventually.
// impl Zero for BFieldElement {
//     fn zero() -> Self {
//         BFieldElement(0)
//     }

//     fn is_zero(&self) -> bool {
//         self.0 == 0
//     }
// }

// impl One for BFieldElement {
//     fn one() -> Self {
//         BFieldElement(1)
//     }

//     fn is_one(&self) -> bool {
//         self.0 == 1
//     }
// }

impl IdentityValues for BFieldElement {
    #[inline]
    fn is_zero(&self) -> bool {
        self.canonical_representation() == 0
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.canonical_representation() == 1
    }

    #[inline]
    fn ring_zero(&self) -> Self {
        BFieldElement(0)
    }

    #[inline]
    fn ring_one(&self) -> Self {
        BFieldElement(1)
    }
}

impl Add for BFieldElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let (result, overflow) = self.0.overflowing_add(other.0);
        let mut val = result.wrapping_sub(Self::QUOTIENT * (overflow as u64));

        // For some reason, this `if` codeblock improves NTT runtime by ~10 % and
        // Rescue prime calculations with up to 45 % for `hash_pair`. I think it has
        // something to do with a compiler optimization but I actually don't
        // understand why this speedup occurs.
        if val > Self::MAX {
            val -= Self::QUOTIENT;
        }

        Self(val)
    }
}

impl AddAssign for BFieldElement {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl SubAssign for BFieldElement {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl MulAssign for BFieldElement {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Mul for BFieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        let val: u64 = Self::mod_reduce(self.0 as u128 * other.0 as u128);
        Self(val)
    }
}

impl Neg for BFieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(Self::QUOTIENT - self.canonical_representation())
    }
}

impl Rem for BFieldElement {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Self(self.0 % other.0)
    }
}

impl Sub for BFieldElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn sub(self, other: Self) -> Self {
        let (result, overflow) = self.0.overflowing_sub(other.canonical_representation());
        Self(result.wrapping_add(Self::QUOTIENT * (overflow as u64)))
    }
}

impl Div for BFieldElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self {
        other.inverse() * self
    }
}

// TODO: We probably wanna make use of Rust's Pow, but for now we copy from ...big:
impl ModPowU64 for BFieldElement {
    #[inline]
    fn mod_pow_u64(&self, pow: u64) -> Self {
        self.mod_pow(pow)
    }
}

impl GetPrimitiveRootOfUnity for BFieldElement {
    fn get_primitive_root_of_unity(&self, n: u64) -> (Option<BFieldElement>, Vec<u64>) {
        // Check if n is one of the values for which we have pre-calculated roots
        if PRIMITIVE_ROOTS.contains_key(&(n as u64)) {
            return (
                Some(BFieldElement::new(PRIMITIVE_ROOTS[&(n as u64)])),
                vec![2],
            );
        }

        let mut primes: Vec<u64> = vec![];

        if n <= 1 {
            return (Some(BFieldElement::ring_one()), primes);
        }

        // Calculate prime factorization of n
        // Check if n = 2^k
        if other::is_power_of_two(n) {
            primes = vec![2];
        } else {
            let mut m = n;
            for prime in FIRST_THOUSAND_PRIMES.iter().map(|&p| p as u64) {
                if m == 1 {
                    break;
                }
                if m % prime == 0 {
                    primes.push(prime);
                    while m % prime == 0 {
                        m /= prime;
                    }
                }
            }
            // This might be prohibitively expensive
            if m > 1 {
                let mut other_primes = other::primes_lt(m)
                    .into_iter()
                    .filter(|&x| n % x == 0)
                    .collect();
                primes.append(&mut other_primes);
            }
        };

        // N must divide the field prime minus one for a primitive nth root of unity to exist
        if !((Self::QUOTIENT - 1) % n as u64).is_zero() {
            return (None, primes);
        }

        let mut primitive_root: Option<BFieldElement> = None;
        let mut candidate: BFieldElement = BFieldElement::ring_one();

        #[allow(clippy::suspicious_operation_groupings)]
        while primitive_root == None && candidate.0 < Self::QUOTIENT {
            if (-candidate.legendre_symbol()).is_one()
                && primes.iter().filter(|&x| n % x == 0).all(|x| {
                    !other::mod_pow_raw(
                        candidate.0 as u128,
                        ((Self::QUOTIENT - 1) / *x as u64) as u64,
                        Self::QUOTIENT as u128,
                    )
                    .is_one()
                })
            {
                primitive_root = Some(candidate.mod_pow(((Self::QUOTIENT - 1) / n as u64) as u64));
            }

            candidate.0 += 1;
        }

        (primitive_root, primes)
    }
}

// TODO: Remove this function when `transition_constraints_afo_named_variables` can
// return a <PF: PrimeField> version of itself.
pub fn lift_coefficients_to_xfield(
    mpolynomial: &MPolynomial<BFieldElement>,
) -> MPolynomial<XFieldElement> {
    let mut new_coefficients: HashMap<Vec<u64>, XFieldElement> = HashMap::new();
    mpolynomial.coefficients.iter().for_each(|(key, value)| {
        new_coefficients.insert(key.to_owned(), value.lift());
    });

    MPolynomial {
        variable_count: mpolynomial.variable_count,
        coefficients: new_coefficients,
    }
}

#[cfg(test)]
mod b_prime_field_element_test {
    use std::collections::hash_map::DefaultHasher;

    use crate::utils::generate_random_numbers_u128;
    use crate::{
        shared_math::{b_field_element::*, polynomial::Polynomial},
        utils::generate_random_numbers,
    };
    use itertools::izip;
    use proptest::prelude::*;

    // TODO: Move this into separate file.
    macro_rules! bfield_elem {
        ($value:expr) => {{
            BFieldElement::new($value)
        }};
    }

    #[test]
    fn test_zero_one() {
        let zero = bfield_elem!(0);
        let one = bfield_elem!(1);

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(!one.is_zero());
        assert!(one.is_one());
        assert!(bfield_elem!(BFieldElement::MAX + 1).is_zero());
        assert!(bfield_elem!(BFieldElement::MAX + 2).is_one());
    }

    #[test]
    fn simple_value_test() {
        let zero: BFieldElement = bfield_elem!(0);
        assert_eq!(0, zero.value());
        let one: BFieldElement = BFieldElement::ring_one();
        assert_eq!(1, one.value());

        let neinneinnein = bfield_elem!(999);
        assert_eq!(999, neinneinnein.value());
    }

    #[test]
    fn simple_generator_test() {
        assert_eq!(bfield_elem!(7), BFieldElement::generator());
        assert!(bfield_elem!(7)
            .mod_pow(((1u128 << 64) - (1u128 << 32)) as u64)
            .is_one());
        assert!(!bfield_elem!(7)
            .mod_pow(((1u128 << 64) - (1u128 << 32)) as u64 / 2)
            .is_one());
    }

    #[test]
    fn simple_lift_test() {
        let zero: BFieldElement = bfield_elem!(0);
        assert!(zero.lift().is_zero());
        assert!(!zero.lift().is_one());

        let one: BFieldElement = bfield_elem!(1);
        assert!(!one.lift().is_zero());
        assert!(one.lift().is_one());

        let five: BFieldElement = bfield_elem!(5);
        let five_lifted: XFieldElement = five.lift();
        assert_eq!(Some(five), five_lifted.unlift());
    }

    #[test]
    fn lift_property_test() {
        let mut rng = rand::thread_rng();
        let elements: Vec<BFieldElement> = BFieldElement::random_elements(100, &mut rng);
        for element in elements {
            assert_eq!(Some(element), element.lift().unlift());
        }
    }

    #[test]
    fn next_and_previous_test() {
        let mut val_a = bfield_elem!(0);
        let mut val_b = bfield_elem!(1);
        let mut val_c = bfield_elem!(BFieldElement::MAX - 1);
        let max = BFieldElement::new(BFieldElement::MAX);
        val_a.next();
        assert!(val_a.is_one());
        val_b.next();
        assert!(!val_b.is_one());
        assert_eq!(BFieldElement::new(2), val_b);
        val_b.next();
        assert_eq!(BFieldElement::new(3), val_b);
        assert_ne!(max, val_c);
        val_c.next();
        assert_eq!(max, val_c);
        val_c.next();
        assert!(val_c.is_zero());
        val_c.next();
        assert!(val_c.is_one());
        val_c.previous();
        assert!(val_c.is_zero());
        val_c.previous();
        assert_eq!(max, val_c);
        val_c.previous();
        assert_eq!(bfield_elem!(BFieldElement::MAX - 1), val_c);
    }

    proptest! {
        #[test]
        fn identity_tests(n in 0u64..BFieldElement::MAX) {
            let zero = bfield_elem!(0);
            let one = bfield_elem!(1);
            let other = bfield_elem!(n);

            prop_assert_eq!(other, zero + other, "left zero identity");
            prop_assert_eq!(other, other + zero, "right zero identity");
            prop_assert_eq!(other, one * other, "left one identity");
            prop_assert_eq!(other, other * one, "right one identity");
        }
    }

    #[test]
    fn inversion_test() {
        let one_inv = bfield_elem!(1);
        let two_inv = bfield_elem!(9223372034707292161);
        let three_inv = bfield_elem!(12297829379609722881);
        let four_inv = bfield_elem!(13835058052060938241);
        let five_inv = bfield_elem!(14757395255531667457);
        let six_inv = bfield_elem!(15372286724512153601);
        let seven_inv = bfield_elem!(2635249152773512046);
        let eight_inv = bfield_elem!(16140901060737761281);
        let nine_inv = bfield_elem!(4099276459869907627);
        let ten_inv = bfield_elem!(16602069662473125889);
        let eightfive_million_sixhundred_and_seventyone_onehundred_and_six_inv =
            bfield_elem!(13115294102219178839);
        assert_eq!(two_inv, bfield_elem!(2).inverse());
        assert_eq!(three_inv, bfield_elem!(3).inverse());
        assert_eq!(four_inv, bfield_elem!(4).inverse());
        assert_eq!(five_inv, bfield_elem!(5).inverse());
        assert_eq!(six_inv, bfield_elem!(6).inverse());
        assert_eq!(seven_inv, bfield_elem!(7).inverse());
        assert_eq!(eight_inv, bfield_elem!(8).inverse());
        assert_eq!(nine_inv, bfield_elem!(9).inverse());
        assert_eq!(ten_inv, bfield_elem!(10).inverse());
        assert_eq!(
            eightfive_million_sixhundred_and_seventyone_onehundred_and_six_inv,
            bfield_elem!(85671106).inverse()
        );

        let inverses = [
            one_inv,
            two_inv,
            three_inv,
            four_inv,
            five_inv,
            six_inv,
            seven_inv,
            eight_inv,
            nine_inv,
            ten_inv,
            eightfive_million_sixhundred_and_seventyone_onehundred_and_six_inv,
        ];
        let values = [
            bfield_elem!(1),
            bfield_elem!(2),
            bfield_elem!(3),
            bfield_elem!(4),
            bfield_elem!(5),
            bfield_elem!(6),
            bfield_elem!(7),
            bfield_elem!(8),
            bfield_elem!(9),
            bfield_elem!(10),
            bfield_elem!(85671106),
        ];
        let calculated_inverses = BFieldElement::batch_inversion(values.to_vec());
        assert_eq!(values.len(), calculated_inverses.len());
        let calculated_inverse_inverses =
            BFieldElement::batch_inversion(calculated_inverses.to_vec());
        for i in 0..calculated_inverses.len() {
            assert_eq!(inverses[i], calculated_inverses[i]);
            assert_eq!(calculated_inverse_inverses[i], values[i]);
        }

        let empty_inversion = BFieldElement::batch_inversion(vec![]);
        assert!(empty_inversion.is_empty());

        let singleton_inversion = BFieldElement::batch_inversion(vec![bfield_elem!(2)]);
        assert_eq!(1, singleton_inversion.len());
        assert_eq!(two_inv, singleton_inversion[0]);

        let duplet_inversion =
            BFieldElement::batch_inversion(vec![bfield_elem!(2), bfield_elem!(1)]);
        assert_eq!(2, duplet_inversion.len());
        assert_eq!(two_inv, duplet_inversion[0]);
        assert_eq!(one_inv, duplet_inversion[1]);
    }

    #[test]
    fn inversion_property_based_test() {
        let rands: Vec<i128> = generate_random_numbers(30, BFieldElement::MAX as i128);
        for rand in rands {
            assert!((bfield_elem!(rand as u64).inverse() * bfield_elem!(rand as u64)).is_one());
        }
    }

    #[test]
    fn batch_inversion_pbt() {
        let test_iterations = 100;
        let mut rng = rand::thread_rng();
        for i in 0..test_iterations {
            let rands: Vec<BFieldElement> = BFieldElement::random_elements(i, &mut rng);
            let rands_inv: Vec<BFieldElement> = BFieldElement::batch_inversion(rands.clone());
            assert_eq!(i as usize, rands_inv.len());
            for (mut rand, rand_inv) in izip!(rands, rands_inv) {
                assert!((rand * rand_inv).is_one());
                assert!((rand_inv * rand).is_one());
                assert_eq!(rand.inverse(), rand_inv);
                rand.next();
                assert!(!(rand * rand_inv).is_one());
            }
        }
    }

    #[test]
    fn power_accumulator_simple_test() {
        let input_a = [
            BFieldElement::new(10),
            BFieldElement::new(100),
            BFieldElement::new(1000),
            BFieldElement::new(1),
        ];
        let input_b = [
            BFieldElement::new(5),
            BFieldElement::new(6),
            BFieldElement::new(7),
            BFieldElement::new(8),
        ];
        let powers: [BFieldElement; 4] = BFieldElement::power_accumulator::<4, 2>(input_a, input_b);
        assert_eq!(BFieldElement::new(50000), powers[0]);
        assert_eq!(BFieldElement::new(600000000), powers[1]);
        assert_eq!(BFieldElement::new(7000000000000), powers[2]);
        assert_eq!(BFieldElement::new(8), powers[3]);
    }

    #[test]
    fn mul_div_plus_minus_neg_property_based_test() {
        let rands: Vec<i128> = generate_random_numbers(300, BFieldElement::QUOTIENT as i128);
        let power_inputs_b: Vec<i128> = generate_random_numbers(6, BFieldElement::QUOTIENT as i128);
        for i in 1..rands.len() {
            let a = bfield_elem!(rands[i - 1] as u64);
            let b = bfield_elem!(rands[i] as u64);
            let ab = a * b;
            let a_o_b = a / b;
            let b_o_a = b / a;
            assert_eq!(a, ab / b);
            assert_eq!(b, ab / a);
            assert_eq!(a, a_o_b * b);
            assert_eq!(b, b_o_a * a);
            assert!((a_o_b * b_o_a).is_one());
            assert_eq!(a * a, a.square());

            assert_eq!(a - b + b, a);
            assert_eq!(b - a + a, b);
            assert!((a - a).is_zero());
            assert!((b - b).is_zero());

            // Test the add/sub/mul assign operators
            let mut a_minus_b = a;
            a_minus_b -= b;
            assert_eq!(a - b, a_minus_b);

            let mut a_plus_b = a;
            a_plus_b += b;
            assert_eq!(a + b, a_plus_b);

            let mut a_mul_b = a;
            a_mul_b *= b;
            assert_eq!(a * b, a_mul_b);
            assert_eq!(b * a, a_mul_b);

            // Test negation
            assert!((-a + a).is_zero());
            assert!((-b + b).is_zero());
            assert!((-ab + ab).is_zero());
            assert!((-a_o_b + a_o_b).is_zero());
            assert!((-b_o_a + b_o_a).is_zero());
            assert!((-a_minus_b + a_minus_b).is_zero());
            assert!((-a_plus_b + a_plus_b).is_zero());
            assert!((-a_mul_b + a_mul_b).is_zero());

            // Test power_accumulator
            let power_input_a = [a, b, ab, a_o_b, b_o_a, a_minus_b];
            let power_input_b = [
                bfield_elem!(power_inputs_b[0] as u64),
                bfield_elem!(power_inputs_b[1] as u64),
                bfield_elem!(power_inputs_b[2] as u64),
                bfield_elem!(power_inputs_b[3] as u64),
                bfield_elem!(power_inputs_b[4] as u64),
                bfield_elem!(power_inputs_b[5] as u64),
            ];
            let powers = BFieldElement::power_accumulator::<6, 4>(power_input_a, power_input_b);
            for ((result_element, input_a), input_b) in powers
                .iter()
                .zip(power_input_a.iter())
                .zip(power_input_b.iter())
            {
                assert_eq!(input_a.mod_pow(16) * *input_b, *result_element);
            }
        }
    }

    #[test]
    fn neg_test() {
        assert_eq!(-BFieldElement::ring_zero(), BFieldElement::ring_zero());
        assert_eq!((-BFieldElement::ring_one()).0, BFieldElement::MAX);
        let max = BFieldElement::new(BFieldElement::MAX);
        let max_plus_one = max + BFieldElement::ring_one();
        let max_plus_two = max_plus_one + BFieldElement::ring_one();
        assert_eq!(BFieldElement::ring_zero(), -max_plus_one);
        assert_eq!(max, -max_plus_two);
    }

    #[test]
    fn equality_and_hash_test() {
        assert_eq!(BFieldElement::ring_zero(), BFieldElement::ring_zero());
        assert_eq!(BFieldElement::ring_one(), BFieldElement::ring_one());
        assert_ne!(BFieldElement::ring_one(), BFieldElement::ring_zero());
        assert_eq!(BFieldElement::new(42), BFieldElement::new(42));
        assert_ne!(BFieldElement::new(42), BFieldElement::new(43));

        assert_eq!(
            BFieldElement::new(102),
            BFieldElement::new(BFieldElement::MAX) + BFieldElement::new(103)
        );
        assert_ne!(
            BFieldElement::new(103),
            BFieldElement::new(BFieldElement::MAX) + BFieldElement::new(103)
        );

        // Verify that hashing works for canonical representations
        let mut hasher_a = DefaultHasher::new();
        let mut hasher_b = DefaultHasher::new();

        BFieldElement::new(42).hash(&mut hasher_a);
        BFieldElement::new(42).hash(&mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());

        // Verify that hashing works for non-canonical representations
        hasher_a = DefaultHasher::new();
        hasher_b = DefaultHasher::new();
        (BFieldElement::new(BFieldElement::MAX) + BFieldElement::new(103)).hash(&mut hasher_a);
        BFieldElement::new(102).hash(&mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }

    #[test]
    fn create_polynomial_test() {
        let a: Polynomial<BFieldElement> =
            Polynomial::new(vec![bfield_elem!(1), bfield_elem!(3), bfield_elem!(7)]);

        let b: Polynomial<BFieldElement> = Polynomial::new(vec![
            bfield_elem!(2),
            bfield_elem!(5),
            bfield_elem!(BFieldElement::MAX as u64),
        ]);

        let expected: Polynomial<BFieldElement> =
            Polynomial::new(vec![bfield_elem!(3), bfield_elem!(8), bfield_elem!(6)]);

        assert_eq!(expected, a + b);
    }

    #[test]
    fn b_field_xgcd_test() {
        let a = 15;
        let b = 25;
        let expected_gcd_ab = 5;
        let (actual_gcd_ab, a_factor, b_factor) = other::xgcd(a, b);

        assert_eq!(expected_gcd_ab, actual_gcd_ab);
        assert_eq!(2, a_factor);
        assert_eq!(-1, b_factor);
        assert_eq!(expected_gcd_ab, a_factor * a + b_factor * b);
    }

    #[test]
    fn mod_pow_test() {
        // These values were found by finding primitive roots of unity and verifying
        // that they are group generators of the right order
        assert!(BFieldElement::new(281474976710656).mod_pow(4).is_one());
        assert_eq!(
            BFieldElement::new(281474976710656),
            BFieldElement::new(281474976710656).mod_pow(5)
        );
        assert!(BFieldElement::new(18446744069414584320).mod_pow(2).is_one()); // 18446744069414584320, 2nd
        assert!(BFieldElement::new(18446744069397807105).mod_pow(8).is_one());
        assert!(BFieldElement::new(2625919085333925275).mod_pow(10).is_one());
        assert!(BFieldElement::new(281474976645120).mod_pow(12).is_one());
    }

    #[test]
    fn get_primitive_root_of_unity_non_powers_of_two_test() {
        // The below list follows from the fact that `prime = 2^32*prod_{i=0}^4(1 + 2^(2^i)) + 1`
        let prime_minus_one_factors = vec![2, 3, 5, 17, 257, 65537];
        for number in prime_minus_one_factors {
            assert!(BFieldElement::ring_one()
                .get_primitive_root_of_unity(number)
                .0
                .is_some());
        }
        assert!(BFieldElement::ring_one()
            .get_primitive_root_of_unity(2u64.pow(32) * 65537u64)
            .0
            .is_some());
        assert!(BFieldElement::ring_one()
            .get_primitive_root_of_unity(2u64.pow(32) * 65537u64 * 257)
            .0
            .is_some());
        assert!(BFieldElement::ring_one()
            .get_primitive_root_of_unity(2u64.pow(32) * 65537u64 * 257 * 17)
            .0
            .is_some());
        assert!(BFieldElement::ring_one()
            .get_primitive_root_of_unity(2u64.pow(32) * 65537u64 * 257 * 17 * 5)
            .0
            .is_some());

        // Largest subgroup of the multiplicative group of the B field
        assert!(BFieldElement::ring_one()
            .get_primitive_root_of_unity(2u64.pow(31) * 65537u64 * 257 * 17 * 5 * 3)
            .0
            .is_some());

        // Negative test for small group sizes
        let non_factors = vec![7, 9, 11, 18];
        for number in non_factors {
            assert!(BFieldElement::ring_one()
                .get_primitive_root_of_unity(number)
                .0
                .is_none());
        }
    }

    #[test]
    fn get_primitive_root_of_unity_test() {
        for i in 1..33 {
            let power = 1 << i;
            let root_result = BFieldElement::ring_one().get_primitive_root_of_unity(power);
            match root_result.0 {
                Some(root) => println!("{} => {},", power, root),
                None => println!("Found no primitive root of unity for n = {}", power),
            };
            let root = root_result.0.unwrap();
            assert!(root.mod_pow(power as u64).is_one());
            assert!(!root.mod_pow(power as u64 / 2).is_one());
        }
    }

    #[test]
    fn lift_coefficients_to_xfield_test() {
        let b_field_mpol = gen_mpolynomial(4, 6, 4, 100);
        let x_field_mpol = super::lift_coefficients_to_xfield(&b_field_mpol);
        assert_eq!(b_field_mpol.degree(), x_field_mpol.degree());
        for (exponents, coefficient) in x_field_mpol.coefficients.iter() {
            assert_eq!(
                coefficient.unlift().unwrap(),
                b_field_mpol.coefficients[exponents]
            );
        }
    }

    fn gen_mpolynomial(
        variable_count: usize,
        term_count: usize,
        exponenent_limit: u128,
        coefficient_limit: u64,
    ) -> MPolynomial<BFieldElement> {
        let mut coefficients: HashMap<Vec<u64>, BFieldElement> = HashMap::new();

        for _ in 0..term_count {
            let key = generate_random_numbers_u128(variable_count, None)
                .iter()
                .map(|x| (*x % exponenent_limit) as u64)
                .collect::<Vec<u64>>();
            let value = gen_bfield_element(coefficient_limit);
            coefficients.insert(key, value);
        }

        MPolynomial {
            variable_count,
            coefficients,
        }
    }

    fn gen_bfield_element(limit: u64) -> BFieldElement {
        let mut rng = rand::thread_rng();

        // adding 1 prevents us from building multivariate polynomial containing zero-coefficients
        let elem = rng.next_u64() % limit + 1;
        BFieldElement::new(elem as u64)
    }
}
