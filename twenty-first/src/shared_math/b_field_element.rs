use super::traits::{FromVecu8, Inverse, PrimitiveRootOfUnity};
use super::x_field_element::XFieldElement;
use crate::shared_math::traits::{CyclicGroupGenerator, FiniteField, ModPowU32, ModPowU64, New};
use crate::util_types::emojihash_trait::{Emojihash, EMOJI_PER_ELEMENT};
use num_traits::{One, Zero};
use rand_distr::{Distribution, Standard};
use std::hash::Hash;

use phf::phf_map;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::convert::{TryFrom, TryInto};
use std::iter::Sum;
use std::num::TryFromIntError;
use std::ops::{AddAssign, MulAssign, SubAssign};
use std::{
    fmt::{self},
    ops::{Add, Div, Mul, Neg, Sub},
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

// BFieldElement ∈ ℤ_{2^64 - 2^32 + 1} using Montgomery representation.
// This implementation follows https://eprint.iacr.org/2022/274.pdf
// and https://github.com/novifinancial/winterfell/pull/101/files
#[derive(Debug, Copy, Clone, Serialize, Deserialize, Default, Hash, PartialEq, Eq)]
pub struct BFieldElement(u64);

pub const BFIELD_ZERO: BFieldElement = BFieldElement::new(0);
pub const BFIELD_ONE: BFieldElement = BFieldElement::new(1);

impl Sum for BFieldElement {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b)
            .unwrap_or_else(BFieldElement::zero)
    }
}

impl BFieldElement {
    pub const BYTES: usize = 8;

    // 2^64 - 2^32 + 1
    pub const QUOTIENT: u64 = 0xffff_ffff_0000_0001u64;
    pub const MAX: u64 = Self::QUOTIENT - 1;

    /// 2^128 mod QUOTIENT; this is used for conversion of elements into Montgomery representation.
    const R2: u64 = 0xFFFFFFFE00000001;

    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(Self::montyred((value as u128) * (Self::R2 as u128)))
    }

    #[inline]
    pub const fn value(&self) -> u64 {
        self.canonical_representation()
    }

    #[inline]
    /// Square the base M times and multiply the result by the tail value
    pub const fn power_accumulator<const N: usize, const M: usize>(
        base: [Self; N],
        tail: [Self; N],
    ) -> [Self; N] {
        let mut result = base;
        let mut i = 0;
        while i < M {
            let mut j = 0;
            while j < N {
                result[j] = Self(Self::montyred(result[j].0 as u128 * result[j].0 as u128));
                j += 1;
            }
            i += 1;
        }

        let mut j = 0;
        while j < N {
            result[j] = Self(Self::montyred(result[j].0 as u128 * tail[j].0 as u128));
            j += 1;
        }
        result
    }

    /// Get a generator for the entire field
    pub const fn generator() -> Self {
        BFieldElement::new(7)
    }

    #[inline]
    pub const fn lift(&self) -> XFieldElement {
        XFieldElement::new_const(*self)
    }

    // You should probably only use `increment` and `decrement` for testing purposes
    pub fn increment(&mut self) {
        *self += Self::one();
    }

    // You should probably only use `increment` and `decrement` for testing purposes
    pub fn decrement(&mut self) {
        *self -= Self::one();
    }

    #[inline]
    const fn canonical_representation(&self) -> u64 {
        Self::montyred(self.0 as u128)
    }

    #[must_use]
    #[inline]
    pub const fn mod_pow(&self, exp: u64) -> Self {
        // Special case for handling 0^0 = 1
        if exp == 0 {
            return BFieldElement::new(1);
        }

        let mut acc = BFieldElement::new(1);
        let bit_length = u64::BITS - exp.leading_zeros();
        let mut i = 0;
        while i < bit_length {
            acc = Self(Self::montyred(acc.0 as u128 * acc.0 as u128));
            if exp & (1 << (bit_length - 1 - i)) != 0 {
                acc = Self(Self::montyred(acc.0 as u128 * self.0 as u128));
            }
            i += 1;
        }

        acc
    }

    /// Convert a `BFieldElement` from a byte slice in native endianness.
    pub fn from_ne_bytes(bytes: &[u8]) -> BFieldElement {
        let mut bytes_copied: [u8; 8] = [0; 8];
        bytes_copied.copy_from_slice(bytes);
        BFieldElement::new(u64::from_ne_bytes(bytes_copied))
    }

    /// Montgomery reduction
    #[inline(always)]
    pub const fn montyred(x: u128) -> u64 {
        // See reference above for a description of the following implementation.
        let xl = x as u64;
        let xh = (x >> 64) as u64;
        let (a, e) = xl.overflowing_add(xl << 32);

        let b = a.wrapping_sub(a >> 32).wrapping_sub(e as u64);

        let (r, c) = xh.overflowing_sub(b);

        // See https://github.com/Neptune-Crypto/twenty-first/pull/70 for various ways
        // of expressing this.
        r.wrapping_sub((1 + !Self::QUOTIENT) * c as u64)
    }

    /// Return the raw bytes or 8-bit chunks of the Montgomery
    /// representation, in little-endian byte order
    pub const fn raw_bytes(&self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Take a slice of 8 bytes and interpret it as an integer in
    /// little-endian byte order, and cast it to a BFieldElement
    /// in Montgomery representation
    pub const fn from_raw_bytes(bytes: &[u8; 8]) -> Self {
        Self(u64::from_le_bytes(*bytes))
    }

    /// Return the raw 16-bit chunks of the Montgomery
    /// representation, in little-endian chunk order
    pub const fn raw_u16s(&self) -> [u16; 4] {
        [
            (self.0 & 0xffff) as u16,
            ((self.0 >> 16) & 0xffff) as u16,
            ((self.0 >> 32) & 0xffff) as u16,
            ((self.0 >> 48) & 0xffff) as u16,
        ]
    }

    /// Take a slice of 4 16-bit chunks and interpret it as an integer in
    /// little-endian chunk order, and cast it to a BFieldElement
    /// in Montgomery representation
    pub const fn from_raw_u16s(chunks: &[u16; 4]) -> Self {
        Self(
            ((chunks[3] as u64) << 48)
                | ((chunks[2] as u64) << 32)
                | ((chunks[1] as u64) << 16)
                | (chunks[0] as u64),
        )
    }

    #[inline]
    pub fn raw_u128(&self) -> u128 {
        self.0.into()
    }

    #[inline]
    pub const fn from_raw_u64(e: u64) -> BFieldElement {
        BFieldElement(e)
    }
}

impl Emojihash for BFieldElement {
    fn emojihash(&self) -> String {
        emojihash::hash(&self.canonical_representation().to_be_bytes())
            .chars()
            .take(EMOJI_PER_ELEMENT)
            .collect::<String>()
    }
}

impl fmt::Display for BFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let canonical_value = Self::canonical_representation(self);
        let cutoff = 256;
        if canonical_value >= Self::QUOTIENT - cutoff {
            write!(f, "-{}", Self::QUOTIENT - canonical_value)
        } else if canonical_value <= cutoff {
            write!(f, "{}", canonical_value)
        } else {
            write!(f, "{:>020}", canonical_value)
        }
    }
}

impl From<u32> for BFieldElement {
    fn from(value: u32) -> Self {
        Self::new(value.into())
    }
}

impl From<u64> for BFieldElement {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<BFieldElement> for u64 {
    fn from(elem: BFieldElement) -> Self {
        elem.canonical_representation()
    }
}

impl From<&BFieldElement> for u64 {
    fn from(elem: &BFieldElement) -> Self {
        elem.canonical_representation()
    }
}

impl TryFrom<BFieldElement> for u32 {
    type Error = TryFromIntError;

    fn try_from(value: BFieldElement) -> Result<Self, Self::Error> {
        u32::try_from(value.canonical_representation())
    }
}

/// Convert a B-field element to a byte array.
/// The client uses this for its database.
impl From<BFieldElement> for [u8; 8] {
    fn from(bfe: BFieldElement) -> Self {
        // It's crucial to map this to the canonical representation
        // before converting. Otherwise the representation is degenerate.
        bfe.canonical_representation().to_le_bytes()
    }
}

impl From<[u8; 8]> for BFieldElement {
    fn from(array: [u8; 8]) -> Self {
        let n: u64 = u64::from_le_bytes(array);
        debug_assert!(
            n <= Self::MAX,
            "Byte representation must represent a valid B field element, less than the quotient."
        );
        BFieldElement::new(n)
    }
}

impl Inverse for BFieldElement {
    #[must_use]
    #[inline]
    fn inverse(&self) -> Self {
        let x = *self;
        assert_ne!(
            x,
            Self::zero(),
            "Attempted to find the multiplicative inverse of zero."
        );

        #[inline(always)]
        const fn exp(base: BFieldElement, exponent: u64) -> BFieldElement {
            let mut res = base;
            let mut i = 0;
            while i < exponent {
                res = Self(BFieldElement::montyred(res.0 as u128 * res.0 as u128));
                i += 1;
            }
            res
        }

        let bin_2_ones = x.square() * x;
        let bin_3_ones = bin_2_ones.square() * x;
        let bin_6_ones = exp(bin_3_ones, 3) * bin_3_ones;
        let bin_12_ones = exp(bin_6_ones, 6) * bin_6_ones;
        let bin_24_ones = exp(bin_12_ones, 12) * bin_12_ones;
        let bin_30_ones = exp(bin_24_ones, 6) * bin_6_ones;
        let bin_31_ones = bin_30_ones.square() * x;
        let bin_31_ones_1_zero = bin_31_ones.square();
        let bin_32_ones = bin_31_ones.square() * x;

        exp(bin_31_ones_1_zero, 32) * bin_32_ones
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
        let mut ret: Vec<Self> = vec![Self::one()];

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

impl Distribution<BFieldElement> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BFieldElement {
        BFieldElement::new(rng.gen_range(0..=BFieldElement::MAX))
    }
}

impl New for BFieldElement {
    fn new_from_usize(&self, value: usize) -> Self {
        Self::new(value as u64)
    }
}

// This is used for: Convert a hash value to a BFieldElement. Consider making From<Blake3Hash> trait
impl FromVecu8 for BFieldElement {
    fn from_vecu8(bytes: Vec<u8>) -> Self {
        // TODO: Right now we only accept if 'bytes' has 8 bytes; while that is true in
        // the single call site this is used, it also seems unnecessarily fragile (when we
        // change from BLAKE3 to Rescue-Prime, the hash length will change and this will be
        // be wrong). We should make this a From<Blake3Hash> to ensure that it has the right
        // length.
        let (eight_bytes, _rest) = bytes.as_slice().split_at(std::mem::size_of::<u64>());
        let coerced: [u8; 8] = eight_bytes.try_into().unwrap();
        coerced.into()
    }
}

impl FiniteField for BFieldElement {}

impl Zero for BFieldElement {
    #[inline]
    fn zero() -> Self {
        BFieldElement::new(0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.canonical_representation() == 0
    }
}

impl One for BFieldElement {
    #[inline]
    fn one() -> Self {
        BFieldElement::new(1)
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.canonical_representation() == 1
    }
}

impl Add for BFieldElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // Compute a + b = a - (p - b).
        let (x1, c1) = self.0.overflowing_sub(Self::QUOTIENT - rhs.0);

        // The following if/else is equivalent to the commented-out code below but
        // the if/else was found to be faster.
        // let adj = 0u32.wrapping_sub(c1 as u32);
        // Self(x1.wrapping_sub(adj as u64))
        // See
        // https://github.com/Neptune-Crypto/twenty-first/pull/70
        if c1 {
            Self(x1.wrapping_add(Self::QUOTIENT))
        } else {
            Self(x1)
        }
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
    fn mul(self, rhs: Self) -> Self {
        Self(Self::montyred((self.0 as u128) * (rhs.0 as u128)))
    }
}

impl Neg for BFieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::zero() - self
    }
}

impl Sub for BFieldElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (x1, c1) = self.0.overflowing_sub(rhs.0);

        // The following code is equivalent to the commented-out code below
        // but they were determined to have near-equiavalent running times. Maybe because
        // subtraction is not used very often.
        // See: https://github.com/Neptune-Crypto/twenty-first/pull/70
        // 1st alternative:
        // if c1 {
        //     Self(x1.wrapping_add(Self::QUOTIENT))
        // } else {
        //     Self(x1)
        // }
        // 2nd alternative:
        // let adj = 0u32.wrapping_sub(c1 as u32);
        // Self(x1.wrapping_sub(adj as u64))
        Self(x1.wrapping_sub((1 + !Self::QUOTIENT) * c1 as u64))
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

impl PrimitiveRootOfUnity for BFieldElement {
    fn primitive_root_of_unity(n: u64) -> Option<BFieldElement> {
        // Check if n is one of the values for which we have pre-calculated roots
        if PRIMITIVE_ROOTS.contains_key(&n) {
            Some(BFieldElement::new(PRIMITIVE_ROOTS[&n]))
        } else if n <= 1 {
            Some(BFieldElement::one())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod b_prime_field_element_test {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    use crate::shared_math::b_field_element::*;
    use crate::shared_math::other::{random_elements, random_elements_array, xgcd};
    use crate::shared_math::polynomial::Polynomial;
    use itertools::izip;
    use proptest::prelude::*;
    use rand::thread_rng;

    #[test]
    fn display_test() {
        // Ensure that display always prints the canonical value, not a number
        // exceeding BFieldElement::QUOTIENT
        let seven: BFieldElement = BFieldElement::new(7);
        let seven_alt: BFieldElement = BFieldElement::new(7 + BFieldElement::QUOTIENT);
        assert_eq!("7", format!("{}", seven));
        assert_eq!("7", format!("{}", seven_alt));

        let minus_one: BFieldElement = BFieldElement::new(BFieldElement::QUOTIENT - 1);
        assert_eq!("-1", format!("{}", minus_one));

        let minus_fifteen: BFieldElement = BFieldElement::new(BFieldElement::QUOTIENT - 15);
        assert_eq!("-15", format!("{}", minus_fifteen));
    }

    #[test]
    fn test_zero_one() {
        let zero = BFieldElement::new(0);
        let one = BFieldElement::new(1);

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(!one.is_zero());
        assert!(one.is_one());
        assert!(BFieldElement::new(BFieldElement::MAX + 1).is_zero());
        assert!(BFieldElement::new(BFieldElement::MAX + 2).is_one());
    }

    #[test]
    fn byte_array_conversion_test() {
        let a = BFieldElement::new(123);
        let array_a: [u8; 8] = a.into();
        assert_eq!(123, array_a[0]);
        (1..7).for_each(|i| {
            assert_eq!(0, array_a[i]);
        });

        let a_converted_back: BFieldElement = array_a.into();
        assert_eq!(a, a_converted_back);

        // Same but with a value above Self::MAX
        let b = BFieldElement::new(123 + BFieldElement::QUOTIENT);
        let array_b: [u8; 8] = b.into();
        assert_eq!(array_a, array_b);

        // Let's also do some PBT
        let xs: Vec<BFieldElement> = random_elements(100);
        for x in xs {
            let array: [u8; 8] = x.into();
            let x_recalculated: BFieldElement = array.into();
            assert_eq!(x, x_recalculated);
        }
    }

    #[should_panic(
        expected = "Byte representation must represent a valid B field element, less than the quotient."
    )]
    #[test]
    fn disallow_conversion_of_u8_array_outside_range() {
        let bad_bfe_array: [u8; 8] = [u8::MAX; 8];
        println!("bad_bfe_array = {:?}", bad_bfe_array);
        let _value: BFieldElement = bad_bfe_array.into();
    }

    #[test]
    fn simple_value_test() {
        let zero: BFieldElement = BFieldElement::new(0);
        assert_eq!(0, zero.value());
        let one: BFieldElement = BFieldElement::one();
        assert_eq!(1, one.value());

        let neinneinnein = BFieldElement::new(999);
        assert_eq!(999, neinneinnein.value());
    }

    #[test]
    fn simple_generator_test() {
        assert_eq!(BFieldElement::new(7), BFieldElement::generator());
        assert!(BFieldElement::new(7)
            .mod_pow(((1u128 << 64) - (1u128 << 32)) as u64)
            .is_one());
        assert!(!BFieldElement::new(7)
            .mod_pow(((1u128 << 64) - (1u128 << 32)) as u64 / 2)
            .is_one());
    }

    #[test]
    fn simple_lift_test() {
        let zero: BFieldElement = BFieldElement::new(0);
        assert!(zero.lift().is_zero());
        assert!(!zero.lift().is_one());

        let one: BFieldElement = BFieldElement::new(1);
        assert!(!one.lift().is_zero());
        assert!(one.lift().is_one());

        let five: BFieldElement = BFieldElement::new(5);
        let five_lifted: XFieldElement = five.lift();
        assert_eq!(Some(five), five_lifted.unlift());
    }

    #[test]
    fn lift_property_test() {
        let elements: Vec<BFieldElement> = random_elements(100);
        for element in elements {
            assert_eq!(Some(element), element.lift().unlift());
        }
    }

    #[test]
    fn next_and_previous_test() {
        let mut val_a = BFieldElement::new(0);
        let mut val_b = BFieldElement::new(1);
        let mut val_c = BFieldElement::new(BFieldElement::MAX - 1);
        let max = BFieldElement::new(BFieldElement::MAX);
        val_a.increment();
        assert!(val_a.is_one());
        val_b.increment();
        assert!(!val_b.is_one());
        assert_eq!(BFieldElement::new(2), val_b);
        val_b.increment();
        assert_eq!(BFieldElement::new(3), val_b);
        assert_ne!(max, val_c);
        val_c.increment();
        assert_eq!(max, val_c);
        val_c.increment();
        assert!(val_c.is_zero());
        val_c.increment();
        assert!(val_c.is_one());
        val_c.decrement();
        assert!(val_c.is_zero());
        val_c.decrement();
        assert_eq!(max, val_c);
        val_c.decrement();
        assert_eq!(BFieldElement::new(BFieldElement::MAX - 1), val_c);
    }

    proptest! {
        #[test]
        fn identity_tests(n in 0u64..BFieldElement::MAX) {
            let zero = BFieldElement::new(0);
            let one = BFieldElement::new(1);
            let other = BFieldElement::new(n);

            prop_assert_eq!(other, zero + other, "left zero identity");
            prop_assert_eq!(other, other + zero, "right zero identity");
            prop_assert_eq!(other, one * other, "left one identity");
            prop_assert_eq!(other, other * one, "right one identity");
        }
    }

    #[test]
    fn inversion_test() {
        let one_inv = BFieldElement::new(1);
        let two_inv = BFieldElement::new(9223372034707292161);
        let three_inv = BFieldElement::new(12297829379609722881);
        let four_inv = BFieldElement::new(13835058052060938241);
        let five_inv = BFieldElement::new(14757395255531667457);
        let six_inv = BFieldElement::new(15372286724512153601);
        let seven_inv = BFieldElement::new(2635249152773512046);
        let eight_inv = BFieldElement::new(16140901060737761281);
        let nine_inv = BFieldElement::new(4099276459869907627);
        let ten_inv = BFieldElement::new(16602069662473125889);
        let eightfive_million_sixhundred_and_seventyone_onehundred_and_six_inv =
            BFieldElement::new(13115294102219178839);

        // With these "alt" values we verify that the degenerated representation of
        // B field elements works.
        let one_alt = BFieldElement::new(BFieldElement::QUOTIENT + 1);
        let two_alt = BFieldElement::new(BFieldElement::QUOTIENT + 2);
        let three_alt = BFieldElement::new(BFieldElement::QUOTIENT + 3);
        assert_eq!(two_inv, BFieldElement::new(2).inverse());
        assert_eq!(three_inv, BFieldElement::new(3).inverse());
        assert_eq!(four_inv, BFieldElement::new(4).inverse());
        assert_eq!(five_inv, BFieldElement::new(5).inverse());
        assert_eq!(six_inv, BFieldElement::new(6).inverse());
        assert_eq!(seven_inv, BFieldElement::new(7).inverse());
        assert_eq!(eight_inv, BFieldElement::new(8).inverse());
        assert_eq!(nine_inv, BFieldElement::new(9).inverse());
        assert_eq!(ten_inv, BFieldElement::new(10).inverse());
        assert_eq!(
            eightfive_million_sixhundred_and_seventyone_onehundred_and_six_inv,
            BFieldElement::new(85671106).inverse()
        );
        assert_eq!(one_inv, one_alt.inverse());
        assert_eq!(two_inv, two_alt.inverse());
        assert_eq!(three_inv, three_alt.inverse());

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
            one_inv,
            two_inv,
            three_inv,
        ];

        let values = [
            BFieldElement::new(1),
            BFieldElement::new(2),
            BFieldElement::new(3),
            BFieldElement::new(4),
            BFieldElement::new(5),
            BFieldElement::new(6),
            BFieldElement::new(7),
            BFieldElement::new(8),
            BFieldElement::new(9),
            BFieldElement::new(10),
            BFieldElement::new(85671106),
            one_alt,
            two_alt,
            three_alt,
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

        let singleton_inversion = BFieldElement::batch_inversion(vec![BFieldElement::new(2)]);
        assert_eq!(1, singleton_inversion.len());
        assert_eq!(two_inv, singleton_inversion[0]);

        let duplet_inversion =
            BFieldElement::batch_inversion(vec![BFieldElement::new(2), BFieldElement::new(1)]);
        assert_eq!(2, duplet_inversion.len());
        assert_eq!(two_inv, duplet_inversion[0]);
        assert_eq!(one_inv, duplet_inversion[1]);
    }

    #[test]
    fn inversion_property_based_test() {
        let elements: Vec<BFieldElement> = random_elements(30);

        for elem in elements {
            if elem.is_zero() {
                continue;
            }

            assert!((elem.inverse() * elem).is_one());
            assert!((elem * elem.inverse()).is_one());
        }
    }

    #[test]
    fn batch_inversion_pbt() {
        let test_iterations = 100;
        for i in 0..test_iterations {
            let rands: Vec<BFieldElement> = random_elements(i);
            let rands_inv: Vec<BFieldElement> = BFieldElement::batch_inversion(rands.clone());
            assert_eq!(i, rands_inv.len());
            for (mut rand, rand_inv) in izip!(rands, rands_inv) {
                assert!((rand * rand_inv).is_one());
                assert!((rand_inv * rand).is_one());
                assert_eq!(rand.inverse(), rand_inv);
                rand.increment();
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
        let elements: Vec<BFieldElement> = random_elements(300);
        let power_input_b: [BFieldElement; 6] = random_elements_array();
        for i in 1..elements.len() {
            let a = elements[i - 1];
            let b = elements[i];

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
    fn mul_div_pbt() {
        // Verify that the mul result is sane
        let rands: Vec<BFieldElement> = random_elements(100);
        for i in 1..rands.len() {
            let prod_mul = rands[i - 1] * rands[i];
            let mut prod_mul_assign = rands[i - 1];
            prod_mul_assign *= rands[i];
            assert_eq!(
                prod_mul, prod_mul_assign,
                "mul and mul_assign must be the same for B field elements"
            );
            assert_eq!(prod_mul / rands[i - 1], rands[i]);
            assert_eq!(prod_mul / rands[i], rands[i - 1]);
        }
    }

    #[test]
    fn add_sub_wrap_around_test() {
        // Ensure that something that exceeds P but is smaller than $2^64$
        // is still the correct field element. The property-based test is unlikely
        // to hit such an element as the chances of doing so are about 2^(-32) for
        // random uniform numbers. So we test this in a separate test
        let element = BFieldElement::new(4);
        let sum = BFieldElement::new(BFieldElement::MAX) + element;
        assert_eq!(BFieldElement::new(3), sum);
        let diff = sum - element;
        assert_eq!(BFieldElement::new(BFieldElement::MAX), diff);
    }

    #[test]
    fn neg_test() {
        assert_eq!(-BFieldElement::zero(), BFieldElement::zero());
        assert_eq!(
            (-BFieldElement::one()).canonical_representation(),
            BFieldElement::MAX
        );
        let max = BFieldElement::new(BFieldElement::MAX);
        let max_plus_one = max + BFieldElement::one();
        let max_plus_two = max_plus_one + BFieldElement::one();
        assert_eq!(BFieldElement::zero(), -max_plus_one);
        assert_eq!(max, -max_plus_two);
    }

    #[test]
    fn equality_and_hash_test() {
        assert_eq!(BFieldElement::zero(), BFieldElement::zero());
        assert_eq!(BFieldElement::one(), BFieldElement::one());
        assert_ne!(BFieldElement::one(), BFieldElement::zero());
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

        std::hash::Hash::hash(&BFieldElement::new(42), &mut hasher_a);
        std::hash::Hash::hash(&BFieldElement::new(42), &mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());

        // Verify that hashing works for non-canonical representations
        hasher_a = DefaultHasher::new();
        hasher_b = DefaultHasher::new();
        let non_canonical = BFieldElement::new(BFieldElement::MAX) + BFieldElement::new(103);
        std::hash::Hash::hash(&(non_canonical), &mut hasher_a);
        std::hash::Hash::hash(&BFieldElement::new(102), &mut hasher_b);
        assert_eq!(hasher_a.finish(), hasher_b.finish());
    }

    #[test]
    fn create_polynomial_test() {
        let a: Polynomial<BFieldElement> = Polynomial::new(vec![
            BFieldElement::new(1),
            BFieldElement::new(3),
            BFieldElement::new(7),
        ]);

        let b: Polynomial<BFieldElement> = Polynomial::new(vec![
            BFieldElement::new(2),
            BFieldElement::new(5),
            BFieldElement::new(BFieldElement::MAX),
        ]);

        let expected: Polynomial<BFieldElement> = Polynomial::new(vec![
            BFieldElement::new(3),
            BFieldElement::new(8),
            BFieldElement::new(6),
        ]);

        assert_eq!(expected, a + b);
    }

    #[test]
    fn b_field_xgcd_test() {
        let a = 15;
        let b = 25;
        let expected_gcd_ab = 5;
        let (actual_gcd_ab, a_factor, b_factor) = xgcd(a, b);

        assert_eq!(expected_gcd_ab, actual_gcd_ab);
        assert_eq!(2, a_factor);
        assert_eq!(-1, b_factor);
        assert_eq!(expected_gcd_ab, a_factor * a + b_factor * b);
    }

    #[test]
    fn mod_pow_test_powers_of_two() {
        let two = BFieldElement::new(2);
        // 2^63 < 2^64, so no wrap-around of B-field element
        for i in 0..64 {
            assert_eq!(BFieldElement::new(1 << i), two.mod_pow(i));
        }
    }

    #[test]
    fn mod_pow_test_powers_of_three() {
        let three = BFieldElement::new(3);
        // 3^40 < 2^64, so no wrap-around of B-field element
        for i in 0..41 {
            assert_eq!(BFieldElement::new(3u64.pow(i as u32)), three.mod_pow(i));
        }
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
    fn get_primitive_root_of_unity_test() {
        for i in 1..33 {
            let power = 1 << i;
            let root_result = BFieldElement::primitive_root_of_unity(power);
            match root_result {
                Some(root) => println!("{} => {},", power, root),
                None => println!("Found no primitive root of unity for n = {}", power),
            };
            let root = root_result.unwrap();
            assert!(root.mod_pow(power).is_one());
            assert!(!root.mod_pow(power / 2).is_one());
        }
    }

    #[test]
    #[should_panic(expected = "Attempted to find the multiplicative inverse of zero.")]
    fn multiplicative_inverse_of_zero() {
        let zero = BFieldElement::zero();
        zero.inverse();
    }

    #[test]
    #[should_panic(expected = "Attempted to find the multiplicative inverse of zero.")]
    fn multiplicative_inverse_of_p() {
        let zero = BFieldElement::new(BFieldElement::QUOTIENT);
        zero.inverse();
    }

    #[test]
    fn u32_conversion() {
        let val = BFieldElement::new(u32::MAX as u64);
        let as_u32: u32 = val.try_into().unwrap();
        assert_eq!(u32::MAX, as_u32);

        for i in 1..100 {
            let invalid_val_0 = BFieldElement::new((u32::MAX as u64) + i);
            let converted_0 = TryInto::<u32>::try_into(invalid_val_0);
            assert!(converted_0.is_err());
        }
    }

    #[test]
    fn uniqueness_of_consecutive_emojis_bfe() {
        let mut prev = BFieldElement::zero().emojihash();
        for n in 1..256 {
            let curr = BFieldElement::new(n).emojihash();
            println!("{}, n: {n}", curr);
            assert_ne!(curr, prev);
            prev = curr
        }

        // Verify that emojihash is independent of representation
        let val = BFieldElement::new(6672);
        let same_val = BFieldElement::new(6672 + BFieldElement::QUOTIENT);
        assert_eq!(val, same_val);
        assert_eq!(val.emojihash(), same_val.emojihash());
    }

    #[test]
    fn inverse_or_zero_bfe() {
        let zero = BFieldElement::zero();
        let one = BFieldElement::one();
        assert_eq!(zero, zero.inverse_or_zero());

        let mut rng = rand::thread_rng();
        let elem: BFieldElement = rng.gen();
        if elem.is_zero() {
            assert_eq!(zero, elem.inverse_or_zero())
        } else {
            assert_eq!(one, elem * elem.inverse_or_zero());
        }
    }

    #[test]
    fn test_random_squares() {
        let mut rng = thread_rng();
        let p = 0xffff_ffff_0000_0001u128;
        for _ in 0..100 {
            let a = rng.next_u64() % (p as u64);
            let asq = (((a as u128) * (a as u128)) % p) as u64;
            let b = BFieldElement::new(a);
            let bsq = BFieldElement::new(asq);
            assert_eq!(bsq, b * b);
            assert_eq!(bsq.value(), (b * b).value());
            assert_eq!(b.value(), a);
            assert_eq!(bsq.value(), asq);
        }
        let one = BFieldElement::new(1);
        assert_eq!(one, one * one);
    }

    #[test]
    fn equals() {
        let a = BFieldElement::one();
        let b = BFieldElement::new(0xffff_ffff_0000_0001u64 - 1)
            * BFieldElement::new(0xffff_ffff_0000_0001u64 - 1);

        // elements are equal
        assert_eq!(a, b);
        assert_eq!(a.value(), b.value());
    }

    #[test]
    fn test_random_raw() {
        let mut rng = thread_rng();
        let p = 0xffff_ffff_0000_0001u128;
        for _ in 0..100 {
            let a = rng.next_u64() % (p as u64);
            let e = BFieldElement::new(a);
            let bytes = e.raw_bytes();
            let c = BFieldElement::from_raw_bytes(&bytes);
            assert_eq!(e, c);
            let mut f = 0u64;
            for (i, b) in bytes.iter().enumerate() {
                f += (*b as u64) << (8 * i);
            }
            assert_eq!(e, BFieldElement(f));

            let chunks = e.raw_u16s();
            let g = BFieldElement::from_raw_u16s(&chunks);
            assert_eq!(e, g);
            let mut h = 0u64;
            for (i, ch) in chunks.iter().enumerate() {
                h += (*ch as u64) << (16 * i);
            }
            assert_eq!(e, BFieldElement(h));
        }
    }

    #[test]
    fn test_fixed_inverse() {
        // (8561862112314395584, 17307602810081694772)
        let a = BFieldElement::new(8561862112314395584);
        let a_inv = a.inverse();
        let a_inv_or_0 = a.inverse_or_zero();
        let expected = BFieldElement::new(17307602810081694772);
        assert_eq!(a_inv, a_inv_or_0);
        assert_eq!(a_inv, expected);
    }

    #[test]
    fn test_fixed_modpow() {
        let exponent = 16608971246357572739u64;
        let base = BFieldElement::new(7808276826625786800);
        let expected = BFieldElement::new(2288673415394035783);
        assert_eq!(base.mod_pow_u64(exponent), expected);
    }

    #[test]
    fn test_fixed_mul() {
        let a = BFieldElement::new(2779336007265862836);
        let b = BFieldElement::new(8146517303801474933);
        let c = a * b;
        let expected = BFieldElement::new(1857758653037316764);
        assert_eq!(c, expected);

        let a = BFieldElement::new(9223372036854775808);
        let b = BFieldElement::new(9223372036854775808);
        let c = a * b;
        let expected = BFieldElement::new(18446744068340842497);
        assert_eq!(c, expected);
    }
}
