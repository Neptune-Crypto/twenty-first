use super::mpolynomial::MPolynomial;
use super::other;
use super::traits::{FromVecu8, Inverse, PrimitiveRootOfUnity};
use super::x_field_element::XFieldElement;
use crate::shared_math::traits::GetRandomElements;
use crate::shared_math::traits::{CyclicGroupGenerator, FiniteField, ModPowU32, ModPowU64, New};
use num_traits::{One, Zero};
use std::hash::{Hash, Hasher};

use phf::phf_map;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::iter::Sum;
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

impl Sum for BFieldElement {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b)
            .unwrap_or_else(BFieldElement::zero)
    }
}

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

    // You should probably only use `increment` and `decrement` for testing purposes
    pub fn increment(&mut self) {
        self.0 = Self::canonical_representation(&(*self + Self::one()));
    }

    // You should probably only use `increment` and `decrement` for testing purposes
    pub fn decrement(&mut self) {
        self.0 = Self::canonical_representation(&(*self - Self::one()));
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
            return BFieldElement::one();
        }

        let mut acc = BFieldElement::one();
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

    /// Convert a byte array to a vector of B field elements
    pub fn from_byte_array<const N: usize>(input: [u8; N]) -> Vec<Self> {
        // Although the output size should be known at compile time, Rust cannot handle constant
        // expressions, so we have to accept a vector as output.
        // Cf. https://github.com/rust-lang/rust/issues/76560
        let bytes_per_element = 7;
        let output_length = N / 7 + if N % 7 == 0 { 0 } else { 1 };
        let full_iterations = N / 7;
        let mut ret: Vec<BFieldElement> = Vec::with_capacity(output_length);
        for i in 0..full_iterations {
            let bytes: [u8; 7] = input[i * bytes_per_element..(i + 1) * bytes_per_element]
                .try_into()
                .expect("slice with incorrect length");

            // I think a specific function can be used here, we shouldn't use a for-loop
            let mut u64_input: [u8; 8] = [0u8; 8];
            u64_input[..bytes_per_element].copy_from_slice(&bytes[..bytes_per_element]);

            let u64_value: u64 = u64::from_le_bytes(u64_input);
            ret.push(BFieldElement(u64_value));
        }

        // Handle any remaining bits, if input length is not a factor of 7.
        if full_iterations + 1 == output_length {
            // We are missing `7 % N` bytes
            let remaining = N % 7;
            let mut u64_input: [u8; 8] = [0u8; 8];
            u64_input[..remaining].copy_from_slice(&input[full_iterations * bytes_per_element..]);
            let u64_value: u64 = u64::from_le_bytes(u64_input);
            ret.push(BFieldElement(u64_value));
        }

        ret
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
        let tmp1 = tmp0.wrapping_sub(Self::LOWER_MASK * (under as u64));

        // compute c * 2^32 - c; this is guaranteed not to underflow
        let tmp2 = (c << 32) - c;

        // add temp values and return the result; because each of the temp may be up to 64 bits,
        // we need to handle potential overflow
        let (result, over) = tmp1.overflowing_add(tmp2);
        result.wrapping_add(Self::LOWER_MASK * (over as u64))
    }
}

impl fmt::Display for BFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let cutoff = 256;
        if self.value() >= Self::QUOTIENT - cutoff {
            write!(f, "-{}", Self::QUOTIENT - self.value())
        } else if self.value() <= cutoff {
            write!(f, "{}", self.value())
        } else {
            write!(f, "{:>020}", self.value())
        }
    }
}

impl From<u32> for BFieldElement {
    fn from(value: u32) -> Self {
        BFieldElement::new(value.into())
    }
}

impl From<u64> for BFieldElement {
    fn from(value: u64) -> Self {
        Self(value)
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
        u32::try_from(value.0)
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
        assert!(
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
        fn exp(base: BFieldElement, exponent: u64) -> BFieldElement {
            let mut res = base;
            for _ in 0..exponent {
                res *= res
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
    fn zero() -> Self {
        BFieldElement(0)
    }

    fn is_zero(&self) -> bool {
        self.canonical_representation() == 0
    }
}

impl One for BFieldElement {
    fn one() -> Self {
        BFieldElement(1)
    }

    fn is_one(&self) -> bool {
        self.canonical_representation() == 1
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

impl PrimitiveRootOfUnity for BFieldElement {
    fn primitive_root_of_unity(n: u64) -> Option<BFieldElement> {
        // Check if n is one of the values for which we have pre-calculated roots
        if PRIMITIVE_ROOTS.contains_key(&(n as u64)) {
            Some(BFieldElement::new(PRIMITIVE_ROOTS[&(n as u64)]))
        } else if n <= 1 {
            Some(BFieldElement::one())
        } else {
            None
        }
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
    use rand::thread_rng;

    // TODO: Move this into separate file.
    macro_rules! bfield_elem {
        ($value:expr) => {{
            BFieldElement::new($value)
        }};
    }

    #[test]
    fn display_test() {
        // Ensure that display always prints the canonical value, not a number
        // exceeding BFieldElement::QUOTIENT
        let seven: BFieldElement = BFieldElement(7);
        let seven_alt: BFieldElement = BFieldElement(7 + BFieldElement::QUOTIENT);
        assert_eq!("7", format!("{}", seven));
        assert_eq!("7", format!("{}", seven_alt));

        let minus_one: BFieldElement = BFieldElement(BFieldElement::QUOTIENT - 1);
        assert_eq!("-1", format!("{}", minus_one));

        let minus_fifteen: BFieldElement = BFieldElement(BFieldElement::QUOTIENT - 15);
        assert_eq!("-15", format!("{}", minus_fifteen));
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
    fn byte_array_conversion_test() {
        let a = BFieldElement::new(123);
        let array_a: [u8; 8] = a.into();
        assert_eq!(123, array_a[0]);
        for i in 1..7 {
            assert_eq!(0, array_a[i]);
        }

        let a_converted_back: BFieldElement = array_a.into();
        assert_eq!(a, a_converted_back);

        // Same but with a value above Self::MAX
        let b = BFieldElement::new(123 + BFieldElement::QUOTIENT);
        let array_b: [u8; 8] = b.into();
        assert_eq!(array_a, array_b);

        // Let's also do some PBT
        let mut prng = thread_rng();
        let xs: Vec<BFieldElement> = BFieldElement::random_elements(100, &mut prng);
        for x in xs {
            let array: [u8; 8] = x.into();
            let x_recalculated: BFieldElement = array.into();
            assert_eq!(x, x_recalculated);
        }
    }

    #[test]
    fn byte_array_conversion_multiple_test() {
        // Ensure we can't overflow
        let byte_array_0: [u8; 7] = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let output = BFieldElement::from_byte_array(byte_array_0);
        assert_eq!(1, output.len());
        assert_eq!(BFieldElement((1 << 56) - 1), output[0]);

        // Ensure we're using little-endianness
        let byte_array_1: [u8; 7] = [100, 0, 0, 0, 0, 0, 0];
        let output = BFieldElement::from_byte_array(byte_array_1);
        assert_eq!(1, output.len());
        assert_eq!(BFieldElement(100), output[0]);

        // Ensure we can handle bigger inputs, with lengths not multiple of 7
        let byte_array_3: [u8; 33] = [
            100, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0,
            0, 0, 150, 0, 0, 0, 0,
        ];
        let output = BFieldElement::from_byte_array(byte_array_3);
        assert_eq!(5, output.len());
        assert_eq!(BFieldElement(100), output[0]);
        assert_eq!(BFieldElement(200), output[1]);
        assert_eq!(BFieldElement(100 * 256 + 100), output[2]);
        assert_eq!(BFieldElement(100), output[3]);
        assert_eq!(BFieldElement(150), output[4]);

        // Assert more sizes can be handled, without crashing
        assert_eq!(1, BFieldElement::from_byte_array([0u8; 1]).len());
        assert_eq!(1, BFieldElement::from_byte_array([0u8; 3]).len());
        assert_eq!(1, BFieldElement::from_byte_array([0u8; 2]).len());
        assert_eq!(1, BFieldElement::from_byte_array([0u8; 4]).len());
        assert_eq!(1, BFieldElement::from_byte_array([0u8; 5]).len());
        assert_eq!(1, BFieldElement::from_byte_array([0u8; 6]).len());
        assert_eq!(2, BFieldElement::from_byte_array([0u8; 8]).len());
        assert_eq!(2, BFieldElement::from_byte_array([0u8; 9]).len());
        assert_eq!(2, BFieldElement::from_byte_array([0u8; 14]).len());
        assert_eq!(3, BFieldElement::from_byte_array([0u8; 15]).len());
    }

    #[should_panic(
        expected = "Byte representation must represent a valid B field element, less than the quotient."
    )]
    #[test]
    fn disallow_conversion_of_u8_array_outside_range() {
        let bad_bfe_array: [u8; 8] = [
            u8::MAX,
            255,
            0xFF,
            0o377,
            0_255,
            'ÿ' as u8,
            0b11111111,
            (('Ǿ' as u16) / 2) as u8,
        ];
        println!("bad_bfe_array = {:?}", bad_bfe_array);
        let _value: BFieldElement = bad_bfe_array.into();
    }

    #[test]
    fn simple_value_test() {
        let zero: BFieldElement = bfield_elem!(0);
        assert_eq!(0, zero.value());
        let one: BFieldElement = BFieldElement::one();
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

        // With these "alt" values we verify that the degenerated representation of
        // B field elements works.
        let one_alt = BFieldElement(BFieldElement::QUOTIENT + 1);
        let two_alt = BFieldElement(BFieldElement::QUOTIENT + 2);
        let three_alt = BFieldElement(BFieldElement::QUOTIENT + 3);
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
        assert_eq!((-BFieldElement::one()).0, BFieldElement::MAX);
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
    fn get_primitive_root_of_unity_test() {
        for i in 1..33 {
            let power = 1 << i;
            let root_result = BFieldElement::primitive_root_of_unity(power);
            match root_result {
                Some(root) => println!("{} => {},", power, root),
                None => println!("Found no primitive root of unity for n = {}", power),
            };
            let root = root_result.unwrap();
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
}
