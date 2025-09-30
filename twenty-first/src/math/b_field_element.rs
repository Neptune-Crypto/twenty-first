use std::convert::TryFrom;
use std::fmt;
use std::fmt::Formatter;
use std::hash::Hash;
use std::iter::Sum;
use std::num::TryFromIntError;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;
use std::str::FromStr;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use get_size2::GetSize;
use num_traits::ConstOne;
use num_traits::ConstZero;
use num_traits::One;
use num_traits::Zero;
use phf::phf_map;
use rand::Rng;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;

use super::traits::Inverse;
use super::traits::PrimitiveRootOfUnity;
use super::x_field_element::XFieldElement;
use crate::error::ParseBFieldElementError;
use crate::math::traits::CyclicGroupGenerator;
use crate::math::traits::FiniteField;
use crate::math::traits::ModPowU32;
use crate::math::traits::ModPowU64;

const PRIMITIVE_ROOTS: phf::Map<u64, u64> = phf_map! {
    0u64 => 1,
    1u64 => 1,
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

/// Base field element ∈ ℤ_{2^64 - 2^32 + 1}.
///
/// In Montgomery representation. This implementation follows <https://eprint.iacr.org/2022/274.pdf>
/// and <https://github.com/novifinancial/winterfell/pull/101/files>.
#[derive(Copy, Clone, Default, Hash, PartialEq, Eq, GetSize)]
#[repr(transparent)]
pub struct BFieldElement(u64);

/// Simplifies constructing [base field element][BFieldElement]s.
///
/// The type [`BFieldElement`] must be in scope for this macro to work.
/// See [`BFieldElement::from`] for supported types.
///
/// # Examples
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = bfe!(42);
/// let b = bfe!(-12); // correctly translates to `BFieldElement::P - 12`
/// let c = bfe!(42 - 12);
/// assert_eq!(a + b, c);
///```
#[macro_export]
macro_rules! bfe {
    ($value:expr) => {
        BFieldElement::from($value)
    };
}

/// Simplifies constructing vectors of [base field element][BFieldElement]s.
///
/// The type [`BFieldElement`] must be in scope for this macro to work. See also [`bfe!`].
///
/// # Examples
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = bfe_vec![1, 2, 3];
/// let b = vec![bfe!(1), bfe!(2), bfe!(3)];
/// assert_eq!(a, b);
/// ```
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = bfe_vec![42; 15];
/// let b = vec![bfe!(42); 15];
/// assert_eq!(a, b);
/// ```
///
#[macro_export]
macro_rules! bfe_vec {
    ($b:expr; $n:expr) => {
        vec![BFieldElement::from($b); $n]
    };
    ($($b:expr),* $(,)?) => {
        vec![$(BFieldElement::from($b)),*]
    };
}

/// Simplifies constructing arrays of [base field element][BFieldElement]s.
///
/// The type [`BFieldElement`] must be in scope for this macro to work. See also [`bfe!`].
///
/// # Examples
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = bfe_array![1, 2, 3];
/// let b = [bfe!(1), bfe!(2), bfe!(3)];
/// assert_eq!(a, b);
/// ```
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = bfe_array![42; 15];
/// let b = [bfe!(42); 15];
/// assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! bfe_array {
    ($b:expr; $n:expr) => {
        [BFieldElement::from($b); $n]
    };
    ($($b:expr),* $(,)?) => {
        [$(BFieldElement::from($b)),*]
    };
}

impl fmt::Debug for BFieldElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BFieldElement").field(&self.value()).finish()
    }
}

impl fmt::LowerHex for BFieldElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.value(), f)
    }
}

impl fmt::UpperHex for BFieldElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.value(), f)
    }
}

impl<'a> Arbitrary<'a> for BFieldElement {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        u.arbitrary().map(BFieldElement::new)
    }
}

impl Serialize for BFieldElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BFieldElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::new(u64::deserialize(deserializer)?))
    }
}

impl Sum for BFieldElement {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b)
            .unwrap_or_else(BFieldElement::zero)
    }
}

impl BFieldElement {
    pub const BYTES: usize = 8;

    /// The base field's prime, _i.e._, 2^64 - 2^32 + 1.
    pub const P: u64 = 0xffff_ffff_0000_0001;
    pub const MAX: u64 = Self::P - 1;

    /// 2^128 mod P; this is used for conversion of elements into Montgomery representation.
    const R2: u64 = 0xffff_fffe_0000_0001;

    /// -2^-1
    pub const MINUS_TWO_INVERSE: Self = Self::new(0x7fff_ffff_8000_0000);

    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(Self::montyred((value as u128) * (Self::R2 as u128)))
    }

    /// Construct a new base field element iff the given value is
    /// [canonical][Self::is_canonical], an error otherwise.
    fn try_new(v: u64) -> Result<Self, ParseBFieldElementError> {
        Self::is_canonical(v)
            .then(|| Self::new(v))
            .ok_or(ParseBFieldElementError::NotCanonical(i128::from(v)))
    }

    #[inline]
    pub const fn value(&self) -> u64 {
        self.canonical_representation()
    }

    #[must_use]
    #[inline]
    pub fn inverse(&self) -> Self {
        #[inline(always)]
        const fn exp(base: BFieldElement, exponent: u64) -> BFieldElement {
            let mut res = base;
            let mut i = 0;
            while i < exponent {
                res = BFieldElement(BFieldElement::montyred(res.0 as u128 * res.0 as u128));
                i += 1;
            }
            res
        }

        let x = *self;
        assert_ne!(
            x,
            Self::zero(),
            "Attempted to find the multiplicative inverse of zero."
        );

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
        let mut acc = BFieldElement::ONE;
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
        r.wrapping_sub((1 + !Self::P) * c as u64)
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

    #[inline]
    pub const fn raw_u64(&self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn is_canonical(x: u64) -> bool {
        x < Self::P
    }
}

impl fmt::Display for BFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let canonical_value = Self::canonical_representation(self);
        let cutoff = 256;
        if canonical_value >= Self::P - cutoff {
            write!(f, "-{}", Self::P - canonical_value)
        } else if canonical_value <= cutoff {
            write!(f, "{canonical_value}")
        } else {
            write!(f, "{canonical_value:>020}")
        }
    }
}

impl FromStr for BFieldElement {
    type Err = ParseBFieldElementError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parsed = s.parse::<i128>().map_err(Self::Err::ParseIntError)?;

        let p = i128::from(Self::P);
        let normalized = match parsed {
            n if n <= -p => return Err(Self::Err::NotCanonical(parsed)),
            n if n < 0 => n + p,
            n => n,
        };

        let bfe_value = u64::try_from(normalized).map_err(|_| Self::Err::NotCanonical(parsed))?;
        Self::try_new(bfe_value)
    }
}

impl From<usize> for BFieldElement {
    fn from(value: usize) -> Self {
        // targets with wider target pointers don't exist at the time of writing
        #[cfg(any(
            target_pointer_width = "16",
            target_pointer_width = "32",
            target_pointer_width = "64",
        ))]
        Self::new(value as u64)
    }
}

impl From<u128> for BFieldElement {
    fn from(value: u128) -> Self {
        fn mod_reduce(x: u128) -> u64 {
            const LOWER_MASK: u64 = 0xFFFF_FFFF;

            let x_lo = x as u64;
            let x_hi = (x >> 64) as u64;
            let x_hi_lo = (x_hi as u32) as u64;
            let x_hi_hi = x_hi >> 32;

            // x_lo - x_hi_hi; potential underflow because `x_hi_hi` may be greater than `x_lo`
            let (tmp0, is_underflow) = x_lo.overflowing_sub(x_hi_hi);
            let tmp1 = tmp0.wrapping_sub(LOWER_MASK * (is_underflow as u64));

            // guaranteed not to underflow
            let tmp2 = (x_hi_lo << 32) - x_hi_lo;

            // adding tmp values gives final result;
            // potential overflow because each of the `tmp`s may be up to 64 bits
            let (result, is_overflow) = tmp1.overflowing_add(tmp2);
            result.wrapping_add(LOWER_MASK * (is_overflow as u64))
        }

        Self::new(mod_reduce(value))
    }
}

macro_rules! impl_from_small_unsigned_int_for_bfe {
    ($($t:ident),+ $(,)?) => {$(
        impl From<$t> for BFieldElement {
            fn from(value: $t) -> Self {
                Self::new(u64::from(value))
            }
        }
    )+};
}

impl_from_small_unsigned_int_for_bfe!(u8, u16, u32, u64);

impl From<isize> for BFieldElement {
    fn from(value: isize) -> Self {
        // targets with wider target pointers don't exist at the time of writing
        #[cfg(target_pointer_width = "16")]
        {
            (value as i16).into()
        }
        #[cfg(target_pointer_width = "32")]
        {
            (value as i32).into()
        }
        #[cfg(target_pointer_width = "64")]
        {
            (value as i64).into()
        }
    }
}

impl From<i64> for BFieldElement {
    fn from(value: i64) -> Self {
        match i128::from(value) {
            0.. => value as u128,
            _ => (value as u128) - BFieldElement::R2 as u128,
        }
        .into()
    }
}

macro_rules! impl_from_small_signed_int_for_bfe {
    ($($t:ident),+ $(,)?) => {$(
        impl From<$t> for BFieldElement {
            fn from(value: $t) -> Self {
                i64::from(value).into()
            }
        }
    )+};
}

impl_from_small_signed_int_for_bfe!(i8, i16, i32);

macro_rules! impl_try_from_bfe_for_int {
    ($($t:ident),+ $(,)?) => {$(
        impl TryFrom<BFieldElement> for $t {
            type Error = TryFromIntError;

            fn try_from(value: BFieldElement) -> Result<Self, Self::Error> {
                $t::try_from(value.canonical_representation())
            }
        }

        impl TryFrom<&BFieldElement> for $t {
            type Error = TryFromIntError;

            fn try_from(value: &BFieldElement) -> Result<Self, Self::Error> {
                $t::try_from(value.canonical_representation())
            }
        }
    )+};
}

impl_try_from_bfe_for_int!(u8, i8, u16, i16, u32, i32, usize, isize);

macro_rules! impl_from_bfe_for_int {
    ($($t:ident),+ $(,)?) => {$(
        impl From<BFieldElement> for $t {
            fn from(elem: BFieldElement) -> Self {
                Self::from(elem.canonical_representation())
            }
        }

        impl From<&BFieldElement> for $t {
            fn from(elem: &BFieldElement) -> Self {
                Self::from(elem.canonical_representation())
            }
        }
    )+};
}

impl_from_bfe_for_int!(u64, u128, i128);

impl From<BFieldElement> for i64 {
    fn from(elem: BFieldElement) -> Self {
        bfe_to_i64(&elem)
    }
}

impl From<&BFieldElement> for i64 {
    fn from(elem: &BFieldElement) -> Self {
        bfe_to_i64(elem)
    }
}

const fn bfe_to_i64(bfe: &BFieldElement) -> i64 {
    let v = bfe.canonical_representation();
    if v <= i64::MAX as u64 {
        v as i64
    } else {
        (v as i128 - BFieldElement::P as i128) as i64
    }
}

/// Convert a B-field element to a byte array.
/// The client uses this for its database.
impl From<BFieldElement> for [u8; BFieldElement::BYTES] {
    fn from(bfe: BFieldElement) -> Self {
        // It's crucial to map this to the canonical representation before converting.
        // Otherwise, the representation is degenerate.
        bfe.canonical_representation().to_le_bytes()
    }
}

impl TryFrom<[u8; BFieldElement::BYTES]> for BFieldElement {
    type Error = ParseBFieldElementError;

    fn try_from(array: [u8; BFieldElement::BYTES]) -> Result<Self, Self::Error> {
        Self::try_new(u64::from_le_bytes(array))
    }
}

impl TryFrom<&[u8]> for BFieldElement {
    type Error = ParseBFieldElementError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        <[u8; BFieldElement::BYTES]>::try_from(bytes)
            .map_err(|_| Self::Error::InvalidNumBytes(bytes.len()))?
            .try_into()
    }
}

impl Inverse for BFieldElement {
    #[inline]
    fn inverse(&self) -> Self {
        self.inverse()
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

impl Distribution<BFieldElement> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BFieldElement {
        BFieldElement::new(rng.random_range(0..=BFieldElement::MAX))
    }
}

impl FiniteField for BFieldElement {}

impl Zero for BFieldElement {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self == &Self::ZERO
    }
}

impl ConstZero for BFieldElement {
    const ZERO: Self = Self::new(0);
}

impl One for BFieldElement {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }

    #[inline]
    fn is_one(&self) -> bool {
        self == &Self::ONE
    }
}

impl ConstOne for BFieldElement {
    const ONE: Self = Self::new(1);
}

impl Add for BFieldElement {
    type Output = Self;

    #[expect(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // Compute a + b = a - (p - b).
        let (x1, c1) = self.0.overflowing_sub(Self::P - rhs.0);

        // The following if/else is equivalent to the commented-out code below but
        // the if/else was found to be faster.
        // let adj = 0u32.wrapping_sub(c1 as u32);
        // Self(x1.wrapping_sub(adj as u64))
        // See
        // https://github.com/Neptune-Crypto/twenty-first/pull/70
        if c1 {
            Self(x1.wrapping_add(Self::P))
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

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (x1, c1) = self.0.overflowing_sub(rhs.0);

        // The following code is equivalent to the commented-out code below
        // but they were determined to have near-equiavalent running times. Maybe because
        // subtraction is not used very often.
        // See: https://github.com/Neptune-Crypto/twenty-first/pull/70
        // 1st alternative:
        // if c1 {
        //     Self(x1.wrapping_add(Self::P))
        // } else {
        //     Self(x1)
        // }
        // 2nd alternative:
        // let adj = 0u32.wrapping_sub(c1 as u32);
        // Self(x1.wrapping_sub(adj as u64))
        Self(x1.wrapping_sub((1 + !Self::P) * c1 as u64))
    }
}

impl Div for BFieldElement {
    type Output = Self;

    #[expect(clippy::suspicious_arithmetic_impl)]
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
        PRIMITIVE_ROOTS.get(&n).map(|&r| BFieldElement::new(r))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    use itertools::izip;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::random;
    use test_strategy::proptest;

    use crate::math::b_field_element::*;
    use crate::math::other::random_elements;
    use crate::math::polynomial::Polynomial;

    impl proptest::arbitrary::Arbitrary for BFieldElement {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb().boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    #[proptest]
    fn get_size(bfe: BFieldElement) {
        prop_assert_eq!(8, bfe.get_size());
    }

    #[proptest]
    fn serialization_and_deserialization_to_and_from_json_is_identity(bfe: BFieldElement) {
        let serialized = serde_json::to_string(&bfe).unwrap();
        let deserialized: BFieldElement = serde_json::from_str(&serialized).unwrap();
        prop_assert_eq!(bfe, deserialized);
    }

    #[proptest]
    fn deserializing_u64_is_like_calling_new(#[strategy(0..=BFieldElement::MAX)] value: u64) {
        let bfe = BFieldElement::new(value);
        let deserialized: BFieldElement = serde_json::from_str(&value.to_string()).unwrap();
        prop_assert_eq!(bfe, deserialized);
    }

    #[test]
    fn parsing_interval_is_open_minus_p_to_p() {
        let p = i128::from(BFieldElement::P);
        let display_then_parse = |v: i128| BFieldElement::from_str(&v.to_string());

        assert!(display_then_parse(-p).is_err());
        assert!(display_then_parse(-p + 1).is_ok());
        assert!(display_then_parse(p - 1).is_ok());
        assert!(display_then_parse(p).is_err());
    }

    #[proptest]
    fn parsing_string_representing_canonical_negative_integer_gives_correct_bfield_element(
        #[strategy(0..=BFieldElement::MAX)] v: u64,
    ) {
        let bfe = BFieldElement::from_str(&(-i128::from(v)).to_string())?;
        prop_assert_eq!(BFieldElement::P - v, bfe.value());
    }

    #[proptest]
    fn parsing_string_representing_canonical_positive_integer_gives_correct_bfield_element(
        #[strategy(0..=BFieldElement::MAX)] v: u64,
    ) {
        let bfe = BFieldElement::from_str(&v.to_string())?;
        prop_assert_eq!(v, bfe.value());
    }

    #[proptest]
    fn parsing_string_representing_too_big_positive_integer_as_bfield_element_gives_error(
        #[strategy(i128::from(BFieldElement::P)..)] v: i128,
    ) {
        let err = BFieldElement::from_str(&v.to_string()).unwrap_err();
        prop_assert_eq!(ParseBFieldElementError::NotCanonical(v), err);
    }

    #[proptest]
    fn parsing_string_representing_too_small_negative_integer_as_bfield_element_gives_error(
        #[strategy(..=i128::from(BFieldElement::P))] v: i128,
    ) {
        let err = BFieldElement::from_str(&v.to_string()).unwrap_err();
        prop_assert_eq!(ParseBFieldElementError::NotCanonical(v), err);
    }

    #[proptest]
    fn zero_is_neutral_element_for_addition(bfe: BFieldElement) {
        let zero = BFieldElement::ZERO;
        prop_assert_eq!(bfe + zero, bfe);
    }

    #[proptest]
    fn one_is_neutral_element_for_multiplication(bfe: BFieldElement) {
        let one = BFieldElement::ONE;
        prop_assert_eq!(bfe * one, bfe);
    }

    #[proptest]
    fn addition_is_commutative(element_0: BFieldElement, element_1: BFieldElement) {
        prop_assert_eq!(element_0 + element_1, element_1 + element_0);
    }

    #[proptest]
    fn multiplication_is_commutative(element_0: BFieldElement, element_1: BFieldElement) {
        prop_assert_eq!(element_0 * element_1, element_1 * element_0);
    }

    #[proptest]

    fn addition_is_associative(
        element_0: BFieldElement,
        element_1: BFieldElement,
        element_2: BFieldElement,
    ) {
        prop_assert_eq!(
            (element_0 + element_1) + element_2,
            element_0 + (element_1 + element_2)
        );
    }

    #[proptest]
    fn multiplication_is_associative(
        element_0: BFieldElement,
        element_1: BFieldElement,
        element_2: BFieldElement,
    ) {
        prop_assert_eq!(
            (element_0 * element_1) * element_2,
            element_0 * (element_1 * element_2)
        );
    }

    #[proptest]
    fn multiplication_distributes_over_addition(
        element_0: BFieldElement,
        element_1: BFieldElement,
        element_2: BFieldElement,
    ) {
        prop_assert_eq!(
            element_0 * (element_1 + element_2),
            element_0 * element_1 + element_0 * element_2
        );
    }

    #[proptest]
    fn multiplication_with_inverse_gives_identity(#[filter(!#bfe.is_zero())] bfe: BFieldElement) {
        prop_assert!((bfe.inverse() * bfe).is_one());
    }

    #[proptest]
    fn division_by_self_gives_identity(#[filter(!#bfe.is_zero())] bfe: BFieldElement) {
        prop_assert!((bfe / bfe).is_one());
    }

    #[proptest]
    fn values_larger_than_modulus_are_handled_correctly(
        #[strategy(BFieldElement::P..)] large_value: u64,
    ) {
        let bfe = BFieldElement::new(large_value);
        let expected_value = large_value - BFieldElement::P;
        prop_assert_eq!(expected_value, bfe.value());
    }

    #[test]
    fn display_test() {
        let seven = BFieldElement::new(7);
        assert_eq!("7", format!("{seven}"));
        assert_eq!("7", format!("{seven:x}"));
        assert_eq!("7", format!("{seven:X}"));
        assert_eq!("0x7", format!("{seven:#x}"));
        assert_eq!("0x7", format!("{seven:#X}"));
        assert_eq!("BFieldElement(7)", format!("{seven:?}"));

        let forty_two = BFieldElement::new(42);
        assert_eq!("42", format!("{forty_two}"));
        assert_eq!("2a", format!("{forty_two:x}"));
        assert_eq!("2A", format!("{forty_two:X}"));
        assert_eq!("0x2a", format!("{forty_two:#x}"));
        assert_eq!("0x2A", format!("{forty_two:#X}"));
        assert_eq!("BFieldElement(42)", format!("{forty_two:?}"));

        let minus_one = BFieldElement::new(BFieldElement::P - 1);
        assert_eq!("-1", format!("{minus_one}"));
        assert_eq!("ffffffff00000000", format!("{minus_one:x}"));
        assert_eq!("FFFFFFFF00000000", format!("{minus_one:X}"));
        assert_eq!("0xffffffff00000000", format!("{minus_one:#x}"));
        assert_eq!("0xFFFFFFFF00000000", format!("{minus_one:#X}"));
        assert_eq!(
            "BFieldElement(18446744069414584320)",
            format!("{minus_one:?}")
        );

        let minus_fifteen = BFieldElement::new(BFieldElement::P - 15);
        assert_eq!("-15", format!("{minus_fifteen}"));
        assert_eq!("fffffffefffffff2", format!("{minus_fifteen:x}"));
        assert_eq!("FFFFFFFEFFFFFFF2", format!("{minus_fifteen:X}"));
        assert_eq!("0xfffffffefffffff2", format!("{minus_fifteen:#x}"));
        assert_eq!("0xFFFFFFFEFFFFFFF2", format!("{minus_fifteen:#X}"));
        assert_eq!(
            "BFieldElement(18446744069414584306)",
            format!("{minus_fifteen:?}")
        );
    }

    #[test]
    fn display_and_from_str_are_reciprocal_unit_test() {
        for bfe in bfe_array![
            -1000, -500, -200, -100, -10, -1, 0, 1, 10, 100, 200, 500, 1000
        ] {
            let bfe_again = bfe.to_string().parse().unwrap();
            assert_eq!(bfe, bfe_again);
        }
    }

    #[proptest]
    fn display_and_from_str_are_reciprocal_prop_test(bfe: BFieldElement) {
        let bfe_again = bfe.to_string().parse()?;
        prop_assert_eq!(bfe, bfe_again);
    }

    #[test]
    fn zero_is_zero() {
        let zero = BFieldElement::zero();
        assert!(zero.is_zero());
        assert_eq!(zero, BFieldElement::ZERO);
    }

    #[proptest]
    fn not_zero_is_nonzero(bfe: BFieldElement) {
        if bfe.value() == 0 {
            return Ok(());
        }
        prop_assert!(!bfe.is_zero());
    }

    #[test]
    fn one_is_one() {
        let one = BFieldElement::one();
        assert!(one.is_one());
        assert_eq!(one, BFieldElement::ONE);
    }

    #[proptest]
    fn not_one_is_not_one(bfe: BFieldElement) {
        if bfe.value() == 1 {
            return Ok(());
        }
        prop_assert!(!bfe.is_one());
    }

    #[test]
    fn one_unequal_zero() {
        let one = BFieldElement::ONE;
        let zero = BFieldElement::ZERO;
        assert_ne!(one, zero);
    }

    #[proptest]
    fn byte_array_of_small_field_elements_is_zero_at_high_indices(value: u8) {
        let bfe = BFieldElement::new(value as u64);
        let byte_array: [u8; 8] = bfe.into();

        prop_assert_eq!(value, byte_array[0]);
        (1..8).for_each(|i| {
            assert_eq!(0, byte_array[i]);
        });
    }

    #[proptest]
    fn byte_array_conversion(bfe: BFieldElement) {
        let array: [u8; 8] = bfe.into();
        let bfe_recalculated: BFieldElement = array.try_into()?;
        prop_assert_eq!(bfe, bfe_recalculated);
    }

    #[proptest]
    fn byte_array_outside_range_is_not_accepted(#[strategy(BFieldElement::P..)] value: u64) {
        let byte_array = value.to_le_bytes();
        prop_assert!(BFieldElement::try_from(byte_array).is_err());
    }

    #[proptest]
    fn value_is_preserved(#[strategy(0..BFieldElement::P)] value: u64) {
        prop_assert_eq!(value, BFieldElement::new(value).value());
    }

    #[test]
    fn supposed_generator_is_generator() {
        let generator = BFieldElement::generator();
        let largest_meaningful_power = BFieldElement::P - 1;
        let generator_pow_p = generator.mod_pow(largest_meaningful_power);
        let generator_pow_p_half = generator.mod_pow(largest_meaningful_power / 2);

        assert_eq!(BFieldElement::ONE, generator_pow_p);
        assert_ne!(BFieldElement::ONE, generator_pow_p_half);
    }

    #[proptest]
    fn lift_then_unlift_preserves_element(bfe: BFieldElement) {
        prop_assert_eq!(Some(bfe), bfe.lift().unlift());
    }

    #[proptest]
    fn increment(mut bfe: BFieldElement) {
        let old_value = bfe.value();
        bfe.increment();
        let expected_value = (old_value + 1) % BFieldElement::P;
        prop_assert_eq!(expected_value, bfe.value());
    }

    #[test]
    fn incrementing_max_value_wraps_around() {
        let mut bfe = BFieldElement::new(BFieldElement::MAX);
        bfe.increment();
        assert_eq!(0, bfe.value());
    }

    #[proptest]
    fn decrement(mut bfe: BFieldElement) {
        let old_value = bfe.value();
        bfe.decrement();
        let expected_value = old_value.checked_sub(1).unwrap_or(BFieldElement::P - 1);
        prop_assert_eq!(expected_value, bfe.value());
    }

    #[test]
    fn decrementing_min_value_wraps_around() {
        let mut bfe = BFieldElement::ZERO;
        bfe.decrement();
        assert_eq!(BFieldElement::MAX, bfe.value());
    }

    #[test]
    fn empty_batch_inversion() {
        let empty_inv = BFieldElement::batch_inversion(vec![]);
        assert!(empty_inv.is_empty());
    }

    #[proptest]
    fn batch_inversion(bfes: Vec<BFieldElement>) {
        let bfes_inv = BFieldElement::batch_inversion(bfes.clone());
        prop_assert_eq!(bfes.len(), bfes_inv.len());
        for (bfe, bfe_inv) in izip!(bfes, bfes_inv) {
            prop_assert_eq!(BFieldElement::ONE, bfe * bfe_inv);
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
        let power_input_b: [BFieldElement; 6] = random();
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
        assert_eq!(-BFieldElement::ZERO, BFieldElement::ZERO);
        assert_eq!(
            (-BFieldElement::ONE).canonical_representation(),
            BFieldElement::MAX
        );
        let max = BFieldElement::new(BFieldElement::MAX);
        let max_plus_one = max + BFieldElement::ONE;
        let max_plus_two = max_plus_one + BFieldElement::ONE;
        assert_eq!(BFieldElement::ZERO, -max_plus_one);
        assert_eq!(max, -max_plus_two);
    }

    #[test]
    fn equality_and_hash_test() {
        assert_eq!(BFieldElement::ZERO, BFieldElement::ZERO);
        assert_eq!(BFieldElement::ONE, BFieldElement::ONE);
        assert_ne!(BFieldElement::ONE, BFieldElement::ZERO);
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
        let a = Polynomial::from([1, 3, 7]);
        let b = Polynomial::from([2, 5, -1]);
        let expected = Polynomial::<BFieldElement>::from([3, 8, 6]);

        assert_eq!(expected, a + b);
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
        assert!(BFieldElement::new(18446744069414584320).mod_pow(2).is_one());
        assert!(BFieldElement::new(18446744069397807105).mod_pow(8).is_one());
        assert!(BFieldElement::new(2625919085333925275).mod_pow(10).is_one());
        assert!(BFieldElement::new(281474976645120).mod_pow(12).is_one());
        assert!(BFieldElement::new(0).mod_pow(0).is_one());
    }

    #[test]
    fn get_primitive_root_of_unity_test() {
        for i in 1..33 {
            let power = 1 << i;
            let root_result = BFieldElement::primitive_root_of_unity(power);
            match root_result {
                Some(root) => println!("{power} => {root},"),
                None => println!("Found no primitive root of unity for n = {power}"),
            };
            let root = root_result.unwrap();
            assert!(root.mod_pow(power).is_one());
            assert!(!root.mod_pow(power / 2).is_one());
        }
    }

    #[test]
    #[should_panic(expected = "Attempted to find the multiplicative inverse of zero.")]
    fn multiplicative_inverse_of_zero() {
        let zero = BFieldElement::ZERO;
        let _ = zero.inverse();
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
    fn inverse_or_zero_bfe() {
        let zero = BFieldElement::ZERO;
        let one = BFieldElement::ONE;
        assert_eq!(zero, zero.inverse_or_zero());

        let mut rng = rand::rng();
        let elem: BFieldElement = rng.random();
        if elem.is_zero() {
            assert_eq!(zero, elem.inverse_or_zero())
        } else {
            assert_eq!(one, elem * elem.inverse_or_zero());
        }
    }

    #[test]
    fn test_random_squares() {
        let mut rng = rand::rng();
        let p = BFieldElement::P;
        for _ in 0..100 {
            let a = rng.random_range(0..p);
            let asq = (((a as u128) * (a as u128)) % (p as u128)) as u64;
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
        let a = BFieldElement::ONE;
        let b = bfe!(BFieldElement::MAX) * bfe!(BFieldElement::MAX);

        // elements are equal
        assert_eq!(a, b);
        assert_eq!(a.value(), b.value());
    }

    #[test]
    fn test_random_raw() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let e: BFieldElement = rng.random();
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
        {
            let a = BFieldElement::new(2779336007265862836);
            let b = BFieldElement::new(8146517303801474933);
            let c = a * b;
            let expected = BFieldElement::new(1857758653037316764);
            assert_eq!(c, expected);
        }

        {
            let a = BFieldElement::new(9223372036854775808);
            let b = BFieldElement::new(9223372036854775808);
            let c = a * b;
            let expected = BFieldElement::new(18446744068340842497);
            assert_eq!(c, expected);
        }
    }

    #[proptest]
    fn conversion_from_i32_to_bfe_is_correct(v: i32) {
        let bfe = BFieldElement::from(v);
        match v {
            0.. => prop_assert_eq!(u64::try_from(v)?, bfe.value()),
            _ => prop_assert_eq!(u64::try_from(-v)?, BFieldElement::P - bfe.value()),
        }
    }

    #[proptest]
    fn conversion_from_isize_to_bfe_is_correct(v: isize) {
        let bfe = BFieldElement::from(v);
        match v {
            0.. => prop_assert_eq!(u64::try_from(v)?, bfe.value()),
            _ => prop_assert_eq!(u64::try_from(-v)?, BFieldElement::P - bfe.value()),
        }
    }

    #[test]
    fn bfield_element_can_be_converted_to_and_from_many_types() {
        let _ = BFieldElement::from(0_u8);
        let _ = BFieldElement::from(0_u16);
        let _ = BFieldElement::from(0_u32);
        let _ = BFieldElement::from(0_u64);
        let _ = BFieldElement::from(0_u128);
        let _ = BFieldElement::from(0_usize);

        let max = bfe!(BFieldElement::MAX);
        assert_eq!(max, BFieldElement::from(-1_i8));
        assert_eq!(max, BFieldElement::from(-1_i16));
        assert_eq!(max, BFieldElement::from(-1_i32));
        assert_eq!(max, BFieldElement::from(-1_i64));
        assert_eq!(max, BFieldElement::from(-1_isize));

        assert!(u8::try_from(BFieldElement::ZERO).is_ok());
        assert!(i8::try_from(BFieldElement::ZERO).is_ok());
        assert!(u16::try_from(BFieldElement::ZERO).is_ok());
        assert!(i16::try_from(BFieldElement::ZERO).is_ok());
        assert!(u32::try_from(BFieldElement::ZERO).is_ok());
        assert!(i32::try_from(BFieldElement::ZERO).is_ok());
        assert!(usize::try_from(BFieldElement::ZERO).is_ok());
        assert!(isize::try_from(BFieldElement::ZERO).is_ok());

        let _ = u64::from(max);
        let _ = i64::from(max);
        let _ = u128::from(max);
        let _ = i128::from(max);
    }

    #[test]
    fn bfield_conversion_works_for_types_min_and_max() {
        let _ = BFieldElement::from(u8::MIN);
        let _ = BFieldElement::from(u8::MAX);
        let _ = BFieldElement::from(u16::MIN);
        let _ = BFieldElement::from(u16::MAX);
        let _ = BFieldElement::from(u32::MIN);
        let _ = BFieldElement::from(u32::MAX);
        let _ = BFieldElement::from(u64::MIN);
        let _ = BFieldElement::from(u64::MAX);
        let _ = BFieldElement::from(u128::MIN);
        let _ = BFieldElement::from(u128::MAX);
        let _ = BFieldElement::from(usize::MIN);
        let _ = BFieldElement::from(usize::MAX);
        let _ = BFieldElement::from(i8::MIN);
        let _ = BFieldElement::from(i8::MAX);
        let _ = BFieldElement::from(i16::MIN);
        let _ = BFieldElement::from(i16::MAX);
        let _ = BFieldElement::from(i32::MIN);
        let _ = BFieldElement::from(i32::MAX);
        let _ = BFieldElement::from(i64::MIN);
        let _ = BFieldElement::from(i64::MAX);
        let _ = BFieldElement::from(isize::MIN);
        let _ = BFieldElement::from(isize::MAX);
    }

    #[proptest]
    fn naive_and_actual_conversion_from_u128_agree(v: u128) {
        fn naive_conversion(x: u128) -> BFieldElement {
            let p = BFieldElement::P as u128;
            let value = (x % p) as u64;
            BFieldElement::new(value)
        }

        prop_assert_eq!(naive_conversion(v), BFieldElement::from(v));
    }

    #[proptest]
    fn naive_and_actual_conversion_from_i64_agree(v: i64) {
        fn naive_conversion(x: i64) -> BFieldElement {
            let p = BFieldElement::P as i128;
            let value = i128::from(x).rem_euclid(p) as u64;
            BFieldElement::new(value)
        }

        prop_assert_eq!(naive_conversion(v), BFieldElement::from(v));
    }

    #[test]
    fn bfe_macro_can_be_used() {
        let b = bfe!(42);
        let _ = bfe!(42u32);
        let _ = bfe!(-1);
        let _ = bfe!(b);
        let _ = bfe!(b.0);
        let _ = bfe!(42_usize);
        let _ = bfe!(-2_isize);

        let c: Vec<BFieldElement> = bfe_vec![1, 2, 3];
        let d: [BFieldElement; 3] = bfe_array![1, 2, 3];
        assert_eq!(c, d);
    }

    #[proptest]
    fn bfe_macro_produces_same_result_as_calling_new(value: u64) {
        prop_assert_eq!(BFieldElement::new(value), bfe!(value));
    }

    #[test]
    fn const_minus_two_inverse_is_really_minus_two_inverse() {
        assert_eq!(bfe!(-2).inverse(), BFieldElement::MINUS_TWO_INVERSE);
    }
}
