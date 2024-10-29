use std::fmt::Display;
use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

use arbitrary::Arbitrary;
use bfieldcodec_derive::BFieldCodec;
use num_traits::ConstOne;
use num_traits::ConstZero;
use num_traits::One;
use num_traits::Zero;
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Standard;
use serde::Deserialize;
use serde::Serialize;

use super::digest::Digest;
use crate::bfe_vec;
use crate::error::TryFromXFieldElementError;
use crate::math::b_field_element::BFieldElement;
use crate::math::polynomial::Polynomial;
use crate::math::traits::CyclicGroupGenerator;
use crate::math::traits::FiniteField;
use crate::math::traits::Inverse;
use crate::math::traits::ModPowU32;
use crate::math::traits::ModPowU64;
use crate::math::traits::PrimitiveRootOfUnity;

pub const EXTENSION_DEGREE: usize = 3;

#[derive(
    Debug, PartialEq, Eq, Copy, Clone, Hash, Serialize, Deserialize, BFieldCodec, Arbitrary,
)]
pub struct XFieldElement {
    pub coefficients: [BFieldElement; EXTENSION_DEGREE],
}

/// Simplifies constructing [extension field element](XFieldElement)s.
///
/// The type [`XFieldElement`] must be in scope for this macro to work.
/// See [`XFieldElement::from`] for supported types.
///
/// # Examples
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe!(1);
/// let b = xfe!([2, 0, 5]);
/// let c = xfe!([3, 0, 2 + 3]);
/// assert_eq!(a + b, c);
/// ```
#[macro_export]
macro_rules! xfe {
    ($value:expr) => {
        XFieldElement::from($value)
    };
}

/// Simplifies constructing vectors of [extension field element][XFieldElement]s.
///
/// The type [`XFieldElement`] must be in scope for this macro to work. See also [`xfe!`].
///
/// # Examples
///
/// Vector of [constants](XFieldElement::new_const).
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe_vec![1, 2, 3];
/// let b = vec![xfe!(1), xfe!(2), xfe!(3)];
/// assert_eq!(a, b);
/// ```
///
/// Vector of general [extension field element](XFieldElement)s.
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe_vec![[1, 0, 0], [0, 2, 0], [0, 0, 3]];
/// let b = vec![xfe!([1, 0, 0]), xfe!([0, 2, 0]), xfe!([0, 0, 3])];
/// assert_eq!(a, b);
/// ```
///
/// Vector with the same [constant](XFieldElement::new_const) for every entry.
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe_vec![42; 15];
/// let b = vec![xfe!(42); 15];
/// assert_eq!(a, b);
/// ```
///
/// Vector with the same general [extension field element](XFieldElement) for every entry.
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe_vec![[42, 43, 44]; 15];
/// let b = vec![xfe!([42, 43, 44]); 15];
/// assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! xfe_vec {
    ($x:expr; $n:expr) => {
        vec![XFieldElement::from($x); $n]
    };
    ([$c0:expr, $c1:expr, $c2:expr]; $n:expr) => {
        vec![XFieldElement::from([$c0, $c1, $c2]); $n]
    };
    ($($x:expr),* $(,)?) => {
        vec![$(XFieldElement::from($x)),*]
    };
    ($([$c0:expr, $c1:expr, $c2:expr]),* $(,)?) => {
        vec![$(XFieldElement::from([$c0, $c1, $c2])),*]
    };
}

/// Simplifies constructing arrays of [extension field element][XFieldElement]s.
///
/// The type [`XFieldElement`] must be in scope for this macro to work. See also [`xfe!`].
///
/// # Examples
///
/// Array of [constants](XFieldElement::new_const).
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe_array![1, 2, 3];
/// let b = [xfe!(1), xfe!(2), xfe!(3)];
/// assert_eq!(a, b);
/// ```
///
/// Array of general [extension field element](XFieldElement)s.
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe_array![[1, 0, 0], [0, 2, 0], [0, 0, 3]];
/// let b = [xfe!([1, 0, 0]), xfe!([0, 2, 0]), xfe!([0, 0, 3])];
/// assert_eq!(a, b);
/// ```
///
/// Array with the same [constant](XFieldElement::new_const) for every entry.
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe_array![42; 15];
/// let b = [xfe!(42); 15];
/// assert_eq!(a, b);
/// ```
///
/// Array with the same general [extension field element](XFieldElement) for every entry.
///
/// ```
/// # use twenty_first::prelude::*;
/// let a = xfe_array![[42, 43, 44]; 15];
/// let b = [xfe!([42, 43, 44]); 15];
/// assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! xfe_array {
    ($x:expr; $n:expr) => {
        [XFieldElement::from($x); $n]
    };
    ([$c0:expr, $c1:expr, $c2:expr]; $n:expr) => {
        [XFieldElement::from([$c0, $c1, $c2]); $n]
    };
    ($($x:expr),* $(,)?) => {
        [$(XFieldElement::from($x)),*]
    };
    ($([$c0:expr, $c1:expr, $c2:expr]),* $(,)?) => {
        [$(XFieldElement::from([$c0, $c1, $c2])),*]
    };
}

impl From<XFieldElement> for Digest {
    /// Interpret the `XFieldElement` as a [`Digest`]. No hashing is performed. This
    /// interpretation can be useful for the
    /// [`AlgebraicHasher`](crate::util_types::algebraic_hasher::AlgebraicHasher) trait and,
    /// by extension, allows building
    /// [`MerkleTree`](crate::util_types::merkle_tree::MerkleTree)s directly from `XFieldElement`s.
    fn from(xfe: XFieldElement) -> Self {
        let [c0, c1, c2] = xfe.coefficients;
        Digest::new([c0, c1, c2, BFieldElement::ZERO, BFieldElement::ZERO])
    }
}

impl TryFrom<Digest> for XFieldElement {
    type Error = TryFromXFieldElementError;

    fn try_from(digest: Digest) -> Result<Self, Self::Error> {
        let [c0, c1, c2, zero_0, zero_1] = digest.values();
        if zero_0 != BFieldElement::ZERO || zero_1 != BFieldElement::ZERO {
            return Err(TryFromXFieldElementError::InvalidDigest);
        }

        Ok(Self::new([c0, c1, c2]))
    }
}

impl Sum for XFieldElement {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or(XFieldElement::ZERO)
    }
}

impl<T> From<T> for XFieldElement
where
    T: Into<BFieldElement>,
{
    fn from(value: T) -> Self {
        Self::new_const(value.into())
    }
}

impl<T> From<[T; EXTENSION_DEGREE]> for XFieldElement
where
    T: Into<BFieldElement>,
{
    fn from(value: [T; EXTENSION_DEGREE]) -> Self {
        Self::new(value.map(Into::into))
    }
}

impl From<Polynomial<'_, BFieldElement>> for XFieldElement {
    fn from(poly: Polynomial<'_, BFieldElement>) -> Self {
        let (_, rem) = poly.naive_divide(&Self::shah_polynomial());
        let mut xfe = [BFieldElement::ZERO; EXTENSION_DEGREE];

        let Ok(rem_degree) = usize::try_from(rem.degree()) else {
            return Self::ZERO;
        };
        xfe[..=rem_degree].copy_from_slice(&rem.coefficients()[..=rem_degree]);

        XFieldElement::new(xfe)
    }
}

impl TryFrom<&[BFieldElement]> for XFieldElement {
    type Error = TryFromXFieldElementError;

    fn try_from(value: &[BFieldElement]) -> Result<Self, Self::Error> {
        value
            .try_into()
            .map(XFieldElement::new)
            .map_err(|_| Self::Error::InvalidLength(value.len()))
    }
}

impl TryFrom<Vec<BFieldElement>> for XFieldElement {
    type Error = TryFromXFieldElementError;

    fn try_from(value: Vec<BFieldElement>) -> Result<Self, Self::Error> {
        XFieldElement::try_from(value.as_ref())
    }
}

impl XFieldElement {
    /// The quotient defining the [field extension](XFieldElement) over the
    /// [base field](BFieldElement), namely x³ - x + 1.
    #[inline]
    pub fn shah_polynomial() -> Polynomial<'static, BFieldElement> {
        Polynomial::new(bfe_vec![1, -1, 0, 1])
    }

    #[inline]
    pub const fn new(coefficients: [BFieldElement; EXTENSION_DEGREE]) -> Self {
        Self { coefficients }
    }

    #[inline]
    pub const fn new_const(element: BFieldElement) -> Self {
        let zero = BFieldElement::ZERO;
        Self::new([element, zero, zero])
    }

    pub fn unlift(&self) -> Option<BFieldElement> {
        if self.coefficients[1].is_zero() && self.coefficients[2].is_zero() {
            Some(self.coefficients[0])
        } else {
            None
        }
    }

    // `increment` and `decrement` are mainly used for testing purposes
    pub fn increment(&mut self, index: usize) {
        self.coefficients[index].increment();
    }

    pub fn decrement(&mut self, index: usize) {
        self.coefficients[index].decrement();
    }
}

impl Inverse for XFieldElement {
    #[must_use]
    fn inverse(&self) -> Self {
        assert!(
            !self.is_zero(),
            "Cannot invert the zero element in the extension field."
        );
        let self_as_poly: Polynomial<BFieldElement> = self.to_owned().into();
        let (_, a, _) = Polynomial::<BFieldElement>::xgcd(self_as_poly, Self::shah_polynomial());
        a.into()
    }
}

impl PrimitiveRootOfUnity for XFieldElement {
    fn primitive_root_of_unity(n: u64) -> Option<XFieldElement> {
        let b_root = BFieldElement::primitive_root_of_unity(n);
        b_root.map(XFieldElement::new_const)
    }
}

impl Distribution<XFieldElement> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> XFieldElement {
        let coefficients = [
            rng.gen::<BFieldElement>(),
            rng.gen::<BFieldElement>(),
            rng.gen::<BFieldElement>(),
        ];
        XFieldElement { coefficients }
    }
}

impl CyclicGroupGenerator for XFieldElement {
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

impl Display for XFieldElement {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(bfe) = self.unlift() {
            return write!(f, "{bfe}_xfe");
        }

        let [c0, c1, c2] = self.coefficients;
        write!(f, "({c2:>020}·x² + {c1:>020}·x + {c0:>020})")
    }
}

impl Zero for XFieldElement {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        self == &Self::ZERO
    }
}

impl ConstZero for XFieldElement {
    const ZERO: Self = Self::new([BFieldElement::ZERO; EXTENSION_DEGREE]);
}

impl One for XFieldElement {
    fn one() -> Self {
        Self::ONE
    }

    fn is_one(&self) -> bool {
        self == &Self::ONE
    }
}

impl ConstOne for XFieldElement {
    const ONE: Self = Self::new([BFieldElement::ONE, BFieldElement::ZERO, BFieldElement::ZERO]);
}

impl FiniteField for XFieldElement {}

impl Add<XFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        let [s0, s1, s2] = self.coefficients;
        let [o0, o1, o2] = other.coefficients;
        let coefficients = [s0 + o0, s1 + o1, s2 + o2];
        Self { coefficients }
    }
}

impl Add<BFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn add(mut self, other: BFieldElement) -> Self {
        self.coefficients[0] += other;
        self
    }
}

/// The `bfe + xfe -> xfe` instance belongs to BFieldElement.
impl Add<XFieldElement> for BFieldElement {
    type Output = XFieldElement;

    #[inline]
    fn add(self, mut other: XFieldElement) -> XFieldElement {
        other.coefficients[0] += self;
        other
    }
}

impl Mul<XFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        // XField * XField means:
        //
        // (ax^2 + bx + c) * (dx^2 + ex + f)   (mod x^3 - x + 1)
        //
        // =   adx^4 + aex^3 + afx^2
        //   + bdx^3 + bex^2 + bfx
        //   + cdx^2 + cex   + cf
        //
        // = adx^4 + (ae + bd)x^3 + (af + be + cd)x^2 + (bf + ce)x + cf   (mod x^3 - x + 1)

        let [c, b, a] = self.coefficients;
        let [f, e, d] = other.coefficients;

        let r0 = c * f - a * e - b * d;
        let r1 = b * f + c * e - a * d + a * e + b * d;
        let r2 = a * f + b * e + c * d + a * d;

        Self::new([r0, r1, r2])
    }
}

/// XField * BField means scalar multiplication of the
/// BFieldElement onto each coefficient of the XField.
impl Mul<BFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: BFieldElement) -> Self {
        let coefficients = self.coefficients.map(|c| c * other);
        Self { coefficients }
    }
}

impl Mul<XFieldElement> for BFieldElement {
    type Output = XFieldElement;

    #[inline]
    fn mul(self, other: XFieldElement) -> XFieldElement {
        let coefficients = other.coefficients.map(|c| c * self);
        XFieldElement { coefficients }
    }
}

impl Neg for XFieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        let coefficients = self.coefficients.map(Neg::neg);
        Self { coefficients }
    }
}

impl Sub<XFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl Sub<BFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, other: BFieldElement) -> Self {
        self + (-other)
    }
}

impl Sub<XFieldElement> for BFieldElement {
    type Output = XFieldElement;

    #[inline]
    fn sub(self, other: XFieldElement) -> XFieldElement {
        self + (-other)
    }
}

impl AddAssign<XFieldElement> for XFieldElement {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.coefficients[0] += rhs.coefficients[0];
        self.coefficients[1] += rhs.coefficients[1];
        self.coefficients[2] += rhs.coefficients[2];
    }
}

impl AddAssign<BFieldElement> for XFieldElement {
    #[inline]
    fn add_assign(&mut self, rhs: BFieldElement) {
        self.coefficients[0] += rhs;
    }
}

impl MulAssign<XFieldElement> for XFieldElement {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<BFieldElement> for XFieldElement {
    #[inline]
    fn mul_assign(&mut self, rhs: BFieldElement) {
        *self = *self * rhs;
    }
}

impl SubAssign<XFieldElement> for XFieldElement {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.coefficients[0] -= rhs.coefficients[0];
        self.coefficients[1] -= rhs.coefficients[1];
        self.coefficients[2] -= rhs.coefficients[2];
    }
}

impl SubAssign<BFieldElement> for XFieldElement {
    #[inline]
    fn sub_assign(&mut self, rhs: BFieldElement) {
        self.coefficients[0] -= rhs;
    }
}

impl Div for XFieldElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self {
        self * other.inverse()
    }
}

impl ModPowU64 for XFieldElement {
    #[inline]
    fn mod_pow_u64(&self, exponent: u64) -> Self {
        let mut x = *self;
        let mut result = Self::one();
        let mut i = exponent;

        while i > 0 {
            if i & 1 == 1 {
                result *= x;
            }

            x *= x;
            i >>= 1;
        }

        result
    }
}

impl ModPowU32 for XFieldElement {
    #[inline]
    fn mod_pow_u32(&self, exp: u32) -> Self {
        self.mod_pow_u64(exp as u64)
    }
}

#[cfg(test)]
mod tests {
    use itertools::izip;
    use itertools::Itertools;
    use num_traits::ConstOne;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::bfe;
    use crate::math::b_field_element::*;
    use crate::math::ntt::intt;
    use crate::math::ntt::ntt;
    use crate::math::other::random_elements;
    use crate::math::x_field_element::*;

    impl proptest::arbitrary::Arbitrary for XFieldElement {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb().boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    #[test]
    fn one_zero_test() {
        let one = XFieldElement::one();
        assert!(one.is_one());
        assert!(one.coefficients[0].is_one());
        assert!(one.coefficients[1].is_zero());
        assert!(one.coefficients[2].is_zero());
        assert_eq!(one, XFieldElement::ONE);
        let zero = XFieldElement::zero();
        assert!(zero.is_zero());
        assert!(zero.coefficients[0].is_zero());
        assert!(zero.coefficients[1].is_zero());
        assert!(zero.coefficients[2].is_zero());
        assert_eq!(zero, XFieldElement::ZERO);
        let two = XFieldElement::new([
            BFieldElement::new(2),
            BFieldElement::ZERO,
            BFieldElement::ZERO,
        ]);
        assert!(!two.is_one());
        assert!(!zero.is_one());
        let one_as_constant_term_0 = XFieldElement::new([
            BFieldElement::new(1),
            BFieldElement::ONE,
            BFieldElement::ZERO,
        ]);
        let one_as_constant_term_1 = XFieldElement::new([
            BFieldElement::new(1),
            BFieldElement::ZERO,
            BFieldElement::ONE,
        ]);
        assert!(!one_as_constant_term_0.is_one());
        assert!(!one_as_constant_term_1.is_one());
        assert!(!one_as_constant_term_0.is_zero());
        assert!(!one_as_constant_term_1.is_zero());
    }

    #[test]
    fn x_field_random_element_generation_test() {
        let rand_xs: Vec<XFieldElement> = random_elements(14);
        assert_eq!(14, rand_xs.len());

        // TODO: Consider doing a statistical test.
        assert!(rand_xs.into_iter().all_unique());
    }

    #[test]
    fn incr_decr_test() {
        let one_const = XFieldElement::new([1, 0, 0].map(BFieldElement::new));
        let two_const = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let one_x = XFieldElement::new([0, 1, 0].map(BFieldElement::new));
        let two_x = XFieldElement::new([0, 2, 0].map(BFieldElement::new));
        let one_x_squared = XFieldElement::new([0, 0, 1].map(BFieldElement::new));
        let two_x_squared = XFieldElement::new([0, 0, 2].map(BFieldElement::new));
        let max_const = XFieldElement::new([BFieldElement::MAX, 0, 0].map(BFieldElement::new));
        let max_x = XFieldElement::new([0, BFieldElement::MAX, 0].map(BFieldElement::new));
        let max_x_squared = XFieldElement::new([0, 0, BFieldElement::MAX].map(BFieldElement::new));
        let mut val = XFieldElement::ZERO;
        val.increment(0);
        assert!(val.is_one());
        val.increment(0);
        assert_eq!(two_const, val);
        val.decrement(0);
        assert!(val.is_one());
        val.decrement(0);
        assert!(val.is_zero());
        val.decrement(0);
        assert_eq!(max_const, val);
        val.decrement(0);
        assert_eq!(max_const - XFieldElement::ONE, val);
        val.decrement(0);
        assert_eq!(max_const - XFieldElement::ONE - XFieldElement::ONE, val);
        val.increment(0);
        val.increment(0);
        val.increment(0);
        assert!(val.is_zero());
        val.increment(1);
        assert_eq!(one_x, val);
        val.increment(1);
        assert_eq!(two_x, val);
        val.decrement(1);
        val.decrement(1);
        assert!(val.is_zero());
        val.decrement(1);
        assert_eq!(max_x, val);
        val.increment(1);
        val.increment(2);
        assert_eq!(one_x_squared, val);
        val.increment(2);
        assert_eq!(two_x_squared, val);
        val.decrement(2);
        val.decrement(2);
        assert!(val.is_zero());
        val.decrement(2);
        assert_eq!(max_x_squared, val);
        val.decrement(1);
        val.decrement(0);
        assert_eq!(max_x_squared + max_x + max_const, val);
        val.decrement(2);
        val.decrement(1);
        val.decrement(0);
        assert_eq!(
            max_x_squared + max_x + max_const - one_const - one_x - one_x_squared,
            val
        );
    }

    #[test]
    fn x_field_add_test() {
        let poly1 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let poly2 = XFieldElement::new([3, 0, 0].map(BFieldElement::new));

        let mut poly_sum = XFieldElement::new([5, 0, 0].map(BFieldElement::new));
        assert_eq!(poly_sum, poly1 + poly2);

        let poly3 = XFieldElement::new([0, 5, 0].map(BFieldElement::new));
        let poly4 = XFieldElement::new([0, 7, 0].map(BFieldElement::new));

        poly_sum = XFieldElement::new([0, 12, 0].map(BFieldElement::new));
        assert_eq!(poly_sum, poly3 + poly4);

        let poly5 = XFieldElement::new([0, 0, 14].map(BFieldElement::new));
        let poly6 = XFieldElement::new([0, 0, 23].map(BFieldElement::new));

        poly_sum = XFieldElement::new([0, 0, 37].map(BFieldElement::new));
        assert_eq!(poly_sum, poly5 + poly6);

        let poly7 = XFieldElement::new([0, 0, BFieldElement::MAX].map(BFieldElement::new));
        let poly8 = XFieldElement::new([0, 0, 23].map(BFieldElement::new));

        poly_sum = XFieldElement::new([0, 0, 22].map(BFieldElement::new));
        assert_eq!(poly_sum, poly7 + poly8);

        let poly9 = XFieldElement::new([BFieldElement::MAX - 2, 12, 4].map(BFieldElement::new));
        let poly10 = XFieldElement::new([2, 45000, BFieldElement::MAX - 3].map(BFieldElement::new));

        poly_sum = XFieldElement::new([BFieldElement::MAX, 45012, 0].map(BFieldElement::new));
        assert_eq!(poly_sum, poly9 + poly10);
    }

    #[test]
    fn x_field_sub_test() {
        let poly1 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let poly2 = XFieldElement::new([3, 0, 0].map(BFieldElement::new));

        let mut poly_diff = XFieldElement::new([1, 0, 0].map(BFieldElement::new));
        assert_eq!(poly_diff, poly2 - poly1);

        let poly3 = XFieldElement::new([0, 5, 0].map(BFieldElement::new));
        let poly4 = XFieldElement::new([0, 7, 0].map(BFieldElement::new));

        poly_diff = XFieldElement::new([0, 2, 0].map(BFieldElement::new));
        assert_eq!(poly_diff, poly4 - poly3);

        let poly5 = XFieldElement::new([0, 0, 14].map(BFieldElement::new));
        let poly6 = XFieldElement::new([0, 0, 23].map(BFieldElement::new));

        poly_diff = XFieldElement::new([0, 0, 9].map(BFieldElement::new));
        assert_eq!(poly_diff, poly6 - poly5);

        let poly7 = XFieldElement::new([0, 0, BFieldElement::MAX].map(BFieldElement::new));
        let poly8 = XFieldElement::new([0, 0, 23].map(BFieldElement::new));

        poly_diff = XFieldElement::new([0, 0, 24].map(BFieldElement::new));
        assert_eq!(poly_diff, poly8 - poly7);

        let poly9 = XFieldElement::new([BFieldElement::MAX - 2, 12, 4].map(BFieldElement::new));
        let poly10 = XFieldElement::new([2, 45000, BFieldElement::MAX - 3].map(BFieldElement::new));

        poly_diff = XFieldElement::new([5, 44988, BFieldElement::MAX - 7].map(BFieldElement::new));
        assert_eq!(poly_diff, poly10 - poly9);
    }

    #[test]
    fn x_field_mul_test() {
        let poly1 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let poly2 = XFieldElement::new([3, 0, 0].map(BFieldElement::new));

        let poly12_product = XFieldElement::new([6, 0, 0].map(BFieldElement::new));
        assert_eq!(poly12_product, poly1 * poly2);

        let poly3 = XFieldElement::new([0, 3, 0].map(BFieldElement::new));
        let poly4 = XFieldElement::new([0, 3, 0].map(BFieldElement::new));

        let poly34_product = XFieldElement::new([0, 0, 9].map(BFieldElement::new));
        assert_eq!(poly34_product, poly3 * poly4);

        let poly5 = XFieldElement::new([125, 0, 0].map(BFieldElement::new));
        let poly6 = XFieldElement::new([0, 0, 5].map(BFieldElement::new));

        let poly56_product = XFieldElement::new([0, 0, 625].map(BFieldElement::new));
        assert_eq!(poly56_product, poly5 * poly6);

        // x^2 * x^2 = x^4 = x^2 - x mod (x^3 - x + 1)
        let poly7 = XFieldElement::new([0, 0, 1].map(BFieldElement::new));
        let poly8 = XFieldElement::new([0, 0, 1].map(BFieldElement::new));

        let poly78_product = XFieldElement::new([0, BFieldElement::MAX, 1].map(BFieldElement::new));
        assert_eq!(poly78_product, poly7 * poly8);

        // x^2 * x = x^3 = x - 1 mod (x^3 - x + 1)
        let poly9 = XFieldElement::new([0, 1, 0].map(BFieldElement::new));
        let poly10 = XFieldElement::new([0, 0, 1].map(BFieldElement::new));

        let poly910_product =
            XFieldElement::new([BFieldElement::MAX, 1, 0].map(BFieldElement::new));
        assert_eq!(poly910_product, poly9 * poly10);

        // (13+2x+3x2)(19+5x2) = 247+122x^2+38x+10x^3+15x^4
        let poly11 = XFieldElement::new([13, 2, 3].map(BFieldElement::new));
        let poly12 = XFieldElement::new([19, 0, 5].map(BFieldElement::new));

        let poly1112_product = XFieldElement::new([237, 33, 137].map(BFieldElement::new));
        assert_eq!(poly1112_product, poly11 * poly12);
    }

    #[test]
    fn x_field_overloaded_arithmetic_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let xfe = rng.gen::<XFieldElement>();
            let bfe = rng.gen::<BFieldElement>();

            // 1. xfe + bfe.lift() = bfe.lift() + xfe
            // 2. xfe + bfe = xfe + bfe.lift()
            // 3. bfe + xfe = xfe + bfe.lift()
            let expected_add = xfe + bfe.lift();
            assert_eq!(expected_add, bfe.lift() + xfe);
            assert_eq!(expected_add, xfe + bfe);
            assert_eq!(expected_add, bfe + xfe);

            // 4. xfe * bfe.lift() = bfe.lift() * xfe
            // 5. xfe * bfe = xfe * bfe.lift()
            // 6. bfe * xfe = xfe * bfe.lift()
            let expected_mul = xfe * bfe.lift();
            assert_eq!(expected_mul, bfe.lift() * xfe);
            assert_eq!(expected_mul, xfe * bfe);
            assert_eq!(expected_mul, bfe * xfe);

            // 7. xfe - bfe = xfe - bfe.lift()
            // 8. bfe - xfe = xfe - bfe.lift()
            assert_eq!(xfe - bfe.lift(), xfe - bfe);
            assert_eq!(bfe.lift() - xfe, bfe - xfe);
        }
    }

    #[test]
    fn x_field_into_test() {
        let zero_poly: XFieldElement = Polynomial::<BFieldElement>::new(vec![]).into();
        assert!(zero_poly.is_zero());

        let shah_zero: XFieldElement = XFieldElement::shah_polynomial().into();
        assert!(shah_zero.is_zero());

        let neg_shah_zero: XFieldElement =
            XFieldElement::shah_polynomial().scalar_mul(bfe!(-1)).into();
        assert!(neg_shah_zero.is_zero());
    }

    #[test]
    fn x_field_xgcp_test() {
        // Verify expected properties of XGCP: symmetry and that gcd is always
        // one. gcd is always one for all field elements.
        let one = XFieldElement::new([1, 0, 0].map(BFieldElement::new));
        let two = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let hundred = XFieldElement::new([100, 0, 0].map(BFieldElement::new));
        let x = XFieldElement::new([0, 1, 0].map(BFieldElement::new));
        let x_squared = XFieldElement::new([0, 0, 1].map(BFieldElement::new));
        let one_one_one = XFieldElement::new([1, 1, 1].map(BFieldElement::new));
        let complex0 = XFieldElement::new([450, 967, 21444444201].map(BFieldElement::new));
        let complex1 = XFieldElement::new([456230, 0, 4563210789].map(BFieldElement::new));
        let complex2 = XFieldElement::new([0, 96701, 456703214].map(BFieldElement::new));
        let complex3 = XFieldElement::new([124504, 9654677, 0].map(BFieldElement::new));
        let complex4 = XFieldElement::new(
            [BFieldElement::MAX, BFieldElement::MAX, BFieldElement::MAX].map(BFieldElement::new),
        );
        let complex5 =
            XFieldElement::new([0, BFieldElement::MAX, BFieldElement::MAX].map(BFieldElement::new));
        let complex6 =
            XFieldElement::new([BFieldElement::MAX, 0, BFieldElement::MAX].map(BFieldElement::new));
        let complex7 =
            XFieldElement::new([BFieldElement::MAX, BFieldElement::MAX, 0].map(BFieldElement::new));

        let x_field_elements = vec![
            one,
            two,
            hundred,
            x,
            x_squared,
            one_one_one,
            complex0,
            complex1,
            complex2,
            complex3,
            complex4,
            complex5,
            complex6,
            complex7,
        ];
        for x_field_element in x_field_elements.iter() {
            let x_field_element_poly: Polynomial<BFieldElement> = (*x_field_element).into();
            // XGCP for x
            let (gcd_0, a_0, b_0) = Polynomial::xgcd(
                x_field_element_poly.clone(),
                XFieldElement::shah_polynomial(),
            );
            let (gcd_1, b_1, a_1) =
                Polynomial::xgcd(XFieldElement::shah_polynomial(), (*x_field_element).into());

            // Verify symmetry, and that all elements are mutual primes, meaning that
            // they form a field
            assert!(gcd_0.is_one());
            assert!(gcd_1.is_one());
            assert_eq!(a_0, a_1);
            assert_eq!(b_0, b_1);

            // Verify Bezout relations: ax + by = gcd
            assert_eq!(
                gcd_0,
                a_0 * x_field_element_poly + b_0 * XFieldElement::shah_polynomial()
            );
        }
    }

    #[test]
    fn x_field_inv_test() {
        let one = XFieldElement::new([1, 0, 0].map(BFieldElement::new));
        let one_inv = one.inverse();
        assert!((one_inv * one).is_one());
        assert!((one * one_inv).is_one());

        let two = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let two_inv = two.inverse();
        assert!((two_inv * two).is_one());
        assert!((two * two_inv).is_one());

        let three = XFieldElement::new([3, 0, 0].map(BFieldElement::new));
        let three_inv = three.inverse();
        assert!((three_inv * three).is_one());
        assert!((three * three_inv).is_one());

        let hundred = XFieldElement::new([100, 0, 0].map(BFieldElement::new));
        let hundred_inv = hundred.inverse();
        assert!((hundred_inv * hundred).is_one());
        assert!((hundred * hundred_inv).is_one());

        let x = XFieldElement::new([0, 1, 0].map(BFieldElement::new));
        let x_inv = x.inverse();
        assert!((x_inv * x).is_one());
        assert!((x * x_inv).is_one());

        // Test batch inversion
        let mut inverses = XFieldElement::batch_inversion(vec![]);
        assert!(inverses.is_empty());
        inverses = XFieldElement::batch_inversion(vec![one]);
        assert_eq!(1, inverses.len());
        assert!(inverses[0].is_one());
        inverses = XFieldElement::batch_inversion(vec![two]);
        assert_eq!(1, inverses.len());
        assert_eq!(two_inv, inverses[0]);
        inverses = XFieldElement::batch_inversion(vec![x]);
        assert_eq!(1, inverses.len());
        assert_eq!(x_inv, inverses[0]);
        inverses = XFieldElement::batch_inversion(vec![two, x]);
        assert_eq!(2, inverses.len());
        assert_eq!(two_inv, inverses[0]);
        assert_eq!(x_inv, inverses[1]);

        let input = vec![one, two, three, hundred, x];
        inverses = XFieldElement::batch_inversion(input.clone());
        let inverses_inverses = XFieldElement::batch_inversion(inverses.clone());
        assert_eq!(input.len(), inverses.len());
        for i in 0..input.len() {
            assert!((inverses[i] * input[i]).is_one());
            assert_eq!(input[i], inverses_inverses[i]);
        }
    }

    #[proptest]
    fn field_element_inversion(
        #[filter(!#x.is_zero())] x: XFieldElement,
        #[filter(!#disturbance.is_zero())]
        #[filter(#x != #disturbance)]
        disturbance: XFieldElement,
    ) {
        let not_x = x - disturbance;
        prop_assert_eq!(XFieldElement::ONE, x * x.inverse());
        prop_assert_eq!(XFieldElement::ONE, not_x * not_x.inverse());
        prop_assert_ne!(XFieldElement::ONE, x * not_x.inverse());
    }

    #[proptest]
    fn field_element_batch_inversion(
        #[filter(!#xs.iter().any(|x| x.is_zero()))] xs: Vec<XFieldElement>,
    ) {
        let inverses = XFieldElement::batch_inversion(xs.clone());
        for (x, inv) in xs.into_iter().zip(inverses) {
            prop_assert_eq!(XFieldElement::ONE, x * inv);
        }
    }

    #[test]
    fn mul_xfe_with_bfe_pbt() {
        let test_iterations = 100;
        let rands_x: Vec<XFieldElement> = random_elements(test_iterations);
        let rands_b: Vec<BFieldElement> = random_elements(test_iterations);
        for (mut x, b) in izip!(rands_x, rands_b) {
            let res_mul = x * b;
            assert_eq!(res_mul.coefficients[0], x.coefficients[0] * b);
            assert_eq!(res_mul.coefficients[1], x.coefficients[1] * b);
            assert_eq!(res_mul.coefficients[2], x.coefficients[2] * b);

            // Also verify that the `MulAssign` implementation agrees with the `Mul` implementation
            x *= b;
            let res_mul_assign = x;
            assert_eq!(res_mul, res_mul_assign);
        }
    }

    #[test]
    fn x_field_division_mul_pbt() {
        let test_iterations = 1000;
        let rands_a: Vec<XFieldElement> = random_elements(test_iterations);
        let rands_b: Vec<XFieldElement> = random_elements(test_iterations);
        for (a, b) in izip!(rands_a, rands_b) {
            let ab = a * b;
            let ba = b * a;
            assert_eq!(ab, ba);
            assert_eq!(ab / b, a);
            assert_eq!(ab / a, b);
            assert_eq!(a * a, a.square());

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

            // Test the add/sub/mul assign operators, when the higher coefficients are zero.
            // Also tests add/sub/mul operators and add/sub/mul assign operators when RHS has
            // the type of B field element. And add/sub/mul operators when LHS is a B-field
            // element and RHS is an X-field element.
            // mul-assign `*=`
            let b_field_b = XFieldElement::new_const(b.coefficients[0]);
            let mut a_mul_b_field_b_as_x = a;
            a_mul_b_field_b_as_x *= b_field_b;
            assert_eq!(a * b_field_b, a_mul_b_field_b_as_x);
            assert_eq!(a, a_mul_b_field_b_as_x / b_field_b);
            assert_eq!(b_field_b, a_mul_b_field_b_as_x / a);
            assert_eq!(a_mul_b_field_b_as_x, a * b.coefficients[0]);
            assert_eq!(a_mul_b_field_b_as_x, b.coefficients[0] * a);
            let mut a_mul_b_field_b_as_b = a;
            a_mul_b_field_b_as_b *= b.coefficients[0];
            assert_eq!(a_mul_b_field_b_as_b, a_mul_b_field_b_as_x);

            // `+=`
            let mut a_plus_b_field_b_as_x = a;
            a_plus_b_field_b_as_x += b_field_b;
            assert_eq!(a + b_field_b, a_plus_b_field_b_as_x);
            assert_eq!(a, a_plus_b_field_b_as_x - b_field_b);
            assert_eq!(b_field_b, a_plus_b_field_b_as_x - a);
            assert_eq!(a_plus_b_field_b_as_x, a + b.coefficients[0]);
            assert_eq!(a_plus_b_field_b_as_x, b.coefficients[0] + a);
            let mut a_plus_b_field_b_as_b = a;
            a_plus_b_field_b_as_b += b.coefficients[0];
            assert_eq!(a_plus_b_field_b_as_b, a_plus_b_field_b_as_x);

            // `-=`
            let mut a_minus_b_field_b_as_x = a;
            a_minus_b_field_b_as_x -= b_field_b;
            assert_eq!(a - b_field_b, a_minus_b_field_b_as_x);
            assert_eq!(a, a_minus_b_field_b_as_x + b_field_b);
            assert_eq!(-b_field_b, a_minus_b_field_b_as_x - a);
            assert_eq!(a_minus_b_field_b_as_x, a - b.coefficients[0]);
            assert_eq!(-a_minus_b_field_b_as_x, b.coefficients[0] - a);
            let mut a_minus_b_field_b_as_b = a;
            a_minus_b_field_b_as_b -= b.coefficients[0];
            assert_eq!(a_minus_b_field_b_as_b, a_minus_b_field_b_as_x);
        }
    }

    #[test]
    fn xfe_mod_pow_zero() {
        assert!(XFieldElement::ZERO.mod_pow_u32(0).is_one());
        assert!(XFieldElement::ZERO.mod_pow_u64(0).is_one());
        assert!(XFieldElement::ONE.mod_pow_u32(0).is_one());
        assert!(XFieldElement::ONE.mod_pow_u64(0).is_one());
    }

    #[proptest]
    fn xfe_mod_pow(base: XFieldElement, #[strategy(0_u32..200)] exponent: u32) {
        let mut acc = XFieldElement::ONE;
        for i in 0..exponent {
            assert_eq!(acc, base.mod_pow_u32(i));
            acc *= base;
        }
    }

    #[test]
    fn xfe_mod_pow_static() {
        let three_to_the_n = |n| xfe!(3).mod_pow_u64(n);
        let actual = [0, 1, 2, 3, 4, 5].map(three_to_the_n);
        let expected = xfe_array![1, 3, 9, 27, 81, 243];
        assert_eq!(expected, actual);
    }

    #[proptest(cases = 100)]
    fn xfe_intt_is_inverse_of_xfe_ntt(
        #[strategy(1..=11)]
        #[map(|log| 1_usize << log)]
        _num_inputs: usize,
        #[strategy(vec(arb(), #_num_inputs))] inputs: Vec<XFieldElement>,
    ) {
        let mut rv = inputs.clone();
        ntt::<XFieldElement>(&mut rv);
        intt::<XFieldElement>(&mut rv);
        prop_assert_eq!(inputs, rv);
    }

    #[proptest(cases = 40)]
    fn xfe_ntt_corresponds_to_polynomial_evaluation(
        #[strategy(1..=11)]
        #[map(|log_2| 1_u64 << log_2)]
        root_order: u64,
        #[strategy(vec(arb(), #root_order as usize))] inputs: Vec<XFieldElement>,
    ) {
        let root = XFieldElement::primitive_root_of_unity(root_order).unwrap();
        let mut rv = inputs.clone();
        ntt::<XFieldElement>(&mut rv);

        let poly = Polynomial::new(inputs);
        let domain = root.get_cyclic_group_elements(None);
        let evaluations = poly.batch_evaluate(&domain);
        prop_assert_eq!(evaluations, rv);
    }

    #[test]
    fn inverse_or_zero_of_zero_is_zero() {
        let zero = XFieldElement::ZERO;
        assert_eq!(zero, zero.inverse_or_zero());
    }

    #[proptest]
    fn inverse_or_zero_of_non_zero_is_inverse(#[filter(!#xfe.is_zero())] xfe: XFieldElement) {
        let inv = xfe.inverse_or_zero();
        prop_assert_ne!(XFieldElement::ZERO, inv);
        prop_assert_eq!(XFieldElement::ONE, xfe * inv);
    }

    #[test]
    #[should_panic(expected = "Cannot invert the zero element in the extension field.")]
    fn multiplicative_inverse_of_zero() {
        let zero = XFieldElement::ZERO;
        let _ = zero.inverse();
    }

    #[proptest]
    fn xfe_to_digest_to_xfe_is_invariant(xfe: XFieldElement) {
        let digest: Digest = xfe.into();
        let xfe2: XFieldElement = digest.try_into().unwrap();
        assert_eq!(xfe, xfe2);
    }

    #[proptest]
    fn converting_random_digest_to_xfield_element_fails(digest: Digest) {
        if XFieldElement::try_from(digest).is_ok() {
            let reason = "Should not be able to convert random `Digest` to an `XFieldElement`.";
            return Err(TestCaseError::Fail(reason.into()));
        }
    }

    #[test]
    fn xfe_macro_can_be_used() {
        let x = xfe!(42);
        let _ = xfe!(42u32);
        let _ = xfe!(-1);
        let _ = xfe!(x);
        let _ = xfe!([x.coefficients[0], x.coefficients[1], x.coefficients[2]]);
        let y = xfe!(bfe!(42));
        assert_eq!(x, y);

        let a = xfe!([bfe!(42), bfe!(43), bfe!(44)]);
        let b = xfe!([42, 43, 44]);
        assert_eq!(a, b);

        let m: [XFieldElement; 3] = xfe_array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let n: Vec<XFieldElement> = xfe_vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        assert_eq!(m.to_vec(), n);
    }

    #[proptest]
    fn xfe_macro_produces_same_result_as_calling_new(coeffs: [BFieldElement; EXTENSION_DEGREE]) {
        let xfe = XFieldElement::new(coeffs);
        prop_assert_eq!(xfe, xfe!(coeffs));
    }

    #[proptest]
    fn xfe_macro_produces_same_result_as_calling_new_const(scalar: BFieldElement) {
        let xfe = XFieldElement::new_const(scalar);
        prop_assert_eq!(xfe, xfe!(scalar));
    }
}
