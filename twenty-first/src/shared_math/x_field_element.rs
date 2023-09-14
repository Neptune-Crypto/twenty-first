use bfieldcodec_derive::BFieldCodec;
use num_traits::{One, Zero};
use rand::Rng;
use rand_distr::{Distribution, Standard};
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use super::digest::Digest;
use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ZERO};
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::{CyclicGroupGenerator, FiniteField, ModPowU32, ModPowU64, New};
use crate::shared_math::traits::{FromVecu8, Inverse, PrimitiveRootOfUnity};
use crate::util_types::emojihash_trait::Emojihash;

pub const EXTENSION_DEGREE: usize = 3;

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, Serialize, Deserialize, BFieldCodec)]
pub struct XFieldElement {
    pub coefficients: [BFieldElement; EXTENSION_DEGREE],
}

impl From<XFieldElement> for Digest {
    /// Interpret the `XFieldElement` as a [`Digest`]. No hashing is performed. This
    /// interpretation can be useful for the
    /// [`AlgebraicHasher`](crate::util_types::algebraic_hasher::AlgebraicHasher) trait and,
    /// by extension, allows building
    /// [`MerkleTree`](crate::util_types::merkle_tree::MerkleTree)s directly from `XFieldElement`s.
    fn from(xfe: XFieldElement) -> Self {
        Digest::new([
            xfe.coefficients[0],
            xfe.coefficients[1],
            xfe.coefficients[2],
            BFIELD_ZERO,
            BFIELD_ZERO,
        ])
    }
}

impl TryFrom<Digest> for XFieldElement {
    type Error = &'static str;

    fn try_from(digest: Digest) -> Result<Self, Self::Error> {
        let digest_values = digest.values();
        let coefficients = digest_values[..EXTENSION_DEGREE].try_into().unwrap();
        let xfe = Self { coefficients };
        match digest_values[3] == BFIELD_ZERO && digest_values[4] == BFIELD_ZERO {
            true => Ok(xfe),
            false => Err("Digest is not an XFieldElement."),
        }
    }
}

impl Sum for XFieldElement {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b)
            .unwrap_or_else(XFieldElement::zero)
    }
}

impl From<u32> for XFieldElement {
    fn from(value: u32) -> Self {
        XFieldElement::new_const(value.into())
    }
}

impl From<BFieldElement> for XFieldElement {
    fn from(bfe: BFieldElement) -> Self {
        bfe.lift()
    }
}

// TODO: Consider moving this to Polynomial file by untying XFieldElement, and\
// instead referring to its internals. Not sure though.
impl From<XFieldElement> for Polynomial<BFieldElement> {
    fn from(item: XFieldElement) -> Self {
        Self {
            coefficients: item.coefficients.to_vec(),
        }
    }
}

impl From<Polynomial<BFieldElement>> for XFieldElement {
    fn from(poly: Polynomial<BFieldElement>) -> Self {
        let (_, rem) = poly.divide(Self::shah_polynomial());
        let zero = BFieldElement::zero();
        let mut rem_arr: [BFieldElement; EXTENSION_DEGREE] = [zero; EXTENSION_DEGREE];

        for i in 0..rem.degree() + 1 {
            rem_arr[i as usize] = rem.coefficients[i as usize];
        }

        XFieldElement::new(rem_arr)
    }
}

impl TryFrom<&[BFieldElement]> for XFieldElement {
    type Error = String;

    fn try_from(value: &[BFieldElement]) -> Result<Self, Self::Error> {
        let len = value.len();
        value.try_into().map(XFieldElement::new).map_err(|_| {
            format!("Expected {EXTENSION_DEGREE} BFieldElements for digest, but got {len}")
        })
    }
}

impl TryFrom<Vec<BFieldElement>> for XFieldElement {
    type Error = String;

    fn try_from(value: Vec<BFieldElement>) -> Result<Self, Self::Error> {
        XFieldElement::try_from(value.as_ref())
    }
}

impl XFieldElement {
    #[inline]
    pub fn shah_polynomial() -> Polynomial<BFieldElement> {
        Polynomial::new(vec![
            BFieldElement::one(),
            -BFieldElement::one(),
            BFieldElement::zero(),
            BFieldElement::one(),
        ])
    }

    #[inline]
    pub const fn new(coefficients: [BFieldElement; EXTENSION_DEGREE]) -> Self {
        Self { coefficients }
    }

    #[inline]
    pub const fn new_u64(coeffs: [u64; EXTENSION_DEGREE]) -> Self {
        Self {
            coefficients: [
                BFieldElement::new(coeffs[0]),
                BFieldElement::new(coeffs[1]),
                BFieldElement::new(coeffs[2]),
            ],
        }
    }

    #[inline]
    pub const fn new_const(element: BFieldElement) -> Self {
        let zero = BFieldElement::new(0);

        Self {
            coefficients: [element, zero, zero],
        }
    }

    pub fn unlift(&self) -> Option<BFieldElement> {
        if self.coefficients[1].is_zero() && self.coefficients[2].is_zero() {
            Some(self.coefficients[0])
        } else {
            None
        }
    }

    /// Derive a sample `XFieldElement` from a random `Digest`.
    ///
    /// The specific elements of the digest (element 2, 3 and 4)
    /// were chosen because the tasm equivalent of this function
    /// can more efficiently pop elements 0 and 1, leading to a
    /// more efficient `sample_weights()` implementation:
    ///
    /// https://github.com/Neptune-Crypto/twenty-first/pull/66#discussion_r1049771105
    pub fn sample(digest: &Digest) -> Self {
        let elements = digest.values();
        XFieldElement::new([elements[2], elements[3], elements[4]])
    }

    // TODO: Move this into Polynomial when PrimeField can implement Zero + One.
    // Division in 𝔽_p[X], not 𝔽_{p^e} ≅ 𝔽[X]/p(x).
    pub fn xgcd(
        mut x: Polynomial<BFieldElement>,
        mut y: Polynomial<BFieldElement>,
    ) -> (
        Polynomial<BFieldElement>,
        Polynomial<BFieldElement>,
        Polynomial<BFieldElement>,
    ) {
        let mut a_factor = Polynomial::new(vec![BFieldElement::one()]);
        let mut a1 = Polynomial::new(vec![BFieldElement::zero()]);
        let mut b_factor = Polynomial::new(vec![BFieldElement::zero()]);
        let mut b1 = Polynomial::new(vec![BFieldElement::one()]);

        while !y.is_zero() {
            let (quotient, remainder): (Polynomial<BFieldElement>, Polynomial<BFieldElement>) =
                x.clone().divide(y.clone());
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

        // The result is valid up to a coefficient, so we normalize the result,
        // to ensure that x has a leading coefficient of 1.
        // TODO: What happens if x is zero here, can it be?
        let lc = x.leading_coefficient().unwrap();
        let scale = lc.inverse();
        (
            x.scalar_mul(scale),
            a_factor.scalar_mul(scale),
            b_factor.scalar_mul(scale),
        )
    }

    // `increment` and `decrement` are mainly used for testing purposes
    pub fn increment(&mut self, index: usize) {
        self.coefficients[index].increment();
    }

    pub fn decrement(&mut self, index: usize) {
        self.coefficients[index].decrement();
    }
}

impl Emojihash for XFieldElement {
    fn emojihash(&self) -> String {
        self.coefficients.emojihash()
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
        if self.coefficients[2].is_zero() && self.coefficients[1].is_zero() {
            write!(f, "{}_xfe", self.coefficients[0])
        } else {
            write!(
                f,
                "({:>020}·x² + {:>020}·x + {:>020})",
                self.coefficients[2], self.coefficients[1], self.coefficients[0],
            )
        }
    }
}

impl FromVecu8 for XFieldElement {
    fn from_vecu8(bytes: Vec<u8>) -> Self {
        // TODO: See note in BFieldElement's From<Vec<u8>>.
        let bytesize = std::mem::size_of::<u64>();
        let (first_eight_bytes, rest) = bytes.as_slice().split_at(bytesize);
        let (second_eight_bytes, rest2) = rest.split_at(bytesize);
        let (third_eight_bytes, _rest3) = rest2.split_at(bytesize);

        let coefficient0 = BFieldElement::from_vecu8(first_eight_bytes.to_vec());
        let coefficient1 = BFieldElement::from_vecu8(second_eight_bytes.to_vec());
        let coefficient2 = BFieldElement::from_vecu8(third_eight_bytes.to_vec());
        XFieldElement::new([coefficient0, coefficient1, coefficient2])
    }
}

impl Zero for XFieldElement {
    fn zero() -> Self {
        Self {
            coefficients: [BFieldElement::zero(); EXTENSION_DEGREE],
        }
    }

    fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }
}

impl One for XFieldElement {
    fn one() -> Self {
        Self {
            coefficients: [
                BFieldElement::one(),
                BFieldElement::zero(),
                BFieldElement::zero(),
            ],
        }
    }

    fn is_one(&self) -> bool {
        self.coefficients[0].is_one()
            & self.coefficients[1].is_zero()
            & self.coefficients[2].is_zero()
    }
}

impl FiniteField for XFieldElement {}

// TODO: Replace this with a From<usize> trait
// This trait is used by INTT
impl New for XFieldElement {
    fn new_from_usize(&self, value: usize) -> Self {
        Self::new_const(BFieldElement::new(value as u64))
    }
}

impl Add<XFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            coefficients: [
                self.coefficients[0] + other.coefficients[0],
                self.coefficients[1] + other.coefficients[1],
                self.coefficients[2] + other.coefficients[2],
            ],
        }
    }
}

impl Add<BFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn add(self, other: BFieldElement) -> Self {
        Self {
            coefficients: [
                self.coefficients[0] + other,
                self.coefficients[1],
                self.coefficients[2],
            ],
        }
    }
}

/// The `bfe + xfe -> xfe` instance belongs to BFieldElement.
impl Add<XFieldElement> for BFieldElement {
    type Output = XFieldElement;

    #[inline]
    fn add(self, other: XFieldElement) -> XFieldElement {
        XFieldElement {
            coefficients: [
                other.coefficients[0] + self,
                other.coefficients[1],
                other.coefficients[2],
            ],
        }
    }
}

/// XField * XField means:
///
/// (ax^2 + bx + c) * (dx^2 + ex + f)   (mod x^3 - x + 1)
///
/// =   adx^4 + aex^3 + afx^2
///   + bdx^3 + bex^2 + bfx
///   + cdx^2 + cex   + cf
///
/// = adx^4 + (ae + bd)x^3 + (af + be + cd)x^2 + (bf + ce)x + cf   (mod x^3 - x + 1)
impl Mul<XFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        // a_0 * x^2 + b_0 * x + c_0
        let a0 = self.coefficients[2];
        let b0 = self.coefficients[1];
        let c0 = self.coefficients[0];

        // a_1 * x^2 + b_1 * x + c_1
        let a1 = other.coefficients[2];
        let b1 = other.coefficients[1];
        let c1 = other.coefficients[0];

        // (a_0 * x^2 + b_0 * x + c_0) * (a_1 * x^2 + b_1 * x + c_1)
        Self {
            coefficients: [
                c0 * c1 - a0 * b1 - b0 * a1,                     // * x^0
                b0 * c1 + c0 * b1 - a0 * a1 + a0 * b1 + b0 * a1, // * x^1
                a0 * c1 + b0 * b1 + c0 * a1 + a0 * a1,           // * x^2
            ],
        }
    }
}

/// XField * BField means scalar multiplication of the
/// BFieldElement onto each coefficient of the XField.
impl Mul<BFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: BFieldElement) -> Self {
        Self {
            coefficients: [
                self.coefficients[0] * other,
                self.coefficients[1] * other,
                self.coefficients[2] * other,
            ],
        }
    }
}

impl Mul<XFieldElement> for BFieldElement {
    type Output = XFieldElement;

    #[inline]
    fn mul(self, other: XFieldElement) -> XFieldElement {
        XFieldElement {
            coefficients: [
                other.coefficients[0] * self,
                other.coefficients[1] * self,
                other.coefficients[2] * self,
            ],
        }
    }
}

impl Neg for XFieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            coefficients: [
                -self.coefficients[0],
                -self.coefficients[1],
                -self.coefficients[2],
            ],
        }
    }
}

impl Sub<XFieldElement> for XFieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

/// Subtracting a BFieldElement from an XFieldElement
///
/// self - other = self + (-other)
///
/// This overloads to Add and Neg
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
mod x_field_element_test {
    use itertools::{izip, Itertools};
    use rand::{random, thread_rng};

    use crate::shared_math::ntt::{intt, ntt};
    use crate::shared_math::other::{log_2_floor, random_elements};
    use crate::shared_math::{b_field_element::*, x_field_element::*};

    #[test]
    fn one_zero_test() {
        let one = XFieldElement::one();
        assert!(one.is_one());
        assert!(one.coefficients[0].is_one());
        assert!(one.coefficients[1].is_zero());
        assert!(one.coefficients[2].is_zero());
        let zero = XFieldElement::zero();
        assert!(zero.is_zero());
        assert!(zero.coefficients[0].is_zero());
        assert!(zero.coefficients[1].is_zero());
        assert!(zero.coefficients[2].is_zero());
        let two = XFieldElement::new([
            BFieldElement::new(2),
            BFieldElement::zero(),
            BFieldElement::zero(),
        ]);
        assert!(!two.is_one());
        assert!(!zero.is_one());
        let one_as_constant_term_0 = XFieldElement::new([
            BFieldElement::new(1),
            BFieldElement::one(),
            BFieldElement::zero(),
        ]);
        let one_as_constant_term_1 = XFieldElement::new([
            BFieldElement::new(1),
            BFieldElement::zero(),
            BFieldElement::one(),
        ]);
        assert!(!one_as_constant_term_0.is_one());
        assert!(!one_as_constant_term_1.is_one());
        assert!(!one_as_constant_term_0.is_zero());
        assert!(!one_as_constant_term_1.is_zero());
    }

    #[test]
    fn x_field_random_element_generation_test() {
        let rand_xs: Vec<XFieldElement> = random_elements(14);

        // Assert correct length
        assert_eq!(14, rand_xs.len());

        // TODO: Consider doing a statistical test.
        // Assert (probable) uniqueness of all generated elements
        assert_eq!(rand_xs.len(), rand_xs.into_iter().unique().count());
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
        let mut val = XFieldElement::zero();
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
        assert_eq!(max_const - XFieldElement::one(), val);
        val.decrement(0);
        assert_eq!(max_const - XFieldElement::one() - XFieldElement::one(), val);
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

        let neg_shah_zero: XFieldElement = XFieldElement::shah_polynomial()
            .scalar_mul(BFieldElement::new(BFieldElement::P - 1))
            .into();
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

    #[test]
    fn x_field_inversion_pbt() {
        let test_iterations = 100;
        let rands: Vec<XFieldElement> = random_elements(test_iterations);
        for mut rand in rands.clone() {
            let rand_inv_original = rand.inverse();
            assert!((rand * rand_inv_original).is_one());
            assert!((rand_inv_original * rand).is_one());

            // Negative test, verify that when decrementing and incrementing
            // by one in the different indices, we get something
            rand.increment(0);
            assert!((rand * rand.inverse()).is_one());
            assert!((rand.inverse() * rand).is_one());
            assert!(!(rand * rand_inv_original).is_one());
            assert!(!(rand_inv_original * rand).is_one());
            rand.decrement(0);

            rand.increment(1);
            assert!((rand * rand.inverse()).is_one());
            assert!((rand.inverse() * rand).is_one());
            assert!(!(rand * rand_inv_original).is_one());
            assert!(!(rand_inv_original * rand).is_one());
            rand.decrement(1);

            rand.increment(2);
            assert!((rand * rand.inverse()).is_one());
            assert!((rand.inverse() * rand).is_one());
            assert!(!(rand * rand_inv_original).is_one());
            assert!(!(rand_inv_original * rand).is_one());
        }

        // Test batch inversion
        let inverses = XFieldElement::batch_inversion(rands.clone());
        for (val, inv) in izip!(rands, inverses) {
            assert!(!val.is_one()); // Pretty small likely this could happen ^^
            assert!((val * inv).is_one());
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
    fn xfe_mod_pow_zero_test() {
        assert!(
            XFieldElement::zero().mod_pow_u32(0).is_one(),
            "0^0 must be 1 for XFE"
        );
        assert!(
            XFieldElement::zero().mod_pow_u64(0).is_one(),
            "0^0 must be 1 for XFE"
        );
        assert!(
            XFieldElement::one().mod_pow_u32(0).is_one(),
            "0^0 must be 1 for XFE"
        );
        assert!(
            XFieldElement::one().mod_pow_u64(0).is_one(),
            "0^0 must be 1 for XFE"
        );
    }

    #[test]
    fn xfe_mod_pow_pbt() {
        let mut rng = thread_rng();
        for _ in 0..3 {
            let exponent: u32 = rng.gen_range(0..200);
            let base: XFieldElement = random();
            let mut acc = XFieldElement::one();

            for i in 0..exponent {
                assert_eq!(acc, base.mod_pow_u32(i));
                acc *= base;
            }
        }
    }

    #[test]
    fn x_field_mod_pow_test() {
        let const_poly = XFieldElement::new([3, 0, 0].map(BFieldElement::new));

        let expecteds = [1u64, 3, 9, 27, 81, 243].iter().map(|&x| {
            XFieldElement::new([
                BFieldElement::new(x),
                BFieldElement::zero(),
                BFieldElement::zero(),
            ])
        });

        let actuals = [0u64, 1, 2, 3, 4, 5]
            .iter()
            .map(|&n| const_poly.mod_pow_u64(n));

        for (expected, actual) in izip!(expecteds, actuals) {
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn x_field_ntt_test() {
        for root_order in [2, 4, 8, 16, 32] {
            // Verify that NTT and INTT are each other's inverses,
            // and that NTT corresponds to polynomial evaluation
            let inputs_u64: Vec<u64> = (0..root_order).collect();
            let inputs: Vec<XFieldElement> = inputs_u64
                .iter()
                .map(|&x| XFieldElement::new_const(BFieldElement::new(x)))
                .collect();
            let root = XFieldElement::primitive_root_of_unity(root_order).unwrap();
            let log_2_of_n = log_2_floor(inputs.len() as u128) as u32;
            let mut rv = inputs.clone();
            ntt::<XFieldElement>(&mut rv, root.unlift().unwrap(), log_2_of_n);

            // The output should be equivalent to evaluating root^i, i = [0..4]
            // over the polynomial with coefficients 1, 2, 3, 4
            let pol_degree_i_minus_1: Polynomial<XFieldElement> = Polynomial::new(inputs.to_vec());
            let x_domain = root.get_cyclic_group_elements(None);
            for i in 0..inputs.len() {
                assert_eq!(pol_degree_i_minus_1.evaluate(&x_domain[i]), rv[i]);
            }

            // Verify that polynomial interpolation produces the same polynomial
            // Slow Lagrange interpolation is very slow for big inputs. Do not increase
            // this above 32 elements!
            let interpolated = Polynomial::<XFieldElement>::lagrange_interpolate(&x_domain, &rv);
            assert_eq!(pol_degree_i_minus_1, interpolated);

            intt::<XFieldElement>(&mut rv, root.unlift().unwrap(), log_2_of_n);
            assert_eq!(inputs, rv);
        }
    }

    #[test]
    fn uniqueness_of_consecutive_emojis_xfe() {
        let rand_xs: Vec<XFieldElement> = random_elements(14);
        let mut prev = XFieldElement::zero().emojihash();
        for xfe in rand_xs {
            let curr = xfe.emojihash();
            println!("{curr}");
            assert_ne!(curr, prev);
            prev = curr
        }
    }

    #[test]
    fn inverse_or_zero_xfe() {
        let zero = XFieldElement::zero();
        assert_eq!(zero, zero.inverse_or_zero());

        let mut rng = rand::thread_rng();
        let elem: XFieldElement = rng.gen();
        if elem.is_zero() {
            assert_eq!(zero, elem.inverse_or_zero())
        } else {
            let one = XFieldElement::one();
            assert_eq!(one, elem * elem.inverse_or_zero());
        }
    }

    fn emojihash_equivalence_prop(elem: XFieldElement) {
        let expected = elem.emojihash();

        let array: [BFieldElement; EXTENSION_DEGREE] = elem.coefficients;
        assert_eq!(expected, array.emojihash());

        let array_slice: &[BFieldElement] = array.as_ref();
        assert_eq!(expected, array_slice.emojihash());

        let vector: Vec<BFieldElement> = elem.coefficients.to_vec();
        assert_eq!(expected, vector.emojihash());

        let vector_slice: &[BFieldElement] = vector.as_ref();
        assert_eq!(expected, vector_slice.emojihash());

        let vector_ref: &Vec<BFieldElement> = vector.as_ref();
        assert_eq!(expected, vector_ref.emojihash());
    }

    #[test]
    fn emojihash_test() {
        emojihash_equivalence_prop(XFieldElement::zero());

        let mut rng = rand::thread_rng();
        emojihash_equivalence_prop(rng.gen());

        let bfes: Vec<BFieldElement> = random_elements(9);
        let xfes: Vec<XFieldElement> = vec![
            XFieldElement::new([bfes[0], bfes[1], bfes[2]]),
            XFieldElement::new([bfes[3], bfes[4], bfes[5]]),
            XFieldElement::new([bfes[6], bfes[7], bfes[8]]),
        ];

        assert_eq!(
            bfes.emojihash().replace(['[', ']', '|'], ""),
            xfes.emojihash().replace(['[', ']', '|'], ""),
        );
    }

    #[test]
    #[should_panic(expected = "Cannot invert the zero element in the extension field.")]
    fn multiplicative_inverse_of_zero() {
        let zero = XFieldElement::zero();
        zero.inverse();
    }

    #[test]
    fn xfe_to_digest_to_xfe() {
        let xfe: XFieldElement = rand::thread_rng().gen();
        let digest: Digest = xfe.into();
        let xfe2: XFieldElement = digest.try_into().unwrap();
        assert_eq!(xfe, xfe2);

        let digest2: Digest = rand::thread_rng().gen();
        match XFieldElement::try_from(digest2) {
            Ok(_) => panic!("Should not be able to convert a random digest to an XFieldElement."),
            Err(_) => println!("Conversion of random digest to XFieldElement failed as expected."),
        }
    }
}
