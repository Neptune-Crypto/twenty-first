use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::GetRandomElements;
use crate::shared_math::traits::{
    CyclicGroupGenerator, FieldBatchInversion, IdentityValues, ModPowU32, ModPowU64, New,
    PrimeFieldElement,
};
use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::ops::Div;
use std::{
    fmt::Display,
    ops::{Add, Mul, Neg, Sub},
};

use super::traits::GetPrimitiveRootOfUnity;

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, Serialize, Deserialize)]
pub struct XFieldElement {
    pub coefficients: [BFieldElement; 3],
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
        let zero = BFieldElement::ring_zero();
        let mut rem_arr: [BFieldElement; 3] = [zero; 3];

        for i in 0..rem.degree() + 1 {
            rem_arr[i as usize] = rem.coefficients[i as usize];
        }

        XFieldElement::new(rem_arr)
    }
}

impl XFieldElement {
    pub fn shah_polynomial() -> Polynomial<BFieldElement> {
        Polynomial::new(vec![
            BFieldElement::ring_one(),
            -BFieldElement::ring_one(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_one(),
        ])
    }

    pub fn new(coefficients: [BFieldElement; 3]) -> Self {
        Self { coefficients }
    }

    pub fn new_const(element: BFieldElement) -> Self {
        let zero = BFieldElement::ring_zero();

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

    pub fn ring_zero() -> Self {
        Self {
            coefficients: [0, 0, 0].map(BFieldElement::new),
        }
    }

    // 0x^2 + 0x + 1
    pub fn ring_one() -> Self {
        Self {
            coefficients: [1, 0, 0].map(BFieldElement::new),
        }
    }

    // Division in 𝔽_p[X], not 𝔽_{p^e} ≅ 𝔽[X]/p(x).
    pub fn xgcd(
        mut x: Polynomial<BFieldElement>,
        mut y: Polynomial<BFieldElement>,
    ) -> (
        Polynomial<BFieldElement>,
        Polynomial<BFieldElement>,
        Polynomial<BFieldElement>,
    ) {
        let mut a_factor = Polynomial::new(vec![BFieldElement::ring_one()]);
        let mut a1 = Polynomial::new(vec![BFieldElement::ring_zero()]);
        let mut b_factor = Polynomial::new(vec![BFieldElement::ring_zero()]);
        let mut b1 = Polynomial::new(vec![BFieldElement::ring_one()]);

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
        let scale = lc.inv();
        (
            x.scalar_mul(scale),
            a_factor.scalar_mul(scale),
            b_factor.scalar_mul(scale),
        )
    }

    pub fn inv(&self) -> Self {
        let self_as_poly: Polynomial<BFieldElement> = self.to_owned().into();
        let (_, a, _) = Self::xgcd(self_as_poly, Self::shah_polynomial());
        a.into()
    }

    // `incr` and `decr` are mainly used for testing purposes
    pub fn incr(&mut self, index: usize) {
        self.coefficients[index].increment();
    }

    pub fn decr(&mut self, index: usize) {
        self.coefficients[index].decrement();
    }

    // TODO: legendre_symbol
}

impl GetPrimitiveRootOfUnity for XFieldElement {
    fn get_primitive_root_of_unity(&self, n: u128) -> (Option<XFieldElement>, Vec<u128>) {
        let (b_root, primes) = self.coefficients[0].get_primitive_root_of_unity(n);
        let x_root = b_root.map(XFieldElement::new_const);

        (x_root, primes)
    }
}

impl FieldBatchInversion for XFieldElement {
    fn batch_inversion(input: Vec<Self>) -> Vec<Self> {
        // Adapted from https://paulmillr.com/posts/noble-secp256k1-fast-ecc/#batch-inversion
        let input_length = input.len();
        if input_length == 0 {
            return Vec::<Self>::new();
        }

        let mut scratch: Vec<Self> = vec![Self::ring_zero(); input_length];
        let mut acc = Self::ring_one();
        scratch[0] = input[0];

        for i in 0..input_length {
            assert!(!input[i].is_zero(), "Cannot do batch inversion on zero");
            scratch[i] = acc;
            acc = acc * input[i];
        }

        acc = acc.inv();

        let mut res = input;
        for i in (0..input_length).rev() {
            let tmp = acc * res[i];
            res[i] = acc * scratch[i];
            acc = tmp;
        }

        res
    }
}

impl GetRandomElements for XFieldElement {
    fn random_elements(length: usize, rng: &mut ThreadRng) -> Vec<Self> {
        let b_values: Vec<BFieldElement> = BFieldElement::random_elements(length * 3, rng);

        let mut values: Vec<XFieldElement> = Vec::with_capacity(length as usize);
        for i in 0..length as usize {
            values.push(XFieldElement::new([
                b_values[3 * i],
                b_values[3 * i + 1],
                b_values[3 * i + 2],
            ]));
        }

        values
    }
}

impl CyclicGroupGenerator for XFieldElement {
    fn get_cyclic_group_elements(&self, max: Option<usize>) -> Vec<Self> {
        let mut val = *self;
        let mut ret: Vec<Self> = vec![Self::ring_one()];

        loop {
            ret.push(val);
            val = val * *self;
            if val.is_one() || max.is_some() && ret.len() >= max.unwrap() {
                break;
            }
        }
        ret
    }
}

impl Display for XFieldElement {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}x^2 + {}x + {}",
            self.coefficients[2], self.coefficients[1], self.coefficients[0]
        )
    }
}

impl From<Vec<u8>> for XFieldElement {
    fn from(bytes: Vec<u8>) -> Self {
        // TODO: See note in BFieldElement's From<Vec<u8>>.
        let bytesize = std::mem::size_of::<u64>();
        let (first_eight_bytes, rest) = bytes.as_slice().split_at(bytesize);
        let (second_eight_bytes, rest2) = rest.split_at(bytesize);
        let (third_eight_bytes, _rest3) = rest2.split_at(bytesize);

        XFieldElement::new([
            first_eight_bytes.to_vec().into(),
            second_eight_bytes.to_vec().into(),
            third_eight_bytes.to_vec().into(),
        ])
    }
}

impl PrimeFieldElement for XFieldElement {
    type Elem = XFieldElement;
}

impl IdentityValues for XFieldElement {
    fn is_zero(&self) -> bool {
        self == &Self::ring_zero()
    }

    fn is_one(&self) -> bool {
        self == &Self::ring_one()
    }

    fn ring_zero(&self) -> Self {
        Self::ring_zero()
    }

    fn ring_one(&self) -> Self {
        Self::ring_one()
    }
}

// TODO: Replace this with a From<usize> trait
// This trait is used by INTT
impl New for XFieldElement {
    fn new_from_usize(&self, value: usize) -> Self {
        Self::new_const(BFieldElement::new(value as u128))
    }
}

// TODO: This is the place we want to define these; other sites can die.
// impl Zero for XFieldElement {
//     fn zero() -> Self {
//         Self::ring_zero()
//     }

//     fn is_zero(&self) -> bool {
//         self == &Self::ring_zero()
//     }
// }

// impl One for XFieldElement {
//     fn one() -> Self {
//         Self::ring_one()
//     }

//     fn is_one(&self) -> bool {
//         self == &Self::ring_one()
//     }
// }

impl Add for XFieldElement {
    type Output = Self;

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

/*

(ax^2 + bx + c) * (dx^2 + ex + f)   (mod x^3 - x + 1)

=   adx^4 + aex^3 + afx^2
  + bdx^3 + bex^2 + bfx
  + cdx^2 + cex   + cf

= adx^4 + (ae + bd)x^3 + (af + be + cd)x^2 + (bf + ce)x + cf   (mod x^3 - x + 1)

= ...

*/
impl Mul for XFieldElement {
    type Output = Self;

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

impl Neg for XFieldElement {
    type Output = Self;

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

impl Sub for XFieldElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        -other + self
    }
}

impl Div for XFieldElement {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self {
        self * other.inv()
    }
}

impl ModPowU64 for XFieldElement {
    fn mod_pow_u64(&self, exponent: u64) -> Self {
        // Special case for handling 0^0 = 1
        if exponent == 0 {
            return Self::ring_one();
        }

        let mut x = *self;
        let mut result = Self::ring_one();
        let mut i = exponent;

        while i > 0 {
            if i % 2 == 1 {
                result = result * x;
            }

            x = x * x;
            i >>= 1;
        }

        result
    }
}

impl ModPowU32 for XFieldElement {
    fn mod_pow_u32(&self, exp: u32) -> Self {
        // TODO: This can be sped up by a factor 2 by implementing
        // it for u32 and not using the 64-bit version
        self.mod_pow_u64(exp as u64)
    }
}

#[cfg(test)]
mod x_field_element_test {
    use itertools::{izip, Itertools};

    use crate::shared_math::{b_field_element::*, ntt, x_field_element::*};
    // use proptest::prelude::*;

    #[test]
    fn one_zero_test() {
        let one = XFieldElement::ring_one();
        assert!(one.is_one());
        assert!(one.coefficients[0].is_one());
        assert!(one.coefficients[1].is_zero());
        assert!(one.coefficients[2].is_zero());
        let zero = XFieldElement::ring_zero();
        assert!(zero.is_zero());
        assert!(zero.coefficients[0].is_zero());
        assert!(zero.coefficients[1].is_zero());
        assert!(zero.coefficients[2].is_zero());
        let two = XFieldElement::new([
            BFieldElement::new(2),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
        ]);
        assert!(!two.is_one());
        assert!(!zero.is_one());
        let one_as_constant_term_0 = XFieldElement::new([
            BFieldElement::new(1),
            BFieldElement::ring_one(),
            BFieldElement::ring_zero(),
        ]);
        let one_as_constant_term_1 = XFieldElement::new([
            BFieldElement::new(1),
            BFieldElement::ring_zero(),
            BFieldElement::ring_one(),
        ]);
        assert!(!one_as_constant_term_0.is_one());
        assert!(!one_as_constant_term_1.is_one());
        assert!(!one_as_constant_term_0.is_zero());
        assert!(!one_as_constant_term_1.is_zero());
    }

    #[test]
    fn x_field_random_element_generation_test() {
        let mut rng = rand::thread_rng();
        let rand_xs = XFieldElement::random_elements(14, &mut rng);

        // Assert correct length
        assert_eq!(14, rand_xs.len());

        // TODO: Consider doing a statistical test.
        // Assert (probable) uniqueness of all generated elements
        assert_eq!(
            rand_xs.len(),
            rand_xs
                .clone()
                .into_iter()
                .unique()
                .collect::<Vec<XFieldElement>>()
                .len()
        );
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
        let mut val = XFieldElement::ring_zero();
        val.incr(0);
        assert!(val.is_one());
        val.incr(0);
        assert_eq!(two_const, val);
        val.decr(0);
        assert!(val.is_one());
        val.decr(0);
        assert!(val.is_zero());
        val.decr(0);
        assert_eq!(max_const, val);
        val.decr(0);
        assert_eq!(max_const - XFieldElement::ring_one(), val);
        val.decr(0);
        assert_eq!(
            max_const - XFieldElement::ring_one() - XFieldElement::ring_one(),
            val
        );
        val.incr(0);
        val.incr(0);
        val.incr(0);
        assert!(val.is_zero());
        val.incr(1);
        assert_eq!(one_x, val);
        val.incr(1);
        assert_eq!(two_x, val);
        val.decr(1);
        val.decr(1);
        assert!(val.is_zero());
        val.decr(1);
        assert_eq!(max_x, val);
        val.incr(1);
        val.incr(2);
        assert_eq!(one_x_squared, val);
        val.incr(2);
        assert_eq!(two_x_squared, val);
        val.decr(2);
        val.decr(2);
        assert!(val.is_zero());
        val.decr(2);
        assert_eq!(max_x_squared, val);
        val.decr(1);
        val.decr(0);
        assert_eq!(max_x_squared + max_x + max_const, val);
        val.decr(2);
        val.decr(1);
        val.decr(0);
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
    fn x_field_into_test() {
        let zero_poly: XFieldElement = Polynomial::<BFieldElement>::new(vec![]).into();
        assert!(zero_poly.is_zero());

        let shah_zero: XFieldElement = XFieldElement::shah_polynomial().into();
        assert!(shah_zero.is_zero());

        let neg_shah_zero: XFieldElement = XFieldElement::shah_polynomial()
            .scalar_mul(BFieldElement::new(BFieldElement::QUOTIENT - 1))
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
            let (gcd_0, a_0, b_0) = XFieldElement::xgcd(
                x_field_element_poly.clone(),
                XFieldElement::shah_polynomial(),
            );
            let (gcd_1, b_1, a_1) = XFieldElement::xgcd(
                XFieldElement::shah_polynomial(),
                x_field_element.clone().into(),
            );

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
        let one_inv = one.inv();
        assert!((one_inv * one).is_one());
        assert!((one * one_inv).is_one());

        let two = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let two_inv = two.inv();
        assert!((two_inv * two).is_one());
        assert!((two * two_inv).is_one());

        let three = XFieldElement::new([3, 0, 0].map(BFieldElement::new));
        let three_inv = three.inv();
        assert!((three_inv * three).is_one());
        assert!((three * three_inv).is_one());

        let hundred = XFieldElement::new([100, 0, 0].map(BFieldElement::new));
        let hundred_inv = hundred.inv();
        assert!((hundred_inv * hundred).is_one());
        assert!((hundred * hundred_inv).is_one());

        let x = XFieldElement::new([0, 1, 0].map(BFieldElement::new));
        let x_inv = x.inv();
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
        let mut rng = rand::thread_rng();
        let rands = XFieldElement::random_elements(test_iterations, &mut rng);
        for mut rand in rands.clone() {
            let rand_inv_original = rand.inv();
            assert!((rand * rand_inv_original).is_one());
            assert!((rand_inv_original * rand).is_one());

            // Negative test, verify that when decrementing and incrementing
            // by one in the different indices, we get something
            rand.incr(0);
            assert!((rand * rand.inv()).is_one());
            assert!((rand.inv() * rand).is_one());
            assert!(!(rand * rand_inv_original).is_one());
            assert!(!(rand_inv_original * rand).is_one());
            rand.decr(0);

            rand.incr(1);
            assert!((rand * rand.inv()).is_one());
            assert!((rand.inv() * rand).is_one());
            assert!(!(rand * rand_inv_original).is_one());
            assert!(!(rand_inv_original * rand).is_one());
            rand.decr(1);

            rand.incr(2);
            assert!((rand * rand.inv()).is_one());
            assert!((rand.inv() * rand).is_one());
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
    fn x_field_division_mul_pbt() {
        let test_iterations = 100;
        let mut rng = rand::thread_rng();
        let rands_a = XFieldElement::random_elements(test_iterations, &mut rng);
        let rands_b = XFieldElement::random_elements(test_iterations, &mut rng);
        for (a, b) in izip!(rands_a, rands_b) {
            let ab = a * b;
            let ba = b * a;
            assert_eq!(ab, ba);
            assert_eq!(ab / b, a);
            assert_eq!(ab / a, b);
        }
    }

    #[test]
    fn x_field_mod_pow_test() {
        let const_poly = XFieldElement::new([3, 0, 0].map(BFieldElement::new));

        let expecteds = [1u64, 3, 9, 27, 81, 243].iter().map(|&x| {
            XFieldElement::new([
                BFieldElement::new(x as u128),
                BFieldElement::ring_zero(),
                BFieldElement::ring_zero(),
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
        for i in [2, 4, 8, 16, 32] {
            // Verify that NTT and INTT are each other's inverses,
            // and that NTT corresponds to polynomial evaluation
            let inputs_u128: Vec<u128> = (0..i).collect();
            let inputs: Vec<XFieldElement> = inputs_u128
                .iter()
                .map(|&x| XFieldElement::new_const(BFieldElement::new(x)))
                .collect();
            let root = XFieldElement::ring_zero()
                .get_primitive_root_of_unity(i)
                .0
                .unwrap();
            let outputs = ntt::ntt(&inputs, &root);
            let inverted_outputs = ntt::intt(&outputs, &root);
            assert_eq!(inputs, inverted_outputs);

            // The output should be equivalent to evaluating root^i, i = [0..4]
            // over the polynomium with coefficients 1, 2, 3, 4
            let pol_degree_i_minus_1: Polynomial<XFieldElement> = Polynomial::new(inputs.to_vec());
            let x_domain = root.get_cyclic_group_elements(None);
            for i in 0..outputs.len() {
                assert_eq!(pol_degree_i_minus_1.evaluate(&x_domain[i]), outputs[i]);
            }

            // Verify that polynomial interpolation produces the same polynomial
            // Slow Lagrange interpolation is very slow for big inputs. Do not increase
            // this above 32 elements!
            let interpolated = Polynomial::slow_lagrange_interpolation_new(&x_domain, &outputs);
            assert_eq!(pol_degree_i_minus_1, interpolated);
        }
    }
}
