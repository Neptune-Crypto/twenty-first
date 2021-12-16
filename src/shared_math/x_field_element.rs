use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::{IdentityValues, ModPowU64};
use serde::{Deserialize, Serialize};

use std::ops::Div;
use std::{
    fmt::Display,
    ops::{Add, Mul, Neg, Sub},
};

use super::traits::New;

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

        // XXX
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
        let lc = x.leading_coefficient(BFieldElement::ring_zero());
        let scale = lc.inv();
        (
            x.scalar_mul(scale),
            a_factor.scalar_mul(scale),
            b_factor.scalar_mul(scale),
        )
    }

    pub fn get_cyclic_group(&self) -> Vec<XFieldElement> {
        let mut val = *self;
        let mut ret: Vec<XFieldElement> = vec![XFieldElement::ring_one()];

        loop {
            ret.push(val);
            val = val * *self;
            if val.is_one() {
                break;
            }
        }
        ret
    }

    pub fn inv(&self) -> Self {
        let self_as_poly: Polynomial<BFieldElement> = self.to_owned().into();
        let (_, a, _) = Self::xgcd(self_as_poly, Self::shah_polynomial());
        a.into()
    }

    pub fn get_primitive_root_of_unity(n: u128) -> (Option<XFieldElement>, Vec<u128>) {
        let (b_root, primes) = BFieldElement::get_primitive_root_of_unity(n);
        let x_root = b_root.map(XFieldElement::new_const);

        (x_root, primes)
    }

    // TODO: legendre_symbol
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
        // ax^2 + bx + c
        let a = self.coefficients[2];
        let b = self.coefficients[1];
        let c = self.coefficients[0];

        // dx^2 + ex + f
        let d = other.coefficients[2];
        let e = other.coefficients[1];
        let f = other.coefficients[0];

        // (ax^2 + bx + c) * (dx^2 + ex + f)
        Self {
            coefficients: [
                c * f - a * e - b * d,                 // * x^0
                b * f + c * e - a * d + a * e + b * d, // * x^1
                a * f + b * e + c * d + a * d,         // * x^2
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

// TODO: ModPow64

#[cfg(test)]
mod x_field_element_test {
    use itertools::izip;

    use crate::shared_math::{b_field_element::*, ntt, x_field_element::*};
    // use proptest::prelude::*;

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
            let root = XFieldElement::get_primitive_root_of_unity(i).0.unwrap();
            let outputs = ntt::ntt(&inputs, &root);
            let inverted_outputs = ntt::intt(&outputs, &root);
            assert_eq!(inputs, inverted_outputs);

            // The output should be equivalent to evaluating root^i, i = [0..4]
            // over the polynomium with coefficients 1, 2, 3, 4
            let pol_degree_i_minus_1: Polynomial<XFieldElement> = Polynomial::new(inputs.to_vec());
            let x_domain = root.get_cyclic_group();
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
