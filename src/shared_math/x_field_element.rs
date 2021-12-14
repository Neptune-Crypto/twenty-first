use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::IdentityValues;
use serde::{Deserialize, Serialize};

use std::{
    fmt::Display,
    ops::{Add, Mul, Neg, Rem, Sub},
};

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, Serialize, Deserialize)]
pub struct XFieldElement {
    pub coefficients: [BFieldElement; 3],
}

impl XFieldElement {
    pub fn new(coefficients: [BFieldElement; 3]) -> Self {
        Self { coefficients }
    }

    // TODO: inv
    // TODO: xgcd
    // TODO: mod_pow, mod_pow_raw
    // TODO: get_primitive_root_of_unity (TBD), primes_lt, legendre_symbol,

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

    // Division in 𝔽_p[X], not 𝔽_p^e ≅ 𝔽[X]/p(x).
    // TODO (Maybe): Could be that this isn't as efficient as possible.
    fn divide_as_polynomial(&self, other: Self) -> (Self, Self) {
        let lhs = Polynomial {
            coefficients: self.coefficients.to_vec(),
        };
        let rhs = Polynomial {
            coefficients: other.coefficients.to_vec(),
        };
        let (quot_as_vec, rem_as_vec) = lhs.divide(rhs);
        let mut quot_coefficients = [BFieldElement::ring_zero(); 3];
        let mut rem_coefficients = [BFieldElement::ring_zero(); 3];
        for i in 0..quot_as_vec.degree() + 1 {
            quot_coefficients[i as usize] = quot_as_vec.coefficients[i as usize];
        }
        for i in 0..rem_as_vec.degree() + 1 {
            rem_coefficients[i as usize] = rem_as_vec.coefficients[i as usize];
        }

        (
            Self {
                coefficients: quot_coefficients,
            },
            Self {
                coefficients: rem_coefficients,
            },
        )
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

// impl Rem for XFieldElement {
//     type Output = Self;

//     fn rem(self, other: Self) -> Self {
//         // XXX
//     }
// }

// impl Div for XFieldElement {
//     type Output = Self;

//     fn add(self, other: Self) -> Self {
//         Self {
//             coefficients: [
//                 self.coefficients[0] + other.coefficients[0],
//                 self.coefficients[1] + other.coefficients[1],
//                 self.coefficients[2] + other.coefficients[2],
//             ],
//         }
//     }
// }

// TODO: ModPow64

#[cfg(test)]
mod x_field_element_test {
    use crate::shared_math::{b_field_element::*, x_field_element::*};
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
    fn divide_as_polynomial_test() {
        // (0x^2 + 0x + 2) / (0x^2 + 0x + 2) = 1, remainder = 0
        let mut poly1 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let mut poly2 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let (quotient_result, remainder_result) = poly1.divide_as_polynomial(poly2);
        let mut quotient_expected = XFieldElement::new([1, 0, 0].map(BFieldElement::new));
        let mut remainder_expected = XFieldElement::new([0, 0, 0].map(BFieldElement::new));
        assert_eq!(quotient_expected, quotient_result);
        assert_eq!(remainder_expected, remainder_result);

        // (0x^2 + 2x + 0) / (0x^2 + 0x + 2) = x, remainder = 0
        poly1 = XFieldElement::new([0, 2, 0].map(BFieldElement::new));
        poly2 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let (quotient_result, remainder_result) = poly1.divide_as_polynomial(poly2);
        quotient_expected = XFieldElement::new([0, 1, 0].map(BFieldElement::new));
        remainder_expected = XFieldElement::new([0, 0, 0].map(BFieldElement::new));
        assert_eq!(quotient_expected, quotient_result);
        assert_eq!(remainder_expected, remainder_result);

        // (2x^2 + 0x + 0) / (0x^2 + 0x + 2) = x^2, remainder = 0
        poly1 = XFieldElement::new([0, 0, 2].map(BFieldElement::new));
        poly2 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let (quotient_result, remainder_result) = poly1.divide_as_polynomial(poly2);
        quotient_expected = XFieldElement::new([0, 0, 1].map(BFieldElement::new));
        remainder_expected = XFieldElement::new([0, 0, 0].map(BFieldElement::new));
        assert_eq!(quotient_expected, quotient_result);
        assert_eq!(remainder_expected, remainder_result);

        // (2x^2 + 17x + 299) / (0x^2 + 0x + 2) = x^2 + ??x + ???, remainder = 0
        poly1 = XFieldElement::new([2, 17, 299].map(BFieldElement::new));
        poly2 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let (quotient_result, remainder_result) = poly1.divide_as_polynomial(poly2);
        quotient_expected = XFieldElement::new(
            [1, 9223372034707292169u128, 9223372034707292310u128].map(BFieldElement::new),
        );
        remainder_expected = XFieldElement::new([0, 0, 0].map(BFieldElement::new));
        assert_eq!(quotient_expected, quotient_result);
        assert_eq!(remainder_expected, remainder_result);
    }
}
