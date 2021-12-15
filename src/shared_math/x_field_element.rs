use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::traits::IdentityValues;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

use std::{
    fmt::Display,
    ops::{Add, Mul, Neg, Rem, Sub},
};

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
    // fn divide_as_polynomial(lhs: Polynomial<BFieldElement>, rhs: Polynomial<BFieldElement>) -> (Polynomial<BFieldElement>, Polynomial<BFieldElement>) {
    // let lhs = Polynomial {
    //     coefficients: self.coefficients.to_vec(),
    // };
    // let rhs = Polynomial {
    //     coefficients: other.coefficients.to_vec(),
    // };
    // lhs.divide(rhs)
    // let mut quot_coefficients = [BFieldElement::ring_zero(); 3];
    // let mut rem_coefficients = [BFieldElement::ring_zero(); 3];
    // for i in 0..quot_as_vec.degree() + 1 {
    //     quot_coefficients[i as usize] = quot_as_vec.coefficients[i as usize];
    // }
    // for i in 0..rem_as_vec.degree() + 1 {
    //     rem_coefficients[i as usize] = rem_as_vec.coefficients[i as usize];
    // }

    // (
    //     Self {
    //         coefficients: quot_coefficients,
    //     },
    //     Self {
    //         coefficients: rem_coefficients,
    //     },
    // )
    // }

    pub fn xgcd(
        mut x: Polynomial<BFieldElement>,
        mut y: Polynomial<BFieldElement>,
    ) -> (
        Polynomial<BFieldElement>,
        Polynomial<BFieldElement>,
        Polynomial<BFieldElement>,
    ) {
        let input_x = x.clone();
        let input_y = y.clone();
        let mut a_factor = Polynomial::new(vec![BFieldElement::ring_one()]);
        let mut a1 = Polynomial::new(vec![BFieldElement::ring_zero()]);
        let mut b_factor = Polynomial::new(vec![BFieldElement::ring_zero()]);
        let mut b1 = Polynomial::new(vec![BFieldElement::ring_one()]);

        while !y.is_zero() {
            println!("x = {},\ny = {}", x, y);
            let (quotient, remainder): (Polynomial<BFieldElement>, Polynomial<BFieldElement>) =
                x.clone().divide(y.clone());
            println!("quotient = {}", quotient);
            println!("remainder = {}", remainder);
            assert_eq!(x, quotient.clone() * y.clone() + remainder.clone()); // numerator = quotient * divisor + remainder
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

            println!("a_factor = {},\nb_factor = {}", a_factor, b_factor);
            println!();
        }

        // gcd = a_factor * x + b_factor * y
        // let gcd = tmp;
        println!("gcd = {}", x);
        assert_eq!(
            x.clone(),
            a_factor.clone() * input_x.clone() + b_factor.clone() * input_y.clone()
        );
        (x, a_factor, b_factor)
    }

    pub fn inv(&self) -> Self {
        println!("self is {}", self);
        println!("BFieldElement::QUOTIENT = {}", BFieldElement::QUOTIENT);
        // 1 = _ * shah + a * self, so 'a' is congruent to self^-1
        let self_as_poly: Polynomial<BFieldElement> = self.to_owned().into();
        let (zzz, qqq, a) = Self::xgcd(Self::shah_polynomial(), self_as_poly.clone());
        println!(
            "xgcd({}, {}) = ({}, {}, {})",
            Self::shah_polynomial(),
            self_as_poly,
            zzz,
            qqq,
            a
        );

        println!("a = {}", a);

        a.into()
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
    fn x_field_inv_test() {
        // TODO: REMOVE
        let two_inv_expected = XFieldElement::new([
            BFieldElement::new(2).inv(),
            BFieldElement::ring_zero(),
            BFieldElement::ring_zero(),
        ]);
        let two = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        assert!((two * two_inv_expected).is_one());
        assert!((two_inv_expected * two).is_one());

        let foo = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
        let foo_inv = foo.inv();
        println!("foo_inv: {}", foo_inv);
        println!("foo: {}", foo);

        let foo_one_left = foo_inv * foo;
        assert_eq!(foo_one_left, XFieldElement::ring_one());

        let foo_one_right = foo * foo_inv;
        assert_eq!(foo_one_right, XFieldElement::ring_one());
    }

    #[test]
    fn x_field_xgcd_test() {
        let a = Polynomial::new(vec![BFieldElement::new(15)]);
        let b = Polynomial::new(vec![BFieldElement::new(25)]);

        let (actual_gcd_ab, a_factor, b_factor) = XFieldElement::xgcd(a.clone(), b.clone());
        assert_eq!(actual_gcd_ab, a * a_factor.clone() + b * b_factor.clone());

        println!("({}, {}, {})", actual_gcd_ab, a_factor, b_factor);
    }

    // #[test]
    // fn divide_as_polynomial_test() {
    //     // (0x^2 + 0x + 2) / (0x^2 + 0x + 2) = 1, remainder = 0
    //     let mut poly1 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
    //     let mut poly2 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
    //     let (quotient_result, remainder_result) = poly1.divide_as_polynomial(poly2);
    //     let mut quotient_expected = XFieldElement::new([1, 0, 0].map(BFieldElement::new));
    //     let mut remainder_expected = XFieldElement::new([0, 0, 0].map(BFieldElement::new));
    //     assert_eq!(quotient_expected, quotient_result);
    //     assert_eq!(remainder_expected, remainder_result);

    //     // (0x^2 + 2x + 0) / (0x^2 + 0x + 2) = x, remainder = 0
    //     poly1 = XFieldElement::new([0, 2, 0].map(BFieldElement::new));
    //     poly2 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
    //     let (quotient_result, remainder_result) = poly1.divide_as_polynomial(poly2);
    //     quotient_expected = XFieldElement::new([0, 1, 0].map(BFieldElement::new));
    //     remainder_expected = XFieldElement::new([0, 0, 0].map(BFieldElement::new));
    //     assert_eq!(quotient_expected, quotient_result);
    //     assert_eq!(remainder_expected, remainder_result);

    //     // (2x^2 + 0x + 0) / (0x^2 + 0x + 2) = x^2, remainder = 0
    //     poly1 = XFieldElement::new([0, 0, 2].map(BFieldElement::new));
    //     poly2 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
    //     let (quotient_result, remainder_result) = poly1.divide_as_polynomial(poly2);
    //     quotient_expected = XFieldElement::new([0, 0, 1].map(BFieldElement::new));
    //     remainder_expected = XFieldElement::new([0, 0, 0].map(BFieldElement::new));
    //     assert_eq!(quotient_expected, quotient_result);
    //     assert_eq!(remainder_expected, remainder_result);

    //     // (2x^2 + 17x + 299) / (0x^2 + 0x + 2) = x^2 + ??x + ???, remainder = 0
    //     poly1 = XFieldElement::new([2, 17, 299].map(BFieldElement::new));
    //     poly2 = XFieldElement::new([2, 0, 0].map(BFieldElement::new));
    //     let (quotient_result, remainder_result) = poly1.divide_as_polynomial(poly2);
    //     quotient_expected = XFieldElement::new(
    //         [1, 9223372034707292169u128, 9223372034707292310u128].map(BFieldElement::new),
    //     );
    //     remainder_expected = XFieldElement::new([0, 0, 0].map(BFieldElement::new));
    //     assert_eq!(quotient_expected, quotient_result);
    //     assert_eq!(remainder_expected, remainder_result);
    // }

    // TODO: Write property-based test of `divide_as_polynomial`
    // #[test]
    // fn divide_as_polynomial_property_based_test() {
    //     let rands: Vec<i128> = generate_random_numbers(30, BFieldElement::MAX as i128);
    //     for rand in rands {
    //         let a =
    //     }
    // }
}
