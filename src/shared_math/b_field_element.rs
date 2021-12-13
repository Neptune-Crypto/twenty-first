use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use super::traits::{IdentityValues, ModPowU64};

// use super::traits::IdentityValues;

// BFieldElement ∈ ℤ_{2^64 - 2^32 + 1}
#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, Serialize, Deserialize)]
pub struct BFieldElement(u128);

impl BFieldElement {
    pub const QUOTIENT: u128 = 0xffff_ffff_0000_0001u128; // 2^64 - 2^32 + 1
    pub const MAX: u128 = Self::QUOTIENT - 1;

    pub fn new(value: u128) -> Self {
        Self {
            0: value % Self::QUOTIENT,
        }
    }

    pub fn inv(&self) -> Self {
        let (_, _, a) = Self::eea(Self::QUOTIENT, self.0);

        Self {
            0: (a % Self::QUOTIENT + Self::QUOTIENT) % Self::QUOTIENT,
        }
    }

    // TODO: Name this collection of traits as something like... FieldElementInternalNumRepresentation
    pub fn eea<T: Zero + One + Rem<Output = T> + Div<Output = T> + Sub<Output = T> + Clone>(
        mut x: T,
        mut y: T,
    ) -> (T, T, T) {
        let (mut a_factor, mut a1, mut b_factor, mut b1) =
            (T::one(), T::zero(), T::zero(), T::one());

        while !y.is_zero() {
            let (quotient, remainder) = (x.clone() / y.clone(), x.clone() % y.clone());
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

        // x is the gcd
        (x, a_factor, b_factor)
    }

    // TODO: Use Rust Pow. TODO: Maybe move this out into a library along with eea().
    // TODO: Name this collection of traits as something like... FieldElementInternalNumRepresentation
    pub fn mod_pow_raw(&self, pow: u64) -> u128 {
        // Special case for handling 0^0 = 1
        if pow == 0 {
            return 1u128;
        }

        let mut acc: u128 = 1;
        let mod_value: u128 = Self::QUOTIENT;
        let res = self.0;

        for i in 0..128 {
            acc = acc * acc % mod_value;
            let set: bool = pow & (1 << (128 - 1 - i)) != 0;
            if set {
                acc = acc * res % mod_value;
            }
        }

        acc
    }

    pub fn mod_pow(&self, pow: u64) -> Self {
        Self {
            0: self.mod_pow_raw(pow),
        }
    }
}

impl fmt::Display for BFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

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
    fn is_zero(&self) -> bool {
        self.0 == 0
    }

    fn is_one(&self) -> bool {
        self.0 == 1
    }

    fn ring_zero(&self) -> Self {
        BFieldElement(0)
    }

    fn ring_one(&self) -> Self {
        BFieldElement(1)
    }
}

impl Add for BFieldElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            0: (self.0 + other.0) % Self::QUOTIENT,
        }
    }
}

impl Mul for BFieldElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            0: (self.0 * other.0) % Self::QUOTIENT,
        }
    }
}

impl Neg for BFieldElement {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            0: Self::QUOTIENT - self.0,
        }
    }
}

impl Rem for BFieldElement {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Self {
            0: self.0 % other.0,
        }
    }
}

impl Sub for BFieldElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        -other + self
    }
}

impl Div for BFieldElement {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            0: other.inv().0 * self.0 % Self::QUOTIENT,
        }
    }
}

// TODO: We probably wanna make use of Rust's Pow, but for now we copy from ...big:
impl ModPowU64 for BFieldElement {
    fn mod_pow_u64(&self, pow: u64) -> Self {
        self.mod_pow(pow)
    }
}

#[cfg(test)]
mod b_prime_field_element_test {
    use crate::shared_math::{b_field_element::*, polynomial::Polynomial};
    use proptest::prelude::*;

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

    proptest! {
        #[test]
        fn identity_tests(n in 0u128..BFieldElement::MAX) {
            let zero = BFieldElement::new(0);
            let one = BFieldElement::new(1);
            let other = BFieldElement::new(n);

            prop_assert_eq!(other, zero + other, "left zero identity");
            prop_assert_eq!(other, other + zero, "right zero identity");
            prop_assert_eq!(other, one * other, "left one identity");
            prop_assert_eq!(other, other * one, "right one identity");
        }
    }

    prop_compose! {
        fn primefield_element_ranged(min: u128, max: u128)
            (n in (min..max)
                .prop_map(BFieldElement::new))
        -> BFieldElement { n }
    }

    fn primefield_element() -> impl Strategy<Value = BFieldElement> {
        let max = BFieldElement::MAX;
        let bar = prop_oneof![
            primefield_element_ranged(0, 0xffff),
            primefield_element_ranged(max - 0xffff, max),
        ];

        return bar;
    }

    proptest! {
        // #[test]
        // fn add_within_range(a in primefield_element(), b in primefield_element()) {
        //     let sum = a + b;
        // }
    }

    #[test]
    fn create_polynomial_test() {
        // use crate::shared_math_
        let a: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::new(1),
                BFieldElement::new(3),
                BFieldElement::new(7),
            ],
        };

        let b: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::new(2),
                BFieldElement::new(5),
                BFieldElement::new(BFieldElement::MAX),
            ],
        };

        let expected: Polynomial<BFieldElement> = Polynomial {
            coefficients: vec![
                BFieldElement::new(3),
                BFieldElement::new(8),
                BFieldElement::new(6),
            ],
        };

        assert_eq!(expected, a + b);
    }
}
