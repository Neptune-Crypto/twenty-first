use num_traits::{One, Zero};
use std::ops::{Add, Mul, Neg, Rem, Sub};

// BPrimeFieldElement ∈ ℤ_{2^64 - 2^32 + 1}
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct BPrimeFieldElement(i128);

impl BPrimeFieldElement {
    const QUOTIENT: i128 = 0xffff_ffff_0000_0001i128; // 2^64 - 2^32 + 1
    const MAX: i128 = Self::QUOTIENT - 1;

    pub fn new(value: i128) -> Self {
        Self {
            0: value % Self::QUOTIENT,
        }
    }
}

impl Zero for BPrimeFieldElement {
    fn zero() -> Self {
        BPrimeFieldElement(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for BPrimeFieldElement {
    fn one() -> Self {
        BPrimeFieldElement(1)
    }

    fn is_one(&self) -> bool {
        self.0 == 1
    }
}

impl Add for BPrimeFieldElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            0: (self.0 + other.0) % Self::QUOTIENT,
        }
    }
}

impl Mul for BPrimeFieldElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            0: (self.0 * other.0) % Self::QUOTIENT,
        }
    }
}

impl Neg for BPrimeFieldElement {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            0: -self.0 % Self::QUOTIENT,
        }
    }
}

impl Rem for BPrimeFieldElement {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        Self {
            0: self.0 % other.0,
        }
    }
}

impl Sub for BPrimeFieldElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            0: (self.0 - other.0) % Self::QUOTIENT,
        }
    }
}

#[cfg(test)]
mod primefield_test {
    use crate::shared_math::prime_field_efficient::*;
    use proptest::prelude::*;

    #[test]
    fn test_zero_one() {
        assert!(BPrimeFieldElement::zero().is_zero());
        assert!(!BPrimeFieldElement::zero().is_one());
        assert!(!BPrimeFieldElement::one().is_zero());
        assert!(BPrimeFieldElement::one().is_one());
        assert!(BPrimeFieldElement::new(BPrimeFieldElement::MAX + 1).is_zero());
        assert!(BPrimeFieldElement::new(BPrimeFieldElement::MAX + 2).is_one());
    }

    proptest! {
        #[test]
        fn identity_tests(n in 0i128..BPrimeFieldElement::MAX) {
            let zero = BPrimeFieldElement::zero();
            let one = BPrimeFieldElement::one();
            let other = BPrimeFieldElement::new(n);

            prop_assert_eq!(other, zero + other, "left zero identity");
            prop_assert_eq!(other, other + zero, "right zero identity");
            prop_assert_eq!(other, one * other, "left one identity");
            prop_assert_eq!(other, other * one, "right one identity");
        }
    }

    prop_compose! {
        fn primefield_element_ranged(min: i128, max: i128)
            (n in (min..max)
                .prop_map(BPrimeFieldElement::new))
        -> BPrimeFieldElement { n }
    }

    fn primefield_element() -> impl Strategy<Value = BPrimeFieldElement> {
        let max = BPrimeFieldElement::MAX;
        let bar = prop_oneof![
            primefield_element_ranged(0, 0xffff),
            primefield_element_ranged(max - 0xffff, max),
        ];

        return bar;
    }

    proptest! {
        #[test]
        fn add_within_range(a in primefield_element(), b in primefield_element()) {
            let sum = a + b;
            assert!()
        }
    }
}
