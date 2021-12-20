use num_traits::Num;
use num_traits::One;
use num_traits::Zero;
use std::convert::From;
use std::fmt::{Debug, Display};
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Rem;
use std::ops::Sub;

use super::prime_field_element::PrimeFieldElement;

// A simple implementation of fractions, with integers, not with finite fields
#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Fraction<T>
where
    T: num_traits::Num + Clone,
{
    dividend: T,
    divisor: T,
}

impl<T: num_traits::Num + Clone + Copy + Display> From<T> for Fraction<T> {
    fn from(item: T) -> Self {
        Self {
            dividend: item,
            divisor: num_traits::one(),
        }
    }
}

impl<T: num_traits::Num + Clone + Copy + Display> std::fmt::Display for Fraction<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} / {})", self.dividend, self.divisor)
    }
}

impl<U: num_traits::Num + Clone + Copy + Debug> Num for Fraction<U> {
    type FromStrRadixErr = &'static str;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let dividend = U::from_str_radix(str, radix);
        match dividend {
            Ok(dividend) => Ok(Self {
                dividend,
                divisor: num_traits::one(),
            }),
            _ => Err("Failed"),
        }
    }
}

impl<T: num_traits::Num + Clone + Copy + Debug> Fraction<T> {
    pub fn reduce(mut dividend: T, mut divisor: T) -> Self {
        let (reducer, ..) = PrimeFieldElement::xgcd(dividend, divisor);
        if reducer != num_traits::one() {
            dividend = dividend / reducer;
            divisor = divisor / reducer;
        }

        // TODO: Minus sign is currently on divisor, would be
        // better to have it on the dividend imo.
        Self { dividend, divisor }
    }

    pub fn new(dividend: T, divisor: T) -> Self {
        Self::reduce(dividend, divisor)
    }

    pub fn get_dividend(&self) -> T {
        self.dividend
    }

    pub fn get_divisor(&self) -> T {
        self.divisor
    }

    fn zero() -> Self {
        Self {
            dividend: num_traits::zero(),
            divisor: num_traits::one(),
        }
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    fn one() -> Self {
        Self {
            dividend: num_traits::one(),
            divisor: num_traits::one(),
        }
    }

    pub fn scalar_mul(&self, scalar: T) -> Self {
        Self::reduce(scalar * self.dividend, self.divisor)
    }

    pub fn scalar_div(&self, scalar: T) -> Self {
        Self::reduce(self.dividend, scalar * self.divisor)
    }
}

impl<U: num_traits::Num + Clone + Copy + Debug> One for Fraction<U> {
    fn one() -> Self {
        Fraction::one()
    }
}

impl<U: num_traits::Num + Clone + Copy + Debug> Zero for Fraction<U> {
    fn zero() -> Self {
        Fraction::zero()
    }

    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

impl<U: num_traits::Num + Clone + Copy + Debug> Div for Fraction<U> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self::reduce(self.dividend * other.divisor, self.divisor * other.dividend)
    }
}

impl<U: num_traits::Num + Clone + Copy + Debug> Add for Fraction<U> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let common_divisor = self.divisor * other.divisor;
        let dividend = self.dividend * other.divisor + other.dividend * self.divisor;
        Self::reduce(dividend, common_divisor)
    }
}

impl<U: num_traits::Num + Clone + Copy + Debug> Sub for Fraction<U> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let common_divisor = self.divisor * other.divisor;
        let dividend = self.dividend * other.divisor - other.dividend * self.divisor;
        Self::reduce(dividend, common_divisor)
    }
}

impl<U: num_traits::Num + Clone + Copy + Debug> Mul for Fraction<U> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::reduce(self.dividend * other.dividend, self.divisor * other.divisor)
    }
}

impl<U: num_traits::Num + Clone + Copy> Rem for Fraction<U> {
    type Output = Self;

    fn rem(self, _: Self) -> Self {
        Self {
            dividend: num_traits::zero(),
            divisor: num_traits::one(),
        }
    }
}

#[cfg(test)]
mod test_fractions {

    #[test]
    fn internal() {
        use super::*;
        let zero = Fraction::zero();
        assert_eq!(zero, Fraction::new(0, 1));
        let one = Fraction::one();
        assert_eq!(one, Fraction::new(1, 1));

        let seven = Fraction::new(7, 1);
        let two = Fraction::new(8, 4);
        assert_eq!(seven + two, Fraction::new(9, 1));
        assert_eq!(seven - two, Fraction::new(5, 1));
        assert_eq!(seven / two, Fraction::new(7, 2));
        assert_eq!(seven * two, Fraction::new(14, 1));
        assert_eq!(seven + zero, Fraction::new(7, 1));
        assert_eq!(seven + one, Fraction::new(8, 1));
        assert_eq!(two + zero, Fraction::new(2, 1));
        assert_eq!(two + one, Fraction::new(3, 1));
        assert_eq!(zero + seven, Fraction::new(7, 1));
        assert_eq!(one + seven, Fraction::new(8, 1));
        assert_eq!(zero + two, Fraction::new(2, 1));
        assert_eq!(one + two, Fraction::new(3, 1));

        let seven_ninths = Fraction::new(7, 9);
        let forty_four_fifths = Fraction::new(44, 5);
        assert_eq!(seven_ninths + forty_four_fifths, Fraction::new(431, 45));
        assert_eq!(seven_ninths - forty_four_fifths, Fraction::new(-361, 45));
        assert_eq!(seven_ninths / forty_four_fifths, Fraction::new(35, 396));
        assert_eq!(seven_ninths * forty_four_fifths, Fraction::new(308, 45));
        assert_eq!(seven_ninths + zero, Fraction::new(7, 9));
        assert_eq!(seven_ninths + one, Fraction::new(16, 9));
        assert_eq!(forty_four_fifths + zero, Fraction::new(44, 5));
        assert_eq!(forty_four_fifths + one, Fraction::new(49, 5));
        assert_eq!(seven_ninths - zero, Fraction::new(7, 9));
        assert_eq!(seven_ninths - one, Fraction::new(-2, 9));
        assert_eq!(forty_four_fifths - zero, Fraction::new(44, 5));
        assert_eq!(forty_four_fifths - one, Fraction::new(39, 5));
        assert_eq!(forty_four_fifths.scalar_mul(5), Fraction::new(44, 1));
        assert_eq!(forty_four_fifths.scalar_mul(2), Fraction::new(88, 5));
        assert_eq!(forty_four_fifths.scalar_div(5), Fraction::new(44, 25));
        assert_eq!(forty_four_fifths.scalar_div(2), Fraction::new(22, 5));
        assert_eq!(forty_four_fifths.scalar_mul(-5), Fraction::new(-44, 1));
        assert_eq!(forty_four_fifths.scalar_mul(-2), Fraction::new(-88, 5));
        assert_eq!(forty_four_fifths.scalar_div(-5), Fraction::new(-44, 25));
        assert_eq!(forty_four_fifths.scalar_div(-2), Fraction::new(-22, 5));

        let ten_fifths = Fraction::new(10, 5);
        assert_eq!(ten_fifths, Fraction::new(2, 1));

        let minus_one = Fraction::new(-1, 1);
        let minus_two = Fraction::new(-2, 1);
        assert_eq!(minus_one + minus_two, Fraction::new(-3, 1));
        assert_eq!(minus_one - minus_two, Fraction::new(1, 1));
        assert_eq!(minus_one / minus_two, Fraction::new(1, 2));
        assert_eq!(minus_one * minus_two, Fraction::new(2, 1));
        assert_eq!(minus_one.scalar_mul(5), Fraction::new(-5, 1));
        assert_eq!(minus_one.scalar_mul(2), Fraction::new(-2, 1));
        assert_eq!(minus_one.scalar_div(5), Fraction::new(-1, 5));
        assert_eq!(minus_one.scalar_div(2), Fraction::new(-1, 2));
        assert_eq!(minus_one.scalar_mul(-5), Fraction::new(5, 1));
        assert_eq!(minus_one.scalar_mul(-2), Fraction::new(2, 1));
        assert_eq!(minus_one.scalar_div(-5), Fraction::new(1, 5));
        assert_eq!(minus_one.scalar_div(-2), Fraction::new(1, 2));

        // Ensure that negative sign is always consistent and unique
        assert_eq!(Fraction::new(1, -2), Fraction::new(-1, 2));
    }
}
