use num_traits::Num;
use num_traits::One;
use num_traits::Zero;
use std::fmt::Display;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Rem;
use std::ops::Sub;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ComplexNumber<T: num_traits::Num + Clone> {
    r: T,
    i: T,
}

impl<T: num_traits::Num + Clone + Copy> ComplexNumber<T> {
    pub fn new(real: T, imaginary: T) -> Self {
        ComplexNumber {
            r: real,
            i: imaginary,
        }
    }

    pub fn get_real(&self) -> T {
        self.r
    }

    pub fn get_imaginary(&self) -> T {
        self.i
    }
}

// I wouldn't know how to implement this for integers (need to take sine and cosine),
// so we just implement it for floats.
impl<T: num_traits::Float + Clone + Copy> ComplexNumber<T> {
    pub fn from_exponential(imaginary: T) -> ComplexNumber<T> {
        ComplexNumber {
            r: imaginary.cos(),
            i: imaginary.sin(),
        }
    }
}

impl<T: num_traits::Num + Clone + Copy + Display> std::fmt::Display for ComplexNumber<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + i{}", self.r, self.i)
    }
}

impl<U: num_traits::Num + Clone + Copy> Num for ComplexNumber<U> {
    type FromStrRadixErr = &'static str;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let real = U::from_str_radix(str, radix);
        match real {
            Ok(r) => Ok(ComplexNumber {
                r,
                i: num_traits::zero(),
            }),
            _ => Err("Failed"),
        }
    }
}

impl<U: num_traits::Num + Clone + Copy> Zero for ComplexNumber<U> {
    fn zero() -> Self {
        ComplexNumber {
            r: num_traits::zero(),
            i: num_traits::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl<U: num_traits::Num + Clone + Copy> One for ComplexNumber<U> {
    fn one() -> Self {
        ComplexNumber {
            r: num_traits::one(),
            i: num_traits::zero(),
        }
    }
}

impl<U: num_traits::Num + Clone + Copy> Div for ComplexNumber<U> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let other_hyp_squared = other.r * other.r + other.i * other.i;
        Self {
            r: (self.r * other.r + self.i * other.i) / other_hyp_squared,
            i: (self.i * other.r - self.r * other.i) / other_hyp_squared,
        }
    }
}

impl<U: num_traits::Num + Clone + Copy> Add for ComplexNumber<U> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            i: self.i + other.i,
        }
    }
}

impl<U: num_traits::Num + Clone + Copy> Sub for ComplexNumber<U> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            r: self.r - other.r,
            i: self.i - other.i,
        }
    }
}

impl<U: num_traits::Num + Clone + Copy> Mul for ComplexNumber<U> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            r: self.r * other.r - self.i * other.i,
            i: self.r * other.i + self.i * other.r,
        }
    }
}

impl<U: num_traits::Num + Clone + Copy> Rem for ComplexNumber<U> {
    type Output = Self;

    // TODO: Not sure if this is correct!
    fn rem(self, _rhs: Self) -> Self::Output {
        Self::zero()
    }
}

#[cfg(test)]
mod test_complex_numbers {

    #[test]
    fn internal() {
        use super::*;
        let zero = ComplexNumber::zero();
        assert_eq!(zero, ComplexNumber::new(0, 0));
        let one = ComplexNumber::one();
        assert_eq!(one, ComplexNumber::new(1, 0));

        let mut a: ComplexNumber<i128> = ComplexNumber::new(6, 4);
        let mut b: ComplexNumber<i128> = ComplexNumber::new(8, -2);
        assert_eq!(a + b, ComplexNumber::new(14i128, 2i128));
        assert_eq!((a + b).r, 14i128);
        assert_eq!((a + b).i, 2i128);

        a = ComplexNumber::new(10, 3);
        b = ComplexNumber::new(7, -4);
        assert_eq!(a - b, ComplexNumber::new(3, 7));

        a = ComplexNumber::new(2, 3);
        b = ComplexNumber::new(1, 5);
        assert_eq!(a * b, ComplexNumber::new(-13, 13));

        a = ComplexNumber::new(2, 1);
        assert_eq!(a * a, ComplexNumber::new(3, 4));

        a = ComplexNumber::new(3, -2);
        b = ComplexNumber::new(1, -4);
        assert_eq!(a * b, ComplexNumber::new(-5, -14));

        a = ComplexNumber::new(3, 4);
        b = ComplexNumber::new(3, -4);
        assert_eq!(a * b, ComplexNumber::new(25, 0));

        a = ComplexNumber::new(3, 0);
        b = ComplexNumber::new(6, 2);
        assert_eq!(a * b, ComplexNumber::new(18, 6));

        let mut a_float = ComplexNumber::new(2.0f64, 5.0f64);
        let mut b_float = ComplexNumber::new(4.0f64, -1.0f64);
        let res = a_float / b_float;
        assert!((res.r - 3.0f64 / 17.0f64).abs() < 0.0001);
        assert!((res.i - 22.0f64 / 17.0f64).abs() < 0.0001);

        a_float = ComplexNumber::new(3.0f64, 2.0f64);
        b_float = ComplexNumber::new(4.0f64, -3.0f64);
        let res = a_float / b_float;
        assert!((res.r - 6.0f64 / 25.0f64).abs() < 0.0001);
        assert!((res.i - 17.0f64 / 25.0f64).abs() < 0.0001);

        a_float = ComplexNumber::new(2.0f64, -1.0f64);
        b_float = ComplexNumber::new(-3.0f64, 6.0f64);
        let res = a_float / b_float;
        assert!((res.r + 4.0f64 / 15.0f64).abs() < 0.0001);
        assert!((res.i + 1.0f64 / 5.0f64).abs() < 0.0001);

        a_float = ComplexNumber::new(-6.0f64, -3.0f64);
        b_float = ComplexNumber::new(4.0f64, 6.0f64);
        let res = a_float / b_float;
        assert!((res.r + 21.0f64 / 26.0f64).abs() < 0.0001);
        assert!((res.i - 6.0f64 / 13.0f64).abs() < 0.0001);
    }
}
