use std::cmp::Ordering::{Equal, Greater, Less};
use std::ops::{Add, Mul, Sub};

use num_bigint::BigUint;
use num_traits::{One, Zero};

#[derive(Clone, Debug)]
pub struct DyadicRational {
    // TODO: Consider changing mantissa type to `u32` or to `U32s<N>`
    mantissa: BigUint,
    exponent: u32,
}

impl Zero for DyadicRational {
    fn zero() -> Self {
        Self {
            mantissa: 0u32.into(),
            exponent: 0,
        }
    }

    fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }
}

impl One for DyadicRational {
    fn one() -> Self {
        Self {
            mantissa: 1u32.into(),
            exponent: 0,
        }
    }

    fn is_one(&self) -> bool {
        *self == Self::one()
    }

    fn set_one(&mut self) {
        *self = One::one();
    }
}

impl DyadicRational {
    pub fn new(mantissa: BigUint, exponent: u32) -> Self {
        Self { mantissa, exponent }
    }

    fn canonize(&mut self) {
        let two: BigUint = BigUint::one() + BigUint::one();
        while (self.mantissa.clone() % two.clone()).is_zero() && self.exponent > 0 {
            self.mantissa /= two.clone();
            self.exponent -= 1;
        }
    }

    fn canonical_representation(&self) -> Self {
        let mut canon = self.clone();
        canon.canonize();
        canon
    }

    pub fn scalar_mul(&mut self, scalar: u32) {
        let scalar_big_uint: BigUint = scalar.into();
        self.mantissa *= scalar_big_uint;
    }

    pub fn divide_by_power_of_two(&mut self, power: u32) {
        self.exponent += power;
    }
}

impl PartialEq for DyadicRational {
    fn eq(&self, other: &Self) -> bool {
        let lhs = Self::canonical_representation(self);
        let rhs = Self::canonical_representation(other);
        lhs.mantissa == rhs.mantissa && lhs.exponent == rhs.exponent
    }
}

impl Eq for DyadicRational {}

impl Mul for DyadicRational {
    type Output = Self;
    fn mul(self, rhs: DyadicRational) -> Self::Output {
        Self::canonical_representation(&Self {
            mantissa: self.mantissa * rhs.mantissa,
            exponent: self.exponent + rhs.exponent,
        })
    }
}

impl Add for DyadicRational {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // make sure self.exponent is the larger exponent
        if self.exponent < rhs.exponent {
            return rhs.add(self);
        }
        // Put on same denominator
        let max_exponent = self.exponent;
        let min_exponent = rhs.exponent;
        let mut shift = max_exponent - min_exponent;

        // Add numerators
        let mut numerator = rhs.mantissa;
        while !shift.is_zero() {
            numerator += numerator.clone();
            shift -= 1;
        }

        numerator += self.mantissa;

        // canonize
        let mut val = Self {
            exponent: max_exponent,
            mantissa: numerator,
        };

        val.canonize();

        val
    }
}

impl PartialOrd for DyadicRational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Make sure self.exponent is the larger exponent
        if self.exponent < other.exponent {
            // Flip result when this function is called recursively with flipped
            // arguments.
            return match other.partial_cmp(self) {
                Some(Greater) => Some(Less),
                Some(Less) => Some(Greater),
                _ => Some(Equal),
            };
        }

        // Put on same denominator
        let max_exponent = self.exponent;
        let min_exponent = other.exponent;
        let mut shift = max_exponent - min_exponent;

        // Add numerators
        let mut shifted_other_mantissa = other.mantissa.clone();
        while !shift.is_zero() {
            shifted_other_mantissa += shifted_other_mantissa.clone();
            shift -= 1;
        }

        self.mantissa.partial_cmp(&shifted_other_mantissa)
    }
}

impl Ord for DyadicRational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Sub for DyadicRational {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // panic if rhs is larger
        assert!(
            self >= rhs,
            "Right-hand-side argument cannot be larger than left-hand-side for subtraction"
        );

        // Put on same denominator
        let (max_exponent, (val_max_exponent, mut val_min_exponent)) =
            if self.exponent > rhs.exponent {
                ((self.exponent), (self, rhs))
            } else {
                ((rhs.exponent), (rhs, self))
            };

        while val_min_exponent.exponent < max_exponent {
            val_min_exponent.mantissa += val_min_exponent.mantissa.clone();
            val_min_exponent.exponent += 1;
        }

        let mut ret = if val_min_exponent.mantissa < val_max_exponent.mantissa {
            Self {
                exponent: max_exponent,
                mantissa: val_max_exponent.mantissa - val_min_exponent.mantissa,
            }
        } else {
            Self {
                exponent: max_exponent,
                mantissa: val_min_exponent.mantissa - val_max_exponent.mantissa,
            }
        };

        ret.canonize();
        ret
    }
}

impl From<u32> for DyadicRational {
    fn from(integer: u32) -> Self {
        Self {
            mantissa: integer.into(),
            exponent: 0u32,
        }
    }
}

#[cfg(test)]
mod dyadic_rationals_tests {
    use rand::{thread_rng, RngCore};

    use super::*;

    #[test]
    fn simple_add_test() {
        let a: DyadicRational = 5.into();
        let b = DyadicRational {
            exponent: 3,
            mantissa: 51u32.into(),
        };

        let c = a + b;
        let expected = DyadicRational {
            exponent: 3,
            mantissa: 91u32.into(),
        };

        assert_eq!(expected, c);
    }

    #[test]
    fn simple_sub_test() {
        let a = DyadicRational {
            exponent: 4,
            mantissa: 17u32.into(),
        };
        let b = DyadicRational {
            exponent: 2,
            mantissa: 2u32.into(),
        };

        let expected = DyadicRational {
            exponent: 4,
            mantissa: 9u32.into(),
        };

        assert_eq!(expected, a - b);
    }

    #[test]
    fn add_sub_pbt() {
        let count: usize = 100;
        let vals: Vec<DyadicRational> = get_rands(2 * count);
        for i in 0..count {
            let sum = vals[2 * i + 1].clone() + vals[2 * i].clone();
            assert_eq!(vals[2 * i], sum.clone() - vals[2 * i + 1].clone());
            assert_eq!(vals[2 * i + 1], sum.clone() - vals[2 * i].clone());

            assert!(sum >= vals[2 * i]);
            assert!(sum >= vals[2 * i + 1]);
        }
    }

    #[test]
    fn equality_test() {
        let a: DyadicRational = 16.into();
        let b = DyadicRational {
            exponent: 2,
            mantissa: 64u32.into(),
        };

        assert_eq!(a, b);

        let c = DyadicRational {
            exponent: 3,
            mantissa: 127u32.into(),
        };

        assert_ne!(a, c);
    }

    #[test]
    fn additive_inverse_test() {
        let a = DyadicRational {
            exponent: 4,
            mantissa: 17u32.into(),
        };
        let b = DyadicRational {
            exponent: 2,
            mantissa: 2u32.into(),
        };

        assert_eq!(b.clone() + a.clone() - a, b);
    }

    fn get_rands(length: usize) -> Vec<DyadicRational> {
        let mut prng = thread_rng();
        let mut ret = Vec::with_capacity(length);

        for _ in 0..length {
            let mantissa: BigUint = prng.next_u64().into();

            // Restrict exponent to a value between 0 and 255
            let exponent: u32 = prng.next_u32() % 0x0100;
            let val = DyadicRational::new(mantissa, exponent);
            ret.push(val);
        }

        ret
    }
}
