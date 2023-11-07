use num_bigint::BigUint;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Debug, Serialize, Deserialize)]
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
        Some(self.cmp(other))
    }
}

impl Ord for DyadicRational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.exponent < other.exponent {
            return other.cmp(self).reverse();
        }

        // We now know that: self.exponent >= other.exponent

        // Put on same denominator
        let max_exponent = self.exponent;
        let min_exponent = other.exponent;
        let shift = max_exponent - min_exponent;

        // Add numerators
        let shifted_other_mantissa = other.mantissa.clone() << shift;

        self.mantissa.cmp(&shifted_other_mantissa)
    }
}

impl Sub for DyadicRational {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // panic if rhs is larger
        assert!(
            self >= rhs,
            "Right-hand-side argument cannot be larger than left-hand-side for subtraction.\nLHS was: {self:?},\nRHS: {rhs:?}"
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
    use rand::{Rng, RngCore};

    use super::*;

    #[test]
    fn canonize_simple_test() {
        let mut a = DyadicRational {
            exponent: 4,
            mantissa: 128u32.into(),
        };
        let expected_a = DyadicRational {
            exponent: 0,
            mantissa: 8u32.into(),
        };

        // Verify that mantissa and exponent are manipulated correctly
        let canonized_a = a.canonical_representation();
        assert_eq!(expected_a.exponent, canonized_a.exponent);
        assert_eq!(expected_a.mantissa, canonized_a.mantissa);

        // Verify that equality operator behaves as expected
        assert_eq!(expected_a, a);

        a.canonize();
        assert_eq!(expected_a.exponent, a.exponent);
        assert_eq!(expected_a.mantissa, a.mantissa);

        let mut b = DyadicRational {
            exponent: 4,
            mantissa: 36u32.into(),
        };
        let expected_b = DyadicRational {
            exponent: 2,
            mantissa: 9u32.into(),
        };

        // Verify that mantissa and exponent are manipulated correctly
        let canonized_b = b.canonical_representation();
        assert_eq!(expected_b.exponent, canonized_b.exponent);
        assert_eq!(expected_b.mantissa, canonized_b.mantissa);

        // Verify that equality operator behaves as expected
        assert_eq!(expected_b, b);

        b.canonize();
        assert_eq!(expected_b.exponent, b.exponent);
        assert_eq!(expected_b.mantissa, b.mantissa);
    }

    #[test]
    fn canonize_pbt() {
        let count: usize = 100;
        let vals: Vec<DyadicRational> = get_rands(2 * count);
        for val in vals {
            assert_eq!(val, val.canonical_representation());
            let mut val_copy = val.clone();
            val_copy.canonize();
            assert_eq!(val, val_copy);
        }
    }

    #[test]
    fn scalar_mul_simple_test() {
        let mut a = DyadicRational {
            exponent: 1,
            mantissa: 7u32.into(),
        };
        let expected_double = DyadicRational {
            exponent: 0,
            mantissa: 7u32.into(),
        };
        let expected_quad = DyadicRational {
            exponent: 0,
            mantissa: 14u32.into(),
        };
        let expected_times_12 = DyadicRational {
            exponent: 0,
            mantissa: 42u32.into(),
        };
        let expected_triple = DyadicRational {
            exponent: 1,
            mantissa: 21u32.into(),
        };

        a.scalar_mul(2);
        assert_eq!(expected_double, a);

        a.scalar_mul(2);
        assert_eq!(expected_quad, a);

        a.scalar_mul(3);
        assert_eq!(expected_times_12, a);

        // Also verify behavior of `divide_by_power_of_two`
        a.divide_by_power_of_two(2);
        assert_eq!(expected_triple, a);
    }

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
    fn ordering_unit_test() {
        let a: DyadicRational = 16.into();
        let b = DyadicRational {
            exponent: 2,
            mantissa: 64u32.into(),
        };

        assert!(!(a > b));
        assert!(!(b > a));
        assert!(!(a < b));
        assert!(!(b < a));
        assert!(a <= b);
        assert!(a >= b);

        let c = DyadicRational {
            exponent: 3,
            mantissa: 127u32.into(),
        };

        assert!(b > c);
        assert!(c < b);
        assert!(a > c);
        assert!(c < a);
    }

    #[test]
    fn mul_simple_test() {
        let a = DyadicRational {
            exponent: 4,
            mantissa: 17u32.into(),
        };
        let b = DyadicRational {
            exponent: 2,
            mantissa: 2u32.into(),
        };

        let expected_prod = DyadicRational {
            exponent: 6,
            mantissa: 34u32.into(),
        };
        let mut calculated = a * b;
        assert_eq!(expected_prod, calculated);
        calculated.canonize();
        assert_eq!(5, calculated.exponent);
        assert_eq!(BigUint::from(17u32), calculated.mantissa);

        // Divide by n and verify result
        calculated.divide_by_power_of_two(5);
        assert_eq!(10, calculated.exponent);
        assert_eq!(BigUint::from(17u32), calculated.mantissa);
    }

    #[test]
    fn mul_div_pow_two_pbt() {
        let count: usize = 100;
        let vals: Vec<DyadicRational> = get_rands(2 * count);
        let two = DyadicRational {
            exponent: 0,
            mantissa: BigUint::from(2u32),
        };
        let three = DyadicRational {
            exponent: 0,
            mantissa: BigUint::from(3u32),
        };
        let seventeen = DyadicRational {
            exponent: 0,
            mantissa: BigUint::from(17u32),
        };
        let one_sixteenth = DyadicRational {
            exponent: 4,
            mantissa: BigUint::from(1u32),
        };
        for val in vals {
            let mut val_local = val.clone();
            let double_by_mul = val.clone() * two.clone();
            val_local.scalar_mul(2);
            assert_eq!(val_local, double_by_mul);

            val_local = val.clone();
            val_local.scalar_mul(3);
            let triple_by_mul = val.clone() * three.clone();
            assert_eq!(val_local, triple_by_mul);

            val_local = val.clone();
            val_local.scalar_mul(17);
            let septagint_by_mul = val.clone() * seventeen.clone();
            assert_eq!(val_local, septagint_by_mul);

            // Divide by 16
            val_local = val.clone();
            val_local.divide_by_power_of_two(4);
            let one_sixteenth_by_mul = val.clone() * one_sixteenth.clone();
            assert_eq!(val_local, one_sixteenth_by_mul);
        }
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
        let mut rng = rand::thread_rng();
        let mut ret = Vec::with_capacity(length);

        for _ in 0..length {
            let mantissa: BigUint = rng.next_u64().into();

            // Restrict exponent to a value between 0 and 255
            let exponent: u32 = rng.gen_range(0..256);
            let val = DyadicRational::new(mantissa, exponent);
            ret.push(val);
        }

        ret
    }
}
