use std::convert::TryFrom;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Rem, Sub};

use get_size::GetSize;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::Rng;
use rand_distr::{Distribution, Standard};
use serde_big_array;
use serde_big_array::BigArray;
use serde_derive::{Deserialize, Serialize};

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::bfield_codec::{BFieldCodec, BFieldCodecError};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, GetSize)]
pub struct U32s<const N: usize> {
    #[serde(with = "BigArray")]
    values: [u32; N],
}

impl<const N: usize> Eq for U32s<N> {}

impl<const N: usize> AsRef<[u32; N]> for U32s<N> {
    fn as_ref(&self) -> &[u32; N] {
        &self.values
    }
}

impl<const N: usize> U32s<N> {
    pub fn new(values: [u32; N]) -> Self {
        Self { values }
    }

    fn set_bit(&mut self, bit_index: usize, val: bool) {
        assert!(bit_index < 32 * N, "bit index exceeded length of U32 array");
        let u32_element_index = bit_index / 32;
        let element_bit_index = bit_index % 32;
        self.values[u32_element_index] = (self.values[u32_element_index]
            & !(1u32 << element_bit_index))
            | ((val as u32) << element_bit_index);
    }

    fn get_bit(&self, bit_index: usize) -> bool {
        assert!(bit_index < 32 * N, "bit index exceeded length of U32 array");
        let u32_element_index = bit_index / 32;
        let element_bit_index = bit_index % 32;
        (self.values[u32_element_index] & (1u32 << element_bit_index)) != 0
    }

    pub fn div_two(&mut self) {
        let mut carry = false;
        for i in (0..N).rev() {
            let new_carry = (self.values[i] & 0x00000001u32) == 1;
            let mut new_cell = self.values[i] >> 1;
            if carry {
                new_cell += 1 << 31;
            }
            carry = new_carry;
            self.values[i] = new_cell;
        }
    }

    // Linter complains about line setting `carry` but it looks fine to me. Maybe a bug?
    #[allow(clippy::shadow_unrelated)]
    pub fn mul_two(&mut self) {
        let mut carry = false;
        for i in 0..N {
            let (temp, carry_mul) = self.values[i].overflowing_mul(2);

            (self.values[i], carry) = temp.overflowing_add(carry as u32);
            carry |= carry_mul;
        }

        assert!(!carry, "Overflow in mul_two");
    }

    /// Returns (quotient, remainder)
    pub fn rem_div(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "Division by zero error");
        let mut quotient = Self::new([0; N]);
        let mut remainder = Self::new([0; N]);
        for i in (0..N * 32).rev() {
            remainder.mul_two();
            remainder.set_bit(0, self.get_bit(i));
            if remainder >= *divisor {
                remainder = remainder - *divisor;
                quotient.set_bit(i, true);
            }
        }

        (quotient, remainder)
    }
}

impl<const N: usize> From<U32s<N>> for BigUint {
    /// Convert a `U32s` to a `BigUInt` using big endian representation
    fn from(u32s: U32s<N>) -> Self {
        let mut acc: BigUint = BigUint::zero();
        for i in (0..N).rev() {
            acc <<= 32;
            let element_value: BigUint = u32s.values[i].into();
            acc += element_value;
        }

        acc
    }
}

impl<const N: usize> From<BigUint> for U32s<N> {
    /// Convert a `BigUInt` to a `U32s` using big endian representation
    fn from(bigint: BigUint) -> Self {
        let mut remaining: BigUint = bigint;
        let mut ret: Self = U32s::zero();
        for i in 0..N {
            ret.values[i] = (remaining.clone() % BigUint::new(vec![0, 1]))
                .try_into()
                .unwrap();
            remaining /= BigUint::new(vec![0, 1]);
        }

        ret
    }
}

impl<const N: usize> From<U32s<N>> for [BFieldElement; N] {
    fn from(value: U32s<N>) -> Self {
        let mut ret = [BFieldElement::zero(); N];
        for (&value_elem, ret_elem) in value.values.iter().zip(ret.iter_mut()) {
            *ret_elem = value_elem.into();
        }

        ret
    }
}

impl<const N: usize> PartialOrd for U32s<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Ord for U32s<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_values = self.values.iter().rev();
        let other_values = other.values.iter().rev();
        self_values.cmp(other_values)
    }
}

impl<const N: usize> Zero for U32s<N> {
    fn zero() -> Self {
        Self::new([0; N])
    }

    fn is_zero(&self) -> bool {
        self.values.iter().all(|x| *x == 0)
    }
}

impl<const N: usize> One for U32s<N> {
    fn one() -> Self {
        let mut ret_array = [0; N];
        ret_array[0] = 1;
        Self::new(ret_array)
    }

    fn is_one(&self) -> bool {
        *self == Self::one()
    }

    fn set_one(&mut self) {
        *self = One::one();
    }
}

impl<const N: usize> Sub for U32s<N> {
    type Output = Self;

    // Linter complains about line setting `carry_old` but it looks fine to me. Maybe a bug?
    #[allow(clippy::shadow_unrelated)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut carry_old = false;
        let mut res: U32s<N> = Self::new([0; N]);
        for i in 0..N {
            let (int, carry_new) = self.values[i].overflowing_sub(rhs.values[i]);
            (res.values[i], carry_old) = int.overflowing_sub(carry_old as u32);
            carry_old = carry_new || carry_old;
        }
        assert!(
            !carry_old,
            "overflow error in subtraction of U32s. Input: ({self:?}-{rhs:?})"
        );
        res
    }
}

impl<const N: usize> Add for U32s<N> {
    type Output = U32s<N>;

    // Linter complains about line setting `carry_old` but it looks fine to me. Maybe a bug?
    #[allow(clippy::shadow_unrelated)]
    fn add(self, other: U32s<N>) -> U32s<N> {
        let mut carry_old = false;
        let mut res: U32s<N> = Self::new([0; N]);
        for i in 0..N {
            let (int, carry_new) = self.values[i].overflowing_add(other.values[i]);
            (res.values[i], carry_old) = int.overflowing_add(carry_old.into());
            carry_old = carry_new || carry_old;
        }
        assert!(
            !carry_old,
            "overflow error in addition of U32s. Input: ({self:?}+{other:?})"
        );

        res
    }
}

impl<const N: usize> Div for U32s<N> {
    type Output = U32s<N>;

    fn div(self, rhs: Self) -> Self::Output {
        self.rem_div(&rhs).0
    }
}

impl<const N: usize> Rem for U32s<N> {
    type Output = U32s<N>;

    fn rem(self, rhs: Self) -> Self::Output {
        self.rem_div(&rhs).1
    }
}

impl<const N: usize> Mul for U32s<N> {
    type Output = U32s<N>;

    // Linter complains about line setting `add_carry` but it looks fine to me. Maybe a bug?
    #[allow(clippy::shadow_unrelated)]
    fn mul(self, other: U32s<N>) -> U32s<N> {
        let mut res: U32s<N> = Self::new([0; N]);
        for i in 0..N {
            let mut add_carry: bool;
            for j in 0..N {
                let hi_lo: u64 = self.values[i] as u64 * other.values[j] as u64;
                let hi: u32 = (hi_lo >> 32) as u32;
                let lo: u32 = hi_lo as u32;
                assert!(
                    i + j < N || hi == 0 && lo == 0,
                    "Overflow in multiplication",
                );

                if hi == 0 && lo == 0 {
                    continue;
                }

                // Use lo result
                (res.values[i + j], add_carry) = res.values[i + j].overflowing_add(lo);
                let mut k = 1;
                while add_carry {
                    assert!(i + j + k < N, "Overflow in multiplication",);
                    (res.values[i + j + k], add_carry) =
                        res.values[i + j + k].overflowing_add(add_carry as u32);
                    k += 1;
                }

                // Use hi result
                if hi == 0 {
                    continue;
                }

                assert!(i + j + 1 < N, "Overflow in multiplication",);
                (res.values[i + j + 1], add_carry) = res.values[i + j + 1].overflowing_add(hi);
                k = 2;
                while add_carry {
                    assert!(i + j + k < N, "Overflow in multiplication",);
                    (res.values[i + j + k], add_carry) =
                        res.values[i + j + k].overflowing_add(add_carry as u32);
                    k += 1;
                }
            }
        }

        res
    }
}

impl<const N: usize> Sum for U32s<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

impl<const N: usize> Distribution<U32s<N>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> U32s<N> {
        let values = rng.sample::<[u32; N], Standard>(Standard);
        U32s { values }
    }
}

impl<const N: usize> From<u32> for U32s<N> {
    fn from(n: u32) -> Self {
        let mut ret = U32s::zero();
        ret.values[0] = n;
        ret
    }
}

impl<const N: usize> TryFrom<u64> for U32s<N> {
    type Error = &'static str;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if N < 2 {
            return Err("U32s<{N}>, N<=1 may not be big enough to hold a u64");
        }
        Ok(U32s::<N>::from(BigUint::from(value)))
    }
}

impl<const N: usize> TryFrom<u128> for U32s<N> {
    type Error = &'static str;

    fn try_from(value: u128) -> Result<Self, Self::Error> {
        if N < 4 {
            return Err("U32s<{N}>, N<=3 may not be big enough to hold a u128");
        }
        Ok(U32s::<N>::from(BigUint::from(value)))
    }
}

impl<const N: usize> Display for U32s<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", BigUint::from(*self))
    }
}

impl<const N: usize> BFieldCodec for U32s<N> {
    type Error = BFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        if N > 0 && sequence.is_empty() {
            return Err(Self::Error::EmptySequence);
        }
        if sequence.len() < N {
            return Err(Self::Error::SequenceTooShort);
        }
        if sequence.len() > N {
            return Err(Self::Error::SequenceTooLong);
        }

        let mut array: [u32; N] = [0u32; N];
        for i in 0..N {
            array[i] = *u32::decode(&sequence[i..i + 1])?;
        }
        Ok(Box::new(Self::new(array)))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.values.into_iter().flat_map(|v| v.encode()).collect()
    }

    fn static_length() -> Option<usize> {
        Some(N)
    }
}

#[cfg(test)]
mod u32s_tests {
    use rand::{random, thread_rng, Rng, RngCore};

    use crate::shared_math::other::random_elements;

    use super::*;

    #[test]
    fn get_size_test() {
        let val_0 = U32s::new([]);
        assert_eq!(0, val_0.get_size());
        let val_1 = U32s::new([1 << 31]);
        assert_eq!(4, val_1.get_size());
        let val_2 = U32s::new([0, 0]);
        assert_eq!(4 * 2, val_2.get_size());
        let val_4: U32s<4> = 9999485u32.into();
        assert_eq!(4 * 4, val_4.get_size());
        let val_5 = U32s::new([(1 << 31) + 2001, 200, 400, 9999, 123456]);
        assert_eq!(4 * 5, val_5.get_size());
    }

    #[test]
    fn u32_conversion_test() {
        let a: U32s<4> = 9999485u32.into();
        assert_eq!(9999485u32, a.values[0]);
        for i in 1..4 {
            assert!(a.values[i].is_zero());
        }
    }

    #[test]
    fn u128_conversion_test() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let a_as_u128: u128 = (rng.next_u64() as u128) << 64 | rng.next_u64() as u128;
            let a: U32s<4> = a_as_u128.try_into().unwrap();
            let a_as_biguint: BigUint = a_as_u128.into();
            assert_eq!(a, Into::<U32s<4>>::into(a_as_biguint));
        }
    }

    #[test]
    fn convert_to_bfields_test() {
        let a = U32s::new([(1 << 31) + 2001, 200, 400, 9999, 123456]);
        let bfes: [BFieldElement; 5] = a.into();
        assert_eq!(bfes[0].value(), (1 << 31) + 2001);
        assert_eq!(bfes[1].value(), 200);
        assert_eq!(bfes[2].value(), 400);
        assert_eq!(bfes[3].value(), 9999);
        assert_eq!(bfes[4].value(), 123456);
    }

    #[test]
    fn simple_add_test() {
        let a = U32s::new([1 << 31, 0, 0, 0]);
        let b = U32s::new([1 << 31, 0, 0, 0]);
        let expected = U32s::new([0, 1, 0, 0]);
        assert_eq!(expected, a + b);
    }

    #[test]
    fn simple_sum_test() {
        let a = U32s::new([1 << 31, 0, 0, 0]);
        let b = U32s::new([1 << 31, 0, 0, 0]);
        assert_eq!(U32s::new([0, 1, 0, 0]), vec![a, b].into_iter().sum());

        let c = U32s::new([1 << 31, 0, 19876, 0]);
        assert_eq!(
            U32s::new([1 << 31, 1, 19876, 0]),
            vec![a, b, c].into_iter().sum()
        );
    }

    #[test]
    #[should_panic]
    fn sum_panic_test() {
        let a = U32s::new([0, 0, 0, 1 << 31]);
        let b = U32s::new([0, 0, 0, 1 << 30]);
        let c = U32s::new([0, 0, 0, 1 << 30]);
        let _res: U32s<4> = vec![a, b, c].into_iter().sum();
    }

    #[test]
    fn simple_sub_test() {
        let a = U32s::new([0, 17, 5, 52]);
        let b = U32s::new([1 << 31, 0, 0, 0]);
        let expected = U32s::new([1 << 31, 16, 5, 52]);
        assert_eq!(expected, a - b);
    }

    #[test]
    fn simple_mul_test() {
        let a = U32s::new([41000, 17, 5, 0]);
        let b = U32s::new([3, 2, 0, 0]);
        let expected = U32s::new([123000, 82051, 49, 10]);
        assert_eq!(expected, a * b);
    }

    #[test]
    fn mul_test_with_carry() {
        let a = U32s::new([1 << 31, 0, 1 << 9, 0]);
        let b = U32s::new([1 << 31, 1 << 17, 0, 0]);
        assert_eq!(
            U32s::new([0, 1 << 30, 1 << 16, (1 << 26) + (1 << 8)]),
            a * b
        );
    }

    #[test]
    #[should_panic(expected = "Overflow in multiplication")]
    fn mul_overflow_test_0() {
        let a = U32s::new([41000, 17, 5, 0]);
        let b = U32s::new([3, 2, 2, 0]);
        let _c = a * b;
    }

    #[test]
    #[should_panic(expected = "Overflow in multiplication")]
    fn mul_overflow_test_1() {
        let a = U32s::new([0, 0, 1, 0]);
        let b = U32s::new([0, 0, 1, 0]);
        let _c = a * b;
    }

    #[test]
    fn mod_div_simple_test_0() {
        let a = U32s::new([12, 0]);
        let b = U32s::new([4, 0]);
        assert_eq!((U32s::new([3, 0]), U32s::new([0, 0])), a.rem_div(&b));
    }

    #[test]
    fn mod_div_simple_test_1() {
        let a = U32s::new([13, 64]);
        let b = U32s::new([4, 0]);
        assert_eq!((U32s::new([3, 16]), U32s::new([1, 0])), a.rem_div(&b));
    }

    #[test]
    fn mod_div_simple_test_2() {
        let a = U32s::new([420_000_000, 0, 420_000_000, 0]);
        assert_eq!(
            (U32s::new([210_000, 0, 210_000, 0]), U32s::zero()),
            a.rem_div(&U32s::new([2000, 0, 0, 0]))
        );
        assert_eq!(
            (
                U32s::new([0, 2_100_000, 0, 0]),
                U32s::new([420_000_000, 0, 0, 0])
            ),
            a.rem_div(&U32s::new([0, 200, 0, 0]))
        );
        assert_eq!(
            (
                U32s::new([21_000_000, 0, 0, 0]),
                U32s::new([420_000_000, 0, 0, 0])
            ),
            a.rem_div(&U32s::new([0, 0, 20, 0]))
        );
    }

    #[test]
    fn set_bit_simple_test() {
        let mut a = U32s::new([12, 0]);
        a.set_bit(10, true);
        assert_eq!(U32s::new([1036, 0]), a);
        a.set_bit(10, false);
        assert_eq!(U32s::new([12, 0]), a);
        a.set_bit(42, true);
        assert_eq!(U32s::new([12, 1024]), a);
        a.set_bit(0, true);
        assert_eq!(U32s::new([13, 1024]), a);
    }

    #[test]
    fn get_bit_test() {
        let a = U32s::new([0x010000, 1024]);
        for i in 0..64 {
            assert_eq!(
                i == 16 || i == 42,
                a.get_bit(i),
                "bit i must match set value for i = {i}"
            );
        }
    }

    #[test]
    fn compare_simple_test() {
        assert!(U32s::new([1]) > U32s::new([0]));
        assert!(U32s::new([100]) > U32s::new([0]));
        assert!(U32s::new([100]) > U32s::new([99]));
        assert!(U32s::new([100, 0]) > U32s::new([99, 0]));
        assert!(U32s::new([0, 1]) > U32s::new([1 << 31, 0]));
        assert!(U32s::new([542, 12]) > U32s::new([1 << 31, 11]));
    }

    #[test]
    fn compare_simple_test_more() {
        assert!(U32s::new([0]) < U32s::new([1]));
        assert!(U32s::new([0]) <= U32s::new([100]));
        assert!(U32s::new([99]) < U32s::new([100]));
        assert!(U32s::new([99, 0]) <= U32s::new([100, 0]));
        assert!(U32s::new([100, 0]) <= U32s::new([100, 0]));
        assert!(U32s::new([100, 0]) >= U32s::new([100, 0]));
        assert!(U32s::new([1 << 31, 0]) < U32s::new([0, 1]));
        assert!(U32s::new([1 << 31, 11]) <= U32s::new([542, 12]));
        assert!(U32s::new([0]) >= U32s::new([0]));
    }

    #[test]
    fn mul_two_div_two_simple_test() {
        // Without carry
        let a = U32s::new([3, 6, 9, 12]);
        let mut x = a;
        x.mul_two();
        assert_eq!(U32s::new([6, 12, 18, 24]), x);
        x.div_two();
        assert_eq!(a, x);

        // With carry
        let b = U32s::new([3, 6, 9 + (1 << 31), 12]);
        x = b;
        x.mul_two();
        assert_eq!(U32s::new([6, 12, 18, 25]), x);
        x.div_two();
        assert_eq!(U32s::new([3, 6, 9 + (1 << 31), 12]), x);
        x.div_two();
        assert_eq!(U32s::new([1, 3 + (1 << 31), 4 + (1 << 30), 6]), x);
    }

    #[test]
    fn mul_two_div_two_pbt() {
        let vals = random_masked_u32s(100, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF]);
        for val in vals {
            let mut calculated = val;
            calculated.mul_two();
            calculated.div_two();
            assert_eq!(val, calculated);
        }
    }

    #[test]
    fn identity_mul_test() {
        let masks: Vec<[u32; 4]> = vec![
            [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
            [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0],
            [0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0],
            [0xFFFFFFFF, 0x0, 0x0, 0x0],
        ];
        for (i, mask) in masks.into_iter().enumerate() {
            let mut rhs = U32s::new([0; 4]);
            rhs.values[i] = 1u32;
            let vals = random_masked_u32s(100, mask);
            for val in vals {
                let mut expected = val;
                expected.values.rotate_right(i);
                assert_eq!(expected, val * rhs);
            }
        }
    }

    #[test]
    fn div_mul_pbt() {
        let count = 40;
        let vals: Vec<U32s<4>> = random_elements(2 * count);
        for i in 0..count {
            let (quot, rem) = vals[2 * i].rem_div(&vals[2 * i + 1]);
            assert_eq!(vals[2 * i], quot * vals[2 * i + 1] + rem);
            assert!(rem < vals[2 * i + 1]);
            assert_eq!(quot, vals[2 * i] / vals[2 * i + 1]);
            assert_eq!(rem, vals[2 * i] % vals[2 * i + 1]);
        }

        // Restrict divisors to 2^64, so quotients are usually in the range of 2^64
        let divisors = random_masked_u32s(2 * count, [0xFFFFFFFF, 0xFFFFFFFF, 0x00, 0x00]);
        for i in 0..2 * count {
            let (quot, rem) = vals[i].rem_div(&divisors[i]);
            assert_eq!(vals[i], quot * divisors[i] + rem);
            assert!(quot * divisors[i] < vals[i]);
            assert!(rem < divisors[i]);
            assert!(quot > U32s::new([0, 1, 0, 0])); // True with a probability of ~=1 - 2^(-33)
        }
    }

    #[test]
    fn biguinteger_conversion_test() {
        let a: U32s<4> = U32s::new([2000u32, 0, 0, 0]);
        let expected_biguint_a: BigUint = 2000u32.into();
        let converted_biguint_a: BigUint = a.into();
        assert_eq!(expected_biguint_a, converted_biguint_a);

        let b: U32s<4> = U32s::new([2000u32, 124, 7, 300001]);
        let expected_biguint_b: BigUint =
            (2000u128 + 124 * (1u128 << 32) + 7 * (1u128 << 64) + 300001 * (1u128 << 96)).into();
        let converted_biguint_b: BigUint = b.into();
        assert_eq!(expected_biguint_b, converted_biguint_b);

        // Converting to BigUint and converting back again is the identity operator
        assert_eq!(a, converted_biguint_a.into());
        assert_eq!(b, converted_biguint_b.clone().into());
        assert_ne!(a, converted_biguint_b.into());
    }

    #[test]
    fn biguinteger_conversion_pbt() {
        let count = 100;
        let inputs: Vec<U32s<5>> = random_elements(count);
        for input in inputs {
            let biguint: BigUint = input.into();
            let back_again: U32s<5> = biguint.into();
            assert_eq!(input, back_again);
        }
    }

    #[test]
    fn sub_add_pbt() {
        let count = 100;
        let mask = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF];
        let inputs = random_masked_u32s(2 * count, mask);

        let zero = U32s::<4>::zero();
        let one = U32s::<4>::one();
        let two = one + one;

        for i in 0..count {
            let sum = inputs[2 * i + 1] + inputs[2 * i];
            assert_eq!(inputs[2 * i], sum - inputs[2 * i + 1]);
            assert_eq!(inputs[2 * i + 1], sum - inputs[2 * i]);

            // Let's also test compare, while we're at it
            assert!(sum >= inputs[2 * i]);
            assert!(sum >= inputs[2 * i + 1]);

            // subtracting/adding one could overflow if LHS is zero, but the chances are negligible (~= 2^(-126))
            assert!(inputs[2 * i] - one < inputs[2 * i]);
            assert!(inputs[2 * i] + one > inputs[2 * i]);
            assert!(inputs[2 * i] == inputs[2 * i]);

            // And simple identities
            assert_eq!(inputs[2 * i], inputs[2 * i] + zero);
            assert_eq!(inputs[2 * i], inputs[2 * i] * one);
            assert_eq!(inputs[2 * i] + inputs[2 * i], inputs[2 * i] * two);
            let mut two_self = inputs[2 * i];
            two_self.mul_two();
            assert_eq!(inputs[2 * i] + inputs[2 * i], two_self);
        }
    }

    #[test]
    fn div_2_pbt() {
        let count = 100;
        let vals: Vec<U32s<4>> = random_elements(count);
        for val in vals {
            let even: bool = (val.values[0] & 0x00000001u32) == 0;
            let mut calculated = val;
            calculated.div_two();
            calculated = calculated + calculated;
            if even {
                assert_eq!(val, calculated);
            } else {
                assert_eq!(val, calculated + U32s::one());
            }
        }
    }

    #[test]
    fn get_bit_set_bit_pbt() {
        let outer_count = 100;
        let inner_count = 20;
        let mut rng = rand::thread_rng();
        let vals: Vec<U32s<4>> = random_elements(outer_count);
        for mut val in vals {
            let bit_value: bool = rng.gen();
            for _ in 0..inner_count {
                let bit_index = rng.gen_range(0..4 * 32);
                val.set_bit(bit_index, bit_value);
                assert_eq!(bit_value, val.get_bit(bit_index));
            }
        }
    }

    fn random_masked_u32s<const N: usize>(count: usize, and_mask: [u32; N]) -> Vec<U32s<N>> {
        let mut elems: Vec<U32s<N>> = random_elements(count);

        for elem in elems.iter_mut() {
            for (value, mask) in elem.values.iter_mut().zip(and_mask) {
                *value &= mask;
            }
        }

        elems
    }

    #[test]
    fn serialization_test() {
        // TODO: You could argue that this test doesn't belong here, as it tests the behavior of
        // an imported library. I included it here, though, because the setup seems a bit clumsy
        // to me so far.
        let s = U32s {
            values: [9788888u32; 64],
        };
        let j = serde_json::to_string(&s).unwrap();
        let s_back = serde_json::from_str::<U32s<64>>(&j).unwrap();
        assert!(s.values[..] == s_back.values[..]);
    }

    #[test]
    fn display_u32s() {
        let v = u64::MAX;
        let u32s = U32s::<4>::try_from(v).unwrap();

        let v_string = format!("{v}");
        let u32s_string = format!("{u32s}");

        assert_eq!(v_string, u32s_string)
    }

    #[ignore]
    #[test]
    fn crash() {
        let _u32s = U32s::<0>::from(0u32);
    }

    #[test]
    fn conversion_test() {
        let u32_max_actual = U32s::<5>::from(u32::MAX);
        let u32_max_expected = U32s::<5>::new([u32::MAX, 0, 0, 0, 0]);
        assert_eq!(u32_max_actual, u32_max_expected);
    }

    #[test]
    fn test_u32s_codec() {
        let array: [u32; 5] = random();
        let u32s = U32s::new(array);
        let encoded = u32s.encode();
        let decoded: U32s<5> = *U32s::decode(&encoded).unwrap();
        assert_eq!(u32s, decoded);
    }
}
