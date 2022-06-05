use std::ops::{Add, Div, Mul, Rem, Sub};

use num_traits::{One, Zero};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct U32s<const N: usize>([u32; N]);

impl<const N: usize> Eq for U32s<N> {}

impl<const N: usize> U32s<N> {
    fn set_bit(&mut self, bit_index: usize, val: bool) {
        assert!(bit_index < 32 * N, "bit index exceeded length of U32 array");
        let u32_element_index = bit_index / 32;
        let element_bit_index = bit_index % 32;
        self.0[u32_element_index] = (self.0[u32_element_index] & !(1u32 << element_bit_index))
            | ((val as u32) << element_bit_index);
    }

    fn get_bit(&self, bit_index: usize) -> bool {
        assert!(bit_index < 32 * N, "bit index exceeded length of U32 array");
        let u32_element_index = bit_index / 32;
        let element_bit_index = bit_index % 32;
        (self.0[u32_element_index] & (1u32 << element_bit_index)) != 0
    }

    pub fn div_two(&mut self) {
        let mut carry = false;
        for i in (0..N).rev() {
            let new_carry = (self.0[i] & 0x00000001u32) == 1;
            let mut new_cell = self.0[i] >> 1;
            if carry {
                new_cell += 1 << 31;
            }
            carry = new_carry;
            self.0[i] = new_cell;
        }
    }

    // Linter complains about line setting `carry` but it looks fine to me. Maybe a bug?
    #[allow(clippy::shadow_unrelated)]
    pub fn mul_two(&mut self) {
        let mut carry = false;
        for i in 0..N {
            let (temp, carry_mul) = self.0[i].overflowing_mul(2);

            (self.0[i], carry) = temp.overflowing_add(carry as u32);
            carry |= carry_mul;
        }

        assert!(!carry, "Overflow in mul_two");
    }

    /// Returns (quotient, remainder)
    pub fn rem_div(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "Division by zero error");
        let mut quotient = Self([0; N]);
        let mut remainder = Self([0; N]);
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

impl<const N: usize> PartialOrd for U32s<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        for i in (0..N).rev() {
            if self.0[i] != other.0[i] {
                return Some(self.0[i].cmp(&other.0[i]));
            }
        }

        Some(std::cmp::Ordering::Equal)
    }
}

impl<const N: usize> Ord for U32s<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<const N: usize> Zero for U32s<N> {
    fn zero() -> Self {
        Self([0; N])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| *x == 0)
    }
}

impl<const N: usize> One for U32s<N> {
    fn one() -> Self {
        let mut ret_array = [0; N];
        ret_array[0] = 1;
        Self(ret_array)
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
        let mut res: U32s<N> = U32s([0; N]);
        for i in 0..N {
            let (int, carry_new) = self.0[i].overflowing_sub(rhs.0[i]);
            (res.0[i], carry_old) = int.overflowing_sub(carry_old as u32);
            carry_old = carry_new || carry_old;
        }
        assert!(
            !carry_old,
            "overflow error in subtraction of U32s. Input: ({:?}-{:?})",
            self, rhs
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
        let mut res: U32s<N> = U32s([0; N]);
        for i in 0..N {
            let (int, carry_new) = self.0[i].overflowing_add(other.0[i]);
            (res.0[i], carry_old) = int.overflowing_add(carry_old.into());
            carry_old = carry_new || carry_old;
        }
        assert!(
            !carry_old,
            "overflow error in addition of U32s. Input: ({:?}+{:?})",
            self, other
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
        let mut res: U32s<N> = U32s([0; N]);
        for i in 0..N {
            let mut add_carry: bool;
            for j in 0..N {
                let hi_lo: u64 = self.0[i] as u64 * other.0[j] as u64;
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
                (res.0[i + j], add_carry) = res.0[i + j].overflowing_add(lo);
                let mut k = 1;
                while add_carry {
                    assert!(i + j + k < N, "Overflow in multiplication",);
                    (res.0[i + j + k], add_carry) =
                        res.0[i + j + k].overflowing_add(add_carry as u32);
                    k += 1;
                }

                // Use hi result
                if hi == 0 {
                    continue;
                }

                assert!(i + j + 1 < N, "Overflow in multiplication",);
                (res.0[i + j + 1], add_carry) = res.0[i + j + 1].overflowing_add(hi);
                k = 2;
                while add_carry {
                    assert!(i + j + k < N, "Overflow in multiplication",);
                    (res.0[i + j + k], add_carry) =
                        res.0[i + j + k].overflowing_add(add_carry as u32);
                    k += 1;
                }
            }
        }

        res
    }
}

#[cfg(test)]
mod u32s_tests {
    use rand::{thread_rng, RngCore};

    use super::*;

    #[test]
    fn simple_add_test() {
        let a = U32s([1 << 31, 0, 0, 0]);
        let b = U32s([1 << 31, 0, 0, 0]);
        let expected = U32s([0, 1, 0, 0]);
        assert_eq!(expected, a + b);
    }

    #[test]
    fn simple_sub_test() {
        let a = U32s([0, 17, 5, 52]);
        let b = U32s([1 << 31, 0, 0, 0]);
        let expected = U32s([1 << 31, 16, 5, 52]);
        assert_eq!(expected, a - b);
    }

    #[test]
    fn simple_mul_test() {
        let a = U32s([41000, 17, 5, 0]);
        let b = U32s([3, 2, 0, 0]);
        let expected = U32s([123000, 82051, 49, 10]);
        assert_eq!(expected, a * b);
    }

    #[test]
    fn mul_test_with_carry() {
        let a = U32s([1 << 31, 0, 1 << 9, 0]);
        let b = U32s([1 << 31, 1 << 17, 0, 0]);
        assert_eq!(U32s([0, 1 << 30, 1 << 16, (1 << 26) + (1 << 8)]), a * b);
    }

    #[test]
    #[should_panic(expected = "Overflow in multiplication")]
    fn mul_overflow_test_0() {
        let a = U32s([41000, 17, 5, 0]);
        let b = U32s([3, 2, 2, 0]);
        let _c = a * b;
    }

    #[test]
    #[should_panic(expected = "Overflow in multiplication")]
    fn mul_overflow_test_1() {
        let a = U32s([0, 0, 1, 0]);
        let b = U32s([0, 0, 1, 0]);
        let _c = a * b;
    }

    #[test]
    fn mod_div_simple_test_0() {
        let a = U32s([12, 0]);
        let b = U32s([4, 0]);
        assert_eq!((U32s([3, 0]), U32s([0, 0])), a.rem_div(&b));
    }

    #[test]
    fn mod_div_simple_test_1() {
        let a = U32s([13, 64]);
        let b = U32s([4, 0]);
        assert_eq!((U32s([3, 16]), U32s([1, 0])), a.rem_div(&b));
    }

    #[test]
    fn mod_div_simple_test_2() {
        let a = U32s([420_000_000, 0, 420_000_000, 0]);
        assert_eq!(
            (U32s([210_000, 0, 210_000, 0]), U32s::zero()),
            a.rem_div(&U32s([2000, 0, 0, 0]))
        );
        assert_eq!(
            (U32s([0, 2_100_000, 0, 0]), U32s([420_000_000, 0, 0, 0])),
            a.rem_div(&U32s([0, 200, 0, 0]))
        );
        assert_eq!(
            (U32s([21_000_000, 0, 0, 0]), U32s([420_000_000, 0, 0, 0])),
            a.rem_div(&U32s([0, 0, 20, 0]))
        );
    }

    #[test]
    fn set_bit_simple_test() {
        let mut a = U32s([12, 0]);
        a.set_bit(10, true);
        assert_eq!(U32s([1036, 0]), a);
        a.set_bit(10, false);
        assert_eq!(U32s([12, 0]), a);
        a.set_bit(42, true);
        assert_eq!(U32s([12, 1024]), a);
        a.set_bit(0, true);
        assert_eq!(U32s([13, 1024]), a);
    }

    #[test]
    fn get_bit_test() {
        let a = U32s([0x010000, 1024]);
        for i in 0..64 {
            assert_eq!(
                i == 16 || i == 42,
                a.get_bit(i),
                "bit i must match set value for i = {}",
                i
            );
        }
    }

    #[test]
    fn compare_simple_test() {
        assert!(U32s([1]) > U32s([0]));
        assert!(U32s([100]) > U32s([0]));
        assert!(U32s([100]) > U32s([99]));
        assert!(U32s([100, 0]) > U32s([99, 0]));
        assert!(U32s([0, 1]) > U32s([1 << 31, 0]));
        assert!(U32s([542, 12]) > U32s([1 << 31, 11]));
    }

    #[test]
    fn mul_two_div_two() {
        let vals = get_u32s::<4>(100, Some([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF]));
        for val in vals {
            let mut calculated = val;
            calculated.mul_two();
            calculated.div_two();
            assert_eq!(val, calculated);
        }
    }

    #[test]
    fn identity_mul_test() {
        let masks: [Option<[u32; 4]>; 4] = [
            Some([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
            Some([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0]),
            Some([0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0]),
            Some([0xFFFFFFFF, 0x0, 0x0, 0x0]),
        ];
        let mut rhs: U32s<4>;
        for i in 0..4 {
            rhs = U32s([0; 4]);
            rhs.0[i] = 1u32;
            let vals = get_u32s::<4>(100, masks[i]);
            for val in vals {
                let mut expected = val;
                expected.0.rotate_right(i);
                assert_eq!(expected, val * rhs);
            }
        }
    }

    #[test]
    fn div_mul_pbt() {
        let count = 40;
        let vals: Vec<U32s<4>> = get_u32s::<4>(2 * count, None);
        for i in 0..count {
            let (quot, rem) = vals[2 * i].rem_div(&vals[2 * i + 1]);
            assert_eq!(vals[2 * i], quot * vals[2 * i + 1] + rem);
            assert!(rem < vals[2 * i + 1]);
            assert_eq!(quot, vals[2 * i] / vals[2 * i + 1]);
            assert_eq!(rem, vals[2 * i] % vals[2 * i + 1]);
        }

        // Restrict divisors to 2^64, so quotients are usually in the range of 2^64
        let divisors: Vec<U32s<4>> =
            get_u32s::<4>(2 * count, Some([0xFFFFFFFF, 0xFFFFFFFF, 0x00, 0x00]));
        for i in 0..2 * count {
            let (quot, rem) = vals[i].rem_div(&divisors[i]);
            assert_eq!(vals[i], quot * divisors[i] + rem);
            assert!(rem < divisors[i]);
            assert!(quot > U32s([0, 1, 0, 0])); // True with a probability of ~=1 - 2^(-33)
        }
    }

    #[test]
    fn sub_add_pbt() {
        let count = 100;
        let inputs = get_u32s::<4>(
            2 * count,
            Some([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF]),
        );

        let one = U32s::<4>::one();
        let zero = U32s::<4>::zero();

        for i in 0..count {
            let sum = inputs[2 * i + 1] + inputs[2 * i];
            assert_eq!(inputs[2 * i], sum - inputs[2 * i + 1]);
            assert_eq!(inputs[2 * i + 1], sum - inputs[2 * i]);

            // Let's also test compare, while we're at it
            assert!(sum >= inputs[2 * i]);
            assert!(sum >= inputs[2 * i + 1]);

            // subtracting one could overflow if LHS is zero, but the chances are negligible
            assert!(inputs[2 * i] - one < inputs[2 * i]);
            assert!(inputs[2 * i] == inputs[2 * i]);

            // And simple identities
            assert_eq!(inputs[2 * i], inputs[2 * i] + zero);
            assert_eq!(inputs[2 * i], inputs[2 * i] * one);
        }
    }

    #[test]
    fn div_2_pbt() {
        let count = 100;
        let vals: Vec<U32s<4>> = get_u32s::<4>(count, None);
        for val in vals {
            let even: bool = (val.0[0] & 0x00000001u32) == 0;
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

    fn get_u32s<const N: usize>(count: usize, and_mask: Option<[u32; N]>) -> Vec<U32s<N>> {
        let mut prng = thread_rng();
        let mut rets: Vec<U32s<N>> = vec![];
        for _ in 0..count {
            let mut a: U32s<N> = U32s([0; N]);
            for i in 0..N {
                a.0[i] = prng.next_u32();
            }

            a = match and_mask {
                None => a,
                Some(mask) => {
                    for i in 0..N {
                        a.0[i] &= mask[i];
                    }
                    a
                }
            };

            rets.push(a);
        }

        rets
    }
}
