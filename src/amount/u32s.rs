use std::ops::{Add, Mul, Sub};

use num_traits::{ops::overflowing::OverflowingMul, One, Zero};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct U32s<const N: usize>([u32; N]);

impl<const N: usize> Zero for U32s<N> {
    fn zero() -> Self {
        Self([0; N])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|x| *x == 0)
    }
}

// impl<const N: usize> One for U32s<N> {
//     fn one() -> Self {}

//     fn is_one(&self) -> bool {
//         *self == Self::one()
//     }

//     fn set_one(&mut self) {
//         // *self = One::one();
//     }
// }

impl<const N: usize> Sub for U32s<N> {
    type Output = Self;

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

impl<const N: usize> Mul for U32s<N> {
    type Output = U32s<N>;
    fn mul(self, other: U32s<N>) -> U32s<N> {
        let mut res: U32s<N> = U32s([0; N]);
        for i in 0..N {
            let mut add_carry: bool = false;
            for j in 0..N {
                // let (lo, hi) = self.0[i].wrapping_mul(other, carry);
                // let (lo, hi) = self.0[i].widening_mul(other.0[j]);
                // let res = self.0[i].widening_mul(other.0[j]);
                // let res = self.0[i].carrying_mul(other.0[j], carry);
                // assert!(i + j > N && )
                let hi_lo: u64 = self.0[i] as u64 * other.0[j] as u64;
                let hi: u32 = (hi_lo >> 32) as u32;
                let lo: u32 = hi_lo as u32;
                assert!(
                    i + j <= N || hi == 0 && lo == 0,
                    "Overflow in multiplication. Got: {:?}*{:?}",
                    self,
                    other
                );

                if hi == 0 && lo == 0 {
                    continue;
                }

                // Use lo result
                (res.0[i + j], add_carry) = res.0[i + j].overflowing_add(lo);
                let mut k = 1;
                while add_carry {
                    (res.0[i + j + k], add_carry) =
                        res.0[i + j + k].overflowing_add(add_carry as u32);
                    k += 1;
                }

                // Use hi result
                if hi == 0 {
                    continue;
                }
            }
        }

        todo!()
        // let mut carry_old = false;
        // let mut res: U32s<N> = U32s([0; N]);
        // for i in 0..N {
        //     let (int, carry_new) = self.0[i].overflowing_add(other.0[i]);
        //     (res.0[i], carry_old) = int.overflowing_add(carry_old.into());
        //     carry_old = carry_new || carry_old;
        // }
        // assert!(
        //     !carry_old,
        //     "overflow error in addition of U32s. Input: ({:?}+{:?})",
        //     self, other
        // );

        // res
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
    fn sub_add_pbt() {
        let count = 100;
        let inputs = get_u32s::<4>(
            2 * count,
            Some([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF]),
        );
        for i in 0..count {
            assert_eq!(
                inputs[2 * i],
                inputs[2 * i + 1] + inputs[2 * i] - inputs[2 * i + 1]
            );
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
