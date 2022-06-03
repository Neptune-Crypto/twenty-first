use std::ops::Add;

#[derive(Clone, Debug, PartialEq)]
pub struct FourU32s([u32; 4]);

impl Add for FourU32s {
    type Output = FourU32s;
    fn add(self, other: FourU32s) -> FourU32s {
        let mut carry_old = false;
        let mut res: FourU32s = FourU32s([0, 0, 0, 0]);
        for i in 0..4 {
            // res.0[i] = self.0[i] + other.0[i] + carry;
            let (int, carry_new) = self.0[i].overflowing_add(other.0[i]);
            (res.0[i], carry_old) = int.overflowing_add(carry_old.into());
            carry_old = carry_new || carry_old;
        }
        assert!(!carry_old, "overflow error in addition of FourU32s");
        res
    }
}

#[cfg(test)]
mod four_32s_tests {
    use super::*;

    #[test]
    fn simple_add_test() {
        let a = FourU32s([1 << 31, 0, 0, 0]);
        let b = FourU32s([1 << 31, 0, 0, 0]);
        let expected = FourU32s([0, 1, 0, 0]);
        assert_eq!(expected, a + b);
    }
}
