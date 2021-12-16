use crate::shared_math::traits::{IdentityValues, ModPowU64};
use crate::utils::FIRST_THOUSAND_PRIMES;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Display},
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

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
        let (_, _, a) = Self::xgcd(Self::QUOTIENT as i128, self.0 as i128);

        Self {
            0: ((a % Self::QUOTIENT as i128 + Self::QUOTIENT as i128) % Self::QUOTIENT as i128)
                as u128,
        }
    }

    // TODO: Name this collection of traits as something like... FieldElementInternalNumRepresentation
    pub fn xgcd<
        T: Zero + One + Rem<Output = T> + Div<Output = T> + Sub<Output = T> + Clone + Display,
    >(
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

    // TODO: Use Rust Pow. TODO: Maybe move this out into a library along with xgcd().
    // TODO: Name this collection of traits as something like... FieldElementInternalNumRepresentation
    fn mod_pow_raw(&self, pow: u64) -> u128 {
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

    // TODO: Currently, IdentityValues has &self as part of its signature, so we hotfix
    // being able to refer to a zero/one element without having an element at hand. This
    // will go away when moving to Zero/One traits.
    pub fn ring_zero() -> Self {
        Self { 0: 0 }
    }

    pub fn ring_one() -> Self {
        Self { 0: 1 }
    }

    pub fn mod_pow(&self, pow: u64) -> Self {
        Self {
            0: self.mod_pow_raw(pow),
        }
    }

    // TODO: Maybe make u128 into <T>?
    pub fn get_primitive_root_of_unity(n: u128) -> (Option<BFieldElement>, Vec<u128>) {
        let mut primes: Vec<u128> = vec![];

        if n <= 1 {
            return (Some(BFieldElement::ring_one()), primes);
        }

        // Calculate prime factorization of n
        // Check if n = 2^k
        if n & (n - 1) == 0 {
            primes = vec![2];
        } else {
            let mut m = n;
            for prime in FIRST_THOUSAND_PRIMES.iter().map(|&p| p as u128) {
                if m == 1 {
                    break;
                }
                if m % prime == 0 {
                    primes.push(prime);
                    while m % prime == 0 {
                        m /= prime;
                    }
                }
            }
            // This might be prohibitively expensive
            if m > 1 {
                let mut other_primes = BFieldElement::primes_lt(m)
                    .into_iter()
                    .filter(|&x| n % x == 0)
                    .collect();
                primes.append(&mut other_primes);
            }
        };

        // N must divide the field prime minus one for a primitive nth root of unity to exist
        if !((Self::QUOTIENT - 1) % n).is_zero() {
            return (None, primes);
        }

        let mut primitive_root: Option<BFieldElement> = None;
        let mut candidate: BFieldElement = BFieldElement::ring_one();
        #[allow(clippy::suspicious_operation_groupings)]
        while primitive_root == None && candidate.0 < Self::QUOTIENT {
            if (-candidate.legendre_symbol()).is_one()
                && primes.iter().filter(|&x| n % x == 0).all(|x| {
                    !candidate
                        .mod_pow_raw(((Self::QUOTIENT - 1) / x) as u64)
                        .is_one()
                })
            {
                primitive_root = Some(candidate.mod_pow(((Self::QUOTIENT - 1) / n) as u64));
            }

            candidate.0 += 1;
        }

        (primitive_root, primes)
    }

    // TODO: Abstract for both i128 and u128 so we don't keep multiple copies of the same algorithm.
    fn primes_lt(bound: u128) -> Vec<u128> {
        let mut primes: Vec<bool> = (0..bound + 1).map(|num| num == 2 || num & 1 != 0).collect();
        let mut num = 3u128;
        while num * num <= bound {
            let mut j = num * num;
            while j <= bound {
                primes[j as usize] = false;
                j += num;
            }
            num += 2;
        }
        primes
            .into_iter()
            .enumerate()
            .skip(2)
            .filter_map(|(i, p)| if p { Some(i as u128) } else { None })
            .collect::<Vec<u128>>()
    }

    pub fn legendre_symbol(&self) -> i8 {
        let elem = self.mod_pow((Self::QUOTIENT - 1) as u64 / 2).0;

        // Ugly hack to force a result in {-1,0,1}
        if elem == Self::QUOTIENT - 1 {
            -1
        } else if elem == 0 {
            0
        } else {
            1
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

    // XXX
    fn div(self, other: Self) -> Self {
        Self {
            0: (other.inv().0 * self.0) % Self::QUOTIENT,
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
    use crate::{
        shared_math::{b_field_element::*, polynomial::Polynomial},
        utils::generate_random_numbers,
    };
    use proptest::prelude::*;

    // TODO: Move this into separate file.
    macro_rules! bfield_elem {
        ($value:expr) => {{
            BFieldElement::new($value)
        }};
    }

    #[test]
    fn test_zero_one() {
        let zero = bfield_elem!(0);
        let one = bfield_elem!(1);

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(!one.is_zero());
        assert!(one.is_one());
        assert!(bfield_elem!(BFieldElement::MAX + 1).is_zero());
        assert!(bfield_elem!(BFieldElement::MAX + 2).is_one());
    }

    proptest! {
        #[test]
        fn identity_tests(n in 0u128..BFieldElement::MAX) {
            let zero = bfield_elem!(0);
            let one = bfield_elem!(1);
            let other = bfield_elem!(n);

            prop_assert_eq!(other, zero + other, "left zero identity");
            prop_assert_eq!(other, other + zero, "right zero identity");
            prop_assert_eq!(other, one * other, "left one identity");
            prop_assert_eq!(other, other * one, "right one identity");
        }
    }

    #[test]
    fn inversion_test() {
        assert_eq!(bfield_elem!(9223372034707292161), bfield_elem!(2).inv());
        assert_eq!(bfield_elem!(12297829379609722881), bfield_elem!(3).inv());
        assert_eq!(bfield_elem!(13835058052060938241), bfield_elem!(4).inv());
        assert_eq!(bfield_elem!(14757395255531667457), bfield_elem!(5).inv());
        assert_eq!(bfield_elem!(15372286724512153601), bfield_elem!(6).inv());
        assert_eq!(bfield_elem!(2635249152773512046), bfield_elem!(7).inv());
        assert_eq!(bfield_elem!(16140901060737761281), bfield_elem!(8).inv());
        assert_eq!(bfield_elem!(4099276459869907627), bfield_elem!(9).inv());
        assert_eq!(bfield_elem!(16602069662473125889), bfield_elem!(10).inv());
        assert_eq!(
            bfield_elem!(11826576560539181984),
            bfield_elem!(8567106).inv()
        );
    }

    #[test]
    fn inversion_property_based_test() {
        let rands: Vec<i128> = generate_random_numbers(30, BFieldElement::MAX as i128);
        for rand in rands {
            assert!((bfield_elem!(rand as u128).inv() * bfield_elem!(rand as u128)).is_one());
        }
    }

    #[test]
    fn mul_div_plus_minus_property_based_test() {
        let rands: Vec<i128> = generate_random_numbers(30, BFieldElement::QUOTIENT as i128);
        for i in 1..rands.len() {
            let a = bfield_elem!(rands[i - 1] as u128);
            let b = bfield_elem!(rands[i] as u128);
            let ab = a * b;
            let a_o_b = a / b;
            let b_o_a = b / a;
            assert_eq!(a, ab / b);
            assert_eq!(b, ab / a);
            assert_eq!(a, a_o_b * b);
            assert_eq!(b, b_o_a * a);

            assert_eq!(a - b + b, a);
            assert_eq!(b - a + a, b);
            assert!((a - a).is_zero());
            assert!((b - b).is_zero());
        }
    }

    #[test]
    fn create_polynomial_test() {
        let a: Polynomial<BFieldElement> =
            Polynomial::new(vec![bfield_elem!(1), bfield_elem!(3), bfield_elem!(7)]);

        let b: Polynomial<BFieldElement> = Polynomial::new(vec![
            bfield_elem!(2),
            bfield_elem!(5),
            bfield_elem!(BFieldElement::MAX),
        ]);

        let expected: Polynomial<BFieldElement> =
            Polynomial::new(vec![bfield_elem!(3), bfield_elem!(8), bfield_elem!(6)]);

        assert_eq!(expected, a + b);
    }

    #[test]
    fn b_field_xgcd_test() {
        let a = 15;
        let b = 25;
        let expected_gcd_ab = 5;
        let (actual_gcd_ab, a_factor, b_factor) = BFieldElement::xgcd(a, b);

        assert_eq!(expected_gcd_ab, actual_gcd_ab);
        assert_eq!(2, a_factor);
        assert_eq!(-1, b_factor);
        assert_eq!(expected_gcd_ab, a_factor * a + b_factor * b);
    }
}
