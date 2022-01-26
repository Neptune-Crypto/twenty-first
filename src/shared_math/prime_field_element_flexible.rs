use crate::shared_math::traits::{
    CyclicGroupGenerator, FieldBatchInversion, FromVecu8, GetPrimitiveRootOfUnity, IdentityValues,
    ModPowU32, ModPowU64, New, PrimeFieldElement,
};
use crate::utils::FIRST_TEN_THOUSAND_PRIMES;
use num_bigint::{BigInt, Sign};
use num_traits::{One, Zero};
use parity_scale_codec::Encode;
use primitive_types::{U256, U512};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub fn get_prime_with_primitive_root_of_unity(
    n: u128,
    min_value: u128,
) -> Option<PrimeFieldElementFlexible> {
    for prime in FIRST_TEN_THOUSAND_PRIMES
        .iter()
        .filter(|&x| *x as u128 > min_value)
    {
        let field_element = PrimeFieldElementFlexible::new(1.into(), (*prime).into());
        if let (Some(j), _) = field_element.get_primitive_root_of_unity(n) {
            return Some(j);
        }
    }

    None
}

// Can only be used to represent up to 256 bits numbers, I think. Because mul.
// If bigger primes are needed, you could convert to BigInt when needed
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PrimeFieldElementFlexible {
    pub q: U512,
    pub value: U512,
}

// Implemented by just displaying the value. If you want the prime, you should call
// the Debug print with `{:?}`.
impl Display for PrimeFieldElementFlexible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl New for PrimeFieldElementFlexible {
    fn new_from_usize(&self, value: usize) -> Self {
        Self {
            value: value.into(),
            q: self.q,
        }
    }
}

impl CyclicGroupGenerator for PrimeFieldElementFlexible {
    fn get_cyclic_group_elements(&self, max: Option<usize>) -> Vec<Self> {
        let mut val = *self;
        let mut ret: Vec<Self> = vec![self.ring_one()];

        loop {
            ret.push(val);
            val *= *self;
            if val.is_one() || max.is_some() && ret.len() >= max.unwrap() {
                break;
            }
        }
        ret
    }
}

impl FieldBatchInversion for PrimeFieldElementFlexible {
    fn batch_inversion(elements: Vec<Self>) -> Vec<Self> {
        // Adapted from https://paulmillr.com/posts/noble-secp256k1-fast-ecc/#batch-inversion
        let input_length = elements.len();
        if input_length == 0 {
            return Vec::<Self>::new();
        }

        let mut scratch: Vec<Self> = vec![elements[0].ring_zero(); input_length];
        let mut acc = elements[0].ring_one();
        scratch[0] = elements[0];

        for i in 0..input_length {
            assert!(!elements[i].is_zero(), "Cannot do batch inversion on zero");
            scratch[i] = acc;
            acc *= elements[i];
        }

        acc = acc.inv();

        let mut res = elements;
        for i in (0..input_length).rev() {
            let tmp = acc * res[i];
            res[i] = acc * scratch[i];
            acc = tmp;
        }

        res
    }
}

impl ModPowU32 for PrimeFieldElementFlexible {
    fn mod_pow_u32(&self, exp: u32) -> Self {
        // TODO: This can be sped up by a factor 2 by implementing
        // it for u32 and not using the 64-bit version
        self.mod_pow(exp.into())
    }
}

impl ModPowU64 for PrimeFieldElementFlexible {
    fn mod_pow_u64(&self, exp: u64) -> Self {
        // TODO: This can be sped up by a factor 2 by implementing
        // it for u32 and not using the 64-bit version
        self.mod_pow(exp.into())
    }
}

impl<'de> Deserialize<'de> for PrimeFieldElementFlexible {
    // TODO: Implement, somehow
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // deserializer.deserialize_bytes(visitor)
        // Ok(Self {
        //     q: 1.into(),
        //     value: 1.into(),
        // })
        todo!()
    }
}

impl Serialize for PrimeFieldElementFlexible {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut bytes: Vec<u8> = Vec::with_capacity(128);
        let mut q_bytes = self.q.encode();
        let mut value_bytes = self.value.encode();
        bytes.append(&mut q_bytes);
        bytes.append(&mut value_bytes);
        serializer.serialize_bytes(&bytes)
    }
}

impl PrimeFieldElementFlexible {
    fn u512_to_bigint(value: U512) -> BigInt {
        let mut val_bytes: Vec<u8> = vec![0; 64];
        value.to_little_endian(&mut val_bytes);
        BigInt::from_bytes_le(Sign::Plus, &val_bytes)
    }

    fn bigint_to_u512(value: BigInt) -> Option<U512> {
        let (sign, bytes) = value.to_bytes_be();
        match sign {
            Sign::Minus => None,
            Sign::Plus => Some(U512::from_big_endian(&bytes)),
            Sign::NoSign => Some(U512::zero()),
        }
    }

    pub fn new(value: U256, q: U256) -> Self {
        Self {
            q: q.into(),
            value: value.into(),
        }
    }

    pub fn new_from_u512(value: U512, q: U512) -> Self {
        Self { q, value }
    }

    pub fn inv(&self) -> Self {
        let mut q_bytes: Vec<u8> = vec![0; 64];
        self.q.to_little_endian(&mut q_bytes);
        let mut val_bytes: Vec<u8> = vec![0; 64];
        self.value.to_little_endian(&mut val_bytes);
        let q_bigint: BigInt = BigInt::from_bytes_le(Sign::Plus, &q_bytes);
        let value_bigint: BigInt = BigInt::from_bytes_le(Sign::Plus, &val_bytes);

        let (_, _, a) = Self::xgcd(q_bigint, value_bigint);
        let (sign, bytes) = a.to_bytes_be();
        let a_u512 = match sign {
            Sign::Minus => self.q - U512::from_big_endian(&bytes),
            Sign::Plus => U512::from_big_endian(&bytes),
            Sign::NoSign => U512::from_big_endian(&bytes),
        };
        Self {
            value: (a_u512 % self.q + self.q) % self.q,
            q: self.q,
        }
    }

    pub fn mod_pow(&self, pow: BigInt) -> Self {
        // Special case for handling 0^0 = 1
        if pow.is_zero() {
            return self.ring_one();
        }

        let mut acc: BigInt = BigInt::one();
        let mod_value: BigInt = Self::u512_to_bigint(self.q);
        let res = Self::u512_to_bigint(self.value);

        let bit_length: u64 = pow.bits();

        for i in 0..bit_length {
            acc = acc.clone() * acc.clone() % mod_value.clone();
            let set: bool =
                !(pow.clone() & Into::<BigInt>::into(1u128 << (bit_length - 1 - i))).is_zero();
            if set {
                acc = acc * res.clone() % mod_value.clone();
            }
        }

        Self {
            q: self.q,
            value: Self::bigint_to_u512(acc).unwrap(),
        }
    }

    pub fn increment(&mut self) {
        self.value = (self.value + 1) % self.q;
    }

    pub fn decrement(&mut self) {
        self.value = (self.q - 1 + self.value) % self.q;
    }

    pub fn xgcd(mut x: BigInt, mut y: BigInt) -> (BigInt, BigInt, BigInt) {
        let (mut a_factor, mut a1, mut b_factor, mut b1) =
            (BigInt::one(), BigInt::zero(), BigInt::zero(), BigInt::one());

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

    // TODO: Consider converting to different input
    pub fn is_prime(n: BigInt, primes: &[BigInt]) -> bool {
        for p in primes {
            let q = n.clone() / p.to_owned();
            if q.clone() < *p {
                return true;
            };
            // let r: i128 = n - q * p;
            let r: BigInt = n.clone() - q.clone() * p.to_owned();
            if r.is_zero() {
                return false;
            };
        }
        panic!("too few primes")
    }

    // Not converted to BigInt since it is infeasible to find all primes below i128.max
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

    fn legendre_symbol(&self) -> BigInt {
        let one = BigInt::one();
        let q_minus_one: BigInt = Self::u512_to_bigint(self.q - 1);
        let two: BigInt = 2.into();
        let elem = self.mod_pow(q_minus_one / two);
        let q_big_int: BigInt = Self::u512_to_bigint(self.q);
        let elem_value_bigint: BigInt = Self::u512_to_bigint(elem.value);

        // Ugly hack to force a result in {-1,0,1}
        if elem_value_bigint == q_big_int - one.clone() {
            -one
        } else {
            elem_value_bigint
        }
    }
}

impl Add for PrimeFieldElementFlexible {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            q: self.q,
            value: (self.value + rhs.value) % self.q,
        }
    }
}

impl Sub for PrimeFieldElementFlexible {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        -rhs + self
    }
}

impl AddAssign for PrimeFieldElementFlexible {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value = (self.value + rhs.value) % self.q;
    }
}

impl SubAssign for PrimeFieldElementFlexible {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value = (self.q - rhs.value + self.value) % self.q;
    }
}

impl MulAssign for PrimeFieldElementFlexible {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.value = (self.value * rhs.value) % self.q;
    }
}

impl Mul for PrimeFieldElementFlexible {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            value: (self.value * other.value) % self.q,
            q: self.q,
        }
    }
}

impl Neg for PrimeFieldElementFlexible {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.q - self.value,
            q: self.q,
        }
    }
}

impl<'a> Div for PrimeFieldElementFlexible {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self {
            value: other.inv().value * self.value % self.q,
            q: self.q,
        }
    }
}

impl PrimeFieldElement for PrimeFieldElementFlexible {
    type Elem = PrimeFieldElementFlexible;
}

impl FromVecu8 for PrimeFieldElementFlexible {
    fn from_vecu8(&self, bytes: Vec<u8>) -> Self {
        // Beware! This function can only generate values between 0 and 2^256.
        // If you want this function to be able to create values up to 2^512,
        // then it would need to read 64 bytes instead of 32 as now. And if
        // feeding it a hash, it would need to be fed a 64 byte hash.
        let bytesize = std::mem::size_of::<u64>();
        let (first_eight_bytes, rest) = bytes.as_slice().split_at(bytesize);
        let (second_eight_bytes, rest) = rest.split_at(bytesize);
        let (third_eight_bytes, rest) = rest.split_at(bytesize);
        let (fourth_eight_bytes, _rest) = rest.split_at(bytesize);
        let mut u256_bytes: Vec<u8> = vec![];
        u256_bytes.extend_from_slice(first_eight_bytes);
        u256_bytes.extend_from_slice(second_eight_bytes);
        u256_bytes.extend_from_slice(third_eight_bytes);
        u256_bytes.extend_from_slice(fourth_eight_bytes);

        let val_u512 = U512::from_big_endian(&u256_bytes);

        PrimeFieldElementFlexible {
            value: val_u512 % self.q,
            q: self.q,
        }
    }
}

impl IdentityValues for PrimeFieldElementFlexible {
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        (self.value - 1).is_zero()
    }

    fn ring_zero(&self) -> Self {
        Self {
            q: self.q,
            value: U512::zero(),
        }
    }

    fn ring_one(&self) -> Self {
        Self {
            q: self.q,
            value: U512::one(),
        }
    }
}

impl GetPrimitiveRootOfUnity for PrimeFieldElementFlexible {
    fn get_primitive_root_of_unity(&self, n: u128) -> (Option<Self>, Vec<u128>) {
        let mut primes: Vec<u128> = vec![];
        // let q_u256: U256 = From::<U256>::from(self.q);
        // let q_as_u256: U256 = self.q.into();

        if n <= 1 {
            return (Some(self.ring_one()), primes);
        }

        // Calculate prime factorization of n
        // Check if n = 2^k
        if n & (n - 1) == 0 {
            primes = vec![2];
        } else {
            let mut m = n;
            for prime in FIRST_TEN_THOUSAND_PRIMES.iter().map(|&p| p as u128) {
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
                let mut other_primes = PrimeFieldElementFlexible::primes_lt(m)
                    .into_iter()
                    .filter(|&x| n % x == 0)
                    .collect();
                primes.append(&mut other_primes);
            }
        };

        // N must divide the field prime minus one for a primitive nth root of unity to exist
        if !((self.q - 1) % n).is_zero() {
            return (None, primes);
        }

        let mut primitive_root: Option<PrimeFieldElementFlexible> = None;
        let mut candidate: PrimeFieldElementFlexible = self.ring_one();
        let q_minus_one: BigInt = Self::u512_to_bigint(self.q - 1);
        #[allow(clippy::suspicious_operation_groupings)]
        while primitive_root == None && candidate.value < self.q {
            if (-candidate.legendre_symbol()).is_one()
                && primes.iter().filter(|&x| n % x == 0).all(|x| {
                    let x_bi: BigInt = (*x).into();
                    !candidate.mod_pow(q_minus_one.clone() / x_bi).is_one()
                })
            {
                let n_bi: BigInt = n.into();
                primitive_root = Some(candidate.mod_pow(q_minus_one.clone() / n_bi));
            }

            candidate.increment();
        }

        (primitive_root, primes)
    }
}

#[cfg(test)]
mod test_modular_arithmetic_flexible {
    #![allow(clippy::just_underscores_and_digits)]
    use std::iter::FromIterator;

    use crate::utils::generate_random_numbers;

    use super::*;

    fn values(prime: u128, values: &[u128]) -> Vec<PrimeFieldElementFlexible> {
        values
            .into_iter()
            .map(|x| PrimeFieldElementFlexible {
                q: prime.into(),
                value: (*x).into(),
            })
            .collect()
    }

    fn big_integers(xs: Vec<u128>) -> Vec<BigInt> {
        xs.iter()
            .map(|x| Into::<BigInt>::into(*x))
            .collect::<Vec<BigInt>>()
    }

    #[test]
    fn arithmetic_flexible_test() {
        // Test addition, subtraction, multiplication, and division
        let prime: U256 = 1931.into();
        let _899_1931 = PrimeFieldElementFlexible::new(899.into(), prime);
        let _31_1931 = PrimeFieldElementFlexible::new(31.into(), prime);
        let _1921_1931 = PrimeFieldElementFlexible::new(1921.into(), prime);
        assert_eq!((_899_1931 + _31_1931).value, 930.into());
        assert_eq!((_899_1931 + _1921_1931).value, 889.into());
        assert_eq!((_899_1931 - _31_1931).value, 868.into());
        assert_eq!((_899_1931 - _1921_1931).value, 909.into());
        assert_eq!((_899_1931 * _31_1931).value, 835.into());
        assert_eq!((_899_1931 * _1921_1931).value, 665.into());
        assert_eq!((_899_1931 / _31_1931).value, 29.into());
        assert_eq!((_899_1931 / _1921_1931).value, 1648.into());
    }

    #[test]
    fn batch_inversion_test_small_no_zeros() {
        let input: Vec<PrimeFieldElementFlexible> = values(5, &vec![1, 2, 3, 4]);
        let output = PrimeFieldElementFlexible::batch_inversion(input);
        let expected_output: Vec<PrimeFieldElementFlexible> = values(5, &vec![1, 3, 2, 4]);
        assert_eq!(expected_output, output);
    }

    #[test]
    #[should_panic]
    fn batch_inversion_test_small_with_zeros() {
        let input: Vec<PrimeFieldElementFlexible> = values(5, &vec![1, 2, 3, 4, 0]);
        PrimeFieldElementFlexible::batch_inversion(input);
    }

    #[test]
    fn batch_inversion_flexible_test_bigger() {
        let input: Vec<PrimeFieldElementFlexible> = values(23, &Vec::from_iter(1..=22));
        let output = PrimeFieldElementFlexible::batch_inversion(input);
        let expected_output: Vec<PrimeFieldElementFlexible> = values(
            23,
            &vec![
                1, 12, 8, 6, 14, 4, 10, 3, 18, 7, 21, 2, 16, 5, 20, 13, 19, 9, 17, 15, 11, 22,
            ],
        );
        assert_eq!(expected_output, output);
    }

    #[test]
    fn sieve_of_eratosthenes() {
        // Find primes below 100
        let expected: Vec<u128> = vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];
        let primes = PrimeFieldElementFlexible::primes_lt(100);
        let bi_primes = big_integers(primes.clone());
        assert_eq!(expected, primes);
        for i in 2..primes.len() {
            assert!(PrimeFieldElementFlexible::is_prime(
                expected[i].into(),
                &bi_primes[0..i]
            ));
        }
    }

    #[test]
    fn get_cyclic_group_elements_test() {
        let value = PrimeFieldElementFlexible::new(40.into(), 113.into());
        let power_series = value.get_cyclic_group_elements(None);
        assert_eq!(value.ring_one(), power_series.first().unwrap().to_owned());
        assert_eq!(value.new_from_usize(40), power_series[1]);
        assert_eq!(value.new_from_usize(18), power_series[2]);
        assert_eq!(value.new_from_usize(42), power_series[3]);
        assert_eq!(value.new_from_usize(98), power_series[4]);
        assert_eq!(value.new_from_usize(48), power_series[7]);
    }

    #[test]
    fn get_primitive_roots_of_unity_12289() {
        let one = PrimeFieldElementFlexible::new(1.into(), 12289.into());
        let root = one.get_primitive_root_of_unity(1024).0.unwrap();
        assert!(root.mod_pow(1024.into()).is_one());
        assert!(!root.mod_pow(512.into()).is_one());

        let _8193 = one.new_from_usize(8193);
        assert!(_8193.mod_pow(1024.into()).is_one());
        assert!(_8193.mod_pow(512.into()).is_one());
        assert!(!_8193.mod_pow(256.into()).is_one());
    }

    #[test]
    fn roots_of_unity_big() {
        let mut prime = 17;
        let one_17 = PrimeFieldElementFlexible::new(1.into(), prime.into());
        for i in 2..17 {
            let (b, _) = one_17.get_primitive_root_of_unity(i);
            if i == 2 || i == 4 || i == 8 || i == 16 {
                assert_ne!(b, None);
            } else {
                assert_eq!(b, None);
            }
        }

        prime = 41;
        let one_41 = PrimeFieldElementFlexible::new(1.into(), prime.into());
        let mut count = 0;
        for i in 2..41 {
            let (b, _) = one_41.get_primitive_root_of_unity(i);
            match b {
                None => {}
                _ => {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 7);

        prime = 761;
        let one_761 = PrimeFieldElementFlexible::new(1.into(), prime.into());
        let (c, _) = one_761.get_primitive_root_of_unity(40);
        let _208: U512 = 208.into();
        assert_eq!(_208, c.unwrap().value);
        let (c, _) = one_761.get_primitive_root_of_unity(760);
        let _6: U512 = 6.into();
        assert_eq!(_6, c.unwrap().value);
    }

    #[test]
    fn arithmetic_pointer_equals() {
        // build the same prime field element twice, but with pointers to the different objects
        let e1 = PrimeFieldElementFlexible::new(16.into(), 101.into());
        let e2 = PrimeFieldElementFlexible::new(16.into(), 101.into());

        // test for equality
        assert!(e1 == e2); // test passes => equality test folows all pointers to test end-values
    }

    #[test]
    fn get_prime_with_primitive_root_of_unity_test() {
        let a = get_prime_with_primitive_root_of_unity(17, 300).unwrap();
        let _307: U512 = 307.into();
        assert_eq!(_307, a.q);
        let _273: U512 = 273.into();
        assert_eq!(_273, a.value);
        assert!(a.mod_pow(17.into()).is_one());
    }

    #[test]
    fn primitive_root_property_based_test() {
        let primes = vec![773i128, 13367, 223, 379, 41];
        for i in 0..primes.len() {
            // println!("Testing prime {}", prime);
            let rands: Vec<i128> = generate_random_numbers(
                30,
                if primes[i] > 1000000 {
                    1000000
                } else {
                    primes[i]
                },
            );
            let one = PrimeFieldElementFlexible::new(1.into(), primes[i].into());
            for elem in rands.iter() {
                let (root, prime_factors): (Option<PrimeFieldElementFlexible>, Vec<u128>) =
                    one.get_primitive_root_of_unity(*elem as u128);
                assert!(prime_factors
                    .clone()
                    .into_iter()
                    .all(|x| (*elem as u128 % x).is_zero()));
                if *elem == 0 {
                    continue;
                }

                // verify that we can build *elem from prime_factors
                let mut m: u128 = *elem as u128;
                for prime in prime_factors.iter() {
                    while m % prime == 0 {
                        m = m / prime;
                    }
                }
                assert!(m.is_one());

                let elem_bi: BigInt = (*elem).into();
                match root {
                    None => (),
                    Some(i) => {
                        // println!(
                        //     "Found primitive root: {}^{} = 1 mod {}",
                        //     i.value, *elem, *prime
                        // );

                        // Verify that i is actually a root of unity
                        assert!(i.mod_pow(elem_bi.clone()).is_one());

                        // Verify that root is actually primitive
                        for prime_factor in prime_factors.clone().iter() {
                            let prime_factor_bi: BigInt = (*prime_factor).into();
                            assert!(!i.mod_pow(elem_bi.clone() / prime_factor_bi).is_one());
                        }
                    }
                }
            }
        }
    }
}
