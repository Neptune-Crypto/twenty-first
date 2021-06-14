use crate::utils::FIRST_THOUSAND_PRIMES;
use serde::{Deserialize, Serialize, Serializer};
use std::fmt;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Rem;
use std::ops::Sub;

#[cfg_attr(feature = "serialization-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PrimeField {
    pub q: i128,
}

impl PrimeField {
    pub fn new(q: i128) -> Self {
        Self { q }
    }

    pub fn batch_inversion(&self, input: Vec<i128>) -> Vec<i128> {
        let size = input.len();
        if size == 0 {
            return Vec::<i128>::new();
        }

        let mut partials0: Vec<i128> = vec![0i128; size];
        let mut result: Vec<i128> = vec![0i128; size];
        partials0[0] = input[0];
        for i in 1..size {
            partials0[i] = partials0[i - 1] * input[i] % self.q;
        }

        // Invert the last element of the `partials` vector
        let (_, inv, _) = PrimeFieldElement::eea(partials0[size - 1], self.q);
        result[size - 1] = (inv % self.q + self.q) % self.q;
        for i in 1..size {
            result[size - i - 1] = result[size - i] * input[size - i] % self.q;
            if size - i - 1 == 0 {}
        }

        for i in 1..size {
            result[i] = result[i] * partials0[i - 1] % self.q;
        }

        result
    }

    pub fn get_primitive_root_of_unity(&self, n: i128) -> (Option<PrimeFieldElement>, Vec<i128>) {
        let mut primes: Vec<i128> = vec![];

        if n <= 1 {
            return (Some(PrimeFieldElement::new(1, self)), primes);
        }

        // Calculate prime factorization of n
        // Check if n = 2^k
        if n & (n - 1) == 0 {
            primes = vec![2];
        } else {
            let mut m = n;
            for prime in FIRST_THOUSAND_PRIMES.iter() {
                if m == 1 {
                    break;
                }
                if m % prime == 0 {
                    primes.push(*prime);
                    while m % prime == 0 {
                        m /= prime;
                    }
                }
            }
            // This might be prohibitively expensive
            if m > 1 {
                let mut other_primes = PrimeFieldElement::primes_lt(m)
                    .into_iter()
                    .filter(|&x| n % x == 0)
                    .collect();
                primes.append(&mut other_primes);
            }
        };

        // N must divide the field prime minus one for a primitive nth root of unity to exist
        if (self.q - 1) % n != 0 {
            return (None, primes);
        }

        let mut primitive_root: Option<PrimeFieldElement> = None;
        let mut candidate: PrimeFieldElement = PrimeFieldElement::new(1, &self);
        #[allow(clippy::suspicious_operation_groupings)]
        while primitive_root == None && candidate.value < self.q {
            if candidate.legendre_symbol() == -1
                && primes
                    .iter()
                    .filter(|&x| n % x == 0)
                    .all(|x| candidate.mod_pow_raw((self.q - 1) / x) != 1)
            {
                primitive_root = Some(candidate.mod_pow((self.q - 1) / n));
            }

            candidate.value += 1;
        }

        (primitive_root, primes)
    }
}

#[cfg_attr(
    feature = "serialization-serde",
    derive(Serialize, Deserialize, Serializer)
)]
#[derive(Debug, Clone, PartialEq, Copy, Serialize)]
pub struct PrimeFieldElement<'a> {
    pub value: i128,
    pub field: &'a PrimeField,
}

impl fmt::Display for PrimeFieldElement<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} mod {}", self.value, self.field.q)
    }
}

impl<'a> PrimeFieldElement<'a> {
    // is_prime and primes_lt have been shamelessly stolen from
    // https://gist.github.com/glebm/440bbe2fc95e7abee40eb260ec82f85c
    pub fn is_prime(n: i128, primes: &[i128]) -> bool {
        for &p in primes {
            let q = n / p;
            if q < p {
                return true;
            };
            let r = n - q * p;
            if r == 0 {
                return false;
            };
        }
        panic!("too few primes")
    }

    fn primes_lt(bound: i128) -> Vec<i128> {
        let mut primes: Vec<bool> = (0..bound + 1).map(|num| num == 2 || num & 1 != 0).collect();
        let mut num = 3i128;
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
            .filter_map(|(i, p)| if p { Some(i as i128) } else { None })
            .collect::<Vec<i128>>()
    }

    pub fn new(value: i128, field: &'a PrimeField) -> Self {
        Self {
            value: (value % field.q + field.q) % field.q,
            field,
        }
    }

    pub fn legendre_symbol(&self) -> i128 {
        let elem = self.mod_pow((self.field.q - 1) / 2).value;

        // Ugly hack to force a result in {-1,0,1}
        if elem == self.field.q - 1 {
            -1
        } else {
            elem
        }
    }

    pub fn from_bytes_raw(modulus: &i128, buf: &[u8]) -> i128 {
        let mut output_int = 0i128;
        for elem in buf.iter() {
            output_int = output_int << 8 ^ *elem as i128;
        }
        output_int % modulus
    }

    fn same_field_check(&self, other: &PrimeFieldElement, operation: &str) {
        if self.field.q != other.field.q {
            panic!(
                "Operation {} is only defined for elements in the same field. Got: q={}, p={}",
                operation, self.field.q, self.field.q
            );
        }
    }

    // Return the greatest common divisor (gcd), and factors a, b s.t. x*a + b*y = gcd(a, b).
    pub fn eea<T: num_traits::Num + Clone + Copy>(mut x: T, mut y: T) -> (T, T, T) {
        let (mut a_factor, mut a1, mut b_factor, mut b1) = (
            num_traits::one(),
            num_traits::zero(),
            num_traits::zero(),
            num_traits::one(),
        );

        while y != num_traits::zero() {
            let (quotient, remainder) = (x / y, x % y);
            let (c, d) = (a_factor - quotient * a1, b_factor - quotient * b1);

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

    pub fn inv(&self) -> Self {
        let (_, _, a) = Self::eea(self.field.q, self.value);
        Self {
            field: self.field,
            value: (a % self.field.q + self.field.q) % self.field.q,
        }
    }

    pub fn mod_pow_raw(&self, pow: i128) -> i128 {
        let mut acc: i128 = 1;
        let mod_value: i128 = self.field.q;
        let res = self.value;

        for i in 0..128 {
            acc = acc * acc % mod_value;
            let set: bool = pow & (1 << (128 - 1 - i)) != 0;
            if set {
                acc = acc * res % mod_value;
            }
        }
        acc
    }

    pub fn mod_pow(&self, pow: i128) -> Self {
        Self {
            value: self.mod_pow_raw(pow),
            field: self.field,
        }
    }
}

impl<'a> Add for PrimeFieldElement<'a> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.same_field_check(&other, "add");
        Self {
            value: (self.value + other.value) % self.field.q,
            field: self.field,
        }
    }
}

impl<'a> Sub for PrimeFieldElement<'a> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.same_field_check(&other, "sub");
        Self {
            value: ((self.value - other.value) % self.field.q + self.field.q) % self.field.q,
            field: self.field,
        }
    }
}

impl<'a> Mul for PrimeFieldElement<'a> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.same_field_check(&other, "mul");
        Self {
            value: self.value * other.value % self.field.q,
            field: self.field,
        }
    }
}

impl<'a> Div for PrimeFieldElement<'a> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.same_field_check(&other, "div");
        Self {
            value: other.inv().value * self.value % self.field.q,
            field: self.field,
        }
    }
}

impl<'a> Rem for PrimeFieldElement<'a> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        self.same_field_check(&other, "rem");
        Self {
            value: 0,
            field: self.field,
        }
    }
}

// p = k*n+1 = 2^32 − 2^20 + 1 = 4293918721
// p-1=2^20*3^2*5*7*13.

#[cfg(test)]
mod test_modular_arithmetic {
    #![allow(clippy::just_underscores_and_digits)]
    use super::*;
    use crate::utils::generate_random_numbers;

    #[test]
    fn batch_inversion_test_small() {
        let input: Vec<i128> = vec![1, 2, 3, 4];
        let field = PrimeField::new(5);
        let output = field.batch_inversion(input);
        assert_eq!(vec![1, 3, 2, 4], output);
    }

    #[test]
    fn batch_inversion_test_bigger() {
        let input: Vec<i128> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        ];
        let field = PrimeField::new(23);
        let output = field.batch_inversion(input);
        assert_eq!(
            vec![1, 12, 8, 6, 14, 4, 10, 3, 18, 7, 21, 2, 16, 5, 20, 13, 19, 9, 17, 15, 11, 22],
            output
        );
    }

    #[test]
    fn sieve_of_eratosthenes() {
        // Find primes below 100
        let expected: Vec<i128> = vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];
        let primes = PrimeFieldElement::primes_lt(100);
        for i in 0..primes.len() {
            assert_eq!(expected[i], primes[i]);
        }
        for i in 2..primes.len() {
            assert!(PrimeFieldElement::is_prime(expected[i], &primes[0..i]));
        }
        println!("sieve successful");
    }

    #[test]
    fn roots_of_unity() {
        let mut a = PrimeField::new(17);
        for i in 2..a.q {
            let (b, _) = a.get_primitive_root_of_unity(i);
            if i == 2 || i == 4 || i == 8 || i == 16 {
                assert_ne!(b, None);
            } else {
                assert_eq!(b, None);
            }
        }

        a = PrimeField::new(41);
        let mut count = 0;
        for i in 2..a.q {
            let (b, _) = a.get_primitive_root_of_unity(i);
            match b {
                None => {}
                _ => {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 7);

        a = PrimeField::new(761);
        let (b, _) = a.get_primitive_root_of_unity(40);
        assert_eq!(208, b.unwrap().value);
        let (b, _) = a.get_primitive_root_of_unity(760);
        assert_eq!(6, b.unwrap().value);
    }

    #[test]
    fn primitive_root_of_unity_mod_7() {
        let field = PrimeField::new(7);
        let (primitive_root, prime_factors) = field.get_primitive_root_of_unity(2);
        assert_eq!(6, primitive_root.unwrap().value);
        assert_eq!(vec![2], prime_factors);
        let (primitive_root, prime_factors) = field.get_primitive_root_of_unity(3);
        assert_eq!(2, primitive_root.unwrap().value);
        assert_eq!(vec![3], prime_factors);
        let (primitive_root, prime_factors) = field.get_primitive_root_of_unity(4);
        assert_eq!(None, primitive_root);
        assert_eq!(vec![2], prime_factors);
        let (primitive_root, prime_factors) = field.get_primitive_root_of_unity(5);
        assert_eq!(None, primitive_root);
        assert_eq!(vec![5], prime_factors);
    }

    // Test vector found in https://www.vitalik.ca/general/2018/07/21/starks_part_3.html
    #[test]
    fn find_16th_root_of_unity_mod_337() {
        let field = PrimeField::new(337);
        let (primitive_root, prime_factors) = field.get_primitive_root_of_unity(16);
        assert!(
            vec![59i128, 146, 30, 297, 278, 191, 307, 40].contains(&primitive_root.unwrap().value)
        );
        assert_eq!(vec![2], prime_factors);
    }

    #[test]
    fn find_40th_root_of_unity_mod_761() {
        let field = PrimeField::new(761);
        let (primitive_root, prime_factors) = field.get_primitive_root_of_unity(40);
        println!("Found: {}", primitive_root.unwrap());
        assert_eq!(208, primitive_root.unwrap().value);
        assert_eq!(vec![2, 5], prime_factors);
    }

    #[test]
    fn primitive_root_property_based_test() {
        let primes = vec![773i128, 13367, 223, 379, 41];
        for prime in primes.iter() {
            // println!("Testing prime {}", prime);
            let field = PrimeField::new(*prime);
            let rands =
                generate_random_numbers(30, if *prime > 1000000 { 1000000 } else { *prime });
            for elem in rands.iter() {
                // println!("elem = {}", *elem);
                let (root, prime_factors) = field.get_primitive_root_of_unity(*elem);
                assert!(prime_factors.iter().all(|&x| *elem % x == 0));
                if *elem == 0 {
                    continue;
                }

                // verify that we can build *elem from prime_factors
                let mut m = *elem;
                for prime in prime_factors.clone() {
                    while m % prime == 0 {
                        m /= prime;
                    }
                }
                assert_eq!(1, m);

                match root {
                    None => (),
                    Some(i) => {
                        // println!(
                        //     "Found primitive root: {}^{} = 1 mod {}",
                        //     i.value, *elem, *prime
                        // );

                        // Verify that i is actually a root of unity
                        assert_eq!(1, i.mod_pow_raw(*elem));

                        // Verify that root is actually primitive
                        for prime_factor in prime_factors.clone().iter() {
                            assert_ne!(1, i.mod_pow_raw(*elem / prime_factor));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn internal() {
        // Test addition, subtraction, multiplication, and division
        let _1931 = PrimeField::new(1931);
        let _899_1931 = PrimeFieldElement::new(899, &_1931);
        let _31_1931 = PrimeFieldElement::new(31, &_1931);
        let _1921_1931 = PrimeFieldElement::new(1921, &_1931);
        assert_eq!((_899_1931 + _31_1931).value, 930);
        assert_eq!((_899_1931 + _1921_1931).value, 889);
        assert_eq!((_899_1931 - _31_1931).value, 868);
        assert_eq!((_899_1931 - _1921_1931).value, 909);
        assert_eq!((_899_1931 * _31_1931).value, 835);
        assert_eq!((_899_1931 * _1921_1931).value, 665);
        assert_eq!((_899_1931 / _31_1931).value, 29);
        assert_eq!((_899_1931 / _1921_1931).value, 1648);

        let field_19 = PrimeField { q: 19 };
        let field_23 = PrimeField { q: 23 };
        let _3_19 = PrimeFieldElement {
            value: 3,
            field: &field_19,
        };
        let _5_19 = PrimeFieldElement {
            value: 5,
            field: &field_19,
        };
        assert_eq!(_3_19.inv(), PrimeFieldElement::new(13, &field_19));
        assert_eq!(
            PrimeFieldElement::new(13, &field_19).mul(PrimeFieldElement {
                value: 5,
                field: &field_19
            }),
            PrimeFieldElement::new(8, &field_19)
        );
        assert_eq!(
            PrimeFieldElement::new(13, &field_19)
                * PrimeFieldElement {
                    value: 5,
                    field: &field_19
                },
            PrimeFieldElement::new(8, &field_19)
        );
        assert_eq!(
            PrimeFieldElement::new(13, &field_19)
                + PrimeFieldElement {
                    value: 5,
                    field: &field_19
                },
            PrimeFieldElement::new(18, &field_19)
        );

        assert_eq!(
            PrimeFieldElement::new(13, &field_19)
                + PrimeFieldElement {
                    value: 13,
                    field: &field_19
                },
            PrimeFieldElement::new(7, &field_19)
        );
        assert_eq!(
            PrimeFieldElement::new(2, &field_23).inv(),
            PrimeFieldElement::new(12, &field_23)
        );

        // Test EAC (Extended Euclidean Algorithm/Extended Greatest Common Divider)
        let (gcd, a_factor, b_factor) = PrimeFieldElement::eea(1914, 899);
        assert_eq!(8, a_factor);
        assert_eq!(-17, b_factor);
        assert_eq!(29, gcd);

        // test mod_pow
        let _1914 = PrimeField::new(1914);
        let _899_1914 = PrimeFieldElement::new(899, &_1914);
        assert_eq!(_899_1914.value, 899);
        assert_eq!(_899_1914.mod_pow(2).value, 493);
        assert_eq!(_899_1914.mod_pow(3).value, 1073);
        assert_eq!(_899_1914.mod_pow(4).value, 1885);
        assert_eq!(_899_1914.mod_pow(5).value, 725);
        assert_eq!(_899_1914.mod_pow(6).value, 1015);
        assert_eq!(_899_1914.mod_pow(7).value, 1421);
        assert_eq!(_899_1914.mod_pow(8).value, 841);
        assert_eq!(_899_1914.mod_pow(9).value, 29);

        // Now with an actual prime
        assert_eq!(_899_1931.value, 899);
        assert_eq!(_899_1931.mod_pow(1).value, 899);
        assert_eq!(_899_1931.mod_pow(2).value, 1043);
        assert_eq!(_899_1931.mod_pow(3).value, 1122);
        assert_eq!(_899_1931.mod_pow(4).value, 696);
        assert_eq!(_899_1931.mod_pow(5).value, 60);
        assert_eq!(_899_1931.mod_pow(6).value, 1803);
        assert_eq!(_899_1931.mod_pow(7).value, 788);
        assert_eq!(_899_1931.mod_pow(8).value, 1666);
        assert_eq!(_899_1931.mod_pow(9).value, 1209);
        assert_eq!(_899_1931.mod_pow(10).value, 1669);
        assert_eq!(_899_1931.mod_pow(11).value, 44);
        assert_eq!(_899_1931.mod_pow(12).value, 936);
        assert_eq!(_899_1931.mod_pow(13).value, 1479);
        assert_eq!(_899_1931.mod_pow(14).value, 1093);
        assert_eq!(_899_1931.mod_pow(15).value, 1659);
        assert_eq!(_899_1931.mod_pow(1930).value, 1); // Fermat's Little Theorem

        // Test the inverse function
        println!("{}", _899_1931.inv().value);
        assert_eq!(_899_1931.inv().value, 784);
        assert_eq!(
            _899_1931 * PrimeFieldElement::new(784, &_1931),
            PrimeFieldElement::new(1, &_1931)
        );
    }
}
