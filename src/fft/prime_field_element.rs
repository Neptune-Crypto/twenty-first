use std::fmt;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Rem;
use std::ops::Sub;

#[derive(Debug, Clone, PartialEq)]
pub struct PrimeField {
    pub q: i64,
}

impl PrimeField {
    pub fn new(q: i64) -> Self {
        Self { q }
    }

    pub fn get_primitive_root_of_unity(&self, n: i64) -> Option<PrimeFieldElement> {
        // Cf. https://www.csd.uwo.ca/~mmorenom/CS874/Lectures/Newton2Hensel.html/node9.html#thrm:PrimitiveRootExistenceCriterium
        // N must divide the field prime minus one for a primitive nth root of unity to exist
        if (self.q - 1) % n != 0 {
            return None;
        }

        let mut candidate_value = 2;
        let mut roots: Vec<i64> = Vec::new();
        let mut field_element: PrimeFieldElement = PrimeFieldElement::new(0, self);
        let mut found: bool = false;
        while candidate_value <= self.q {
            // if candidate_value % 1000 == 0 {
            //     println!("candidate value: {}", candidate_value);
            // }

            field_element = PrimeFieldElement::new(candidate_value, &self);
            let mod_pow = field_element.mod_pow(n as i64);
            if mod_pow.value == 1 {
                roots.push(candidate_value);
                // println!("{} ^ N == 1", candidate_value);
                // println!("Roots: {:?}", roots);

                // candidate is Nth prime. Now check that it is primitive prime
                // cf. this link we must check that for all primes p dividing N that
                // that y^(N/p) != 1.
                // https://en.wikipedia.org/wiki/Root_of_unity_modulo_n#Finding_an_n_with_a_primitive_k-th_root_of_unity_modulo_n
                // find all primes, p_i, less than sqrt(candidate_value) and check, for all p_i
                // candidate_value ^ (n/p_i) != 1
                let bound = n / 2;
                let primes = PrimeFieldElement::primes_lt(bound as i64);
                if primes
                    .iter()
                    .filter(|&x| n % x == 0)
                    .all(|x| field_element.mod_pow_raw(n / x) != 1)
                {
                    println!(
                        "Found {} primitive root: {} of mod {}",
                        n, candidate_value, self.q
                    );
                    found = true;
                    break;
                }
            }
            candidate_value += 1;
        }

        if found {
            Some(field_element)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct PrimeFieldElement<'a> {
    pub value: i64,
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
    pub fn is_prime(n: i64, primes: &[i64]) -> bool {
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

    fn primes_lt(bound: i64) -> Vec<i64> {
        let mut primes: Vec<bool> = (0..bound + 1).map(|num| num == 2 || num & 1 != 0).collect();
        let mut num = 3i64;
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
            .filter_map(|(i, p)| if p { Some(i as i64) } else { None })
            .collect::<Vec<i64>>()
    }

    pub fn new(value: i64, field: &'a PrimeField) -> Self {
        Self {
            value: (value % field.q + field.q) % field.q,
            field,
        }
    }

    pub fn legendre_symbol(&self) -> i64 {
        let elem = self.mod_pow((self.field.q - 1) / 2).value;

        // Ugly hack to force a result in {-1,0,1}
        if elem == self.field.q - 1 {
            -1
        } else {
            elem
        }
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

    pub fn mod_pow_raw(&self, pow: i64) -> i64 {
        let mut acc: i64 = 1;
        let mod_value: i64 = self.field.q;
        let res = self.value;

        for i in 0..64 {
            acc = acc * acc % mod_value;
            let set: bool = pow & (1 << (64 - 1 - i)) != 0;
            if set {
                acc = acc * res % mod_value;
            }
        }
        acc
    }

    pub fn mod_pow(&self, pow: i64) -> Self {
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

    #[test]
    fn sieve_of_eratosthenes() {
        use super::*;

        // Find primes below 100
        let expected: Vec<i64> = vec![
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
        use super::*;

        let mut a = PrimeField::new(17);
        for i in 2..a.q {
            let b = a.get_primitive_root_of_unity(i);
            if i == 2 || i == 4 || i == 8 || i == 16 {
                assert_ne!(b, None);
            } else {
                assert_eq!(b, None);
            }
        }

        a = PrimeField::new(41);
        let mut count = 0;
        for i in 2..a.q {
            let b = a.get_primitive_root_of_unity(i);
            match b {
                None => {}
                _ => {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 7);

        a = PrimeField::new(761);
        let mut b = a.get_primitive_root_of_unity(40).unwrap();
        assert_eq!(35, b.value);
        b = a.get_primitive_root_of_unity(760).unwrap();
        assert_eq!(6, b.value);
    }

    #[test]
    fn internal() {
        use super::*;

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
