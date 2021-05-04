use std::fmt;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Rem;
use std::ops::Sub;

#[derive(Debug, Clone, PartialEq)]
pub struct PrimeField {
    pub q: i128,
}

impl PrimeField {
    pub fn new(q: i128) -> Self {
        Self { q }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
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
    pub fn new(value: i128, field: &'a PrimeField) -> Self {
        Self {
            value: (value % field.q + field.q) % field.q,
            field,
        }
    }

    pub fn legendre_symbol(&self) -> i128 {
        self.mod_pow((self.field.q - 1) / 2).value
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
    fn eea(mut x: i128, mut y: i128) -> (i128, i128, i128) {
        let (mut a_factor, mut a1, mut b_factor, mut b1) = (1, 0, 0, 1);

        while y != 0 {
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

    pub fn mod_pow(&self, pow: i128) -> Self {
        let mut acc = Self {
            value: 1,
            field: self.field,
        };
        let res = *self;

        for i in 0..128 {
            acc = acc * acc;
            let set: bool = pow & (1 << (128 - 1 - i)) != 0;
            if set {
                acc = acc * res;
            }
        }
        acc
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
