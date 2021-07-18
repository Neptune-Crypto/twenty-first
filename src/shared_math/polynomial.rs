use crate::shared_math::traits::IdentityValues;
use crate::utils::has_unique_elements;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use std::convert::From;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Rem;
use std::ops::Sub;

fn pretty_print_coefficients_generic<T: Add + Div + Mul + Rem + Sub + IdentityValues + Display>(
    coefficients: &[T],
) -> String {
    if coefficients.is_empty() {
        return String::from("0");
    }

    let mut outputs: Vec<String> = Vec::new();
    let mut pol_degree = coefficients.len() - 1;
    // reduce pol_degree to skip trailing zeros
    while coefficients[pol_degree].is_zero() {
        pol_degree -= 1;
    }

    // for every nonzero term, in descending order
    for i in 0..=pol_degree {
        let pow = pol_degree - i;
        if coefficients[pow].is_zero() {
            continue;
        }

        outputs.push(format!(
            "{}{}{}", // { + } { 7 } { x^3 }
            if i == 0 { "" } else { " + " },
            if coefficients[pow].is_one() {
                String::from("")
            } else {
                coefficients[pow].to_string()
            },
            if pow == 0 && coefficients[pow].is_one() {
                let one: T = coefficients[pow].ring_one();
                one.to_string()
            } else if pow == 0 {
                String::from("")
            } else if pow == 1 {
                String::from("x")
            } else {
                let mut result = "x^".to_owned();
                let borrowed_string = pow.to_string().to_owned();
                result.push_str(&borrowed_string);
                result
            }
        ));
    }
    outputs.join("")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial<
    T: Add + Div + Mul + Rem + Sub + IdentityValues + Clone + PartialEq + Eq + Hash + Display + Debug,
> {
    pub coefficients: Vec<T>,
}

impl<
        T: Add
            + Div
            + Mul
            + Rem
            + Sub
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Debug,
    > std::fmt::Display for Polynomial<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}",
            pretty_print_coefficients_generic(&self.coefficients)
        )
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Rem
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + Display
            + Debug
            + PartialEq
            + Eq
            + Hash,
    > Polynomial<U>
{
    pub fn normalize(&mut self) {
        while !self.coefficients.is_empty() && self.coefficients.last().unwrap().is_zero() {
            self.coefficients.pop();
        }
    }

    pub fn ring_zero() -> Self {
        Self {
            coefficients: vec![],
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty() || self.coefficients.iter().all(|x| x.is_zero())
    }

    pub fn evaluate(&self, x: &U) -> U {
        let mut acc = x.ring_zero();
        for c in self.coefficients.iter().rev() {
            acc = c.to_owned() + x.to_owned() * acc;
        }

        acc
    }

    pub fn lagrange_interpolation_2(point0: &(U, U), point1: &(U, U)) -> (U, U) {
        let x_diff = point0.0.clone() - point1.0.clone();
        let x_diff_inv = point0.0.ring_one() / x_diff;
        let a = (point0.1.clone() - point1.1.clone()) * x_diff_inv;
        let b = point0.1.clone() - a.clone() * point0.0.clone();

        (a, b)
    }

    pub fn are_colinear(points: &[(U, U)]) -> bool {
        if points.len() < 3 {
            println!("Too few points received. Got: {} points", points.len());
            return false;
        }

        if !has_unique_elements(points.iter().map(|p| p.0.clone())) {
            println!("Non-unique element spotted Got: {:?}", points);
            return false;
        }

        // Find 1st degree polynomial from first two points
        let one: U = points[0].0.ring_one();
        let x_diff: U = points[0].0.clone() - points[1].0.clone();
        let x_diff_inv = one / x_diff;
        let a = (points[0].1.clone() - points[1].1.clone()) * x_diff_inv;
        let b = points[0].1.clone() - a.clone() * points[0].0.clone();
        for point in points.iter().skip(2) {
            let expected = a.clone() * point.0.clone() + b.clone();
            if point.1 != expected {
                println!(
                    "L({}) = {}, expected L({}) = {}, Found: L(x) = {}x + {} from {{({},{}),({},{})}}",
                    point.0,
                    point.1,
                    point.0,
                    expected,
                    a,
                    b,
                    points[0].0,
                    points[0].1,
                    points[1].0,
                    points[1].1
                );
                return false;
            }
        }

        true
    }

    // Any fast interpolation will use NTT, so this is mainly used for testing/integrity
    // purposes. This also means that it is not pivotal that this function has an optimal
    // runtime.
    pub fn slow_lagrange_interpolation(points: &[(U, U)]) -> Self {
        // calculate a reversed representation of the coefficients of
        // prod_{i=0}^{N}((x- q_i))
        fn prod_helper<T: IdentityValues + Sub<Output = T> + Mul<Output = T> + Clone>(
            input: &[T],
        ) -> Vec<T> {
            if let Some((q_j, elements)) = input.split_first() {
                let one: T = q_j.ring_one();
                let zero: T = q_j.ring_zero();
                let minus_q_j = zero.clone() - q_j.to_owned();
                match elements {
                    // base case is `x - q_j` := [1, -q_j]
                    [] => vec![one, minus_q_j],
                    _ => {
                        // The recursive call calculates (x-q_j)*rec = x*rec - q_j*rec := [0, rec] .- q_j*[rec]
                        let mut rec = prod_helper(elements);
                        rec.push(zero);
                        let mut i = rec.len() - 1;
                        while i > 0 {
                            rec[i] = rec[i].clone() - q_j.to_owned() * rec[i - 1].clone();
                            i -= 1;
                        }
                        rec
                    }
                }
            } else {
                panic!("Empty array received");
            }
        }

        if !has_unique_elements(points.iter().map(|x| x.0.clone())) {
            panic!("Repeated x values received. Got: {:?}", points);
        }

        let roots: Vec<U> = points.iter().map(|x| x.0.clone()).collect();
        let mut big_pol_coeffs = prod_helper(&roots);

        big_pol_coeffs.reverse();
        let big_pol = Self {
            coefficients: big_pol_coeffs.iter().map(|x| x.to_owned()).collect(),
        };
        let zero: U = points[0].0.ring_zero();
        let one: U = points[0].0.ring_one();
        let mut coefficients: Vec<U> = vec![zero.clone(); points.len()];
        for point in points.iter() {
            // create a PrimeFieldPolynomial that is zero at all other points than this
            // coeffs_j = prod_{i=0, i != j}^{N}((x- q_i))
            let my_div_coefficients = vec![zero.clone() - point.0.clone(), one.clone()];
            let mut my_pol = Self {
                coefficients: my_div_coefficients.iter().map(|x| x.to_owned()).collect(),
            };
            my_pol = big_pol.clone() / my_pol.clone();

            let mut divisor = one.clone();
            for root in roots.iter() {
                if *root == point.0 {
                    continue;
                }
                divisor = divisor * (point.0.clone() - root.to_owned());
            }

            let mut my_coeffs: Vec<U> = my_pol.coefficients.iter().map(|x| x.to_owned()).collect();
            for coeff in my_coeffs.iter_mut() {
                *coeff = coeff.to_owned() * point.1.clone();
                *coeff = coeff.to_owned() / divisor.clone();
            }

            for i in 0..my_coeffs.len() {
                coefficients[i] = coefficients[i].clone() + my_coeffs[i].clone();
            }
        }

        Self { coefficients }
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Rem
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + std::fmt::Debug
            + std::fmt::Display
            + PartialEq
            + Eq
            + Hash,
    > Polynomial<U>
{
    fn multiply(self, other: Self) -> Self {
        let degree_lhs = self.degree();
        let degree_rhs = other.degree();

        if degree_lhs < 0 || degree_rhs < 0 {
            return Self::ring_zero();
            // return self.zero();
        }

        // allocate right number of coefficients, initialized to zero
        let elem = self.coefficients[0].clone();
        let mut result_coeff: Vec<U> =
            //vec![U::zero_from_field(field: U); degree_lhs as usize + degree_rhs as usize + 1];
            vec![elem.ring_zero(); degree_lhs as usize + degree_rhs as usize + 1];

        // for all pairs of coefficients, add product to result vector in appropriate coordinate
        for i in 0..=degree_lhs as usize {
            for j in 0..=degree_rhs as usize {
                let mul: U = self.coefficients[i].clone() * other.coefficients[j].clone();
                result_coeff[i + j] = result_coeff[i + j].clone() + mul;
            }
        }

        // build and return Polynomial object
        Self {
            coefficients: result_coeff,
        }
    }

    pub fn scalar_mul(&self, scalar: U) -> Self {
        let mut coefficients: Vec<U> = vec![];
        for i in 0..self.coefficients.len() {
            coefficients.push(self.coefficients[i].clone() * scalar.clone());
        }

        Self { coefficients }
    }

    pub fn divide(&self, divisor: Self) -> (Self, Self) {
        let degree_lhs = self.degree();
        let degree_rhs = divisor.degree();
        // cannot divide by zero
        if degree_rhs < 0 {
            panic!(
                "Cannot divide polynomial by zero. Got: ({:?})/({:?})",
                self, divisor
            );
        }

        // zero divided by anything guves zero
        if degree_lhs < 0 {
            return (Self::ring_zero(), Self::ring_zero());
        }

        // quotient is built from back to front so must be reversed
        let mut quotient = Vec::with_capacity((degree_lhs - degree_rhs + 1) as usize);
        let mut remainder = self.clone();

        let dlc: U = divisor.coefficients[degree_rhs as usize].clone(); // divisor leading coefficient
        let inv = dlc.ring_one() / dlc;

        let mut i = 0;
        while i + degree_rhs <= degree_lhs {
            // calculate next quotient coefficient, and set leading coefficient
            // of remainder remainder is 0 by removing it
            let rlc: U = remainder.coefficients.last().unwrap().to_owned();
            let q: U = rlc * inv.clone();
            quotient.push(q.clone());
            remainder.coefficients.pop();
            if q.is_zero() {
                i += 1;
                continue;
            }

            // Calculate the new remainder
            for j in 0..degree_rhs as usize {
                let rem_length = remainder.coefficients.len();
                remainder.coefficients[rem_length - j - 1] = remainder.coefficients
                    [rem_length - j - 1]
                    .clone()
                    - q.clone() * divisor.coefficients[divisor.coefficients.len() - j - 2].clone();
            }

            i += 1;
        }

        quotient.reverse();
        let quotient_pol = Self {
            coefficients: quotient,
        };

        (quotient_pol, remainder)
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Rem
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Display
            + Debug
            + Hash,
    > Div for Polynomial<U>
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let (quotient, _): (Self, Self) = self.divide(other);
        quotient
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Rem
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Debug,
    > Rem for Polynomial<U>
{
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        let (_, remainder): (Self, Self) = self.divide(other);
        remainder
    }
}

impl<
        U: Add<Output = U>
            + Div
            + Mul
            + Rem
            + Sub
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Debug,
    > Add for Polynomial<U>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let summed: Vec<U> = self
            .coefficients
            .iter()
            .zip_longest(other.coefficients.iter())
            .map(|a: itertools::EitherOrBoth<&U, &U>| match a {
                Both(l, r) => l.to_owned() + r.to_owned(),
                Left(l) => l.to_owned(),
                Right(r) => r.to_owned(),
            })
            .collect();

        Self {
            coefficients: summed,
        }
    }
}

impl<
        U: Add
            + Div
            + Mul
            + Rem
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Display
            + Debug,
    > Sub for Polynomial<U>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let summed: Vec<U> = self
            .coefficients
            .iter()
            .zip_longest(other.coefficients.iter())
            .map(|a: itertools::EitherOrBoth<&U, &U>| match a {
                Both(l, r) => l.to_owned() - r.to_owned(),
                Left(l) => l.to_owned(),
                Right(r) => r.ring_zero() - r.to_owned(),
            })
            .collect();

        Self {
            coefficients: summed,
        }
    }
}

impl<
        U: Add
            + Div
            + Mul
            + Rem
            + Sub
            + IdentityValues
            + Clone
            + PartialEq
            + Eq
            + Hash
            + Debug
            + Display,
    > Polynomial<U>
{
    pub fn degree(&self) -> isize {
        let mut deg = self.coefficients.len() as isize - 1;
        while deg > 0 && self.coefficients[deg as usize].is_zero() {
            deg -= 1;
        }

        deg // -1 for the zero polynomial
    }
}

impl<
        U: Add<Output = U>
            + Div<Output = U>
            + Mul<Output = U>
            + Rem
            + Sub<Output = U>
            + IdentityValues
            + Clone
            + std::fmt::Debug
            + std::fmt::Display
            + PartialEq
            + Eq
            + Hash,
    > Mul for Polynomial<U>
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::multiply(self, other)
    }
}

#[cfg(test)]
mod test_polynomials {
    #![allow(clippy::just_underscores_and_digits)]
    use super::super::prime_field_element::{PrimeField, PrimeFieldElement};
    use super::super::prime_field_element_big::{PrimeFieldBig, PrimeFieldElementBig};
    use super::*;
    use crate::utils::generate_random_numbers;
    use num_bigint::BigInt;

    fn b(x: i128) -> BigInt {
        Into::<BigInt>::into(x)
    }

    fn pf(value: i128, field: &PrimeField) -> PrimeFieldElement {
        PrimeFieldElement::new(value, field)
    }

    #[allow(clippy::needless_lifetimes)] // Suppress wrong warning (fails to compile without lifetime, I think)
    fn pfb<'a>(value: i128, field: &'a PrimeFieldBig) -> PrimeFieldElementBig {
        PrimeFieldElementBig::new(b(value), field)
    }

    #[test]
    fn polynomial_evaluate_test_big() {
        let prime_modulus = 71;
        let _71 = PrimeFieldBig::new(b(prime_modulus));
        let parabola = Polynomial::<PrimeFieldElementBig> {
            coefficients: vec![
                PrimeFieldElementBig::new(b(7), &_71),
                PrimeFieldElementBig::new(b(3), &_71),
                PrimeFieldElementBig::new(b(2), &_71),
            ],
        };
        assert_eq!(pfb(7, &_71), parabola.evaluate(&pfb(0, &_71)));
        assert_eq!(pfb(12, &_71), parabola.evaluate(&pfb(1, &_71)));
        assert_eq!(pfb(21, &_71), parabola.evaluate(&pfb(2, &_71)));
        assert_eq!(pfb(34, &_71), parabola.evaluate(&pfb(3, &_71)));
        assert_eq!(pfb(51, &_71), parabola.evaluate(&pfb(4, &_71)));
        assert_eq!(pfb(1, &_71), parabola.evaluate(&pfb(5, &_71)));
        assert_eq!(pfb(26, &_71), parabola.evaluate(&pfb(6, &_71)));
    }

    #[test]
    fn polynomial_evaluate_test() {
        let prime_modulus = 71;
        let _71 = PrimeField::new(prime_modulus);
        let parabola = Polynomial::<PrimeFieldElement> {
            coefficients: vec![
                PrimeFieldElement::new(7, &_71),
                PrimeFieldElement::new(3, &_71),
                PrimeFieldElement::new(2, &_71),
            ],
        };
        assert_eq!(
            PrimeFieldElement::new(7, &_71),
            parabola.evaluate(&PrimeFieldElement::new(0, &_71))
        );
        assert_eq!(
            PrimeFieldElement::new(12, &_71),
            parabola.evaluate(&PrimeFieldElement::new(1, &_71))
        );
        assert_eq!(
            PrimeFieldElement::new(21, &_71),
            parabola.evaluate(&PrimeFieldElement::new(2, &_71))
        );
        assert_eq!(
            PrimeFieldElement::new(34, &_71),
            parabola.evaluate(&PrimeFieldElement::new(3, &_71))
        );
        assert_eq!(
            PrimeFieldElement::new(51, &_71),
            parabola.evaluate(&PrimeFieldElement::new(4, &_71))
        );
        assert_eq!(
            PrimeFieldElement::new(1, &_71),
            parabola.evaluate(&PrimeFieldElement::new(5, &_71))
        );
        assert_eq!(
            PrimeFieldElement::new(26, &_71),
            parabola.evaluate(&PrimeFieldElement::new(6, &_71))
        );
    }

    #[test]
    fn normalize_test() {
        let prime_modulus = 71;
        let _71 = PrimeField::new(prime_modulus);
        let _0_71 = PrimeFieldElement::new(0, &_71);
        let _1_71 = PrimeFieldElement::new(1, &_71);
        let _6_71 = PrimeFieldElement::new(6, &_71);
        let _12_71 = PrimeFieldElement::new(12, &_71);
        let _71 = PrimeField::new(prime_modulus);
        let zero: Polynomial<PrimeFieldElement> = Polynomial::ring_zero();
        let mut mut_one: Polynomial<PrimeFieldElement> = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_1_71],
        };
        let one: Polynomial<PrimeFieldElement> = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_1_71],
        };
        let mut a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![],
        };
        a.normalize();
        assert_eq!(zero, a);
        mut_one.normalize();
        assert_eq!(one, mut_one);

        // trailing zeros are removed
        a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_1_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElement> {
                coefficients: vec![_1_71],
            },
            a
        );
        a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_1_71, _0_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElement> {
                coefficients: vec![_1_71],
            },
            a
        );

        // but leading zeros are not removed
        a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_0_71, _1_71, _0_71, _0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElement> {
                coefficients: vec![_0_71, _1_71],
            },
            a
        );
        a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_0_71, _1_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElement> {
                coefficients: vec![_0_71, _1_71],
            },
            a
        );
        a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_0_71],
        };
        a.normalize();
        assert_eq!(
            Polynomial::<PrimeFieldElement> {
                coefficients: vec![],
            },
            a
        );
    }

    #[test]
    fn slow_lagrange_interpolation_test() {
        let field = PrimeField::new(7);
        let points = &[
            (pf(0, &field), pf(6, &field)),
            (pf(1, &field), pf(6, &field)),
            (pf(2, &field), pf(2, &field)),
        ];

        let interpolation_result = Polynomial::slow_lagrange_interpolation(points);
        let expected_result = Polynomial {
            coefficients: vec![pf(6, &field), pf(2, &field), pf(5, &field)],
        };
        assert_eq!(expected_result, interpolation_result);

        // Use the same numbers to test evaluation
        for point in points.iter() {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }
    }

    #[test]
    fn slow_lagrange_interpolation_test_big() {
        let field = PrimeFieldBig::new(b(7));
        let points = &[
            (pfb(0, &field), pfb(6, &field)),
            (pfb(1, &field), pfb(6, &field)),
            (pfb(2, &field), pfb(2, &field)),
        ];

        let interpolation_result = Polynomial::slow_lagrange_interpolation(points);
        let expected_result = Polynomial {
            coefficients: vec![pfb(6, &field), pfb(2, &field), pfb(5, &field)],
        };
        assert_eq!(expected_result, interpolation_result);

        // Use the same numbers to test evaluation
        for point in points.iter() {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }
    }

    #[test]
    fn property_based_slow_lagrange_interpolation_test() {
        // Autogenerate a `number_of_points - 1` degree polynomial
        // We start by autogenerating the polynomial, as we would get a polynomial
        // with fractional coefficients if we autogenerated the points and derived the polynomium
        // from that.
        let field = PrimeField::new(999983i128);
        let number_of_points = 50usize;
        let coefficients: Vec<PrimeFieldElement> =
            generate_random_numbers(number_of_points, field.q)
                .iter()
                .map(|x| PrimeFieldElement::new(*x as i128, &field))
                .collect();

        let pol: Polynomial<PrimeFieldElement> = Polynomial { coefficients };

        // Evaluate polynomial in `number_of_points` points
        let points: Vec<(PrimeFieldElement, PrimeFieldElement)> = (0..number_of_points)
            .map(|x| {
                let x = PrimeFieldElement::new(x as i128, &field);
                (x, pol.evaluate(&x))
            })
            .collect();

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result: Polynomial<PrimeFieldElement> =
            Polynomial::slow_lagrange_interpolation(&points);
        assert_eq!(interpolation_result, pol);
        for point in points {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }
    }

    #[test]
    fn property_based_slow_lagrange_interpolation_test_big() {
        // Autogenerate a `number_of_points - 1` degree polynomial
        // We start by autogenerating the polynomial, as we would get a polynomial
        // with fractional coefficients if we autogenerated the points and derived the polynomium
        // from that.
        let field = PrimeFieldBig::new(b(999983i128));
        let number_of_points = 50usize;
        let coefficients: Vec<PrimeFieldElementBig> =
            generate_random_numbers(number_of_points, 999983i128)
                .iter()
                .map(|x| pfb(*x as i128, &field))
                .collect();

        let pol: Polynomial<PrimeFieldElementBig> = Polynomial { coefficients };

        // Evaluate polynomial in `number_of_points` points
        let points: Vec<(PrimeFieldElementBig, PrimeFieldElementBig)> = (0..number_of_points)
            .map(|x| {
                let x = pfb(x as i128, &field);
                (x.clone(), pol.evaluate(&x))
            })
            .collect();

        // Derive the `number_of_points - 1` degree polynomium from these `number_of_points` points,
        // evaluate the point values, and verify that they match the original values
        let interpolation_result: Polynomial<PrimeFieldElementBig> =
            Polynomial::slow_lagrange_interpolation(&points);
        assert_eq!(interpolation_result, pol);
        for point in points {
            assert_eq!(point.1, interpolation_result.evaluate(&point.0));
        }
    }

    #[test]
    fn lagrange_interpolation_2_test() {
        let field = PrimeField::new(5);
        assert_eq!(
            (pf(1, &field), pf(0, &field)),
            Polynomial::lagrange_interpolation_2(
                &(pf(1, &field), pf(1, &field)),
                &(pf(2, &field), pf(2, &field))
            )
        );
        assert_eq!(
            (pf(4, &field), pf(4, &field)),
            Polynomial::lagrange_interpolation_2(
                &(pf(1, &field), pf(3, &field)),
                &(pf(2, &field), pf(2, &field))
            )
        );
        assert_eq!(
            (pf(4, &field), pf(2, &field)),
            Polynomial::lagrange_interpolation_2(
                &(pf(15, &field), pf(92, &field)),
                &(pf(19, &field), pf(108, &field))
            )
        );

        assert_eq!(
            (pf(3, &field), pf(2, &field)),
            Polynomial::lagrange_interpolation_2(
                &(pf(1, &field), pf(0, &field)),
                &(pf(2, &field), pf(3, &field))
            )
        );

        let field_big = PrimeFieldBig::new(b(5));
        assert_eq!(
            (pfb(1, &field_big), pfb(0, &field_big)),
            Polynomial::lagrange_interpolation_2(
                &(pfb(1, &field_big), pfb(1, &field_big)),
                &(pfb(2, &field_big), pfb(2, &field_big))
            )
        );
        assert_eq!(
            (pfb(4, &field_big), pfb(4, &field_big)),
            Polynomial::lagrange_interpolation_2(
                &(pfb(1, &field_big), pfb(3, &field_big)),
                &(pfb(2, &field_big), pfb(2, &field_big))
            )
        );
        assert_eq!(
            (pfb(4, &field_big), pfb(2, &field_big)),
            Polynomial::lagrange_interpolation_2(
                &(pfb(15, &field_big), pfb(92, &field_big)),
                &(pfb(19, &field_big), pfb(108, &field_big))
            )
        );
        assert_eq!(
            (pfb(3, &field_big), pfb(2, &field_big)),
            Polynomial::lagrange_interpolation_2(
                &(pfb(1, &field_big), pfb(0, &field_big)),
                &(pfb(2, &field_big), pfb(3, &field_big))
            )
        );
    }

    #[test]
    fn polynomial_are_colinear_test() {
        let field = PrimeField::new(5);
        assert!(Polynomial::are_colinear(&[
            (pf(1, &field), pf(1, &field)),
            (pf(2, &field), pf(2, &field)),
            (pf(3, &field), pf(3, &field))
        ]));
        assert!(Polynomial::are_colinear(&[
            (pf(1, &field), pf(1, &field)),
            (pf(2, &field), pf(7, &field)),
            (pf(3, &field), pf(3, &field))
        ]));
        assert!(Polynomial::are_colinear(&[
            (pf(1, &field), pf(3, &field)),
            (pf(2, &field), pf(2, &field)),
            (pf(3, &field), pf(1, &field))
        ]));
        assert!(Polynomial::are_colinear(&[
            (pf(1, &field), pf(1, &field)),
            (pf(7, &field), pf(7, &field)),
            (pf(3, &field), pf(3, &field))
        ]));
        assert!(!Polynomial::are_colinear(&[
            (pf(1, &field), pf(1, &field)),
            (pf(2, &field), pf(2, &field)),
            (pf(3, &field), pf(4, &field))
        ]));
        assert!(!Polynomial::are_colinear(&[
            (pf(1, &field), pf(1, &field)),
            (pf(2, &field), pf(3, &field)),
            (pf(3, &field), pf(3, &field))
        ]));
        assert!(!Polynomial::are_colinear(&[
            (pf(1, &field), pf(0, &field)),
            (pf(2, &field), pf(3, &field)),
            (pf(3, &field), pf(3, &field))
        ]));
        assert!(Polynomial::are_colinear(&[
            (pf(15, &field), pf(92, &field)),
            (pf(11, &field), pf(76, &field)),
            (pf(19, &field), pf(108, &field))
        ]));
        assert!(!Polynomial::are_colinear(&[
            (pf(12, &field), pf(92, &field)),
            (pf(11, &field), pf(76, &field)),
            (pf(19, &field), pf(108, &field))
        ]));

        // Disallow repeated x-values
        assert!(!Polynomial::are_colinear(&[
            (pf(12, &field), pf(92, &field)),
            (pf(11, &field), pf(76, &field)),
            (pf(11, &field), pf(108, &field))
        ]));

        // Disallow args with less than three points
        assert!(!Polynomial::are_colinear(&[
            (pf(12, &field), pf(92, &field)),
            (pf(11, &field), pf(76, &field))
        ]));
    }

    #[test]
    fn polynomial_are_colinear_test_big() {
        let field = PrimeFieldBig::new(b(5));
        assert!(Polynomial::are_colinear(&[
            (pfb(1, &field), pfb(1, &field)),
            (pfb(2, &field), pfb(2, &field)),
            (pfb(3, &field), pfb(3, &field))
        ]));
        assert!(Polynomial::are_colinear(&[
            (pfb(1, &field), pfb(1, &field)),
            (pfb(2, &field), pfb(7, &field)),
            (pfb(3, &field), pfb(3, &field))
        ]));
        assert!(Polynomial::are_colinear(&[
            (pfb(1, &field), pfb(3, &field)),
            (pfb(2, &field), pfb(2, &field)),
            (pfb(3, &field), pfb(1, &field))
        ]));
        assert!(Polynomial::are_colinear(&[
            (pfb(1, &field), pfb(1, &field)),
            (pfb(7, &field), pfb(7, &field)),
            (pfb(3, &field), pfb(3, &field))
        ]));
        assert!(!Polynomial::are_colinear(&[
            (pfb(1, &field), pfb(1, &field)),
            (pfb(2, &field), pfb(2, &field)),
            (pfb(3, &field), pfb(4, &field))
        ]));
        assert!(!Polynomial::are_colinear(&[
            (pfb(1, &field), pfb(1, &field)),
            (pfb(2, &field), pfb(3, &field)),
            (pfb(3, &field), pfb(3, &field))
        ]));
        assert!(!Polynomial::are_colinear(&[
            (pfb(1, &field), pfb(0, &field)),
            (pfb(2, &field), pfb(3, &field)),
            (pfb(3, &field), pfb(3, &field))
        ]));
        assert!(Polynomial::are_colinear(&[
            (pfb(15, &field), pfb(92, &field)),
            (pfb(11, &field), pfb(76, &field)),
            (pfb(19, &field), pfb(108, &field))
        ]));
        assert!(!Polynomial::are_colinear(&[
            (pfb(12, &field), pfb(92, &field)),
            (pfb(11, &field), pfb(76, &field)),
            (pfb(19, &field), pfb(108, &field))
        ]));

        // Disallow repeated x-values
        assert!(!Polynomial::are_colinear(&[
            (pfb(12, &field), pfb(92, &field)),
            (pfb(11, &field), pfb(76, &field)),
            (pfb(11, &field), pfb(108, &field))
        ]));

        // Disallow args with less than three points
        assert!(!Polynomial::are_colinear(&[
            (pfb(12, &field), pfb(92, &field)),
            (pfb(11, &field), pfb(76, &field))
        ]));
    }

    #[test]
    fn polynomial_arithmetic_property_based_test() {
        let prime_modulus = 71;
        let _71 = PrimeField::new(prime_modulus);
        let a_degree = 20;
        for i in 0..20 {
            let mut a = Polynomial::<PrimeFieldElement> {
                coefficients: generate_random_numbers(a_degree, prime_modulus)
                    .iter()
                    .map(|x| PrimeFieldElement::new(*x, &_71))
                    .collect(),
            };
            a.normalize();
            let mut b = Polynomial::<PrimeFieldElement> {
                coefficients: generate_random_numbers(a_degree + i, prime_modulus)
                    .iter()
                    .map(|x| PrimeFieldElement::new(*x, &_71))
                    .collect(),
            };
            b.normalize();

            let mul_a_b: Polynomial<PrimeFieldElement> = a.clone() * b.clone();
            let mul_b_a: Polynomial<PrimeFieldElement> = b.clone() * a.clone();
            let add_a_b: Polynomial<PrimeFieldElement> = a.clone() + b.clone();
            let add_b_a: Polynomial<PrimeFieldElement> = b.clone() + a.clone();
            let sub_a_b: Polynomial<PrimeFieldElement> = a.clone() - b.clone();
            let sub_b_a: Polynomial<PrimeFieldElement> = b.clone() - a.clone();

            let mut res = mul_a_b.clone() / b.clone();
            res.normalize();
            assert_eq!(res, a);
            res = mul_b_a.clone() / a.clone();
            res.normalize();
            assert_eq!(res, b);
            res = add_a_b.clone() - b.clone();
            res.normalize();
            assert_eq!(res, a);
            res = sub_a_b.clone() + b.clone();
            res.normalize();
            assert_eq!(res, a);
            res = add_b_a.clone() - a.clone();
            res.normalize();
            assert_eq!(res, b);
            res = sub_b_a.clone() + a.clone();
            res.normalize();
            assert_eq!(res, b);
            assert_eq!(add_a_b, add_b_a);
            assert_eq!(mul_a_b, mul_b_a);
            assert!(a.degree() < a_degree as isize);
            assert!(b.degree() < (a_degree + i) as isize);
            assert!(mul_a_b.degree() <= ((a_degree - 1) * 2 + i) as isize);
            assert!(add_a_b.degree() < (a_degree + i) as isize);
        }
    }

    #[test]
    fn polynomial_arithmetic_property_based_test_big() {
        let prime_modulus = 71;
        let _71 = PrimeFieldBig::new(b(prime_modulus));
        let a_degree = 20;
        for i in 0..20 {
            let mut a = Polynomial::<PrimeFieldElementBig> {
                coefficients: generate_random_numbers(a_degree, prime_modulus)
                    .iter()
                    .map(|x| pfb(*x, &_71))
                    .collect(),
            };
            a.normalize();
            let mut b = Polynomial::<PrimeFieldElementBig> {
                coefficients: generate_random_numbers(a_degree + i, prime_modulus)
                    .iter()
                    .map(|x| pfb(*x, &_71))
                    .collect(),
            };
            b.normalize();

            let mul_a_b: Polynomial<PrimeFieldElementBig> = a.clone() * b.clone();
            let mul_b_a: Polynomial<PrimeFieldElementBig> = b.clone() * a.clone();
            let add_a_b: Polynomial<PrimeFieldElementBig> = a.clone() + b.clone();
            let add_b_a: Polynomial<PrimeFieldElementBig> = b.clone() + a.clone();
            let sub_a_b: Polynomial<PrimeFieldElementBig> = a.clone() - b.clone();
            let sub_b_a: Polynomial<PrimeFieldElementBig> = b.clone() - a.clone();

            let mut res = mul_a_b.clone() / b.clone();
            res.normalize();
            assert_eq!(res, a);
            res = mul_b_a.clone() / a.clone();
            res.normalize();
            assert_eq!(res, b);
            res = add_a_b.clone() - b.clone();
            res.normalize();
            assert_eq!(res, a);
            res = sub_a_b.clone() + b.clone();
            res.normalize();
            assert_eq!(res, a);
            res = add_b_a.clone() - a.clone();
            res.normalize();
            assert_eq!(res, b);
            res = sub_b_a.clone() + a.clone();
            res.normalize();
            assert_eq!(res, b);
            assert_eq!(add_a_b, add_b_a);
            assert_eq!(mul_a_b, mul_b_a);
            assert!(a.degree() < a_degree as isize);
            assert!(b.degree() < (a_degree + i) as isize);
            assert!(mul_a_b.degree() <= ((a_degree - 1) * 2 + i) as isize);
            assert!(add_a_b.degree() < (a_degree + i) as isize);
        }
    }

    #[test]
    fn polynomial_arithmetic_division_test() {
        let _71 = PrimeField::new(71);
        let a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![PrimeFieldElement::new(17, &_71)],
        };
        let b = Polynomial::<PrimeFieldElement> {
            coefficients: vec![PrimeFieldElement::new(17, &_71)],
        };
        let one = Polynomial::<PrimeFieldElement> {
            coefficients: vec![PrimeFieldElement::new(1, &_71)],
        };
        let zero = Polynomial::<PrimeFieldElement> {
            coefficients: vec![PrimeFieldElement::new(0, &_71)],
        };
        assert_eq!(one, a / b.clone());
        assert_eq!(zero, zero.clone() / b.clone());

        let x: Polynomial<PrimeFieldElement> = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(1, &_71),
            ],
        };
        let mut prod_x = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(1, &_71),
            ],
        };
        let mut expected_quotient = Polynomial {
            coefficients: vec![PrimeFieldElement::new(1, &_71)],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());
        assert_eq!(zero, zero.clone() / b);

        prod_x = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(1, &_71),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![PrimeFieldElement::new(1, &_71)],
        };
        assert_eq!(expected_quotient, prod_x / (x.clone() * x.clone()));

        prod_x = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(1, &_71),
                PrimeFieldElement::new(2, &_71),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(1, &_71),
                PrimeFieldElement::new(2, &_71),
            ],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());

        prod_x = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(1, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(2, &_71),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(2, &_71),
            ],
        };
        assert_eq!(expected_quotient, prod_x / x.clone());

        prod_x = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(48, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(25, &_71),
                PrimeFieldElement::new(11, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(64, &_71),
                PrimeFieldElement::new(16, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(30, &_71),
            ],
        };
        expected_quotient = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(48, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(25, &_71),
                PrimeFieldElement::new(11, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(64, &_71),
                PrimeFieldElement::new(16, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(30, &_71),
            ],
        };
        assert_eq!(expected_quotient, prod_x.clone() / x.clone());

        expected_quotient = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(25, &_71),
                PrimeFieldElement::new(11, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(64, &_71),
                PrimeFieldElement::new(16, &_71),
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(30, &_71),
            ],
        };
        assert_eq!(expected_quotient, prod_x.clone() / (x.clone() * x.clone()));
        assert_eq!(
            Polynomial {
                coefficients: vec![
                    PrimeFieldElement::new(0, &_71),
                    PrimeFieldElement::new(48, &_71),
                ],
            },
            prod_x % (x.clone() * x.clone())
        );
    }

    #[test]
    fn polynomial_arithmetic_test_linear_combination() {
        let field = PrimeFieldBig::new(b(167772161));
        let tq = Polynomial::<PrimeFieldElementBig> {
            coefficients: vec![
                pfb(76432291, &field),
                pfb(6568597, &field),
                pfb(37593670, &field),
                pfb(164656139, &field),
                pfb(100728053, &field),
                pfb(8855557, &field),
                pfb(84827854, &field),
            ],
        };
        let ti = Polynomial::<PrimeFieldElementBig> {
            coefficients: vec![
                pfb(137616711, &field),
                pfb(15613095, &field),
                pfb(114041830, &field),
                pfb(68272686, &field),
            ],
        };
        let bq = Polynomial::<PrimeFieldElementBig> {
            coefficients: vec![pfb(43152288, &field), pfb(68272686, &field)],
        };
        let x_to_3 = Polynomial::<PrimeFieldElementBig> {
            coefficients: vec![
                pfb(0, &field),
                pfb(0, &field),
                pfb(0, &field),
                pfb(1, &field),
            ],
        };
        let ks = vec![
            pfb(132934501, &field),
            pfb(57662258, &field),
            pfb(76229169, &field),
            pfb(82319948, &field),
        ];
        let expected_lc = Polynomial::<PrimeFieldElementBig> {
            coefficients: vec![
                pfb(2792937, &field),
                pfb(39162406, &field),
                pfb(7217300, &field),
                pfb(58955792, &field),
                pfb(3275580, &field),
                pfb(58708383, &field),
                pfb(3119620, &field),
            ],
        };
        let linear_combination = tq
            + ti.scalar_mul(ks[0].clone())
            + (ti * x_to_3.clone()).scalar_mul(ks[1].clone())
            + bq.scalar_mul(ks[2].clone())
            + (bq * x_to_3).scalar_mul(ks[3].clone());
        assert_eq!(expected_lc, linear_combination);

        let x_values: Vec<PrimeFieldElementBig> = vec![
            pfb(1, &field),
            pfb(116878283, &field),
            pfb(71493608, &field),
            pfb(131850885, &field),
            pfb(65249968, &field),
            pfb(26998229, &field),
            pfb(30406922, &field),
            pfb(40136459, &field),
            pfb(167772160, &field),
            pfb(50893878, &field),
            pfb(96278553, &field),
            pfb(35921276, &field),
            pfb(102522193, &field),
            pfb(140773932, &field),
            pfb(137365239, &field),
            pfb(127635702, &field),
        ];
        let expected_y_values: Vec<PrimeFieldElementBig> = vec![
            pfb(5459857, &field),
            pfb(148657471, &field),
            pfb(30002611, &field),
            pfb(66137138, &field),
            pfb(8094868, &field),
            pfb(56386222, &field),
            pfb(156375138, &field),
            pfb(54481212, &field),
            pfb(27351017, &field),
            pfb(142491681, &field),
            pfb(27138843, &field),
            pfb(146662298, &field),
            pfb(151140487, &field),
            pfb(131629901, &field),
            pfb(120097158, &field),
            pfb(114758378, &field),
        ];
        for i in 0..16 {
            assert_eq!(
                expected_y_values[i],
                linear_combination.evaluate(&x_values[i])
            );
        }
    }

    #[test]
    fn polynomial_arithmetic_test() {
        let _71 = PrimeField::new(71);
        let _6_71 = PrimeFieldElement::new(6, &_71);
        let _12_71 = PrimeFieldElement::new(12, &_71);
        let _16_71 = PrimeFieldElement::new(16, &_71);
        let _17_71 = PrimeFieldElement::new(17, &_71);
        let _22_71 = PrimeFieldElement::new(22, &_71);
        let _28_71 = PrimeFieldElement::new(28, &_71);
        let _33_71 = PrimeFieldElement::new(33, &_71);
        let _38_71 = PrimeFieldElement::new(38, &_71);
        let _49_71 = PrimeFieldElement::new(49, &_71);
        let _60_71 = PrimeFieldElement::new(60, &_71);
        let _64_71 = PrimeFieldElement::new(64, &_71);
        let _65_71 = PrimeFieldElement::new(65, &_71);
        let _66_71 = PrimeFieldElement::new(66, &_71);
        let mut a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_17_71],
        };
        let mut b = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_16_71],
        };
        let mut sum = a + b;
        let mut expected_sum = Polynomial {
            coefficients: vec![_33_71],
        };
        assert_eq!(expected_sum, sum);

        // Verify overflow handling
        a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_66_71],
        };
        b = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_65_71],
        };
        sum = a + b;
        expected_sum = Polynomial {
            coefficients: vec![_60_71],
        };
        assert_eq!(expected_sum, sum);

        // Verify handling of multiple indices
        a = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_66_71, _66_71, _66_71],
        };
        b = Polynomial::<PrimeFieldElement> {
            coefficients: vec![_33_71, _33_71, _17_71, _65_71],
        };
        sum = a.clone() + b.clone();
        expected_sum = Polynomial {
            coefficients: vec![_28_71, _28_71, _12_71, _65_71],
        };
        assert_eq!(expected_sum, sum);

        let mut diff = a.clone() - b.clone();
        let mut expected_diff = Polynomial {
            coefficients: vec![_33_71, _33_71, _49_71, _6_71],
        };
        assert_eq!(expected_diff, diff);

        diff = b.clone() - a.clone();
        expected_diff = Polynomial {
            coefficients: vec![_38_71, _38_71, _22_71, _65_71],
        };
        assert_eq!(expected_diff, diff);

        // Test multiplication
        let mut prod = a.clone() * b.clone();
        let mut expected_prod = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(48, &_71),
                PrimeFieldElement::new(25, &_71),
                PrimeFieldElement::new(11, &_71),
                PrimeFieldElement::new(64, &_71),
                PrimeFieldElement::new(16, &_71),
                PrimeFieldElement::new(30, &_71),
            ],
        };
        assert_eq!(expected_prod, prod);
        assert_eq!(5, prod.degree());
        assert_eq!(2, a.degree());
        assert_eq!(3, b.degree());

        let zero: Polynomial<PrimeFieldElement> = Polynomial {
            coefficients: vec![],
        };
        let zero_alt: Polynomial<PrimeFieldElement> = Polynomial::ring_zero();
        assert_eq!(zero, zero_alt);
        let one: Polynomial<PrimeFieldElement> = Polynomial {
            coefficients: vec![PrimeFieldElement::new(1, &_71)],
        };
        // let five: Polynomial<PrimeFieldElement> = Polynomial {
        //     coefficients: vec![PrimeFieldElement::new(5, &_71)],
        // };
        let x: Polynomial<PrimeFieldElement> = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(1, &_71),
            ],
        };
        assert_eq!(-1, zero.degree());
        assert_eq!(0, one.degree());
        assert_eq!(1, x.degree());
        assert_eq!(zero, prod.clone() * zero.clone());
        assert_eq!(prod, prod.clone() * one.clone());
        assert_eq!(x, x.clone() * one.clone());

        assert_eq!("0", zero.to_string());
        assert_eq!("1", one.to_string());
        assert_eq!("x", x.to_string());
        assert_eq!("66x^2 + 66x + 66", a.to_string());

        expected_prod = Polynomial {
            coefficients: vec![
                PrimeFieldElement::new(0, &_71),
                PrimeFieldElement::new(48, &_71),
                PrimeFieldElement::new(25, &_71),
                PrimeFieldElement::new(11, &_71),
                PrimeFieldElement::new(64, &_71),
                PrimeFieldElement::new(16, &_71),
                PrimeFieldElement::new(30, &_71),
            ],
        };
        prod = prod.clone() * x.clone();
        assert_eq!(expected_prod, prod);
        assert_eq!(
            "30x^6 + 16x^5 + 64x^4 + 11x^3 + 25x^2 + 48x",
            prod.to_string()
        );
    }
}
