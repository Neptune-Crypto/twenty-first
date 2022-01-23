// use crate::shared_math::prime_field_element_big::{PrimeFieldBig, PrimeFieldElementBig};
use crate::shared_math::traits::{IdentityValues, New};
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;

fn slow_ntt_base_case<T: Add<Output = T> + Mul<Output = T> + Clone>(x: &[T], omega: &T) -> Vec<T> {
    vec![
        x[0].clone() + x[1].clone(),
        x[0].clone() + omega.to_owned() * x[1].clone(),
    ]
}

fn slow_ntt_recursive<
    T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + IdentityValues + Clone,
>(
    x: &[T],
    omega: &T,
) -> Vec<T> {
    let n: usize = x.len();
    if n == 2 {
        return slow_ntt_base_case(x, omega);
    }
    // else,
    // split by parity
    let mut evens: Vec<T> = Vec::with_capacity(n / 2);
    let mut odds: Vec<T> = Vec::with_capacity(n / 2);
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if i % 2 == 1 {
            odds.push(x[i].clone());
        } else {
            evens.push(x[i].clone());
        }
    }

    // recursion step
    let omega_squared = omega.to_owned() * omega.to_owned();
    let (even, odd) = (
        slow_ntt_recursive(&evens, &omega_squared),
        slow_ntt_recursive(&odds, &omega_squared),
    );

    // hadamard product
    let mut omegai: T = omega.ring_one();
    let mut result: Vec<T> = Vec::with_capacity(n);
    for i in 0..(n / 2) {
        result.push(even[i].clone() + odd[i].clone() * omegai.clone());
        omegai = omegai.clone() * omega.to_owned();
    }
    //omegai = -omegai.clone();
    omegai = -omegai.ring_one();
    for i in 0..(n / 2) {
        result.push(even[i].clone() + odd[i].clone() * omegai.clone());
        omegai = omegai.clone() * omega.to_owned();
    }
    result
}

pub fn slow_ntt<T: Add<Output = T> + Mul<Output = T> + Neg<Output = T> + IdentityValues + Clone>(
    x: &[T],
    omega: &T,
) -> Vec<T> {
    // test if n (=length of x) is a power of 2
    let n = x.len();
    if n & (n - 1) != 0 {
        panic!("ntt must operate on vector of length power of two");
    }

    // test if omega is an nth primitive root of unity
    let mut acc = 1usize;
    let mut omega2i = omega.clone();
    while acc != n {
        // omega2i = omega^(acc)
        if omega2i.is_one() {
            panic!("ntt needs primitive nth root of unity but omega has lower order");
        }
        omega2i = omega2i.clone() * omega2i.clone();
        acc <<= 1;
    }
    if !omega2i.is_one() {
        panic!("ntt needs primitive nth root of unity but order of omega does not match n");
    }

    slow_ntt_recursive(x, omega)
}

pub fn slow_intt<
    T: Add<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
        + Div<Output = T>
        + IdentityValues
        + Clone
        + New,
>(
    x: &[T],
    omega: &T,
) -> Vec<T> {
    // const fn num_bits<T>() -> u64 {
    //     std::mem::size_of::<T>() as u64 * 8
    // }
    let n: T = omega.new_from_usize(x.len());
    // let n: T = omega.new(x.len().to_bytes());
    slow_ntt(x, &(omega.ring_one() / omega.to_owned()))
        .into_iter()
        .map(|x: T| x / n.clone())
        .collect()
}

#[cfg(test)]
mod ntt_tests {
    use super::super::prime_field_element_big::{PrimeFieldBig, PrimeFieldElementBig};
    use super::*;
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::polynomial::Polynomial;
    use crate::shared_math::traits::GetPrimitiveRootOfUnity;
    use num_bigint::BigInt;

    fn b(x: i128) -> BigInt {
        Into::<BigInt>::into(x)
    }

    // fn bs(xs: Vec<i128>) -> Vec<BigInt> {
    //     xs.iter()
    //         .map(|x| Into::<BigInt>::into(*x))
    //         .collect::<Vec<BigInt>>()
    // }

    fn pfb(value: i128) -> PrimeFieldBig {
        PrimeFieldBig::new(b(value))
    }

    fn pfeb(value: i128, field: &PrimeFieldBig) -> PrimeFieldElementBig {
        PrimeFieldElementBig::new(b(value), field)
    }

    #[test]

    fn test_ntt_simple_2() {
        let field = pfb(5);
        let generator = pfeb(4, &field);
        let input = vec![pfeb(1, &field), pfeb(4, &field)];
        let expected_output = vec![pfeb(0, &field), pfeb(2, &field)];
        let actual_output = slow_ntt(&input, &generator);
        assert_eq!(expected_output, actual_output);
    }

    #[test]
    fn b_field_ntt_with_length_4() {
        let input = vec![
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        let omega = BFieldElement::ring_one()
            .get_primitive_root_of_unity(4)
            .0
            .unwrap();
        let actual_output = slow_ntt(&input, &omega);
        let expected = vec![
            BFieldElement::new(5),
            BFieldElement::new(1125899906842625),
            BFieldElement::new(18446744069414584318),
            BFieldElement::new(18445618169507741698),
        ];
        assert_eq!(expected, actual_output);
        println!("actual_output = {:?}", actual_output);
    }

    #[test]
    fn b_field_ntt_with_length_32() {
        let input = vec![
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        let omega = BFieldElement::ring_one()
            .get_primitive_root_of_unity(32)
            .0
            .unwrap();
        let actual_output = slow_ntt(&input, &omega);
        println!("actual_output = {:?}", actual_output);
        let expected = vec![
            BFieldElement::new(20),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(18446744069146148869),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(4503599627370500),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(18446726477228544005),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(18446744069414584309),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(268435460),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(18442240469787213829),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(17592186040324),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        assert_eq!(expected, actual_output);
    }

    #[test]
    fn test_ntt_simple_4() {
        let field = pfb(5);
        let generator = pfeb(2, &field);
        let input = vec![
            pfeb(1, &field),
            pfeb(4, &field),
            pfeb(0, &field),
            pfeb(0, &field),
        ];
        let expected_output = vec![
            pfeb(0, &field),
            pfeb(4, &field),
            pfeb(2, &field),
            pfeb(3, &field),
        ];
        let actual_output = slow_ntt(&input, &generator);
        assert_eq!(expected_output, actual_output);
    }

    #[test]
    fn fast_polynomial_functions_property_based_test() {
        let prime = 167772161; // 2^25*5+1
        let field: PrimeFieldBig = pfb(prime);
        for &size in &[2, 4, 8, 1024, 2048] {
            let input_y_values: Vec<PrimeFieldElementBig> = (0..size)
                .map(|_| pfeb(rand::random::<u32>() as i128 % prime, &field))
                .collect();
            let (root_option, _): (Option<PrimeFieldElementBig>, Vec<BigInt>) =
                field.get_primitive_root_of_unity(size);
            let coefficients: Vec<PrimeFieldElementBig> =
                slow_intt(&input_y_values, &root_option.clone().unwrap());
            let output_y_values: Vec<PrimeFieldElementBig> =
                slow_ntt(&coefficients, &root_option.clone().unwrap());
            assert_eq!(input_y_values, output_y_values);
            assert_ne!(input_y_values, coefficients);

            if size < 17 {
                let poly: Polynomial<PrimeFieldElementBig> = Polynomial { coefficients };
                let mut x = pfeb(1, &field);
                // for i in 0..size as usize {
                for input_y_value in input_y_values.iter() {
                    let evaluated_y = poly.evaluate(&x);
                    assert_eq!(*input_y_value, evaluated_y);
                    x = x.clone() * root_option.clone().unwrap();
                }
            }
        }
    }
}
