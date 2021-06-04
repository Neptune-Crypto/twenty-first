use crate::shared_math::complex_number::ComplexNumber;
use crate::shared_math::prime_field_element::{PrimeField, PrimeFieldElement};
use crate::shared_math::vector::{Matrix, Vector};
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::time::Instant;

pub fn dft_finite_fields<'a>(
    x: &[PrimeFieldElement<'a>],
    omega: &PrimeFieldElement<'a>,
    mod_pows: &mut HashMap<usize, PrimeFieldElement<'a>>,
    omega_power: usize,
) -> Vec<PrimeFieldElement<'a>> {
    let pow = match mod_pows.get_key_value(&omega_power) {
        None => {
            let val = omega.mod_pow(omega_power as i128);
            mod_pows.insert(omega_power, val);
            val
        }
        Some(i) => *i.1,
    };

    vec![x[0] + x[1], x[0] + pow * x[1]]
}

fn ntt_fft_helper<'a>(
    x: Vec<PrimeFieldElement<'a>>,
    omega: &PrimeFieldElement<'a>,
    mod_pows: &mut HashMap<usize, PrimeFieldElement<'a>>,
    omega_power: usize,
) -> Vec<PrimeFieldElement<'a>> {
    let size: usize = x.len();
    if size % 2 == 1 {
        panic!("size of input must be a power of 2");
    } else if size == 2 {
        dft_finite_fields(&x, omega, mod_pows, omega_power)
    } else {
        // split by parity
        let mut x_even: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(size / 2);
        let mut x_odd: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(size / 2);
        #[allow(clippy::needless_range_loop)]
        for i in 0..size {
            if i % 2 == 1 {
                x_odd.push(x[i]);
            } else {
                x_even.push(x[i]);
            }
        }

        // Recursive call, using: omega -> omega^2
        let (even, odd) = (
            ntt_fft_helper(x_even, &omega, mod_pows, omega_power * 2),
            ntt_fft_helper(x_odd, &omega, mod_pows, omega_power * 2),
        );

        // Calculate all values omega^(omega_power * j), for j=0..size
        let mut factor_values: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(size);
        for j in 0..(size / 2) {
            let exponent = j * omega_power;
            let pow = match mod_pows.get_key_value(&exponent) {
                None => {
                    let val = omega.mod_pow(exponent as i128);
                    mod_pows.insert(exponent, val);
                    val
                }
                Some(i) => *i.1,
            };
            factor_values.push(pow);
        }
        let minus_one = PrimeFieldElement::new(-1, omega.field);
        for j in 0..(size / 2) {
            factor_values.push(minus_one * factor_values[j]);
        }

        // split by middle
        let fst_half_factors: &[PrimeFieldElement<'a>] = &factor_values[0..size / 2];
        let snd_half_factors: &[PrimeFieldElement<'a>] = &factor_values[size / 2..];

        // hadamard products
        let mut res: Vec<PrimeFieldElement> = Vec::with_capacity(size);
        for i in 0..(size / 2) {
            res.push(even[i] + odd[i] * fst_half_factors[i]);
        }
        for i in 0..(size / 2) {
            res.push(even[i] + odd[i] * snd_half_factors[i]);
        }

        res
    }
}

pub fn ntt_fft<'a>(
    x: Vec<PrimeFieldElement<'a>>,
    omega: &PrimeFieldElement<'a>,
) -> Vec<PrimeFieldElement<'a>> {
    // Verify that ω^N = 1, N is length of x
    let mut mod_pows: HashMap<usize, PrimeFieldElement<'a>> = HashMap::new();
    if omega.mod_pow(x.len() as i128).value != 1 {
        panic!("ntt_fft called with ω^len != 1. Got: {:?}", omega);
    }
    ntt_fft_helper(x, omega, &mut mod_pows, 1)
}

pub fn intt_fft<'a>(
    x: Vec<PrimeFieldElement<'a>>,
    omega: &PrimeFieldElement<'a>,
) -> Vec<PrimeFieldElement<'a>> {
    let length = PrimeFieldElement::new(x.len() as i128, &omega.field);
    let omega_inv = &omega.inv();
    let res_scaled = ntt_fft(x, &omega_inv);

    res_scaled
        .into_iter()
        .map(|x: PrimeFieldElement| x / length)
        .collect()
}

// Interpolate a polynomial with y values as given in the input, and with x values
// which are successive powers of the primitive root of unity given in the input.
// The primitive root of unity in the input must fulfill
// `primitive_root_of_unity` ** y_values.len() == 1
pub fn fast_polynomial_interpolate(
    y_values: &[i128],
    prime: i128,
    primitive_root_of_unity: i128,
) -> Vec<i128> {
    // TODO: This function can probably be made faster if we don't have to send prime field
    // elements to the intt/ntt functions but can just send vectors of atomic values, like i128.
    let field = PrimeField::new(prime);
    let omega = PrimeFieldElement::new(primitive_root_of_unity, &field);
    let intt_inputs = y_values
        .iter()
        .map(|&x| PrimeFieldElement::new(x, &field))
        .collect();
    intt_fft(intt_inputs, &omega)
        .iter()
        .map(|&x| x.value)
        .collect()
}

// Evaluate a polynomial with a a list of coefficients on x values which are successive powers
// of the primitive root of unity given in the input.
// The primitive root of unity in the input must fulfill
// `primitive_root_of_unity` ** y_values.len() == 1
pub fn fast_polynomial_evaluate(
    pol_coefficients: &[i128],
    prime: i128,
    primitive_root_of_unity: i128,
) -> Vec<i128> {
    // TODO: This function can probably be made faster if we don't have to send prime field
    // elements to the intt/ntt functions but can just send vectors of atomic values, like i128.
    let field = PrimeField::new(prime);
    let omega = PrimeFieldElement::new(primitive_root_of_unity, &field);
    let ntt_inputs = pol_coefficients
        .iter()
        .map(|x| PrimeFieldElement::new(*x, &field))
        .collect();
    ntt_fft(ntt_inputs, &omega)
        .iter()
        .map(|&x| x.value)
        .collect()
}

// FFT has a runtime of O(N*log(N)) whereas the DFT
// algorithm has a runtime of O(N^2).

// Forward Discrete Fourier Transform transforms a sequence of equally spaced
// complex numbers x_n into another sequence of complex numbers x_k.
// X_k = sum_{n=0..N-1}x_n*exp(-i2πkn/N)

// Inverse Discrete Fourier Transform
// x_n = 1/N sum_{k=0..N-1}X_kexp(i2πkn/N)

// The transformation x_n -> X_k is a transformation from configuration
// space to frequency space.

// Expressing the forward-DFT in linear algebra, we get:
// X_j = M_jk*x_k, where M_jk = exp(-i2πjk/N)
pub fn dtf_slow(x: &Vector<ComplexNumber<f64>>) -> Vector<ComplexNumber<f64>> {
    // e^(ix) = cos(x) + isin(x)
    let size: usize = x.height();
    let mut m: Matrix<ComplexNumber<f64>> = Matrix::zeros(size, size);
    for j in 0..size {
        for k in 0..size {
            m.set(
                j,
                k,
                ComplexNumber::from_exponential(
                    -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (size as f64),
                ),
            );
        }
    }
    x.mul(&m) // = Mx
}

pub fn fft(x: Vector<ComplexNumber<f64>>) -> Vector<ComplexNumber<f64>> {
    let size: usize = x.height();
    if size % 2 == 1 {
        panic!("size of input must be a power of 2");
    } else if size <= 4 {
        dtf_slow(&x)
    } else {
        let (x_even, x_odd) = x.split_by_parity();
        let (even, odd) = (fft(x_even), fft(x_odd));
        let mut factor_values = Vec::with_capacity(size);
        for i in 0..size {
            factor_values.push(ComplexNumber::from_exponential(
                -2.0 * std::f64::consts::PI * i as f64 / size as f64,
            ));
        }
        let factor = Vector::from(factor_values);
        let (fst_half_factors, snd_half_factors) = factor.split_by_middle();
        (even.clone() + odd.clone().hadamard_product(fst_half_factors))
            .concat(even + odd.hadamard_product(snd_half_factors))
    }
}

pub fn test() {
    println!("Hello World!");
    let mut vector: Vector<i64> = Vector::zeros(5);
    vector.set(0, -23);
    vector.set(1, 1);
    vector.set(2, 2);
    vector.set(3, 3);
    vector.set(4, 4);
    let mut val: i64 = vector.get(0);
    println!("val = {}", val);
    let mut matrix: Matrix<i64> = Matrix::zeros(5, 5);
    matrix.set(0, 0, 2);
    matrix.set(1, 4, 3);
    matrix.set(2, 3, -2);
    matrix.set(3, 2, -1);
    matrix.set(4, 1, 1);
    val = matrix.get(0, 0);
    println!("val = {}", val);
    println!("matrix height: {}", matrix.height());
    println!("matrix length: {}", matrix.length());
    let vector_transformed = vector.mul(&matrix);
    println!(
        "Matrix transformation M: {} -> {}",
        vector, vector_transformed
    );
    let new_vector = Vector::from(vec![1, 1]);
    let new_matrix = Matrix::try_from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
    let new_vector_transformed = new_vector.mul(&new_matrix);
    println!("M: {} -> {}", new_vector, new_vector_transformed);

    // Complex numbers
    let j = ComplexNumber::new(0, 1);
    let one = ComplexNumber::new(1, 0);
    let mul_result = j * one;
    println!("{}", mul_result);
    println!("{}", mul_result.get_real());
    println!("{}", mul_result.get_imaginary());
    println!("{:?}", mul_result);

    // Complex vectors
    let unity = ComplexNumber::new(1.0f64, 0.0f64);
    let origo = ComplexNumber::new(0.0f64, 0.0f64);
    let complex_vector = Vector::from(vec![unity, origo]);
    println!("{}", complex_vector);

    // DFT implementation, pulse at origo
    println!("Starting DFT timer");
    let now = Instant::now();
    let mut impulse_data = vec![ComplexNumber::zero(); 1024];
    impulse_data[0] = ComplexNumber::one();
    let mut impulse = Vector::from(impulse_data);

    #[allow(unused_variables)] // Ignore warnings since we only are interested in runtime
    let frequency_domain = dtf_slow(&impulse);

    // DFT implementation, pulse at one
    impulse_data = vec![ComplexNumber::zero(); 1024];
    impulse_data[0] = ComplexNumber::zero();
    impulse_data[1] = ComplexNumber::one();
    let mut impulse_new = Vector::from(impulse_data);
    #[allow(unused_variables)]
    let frequency_domain_new = dtf_slow(&impulse_new);

    println!(
        "Running DFT twice took {} milli seconds",
        now.elapsed().as_millis()
    );

    // FFT implementation, pulse at origo
    println!("Starting FFT timer");
    let mut now = Instant::now();
    impulse_data = vec![ComplexNumber::zero(); 1024];
    impulse_data[0] = ComplexNumber::one();
    impulse = Vector::from(impulse_data);

    #[allow(unused_variables)] // Ignore warnings since we only are interested in runtime
    let frequency_domain = fft(impulse);

    // FFT implementation, pulse at one
    impulse_data = vec![ComplexNumber::zero(); 1024];
    impulse_data[0] = ComplexNumber::zero();
    impulse_data[1] = ComplexNumber::one();
    impulse_new = Vector::from(impulse_data);
    #[allow(unused_variables)]
    let frequency_domain_new = fft(impulse_new);

    println!(
        "Running FFT twice took {} milli seconds",
        now.elapsed().as_millis()
    );

    // Prime field stuff
    let field: PrimeField = PrimeField::new(17);
    let prime_field_element = PrimeFieldElement::new(16, &field); // chosen since 16^2 = 1 mod 17
    let _another_prime_field_element = PrimeFieldElement::new(6, &field);
    println!(
        "The inverse of {} is {}",
        prime_field_element,
        prime_field_element.inv()
    );
    println!(
        "The Legendre symbol of {} is {}",
        prime_field_element,
        prime_field_element.legendre_symbol()
    );
    println!(
        "The {}^3 is {}",
        prime_field_element,
        prime_field_element.mod_pow(3)
    );
    let input = vec![prime_field_element, _another_prime_field_element];
    let res = ntt_fft(input.clone(), &prime_field_element);
    let inverse = intt_fft(res.clone(), &prime_field_element);
    println!("{:?} -> {:?} -> {:?}", input, res, inverse);

    let root = 2;
    let (a, _) = field.get_primitive_root_of_unity(root);
    println!(
        "Found solution to x^{} mod {} = 1: x = {}",
        root,
        field.q,
        a.unwrap()
    );

    // Primes
    let val = 11;
    println!(
        "Is {} a prime? {}.",
        val,
        PrimeFieldElement::is_prime(val, &[2, 3, 5])
    );
    let val = 21;
    println!(
        "Is {} a prime? {}.",
        val,
        PrimeFieldElement::is_prime(val, &[2, 3, 5])
    );

    // Time NTT implementation
    let range = 1048576; // 8192 ;
    let prime = 167772161; // = 5 * 2^25 + 1
    let mut ntt_input: Vec<PrimeFieldElement> = Vec::with_capacity(range);
    let new_field = PrimeField::new(prime);
    for _ in 0..range {
        ntt_input.push(PrimeFieldElement::new(
            rand::random::<u32>() as i128 % prime,
            &new_field,
        ))
    }
    let (omega, _) = new_field.get_primitive_root_of_unity(ntt_input.len() as i128);
    println!("Found omega: {}", omega.unwrap());
    now = Instant::now();
    let output = ntt_fft(ntt_input.clone(), &omega.unwrap());
    println!(
        "Running NTT on a dataset of size {} took {} milli seconds",
        range,
        now.elapsed().as_millis()
    );

    // Verify correctness
    let res = intt_fft(output, &omega.unwrap());
    for i in 0..range {
        assert_eq!(res[i].value, ntt_input[i].value);
    }

    println!("Inversion succeeded");
}

#[cfg(test)]
mod test_vectors {
    use super::*;
    use crate::shared_math::polynomial_quotient_ring::PolynomialQuotientRing;
    use crate::shared_math::prime_field_polynomial::PrimeFieldPolynomial;

    #[test]
    fn fast_polynomial_functions_property_based_test() {
        let prime = 167772161; // 2^25*5+1
        let field = PrimeField::new(prime);
        for &size in &[2, 4, 8, 1024, 2048] {
            let input_y_values: Vec<i128> = (0..size)
                .map(|_| rand::random::<u32>() as i128 % prime)
                .collect();
            let (root_option, _) = field.get_primitive_root_of_unity(size);
            let coefficients =
                fast_polynomial_interpolate(&input_y_values, prime, root_option.unwrap().value);
            let output_y_values =
                fast_polynomial_evaluate(&coefficients, prime, root_option.unwrap().value);
            assert_eq!(input_y_values, output_y_values);
            assert_ne!(input_y_values, coefficients);
            if size < 17 {
                let pqr = PolynomialQuotientRing::new(256, prime);
                let pol = PrimeFieldPolynomial {
                    coefficients,
                    pqr: &pqr,
                };
                #[allow(clippy::needless_range_loop)]
                for i in 0..size as usize {
                    let x_value = root_option.unwrap().mod_pow(i as i128);
                    let evaluated_y = pol.evaluate(&x_value);
                    assert_eq!(input_y_values[i], evaluated_y.value);
                }
            }
        }
    }

    #[test]
    fn use_ntt_for_prime_field_polynomial_evaluated() {
        let field = PrimeField::new(337i128);
        let pqr = PolynomialQuotientRing::new(256, field.q);
        let primitive_eighth_root = PrimeFieldElement::new(85, &field);
        let input_y_values = vec![3i128, 1, 4, 1, 5, 9, 2, 6];
        let fast_coefficients = fast_polynomial_interpolate(&input_y_values, field.q, 85);
        let expected_coefficients = vec![46i128, 169, 29, 149, 126, 262, 140, 93];
        assert_eq!(expected_coefficients, fast_coefficients);
        let pol = PrimeFieldPolynomial {
            coefficients: fast_coefficients.to_vec(),
            pqr: &pqr,
        };

        let x_values: Vec<PrimeFieldElement> =
            (0..8).map(|x| primitive_eighth_root.mod_pow(x)).collect();
        let values: Vec<i128> = x_values.iter().map(|x| pol.evaluate(x).value).collect();
        assert_eq!(input_y_values, values);
        let fast_values: Vec<i128> = fast_polynomial_evaluate(&pol.coefficients[..], field.q, 85);
        println!("Fast values = {:?}", fast_values);
        assert_eq!(fast_values, input_y_values);
    }

    // test vectors found here:
    // https://math.stackexchange.com/questions/1437624/number-theoretic-transform-ntt-example-not-working-out
    #[test]
    fn finite_field_fft_simple() {
        let field = PrimeField::new(5);
        let generator: PrimeFieldElement = PrimeFieldElement::new(4, &field);
        let input = vec![
            PrimeFieldElement::new(1, &field),
            PrimeFieldElement::new(4, &field),
        ];
        let expected_output = vec![
            PrimeFieldElement::new(0, &field),
            PrimeFieldElement::new(2, &field),
        ];
        let output = ntt_fft(input.clone(), &generator);
        let result = intt_fft(output.clone(), &generator);

        for i in 0..result.len() {
            assert_eq!(result[i], input[i]);
            assert_eq!(expected_output[i], output[i]);
        }
    }

    // test vectors found here:
    // https://math.stackexchange.com/questions/1437624/number-theoretic-transform-ntt-example-not-working-out
    #[test]
    fn finite_field_fft_four_elements() {
        let field = PrimeField::new(5);
        let generator: PrimeFieldElement = PrimeFieldElement::new(2, &field);
        let input = vec![
            PrimeFieldElement::new(1, &field),
            PrimeFieldElement::new(4, &field),
            PrimeFieldElement::new(0, &field),
            PrimeFieldElement::new(0, &field),
        ];
        let expected_output = vec![
            PrimeFieldElement::new(0, &field),
            PrimeFieldElement::new(4, &field),
            PrimeFieldElement::new(2, &field),
            PrimeFieldElement::new(3, &field),
        ];
        let output = ntt_fft(input.clone(), &generator);
        let result = intt_fft(output.clone(), &generator);
        for i in 0..result.len() {
            assert_eq!(result[i], input[i]);
            assert_eq!(output[i], expected_output[i]);
        }
    }

    #[test]
    fn finite_field_fft() {
        let field = PrimeField::new(17);
        let mut generator: PrimeFieldElement = PrimeFieldElement::new(0, &field);

        // Find a generator for the set Z_p^*. If g is a generator of this set,
        // then g is an Nth primitive root of unity which is the "building blocks"
        // for the NTT.
        for i in 2..17 {
            let elem = PrimeFieldElement::new(i, &field);
            if elem.legendre_symbol() != 1 {
                generator = elem;
                break;
            }
        }
        let one = PrimeFieldElement::new(1, &field);
        let zero = PrimeFieldElement::new(0, &field);
        let mut input = vec![zero; 16];
        input[0] = one; // input = [ 1, 0, 0, 0, ... ]
        let output = ntt_fft(input.clone(), &generator);
        let result = intt_fft(output, &generator);
        for i in 0..result.len() {
            assert_eq!(result[i], input[i]);
        }
    }

    #[test]
    fn finite_field_inversion_fft() {
        // Time NTT implementation
        let range = 4096; // 8192 ;
        let prime = 167772161; // = 5 * 2^25 + 1
        let mut ntt_input: Vec<PrimeFieldElement> = Vec::with_capacity(range);
        let new_field = PrimeField::new(prime);
        for _ in 0..range {
            ntt_input.push(PrimeFieldElement::new(
                rand::random::<u32>() as i128 % prime,
                &new_field,
            ))
        }
        let (omega, _) = new_field.get_primitive_root_of_unity(ntt_input.len() as i128);
        let output = ntt_fft(ntt_input.clone(), &omega.unwrap());

        // Verify that intt . ntt = I
        let res = intt_fft(output, &omega.unwrap());
        for i in 0..range {
            assert_eq!(res[i].value, ntt_input[i].value);
        }
    }

    #[test]
    fn internal() {
        let ft_size = 8;
        let mut impulse_data = vec![ComplexNumber::zero(); ft_size];
        impulse_data[0] = ComplexNumber::one();
        let mut impulse = Vector::from(impulse_data);

        let frequency_domain_dft = dtf_slow(&impulse);

        // DFT implementation, pulse at one
        impulse_data = vec![ComplexNumber::zero(); ft_size];
        impulse_data[0] = ComplexNumber::zero();
        impulse_data[1] = ComplexNumber::one();
        let mut impulse_new = Vector::from(impulse_data);
        let frequency_domain_new_dft = dtf_slow(&impulse_new);

        impulse_data = vec![ComplexNumber::zero(); ft_size];
        impulse_data[0] = ComplexNumber::one();
        impulse = Vector::from(impulse_data);

        #[allow(unused_variables)] // Ignore warnings since we only are interested in runtime
        let frequency_domain_fft = fft(impulse);

        // FFT implementation, pulse at one
        impulse_data = vec![ComplexNumber::zero(); ft_size];
        impulse_data[0] = ComplexNumber::zero();
        impulse_data[1] = ComplexNumber::one();
        impulse_new = Vector::from(impulse_data);
        #[allow(unused_variables)]
        let frequency_domain_new_fft = fft(impulse_new);
        for i in 0..ft_size {
            assert!(
                (frequency_domain_dft.get(i) - frequency_domain_fft.get(i)).get_real() < 0.0001
            );
            assert!(
                (frequency_domain_dft.get(i) - frequency_domain_fft.get(i)).get_imaginary()
                    < 0.0001
            );
            assert!(
                (frequency_domain_new_dft.get(i) - frequency_domain_new_fft.get(i)).get_real()
                    < 0.0001
            );
            assert!(
                (frequency_domain_new_dft.get(i) - frequency_domain_new_fft.get(i)).get_imaginary()
                    < 0.0001
            );
        }
    }
}
