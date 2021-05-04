mod complex_number;
mod vector;
use complex_number::ComplexNumber;
mod prime_field_element;
use num_traits::{One, Zero};
use prime_field_element::{PrimeField, PrimeFieldElement};
use std::convert::TryFrom;
use std::time::Instant;
use vector::{Matrix, Vector};

pub fn dft_finite_fields<'a>(
    x: &[PrimeFieldElement<'a>],
    omega: &PrimeFieldElement<'a>,
) -> Vec<PrimeFieldElement<'a>> {
    vec![x[0] + x[1], x[0] + *omega * x[1]]
}

fn ntt_fft_helper<'a>(
    x: Vec<PrimeFieldElement<'a>>,
    omega: &PrimeFieldElement<'a>,
) -> Vec<PrimeFieldElement<'a>> {
    let size: usize = x.len();
    if size % 2 == 1 {
        panic!("size of input must be a power of 2");
    } else if size == 2 {
        dft_finite_fields(&x, omega)
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

        // Recursive call
        let omega_squared = *omega * *omega;
        let (even, odd) = (
            ntt_fft_helper(x_even, &omega_squared),
            ntt_fft_helper(x_odd, &omega_squared),
        );

        // Calculate all values omega^j, for j=0..size
        let mut factor_values: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(size);
        for j in 0..size {
            let pow = omega.mod_pow(j as i128);
            factor_values.push(pow);
        }

        // split by middle
        let mut fst_half_factors: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(size / 2);
        let mut snd_half_factors: Vec<PrimeFieldElement<'a>> = Vec::with_capacity(size / 2);
        #[allow(clippy::needless_range_loop)]
        for i in 0..(size / 2) {
            fst_half_factors.push(factor_values[i]);
        }
        #[allow(clippy::needless_range_loop)]
        for i in (size / 2)..size {
            snd_half_factors.push(factor_values[i]);
        }

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
    if omega.mod_pow(x.len() as i128).value != 1 {
        panic!("ntt_fft called with ω^len != 1. Got: {:?}", omega);
    }
    ntt_fft_helper(x, omega)
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
    let mut vector: Vector<i128> = Vector::zeros(5);
    vector.set(0, -23);
    vector.set(1, 1);
    vector.set(2, 2);
    vector.set(3, 3);
    vector.set(4, 4);
    let mut val: i128 = vector.get(0);
    println!("val = {}", val);
    let mut matrix: Matrix<i128> = Matrix::zeros(5, 5);
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
    let now = Instant::now();
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
}

#[cfg(test)]
mod test_vectors {
    // test vectors found here:
    // https://math.stackexchange.com/questions/1437624/number-theoretic-transform-ntt-example-not-working-out
    #[test]
    fn finite_field_fft_simple() {
        use super::*;
        let field = PrimeField::new(5);
        let generator: PrimeFieldElement = PrimeFieldElement::new(4, &field);
        let input = vec![
            PrimeFieldElement::new(1, &field),
            PrimeFieldElement::new(4, &field),
        ];
        let output = ntt_fft(input.clone(), &generator);
        let result = intt_fft(output, &generator);

        for i in 0..result.len() {
            assert_eq!(result[i], input[i]);
        }
    }

    // test vectors found here:
    // https://math.stackexchange.com/questions/1437624/number-theoretic-transform-ntt-example-not-working-out
    #[test]
    fn finite_field_fft_four_elements() {
        use super::*;
        let field = PrimeField::new(5);
        let generator: PrimeFieldElement = PrimeFieldElement::new(2, &field);
        let input = vec![
            PrimeFieldElement::new(1, &field),
            PrimeFieldElement::new(4, &field),
            PrimeFieldElement::new(0, &field),
            PrimeFieldElement::new(0, &field),
        ];
        let output = ntt_fft(input.clone(), &generator);
        let result = intt_fft(output, &generator);
        for i in 0..result.len() {
            assert_eq!(result[i], input[i]);
        }
    }

    #[test]
    fn finite_field_fft() {
        use super::*;
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
    fn internal() {
        use super::*;

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
