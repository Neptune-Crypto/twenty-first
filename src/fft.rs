mod complex_number;
mod vector;
use complex_number::ComplexNumber;
use num_traits::{One, Zero};
use std::convert::TryFrom;
use std::time::Instant;
use vector::{Matrix, Vector};

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
                    -2.0 * std::f64::consts::PI * k as f64 * j as f64 / size as f64,
                ),
            );
        }
    }
    x.mul(&m)
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
        // let factor = ComplexNumber::from_exponential(-2.0 * std::f64::consts::PI);
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
    // let mut new_matrix = Matrix::try_from(rows: Vec<Vec<T>>)
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
    // println!("DFT: {} -> {}", impulse, frequency_domain);
    // println!("DFT: {} -> {}", impulse, frequency_domain_new);
}

#[cfg(test)]
mod test_vectors {

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
        println!("ft_size = {}", ft_size);
        println!("dft_height = {}", frequency_domain_dft.height());
        println!("fft_height = {}", frequency_domain_fft.height());
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
