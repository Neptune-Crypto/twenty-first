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
    println!("{:?}", mul_result);

    // Complex vectors
    let unity = ComplexNumber::new(1.0f64, 0.0f64);
    let origo = ComplexNumber::new(0.0f64, 0.0f64);
    let complex_vector = Vector::from(vec![unity, origo]);
    println!("{}", complex_vector);

    // DFT implementation, pulse at origo
    println!("Starting timer");
    let now = Instant::now();
    let mut impulse_data = vec![ComplexNumber::zero(); 1024];
    impulse_data[0] = ComplexNumber::one();
    let impulse = Vector::from(impulse_data);

    #[allow(unused_variables)] // Ignore warnings since we only are interested in runtime
    let frequency_domain = dtf_slow(&impulse);

    // DFT implementation, pulse at one
    impulse_data = vec![ComplexNumber::zero(); 1024];
    impulse_data[0] = ComplexNumber::zero();
    impulse_data[1] = ComplexNumber::one();
    let impulse_new = Vector::from(impulse_data);
    #[allow(unused_variables)]
    let frequency_domain_new = dtf_slow(&impulse_new);

    println!(
        "Running DFT twice took {} milli seconds",
        now.elapsed().as_millis()
    );
    // println!("DFT: {} -> {}", impulse, frequency_domain);
    // println!("DFT: {} -> {}", impulse, frequency_domain_new);
}
