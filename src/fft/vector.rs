use std::convert::TryFrom;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct Matrix<T: num_traits::Num + Clone> {
    pub length: usize,
    pub height: usize,
    values: Vec<T>,
}

impl<T: num_traits::Num + Clone> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = &'static str;
    // Rewrite using match and pattern matching
    fn try_from(rows: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        match rows.as_slice() {
            [] => Err("Matrix must have at least one row. Got empty vector"),
            [first_row, ..] => {
                if rows.iter().any(|x| first_row.len() != x.len()) {
                    Err("All rows must have the same length.")
                } else {
                    Ok(Matrix {
                        height: rows.len(),
                        length: first_row.len(),
                        values: rows.into_iter().flatten().collect(),
                    })
                }
            }
        }
    }
}

impl<U> Matrix<U>
where
    U: num_traits::Num + Clone + Copy,
{
    pub fn zeros(length: usize, height: usize) -> Self {
        Matrix {
            length,
            height,
            values: vec![num_traits::zero(); length * height],
        }
    }

    // Will panic if row/column exceeds height/length
    pub fn set(&mut self, row: usize, column: usize, value: U) {
        if row >= self.height || column >= self.length {
            panic!(
                "Index outside of allowable range. Got: ({},{}). Dimensions: ({},{})",
                row, column, self.height, self.length
            );
        }
        self.values[row * self.length + column] = value;
    }

    pub fn get(&self, row: usize, column: usize) -> U {
        if row >= self.height || column >= self.length {
            panic!(
                "Index outside of allowable range. Got: ({},{}). Dimensions: ({},{})",
                row, column, self.height, self.length
            );
        }
        self.values[row * self.length + column]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T: num_traits::Num + Clone + Copy> {
    values: Vec<T>,
    pub height: usize,
}

impl<T: num_traits::Num + Clone + Copy + Display> From<Vec<T>> for Vector<T> {
    fn from(values: Vec<T>) -> Self {
        Vector {
            height: values.len(),
            values,
        }
    }
}

impl<T: num_traits::Num + Clone + Copy + Display> std::fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let output = self
            .values
            .iter()
            .map(|x: &T| format!("{}", *x))
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "({})", output)
    }
}

impl<U> Vector<U>
where
    U: num_traits::Num + Clone + Copy + Display,
{
    pub fn zeros(height: usize) -> Self {
        Vector {
            height,
            values: vec![num_traits::zero(); height],
        }
    }

    pub fn set(&mut self, row: usize, value: U) {
        if row >= self.height {
            panic!(
                "Index outside of allowable range. Got: index={}. height: {}",
                row, self.height
            );
        }
        self.values[row] = value;
    }

    pub fn get(&self, row: usize) -> U {
        if row >= self.height {
            panic!(
                "Index outside of allowable range. Got: index={}. height: {}",
                row, self.height
            );
        }
        self.values[row]
    }

    // Returns a new vector, as the allocation for a new vector is needed
    // to prevent a wrong calculation anyway.
    // Also: the height may change as a result of matrix multiplication.
    pub fn mul(&self, matrix: &Matrix<U>) -> Vector<U> {
        if self.height != matrix.length {
            panic!(
                "Incompatible matrix and vector size. vector height = {}, matrix length = {}",
                self.height, matrix.length
            );
        }
        let mut new_vector = Vector {
            values: Vec::with_capacity(matrix.height),
            height: matrix.height,
        };
        for i in 0..matrix.height {
            let mut res: U = num_traits::zero();
            for j in 0..matrix.length {
                res = res + self.values[j] * matrix.values[j + matrix.length * i];
            }
            new_vector.values.push(res);
        }
        new_vector
    }
}

#[cfg(test)]
mod test_vectors {

    #[test]
    fn internal() {
        use super::*;
        let mut vector: Vector<i128> = Vector::zeros(5);
        vector.set(0, -23);
        vector.set(1, 1);
        vector.set(2, 2);
        vector.set(3, 3);
        vector.set(4, 4);
        assert_eq!(vector.get(0), -23i128);
        assert_eq!(vector.get(4), 4i128);

        let mut matrix: Matrix<i128> = Matrix::zeros(5, 5);
        matrix.set(0, 0, 2);
        matrix.set(1, 4, 3);
        matrix.set(2, 3, -2);
        matrix.set(3, 2, -1);
        matrix.set(4, 1, 1);
        assert_eq!(matrix.get(0, 0), 2i128);
        assert_eq!(matrix.get(0, 1), 0i128);
        assert_eq!(matrix.get(3, 2), -1i128);
        let vector_transformed = vector.mul(&matrix);
        let mut expected_vector = Vector::from(vec![-46, 12, -6, -2, 1]);
        assert_eq!(expected_vector, vector_transformed);
        let new_vector = Vector::from(vec![1, 1]);

        let new_matrix = Matrix::try_from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
        let new_vector_transformed = new_vector.mul(&new_matrix);
        expected_vector = Vector::from(vec![3i128, 7, 11]);
        assert_eq!(expected_vector, new_vector_transformed);

        // Verify that all row lengths must be equal when creating matrices
        assert!(Matrix::try_from(vec![vec![1, 2], vec![3, 4, 2], vec![5, 6]]).is_err());
    }
}
