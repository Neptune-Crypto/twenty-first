//! Provides a DB backed Vector API that is thread-safe, cached, and atomic

// We have split storage_vec into individual files, but for compatibility
// we still keep everything in mod storage_vec.
//
// To accomplish that, we keep the sub modules private, and
// add `pub use sub_module::*`.

#![allow(missing_docs)]
mod iterators;
pub mod traits;

pub type Index = u64;
pub use iterators::*;

// note: we keep ordinary_vec around because it is
// used in DocTest examples, as it does not require DB.

mod ordinary_vec;
mod ordinary_vec_private;
pub use ordinary_vec::*;

#[cfg(test)]
mod tests {

    use super::traits::*;
    use super::*;

    use itertools::Itertools;
    use rand::{Rng, RngCore};
    use std::collections::HashMap;

    /// Return a persisted vector and a regular in-memory vector with the same elements
    fn get_persisted_vec_with_length(length: Index, _name: &str) -> (OrdinaryVec<u64>, Vec<u64>) {
        let mut persisted_vec: OrdinaryVec<u64> = Default::default();
        let mut regular_vec = vec![];

        let mut rng = rand::thread_rng();
        for _ in 0..length {
            let value = rng.next_u64();
            persisted_vec.push(value);
            regular_vec.push(value);
        }

        // Sanity checks
        assert_eq!(persisted_vec.len(), regular_vec.len() as u64);

        (persisted_vec, regular_vec)
    }

    fn simple_prop<Storage: StorageVec<[u8; 13]>>(mut delegated_db_vec: Storage) {
        assert_eq!(
            0,
            delegated_db_vec.len(),
            "Length must be zero at initialization"
        );
        assert!(
            delegated_db_vec.is_empty(),
            "Vector must be empty at initialization"
        );

        // push two values, check length.
        delegated_db_vec.push([42; 13]);
        delegated_db_vec.push([44; 13]);
        assert_eq!(2, delegated_db_vec.len());
        assert!(!delegated_db_vec.is_empty());

        // Check `get`, `set`, and `get_many`
        assert_eq!([44; 13], delegated_db_vec.get(1));
        assert_eq!([42; 13], delegated_db_vec.get(0));
        assert_eq!(vec![[42; 13], [44; 13]], delegated_db_vec.get_many(&[0, 1]));
        assert_eq!(vec![[44; 13], [42; 13]], delegated_db_vec.get_many(&[1, 0]));
        assert_eq!(vec![[42; 13]], delegated_db_vec.get_many(&[0]));
        assert_eq!(vec![[44; 13]], delegated_db_vec.get_many(&[1]));
        assert_eq!(Vec::<[u8; 13]>::default(), delegated_db_vec.get_many(&[]));

        delegated_db_vec.set(0, [101; 13]);
        delegated_db_vec.set(1, [200; 13]);
        assert_eq!(vec![[101; 13]], delegated_db_vec.get_many(&[0]));
        assert_eq!(Vec::<[u8; 13]>::default(), delegated_db_vec.get_many(&[]));
        assert_eq!(vec![[200; 13]], delegated_db_vec.get_many(&[1]));
        assert_eq!(vec![[200; 13]; 2], delegated_db_vec.get_many(&[1, 1]));
        assert_eq!(vec![[200; 13]; 3], delegated_db_vec.get_many(&[1, 1, 1]));
        assert_eq!(
            vec![[200; 13], [101; 13], [200; 13]],
            delegated_db_vec.get_many(&[1, 0, 1])
        );

        // test set_many, get_many.  pass array to set_many
        delegated_db_vec.set_many([(0, [41; 13]), (1, [42; 13])]);
        // get in reverse order
        assert_eq!(vec![[42; 13], [41; 13]], delegated_db_vec.get_many(&[1, 0]));

        // set values back how they were before prior set_many() passing HashMap
        delegated_db_vec.set_many(HashMap::from([(0, [101; 13]), (1, [200; 13])]));

        // Pop two values, check length and return value of further pops
        assert_eq!([200; 13], delegated_db_vec.pop().unwrap());
        assert_eq!(1, delegated_db_vec.len());
        assert_eq!([101; 13], delegated_db_vec.pop().unwrap());
        assert!(delegated_db_vec.pop().is_none());
        assert_eq!(0, delegated_db_vec.len());
        assert!(delegated_db_vec.pop().is_none());
        assert_eq!(Vec::<[u8; 13]>::default(), delegated_db_vec.get_many(&[]));
    }

    #[test]
    fn test_simple_prop() {
        let delegated_db_vec: OrdinaryVec<[u8; 13]> = Default::default();
        simple_prop(delegated_db_vec);

        let ordinary_vec: OrdinaryVec<[u8; 13]> = Default::default();
        simple_prop(ordinary_vec);
    }

    #[test]
    fn multiple_vectors_in_one_db() {
        let mut delegated_db_vec_a: OrdinaryVec<u128> = Default::default();
        let delegated_db_vec_b: OrdinaryVec<u128> = Default::default();

        // push values to vec_a, verify vec_b is not affected
        delegated_db_vec_a.push(1000);
        delegated_db_vec_a.push(2000);
        delegated_db_vec_a.push(3000);

        assert_eq!(3, delegated_db_vec_a.len());
        assert_eq!(0, delegated_db_vec_b.len());
    }

    #[test]
    fn test_set_many() {
        let mut delegated_db_vec_a: OrdinaryVec<u128> = Default::default();

        delegated_db_vec_a.push(10);
        delegated_db_vec_a.push(20);
        delegated_db_vec_a.push(30);
        delegated_db_vec_a.push(40);

        // Allow `set_many` with empty input
        delegated_db_vec_a.set_many([]);
        assert_eq!(vec![10, 20, 30], delegated_db_vec_a.get_many(&[0, 1, 2]));

        // Perform an actual update with `set_many`
        let updates = [(0, 100), (1, 200), (2, 300), (3, 400)];
        delegated_db_vec_a.set_many(updates);

        assert_eq!(vec![100, 200, 300], delegated_db_vec_a.get_many(&[0, 1, 2]));

        #[allow(clippy::shadow_unrelated)]
        let updates = HashMap::from([(0, 1000), (1, 2000), (2, 3000)]);
        delegated_db_vec_a.set_many(updates);

        assert_eq!(
            vec![1000, 2000, 3000],
            delegated_db_vec_a.get_many(&[0, 1, 2])
        );
    }

    #[test]
    fn test_set_all() {
        let mut delegated_db_vec_a: OrdinaryVec<u128> = Default::default();

        delegated_db_vec_a.push(10);
        delegated_db_vec_a.push(20);
        delegated_db_vec_a.push(30);

        let updates = [100, 200, 300];
        delegated_db_vec_a.set_all(updates);

        assert_eq!(vec![100, 200, 300], delegated_db_vec_a.get_many(&[0, 1, 2]));

        #[allow(clippy::shadow_unrelated)]
        let updates = vec![1000, 2000, 3000];
        delegated_db_vec_a.set_all(updates);

        assert_eq!(
            vec![1000, 2000, 3000],
            delegated_db_vec_a.get_many(&[0, 1, 2])
        );
    }

    #[test]
    fn get_many_ordering_of_outputs() {
        let mut delegated_db_vec_a: OrdinaryVec<u128> = Default::default();

        delegated_db_vec_a.push(1000);
        delegated_db_vec_a.push(2000);
        delegated_db_vec_a.push(3000);

        // Test `get_many` ordering of outputs
        assert_eq!(
            vec![1000, 2000, 3000],
            delegated_db_vec_a.get_many(&[0, 1, 2])
        );
        assert_eq!(
            vec![2000, 3000, 1000],
            delegated_db_vec_a.get_many(&[1, 2, 0])
        );
        assert_eq!(
            vec![3000, 1000, 2000],
            delegated_db_vec_a.get_many(&[2, 0, 1])
        );
        assert_eq!(
            vec![2000, 1000, 3000],
            delegated_db_vec_a.get_many(&[1, 0, 2])
        );
        assert_eq!(
            vec![3000, 2000, 1000],
            delegated_db_vec_a.get_many(&[2, 1, 0])
        );
        assert_eq!(
            vec![1000, 3000, 2000],
            delegated_db_vec_a.get_many(&[0, 2, 1])
        );
    }

    #[test]
    fn delegated_vec_pbt() {
        let (mut persisted_vector, mut normal_vector) =
            get_persisted_vec_with_length(10000, "vec 1");

        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            match rng.gen_range(0..=4) {
                0 => {
                    // `push`
                    let push_val = rng.next_u64();
                    persisted_vector.push(push_val);
                    normal_vector.push(push_val);
                }
                1 => {
                    // `pop`
                    let persisted_pop_val = persisted_vector.pop().unwrap();
                    let normal_pop_val = normal_vector.pop().unwrap();
                    assert_eq!(persisted_pop_val, normal_pop_val);
                }
                2 => {
                    // `get_many`
                    let index = rng.gen_range(0..normal_vector.len());
                    assert_eq!(Vec::<u64>::default(), persisted_vector.get_many(&[]));
                    assert_eq!(normal_vector[index], persisted_vector.get(index as u64));
                    assert_eq!(
                        vec![normal_vector[index]],
                        persisted_vector.get_many(&[index as u64])
                    );
                    assert_eq!(
                        vec![normal_vector[index], normal_vector[index]],
                        persisted_vector.get_many(&[index as u64, index as u64])
                    );
                }
                3 => {
                    // `set`
                    let value = rng.next_u64();
                    let index = rng.gen_range(0..normal_vector.len());
                    normal_vector[index] = value;
                    persisted_vector.set(index as u64, value);
                }
                4 => {
                    // `set_many`
                    let indices: Vec<u64> = (0..rng.gen_range(0..10))
                        .map(|_| rng.gen_range(0..normal_vector.len() as u64))
                        .unique()
                        .collect();
                    let values: Vec<u64> = (0..indices.len()).map(|_| rng.next_u64()).collect_vec();
                    let update: Vec<(u64, u64)> =
                        indices.into_iter().zip_eq(values.into_iter()).collect();
                    for (key, val) in update.iter() {
                        normal_vector[*key as usize] = *val;
                    }
                    persisted_vector.set_many(update);
                }
                _ => unreachable!(),
            }
        }

        // Check equality after above loop
        assert_eq!(normal_vector.len(), persisted_vector.len() as usize);
        for (i, nvi) in normal_vector.iter().enumerate() {
            assert_eq!(*nvi, persisted_vector.get(i as u64));
        }

        // Check equality using `get_many`
        assert_eq!(
            normal_vector,
            persisted_vector.get_many(&(0..normal_vector.len() as u64).collect_vec())
        );
    }

    #[should_panic(expected = "Out-of-bounds. Got index 3 but length was 1.")]
    #[test]
    fn panic_on_out_of_bounds_get() {
        let (delegated_db_vec, _) = get_persisted_vec_with_length(1, "unit test vec 0");
        delegated_db_vec.get(3);
    }

    #[should_panic(expected = "Out-of-bounds. Got index 3 but length was 1.")]
    #[test]
    fn panic_on_out_of_bounds_get_many() {
        let (delegated_db_vec, _) = get_persisted_vec_with_length(1, "unit test vec 0");
        delegated_db_vec.get_many(&[3]);
    }

    #[should_panic(expected = "index out of bounds: the len is 1 but the index is 1")]
    #[test]
    fn panic_on_out_of_bounds_set() {
        let (mut delegated_db_vec, _) = get_persisted_vec_with_length(1, "unit test vec 0");
        delegated_db_vec.set(1, 3000);
    }

    #[should_panic(expected = "index out of bounds: the len is 1 but the index is 1")]
    #[test]
    fn panic_on_out_of_bounds_set_many() {
        let (mut delegated_db_vec, _) = get_persisted_vec_with_length(1, "unit test vec 0");

        // attempt to set 2 values, when only one is in vector.
        delegated_db_vec.set_many([(0, 0), (1, 1)]);
    }

    #[should_panic(expected = "size-mismatch.  input has 2 elements and target has 1 elements.")]
    #[test]
    fn panic_on_size_mismatch_set_all() {
        let (mut delegated_db_vec, _) = get_persisted_vec_with_length(1, "unit test vec 0");

        // attempt to set 2 values, when only one is in vector.
        delegated_db_vec.set_all([1, 2]);
    }

    #[should_panic(expected = "Out-of-bounds. Got index 11 but length was 11.")]
    #[test]
    fn panic_on_out_of_bounds_get_even_though_value_exists_in_persistent_memory() {
        let (mut delegated_db_vec, _) = get_persisted_vec_with_length(12, "unit test vec 0");
        delegated_db_vec.pop();
        delegated_db_vec.get(11);
    }

    #[should_panic(expected = "index out of bounds: the len is 11 but the index is 11")]
    #[test]
    fn panic_on_out_of_bounds_set_even_though_value_exists_in_persistent_memory() {
        let (mut delegated_db_vec, _) = get_persisted_vec_with_length(12, "unit test vec 0");
        delegated_db_vec.pop();
        delegated_db_vec.set(11, 5000);
    }
}
