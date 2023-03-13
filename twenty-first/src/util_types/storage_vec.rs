use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use rusty_leveldb::{WriteBatch, DB};
use serde::{de::DeserializeOwned, Serialize};

pub type IndexType = u64;

pub trait StorageVec<T> {
    fn is_empty(&self) -> bool;
    fn len(&self) -> IndexType;
    fn get(&self, index: IndexType) -> T;
    fn set(&mut self, index: IndexType, value: T);
    fn pop(&mut self) -> Option<T>;
    fn push(&mut self, value: T);
}

pub enum WriteElement<T: Serialize + DeserializeOwned> {
    OverWrite((IndexType, T)),
    Push(T),
    Pop,
}

pub struct RustyLevelDbVec<T: Serialize + DeserializeOwned> {
    key_prefix: u8,
    db: Arc<Mutex<DB>>,
    write_queue: VecDeque<WriteElement<T>>,
    length: IndexType,
    cache: HashMap<IndexType, T>,
    name: String,
}

impl<T: Serialize + DeserializeOwned + Clone> StorageVec<T> for RustyLevelDbVec<T> {
    fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn len(&self) -> IndexType {
        self.length
    }

    fn get(&self, index: IndexType) -> T {
        // Disallow getting values out-of-bounds
        assert!(
            index < self.len(),
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self.length,
            self.name
        );

        // try cache first
        if self.cache.contains_key(&index) {
            return self.cache[&index].clone();
        }

        // then try persistent storage
        let db_key = self.get_index_key(index);
        let db_val = self.db.lock().unwrap().get(&db_key).unwrap_or_else(|| {
            panic!(
                "Element with index {index} does not exist in {}. This should not happen",
                self.name
            )
        });
        bincode::deserialize(&db_val).unwrap()
    }

    fn set(&mut self, index: IndexType, value: T) {
        // Disallow setting values out-of-bounds
        assert!(
            index < self.len(),
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self.length,
            self.name
        );

        let _old_value = self.cache.insert(index, value.clone());

        // TODO: If `old_value` is Some(*) use it to remove the corresponding
        // element in the `write_queue` to reduce disk IO.

        self.write_queue
            .push_back(WriteElement::OverWrite((index, value)));
    }

    fn pop(&mut self) -> Option<T> {
        // add to write queue
        self.write_queue.push_back(WriteElement::Pop);

        // If vector is empty, return None
        if self.length == 0 {
            return None;
        }

        // Update length
        self.length -= 1;

        // try cache first
        if self.cache.contains_key(&self.length) {
            self.cache.remove(&self.length)
        } else {
            // then try persistent storage
            let db_key = self.get_index_key(self.length);
            self.db
                .lock()
                .unwrap()
                .get(&db_key)
                .map(|bytes| bincode::deserialize(&bytes).unwrap())
        }
    }

    fn push(&mut self, value: T) {
        // add to write queue
        self.write_queue
            .push_back(WriteElement::Push(value.clone()));

        // record in cache
        let _old_value = self.cache.insert(self.length, value);

        // TODO: if `old_value` is Some(_) then use it to remove the corresponding
        // element from the `write_queue` to reduce disk operations

        // update length
        self.length += 1;
    }
}

impl<T: Serialize + DeserializeOwned> RustyLevelDbVec<T> {
    // Return the key used to store the length of the persisted vector
    fn get_length_key(key_prefix: u8) -> [u8; 2] {
        const LENGTH_KEY: u8 = 0u8;
        [key_prefix, LENGTH_KEY]
    }

    /// Return the length at the last write to disk
    fn persisted_length(&self) -> IndexType {
        let key = Self::get_length_key(self.key_prefix);
        match self.db.lock().unwrap().get(&key) {
            Some(value) => bincode::deserialize(&value).unwrap(),
            None => 0,
        }
    }

    /// Return the level-DB key used to store the element at an index
    fn get_index_key(&self, index: IndexType) -> [u8; 9] {
        vec![vec![self.key_prefix], bincode::serialize(&index).unwrap()]
            .concat()
            .try_into()
            .unwrap()
    }

    pub fn new(db: Arc<Mutex<DB>>, key_prefix: u8, name: &str) -> Self {
        let length_key = Self::get_length_key(key_prefix);
        let length = match db.lock().unwrap().get(&length_key) {
            Some(length_bytes) => bincode::deserialize(&length_bytes).unwrap(),
            None => 0,
        };
        let cache = HashMap::new();
        Self {
            key_prefix,
            db,
            write_queue: VecDeque::default(),
            length,
            cache,
            name: name.to_string(),
        }
    }

    /// Collect all added elements that have not yet bit persisted
    pub fn pull_queue(&mut self, write_batch: &mut WriteBatch) {
        let original_length = self.persisted_length();
        let mut length = original_length;
        while let Some(write_element) = self.write_queue.pop_front() {
            match write_element {
                WriteElement::OverWrite((i, t)) => {
                    let key = self.get_index_key(i);
                    let value = bincode::serialize(&t).unwrap();
                    write_batch.put(&key, &value);
                }
                WriteElement::Push(t) => {
                    let key =
                        vec![vec![self.key_prefix], bincode::serialize(&length).unwrap()].concat();
                    length += 1;
                    let value = bincode::serialize(&t).unwrap();
                    write_batch.put(&key, &value);
                }
                WriteElement::Pop => {
                    let key = vec![
                        vec![self.key_prefix],
                        bincode::serialize(&(length - 1)).unwrap(),
                    ]
                    .concat();
                    length -= 1;
                    write_batch.delete(&key);
                }
            };
        }

        if original_length != length {
            let key = Self::get_length_key(self.key_prefix);
            write_batch.put(&key, &bincode::serialize(&self.length).unwrap());
        }

        self.cache.clear();
    }
}

pub struct OrdinaryVec<T>(Vec<T>);

impl<T: Clone> StorageVec<T> for OrdinaryVec<T> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn len(&self) -> IndexType {
        self.0.len() as IndexType
    }

    fn get(&self, index: IndexType) -> T {
        self.0[index as usize].clone()
    }

    fn set(&mut self, index: IndexType, value: T) {
        self.0[index as usize] = value;
    }

    fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    fn push(&mut self, value: T) {
        self.0.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, RngCore};
    use rusty_leveldb::DB;

    fn get_test_db() -> Arc<Mutex<DB>> {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        Arc::new(Mutex::new(db))
    }

    /// Return a persisted vector and a regular in-memory vector with the same elements
    fn get_persisted_vec_with_length(
        length: IndexType,
        name: &str,
    ) -> (RustyLevelDbVec<u64>, Vec<u64>, Arc<Mutex<DB>>) {
        let db = get_test_db();
        let mut persisted_vec = RustyLevelDbVec::new(db.clone(), 0, name);
        let mut regular_vec = vec![];

        let mut rng = rand::thread_rng();
        for _ in 0..length {
            let value = rng.next_u64();
            persisted_vec.push(value);
            regular_vec.push(value);
        }

        let mut write_batch = WriteBatch::new();
        persisted_vec.pull_queue(&mut write_batch);
        assert!(db.lock().unwrap().write(write_batch, true).is_ok());

        // Sanity checks
        assert!(persisted_vec.cache.is_empty());
        assert_eq!(persisted_vec.len(), regular_vec.len() as u64);

        (persisted_vec, regular_vec, db)
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

        // Check `get` and `set`
        assert_eq!([44; 13], delegated_db_vec.get(1));
        assert_eq!([42; 13], delegated_db_vec.get(0));

        delegated_db_vec.set(0, [101; 13]);
        delegated_db_vec.set(1, [200; 13]);

        // Pop two values, check length and return value of further pops
        assert_eq!([200; 13], delegated_db_vec.pop().unwrap());
        assert_eq!(1, delegated_db_vec.len());
        assert_eq!([101; 13], delegated_db_vec.pop().unwrap());
        assert!(delegated_db_vec.pop().is_none());
        assert_eq!(0, delegated_db_vec.len());
        assert!(delegated_db_vec.pop().is_none());
    }

    #[test]
    fn test_simple_prop() {
        let db = get_test_db();
        let delegated_db_vec: RustyLevelDbVec<[u8; 13]> =
            RustyLevelDbVec::new(db, 0, "unit test vec 0");
        simple_prop(delegated_db_vec);

        let ordinary_vec = OrdinaryVec::<[u8; 13]>(vec![]);
        simple_prop(ordinary_vec);
    }

    #[test]
    fn multiple_vectors_in_one_db() {
        let db = get_test_db();
        let mut delegated_db_vec_a: RustyLevelDbVec<u128> =
            RustyLevelDbVec::new(db.clone(), 0, "unit test vec a");
        let mut delegated_db_vec_b: RustyLevelDbVec<u128> =
            RustyLevelDbVec::new(db.clone(), 1, "unit test vec b");

        // push values to vec_a, verify vec_b is not affected
        delegated_db_vec_a.push(1000);
        delegated_db_vec_a.push(2000);
        delegated_db_vec_a.push(3000);

        assert_eq!(3, delegated_db_vec_a.len());
        assert_eq!(0, delegated_db_vec_b.len());
        assert_eq!(3, delegated_db_vec_a.cache.len());
        assert!(delegated_db_vec_b.cache.is_empty());

        // Get all entries to write to database. Write all entries.
        assert_eq!(0, delegated_db_vec_a.persisted_length());
        assert_eq!(0, delegated_db_vec_b.persisted_length());
        assert_eq!(3, delegated_db_vec_a.len());
        assert_eq!(0, delegated_db_vec_b.len());
        let mut write_batch = WriteBatch::new();
        delegated_db_vec_a.pull_queue(&mut write_batch);
        delegated_db_vec_b.pull_queue(&mut write_batch);
        assert_eq!(0, delegated_db_vec_a.persisted_length());
        assert_eq!(0, delegated_db_vec_b.persisted_length());
        assert_eq!(3, delegated_db_vec_a.len());
        assert_eq!(0, delegated_db_vec_b.len());

        assert!(
            db.lock().unwrap().write(write_batch, true).is_ok(),
            "DB write must succeed"
        );
        assert_eq!(3, delegated_db_vec_a.persisted_length());
        assert_eq!(0, delegated_db_vec_b.persisted_length());
        assert_eq!(3, delegated_db_vec_a.len());
        assert_eq!(0, delegated_db_vec_b.len());
        assert!(delegated_db_vec_a.cache.is_empty());
        assert!(delegated_db_vec_b.cache.is_empty());
    }

    #[test]
    fn delegated_vec_pbt() {
        let (mut persisted_vector, mut normal_vector, db) =
            get_persisted_vec_with_length(10000, "vec 1");

        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            match rng.gen_range(0..4) {
                0 => {
                    let push_val = rng.next_u64();
                    persisted_vector.push(push_val);
                    normal_vector.push(push_val);
                }
                1 => {
                    let persisted_pop_val = persisted_vector.pop().unwrap();
                    let normal_pop_val = normal_vector.pop().unwrap();
                    assert_eq!(persisted_pop_val, normal_pop_val);
                }
                2 => {
                    let index = rng.gen_range(0..normal_vector.len());
                    assert_eq!(normal_vector[index], persisted_vector.get(index as u64));
                }
                3 => {
                    let value = rng.next_u64();
                    let index = rng.gen_range(0..normal_vector.len());
                    normal_vector[index] = value;
                    persisted_vector.set(index as u64, value);
                }
                _ => panic!("Bad range"),
            }
        }

        // Check equality after above loop
        assert_eq!(normal_vector.len(), persisted_vector.len() as usize);
        for (i, nvi) in normal_vector.iter().enumerate() {
            assert_eq!(*nvi, persisted_vector.get(i as u64));
        }

        // Check equality after persisting updates
        let mut write_batch = WriteBatch::new();
        persisted_vector.pull_queue(&mut write_batch);
        db.lock().unwrap().write(write_batch, true).unwrap();

        assert_eq!(normal_vector.len(), persisted_vector.len() as usize);
        assert_eq!(
            normal_vector.len(),
            persisted_vector.persisted_length() as usize
        );

        // Check equality after write
        assert_eq!(normal_vector.len(), persisted_vector.len() as usize);
        for (i, nvi) in normal_vector.iter().enumerate() {
            assert_eq!(*nvi, persisted_vector.get(i as u64));
        }
    }

    #[should_panic(
        expected = "Out-of-bounds. Got 3 but length was 1. persisted vector name: unit test vec 0"
    )]
    #[test]
    fn panic_on_out_of_bounds_get() {
        let (delegated_db_vec, _, _) = get_persisted_vec_with_length(1, "unit test vec 0");
        delegated_db_vec.get(3);
    }

    #[should_panic(
        expected = "Out-of-bounds. Got 1 but length was 1. persisted vector name: unit test vec 0"
    )]
    #[test]
    fn panic_on_out_of_bounds_set() {
        let (mut delegated_db_vec, _, _) = get_persisted_vec_with_length(1, "unit test vec 0");
        delegated_db_vec.set(1, 3000);
    }

    #[should_panic(
        expected = "Out-of-bounds. Got 11 but length was 11. persisted vector name: unit test vec 0"
    )]
    #[test]
    fn panic_on_out_of_bounds_get_even_though_value_exists_in_persistent_memory() {
        let (mut delegated_db_vec, _, _) = get_persisted_vec_with_length(12, "unit test vec 0");
        delegated_db_vec.pop();
        delegated_db_vec.get(11);
    }

    #[should_panic(
        expected = "Out-of-bounds. Got 11 but length was 11. persisted vector name: unit test vec 0"
    )]
    #[test]
    fn panic_on_out_of_bounds_set_even_though_value_exists_in_persistent_memory() {
        let (mut delegated_db_vec, _, _) = get_persisted_vec_with_length(12, "unit test vec 0");
        delegated_db_vec.pop();
        delegated_db_vec.set(11, 5000);
    }
}
