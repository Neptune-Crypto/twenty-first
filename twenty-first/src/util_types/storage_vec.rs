use crate::util_types::level_db::DB;
use itertools::Itertools;
use leveldb::{batch::WriteBatch, options::ReadOptions};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

pub type Index = u64;

pub trait StorageVec<T> {
    /// check if collection is empty
    fn is_empty(&self) -> bool;

    /// get collection length
    fn len(&self) -> Index;

    /// get single element at index
    fn get(&self, index: Index) -> T;

    /// get multiple elements matching indices
    fn get_many(&self, indices: &[Index]) -> Vec<T>;

    /// get all elements
    fn get_all(&self) -> Vec<T>;

    /// set a single element.
    fn set(&mut self, index: Index, value: T);

    /// set multiple elements.
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>);

    /// set all elements with a simple list of values in an array or Vec.
    ///
    /// calls ::set_many() internally.
    ///
    /// panics if input length does not match target length.
    ///
    /// note: casts the array's indexes from usize to Index.
    fn set_all(&mut self, vals: &[T])
    where
        T: Clone,
    {
        assert!(
            vals.len() as Index == self.len(),
            "size-mismatch.  input has {} elements and target has {} elements.",
            vals.len(),
            self.len(),
        );

        self.set_many(
            vals.iter()
                .enumerate()
                .map(|(i, v)| (i as Index, v.clone())),
        );
    }

    /// pop an element from end of collection
    fn pop(&mut self) -> Option<T>;

    /// push an element to end of collection
    fn push(&mut self, value: T);
}

pub enum WriteElement<T: Serialize + DeserializeOwned> {
    OverWrite((Index, T)),
    Push(T),
    Pop,
}

pub struct RustyLevelDbVec<T: Serialize + DeserializeOwned> {
    key_prefix: u8,
    db: Arc<DB>,
    write_queue: VecDeque<WriteElement<T>>,
    length: Index,
    cache: HashMap<Index, T>,
    name: String,
}

impl<T: Serialize + DeserializeOwned + Clone> StorageVec<T> for RustyLevelDbVec<T> {
    fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn len(&self) -> Index {
        self.length
    }

    fn get(&self, index: Index) -> T {
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
        let db_val = self
            .db
            .get_u8(&ReadOptions::new(), &db_key)
            .unwrap_or_else(|_| {
                panic!(
                    "Element with index {index} does not exist in {}. This should not happen",
                    self.name
                )
            })
            .unwrap();
        bincode::deserialize(&db_val).unwrap()
    }

    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        fn sort_to_match_requested_index_order<T>(indexed_elements: HashMap<usize, T>) -> Vec<T> {
            let mut elements = indexed_elements.into_iter().collect_vec();
            elements.sort_unstable_by_key(|&(index_position, _)| index_position);
            elements.into_iter().map(|(_, element)| element).collect()
        }

        assert!(
            indices.iter().all(|x| *x < self.len()),
            "Out-of-bounds. Got indices {indices:?} but length was {}. persisted vector name: {}",
            self.len(),
            self.name
        );

        let (indices_of_elements_in_cache, indices_of_elements_not_in_cache): (Vec<_>, Vec<_>) =
            indices
                .iter()
                .copied()
                .enumerate()
                .partition(|&(_, index)| self.cache.contains_key(&index));

        let mut fetched_elements = HashMap::with_capacity(indices.len());
        for (index_position, index) in indices_of_elements_in_cache {
            let element = self.cache[&index].clone();
            fetched_elements.insert(index_position, element);
        }

        let no_need_to_lock_database = indices_of_elements_not_in_cache.is_empty();
        if no_need_to_lock_database {
            return sort_to_match_requested_index_order(fetched_elements);
        }

        // let db_reader = self.db;

        let elements_fetched_from_db = indices_of_elements_not_in_cache
            .iter()
            .map(|&(_, index)| self.get_index_key(index))
            .map(|key| self.db.get_u8(&ReadOptions::new(), &key).unwrap().unwrap())
            .map(|db_element| bincode::deserialize(&db_element).unwrap());

        let indexed_fetched_elements_from_db = indices_of_elements_not_in_cache
            .iter()
            .map(|&(index_position, _)| index_position)
            .zip_eq(elements_fetched_from_db);
        fetched_elements.extend(indexed_fetched_elements_from_db);

        sort_to_match_requested_index_order(fetched_elements)
    }

    /// Return all stored elements in a vector, whose index matches the StorageVec's.
    /// It's the caller's responsibility that there is enough memory to store all elements.
    fn get_all(&self) -> Vec<T> {
        let length = self.len();

        let (indices_of_elements_in_cache, indices_of_elements_not_in_cache): (Vec<_>, Vec<_>) =
            (0..length).partition(|index| self.cache.contains_key(index));

        let mut fetched_elements: Vec<Option<T>> = vec![None; length as usize];
        for index in indices_of_elements_in_cache {
            let element = self.cache[&index].clone();
            fetched_elements[index as usize] = Some(element);
        }

        let no_need_to_lock_database = indices_of_elements_not_in_cache.is_empty();
        if no_need_to_lock_database {
            return fetched_elements
                .into_iter()
                .map(|x| x.unwrap())
                .collect_vec();
        }

        // let db_reader = self.db;
        for index in indices_of_elements_not_in_cache {
            let key = self.get_index_key(index);
            let element = self.db.get_u8(&ReadOptions::new(), &key).unwrap().unwrap();
            let element = bincode::deserialize(&element).unwrap();
            fetched_elements[index as usize] = Some(element);
        }

        fetched_elements
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec()
    }

    fn set(&mut self, index: Index, value: T) {
        // Disallow setting values out-of-bounds
        assert!(
            index < self.len(),
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self.length,
            self.name
        );

        if let Some(_old_val) = self.cache.insert(index, value.clone()) {
            // If cache entry exists, we remove any corresponding
            // OverWrite ops in the `write_queue` to reduce disk IO.

            // logic: retain all ops that are not overwrite, and
            // overwrite ops that do not have an index matching cache_index.
            self.write_queue.retain(|op| match op {
                WriteElement::OverWrite((i, _)) => *i != index,
                _ => true,
            })
        }

        self.write_queue
            .push_back(WriteElement::OverWrite((index, value)));
    }

    /// set multiple elements.
    ///
    /// panics if key_vals contains an index not in the collection
    ///
    /// It is the caller's responsibility to ensure that index values are
    /// unique.  If not, the last value with the same index will win.
    /// For unordered collections such as HashMap, the behavior is undefined.
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        for (index, value) in key_vals.into_iter() {
            self.set(index, value);
        }
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
                .get_u8(&ReadOptions::new(), &db_key)
                .unwrap()
                .map(|bytes| bincode::deserialize(&bytes).unwrap())
        }
    }

    fn push(&mut self, value: T) {
        // add to write queue
        self.write_queue
            .push_back(WriteElement::Push(value.clone()));

        // record in cache
        let _old_value = self.cache.insert(self.length, value);

        // note: we cannot naively remove any previous `Push` ops with
        // this value from the write_queue (to reduce disk i/o) because
        // there might be corresponding `Pop` op(s).

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
    fn persisted_length(&self) -> Index {
        let key = Self::get_length_key(self.key_prefix);
        match self.db.get_u8(&ReadOptions::new(), &key).unwrap() {
            Some(value) => bincode::deserialize(&value).unwrap(),
            None => 0,
        }
    }

    /// Return the level-DB key used to store the element at an index
    fn get_index_key(&self, index: Index) -> [u8; 9] {
        [vec![self.key_prefix], bincode::serialize(&index).unwrap()]
            .concat()
            .try_into()
            .unwrap()
    }

    pub fn new(db: Arc<DB>, key_prefix: u8, name: &str) -> Self {
        let length_key = Self::get_length_key(key_prefix);
        let length = match db.get_u8(&ReadOptions::new(), &length_key).unwrap() {
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
                    write_batch.put_u8(&key, &value);
                }
                WriteElement::Push(t) => {
                    let key =
                        [vec![self.key_prefix], bincode::serialize(&length).unwrap()].concat();
                    length += 1;
                    let value = bincode::serialize(&t).unwrap();
                    write_batch.put(&key, &value);
                }
                WriteElement::Pop => {
                    let key = [
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
            write_batch.put_u8(&key, &bincode::serialize(&self.length).unwrap());
        }

        self.cache.clear();
    }
}

pub struct OrdinaryVec<T>(Vec<T>);

impl<T: Clone> StorageVec<T> for OrdinaryVec<T> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn len(&self) -> Index {
        self.0.len() as Index
    }

    fn get(&self, index: Index) -> T {
        self.0[index as usize].clone()
    }

    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        indices
            .iter()
            .map(|index| self.0[*index as usize].clone())
            .collect()
    }

    fn get_all(&self) -> Vec<T> {
        self.0.clone()
    }

    fn set(&mut self, index: Index, value: T) {
        // note: on 32 bit systems, this could panic.
        self.0[index as usize] = value;
    }

    /// set multiple elements.
    ///
    /// panics if key_vals contains an index not in the collection
    ///
    /// It is the caller's responsibility to ensure that index values are
    /// unique.  If not, the last value with the same index will win.
    /// For unordered collections such as HashMap, the behavior is undefined.
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        for (index, value) in key_vals.into_iter() {
            // note: on 32 bit systems, this could panic.
            self.0[index as usize] = value;
        }
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
    use std::sync::Arc;

    use super::*;
    use rand::{Rng, RngCore};

    use leveldb::{
        batch::{Batch, WriteBatch},
        options::WriteOptions,
    };

    // todo: delete fn
    fn get_test_db(destroy_db_on_drop: bool) -> Arc<DB> {
        Arc::new(DB::open_new_test_database(destroy_db_on_drop, None).unwrap())
    }

    // todo: delete fn
    fn open_test_db(path: &std::path::Path, destroy_db_on_drop: bool) -> Arc<DB> {
        Arc::new(DB::open_test_database(path, destroy_db_on_drop, None).unwrap())
    }

    /// Return a persisted vector and a regular in-memory vector with the same elements
    fn get_persisted_vec_with_length(
        length: Index,
        name: &str,
    ) -> (RustyLevelDbVec<u64>, Vec<u64>, Arc<DB>) {
        let db = get_test_db(true);
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
        assert!(db.write(&WriteOptions::new(), &write_batch).is_ok());

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
        let db = get_test_db(true);
        let delegated_db_vec: RustyLevelDbVec<[u8; 13]> =
            RustyLevelDbVec::new(db, 0, "unit test vec 0");
        simple_prop(delegated_db_vec);

        let ordinary_vec = OrdinaryVec::<[u8; 13]>(vec![]);
        simple_prop(ordinary_vec);
    }

    #[test]
    fn multiple_vectors_in_one_db() {
        let db = get_test_db(true);
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
            db.write(&WriteOptions::new(), &write_batch).is_ok(),
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
    fn rusty_level_db_set_many() {
        let db = get_test_db(true);
        let mut delegated_db_vec_a: RustyLevelDbVec<u128> =
            RustyLevelDbVec::new(db.clone(), 0, "unit test vec a");

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

        // Persist
        let mut write_batch = WriteBatch::new();
        delegated_db_vec_a.pull_queue(&mut write_batch);
        assert!(
            db.write(&WriteOptions::new(), &write_batch).is_ok(),
            "DB write must succeed"
        );
        assert_eq!(4, delegated_db_vec_a.persisted_length());

        // Check values after persisting
        assert_eq!(
            vec![1000, 2000, 3000],
            delegated_db_vec_a.get_many(&[0, 1, 2])
        );
        assert_eq!(
            vec![1000, 2000, 3000, 400],
            delegated_db_vec_a.get_many(&[0, 1, 2, 3])
        );
    }

    #[test]
    fn rusty_level_db_set_all() {
        let db = get_test_db(true);
        let mut delegated_db_vec_a: RustyLevelDbVec<u128> =
            RustyLevelDbVec::new(db.clone(), 0, "unit test vec a");

        delegated_db_vec_a.push(10);
        delegated_db_vec_a.push(20);
        delegated_db_vec_a.push(30);

        let updates = [100, 200, 300];
        delegated_db_vec_a.set_all(&updates);

        assert_eq!(vec![100, 200, 300], delegated_db_vec_a.get_many(&[0, 1, 2]));

        #[allow(clippy::shadow_unrelated)]
        let updates = vec![1000, 2000, 3000];
        delegated_db_vec_a.set_all(&updates);

        assert_eq!(
            vec![1000, 2000, 3000],
            delegated_db_vec_a.get_many(&[0, 1, 2])
        );

        // Persist
        let mut write_batch = WriteBatch::new();
        delegated_db_vec_a.pull_queue(&mut write_batch);
        assert!(
            db.write(&WriteOptions::new(), &write_batch).is_ok(),
            "DB write must succeed"
        );
        assert_eq!(3, delegated_db_vec_a.persisted_length());

        // Check values after persisting
        assert_eq!(
            vec![1000, 2000, 3000],
            delegated_db_vec_a.get_many(&[0, 1, 2])
        );
    }

    #[test]
    fn db_close_and_reload() {
        let db = get_test_db(false);
        let db_path = db.path.clone();

        let mut vec: RustyLevelDbVec<u128> = RustyLevelDbVec::new(db, 0, "vec1");
        vec.push(1000);

        assert_eq!(1, vec.len());

        let mut write_batch = WriteBatch::new();
        vec.pull_queue(&mut write_batch);
        assert!(vec.db.write(&WriteOptions::new(), &write_batch).is_ok());

        drop(vec); // this will drop (close) the Db

        let db2 = open_test_db(&db_path, true);
        let mut vec2: RustyLevelDbVec<u128> = RustyLevelDbVec::new(db2, 0, "vec1");

        assert_eq!(1, vec2.len());
        assert_eq!(1000, vec2.pop().unwrap());
    }

    #[test]
    fn get_many_ordering_of_outputs() {
        let db = get_test_db(true);
        let mut delegated_db_vec_a: RustyLevelDbVec<u128> =
            RustyLevelDbVec::new(db.clone(), 0, "unit test vec a");

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

        // Persist
        let mut write_batch = WriteBatch::new();
        delegated_db_vec_a.pull_queue(&mut write_batch);
        assert!(
            db.write(&WriteOptions::new(), &write_batch).is_ok(),
            "DB write must succeed"
        );
        assert_eq!(3, delegated_db_vec_a.persisted_length());

        // Check ordering after persisting
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
        let (mut persisted_vector, mut normal_vector, db) =
            get_persisted_vec_with_length(10000, "vec 1");

        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            match rng.gen_range(0..=5) {
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
                5 => {
                    // persist
                    let mut write_batch = WriteBatch::new();
                    persisted_vector.pull_queue(&mut write_batch);
                    db.write(&WriteOptions::new(), &write_batch).unwrap();
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

        // Check equality after persisting updates
        let mut write_batch = WriteBatch::new();
        persisted_vector.pull_queue(&mut write_batch);
        db.write(&WriteOptions::new(), &write_batch).unwrap();

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

        // Check equality using `get_many`
        assert_eq!(
            normal_vector,
            persisted_vector.get_many(&(0..normal_vector.len() as u64).collect_vec())
        );
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
        expected = "Out-of-bounds. Got indices [3] but length was 1. persisted vector name: unit test vec 0"
    )]
    #[test]
    fn panic_on_out_of_bounds_get_many() {
        let (delegated_db_vec, _, _) = get_persisted_vec_with_length(1, "unit test vec 0");
        delegated_db_vec.get_many(&[3]);
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
        expected = "Out-of-bounds. Got 1 but length was 1. persisted vector name: unit test vec 0"
    )]
    #[test]
    fn panic_on_out_of_bounds_set_many() {
        let (mut delegated_db_vec, _, _) = get_persisted_vec_with_length(1, "unit test vec 0");

        // attempt to set 2 values, when only one is in vector.
        delegated_db_vec.set_many([(0, 0), (1, 1)]);
    }

    #[should_panic(expected = "size-mismatch.  input has 2 elements and target has 1 elements.")]
    #[test]
    fn panic_on_size_mismatch_set_all() {
        let (mut delegated_db_vec, _, _) = get_persisted_vec_with_length(1, "unit test vec 0");

        // attempt to set 2 values, when only one is in vector.
        delegated_db_vec.set_all(&[1, 2]);
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
