use super::super::level_db::DB;
use super::storage_vec_trait::{Index, StorageVec, WriteElement};
use itertools::Itertools;
use leveldb::{batch::WriteBatch, options::ReadOptions};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// This is the private impl of RustyLevelDbVec.
///
/// RustyLevelDbVec is a public wrapper that adds RwLock around
/// all accesses to RustyLevelDbVecPrivate
pub(crate) struct RustyLevelDbVecPrivate<T: Serialize + DeserializeOwned> {
    key_prefix: u8,
    pub(super) db: Arc<DB>,
    write_queue: VecDeque<WriteElement<T>>,
    length: Index,
    pub(super) cache: HashMap<Index, T>,
    name: String,
}

impl<T: Serialize + DeserializeOwned + Clone> StorageVec<T> for RustyLevelDbVecPrivate<T> {
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

impl<T: Serialize + DeserializeOwned> RustyLevelDbVecPrivate<T> {
    // Return the key used to store the length of the persisted vector
    pub(crate) fn get_length_key(key_prefix: u8) -> [u8; 2] {
        const LENGTH_KEY: u8 = 0u8;
        [key_prefix, LENGTH_KEY]
    }

    /// Return the length at the last write to disk
    pub(crate) fn persisted_length(&self) -> Index {
        let key = Self::get_length_key(self.key_prefix);
        match self.db.get_u8(&ReadOptions::new(), &key).unwrap() {
            Some(value) => bincode::deserialize(&value).unwrap(),
            None => 0,
        }
    }

    /// Return the level-DB key used to store the element at an index
    pub(crate) fn get_index_key(&self, index: Index) -> [u8; 9] {
        [vec![self.key_prefix], bincode::serialize(&index).unwrap()]
            .concat()
            .try_into()
            .unwrap()
    }

    pub(crate) fn new(db: Arc<DB>, key_prefix: u8, name: &str) -> Self {
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
    pub(crate) fn pull_queue(&mut self, write_batch: &mut WriteBatch) {
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
