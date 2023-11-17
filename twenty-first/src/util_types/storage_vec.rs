use lending_iterator::prelude::*;
use lending_iterator::{gat, LendingIterator};
use std::iter::Iterator;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use rusty_leveldb::{WriteBatch, DB};
use serde::{de::DeserializeOwned, Serialize};

pub type Index = u64;

pub trait StorageVec<T> {
    /// check if collection is empty
    fn is_empty(&self) -> bool;

    /// get collection length
    fn len(&self) -> Index;

    /// get single element at index
    fn get(&self, index: Index) -> T;

    /// retrieve a `StorageVecSetter` at index.
    ///
    /// The returned value can be used to read or mutate the value.
    #[inline]
    fn get_mut(&mut self, index: Index) -> Option<StorageVecSetter<Self, T>> {
        let value = self.get(index);
        Some(StorageVecSetter {
            vec: self,
            index,
            value,
        })
    }

    /// get elements matching `indices` as a Vec
    ///
    /// This is a convenience method. For large collections
    /// it will be more efficient to use `iter` and avoid
    /// allocating a Vec
    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        self.many_iter(indices.to_vec()).map(|(_i, v)| v).collect()
    }

    /// returns iterator over multiple elements matching `indices`
    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_>;

    /// returns a mutable iterator over elements matching `indices``
    #[inline]
    fn many_iter_mut(
        &mut self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> ManyIterMut<Self, T>
    where
        Self: Sized,
    {
        ManyIterMut::new(indices, self)
    }

    /// returns a mutable iterator over all elements
    #[inline]
    fn iter_mut(&mut self) -> ManyIterMut<Self, T>
    where
        Self: Sized,
    {
        ManyIterMut::new(0..self.len(), self)
    }

    /// get all elements
    ///
    /// This is a convenience method. For large collections
    /// it will be more efficient to use `get_all_iter` directly
    /// and avoid allocating a Vec
    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.iter().map(|(_i, v)| v).collect()
    }

    /// get all elements and return as an iterator
    #[inline]
    fn iter(&self) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        self.many_iter(0..self.len())
    }

    /// set a single element.
    fn set(&mut self, index: Index, value: T);

    /// set multiple elements.
    //
    //  note: set_many is mostly redundant now that many_iter_mut() exists.
    //        We keep it for now because implementors can lock once rather
    //        than for each element.  eg: DbtVec::set_many
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>);

    /// set elements from start to vals.count()
    ///
    /// calls ::set_many() internally.
    ///
    /// note: casts the array's indexes from usize to Index
    ///       so
    #[inline]
    fn set_first_n(&mut self, vals: impl IntoIterator<Item = T>) {
        self.set_many((0..).zip(vals));
    }

    /// set all elements with a list of values and validates that input length matches target length.
    ///
    /// calls ::set_many() internally.
    ///
    /// panics if input length does not match target length.
    ///
    /// note: casts the input value's length from usize to Index
    ///       so will panic if vals contains more than 2^32 items
    #[inline]
    fn set_all(&mut self, vals: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = T>>) {
        let iter = vals.into_iter();

        assert!(
            iter.len() as Index == self.len(),
            "size-mismatch.  input has {} elements and target has {} elements.",
            iter.len(),
            self.len(),
        );

        self.set_first_n(iter);
    }

    /// pop an element from end of collection
    fn pop(&mut self) -> Option<T>;

    /// push an element to end of collection
    fn push(&mut self, value: T);
}

/// a mutating iterator for StorageVec trait
pub struct ManyIterMut<'a, V, T>
where
    V: StorageVec<T> + ?Sized,
{
    indices: Box<dyn Iterator<Item = Index>>,
    data: &'a mut V,
    phantom: std::marker::PhantomData<T>,
}

impl<'a, V, T> ManyIterMut<'a, V, T>
where
    V: StorageVec<T>,
{
    fn new<I>(indices: I, data: &'a mut V) -> Self
    where
        I: IntoIterator<Item = Index> + 'static,
    {
        Self {
            indices: Box::new(indices.into_iter()),
            data,
            phantom: Default::default(),
        }
    }
}

// LendingIterator trait gives us all the nice iterator type functions.
// We only have to impl next()
#[gat]
impl<'a, V, T: 'a> LendingIterator for ManyIterMut<'a, V, T>
where
    V: StorageVec<T>,
{
    type Item<'b> = StorageVecSetter<'b, V, T>
    where
        Self: 'b;

    fn next(&mut self) -> Option<Self::Item<'_>> {
        if let Some(i) = Iterator::next(&mut self.indices) {
            self.data.get_mut(i)
        } else {
            None
        }
    }
}

/// used for accessing and setting values returned from StorageVec::get_mut() and mutable iterators
pub struct StorageVecSetter<'a, V, T>
where
    V: StorageVec<T> + ?Sized,
{
    vec: &'a mut V,
    index: Index,
    value: T,
}

impl<'a, V, T> StorageVecSetter<'a, V, T>
where
    V: StorageVec<T> + ?Sized,
{
    pub fn set(&mut self, value: T) {
        self.vec.set(self.index, value);
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

pub enum WriteElement<T: Serialize + DeserializeOwned> {
    OverWrite((Index, T)),
    Push(T),
    Pop,
}

pub struct RustyLevelDbVec<T: Serialize + DeserializeOwned> {
    key_prefix: u8,
    db: Arc<Mutex<DB>>,
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
        let db_val = self.db.lock().unwrap().get(&db_key).unwrap_or_else(|| {
            panic!(
                "Element with index {index} does not exist in {}. This should not happen",
                self.name
            )
        });
        bincode::deserialize(&db_val).unwrap()
    }

    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        // note: this lock is moved into the iterator closure and is not
        //       released until the caller drops the returned iterator
        let mut db_reader = self.db.lock().expect("get_many: db-locking must succeed");

        Box::new(indices.into_iter().map(move |i| {
            assert!(
                i < self.length,
                "Out-of-bounds. Got index {} but length was {}. persisted vector name: {}",
                i,
                self.length,
                self.name
            );

            if self.cache.contains_key(&i) {
                (i, self.cache[&i].clone())
            } else {
                let key = self.get_index_key(i);
                let db_element = db_reader.get(&key).unwrap();
                (i, bincode::deserialize(&db_element).unwrap())
            }
        }))
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
        match self.db.lock().unwrap().get(&key) {
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
            write_batch.put(&key, &bincode::serialize(&self.length).unwrap());
        }

        self.cache.clear();
    }
}

pub struct OrdinaryVec<T>(Vec<T>);

// Some niceties for OrdinaryVec

impl<T> IntoIterator for OrdinaryVec<T> {
    type Item = T;
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// We deref to slice so that we can reuse the slice impls
impl<T> std::ops::Deref for OrdinaryVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0[..]
    }
}

impl<T> std::ops::DerefMut for OrdinaryVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0[..]
    }
}

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

    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        Box::new(
            indices
                .into_iter()
                .map(|index| (index, self.0[index as usize].clone())),
        )
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
    use super::*;
    use itertools::Itertools;
    use rand::{Rng, RngCore};
    use rusty_leveldb::DB;

    fn get_test_db() -> Arc<Mutex<DB>> {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        Arc::new(Mutex::new(db))
    }

    /// Return a persisted vector and a regular in-memory vector with the same elements
    fn get_persisted_vec_with_length(
        length: Index,
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
    fn rusty_level_db_set_many() {
        let db = get_test_db();
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
            db.lock().unwrap().write(write_batch, true).is_ok(),
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
        let db = get_test_db();
        let mut delegated_db_vec_a: RustyLevelDbVec<u128> =
            RustyLevelDbVec::new(db.clone(), 0, "unit test vec a");

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

        // Persist
        let mut write_batch = WriteBatch::new();
        delegated_db_vec_a.pull_queue(&mut write_batch);
        assert!(
            db.lock().unwrap().write(write_batch, true).is_ok(),
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
    fn get_many_ordering_of_outputs() {
        let db = get_test_db();
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
            db.lock().unwrap().write(write_batch, true).is_ok(),
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
                    db.lock().unwrap().write(write_batch, true).unwrap();
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

        // Check equality using `get_many`
        assert_eq!(
            normal_vector,
            persisted_vector.get_many(&(0..normal_vector.len() as u64).collect_vec())
        );
    }

    #[test]
    // This tests that we can obtain a subset of elements with
    // many_iter() and feed the results directly into set_many()
    // and the collection should be unchanged, meaning that the
    // index values were preserved.
    fn many_iter_indexes() {
        let db = get_test_db();
        let mut v: RustyLevelDbVec<u128> = RustyLevelDbVec::new(db.clone(), 0, "unit test vec a");

        let pattern = vec![100, 200, 300, 400];
        v.push(pattern[0]);
        v.push(pattern[1]);
        v.push(pattern[2]);
        v.push(pattern[3]);

        let set: Vec<_> = v.many_iter([1, 3]).collect();

        // set indexes 1 and 3 to 0, to prove set_many is effecting change.
        let pattern_wipe = vec![100, 0, 300, 0];
        v.set_all(pattern_wipe.clone());
        assert_eq!(pattern_wipe, v.get_all());

        // now send the fetched values to set_many and we
        // should have restored the original pattern.
        v.set_many(set);
        assert_eq!(pattern, v.get_all());
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
        expected = "Out-of-bounds. Got index 3 but length was 1. persisted vector name: unit test vec 0"
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
        delegated_db_vec.set_all([1, 2]);
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
