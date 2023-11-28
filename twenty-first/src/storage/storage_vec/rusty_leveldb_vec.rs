use super::super::level_db::DB;
use super::rusty_leveldb_vec_private::RustyLevelDbVecPrivate;
use super::storage_vec_trait::{Index, StorageVec};
use leveldb::batch::WriteBatch;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// RustyLevelDbVec is a concurrency safe database-backed
/// Vec with in memory read/write caching for all operations.
#[derive(Debug, Clone)]
pub struct RustyLevelDbVec<T: Serialize + DeserializeOwned> {
    inner: Arc<RwLock<RustyLevelDbVecPrivate<T>>>,
}

impl<T: Serialize + DeserializeOwned + Clone> StorageVec<T> for RustyLevelDbVec<T> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.read_lock().is_empty()
    }

    #[inline]
    fn len(&self) -> Index {
        self.read_lock().len()
    }

    #[inline]
    fn get(&self, index: Index) -> T {
        self.read_lock().get(index)
    }

    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        // note: this lock is moved into the iterator closure and is not
        //       released until caller drops the returned iterator
        let inner = self.read_lock();

        Box::new(indices.into_iter().map(move |i| {
            assert!(
                i < inner.len(),
                "Out-of-bounds. Got index {} but length was {}. persisted vector name: {}",
                i,
                inner.len(),
                inner.name
            );

            if inner.cache.contains_key(&i) {
                (i, inner.cache[&i].clone())
            } else {
                let key = inner.get_index_key(i);
                (i, inner.get_u8(&key))
            }
        }))
    }

    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        self.read_lock().get_many(indices)
    }

    /// Return all stored elements in a vector, whose index matches the StorageVec's.
    /// It's the caller's responsibility that there is enough memory to store all elements.
    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.read_lock().get_all()
    }

    #[inline]
    fn set(&mut self, index: Index, value: T) {
        self.write_lock().set(index, value)
    }

    /// set multiple elements.
    ///
    /// panics if key_vals contains an index not in the collection
    ///
    /// It is the caller's responsibility to ensure that index values are
    /// unique.  If not, the last value with the same index will win.
    /// For unordered collections such as HashMap, the behavior is undefined.
    #[inline]
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        self.write_lock().set_many(key_vals)
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.write_lock().pop()
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.write_lock().push(value)
    }
}

impl<T: Serialize + DeserializeOwned> RustyLevelDbVec<T> {
    // Return the key used to store the length of the persisted vector
    #[inline]
    pub fn get_length_key(key_prefix: u8) -> [u8; 2] {
        RustyLevelDbVecPrivate::<T>::get_length_key(key_prefix)
    }

    /// Return the length at the last write to disk
    #[inline]
    pub fn persisted_length(&self) -> Index {
        self.read_lock().persisted_length()
    }

    /// Return the level-DB key used to store the element at an index
    #[inline]
    pub fn get_index_key(&self, index: Index) -> [u8; 9] {
        self.read_lock().get_index_key(index)
    }

    #[inline]
    pub fn new(db: Arc<DB>, key_prefix: u8, name: &str) -> Self {
        Self {
            inner: Arc::new(RwLock::new(RustyLevelDbVecPrivate::<T>::new(
                db, key_prefix, name,
            ))),
        }
    }

    /// Collect all added elements that have not yet bit persisted
    #[inline]
    pub fn pull_queue(&mut self, write_batch: &mut WriteBatch) {
        self.write_lock().pull_queue(write_batch)
    }

    #[inline]
    pub(super) fn read_lock(&self) -> RwLockReadGuard<'_, RustyLevelDbVecPrivate<T>> {
        self.inner
            .read()
            .expect("should have acquired RustyLevelDbVec read lock")
    }

    #[inline]
    pub(super) fn write_lock(&mut self) -> RwLockWriteGuard<'_, RustyLevelDbVecPrivate<T>> {
        self.inner
            .write()
            .expect("should have acquired RustyLevelDbVec write lock")
    }
}
