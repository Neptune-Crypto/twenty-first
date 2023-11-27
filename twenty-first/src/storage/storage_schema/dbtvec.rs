use super::super::storage_vec::{Index, StorageVec};
use super::dbtvec_private::DbtVecPrivate;
use super::{DbTable, StorageReader, WriteOperation};
use std::{
    fmt::Debug,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// A DB-backed Vec for use with DBSchema
///
/// This type is concurrency-safe.  A single RwLock is employed
/// for all read and write ops.  Callers do not need to perform
/// any additional locking.
///
/// Also because the locking is fully encapsulated within DbtVec
/// there is no possibility of a caller holding a lock too long
/// by accident or encountering ordering deadlock issues.
///
/// `DbtSingleton` is a NewType around Arc<RwLock<..>>.  Thus it
/// can be cheaply cloned to create a reference as if it were an
/// Arc.
pub struct DbtVec<K, V, Index, T> {
    inner: Arc<RwLock<DbtVecPrivate<K, V, Index, T>>>,
}

impl<K, V, T> Clone for DbtVec<K, V, Index, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<K, V, T> DbtVec<K, V, Index, T>
where
    K: From<(K, K)>,
    K: From<u8>,
    K: From<Index>,
    Index: From<V> + From<u64> + Clone,
    T: Clone,
{
    // DbtVec cannot be instantiated directly outside of this crate.
    #[inline]
    pub(crate) fn new(
        reader: Arc<dyn StorageReader<K, V> + Send + Sync>,
        key_prefix: u8,
        name: &str,
    ) -> Self {
        let vec = DbtVecPrivate::<K, V, Index, T>::new(reader, key_prefix, name);

        Self {
            inner: Arc::new(RwLock::new(vec)),
        }
    }

    // This is a private method, but we allow unit tests in super to use it.
    #[inline]
    pub(super) fn read_lock(&self) -> RwLockReadGuard<'_, DbtVecPrivate<K, V, Index, T>> {
        self.inner.read().unwrap()
    }

    // This is a private method, but we allow unit tests in super to use it.
    #[inline]
    pub(super) fn write_lock(&mut self) -> RwLockWriteGuard<'_, DbtVecPrivate<K, V, Index, T>> {
        self.inner.write().unwrap()
    }
}

impl<K, V, T> StorageVec<T> for DbtVec<K, V, Index, T>
where
    K: From<Index>,
    V: From<T>,
    T: Clone + From<V> + Debug,
    K: From<(K, K)>,
    K: From<u8>,
    Index: From<V> + From<u64>,
{
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

    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        self.read_lock().get_many(indices)
    }

    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.read_lock().get_all()
    }

    #[inline]
    fn set(&mut self, index: Index, value: T) {
        self.write_lock().set(index, value)
    }

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

impl<K, V, T> DbTable<K, V> for DbtVec<K, V, Index, T>
where
    K: From<Index>,
    V: From<T>,
    T: Clone,
    T: From<V>,
    K: From<(K, K)>,
    K: From<u8>,
    Index: From<V>,
    V: From<Index>,
{
    /// Collect all added elements that have not yet been persisted
    #[inline]
    fn pull_queue(&mut self) -> Vec<WriteOperation<K, V>> {
        self.write_lock().pull_queue()
    }

    #[inline]
    fn restore_or_new(&mut self) {
        self.write_lock().restore_or_new()
    }
}
