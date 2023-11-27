use std::{
    fmt::Debug,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use super::{
    dbtsingleton_private::DbtSingletonPrivate, DbTable, StorageReader, StorageSingleton,
    WriteOperation,
};

/// Singleton type created by [`DbSchema`]
///
/// This type is concurrency-safe.  A single RwLock is employed
/// for all read and write ops.  Callers do not need to perform
/// any additional locking.
///
/// Also because the locking is fully encapsulated within DbtSingleton
/// there is no possibility of a caller holding a lock too long
/// by accident or encountering ordering deadlock issues.
///
/// `DbtSingleton` is a NewType around Arc<RwLock<..>>.  Thus it
/// can be cheaply cloned to create a reference as if it were an
/// Arc.
pub struct DbtSingleton<K, V, T> {
    // note: Arc is not needed, because we never hand out inner to anyone.
    inner: Arc<RwLock<DbtSingletonPrivate<K, V, T>>>,
}

// We manually impl Clone so that callers can make reference clones.
impl<K, V, T> Clone for DbtSingleton<K, V, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<K, V, T> DbtSingleton<K, V, T>
where
    T: Default,
{
    // DbtSingleton can not be instantiated directly outside of this crate.
    #[inline]
    pub(crate) fn new(key: K, reader: Arc<dyn StorageReader<K, V> + Sync + Send>) -> Self {
        let singleton = DbtSingletonPrivate::<K, V, T> {
            current_value: Default::default(),
            old_value: Default::default(),
            key,
            reader,
        };
        Self {
            inner: Arc::new(RwLock::new(singleton)),
        }
    }

    // This is a private method, but we allow unit tests in super to use it.
    #[inline]
    pub(super) fn read_lock(&self) -> RwLockReadGuard<'_, DbtSingletonPrivate<K, V, T>> {
        self.inner
            .read()
            .expect("should have acquired read lock for DbtSingletonPrivate")
    }

    // This is a private method, but we allow unit tests in super to use it.
    #[inline]
    pub(super) fn write_lock(&self) -> RwLockWriteGuard<'_, DbtSingletonPrivate<K, V, T>> {
        self.inner
            .write()
            .expect("should have acquired write lock for DbtSingletonPrivate")
    }
}

impl<K, V, T> DbTable<K, V> for DbtSingleton<K, V, T>
where
    T: Eq + Clone + Default + From<V>,
    V: From<T> + Debug,
    K: Clone,
{
    #[inline]
    fn pull_queue(&mut self) -> Vec<WriteOperation<K, V>> {
        self.write_lock().pull_queue()
    }

    #[inline]
    fn restore_or_new(&mut self) {
        self.write_lock().restore_or_new()
    }
}

impl<K, V, T> StorageSingleton<T> for DbtSingleton<K, V, T>
where
    T: Clone + From<V> + Default,
{
    #[inline]
    fn get(&self) -> T {
        self.read_lock().current_value.clone()
    }

    #[inline]
    fn set(&mut self, t: T) {
        self.write_lock().current_value = t;
    }
}
