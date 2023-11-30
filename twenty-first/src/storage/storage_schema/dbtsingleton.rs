use std::{
    fmt::Debug,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use super::{dbtsingleton_private::DbtSingletonPrivate, traits::*, WriteOperation};

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

impl<K, V, T> StorageSingletonReads<T> for DbtSingleton<K, V, T>
where
    T: Clone + From<V> + Default,
{
    #[inline]
    fn get(&self) -> T {
        self.read_lock().current_value.clone()
    }
}

impl<K, V, T> StorageSingletonImmutableWrites<T> for DbtSingleton<K, V, T>
where
    T: Clone + From<V> + Default,
{
    #[inline]
    fn set(&self, t: T) {
        self.write_lock().current_value = t;
    }
}

impl<K, V, T> StorageSingleton<T> for DbtSingleton<K, V, T> where T: Clone + From<V> + Default {}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtSingleton<ParentKey, ParentValue, T>
where
    T: Eq + Clone + Default + From<ParentValue>,
    ParentValue: From<T> + Debug,
    ParentKey: Clone,
{
    #[inline]
    fn pull_queue(&self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        let mut lock = self.write_lock();

        if lock.current_value == lock.old_value {
            vec![]
        } else {
            lock.old_value = lock.current_value.clone();
            vec![WriteOperation::Write(
                lock.key.clone(),
                lock.current_value.clone().into(),
            )]
        }
    }

    #[inline]
    fn restore_or_new(&self) {
        let mut lock = self.write_lock();
        lock.current_value = match lock.reader.get(lock.key.clone()) {
            Some(value) => value.into(),
            None => T::default(),
        }
    }
}
