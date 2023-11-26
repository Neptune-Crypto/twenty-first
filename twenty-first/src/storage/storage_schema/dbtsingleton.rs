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
pub struct DbtSingleton<ParentKey, ParentValue, T> {
    // note: Arc is not needed, because we never hand out inner to anyone.
    inner: RwLock<DbtSingletonPrivate<ParentKey, ParentValue, T>>,
}

impl<ParentKey, ParentValue, T> DbtSingleton<ParentKey, ParentValue, T>
where
    T: Default,
{
    // DbtSingleton can not be instantiated directly outside of this crate.
    #[inline]
    pub(crate) fn new(
        key: ParentKey,
        reader: Arc<dyn StorageReader<ParentKey, ParentValue> + Sync + Send>,
    ) -> Self {
        let singleton = DbtSingletonPrivate::<ParentKey, ParentValue, T> {
            current_value: Default::default(),
            old_value: Default::default(),
            key,
            reader,
        };
        Self {
            inner: RwLock::new(singleton),
        }
    }

    #[inline]
    pub(crate) fn read_lock(
        &self,
    ) -> RwLockReadGuard<'_, DbtSingletonPrivate<ParentKey, ParentValue, T>> {
        self.inner.read().unwrap()
    }

    #[inline]
    pub(crate) fn write_lock(
        &self,
    ) -> RwLockWriteGuard<'_, DbtSingletonPrivate<ParentKey, ParentValue, T>> {
        self.inner.write().unwrap()
    }
}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtSingleton<ParentKey, ParentValue, T>
where
    T: Eq + Clone + Default + From<ParentValue>,
    ParentValue: From<T> + Debug,
    ParentKey: Clone,
{
    #[inline]
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        self.write_lock().pull_queue()
    }

    #[inline]
    fn restore_or_new(&mut self) {
        self.write_lock().restore_or_new()
    }
}

impl<ParentKey, ParentValue, T> StorageSingleton<T> for DbtSingleton<ParentKey, ParentValue, T>
where
    T: Clone + From<ParentValue> + Default,
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
