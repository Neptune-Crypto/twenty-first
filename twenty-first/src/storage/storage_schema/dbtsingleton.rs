use std::{
    fmt::Debug,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use super::{
    dbtsingleton_private::DbtSingletonPrivate, traits::*, RustyKey, RustyValue, WriteOperation,
};

/// Singleton type created by [`super::DbtSchema`]
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
pub struct DbtSingleton<V> {
    // note: Arc is not needed, because we never hand out inner to anyone.
    inner: Arc<RwLock<DbtSingletonPrivate<V>>>,
}

// We manually impl Clone so that callers can make reference clones.
impl<V> Clone for DbtSingleton<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<V> DbtSingleton<V>
where
    V: Default,
{
    // DbtSingleton can not be instantiated directly outside of this crate.
    #[inline]
    pub(crate) fn new(key: RustyKey, reader: Arc<dyn StorageReader + Sync + Send>) -> Self {
        let singleton = DbtSingletonPrivate::<V> {
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
    pub(super) fn read_lock(&self) -> RwLockReadGuard<'_, DbtSingletonPrivate<V>> {
        self.inner
            .read()
            .expect("should have acquired read lock for DbtSingletonPrivate")
    }

    // This is a private method, but we allow unit tests in super to use it.
    #[inline]
    pub(super) fn write_lock(&self) -> RwLockWriteGuard<'_, DbtSingletonPrivate<V>> {
        self.inner
            .write()
            .expect("should have acquired write lock for DbtSingletonPrivate")
    }
}

impl<V> StorageSingletonReads<V> for DbtSingleton<V>
where
    V: Clone + From<V> + Default,
{
    #[inline]
    fn get(&self) -> V {
        self.read_lock().current_value.clone()
    }
}

impl<V> StorageSingletonImmutableWrites<V> for DbtSingleton<V>
where
    V: Clone + From<V> + Default,
    V: From<RustyValue>,
{
    #[inline]
    fn set(&self, t: V) {
        self.write_lock().current_value = t;
    }
}

impl<V> StorageSingleton<V> for DbtSingleton<V> where V: Clone + From<V> + Default + From<RustyValue>
{}

impl<V> DbTable for DbtSingleton<V>
where
    V: Eq + Clone + Default + Debug,
    V: From<RustyValue>,
    RustyValue: From<V>,
{
    #[inline]
    fn pull_queue(&self) -> Vec<WriteOperation> {
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
            None => V::default(),
        }
    }
}
