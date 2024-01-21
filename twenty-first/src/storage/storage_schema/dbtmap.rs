use serde::{de::DeserializeOwned, Serialize};

use super::dbtmap_private::DbtMapPrivate;
use super::{traits::*, WriteOperation};
use crate::sync::{AtomicRw, LockCallbackFn};
use itertools::Itertools;
use std::hash::Hash;
use std::{fmt::Debug, sync::Arc};

/// A DB-backed Map for use with DBSchema
///
/// This type is concurrency-safe.  A single RwLock is employed
/// for all read and write ops.  Callers do not need to perform
/// any additional locking.
///
/// `DbtMap` is a NewType around Arc<RwLock<..>>.  Thus it
/// can be cheaply cloned to create a reference as if it were an
/// Arc.
///
/// Important!  This type should only be used for storing
/// a small number of values, eg under 1000, although there
/// is no fixed limit.
///
/// DbtMap is essentially creating a logical sub-set of key/val
/// pairs inside the full set, which is the LevelDB database.
///
/// In order to persist its list of known keys, DbtMap stores
/// that list in a single LevelDB key, which must be updated for
/// every insert or delete.
///
/// Additionally, DbtMap keeps a copy of the key-list in RAM,
/// which is loaded from the DB during initialization.
///
/// For this reason, this type is not suited to storing
/// huge numbers of entries.
///
#[derive(Debug)]
pub struct DbtMap<K, V> {
    inner: AtomicRw<DbtMapPrivate<K, V>>,
}

impl<K, V> Clone for DbtMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<K, V> DbtMap<K, V>
where
    K: DeserializeOwned + Serialize + Eq + Hash + Clone,
    V: DeserializeOwned + Serialize + Clone,
{
    // DbtMap cannot be instantiated directly outside of this crate.
    #[inline]
    pub(crate) fn new(
        reader: Arc<dyn StorageReader + Send + Sync>,
        key_prefix: u8,
        name: &str,
        lock_name: String,
        lock_callback_fn: Option<LockCallbackFn>,
    ) -> Self {
        let vec = DbtMapPrivate::<K, V>::new(reader, key_prefix, name);

        Self {
            inner: AtomicRw::from((vec, Some(lock_name), lock_callback_fn)),
        }
    }

    /// check if map is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.lock(|inner| inner.is_empty())
    }

    /// get length of map
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.lock(|inner| inner.len())
    }

    /// get value identified by key
    #[inline]
    pub fn get(&self, k: &K) -> Option<V> {
        self.inner
            .lock(|inner| inner.get(k).map(|v| v.into_owned()))
    }

    /// insert value identified by key
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> bool {
        self.inner.lock_mut(|inner| inner.insert(k, v))
    }

    /// remove entry identified by key
    #[inline]
    pub fn remove(&mut self, k: &K) -> bool {
        self.inner.lock_mut(|inner| inner.remove(k))
    }

    /// clear all entries
    #[inline]
    pub fn clear(&mut self) {
        self.inner.lock_mut(|inner| inner.clear());
    }

    /// get an iterator over all keys and values
    ///
    /// The returned iterator holds a read-lock over the collection contents.
    /// This enables consistent (snapshot) reads because any writer must
    /// wait until the lock is released.
    ///
    /// The lock is not released until the iterator is dropped, so it is
    /// important to drop the iterator immediately after use.  Typical
    /// for-loop usage does this automatically.
    ///
    /// note: this call performs an internal collect() over all values.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (K, V)> {
        let inner = self.inner.lock_guard();
        inner
            .iter()
            .map(|(k, v)| (k.clone(), v.into_owned()))
            .collect_vec()
            .into_iter()
    }

    /// get an iterator over all keys
    ///
    /// The returned iterator holds a read-lock over the collection contents.
    /// This enables consistent (snapshot) reads because any writer must
    /// wait until the lock is released.
    ///
    /// The lock is not released until the iterator is dropped, so it is
    /// important to drop the iterator immediately after use.  Typical
    /// for-loop usage does this automatically.
    ///
    /// note: this call performs an internal collect() over all values.
    #[inline]
    pub fn iter_keys(&self) -> impl Iterator<Item = K> {
        let inner = self.inner.lock_guard();
        inner.iter_keys().cloned().collect_vec().into_iter()
    }

    /// get an iterator over all values
    ///
    /// The returned iterator holds a read-lock over the collection contents.
    /// This enables consistent (snapshot) reads because any writer must
    /// wait until the lock is released.
    ///
    /// The lock is not released until the iterator is dropped, so it is
    /// important to drop the iterator immediately after use.  Typical
    /// for-loop usage does this automatically.
    ///
    /// note: this call performs an internal collect() over all values.
    #[inline]
    pub fn iter_values(&self) -> impl Iterator<Item = V> {
        let inner = self.inner.lock_guard();
        inner
            .iter_values()
            .map(|v| v.into_owned())
            .collect_vec()
            .into_iter()
    }
}

impl<K, V> DbTable for DbtMap<K, V>
where
    K: DeserializeOwned + Serialize + Eq + Hash + Clone + Debug,
    V: DeserializeOwned + Serialize + Clone,
{
    /// Collect all added elements that have not yet been persisted
    ///
    /// note: this clears the internal cache.  Thus the cache does
    /// not grow unbounded, so long as `pull_queue()` is called
    /// regularly.  It also means the cache must be rebuilt after
    /// each call (batch write)
    fn pull_queue(&mut self) -> Vec<WriteOperation> {
        self.inner.lock_mut(|inner| inner.pull_queue())
    }

    #[inline]
    fn restore_or_new(&mut self) {
        self.inner.lock_mut(|inner| inner.restore_or_new())
    }
}
