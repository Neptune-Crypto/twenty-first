use super::{traits::*, WriteOperation};
use super::{RustyKey, RustyValue};
use serde::{de::DeserializeOwned, Serialize};
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::{collections::HashMap, sync::Arc};

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
/// So in this design, for a DbtMap with prefix `0` and values
/// `{(10,100), (20,200), (30,300)}`, the levelDB representation logically
/// looks like:
/// ```text
///   0_kl: [10,20,30],
///   0_10: 100,
///   0_20: 200,
///   0_30: 300
/// ```
///
/// So the entire `0_kl` list must be updated for each insertion/removal.
///
/// An alternative design is possible where the DbtMap
/// keys are stored in an ascending list of LevelDB keys,
/// eg:
/// ```text
///   0_kl_0: 10,
///   0_kl_1: 20,
///   0_kl_2: 30,
///   0_10: 100,
///   0_20: 200,
///   0_30: 300
/// ```
/// Here only a single small `0_kl_x` key needs to be added or removed
/// for each insertion/removal.
///
/// This design could scale better to large numbers of entries
/// but would be slower for read access without all keys
/// cached in RAM.
pub(crate) struct DbtMapPrivate<K, V> {
    pub(super) name: String,
    pub(super) reader: Arc<dyn StorageReader + Send + Sync>,
    pub(super) key_prefix: u8,
    pub(super) keys: HashSet<K>,
    pub(super) write_queue: Vec<WriteOperation>,
    pub(super) write_cache: HashMap<K, V>,
}

impl<K, V> Debug for DbtMapPrivate<K, V>
where
    K: Debug,
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_struct("DbtMapPrivate")
            .field("name", &self.name)
            .field("reader", &"Arc<dyn StorageReader + Send + Sync>")
            .field("key_prefix", &self.key_prefix)
            .field("keys", &self.keys)
            .field("write_queue", &self.write_queue)
            .field("write_cache", &self.write_cache)
            .finish()
    }
}

impl<K, V> DbtMapPrivate<K, V>
where
    K: DeserializeOwned + Serialize + Eq + Hash + Clone,
    V: DeserializeOwned + Serialize + Clone,
{
    #[inline]
    pub(crate) fn new(
        reader: Arc<dyn StorageReader + Send + Sync>,
        key_prefix: u8,
        name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            key_prefix,
            reader,
            keys: HashSet::default(),
            write_queue: Vec::default(),
            write_cache: HashMap::default(),
        }
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.keys.contains(k)
    }

    #[inline]
    pub fn get(&self, k: &K) -> Option<Cow<V>> {
        if let Some(v) = self.write_cache.get(k) {
            return Some(Cow::Borrowed(v));
        }

        if !self.contains_key(k) {
            return None;
        }

        // try persistent storage
        let db_key: RustyKey = self.db_key(k);
        self.reader.get(db_key).map(|v| Cow::Owned(v.into_any()))
    }

    pub fn insert(&mut self, k: K, v: V) -> bool {
        let new = self.keys.insert(k.clone());
        if new {
            // key list grew by 1.  Set entire key-list
            self.write_queue.push(WriteOperation::Write(
                self.keylist_db_key(),
                RustyValue::from_any(&self.keys),
            ));
        }

        self.write_queue.push(WriteOperation::Write(
            self.db_key(&k),
            RustyValue::from_any(&v),
        ));

        let _ = self.write_cache.insert(k, v);

        new
    }

    /// This will remove the entry identified by `k` if it
    /// exists.
    pub fn remove(&mut self, k: &K) -> bool {
        // no-op if key not found.
        if !self.keys.remove(k) {
            return false;
        }

        self.write_queue.push(WriteOperation::Write(
            self.keylist_db_key(),
            RustyValue::from_any(&self.keys),
        ));

        self.write_cache.remove(k);

        // add to write queue
        self.write_queue
            .push(WriteOperation::Delete(self.db_key(k)));

        true
    }

    // Return the key used to store the length of the vector
    #[inline]
    pub(super) fn keylist_db_key(&self) -> RustyKey {
        let prefix_rk: RustyKey = self.key_prefix.into();
        let keylist_rk = RustyKey::from(b"_kl".as_ref());

        // This concatenates prefix + "_kl" to form the
        // real Key as used in LevelDB
        (prefix_rk, keylist_rk).into()
    }

    pub(super) fn persisted_keys(&self) -> Option<HashSet<K>> {
        self.reader.get(self.keylist_db_key()).map(|v| v.into_any())
    }

    /// Return the key of K type used to store the element at a given usize of usize type
    #[inline]
    pub(super) fn db_key(&self, k: &K) -> RustyKey {
        let prefix_rk: RustyKey = self.key_prefix.into();
        let k_rk = RustyKey::from_any(&k);

        // This concatenates prefix + "_kl" to form the
        // real Key as used in LevelDB
        (prefix_rk, k_rk).into()
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        for k in self.keys.clone().iter() {
            self.remove(k);
        }
        self.keys.clear()
    }

    pub(super) fn iter_keys(&self) -> impl Iterator<Item = &K> {
        self.keys.iter()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&K, Cow<'_, V>)> {
        self.keys.iter().map(|k| (k, self.get(k).unwrap()))
    }

    pub(super) fn iter_values(&self) -> impl Iterator<Item = Cow<'_, V>> {
        self.keys.iter().map(|k| self.get(k).unwrap())
    }
}

impl<K, V> DbTable for DbtMapPrivate<K, V>
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
        self.write_cache.clear();
        std::mem::take(&mut self.write_queue)
    }

    #[inline]
    fn restore_or_new(&mut self) {
        self.write_cache.clear();
        self.write_queue.clear();

        if let Some(keys) = self.persisted_keys() {
            self.keys = keys;
        }
    }
}
