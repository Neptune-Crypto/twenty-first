use super::super::storage_vec::Index;
use super::{
    traits::{DbTable, StorageReader},
    DbtSingleton, DbtVec,
};
use std::sync::Arc;

/// Provides a virtual database schema.
///
/// `DbtSchema` can create any number of instances of types that
/// implement [`DbTable`].
///
/// The application can perform writes to any subset of the
/// instances and then persist (write) the data atomically
/// to the database.
///
/// Thus we get something like relational DB transactions using
/// LevelDB key/val store.
pub struct DbtSchema<
    ParentKey,
    ParentValue,
    Reader: StorageReader<ParentKey, ParentValue> + Send + Sync,
> {
    /// These are the tables known by this `DbtSchema` instance.
    ///
    /// Implementor(s) of [`super::traits::StorageWriter`] will iterate over these
    /// tables, collect the pending operations, and write them
    /// atomically to the DB.
    pub tables: Vec<Box<dyn DbTable<ParentKey, ParentValue> + Send + Sync>>,

    /// Database Reader
    pub reader: Arc<Reader>,
}

impl<
        ParentKey,
        ParentValue,
        Reader: StorageReader<ParentKey, ParentValue> + 'static + Sync + Send,
    > DbtSchema<ParentKey, ParentValue, Reader>
{
    /// Create a new DbtVec
    ///
    /// The `DbtSchema` will keep a reference to the `DbtVec`. In this way,
    /// the Schema becomes aware of any write operations and later
    /// a [`super::traits::StorageWriter`] impl can write them all out.
    ///
    /// deprecated: please use [`Self::create_tables()`] instead.
    #[inline]
    pub fn new_vec<I, T>(&mut self, name: &str) -> DbtVec<ParentKey, ParentValue, Index, T>
    where
        ParentKey: From<Index> + 'static,
        ParentValue: From<T> + 'static,
        T: Clone + From<ParentValue> + 'static,
        ParentKey: From<(ParentKey, ParentKey)>,
        ParentKey: From<u8>,
        Index: From<ParentValue>,
        ParentValue: From<Index>,
        Index: From<u64> + 'static,
        DbtVec<ParentKey, ParentValue, Index, T>: DbTable<ParentKey, ParentValue> + Send + Sync,
    {
        assert!(self.tables.len() < 255);
        let reader = self.reader.clone();
        let key_prefix = self.tables.len() as u8;
        let vector = DbtVec::<ParentKey, ParentValue, Index, T>::new(reader, key_prefix, name);

        self.tables.push(Box::new(vector.clone()));
        vector
    }

    // possible future extension
    // fn new_hashmap<K, V>(&self) -> Arc<RefCell<DbtHashMap<K, V>>> { }

    /// Create a new DbtSingleton
    ///
    /// The `DbtSchema` will keep a reference to the `DbtSingleton`.
    /// In this way, the Schema becomes aware of any write operations
    /// and later a [`super::traits::StorageWriter`] impl can write them all out.
    ///
    /// deprecated: please use [`Self::create_tables()`] instead.
    #[inline]
    pub fn new_singleton<S>(&mut self, key: ParentKey) -> DbtSingleton<ParentKey, ParentValue, S>
    where
        S: Default + Eq + Clone + 'static,
        ParentKey: 'static,
        ParentValue: From<S> + 'static,
        ParentKey: From<(ParentKey, ParentKey)> + From<u8>,
        DbtSingleton<ParentKey, ParentValue, S>: DbTable<ParentKey, ParentValue> + Send + Sync,
    {
        let singleton = DbtSingleton::<ParentKey, ParentValue, S>::new(key, self.reader.clone());
        self.tables.push(Box::new(singleton.clone()));
        singleton
    }
}

/// Defines the types of tables that can be created by [`super::DbtSchema`]
#[derive(Clone)]
pub enum SchemaTable<K, V, T> {
    /// Singleton table ([`super::DbtSingleton`])
    Singleton(DbtSingleton<K, V, T>),
    /// Vec table ([`super::DbtVec`])
    Vec(DbtVec<K, V, Index, T>),
}

use std::collections::BTreeMap;
use std::sync::RwLock;

/// (String, SchemaTable) key/val tuple
pub type SchemaTableWithString<K, V, T> = (String, SchemaTable<K, V, T>);

/// (&str, SchemaTable) key/val tuple
pub type SchemaTableWithStr<'a, K, V, T> = (&'a str, SchemaTable<K, V, T>);

/// Map containing a named set of SchemaTable.
///
/// This type alias simplifies passing around a named set of SchemaTable
pub type TableGroup<K, V, T> = BTreeMap<String, SchemaTable<K, V, T>>;

/// Represents a group of tables over which all reads and writes are atomic.
pub struct AtomicSchema<K, V, T>(Arc<RwLock<TableGroup<K, V, T>>>);

impl<K, V, T> AtomicSchema<K, V, T> {
    /// Atomically perform read operations over a [`super::TableGroup`]
    pub fn perform_read_ops<'b, F>(&'b self, f: F)
    where
        F: Fn(std::sync::RwLockReadGuard<'b, TableGroup<K, V, T>>),
    {
        let lock = self.0.read().unwrap();
        f(lock)
    }

    /// Atomically perform write operations over a [`super::TableGroup`]
    pub fn perform_write_ops<'b, F>(&'b self, f: F)
    where
        F: Fn(std::sync::RwLockWriteGuard<'b, TableGroup<K, V, T>>),
    {
        let lock = self.0.write().unwrap();
        f(lock)
    }
}

impl<K, V, T> FromIterator<SchemaTableWithString<K, V, T>> for AtomicSchema<K, V, T> {
    fn from_iter<I: IntoIterator<Item = SchemaTableWithString<K, V, T>>>(v: I) -> Self {
        Self(Arc::new(RwLock::new(v.into_iter().collect())))
    }
}

impl<'a, K, V, T> FromIterator<SchemaTableWithStr<'a, K, V, T>> for AtomicSchema<K, V, T> {
    fn from_iter<I: IntoIterator<Item = SchemaTableWithStr<'a, K, V, T>>>(v: I) -> Self {
        Self(Arc::new(RwLock::new(
            v.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        )))
    }
}

impl<K, V, T> FromIterator<SchemaTable<K, V, T>> for AtomicSchema<K, V, T> {
    fn from_iter<I: IntoIterator<Item = SchemaTable<K, V, T>>>(v: I) -> Self {
        Self(Arc::new(RwLock::new(
            v.into_iter()
                .enumerate()
                .map(|(i, v)| (format!("table_{}", i + 1), v))
                .collect(),
        )))
    }
}
