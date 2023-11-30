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
