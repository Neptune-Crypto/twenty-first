use super::super::storage_vec::Index;
use super::{DbTable, DbtSingleton, DbtSingletonReference, DbtVec, DbtVecReference, StorageReader};
use std::{cell::RefCell, sync::Arc};

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
pub struct DbtSchema<ParentKey, ParentValue, Reader: StorageReader<ParentKey, ParentValue>> {
    pub(crate) tables: Vec<Arc<RefCell<dyn DbTable<ParentKey, ParentValue> + Send + Sync>>>,
    pub(crate) reader: Arc<Reader>,
}

impl<
        ParentKey,
        ParentValue,
        Reader: StorageReader<ParentKey, ParentValue> + 'static + Sync + Send,
    > DbtSchema<ParentKey, ParentValue, Reader>
{
    /// Create a new DbtVecReference
    #[inline]
    pub fn new_vec<I, T>(&mut self, name: &str) -> DbtVecReference<ParentKey, ParentValue, T>
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

        let arc_refcell_vector = Arc::new(RefCell::new(vector));
        self.tables.push(arc_refcell_vector.clone());
        arc_refcell_vector
    }

    // possible future extension
    // fn new_hashmap<K, V>(&self) -> Arc<RefCell<DbtHashMap<K, V>>> { }

    /// Create a new DbtSingletonReference
    #[inline]
    pub fn new_singleton<S>(
        &mut self,
        key: ParentKey,
    ) -> DbtSingletonReference<ParentKey, ParentValue, S>
    where
        S: Default + Eq + Clone + 'static,
        ParentKey: 'static,
        ParentValue: From<S> + 'static,
        ParentKey: From<(ParentKey, ParentKey)> + From<u8>,
        DbtSingleton<ParentKey, ParentValue, S>: DbTable<ParentKey, ParentValue> + Send + Sync,
    {
        let singleton = DbtSingleton::<ParentKey, ParentValue, S>::new(key, self.reader.clone());
        let arc_refcell_singleton = Arc::new(RefCell::new(singleton));
        self.tables.push(arc_refcell_singleton.clone());
        arc_refcell_singleton
    }
}
