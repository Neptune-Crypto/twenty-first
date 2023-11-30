use super::super::storage_vec::{traits::*, Index};
use super::dbtvec_private::DbtVecPrivate;
use super::{traits::*, VecWriteOperation, WriteOperation};
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
}

impl<K, V, T> StorageVecRwLock<T> for DbtVec<K, V, Index, T> {
    type LockedData = DbtVecPrivate<K, V, Index, T>;

    #[inline]
    fn write_lock(&self) -> RwLockWriteGuard<'_, Self::LockedData> {
        self.inner
            .write()
            .expect("should have acquired DbtVec write lock")
    }

    // This is a private method, but we allow unit tests in super to use it.
    #[inline]
    fn read_lock(&self) -> RwLockReadGuard<'_, Self::LockedData> {
        self.inner.read().expect("should acquire DbtVec read lock")
    }
}

impl<K, V, T> StorageVecReads<T> for DbtVec<K, V, Index, T>
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

    // this fn is here to satisfy the trait but is actually
    // implemented by DbtVec
    // todo
    // fn iter_keys<'a>(&'a self) -> Box<dyn Iterator<Item = Index> + '_> {
    //     unreachable!()
    // }

    #[inline]
    fn many_iter<'a>(
        &'a self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        let inner = self.read_lock();

        Box::new(indices.into_iter().map(move |i| {
            assert!(
                i < inner.len(),
                "Out-of-bounds. Got index {} but length was {}. persisted vector name: {}",
                i,
                inner.len(),
                inner.name
            );

            if inner.cache.contains_key(&i) {
                (i, inner.cache[&i].clone())
            } else {
                let key = inner.get_index_key(i);
                let db_element = inner.reader.get(key).unwrap();
                (i, T::from(db_element))
            }
        }))
    }

    #[inline]
    fn many_iter_values<'a>(
        &'a self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = T> + '_> {
        let inner = self.read_lock();

        Box::new(indices.into_iter().map(move |i| {
            assert!(
                i < inner.len(),
                "Out-of-bounds. Got index {} but length was {}. persisted vector name: {}",
                i,
                inner.len(),
                inner.name
            );

            if inner.cache.contains_key(&i) {
                inner.cache[&i].clone()
            } else {
                let key = inner.get_index_key(i);
                let db_element = inner.reader.get(key).unwrap();
                T::from(db_element)
            }
        }))
    }

    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        self.read_lock().get_many(indices)
    }

    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.read_lock().get_all()
    }
}

impl<K, V, T> StorageVecImmutableWrites<T> for DbtVec<K, V, Index, T>
where
    K: From<Index>,
    V: From<T>,
    T: Clone + From<V> + Debug,
    K: From<(K, K)>,
    K: From<u8>,
    Index: From<V> + From<u64>,
{
    // type LockedData = DbtVecPrivate<K, V, Index, T>;

    #[inline]
    fn set(&self, index: Index, value: T) {
        self.write_lock().set(index, value)
    }

    #[inline]
    fn set_many(&self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        self.write_lock().set_many(key_vals)
    }

    #[inline]
    fn pop(&self) -> Option<T> {
        self.write_lock().pop()
    }

    #[inline]
    fn push(&self, value: T) {
        self.write_lock().push(value)
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
}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtVec<ParentKey, ParentValue, Index, T>
where
    ParentKey: From<Index>,
    ParentValue: From<T>,
    T: Clone,
    T: From<ParentValue>,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    Index: From<ParentValue>,
    ParentValue: From<Index>,
{
    /// Collect all added elements that have not yet been persisted
    fn pull_queue(&self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        let mut lock = self.write_lock();

        let maybe_original_length = lock.persisted_length();
        // necessary because we need maybe_original_length.is_none() later
        let original_length = maybe_original_length.unwrap_or(0);
        let mut length = original_length;
        let mut queue = vec![];
        while let Some(write_element) = lock.write_queue.pop_front() {
            match write_element {
                VecWriteOperation::OverWrite((i, t)) => {
                    let key = lock.get_index_key(i);
                    queue.push(WriteOperation::Write(key, Into::<ParentValue>::into(t)));
                }
                VecWriteOperation::Push(t) => {
                    let key = lock.get_index_key(length);
                    length += 1;
                    queue.push(WriteOperation::Write(key, Into::<ParentValue>::into(t)));
                }
                VecWriteOperation::Pop => {
                    let key = lock.get_index_key(length - 1);
                    length -= 1;
                    queue.push(WriteOperation::Delete(key));
                }
            };
        }

        if original_length != length || maybe_original_length.is_none() {
            let key =
                DbtVecPrivate::<ParentKey, ParentValue, Index, T>::get_length_key(lock.key_prefix);
            queue.push(WriteOperation::Write(
                key,
                Into::<ParentValue>::into(length),
            ));
        }

        lock.cache.clear();

        queue
    }

    #[inline]
    fn restore_or_new(&self) {
        let mut lock = self.write_lock();

        if let Some(length) = lock
            .reader
            .get(DbtVecPrivate::<ParentKey, ParentValue, Index, T>::get_length_key(lock.key_prefix))
        {
            lock.current_length = Some(length.into());
        } else {
            lock.current_length = Some(0);
        }
        lock.cache.clear();
        lock.write_queue.clear();
    }
}
