use super::super::storage_vec::traits::*;
use super::super::storage_vec::Index;
use super::dbtvec_private::DbtVecPrivate;
use super::{traits::*, RustyValue, VecWriteOperation, WriteOperation};
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
pub struct DbtVec<V> {
    inner: Arc<RwLock<DbtVecPrivate<V>>>,
}

impl<V> Clone for DbtVec<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<V> DbtVec<V>
where
    V: Clone,
{
    // DbtVec cannot be instantiated directly outside of this crate.
    #[inline]
    pub(crate) fn new(
        reader: Arc<dyn StorageReader + Send + Sync>,
        key_prefix: u8,
        name: &str,
    ) -> Self {
        let vec = DbtVecPrivate::<V>::new(reader, key_prefix, name);

        Self {
            inner: Arc::new(RwLock::new(vec)),
        }
    }
}

impl<V> StorageVecRwLock<V> for DbtVec<V> {
    type LockedData = DbtVecPrivate<V>;

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

impl<V> StorageVecReads<V> for DbtVec<V>
where
    V: Clone + Debug,
    V: From<RustyValue>,
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
    fn get(&self, index: Index) -> V {
        self.read_lock().get(index)
    }

    // this fn is here to satisfy the trait but is actually
    // implemented by DbtVec
    // todo
    // fn iter_keys<'a>(&'a self) -> Box<dyn Iterator<Item = K> + '_> {
    //     unreachable!()
    // }

    #[inline]
    fn many_iter<'a>(
        &'a self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, V)> + '_> {
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
                (i, V::from(db_element))
            }
        }))
    }

    #[inline]
    fn many_iter_values<'a>(
        &'a self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = V> + '_> {
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
                V::from(db_element)
            }
        }))
    }

    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<V> {
        self.read_lock().get_many(indices)
    }

    #[inline]
    fn get_all(&self) -> Vec<V> {
        self.read_lock().get_all()
    }
}

impl<V> StorageVecImmutableWrites<V> for DbtVec<V>
where
    V: Clone + Debug + From<RustyValue>,
    Index: From<V> + From<u64>,
{
    // type LockedData = DbtVecPrivate<V>;

    #[inline]
    fn set(&self, index: Index, value: V) {
        self.write_lock().set(index, value)
    }

    #[inline]
    fn set_many(&self, key_vals: impl IntoIterator<Item = (Index, V)>) {
        self.write_lock().set_many(key_vals)
    }

    #[inline]
    fn pop(&self) -> Option<V> {
        self.write_lock().pop()
    }

    #[inline]
    fn push(&self, value: V) {
        self.write_lock().push(value)
    }

    #[inline]
    fn clear(&self) {
        self.write_lock().clear()
    }
}

impl<V> StorageVec<V> for DbtVec<V>
where
    V: Clone + Debug + From<RustyValue>,
    Index: From<V> + From<u64>,
{
}

impl<V> DbTable for DbtVec<V>
where
    V: Clone,
    Index: From<V>,
    V: From<Index>,
    V: From<RustyValue>,
    RustyValue: From<V>,
{
    /// Collect all added elements that have not yet been persisted
    fn pull_queue(&self) -> Vec<WriteOperation> {
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
                    queue.push(WriteOperation::Write(key, t.into()));
                }
                VecWriteOperation::Push(t) => {
                    let key = lock.get_index_key(length);
                    length += 1;
                    queue.push(WriteOperation::Write(key, t.into()));
                }
                VecWriteOperation::Pop => {
                    let key = lock.get_index_key(length - 1);
                    length -= 1;
                    queue.push(WriteOperation::Delete(key));
                }
            };
        }

        if original_length != length || maybe_original_length.is_none() {
            let key = DbtVecPrivate::<V>::get_length_key(lock.key_prefix);
            queue.push(WriteOperation::Write(key, length.into()));
        }

        lock.cache.clear();

        queue
    }

    #[inline]
    fn restore_or_new(&self) {
        let mut lock = self.write_lock();

        if let Some(length) = lock
            .reader
            .get(DbtVecPrivate::<V>::get_length_key(lock.key_prefix))
        {
            lock.current_length = Some(length.into());
        } else {
            lock.current_length = Some(0);
        }
        lock.cache.clear();
        lock.write_queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::storage_vec::traits::tests as traits_tests;
    use super::super::{RustyKey, RustyValue, SimpleRustyStorage};
    use super::*;
    use crate::storage::level_db::DB;

    fn gen_concurrency_test_vec() -> DbtVec<RustyKey, RustyValue, u64> {
        // open new DB that will be removed on close.
        let db = DB::open_new_test_database(true, None, None, None).unwrap();
        let mut rusty_storage = SimpleRustyStorage::new(db);
        rusty_storage.schema.new_vec::<u64>("atomicity-test-vector")
    }

    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: Any { .. }")]
    #[test]
    fn non_atomic_set_and_get() {
        traits_tests::concurrency::non_atomic_set_and_get(&gen_concurrency_test_vec());
    }

    #[test]
    fn atomic_setmany_and_getmany() {
        traits_tests::concurrency::atomic_setmany_and_getmany(&gen_concurrency_test_vec());
    }

    #[test]
    fn atomic_setall_and_getall() {
        traits_tests::concurrency::atomic_setall_and_getall(&gen_concurrency_test_vec());
    }

    #[test]
    fn atomic_iter_mut_and_iter() {
        traits_tests::concurrency::atomic_iter_mut_and_iter(&gen_concurrency_test_vec());
    }
}
