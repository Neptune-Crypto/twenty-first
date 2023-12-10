use super::super::storage_vec::traits::*;
use super::super::storage_vec::Index;
use super::dbtvec_private::DbtVecPrivate;
use super::{traits::*, RustyValue, VecWriteOperation, WriteOperation};
use crate::sync::AtomicRw;
use std::{
    fmt::Debug,
    sync::{Arc, RwLockReadGuard, RwLockWriteGuard},
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
#[derive(Debug)]
pub struct DbtVec<V> {
    inner: AtomicRw<DbtVecPrivate<V>>,
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
            inner: AtomicRw::from(vec),
        }
    }
}

impl<V> StorageVecRwLock<V> for DbtVec<V> {
    type LockedData = DbtVecPrivate<V>;

    #[inline]
    fn write_lock(&self) -> RwLockWriteGuard<'_, Self::LockedData> {
        self.inner.guard_mut()
    }

    // This is a private method, but we allow unit tests in super to use it.
    #[inline]
    fn read_lock(&self) -> RwLockReadGuard<'_, Self::LockedData> {
        self.inner.guard()
    }
}

impl<V> StorageVec<V> for DbtVec<V>
where
    V: Clone + Debug,
    V: From<RustyValue>,
{
    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.with(|inner| inner.is_empty())
    }

    #[inline]
    fn len(&self) -> Index {
        self.inner.with(|inner| inner.len())
    }

    #[inline]
    fn get(&self, index: Index) -> V {
        self.inner.with(|inner| inner.get(index))
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
        let inner = self.inner.guard();
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
        let inner = self.inner.guard();
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
        self.inner.with(|inner| inner.get_many(indices))
    }

    #[inline]
    fn get_all(&self) -> Vec<V> {
        self.inner.with(|inner| inner.get_all())
    }

    #[inline]
    fn set(&self, index: Index, value: V) {
        self.inner.with_mut(|inner| inner.set(index, value));
    }

    #[inline]
    fn set_many(&self, key_vals: impl IntoIterator<Item = (Index, V)>) {
        self.inner.with_mut(|inner| inner.set_many(key_vals));
    }

    #[inline]
    fn pop(&self) -> Option<V> {
        self.inner.with_mut(|inner| inner.pop())
    }

    #[inline]
    fn push(&self, value: V) {
        self.inner.with_mut(|inner| inner.push(value));
    }

    #[inline]
    fn clear(&self) {
        self.inner.with_mut(|inner| inner.clear());
    }
}

impl<V> DbTable for DbtVec<V>
where
    V: Clone,
    V: From<RustyValue>,
    RustyValue: From<V>,
{
    /// Collect all added elements that have not yet been persisted
    fn pull_queue(&self) -> Vec<WriteOperation> {
        self.inner.with_mut(|inner| {
            let maybe_original_length = inner.persisted_length();
            // necessary because we need maybe_original_length.is_none() later
            let original_length = maybe_original_length.unwrap_or(0);
            let mut length = original_length;
            let mut queue = vec![];
            while let Some(write_element) = inner.write_queue.pop_front() {
                match write_element {
                    VecWriteOperation::OverWrite((i, t)) => {
                        let key = inner.get_index_key(i);
                        queue.push(WriteOperation::Write(key, t.into()));
                    }
                    VecWriteOperation::Push(t) => {
                        let key = inner.get_index_key(length);
                        length += 1;
                        queue.push(WriteOperation::Write(key, t.into()));
                    }
                    VecWriteOperation::Pop => {
                        let key = inner.get_index_key(length - 1);
                        length -= 1;
                        queue.push(WriteOperation::Delete(key));
                    }
                };
            }

            if original_length != length || maybe_original_length.is_none() {
                let key = DbtVecPrivate::<V>::get_length_key(inner.key_prefix);
                queue.push(WriteOperation::Write(key, length.into()));
            }

            inner.cache.clear();

            queue
        })
    }

    #[inline]
    fn restore_or_new(&self) {
        self.inner.with_mut(|inner| {
            if let Some(length) = inner
                .reader
                .get(DbtVecPrivate::<V>::get_length_key(inner.key_prefix))
            {
                inner.current_length = Some(length.into());
            } else {
                inner.current_length = Some(0);
            }
            inner.cache.clear();
            inner.write_queue.clear();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::storage_vec::traits::tests as traits_tests;
    use super::super::SimpleRustyStorage;
    use super::*;
    use crate::storage::level_db::DB;

    fn gen_concurrency_test_vec() -> DbtVec<u64> {
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
