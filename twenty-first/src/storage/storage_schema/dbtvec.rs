use super::super::storage_vec::traits::*;
use super::super::storage_vec::Index;
use super::dbtvec_private::DbtVecPrivate;
use super::{traits::*, RustyValue, VecWriteOperation, WriteOperation};
use crate::sync::{AtomicRwReadGuard, AtomicRwWriteGuard};
use serde::{de::DeserializeOwned, Serialize};
use std::{cell::RefCell, fmt::Debug, rc::Rc, sync::Arc};

/// A DB-backed Vec for use with DBSchema
///
/// This type is NOT concurrency-safe.
///
/// `DbtVec` is a NewType around Rc<RefCell<..>>.  Thus it
/// can be cheaply cloned to create a reference as if it were an
/// Rc.
#[derive(Debug)]
pub struct DbtVec<V> {
    inner: Rc<RefCell<DbtVecPrivate<V>>>,
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
            inner: Rc::new(RefCell::new(vec)),
        }
    }
}

impl<V> StorageVec<V> for DbtVec<V>
where
    V: Clone + Debug,
    V: DeserializeOwned,
{
    #[inline]
    fn is_empty(&self) -> bool {
        self.inner.borrow().is_empty()
    }

    #[inline]
    fn len(&self) -> Index {
        self.inner.borrow().len()
    }

    #[inline]
    fn get(&self, index: Index) -> V {
        self.inner.borrow().get(index)
    }

    #[inline]
    fn many_iter<'a>(
        &'a self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, V)> + '_> {
        let inner = self.inner.borrow();
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
                (i, db_element.into_any())
            }
        }))
    }

    #[inline]
    fn many_iter_values<'a>(
        &'a self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = V> + '_> {
        let inner = self.inner.borrow();
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
                db_element.into_any()
            }
        }))
    }

    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<V> {
        self.inner.borrow().get_many(indices)
    }

    #[inline]
    fn get_all(&self) -> Vec<V> {
        self.inner.borrow().get_all()
    }

    #[inline]
    fn set(&self, index: Index, value: V) {
        self.inner.borrow_mut().set(index, value);
    }

    #[inline]
    fn set_many(&self, key_vals: impl IntoIterator<Item = (Index, V)>) {
        self.inner.borrow_mut().set_many(key_vals);
    }

    #[inline]
    fn pop(&self) -> Option<V> {
        self.inner.borrow_mut().pop()
    }

    #[inline]
    fn push(&self, value: V) {
        self.inner.borrow_mut().push(value);
    }

    #[inline]
    fn clear(&self) {
        self.inner.borrow_mut().clear();
    }
}

impl<T> StorageVecRwLock<T> for DbtVec<T> {
    type LockedData = DbtVecPrivate<T>;

    #[inline]
    fn try_write_lock(&self) -> Option<AtomicRwWriteGuard<'_, Self::LockedData>> {
        None
    }

    #[inline]
    fn try_read_lock(&self) -> Option<AtomicRwReadGuard<'_, Self::LockedData>> {
        None
    }
}

impl<V> DbTable for DbtVec<V>
where
    V: Clone,
    V: Serialize + DeserializeOwned,
{
    /// Collect all added elements that have not yet been persisted
    ///
    /// note: this clears the internal cache.  Thus the cache does
    /// not grow unbounded, so long as `pull_queue()` is called
    /// regularly.  It also means the cache must be rebuilt after
    /// each call (batch write)
    fn pull_queue(&self) -> Vec<WriteOperation> {
        let mut inner = self.inner.borrow_mut();

        let maybe_original_length = inner.persisted_length();
        // necessary because we need maybe_original_length.is_none() later
        let original_length = maybe_original_length.unwrap_or(0);
        let mut length = original_length;
        let mut queue = vec![];
        while let Some(write_element) = inner.write_queue.pop_front() {
            match write_element {
                VecWriteOperation::OverWrite((i, t)) => {
                    let key = inner.get_index_key(i);
                    queue.push(WriteOperation::Write(key, RustyValue::from_any(&t)));
                }
                VecWriteOperation::Push(t) => {
                    let key = inner.get_index_key(length);
                    length += 1;
                    queue.push(WriteOperation::Write(key, RustyValue::from_any(&t)));
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
            queue.push(WriteOperation::Write(key, RustyValue::from_any(&length)));
        }

        inner.cache.clear();

        queue
    }

    #[inline]
    fn restore_or_new(&self) {
        let mut inner = self.inner.borrow_mut();

        if let Some(length) = inner
            .reader
            .get(DbtVecPrivate::<V>::get_length_key(inner.key_prefix))
        {
            inner.current_length = Some(length.into_any());
        } else {
            inner.current_length = Some(0);
        }
        inner.cache.clear();
        inner.write_queue.clear();
    }
}
