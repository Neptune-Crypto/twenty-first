use super::ordinary_vec_private::OrdinaryVecPrivate;
use super::{traits::*, Index};
use crate::sync::{AtomicRwReadGuard, AtomicRwWriteGuard};
use std::{cell::RefCell, rc::Rc};

/// A NewType for [`Vec`] that implements [`StorageVec`] trait
///
/// note: `OrdinaryVec` is NOT thread-safe.
#[derive(Debug, Clone, Default)]
pub struct OrdinaryVec<T>(Rc<RefCell<OrdinaryVecPrivate<T>>>);

impl<T> From<Vec<T>> for OrdinaryVec<T> {
    fn from(v: Vec<T>) -> Self {
        Self(Rc::new(RefCell::new(OrdinaryVecPrivate(v))))
    }
}

impl<T> StorageVecRwLock<T> for OrdinaryVec<T> {
    type LockedData = OrdinaryVecPrivate<T>;

    #[inline]
    fn try_write_lock(&self) -> Option<AtomicRwWriteGuard<'_, Self::LockedData>> {
        None
    }

    #[inline]
    fn try_read_lock(&self) -> Option<AtomicRwReadGuard<'_, Self::LockedData>> {
        None
    }
}

impl<T: Clone> StorageVec<T> for OrdinaryVec<T> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }

    #[inline]
    fn len(&self) -> Index {
        self.0.borrow().len()
    }

    #[inline]
    fn get(&self, index: Index) -> T {
        self.0.borrow().get(index)
    }

    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        // note: this lock is moved into the iterator closure and is not
        //       released until caller drops the returned iterator
        let inner = self.0.borrow();

        Box::new(indices.into_iter().map(move |i| {
            assert!(
                i < inner.len(),
                "Out-of-bounds. Got index {} but length was {}.",
                i,
                inner.len(),
            );
            (i, inner.get(i))
        }))
    }

    fn many_iter_values(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = T> + '_> {
        // note: this lock is moved into the iterator closure and is not
        //       released until caller drops the returned iterator
        let inner = self.0.borrow();

        Box::new(indices.into_iter().map(move |i| {
            assert!(
                i < inner.len(),
                "Out-of-bounds. Got index {} but length was {}.",
                i,
                inner.len(),
            );
            inner.get(i)
        }))
    }

    #[inline]
    fn set(&self, index: Index, value: T) {
        // note: on 32 bit systems, this could panic.
        self.0.borrow_mut().set(index, value);
    }

    #[inline]
    fn set_many(&self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        self.0.borrow_mut().set_many(key_vals);
    }

    #[inline]
    fn pop(&self) -> Option<T> {
        self.0.borrow_mut().pop()
    }

    #[inline]
    fn push(&self, value: T) {
        self.0.borrow_mut().push(value);
    }

    #[inline]
    fn clear(&self) {
        self.0.borrow_mut().clear();
    }
}
