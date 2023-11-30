use super::ordinary_vec_private::OrdinaryVecPrivate;
use super::{traits::*, Index};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// OrdinaryVec is a wrapper that adds RwLock and atomic snapshot
/// guarantees around all accesses to an ordinary `Vec<T>`
#[derive(Debug, Clone)]
pub struct OrdinaryVec<T>(Arc<RwLock<OrdinaryVecPrivate<T>>>);

impl<T> From<Vec<T>> for OrdinaryVec<T> {
    fn from(v: Vec<T>) -> Self {
        Self(Arc::new(RwLock::new(OrdinaryVecPrivate(v))))
    }
}

impl<T> From<&[T]> for OrdinaryVec<T>
where
    T: Copy,
{
    fn from(v: &[T]) -> Self {
        Self(Arc::new(RwLock::new(OrdinaryVecPrivate(v.to_vec()))))
    }
}

impl<T: Clone> StorageVecReads<T> for OrdinaryVec<T> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.read().unwrap().is_empty()
    }

    #[inline]
    fn len(&self) -> Index {
        self.0.read().unwrap().len()
    }

    #[inline]
    fn get(&self, index: Index) -> T {
        self.0.read().unwrap().get(index)
    }

    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        // note: this lock is moved into the iterator closure and is not
        //       released until caller drops the returned iterator
        let inner = self.0.read().unwrap();

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
        let inner = self.0.read().unwrap();

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
}

impl<T: Clone> StorageVecImmutableWrites<T> for OrdinaryVec<T> {
    #[inline]
    fn set(&self, index: Index, value: T) {
        // note: on 32 bit systems, this could panic.
        self.0.write().unwrap().set(index, value);
    }

    #[inline]
    fn pop(&self) -> Option<T> {
        self.0.write().unwrap().pop()
    }

    #[inline]
    fn push(&self, value: T) {
        self.0.write().unwrap().push(value);
    }
}

impl<T> StorageVecRwLock<T> for OrdinaryVec<T> {
    type LockedData = OrdinaryVecPrivate<T>;

    #[inline]
    fn write_lock(&self) -> RwLockWriteGuard<'_, Self::LockedData> {
        self.0
            .write()
            .expect("should have acquired OrdinaryVec write lock")
    }

    #[inline]
    fn read_lock(&self) -> RwLockReadGuard<'_, Self::LockedData> {
        self.0
            .read()
            .expect("should have acquired OrdinaryVec read lock")
    }
}

impl<T: Clone> StorageVec<T> for OrdinaryVec<T> {}
