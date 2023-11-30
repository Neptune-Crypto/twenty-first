use super::{traits::*, Index};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// OrdinaryVec is a public wrapper that adds RwLock around
/// all accesses to an ordinary Vec<T>
#[derive(Debug, Clone)]
pub struct OrdinaryVec<T>(Arc<RwLock<Vec<T>>>);

impl<T> From<Vec<T>> for OrdinaryVec<T> {
    fn from(v: Vec<T>) -> Self {
        Self(Arc::new(RwLock::new(v)))
    }
}

impl<T: Clone> StorageVecReads<T> for OrdinaryVec<T> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.read().unwrap().is_empty()
    }

    #[inline]
    fn len(&self) -> Index {
        self.0.read().unwrap().len() as Index
    }

    #[inline]
    fn get(&self, index: Index) -> T {
        self.0.write().unwrap()[index as usize].clone()
    }

    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        Box::new(
            indices
                .into_iter()
                .map(|index| (index, self.0.read().unwrap()[index as usize].clone())),
        )
    }

    fn many_iter_values(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = T> + '_> {
        Box::new(
            indices
                .into_iter()
                .map(|index| self.0.read().unwrap()[index as usize].clone()),
        )
    }

    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        indices
            .iter()
            .map(|index| self.0.read().unwrap()[*index as usize].clone())
            .collect()
    }

    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.0.read().unwrap().clone()
    }
}

impl<T: Clone> StorageVecImmutableWrites<T> for OrdinaryVec<T> {
    #[inline]
    fn set(&self, index: Index, value: T) {
        // note: on 32 bit systems, this could panic.
        self.0.write().unwrap()[index as usize] = value;
    }

    /// set multiple elements.
    ///
    /// panics if key_vals contains an index not in the collection
    ///
    /// It is the caller's responsibility to ensure that index values are
    /// unique.  If not, the last value with the same index will win.
    /// For unordered collections such as HashMap, the behavior is undefined.
    #[inline]
    fn set_many(&self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        for (index, value) in key_vals.into_iter() {
            // note: on 32 bit systems, this could panic.
            self.0.write().unwrap()[index as usize] = value;
        }
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
    type LockedData = Vec<T>;

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
