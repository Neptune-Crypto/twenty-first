use super::storage_vec_trait::{Index, StorageVec};

pub struct OrdinaryVec<T>(Vec<T>);

impl<T> From<Vec<T>> for OrdinaryVec<T> {
    fn from(v: Vec<T>) -> Self {
        Self(v)
    }
}

impl<T: Clone> StorageVec<T> for OrdinaryVec<T> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn len(&self) -> Index {
        self.0.len() as Index
    }

    #[inline]
    fn get(&self, index: Index) -> T {
        self.0[index as usize].clone()
    }

    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        indices
            .iter()
            .map(|index| self.0[*index as usize].clone())
            .collect()
    }

    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.0.clone()
    }

    #[inline]
    fn set(&mut self, index: Index, value: T) {
        // note: on 32 bit systems, this could panic.
        self.0[index as usize] = value;
    }

    /// set multiple elements.
    ///
    /// panics if key_vals contains an index not in the collection
    ///
    /// It is the caller's responsibility to ensure that index values are
    /// unique.  If not, the last value with the same index will win.
    /// For unordered collections such as HashMap, the behavior is undefined.
    #[inline]
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        for (index, value) in key_vals.into_iter() {
            // note: on 32 bit systems, this could panic.
            self.0[index as usize] = value;
        }
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.0.push(value);
    }
}
