use super::iterators::{ManyIterMut, StorageSetter};
use serde::{de::DeserializeOwned, Serialize};

pub type Index = u64;

pub trait StorageVec<T> {
    /// check if collection is empty
    fn is_empty(&self) -> bool;

    /// get collection length
    fn len(&self) -> Index;

    /// get single element at index
    fn get(&self, index: Index) -> T;

    #[inline]
    fn get_mut(&mut self, index: Index) -> Option<StorageSetter<Self, T>> {
        let value = self.get(index);
        Some(StorageSetter {
            vec: self,
            index,
            value,
        })
    }

    /// get multiple elements matching indices
    ///
    /// This is a convenience method. For large collections
    /// it will be more efficient to use `many_iter` directly
    /// and avoid allocating a Vec
    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        self.many_iter(indices.to_vec()).map(|(_i, v)| v).collect()
    }

    /// get all elements
    ///
    /// This is a convenience method. For large collections
    /// it may be more efficient to use `iter` directly
    /// and avoid allocating a Vec
    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.iter().map(|(_i, v)| v).collect()
    }

    /// get an iterator over all elements
    #[inline]
    fn iter(&self) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        self.many_iter(0..self.len())
    }

    /// get a mutable iterator over all elements
    #[inline]
    fn iter_mut(&mut self) -> ManyIterMut<Self, T>
    where
        Self: Sized,
    {
        ManyIterMut::new(0..self.len(), self)
    }

    /// get an iterator over elements matching indices
    fn many_iter(
        &self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_>;

    /// get a mutable iterator over elements matching indices
    #[inline]
    fn many_iter_mut(
        &mut self,
        indices: impl IntoIterator<Item = Index> + 'static,
    ) -> ManyIterMut<Self, T>
    where
        Self: Sized,
    {
        ManyIterMut::new(indices, self)
    }

    /// set a single element.
    fn set(&mut self, index: Index, value: T);

    /// set multiple elements.
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        for (key, val) in key_vals.into_iter() {
            self.set(key, val)
        }
    }

    /// set elements from start to vals.count()
    #[inline]
    fn set_first_n(&mut self, vals: impl IntoIterator<Item = T>) {
        self.set_many((0..).zip(vals));
    }

    /// set all elements with a simple list of values in an array or Vec
    /// and validates that input length matches target length.
    ///
    /// calls ::set_many() internally.
    ///
    /// panics if input length does not match target length.
    ///
    /// note: casts the input value's length from usize to Index
    ///       so will panic if vals contains more than 2^32 items
    #[inline]
    fn set_all(&mut self, vals: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = T>>) {
        let iter = vals.into_iter();

        assert!(
            iter.len() as Index == self.len(),
            "size-mismatch.  input has {} elements and target has {} elements.",
            iter.len(),
            self.len(),
        );

        self.set_first_n(iter);
    }

    /// pop an element from end of collection
    fn pop(&mut self) -> Option<T>;

    /// push an element to end of collection
    fn push(&mut self, value: T);
}

pub enum WriteElement<T: Serialize + DeserializeOwned> {
    OverWrite((Index, T)),
    Push(T),
    Pop,
}
