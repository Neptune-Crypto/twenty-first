use lending_iterator::prelude::*;
use lending_iterator::{gat, LendingIterator};
use std::iter::Iterator;
// use std::{
//     collections::{HashMap, VecDeque},
//     sync::{Arc, Mutex},
// };
use super::{
    traits::{StorageVecImmutableWrites, StorageVecReads},
    Index,
};

/// a mutating iterator for StorageVec trait
pub struct ManyIterMut<'a, V, T>
where
    V: StorageVecImmutableWrites<T> + ?Sized,
{
    indices: Box<dyn Iterator<Item = Index>>,
    data: &'a V,
    phantom: std::marker::PhantomData<T>,
}

impl<'a, V, T> ManyIterMut<'a, V, T>
where
    V: StorageVecImmutableWrites<T>,
{
    pub(super) fn new<I>(indices: I, data: &'a V) -> Self
    where
        I: IntoIterator<Item = Index> + 'static,
    {
        Self {
            indices: Box::new(indices.into_iter()),
            data,
            phantom: Default::default(),
        }
    }
}

// LendingIterator trait gives us all the nice iterator type functions.
// We only have to impl next()
#[gat]
impl<'a, V, T: 'a> LendingIterator for ManyIterMut<'a, V, T>
where
    V: StorageVecImmutableWrites<T> + StorageVecReads<T>,
{
    type Item<'b> = StorageSetter<'b, V, T>
    where
        Self: 'b;

    fn next(&mut self) -> Option<Self::Item<'_>> {
        if let Some(i) = Iterator::next(&mut self.indices) {
            self.data.get_mut(i)
        } else {
            None
        }
    }
}

/// used for accessing and setting values returned from StorageVec::get_mut() and mutable iterators
pub struct StorageSetter<'a, V, T>
where
    V: StorageVecImmutableWrites<T> + ?Sized,
{
    pub(super) vec: &'a V,
    pub(super) index: Index,
    pub(super) value: T,
}

impl<'a, V, T> StorageSetter<'a, V, T>
where
    V: StorageVecImmutableWrites<T> + ?Sized,
{
    pub fn set(&mut self, value: T) {
        self.vec.set(self.index, value);
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}
