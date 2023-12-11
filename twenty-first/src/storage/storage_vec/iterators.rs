use super::{traits::*, Index};
use lending_iterator::prelude::*;
use lending_iterator::{gat, LendingIterator};
use std::iter::Iterator;
use std::marker::PhantomData;
use std::sync::RwLockWriteGuard;

/// A mutating iterator for [`StorageVec`] trait
///
/// Important: This iterator holds a write lock over `StorageVecRwLock::LockedData`
/// which will not be released until the iterator is dropped.
///
/// See examples for [`StorageVec::iter_mut()`].
#[allow(private_bounds)]
pub struct ManyIterMut<'a, V, T>
where
    V: StorageVec<T> + StorageVecRwLock<T> + ?Sized,
{
    indices: Box<dyn Iterator<Item = Index>>,
    write_lock: RwLockWriteGuard<'a, V::LockedData>,
    phantom_t: PhantomData<T>,
    phantom_d: PhantomData<V>,
}

#[allow(private_bounds)]
impl<'a, V, T> ManyIterMut<'a, V, T>
where
    V: StorageVec<T> + StorageVecRwLock<T> + ?Sized,
{
    pub(super) fn new<I>(indices: I, data: &'a V) -> Self
    where
        I: IntoIterator<Item = Index> + 'static,
    {
        Self {
            indices: Box::new(indices.into_iter()),
            write_lock: data.write_lock(),
            phantom_t: Default::default(),
            phantom_d: Default::default(),
        }
    }
}

// LendingIterator trait gives us all the nice iterator type functions.
// We only have to impl next()
#[allow(private_bounds)]
#[gat]
impl<'a, V, T: 'a> LendingIterator for ManyIterMut<'a, V, T>
where
    V: StorageVec<T> + StorageVecRwLock<T> + ?Sized,
    V::LockedData: StorageVecLockedData<T>,
{
    type Item<'b> = StorageSetter<'a, 'b, V, T>
    where
        Self: 'b;

    fn next(&mut self) -> Option<Self::Item<'_>> {
        if let Some(i) = Iterator::next(&mut self.indices) {
            let value = self.write_lock.get(i);
            Some(StorageSetter {
                phantom: Default::default(),
                write_lock: &mut self.write_lock,
                index: i,
                value,
            })
        } else {
            None
        }
    }
}

/// used for accessing and setting values returned from StorageVec::get_mut() and mutable iterators
#[allow(private_bounds)]
pub struct StorageSetter<'c, 'd, V, T>
where
    V: StorageVec<T> + StorageVecRwLock<T> + ?Sized,
{
    phantom: PhantomData<V>,
    write_lock: &'d mut RwLockWriteGuard<'c, V::LockedData>,
    index: Index,
    value: T,
}

#[allow(private_bounds)]
impl<'a, 'b, V, T> StorageSetter<'a, 'b, V, T>
where
    V: StorageVec<T> + StorageVecRwLock<T> + ?Sized,
    V::LockedData: StorageVecLockedData<T>,
{
    pub fn set(&mut self, value: T) {
        self.write_lock.set(self.index, value);
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}
