use super::super::storage_vec::{Index, StorageVec};
use super::{DbTable, DbtVec, WriteOperation};
use std::{cell::RefCell, fmt::Debug, sync::Arc};

pub type DbtVecReference<PK, PV, T> = Arc<RefCell<DbtVec<PK, PV, Index, T>>>;

impl<ParentKey, ParentValue, T> StorageVec<T> for DbtVecReference<ParentKey, ParentValue, T>
where
    ParentKey: From<Index>,
    ParentValue: From<T>,
    T: Clone + From<ParentValue> + Debug,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    Index: From<ParentValue> + From<u64>,
{
    #[inline]
    fn is_empty(&self) -> bool {
        self.borrow().is_empty()
    }

    #[inline]
    fn len(&self) -> Index {
        self.borrow().len()
    }

    #[inline]
    fn get(&self, index: Index) -> T {
        self.borrow().get(index)
    }

    #[inline]
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        self.borrow().get_many(indices)
    }

    #[inline]
    fn get_all(&self) -> Vec<T> {
        self.borrow().get_all()
    }

    #[inline]
    fn set(&mut self, index: Index, value: T) {
        self.borrow_mut().set(index, value)
    }

    #[inline]
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        self.borrow_mut().set_many(key_vals)
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.borrow_mut().pop()
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.borrow_mut().push(value)
    }
}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtVecReference<ParentKey, ParentValue, T>
where
    ParentKey: From<Index>,
    ParentValue: From<T>,
    T: Clone,
    T: From<ParentValue>,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    Index: From<ParentValue>,
    ParentValue: From<Index>,
{
    /// Collect all added elements that have not yet been persisted
    #[inline]
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        self.borrow_mut().pull_queue()
    }

    #[inline]
    fn restore_or_new(&mut self) {
        self.borrow_mut().restore_or_new()
    }
}
