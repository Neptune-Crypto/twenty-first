use super::traits::StorageSingleton;
use super::{DbTable, DbtSingleton, WriteOperation};
use std::sync::Arc;
use std::{cell::RefCell, fmt::Debug};

/// Singleton reference type created by [`DbSchema`]
///
/// This type is concurrency-safe. Callers do not need to perform
/// any additional locking.
pub type DbtSingletonReference<PK, PV, T> = Arc<RefCell<DbtSingleton<PK, PV, T>>>;

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtSingletonReference<ParentKey, ParentValue, T>
where
    T: Eq + Clone + Default + From<ParentValue>,
    ParentValue: From<T> + Debug,
    ParentKey: Clone,
{
    #[inline]
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        self.borrow_mut().pull_queue()
    }

    #[inline]
    fn restore_or_new(&mut self) {
        self.borrow_mut().restore_or_new()
    }
}

impl<ParentKey, ParentValue, T> StorageSingleton<T>
    for DbtSingletonReference<ParentKey, ParentValue, T>
where
    T: Clone + From<ParentValue> + Default,
{
    #[inline]
    fn get(&self) -> T {
        self.borrow().get()
    }

    #[inline]
    fn set(&mut self, t: T) {
        self.borrow_mut().set(t)
    }
}
