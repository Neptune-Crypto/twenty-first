use super::traits::*;
use std::sync::Arc;

// note: no locking is required in `DbtSingletonPrivate` because locking
// is performed in the `DbtSingleton` public wrapper.
pub(crate) struct DbtSingletonPrivate<ParentKey, ParentValue, T> {
    pub(super) current_value: T,
    pub(super) old_value: T,
    pub(super) key: ParentKey,
    pub(super) reader: Arc<dyn StorageReader<ParentKey, ParentValue> + Sync + Send>,
}

impl<ParentKey, ParentValue, T> StorageSingletonReads<T>
    for DbtSingletonPrivate<ParentKey, ParentValue, T>
where
    T: Clone + From<ParentValue>,
{
    fn get(&self) -> T {
        self.current_value.clone()
    }
}

impl<ParentKey, ParentValue, T> StorageSingletonMutableWrites<T>
    for DbtSingletonPrivate<ParentKey, ParentValue, T>
where
    T: Clone + From<ParentValue>,
{
    fn set(&mut self, t: T) {
        self.current_value = t;
    }
}
