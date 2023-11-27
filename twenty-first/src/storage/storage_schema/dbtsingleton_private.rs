use super::{DbTable, StorageReader, StorageSingleton, WriteOperation};
use std::{fmt::Debug, sync::Arc};

// note: no locking is required in `DbtSingletonPrivate` because locking
// is performed in the `DbtSingleton` public wrapper.
pub(crate) struct DbtSingletonPrivate<ParentKey, ParentValue, T> {
    pub(super) current_value: T,
    pub(super) old_value: T,
    pub(super) key: ParentKey,
    pub(super) reader: Arc<dyn StorageReader<ParentKey, ParentValue> + Sync + Send>,
}

impl<ParentKey, ParentValue, T> StorageSingleton<T>
    for DbtSingletonPrivate<ParentKey, ParentValue, T>
where
    T: Clone + From<ParentValue>,
{
    fn get(&self) -> T {
        self.current_value.clone()
    }

    fn set(&mut self, t: T) {
        self.current_value = t;
    }
}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtSingletonPrivate<ParentKey, ParentValue, T>
where
    T: Eq + Clone + Default + From<ParentValue>,
    ParentValue: From<T> + Debug,
    ParentKey: Clone,
{
    #[inline]
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        if self.current_value == self.old_value {
            vec![]
        } else {
            self.old_value = self.current_value.clone();
            vec![WriteOperation::Write(
                self.key.clone(),
                self.current_value.clone().into(),
            )]
        }
    }

    #[inline]
    fn restore_or_new(&mut self) {
        self.current_value = match self.reader.get(self.key.clone()) {
            Some(value) => value.into(),
            None => T::default(),
        }
    }
}
