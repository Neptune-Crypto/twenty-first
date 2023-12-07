use super::traits::*;
use super::RustyKey;
use std::sync::Arc;

// note: no locking is required in `DbtSingletonPrivate` because locking
// is performed in the `DbtSingleton` public wrapper.
pub(crate) struct DbtSingletonPrivate<V> {
    pub(super) current_value: V,
    pub(super) old_value: V,
    pub(super) key: RustyKey,
    pub(super) reader: Arc<dyn StorageReader + Sync + Send>,
}

impl<V> StorageSingletonReads<V> for DbtSingletonPrivate<V>
where
    V: Clone + From<V>,
{
    fn get(&self) -> V {
        self.current_value.clone()
    }
}

impl<V> StorageSingletonMutableWrites<V> for DbtSingletonPrivate<V>
where
    V: Clone + From<V>,
{
    fn set(&mut self, v: V) {
        self.current_value = v;
    }
}
