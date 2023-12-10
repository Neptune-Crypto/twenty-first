use super::traits::*;
use super::RustyKey;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

// note: no locking is required in `DbtSingletonPrivate` because locking
// is performed in the `DbtSingleton` public wrapper.
pub(crate) struct DbtSingletonPrivate<V> {
    pub(super) current_value: V,
    pub(super) old_value: V,
    pub(super) key: RustyKey,
    pub(super) reader: Arc<dyn StorageReader + Sync + Send>,
}

impl<V> Debug for DbtSingletonPrivate<V>
where
    V: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_struct("DbtSingletonPrivate")
            .field("current_value", &self.current_value)
            .field("old_value", &self.old_value)
            .field("key", &self.key)
            .field("reader", &"Arc<dyn StorageReader + Send + Sync>")
            .finish()
    }
}

impl<V: Clone> DbtSingletonPrivate<V> {
    pub(super) fn get(&self) -> V {
        self.current_value.clone()
    }

    pub(super) fn set(&mut self, v: V) {
        self.current_value = v;
    }
}
