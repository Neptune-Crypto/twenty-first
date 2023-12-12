use std::{fmt::Debug, sync::Arc};

use super::{
    dbtsingleton_private::DbtSingletonPrivate, traits::*, RustyKey, RustyValue, WriteOperation,
};
use crate::sync::AtomicRw;
use serde::{de::DeserializeOwned, Serialize};

/// Singleton type created by [`super::DbtSchema`]
///
/// This type is concurrency-safe.  A single RwLock is employed
/// for all read and write ops.  Callers do not need to perform
/// any additional locking.
///
/// Also because the locking is fully encapsulated within DbtSingleton
/// there is no possibility of a caller holding a lock too long
/// by accident or encountering ordering deadlock issues.
///
/// `DbtSingleton` is a NewType around Arc<RwLock<..>>.  Thus it
/// can be cheaply cloned to create a reference as if it were an
/// Arc.
#[derive(Debug)]
pub struct DbtSingleton<V> {
    // note: Arc is not needed, because we never hand out inner to anyone.
    inner: AtomicRw<DbtSingletonPrivate<V>>,
}

// We manually impl Clone so that callers can make reference clones.
impl<V> Clone for DbtSingleton<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<V> DbtSingleton<V>
where
    V: Default,
{
    // DbtSingleton can not be instantiated directly outside of this crate.
    #[inline]
    pub(crate) fn new(key: RustyKey, reader: Arc<dyn StorageReader + Sync + Send>) -> Self {
        let singleton = DbtSingletonPrivate::<V> {
            current_value: Default::default(),
            old_value: Default::default(),
            key,
            reader,
        };
        Self {
            inner: AtomicRw::from(singleton),
        }
    }
}

impl<V> StorageSingleton<V> for DbtSingleton<V>
where
    V: Clone + From<V> + Default,
    V: DeserializeOwned,
{
    #[inline]
    fn get(&self) -> V {
        self.inner.with(|inner| inner.get())
    }

    #[inline]
    fn set(&self, t: V) {
        self.inner.with_mut(|inner| inner.set(t));
    }
}

impl<V> DbTable for DbtSingleton<V>
where
    V: Eq + Clone + Default + Debug,
    V: Serialize + DeserializeOwned,
{
    #[inline]
    fn pull_queue(&self) -> Vec<WriteOperation> {
        self.inner.with_mut(|inner| {
            if inner.current_value == inner.old_value {
                vec![]
            } else {
                inner.old_value = inner.current_value.clone();
                vec![WriteOperation::Write(
                    inner.key.clone(),
                    RustyValue::from_any(&inner.current_value),
                )]
            }
        })
    }

    #[inline]
    fn restore_or_new(&self) {
        self.inner.with_mut(|inner| {
            inner.current_value = match inner.reader.get(inner.key.clone()) {
                Some(value) => value.into_any(),
                None => V::default(),
            }
        });
    }
}
