use std::sync::Arc;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

use super::{
    dbtsingleton_private::DbtSingletonPrivate, traits::*, RustyKey, RustyValue, WriteOperation,
};
use serde::{de::DeserializeOwned, Serialize};

/// Singleton type created by [`super::DbtSchema`]
///
/// This type is concurrency-safe.  A single RwLock is employed
/// for all read and write ops.  Callers do not need to perform
/// any additional locking.
///
/// Also because the locking is fully encapsulated within DbtSingletonNoLock
/// there is no possibility of a caller holding a lock too long
/// by accident or encountering ordering deadlock issues.
///
/// `DbtSingletonNoLock` is a NewType around Arc<RwLock<..>>.  Thus it
/// can be cheaply cloned to create a reference as if it were an
/// Arc.
#[derive(Debug)]
pub struct DbtSingletonNoLock<V> {
    inner: Rc<RefCell<DbtSingletonPrivate<V>>>,
}

// We manually impl Clone so that callers can make reference clones.
impl<V> Clone for DbtSingletonNoLock<V> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<V> DbtSingletonNoLock<V>
where
    V: Default,
{
    // DbtSingletonNoLock can not be instantiated directly outside of this crate.
    #[inline]
    pub(crate) fn new(key: RustyKey, reader: Arc<dyn StorageReader + Sync + Send>) -> Self {
        let singleton = DbtSingletonPrivate::<V> {
            current_value: Default::default(),
            old_value: Default::default(),
            key,
            reader,
        };
        Self {
            inner: Rc::new(RefCell::new(singleton)),
        }
    }
}

impl<V> StorageSingleton<V> for DbtSingletonNoLock<V>
where
    V: Clone + From<V> + Default,
    V: DeserializeOwned,
{
    #[inline]
    fn get(&self) -> V {
        self.inner.borrow().get()
    }

    #[inline]
    fn set(&self, t: V) {
        self.inner.borrow_mut().set(t);
    }
}

impl<V> DbTable for DbtSingletonNoLock<V>
where
    V: Eq + Clone + Default + Debug,
    V: Serialize + DeserializeOwned,
{
    #[inline]
    fn pull_queue(&self) -> Vec<WriteOperation> {
        let inner = self.inner.borrow();
        if inner.current_value == inner.old_value {
            vec![]
        } else {
            self.inner.borrow_mut().old_value = inner.current_value.clone();
            vec![WriteOperation::Write(
                inner.key.clone(),
                RustyValue::from_any(&inner.current_value),
            )]
        }
    }

    #[inline]
    fn restore_or_new(&self) {
        let inner = self.inner.borrow();
        self.inner.borrow_mut().current_value = match inner.reader.get(inner.key.clone()) {
            Some(value) => value.into_any(),
            None => V::default(),
        }
    }
}
