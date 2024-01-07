use std::sync::Arc;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

use super::{
    dbtsingleton_private::DbtSingletonPrivate, traits::*, RustyKey, RustyValue, WriteOperation,
};
use serde::{de::DeserializeOwned, Serialize};

/// Singleton type created by [`super::DbtSchema`]
///
/// This type is NOT concurrency-safe.
///
/// `DbtSingleton` is a NewType around Rc<RefCell<..>>.  Thus it
/// can be cheaply cloned to create a reference as if it were an
/// Rc.
#[derive(Debug)]
pub struct DbtSingleton<V> {
    inner: Rc<RefCell<DbtSingletonPrivate<V>>>,
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
            inner: Rc::new(RefCell::new(singleton)),
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
        self.inner.borrow().get()
    }

    #[inline]
    fn set(&self, t: V) {
        self.inner.borrow_mut().set(t);
    }
}

impl<V> DbTable for DbtSingleton<V>
where
    V: Eq + Clone + Default + Debug,
    V: Serialize + DeserializeOwned,
{
    #[inline]
    fn pull_queue(&self) -> Vec<WriteOperation> {
        if self.inner.borrow().current_value == self.inner.borrow().old_value {
            vec![]
        } else {
            let mut inner = self.inner.borrow_mut();
            inner.old_value = inner.current_value.clone();
            vec![WriteOperation::Write(
                inner.key.clone(),
                RustyValue::from_any(&inner.current_value),
            )]
        }
    }

    #[inline]
    fn restore_or_new(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.current_value = match inner.reader.get(inner.key.clone()) {
            Some(value) => value.into_any(),
            None => V::default(),
        }
    }
}
