use super::shared::LockAcquisition;
use super::traits::Atomic;
use super::{LockCallbackFn, LockCallbackInfo, LockEvent, LockType};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// An `Arc<RwLock<T>>` wrapper to make data thread-safe and easy to work with.
///
/// # Example
/// ```
/// # use twenty_first::sync::{AtomicRw, traits::*};
/// struct Car {
///     year: u16,
/// };
/// let atomic_car = AtomicRw::from(Car{year: 2016});
/// atomic_car.lock(|c| println!("year: {}", c.year));
/// atomic_car.lock_mut(|mut c| c.year = 2023);
/// ```
///
/// It is also possible to provide a name and callback fn
/// during instantiation.  In this way, the application
/// can easily trace lock acquisitions.
///
/// # Examples
/// ```
/// # use twenty_first::sync::{AtomicRw, LockEvent, LockCallbackFn};
/// struct Car {
///     year: u16,
/// };
///
/// pub fn log_lock_event(lock_event: LockEvent) {
///     match lock_event {
///         LockEvent::Acquire{info, acquired} =>
///             println!(
///                 "thread {{name: `{}`, id: {:?}}} acquired `{}` lock `{}` for {}",
///                 std::thread::current().name().unwrap_or("?"),
///                 std::thread::current().id(),
///                 info.lock_type(),
///                 info.name().unwrap_or("?"),
///                 acquired,
///             ),
///         LockEvent::Release{info, acquired} =>
///             println!(
///                 "thread {{name: `{}`, id: {:?}}} released `{}` lock `{}` for {}",
///                 std::thread::current().name().unwrap_or("?"),
///                 std::thread::current().id(),
///                 info.lock_type(),
///                 info.name().unwrap_or("?"),
///                 acquired,
///             ),
///     }
///
/// }
/// const LOG_LOCK_EVENT_CB: LockCallbackFn = log_lock_event;
///
/// let atomic_car = AtomicRw::<Car>::from((Car{year: 2016}, Some("car"), Some(LOG_LOCK_EVENT_CB)));
/// atomic_car.lock(|c| {println!("year: {}", c.year)});
/// atomic_car.lock_mut(|mut c| {c.year = 2023});
/// ```
///
/// results in:
/// ```text
/// thread {name: `main`, id: ThreadId(1)} acquired `RwLock` lock `car` for Read
/// year: 2016
/// thread {name: `main`, id: ThreadId(1)} released `RwLock` lock `car` for Read
/// thread {name: `main`, id: ThreadId(1)} acquired `RwLock` lock `car` for Write
/// thread {name: `main`, id: ThreadId(1)} released `RwLock` lock `car` for Write
/// ```
#[derive(Debug)]
pub struct AtomicRw<T> {
    inner: Arc<RwLock<T>>,
    lock_callback_info: LockCallbackInfo,
}

impl<T: Default> Default for AtomicRw<T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
            lock_callback_info: LockCallbackInfo::new(LockType::RwLock, None, None),
        }
    }
}

impl<T> From<T> for AtomicRw<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(t)),
            lock_callback_info: LockCallbackInfo::new(LockType::RwLock, None, None),
        }
    }
}
impl<T> From<(T, Option<String>, Option<LockCallbackFn>)> for AtomicRw<T> {
    /// Create from an optional name and an optional callback function, which
    /// is called when a lock is acquired.
    #[inline]
    fn from(v: (T, Option<String>, Option<LockCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(RwLock::new(v.0)),
            lock_callback_info: LockCallbackInfo::new(LockType::RwLock, v.1, v.2),
        }
    }
}
impl<T> From<(T, Option<&str>, Option<LockCallbackFn>)> for AtomicRw<T> {
    /// Create from a name ref and an optional callback function, which
    /// is called when a lock is acquired.
    #[inline]
    fn from(v: (T, Option<&str>, Option<LockCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(RwLock::new(v.0)),
            lock_callback_info: LockCallbackInfo::new(
                LockType::RwLock,
                v.1.map(|s| s.to_owned()),
                v.2,
            ),
        }
    }
}

impl<T> Clone for AtomicRw<T> {
    fn clone(&self) -> Self {
        Self {
            lock_callback_info: self.lock_callback_info.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl<T> From<RwLock<T>> for AtomicRw<T> {
    #[inline]
    fn from(t: RwLock<T>) -> Self {
        Self {
            inner: Arc::new(t),
            lock_callback_info: LockCallbackInfo::new(LockType::RwLock, None, None),
        }
    }
}
impl<T> From<(RwLock<T>, Option<String>, Option<LockCallbackFn>)> for AtomicRw<T> {
    /// Create from an `RwLock<T>` plus an optional name
    /// and an optional callback function, which is called
    /// when a lock is acquired.
    #[inline]
    fn from(v: (RwLock<T>, Option<String>, Option<LockCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(v.0),
            lock_callback_info: LockCallbackInfo::new(LockType::RwLock, v.1, v.2),
        }
    }
}

impl<T> TryFrom<AtomicRw<T>> for RwLock<T> {
    type Error = Arc<RwLock<T>>;
    fn try_from(t: AtomicRw<T>) -> Result<RwLock<T>, Self::Error> {
        Arc::<RwLock<T>>::try_unwrap(t.inner)
    }
}

impl<T> From<Arc<RwLock<T>>> for AtomicRw<T> {
    #[inline]
    fn from(t: Arc<RwLock<T>>) -> Self {
        Self {
            inner: t,
            lock_callback_info: LockCallbackInfo::new(LockType::RwLock, None, None),
        }
    }
}
impl<T> From<(Arc<RwLock<T>>, Option<String>, Option<LockCallbackFn>)> for AtomicRw<T> {
    /// Create from an `Arc<RwLock<T>>` plus an optional name and
    /// an optional callback function, which is called when a lock
    /// is acquired.
    #[inline]
    fn from(v: (Arc<RwLock<T>>, Option<String>, Option<LockCallbackFn>)) -> Self {
        Self {
            inner: v.0,
            lock_callback_info: LockCallbackInfo::new(LockType::RwLock, v.1, v.2),
        }
    }
}

impl<T> From<AtomicRw<T>> for Arc<RwLock<T>> {
    #[inline]
    fn from(t: AtomicRw<T>) -> Self {
        t.inner
    }
}

// note: we impl the Atomic trait methods here also so they
// can be used without caller having to use the trait.
impl<T> AtomicRw<T> {
    /// Acquire read lock and return an `RwLockReadGuard`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// let year = atomic_car.lock_guard().year;
    /// ```
    pub fn lock_guard(&self) -> AtomicRwReadGuard<T> {
        let guard = self.inner.read().expect("Read lock should succeed");
        AtomicRwReadGuard::new(guard, &self.lock_callback_info)
    }

    /// Acquire write lock and return an `RwLockWriteGuard`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// atomic_car.lock_guard_mut().year = 2022;
    /// ```
    pub fn lock_guard_mut(&self) -> AtomicRwWriteGuard<T> {
        let guard = self.inner.write().expect("Write lock should succeed");
        AtomicRwWriteGuard::new(guard, &self.lock_callback_info)
    }

    /// Immutably access the data of type `T` in a closure and possibly return a result of type `R`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// atomic_car.lock(|c| println!("year: {}", c.year));
    /// let year = atomic_car.lock(|c| c.year);
    /// ```
    pub fn lock<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        let guard = self.inner.read().expect("Read lock should succeed");
        let my_guard = AtomicRwReadGuard::new(guard, &self.lock_callback_info);
        f(&my_guard)
    }

    /// Mutably access the data of type `T` in a closure and possibly return a result of type `R`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// atomic_car.lock_mut(|mut c| {c.year = 2022});
    /// let year = atomic_car.lock_mut(|mut c| {c.year = 2023; c.year});
    /// ```
    pub fn lock_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        let guard = self.inner.write().expect("Write lock should succeed");
        let mut my_guard = AtomicRwWriteGuard::new(guard, &self.lock_callback_info);
        f(&mut my_guard)
    }

    /// get copy of the locked value T (if T implements Copy).
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// let atomic_u64 = AtomicRw::from(25u64);
    /// let age = atomic_u64.get();
    /// ```
    #[inline]
    pub fn get(&self) -> T
    where
        T: Copy,
    {
        self.lock(|v| *v)
    }

    /// set the locked value T (if T implements Copy).
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// let atomic_bool = AtomicRw::from(false);
    /// atomic_bool.set(true);
    /// ```
    #[inline]
    pub fn set(&self, value: T)
    where
        T: Copy,
    {
        self.lock_mut(|v| *v = value)
    }

    /// retrieve lock name if present, or None
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.lock_callback_info.lock_info_owned.name.as_deref()
    }
}

impl<T> Atomic<T> for AtomicRw<T> {
    #[inline]
    fn lock<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        AtomicRw::<T>::lock(self, f)
    }

    #[inline]
    fn lock_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        AtomicRw::<T>::lock_mut(self, f)
    }
}

/// A wrapper for [RwLockReadGuard](std::sync::RwLockReadGuard) that
/// can optionally call a callback to notify when the
/// lock is acquired or released.
pub struct AtomicRwReadGuard<'a, T> {
    guard: RwLockReadGuard<'a, T>,
    lock_callback_info: &'a LockCallbackInfo,
}

impl<'a, T> AtomicRwReadGuard<'a, T> {
    fn new(guard: RwLockReadGuard<'a, T>, lock_callback_info: &'a LockCallbackInfo) -> Self {
        if let Some(cb) = lock_callback_info.lock_callback_fn {
            cb(LockEvent::Acquire {
                info: lock_callback_info.lock_info_owned.as_lock_info(),
                acquired: LockAcquisition::Read,
            });
        }
        Self {
            guard,
            lock_callback_info,
        }
    }
}

impl<'a, T> Drop for AtomicRwReadGuard<'a, T> {
    fn drop(&mut self) {
        let lock_callback_info = self.lock_callback_info;
        if let Some(cb) = lock_callback_info.lock_callback_fn {
            cb(LockEvent::Release {
                info: lock_callback_info.lock_info_owned.as_lock_info(),
                acquired: LockAcquisition::Read,
            });
        }
    }
}

impl<'a, T> Deref for AtomicRwReadGuard<'a, T> {
    type Target = RwLockReadGuard<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a, T> DerefMut for AtomicRwReadGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

/// A wrapper for [RwLockWriteGuard](std::sync::RwLockWriteGuard) that
/// can optionally call a callback to notify when the
/// lock is acquired or released.
pub struct AtomicRwWriteGuard<'a, T> {
    guard: RwLockWriteGuard<'a, T>,
    lock_callback_info: &'a LockCallbackInfo,
}

impl<'a, T> AtomicRwWriteGuard<'a, T> {
    fn new(guard: RwLockWriteGuard<'a, T>, lock_callback_info: &'a LockCallbackInfo) -> Self {
        if let Some(cb) = lock_callback_info.lock_callback_fn {
            cb(LockEvent::Acquire {
                info: lock_callback_info.lock_info_owned.as_lock_info(),
                acquired: LockAcquisition::Write,
            });
        }
        Self {
            guard,
            lock_callback_info,
        }
    }
}

impl<'a, T> Drop for AtomicRwWriteGuard<'a, T> {
    fn drop(&mut self) {
        let lock_callback_info = self.lock_callback_info;
        if let Some(cb) = lock_callback_info.lock_callback_fn {
            cb(LockEvent::Release {
                info: lock_callback_info.lock_info_owned.as_lock_info(),
                acquired: LockAcquisition::Write,
            });
        }
    }
}

impl<'a, T> Deref for AtomicRwWriteGuard<'a, T> {
    type Target = RwLockWriteGuard<'a, T>;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a, T> DerefMut for AtomicRwWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Verify (compile-time) that AtomicRw::lock() and ::lock_mut() accept mutable values.  (FnMut)
    fn mutable_assignment() {
        let name = "Jim".to_string();
        let atomic_name = AtomicRw::from(name);

        let mut new_name: String = Default::default();
        atomic_name.lock(|n| new_name = n.to_string());
        atomic_name.lock_mut(|n| new_name = n.to_string());
    }
}
