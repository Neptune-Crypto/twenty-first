use super::traits::Atomic;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

type AcquiredCallbackFn = fn(is_mut: bool, name: Option<&str>);

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
/// # use twenty_first::sync::AtomicRw;
/// struct Car {
///     year: u16,
/// };
///
/// pub fn log_lock_acquired(is_mut: bool, name: Option<&str>) {
///     println!(
///         "thread {{name: `{}`, id: {:?}}} acquired lock `{}` for {}",
///         std::thread::current().name().unwrap_or("?"),
///         std::thread::current().id(),
///         name.unwrap_or("?"),
///         if is_mut { "write" } else { "read" }
///     );
/// }
/// const LOG_LOCK_ACQUIRED_CB: fn(is_mut: bool, name: Option<&str>) = log_lock_acquired;
///
/// let atomic_car = AtomicRw::<Car>::from((Car{year: 2016}, Some("car"), Some(LOG_LOCK_ACQUIRED_CB)));
/// atomic_car.lock(|c| {println!("year: {}", c.year)});
/// atomic_car.lock_mut(|mut c| {c.year = 2023});
/// ```
///
/// results in:
#[derive(Debug, Default)]
pub struct AtomicRw<T> {
    inner: Arc<RwLock<T>>,
    name: Option<String>,
    acquired_callback: Option<AcquiredCallbackFn>,
}
impl<T> From<T> for AtomicRw<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(t)),
            name: None,
            acquired_callback: None,
        }
    }
}
impl<T> From<(T, Option<String>, Option<AcquiredCallbackFn>)> for AtomicRw<T> {
    /// Create from an optional name and an optional callback function, which
    /// is called when a lock is acquired.
    #[inline]
    fn from(v: (T, Option<String>, Option<AcquiredCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(RwLock::new(v.0)),
            name: v.1,
            acquired_callback: v.2,
        }
    }
}
impl<T> From<(T, Option<&str>, Option<AcquiredCallbackFn>)> for AtomicRw<T> {
    /// Create from a name ref and an optional callback function, which
    /// is called when a lock is acquired.
    #[inline]
    fn from(v: (T, Option<&str>, Option<AcquiredCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(RwLock::new(v.0)),
            name: v.1.map(|s| s.to_owned()),
            acquired_callback: v.2,
        }
    }
}

impl<T> Clone for AtomicRw<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            inner: self.inner.clone(),
            acquired_callback: None,
        }
    }
}

impl<T> From<RwLock<T>> for AtomicRw<T> {
    #[inline]
    fn from(t: RwLock<T>) -> Self {
        Self {
            name: None,
            inner: Arc::new(t),
            acquired_callback: None,
        }
    }
}
impl<T> From<(RwLock<T>, Option<String>, Option<AcquiredCallbackFn>)> for AtomicRw<T> {
    /// Create from an RwLock<T> plus an optional name
    /// and an optional callback function, which is called
    /// when a lock is acquired.
    #[inline]
    fn from(v: (RwLock<T>, Option<String>, Option<AcquiredCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(v.0),
            name: v.1,
            acquired_callback: v.2,
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
            name: None,
            inner: t,
            acquired_callback: None,
        }
    }
}
impl<T> From<(Arc<RwLock<T>>, Option<String>, Option<AcquiredCallbackFn>)> for AtomicRw<T> {
    /// Create from an `Arc<RwLock<T>>` plus an optional name and
    /// an optional callback function, which is called when a lock
    /// is acquired.
    #[inline]
    fn from(v: (Arc<RwLock<T>>, Option<String>, Option<AcquiredCallbackFn>)) -> Self {
        Self {
            inner: v.0,
            name: v.1,
            acquired_callback: v.2,
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
    pub fn lock_guard(&self) -> RwLockReadGuard<T> {
        let guard = self.inner.read().expect("Read lock should succeed");
        if let Some(cb) = self.acquired_callback {
            cb(false, self.name.as_deref());
        }
        guard
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
    pub fn lock_guard_mut(&self) -> RwLockWriteGuard<T> {
        let guard = self.inner.write().expect("Write lock should succeed");
        if let Some(cb) = self.acquired_callback {
            cb(false, self.name.as_deref());
        }
        guard
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
        let lock = self.inner.read().expect("Read lock should succeed");
        if let Some(cb) = self.acquired_callback {
            cb(false, self.name.as_deref());
        }
        f(&lock)
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
        let mut lock = self.inner.write().expect("Write lock should succeed");
        if let Some(cb) = self.acquired_callback {
            cb(false, self.name.as_deref());
        }
        f(&mut lock)
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
        self.name.as_deref()
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
