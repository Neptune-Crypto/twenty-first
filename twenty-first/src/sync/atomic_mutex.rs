use super::traits::Atomic;
use std::sync::{Arc, Mutex, MutexGuard};

type AcquiredCallbackFn = fn(is_mut: bool, name: Option<&str>);

/// An `Arc<Mutex<T>>` wrapper to make data thread-safe and easy to work with.
///
/// # Example
/// ```
/// # use twenty_first::sync::{AtomicMutex, traits::*};
/// struct Car {
///     year: u16,
/// };
/// let atomic_car = AtomicMutex::from(Car{year: 2016});
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
/// # use twenty_first::sync::AtomicMutex;
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
/// let atomic_car = AtomicMutex::<Car>::from((Car{year: 2016}, Some("car"), Some(LOG_LOCK_ACQUIRED_CB)));
/// atomic_car.lock(|c| {println!("year: {}", c.year)});
/// atomic_car.lock_mut(|mut c| {c.year = 2023});
/// ```
///
/// results in:
#[derive(Debug, Default)]
pub struct AtomicMutex<T> {
    inner: Arc<Mutex<T>>,
    name: Option<String>,
    acquired_callback: Option<AcquiredCallbackFn>,
}
impl<T> From<T> for AtomicMutex<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(t)),
            name: None,
            acquired_callback: None,
        }
    }
}
impl<T> From<(T, Option<String>, Option<AcquiredCallbackFn>)> for AtomicMutex<T> {
    /// Create from an optional name and an optional callback function, which
    /// is called when a lock is acquired.
    #[inline]
    fn from(v: (T, Option<String>, Option<AcquiredCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(Mutex::new(v.0)),
            name: v.1,
            acquired_callback: v.2,
        }
    }
}
impl<T> From<(T, Option<&str>, Option<AcquiredCallbackFn>)> for AtomicMutex<T> {
    /// Create from a name ref and an optional callback function, which
    /// is called when a lock is acquired.
    #[inline]
    fn from(v: (T, Option<&str>, Option<AcquiredCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(Mutex::new(v.0)),
            name: v.1.map(|s| s.to_owned()),
            acquired_callback: v.2,
        }
    }
}

impl<T> Clone for AtomicMutex<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            inner: self.inner.clone(),
            acquired_callback: None,
        }
    }
}

impl<T> From<Mutex<T>> for AtomicMutex<T> {
    #[inline]
    fn from(t: Mutex<T>) -> Self {
        Self {
            name: None,
            inner: Arc::new(t),
            acquired_callback: None,
        }
    }
}
impl<T> From<(Mutex<T>, Option<String>, Option<AcquiredCallbackFn>)> for AtomicMutex<T> {
    /// Create from an Mutex<T> plus an optional name
    /// and an optional callback function, which is called
    /// when a lock is acquired.
    #[inline]
    fn from(v: (Mutex<T>, Option<String>, Option<AcquiredCallbackFn>)) -> Self {
        Self {
            inner: Arc::new(v.0),
            name: v.1,
            acquired_callback: v.2,
        }
    }
}

impl<T> TryFrom<AtomicMutex<T>> for Mutex<T> {
    type Error = Arc<Mutex<T>>;
    fn try_from(t: AtomicMutex<T>) -> Result<Mutex<T>, Self::Error> {
        Arc::<Mutex<T>>::try_unwrap(t.inner)
    }
}

impl<T> From<Arc<Mutex<T>>> for AtomicMutex<T> {
    #[inline]
    fn from(t: Arc<Mutex<T>>) -> Self {
        Self {
            name: None,
            inner: t,
            acquired_callback: None,
        }
    }
}
impl<T> From<(Arc<Mutex<T>>, Option<String>, Option<AcquiredCallbackFn>)> for AtomicMutex<T> {
    /// Create from an `Arc<Mutex<T>>` plus an optional name and
    /// an optional callback function, which is called when a lock
    /// is acquired.
    #[inline]
    fn from(v: (Arc<Mutex<T>>, Option<String>, Option<AcquiredCallbackFn>)) -> Self {
        Self {
            inner: v.0,
            name: v.1,
            acquired_callback: v.2,
        }
    }
}

impl<T> From<AtomicMutex<T>> for Arc<Mutex<T>> {
    #[inline]
    fn from(t: AtomicMutex<T>) -> Self {
        t.inner
    }
}

// note: we impl the Atomic trait methods here also so they
// can be used without caller having to use the trait.
impl<T> AtomicMutex<T> {
    /// Acquire read lock and return an `MutexGuard`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// let year = atomic_car.lock_guard().year;
    /// ```
    pub fn lock_guard(&self) -> MutexGuard<T> {
        let guard = self.inner.lock().expect("Read lock should succeed");
        if let Some(cb) = self.acquired_callback {
            cb(false, self.name.as_deref());
        }
        guard
    }

    /// Acquire write lock and return an `MutexGuard`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// atomic_car.lock_guard_mut().year = 2022;
    /// ```
    pub fn lock_guard_mut(&self) -> MutexGuard<T> {
        let guard = self.inner.lock().expect("Write lock should succeed");
        if let Some(cb) = self.acquired_callback {
            cb(false, self.name.as_deref());
        }
        guard
    }

    /// Immutably access the data of type `T` in a closure and possibly return a result of type `R`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// atomic_car.lock(|c| println!("year: {}", c.year));
    /// let year = atomic_car.lock(|c| c.year);
    /// ```
    pub fn lock<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        let lock = self.inner.lock().expect("Read lock should succeed");
        if let Some(cb) = self.acquired_callback {
            cb(false, self.name.as_deref());
        }
        f(&lock)
    }

    /// Mutably access the data of type `T` in a closure and possibly return a result of type `R`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// atomic_car.lock_mut(|mut c| {c.year = 2022});
    /// let year = atomic_car.lock_mut(|mut c| {c.year = 2023; c.year});
    /// ```
    pub fn lock_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        let mut lock = self.inner.lock().expect("Write lock should succeed");
        if let Some(cb) = self.acquired_callback {
            cb(false, self.name.as_deref());
        }
        f(&mut lock)
    }

    /// get copy of the locked value T (if T implements Copy).
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// let atomic_u64 = AtomicMutex::from(25u64);
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
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// let atomic_bool = AtomicMutex::from(false);
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

impl<T> Atomic<T> for AtomicMutex<T> {
    #[inline]
    fn lock<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        AtomicMutex::<T>::lock(self, f)
    }

    #[inline]
    fn lock_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        AtomicMutex::<T>::lock_mut(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Verify (compile-time) that AtomicMutex::lock() and ::lock_mut() accept mutable values.  (FnMut)
    fn mutable_assignment() {
        let name = "Jim".to_string();
        let atomic_name = AtomicMutex::from(name);

        let mut new_name: String = Default::default();
        atomic_name.lock(|n| new_name = n.to_string());
        atomic_name.lock_mut(|n| new_name = n.to_string());
    }
}
