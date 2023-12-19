use super::traits::Atomic;
use std::sync::{Arc, Mutex, MutexGuard};

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
#[derive(Debug, Default, Clone)]
pub struct AtomicMutex<T>(Arc<Mutex<T>>);
impl<T> From<T> for AtomicMutex<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self(Arc::new(Mutex::new(t)))
    }
}

impl<T> From<Mutex<T>> for AtomicMutex<T> {
    #[inline]
    fn from(t: Mutex<T>) -> Self {
        Self(Arc::new(t))
    }
}

impl<T> TryFrom<AtomicMutex<T>> for Mutex<T> {
    type Error = Arc<Mutex<T>>;

    #[inline]
    fn try_from(t: AtomicMutex<T>) -> Result<Mutex<T>, Self::Error> {
        Arc::<Mutex<T>>::try_unwrap(t.0)
    }
}

impl<T> From<Arc<Mutex<T>>> for AtomicMutex<T> {
    #[inline]
    fn from(t: Arc<Mutex<T>>) -> Self {
        Self(t)
    }
}

impl<T> From<AtomicMutex<T>> for Arc<Mutex<T>> {
    #[inline]
    fn from(t: AtomicMutex<T>) -> Self {
        t.0
    }
}

// note: we impl the Atomic trait methods here also so they
// can be used without caller having to use the trait.
impl<T> AtomicMutex<T> {
    /// Acquire lock and return a `MutexGuard`
    ///
    /// note: this method is exactly the same as [`lock_guard_mut()`](Self::lock_guard_mut).
    /// It exists only for compatibility with [`AtomicRw`](super::AtomicRw) so
    /// they can be used interchangeably.
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// atomic_car.lock_guard().year = 2022;
    /// ```
    pub fn lock_guard(&self) -> MutexGuard<T> {
        self.0.lock().expect("Mutex lock should succeed")
    }

    /// Acquire lock and return a `MutexGuard`
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
        self.0.lock().expect("Mutex lock should succeed")
    }

    /// Immutably access the data of type `T` in a closure and return a result
    ///
    /// # Example
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
        let mut lock = self.0.lock().expect("Write lock should succeed");
        f(&mut lock)
    }

    /// Mutably access the data of type `T` in a closure and return a result
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// atomic_car.lock_mut(|mut c| c.year = 2022);
    /// let year = atomic_car.lock_mut(|mut c| {c.year = 2023; c.year});
    /// ```
    pub fn lock_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        let mut lock = self.0.lock().expect("Write lock should succeed");
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
}

impl<T> Atomic<T> for AtomicMutex<T> {
    fn lock<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        AtomicMutex::<T>::lock(self, f)
    }

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
    // Verify (compile-time) that AtomicRw::lock() and ::lock_mut() accept mutable values.  (FnOnce)
    fn mutable_assignment() {
        let name = "Jim".to_string();
        let atomic_name = AtomicMutex::from(name);

        let mut new_name: String = Default::default();
        atomic_name.lock(|n| new_name = n.to_string());
        atomic_name.lock_mut(|n| new_name = n.to_string());
    }
}
