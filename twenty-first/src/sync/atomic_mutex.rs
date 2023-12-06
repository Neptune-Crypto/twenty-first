use super::traits::Atomic;
use std::sync::{Arc, Mutex};

/// An Arc<Mutex<T>> wrapper to make data thread-safe and easy to work with.
///
/// # Example
/// ```
/// # use twenty_first::sync::{AtomicMutex, traits::*};
/// struct Car {
///     year: u16,
/// };
/// let atomic_car = AtomicMutex::from(Car{year: 2016});
/// atomic_car.with(|c| println!("year: {}", c.year));
/// atomic_car.with_mut(|mut c| c.year = 2023);
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
    /// Immutably access the data of type `T` in a closure
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// atomic_car.with(|c| println!("year: {}", c.year));
    /// ```
    #[inline]
    pub fn with<F>(&self, f: F)
    where
        F: Fn(&T),
    {
        let lock = self.0.lock().expect("Mutex lock should succeed");
        f(&lock)
    }

    /// Mutably access the data of type `T` in a closure
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// atomic_car.with_mut(|mut c| c.year = 2023);
    /// ```
    #[inline]
    pub fn with_mut<F>(&self, f: F)
    where
        F: Fn(&mut T),
    {
        let mut lock = self.0.lock().expect("Mutex lock should succeed");
        f(&mut lock)
    }
}

impl<T> Atomic<T> for AtomicMutex<T> {
    #[inline]
    fn with<F>(&self, f: F)
    where
        F: Fn(&T),
    {
        AtomicMutex::<T>::with(self, f)
    }

    #[inline]
    fn with_mut<F>(&self, f: F)
    where
        F: Fn(&mut T),
    {
        AtomicMutex::<T>::with_mut(self, f)
    }
}
