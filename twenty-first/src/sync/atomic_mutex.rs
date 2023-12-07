use super::traits::Atomic;
use std::sync::{Arc, Mutex};

/// An `Arc<Mutex<T>>` wrapper to make data thread-safe and easy to work with.
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
    /// Immutably access the data of type `T` in a closure and return a result
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicMutex, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicMutex::from(Car{year: 2016});
    /// atomic_car.with(|c| println!("year: {}", c.year));
    /// let year = atomic_car.with(|c| c.year);
    /// ```
    pub fn with<R, F>(&self, mut f: F) -> R
    where
        F: FnMut(&T) -> R,
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
    /// atomic_car.with_mut(|mut c| c.year = 2022);
    /// let year = atomic_car.with_mut(|mut c| {c.year = 2023; c.year});
    /// ```
    pub fn with_mut<R, F>(&self, mut f: F) -> R
    where
        F: FnMut(&mut T) -> R,
    {
        let mut lock = self.0.lock().expect("Write lock should succeed");
        f(&mut lock)
    }
}

impl<T> Atomic<T> for AtomicMutex<T> {
    fn with<R, F>(&self, f: F) -> R
    where
        F: FnMut(&T) -> R,
    {
        AtomicMutex::<T>::with(self, f)
    }

    fn with_mut<R, F>(&self, f: F) -> R
    where
        F: FnMut(&mut T) -> R,
    {
        AtomicMutex::<T>::with_mut(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Verify (compile-time) that AtomicRw::with() and ::with_mut() accept mutable values.  (FnMut)
    fn mutable_assignment() {
        let name = "Jim".to_string();
        let atomic_name = AtomicMutex::from(name);

        let mut new_name: String = Default::default();
        atomic_name.with(|n| new_name = n.to_string());
        atomic_name.with_mut(|n| new_name = n.to_string());
    }
}
