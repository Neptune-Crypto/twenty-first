use super::traits::Atomic;
use std::sync::{Arc, RwLock};

/// An `Arc<RwLock<T>>` wrapper to make data thread-safe and easy to work with.
///
/// # Example
/// ```
/// # use twenty_first::sync::{AtomicRw, traits::*};
/// struct Car {
///     year: u16,
/// };
/// let atomic_car = AtomicRw::from(Car{year: 2016});
/// atomic_car.with(|c| println!("year: {}", c.year));
/// atomic_car.with_mut(|mut c| c.year = 2023);
/// ```
#[derive(Debug, Default, Clone)]
pub struct AtomicRw<T>(Arc<RwLock<T>>);
impl<T> From<T> for AtomicRw<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self(Arc::new(RwLock::new(t)))
    }
}

impl<T> From<RwLock<T>> for AtomicRw<T> {
    #[inline]
    fn from(t: RwLock<T>) -> Self {
        Self(Arc::new(t))
    }
}

impl<T> TryFrom<AtomicRw<T>> for RwLock<T> {
    type Error = Arc<RwLock<T>>;
    fn try_from(t: AtomicRw<T>) -> Result<RwLock<T>, Self::Error> {
        Arc::<RwLock<T>>::try_unwrap(t.0)
    }
}

impl<T> From<Arc<RwLock<T>>> for AtomicRw<T> {
    #[inline]
    fn from(t: Arc<RwLock<T>>) -> Self {
        Self(t)
    }
}

impl<T> From<AtomicRw<T>> for Arc<RwLock<T>> {
    #[inline]
    fn from(t: AtomicRw<T>) -> Self {
        t.0
    }
}

// note: we impl the Atomic trait methods here also so they
// can be used without caller having to use the trait.
impl<T> AtomicRw<T> {
    /// Immutably access the data of type `T` in a closure and possibly return a result of type `R`
    ///
    /// # Examples
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// atomic_car.with(|c| println!("year: {}", c.year));
    /// let year = atomic_car.with(|c| c.year);
    /// ```
    pub fn with<R, F>(&self, mut f: F) -> R
    where
        F: FnMut(&T) -> R,
    {
        let mut lock = self.0.write().expect("Write lock should succeed");
        f(&mut lock)
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
    /// atomic_car.with_mut(|mut c| {c.year = 2022});
    /// let year = atomic_car.with_mut(|mut c| {c.year = 2023; c.year});
    /// ```
    pub fn with_mut<R, F>(&self, mut f: F) -> R
    where
        F: FnMut(&mut T) -> R,
    {
        let mut lock = self.0.write().expect("Write lock should succeed");
        f(&mut lock)
    }
}

impl<T> Atomic<T> for AtomicRw<T> {
    #[inline]
    fn with<R, F>(&self, f: F) -> R
    where
        F: FnMut(&T) -> R,
    {
        AtomicRw::<T>::with(self, f)
    }

    #[inline]
    fn with_mut<R, F>(&self, f: F) -> R
    where
        F: FnMut(&mut T) -> R,
    {
        AtomicRw::<T>::with_mut(self, f)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    // Verify (compile-time) that AtomicRw::with() and ::with_mut() accept mutable values.  (FnMut)
    fn mutable_assignment() {
        let name = "Jim".to_string();
        let atomic_name = AtomicRw::from(name);

        let mut new_name: String = Default::default();
        atomic_name.with(|n| new_name = n.to_string());
        atomic_name.with_mut(|n| new_name = n.to_string());
    }
}
