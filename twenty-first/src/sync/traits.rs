//! Traits that define the [`sync`](crate::sync) interface

pub trait Atomic<T> {
    /// Immutably access the data of type `T` in a closure and possibly return a result of type `R`
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// atomic_car.with(|c| {println!("year: {}", c.year); });
    /// let year = atomic_car.with(|c| c.year);
    /// ```
    fn with<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R;

    /// Mutably access the data of type `T` in a closure and possibly return a result of type `R`
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// atomic_car.with_mut(|mut c| {c.year = 2022;});
    /// let year = atomic_car.with_mut(|mut c| {c.year = 2023; c.year});
    /// ```
    fn with_mut<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R;
}
