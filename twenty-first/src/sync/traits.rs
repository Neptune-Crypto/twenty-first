pub trait Atomic<T> {
    /// Immutably access the data of type `T` in a closure
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// atomic_car.with(|c| println!("year: {}", c.year));
    /// ```
    fn with<F>(&self, f: F)
    where
        F: Fn(&T);

    /// Mutably access the data of type `T` in a closure
    ///
    /// # Example
    /// ```
    /// # use twenty_first::sync::{AtomicRw, traits::*};
    /// struct Car {
    ///     year: u16,
    /// };
    /// let atomic_car = AtomicRw::from(Car{year: 2016});
    /// atomic_car.with_mut(|mut c| c.year = 2023);
    /// ```
    fn with_mut<F>(&self, f: F)
    where
        F: Fn(&mut T);
}
