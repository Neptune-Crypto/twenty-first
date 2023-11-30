//! Traits that define the StorageSchema interface
//!
//! It is recommended to wildcard import these with
//! `use twenty_first::storage::storage_vec::traits::*`

use super::WriteOperation;

/// Defines table interface for types used by [`super::DbtSchema`]
pub trait DbTable<ParentKey, ParentValue> {
    /// Retrieve all unwritten operations and empty write-queue
    fn pull_queue(&self) -> Vec<WriteOperation<ParentKey, ParentValue>>;
    /// Restore existing table if present, else create a new one
    fn restore_or_new(&self);
}

/// Defines storage singleton for types created by [`super::DbtSchema`]
pub trait StorageSingletonReads<T>
where
    T: Clone,
{
    /// Retrieve value
    fn get(&self) -> T;
}

/// Defines storage singleton mutable write ops for types created by [`super::DbtSchema`]
pub(super) trait StorageSingletonMutableWrites<T>
where
    T: Clone,
{
    /// Set value
    fn set(&mut self, t: T);
}

/// Defines storage singleton immutable write ops for types created by [`super::DbtSchema`]
pub trait StorageSingletonImmutableWrites<T>
where
    T: Clone,
{
    /// Set value
    fn set(&self, t: T);
}

/// Defines storage singleton read ops for types created by [`super::DbtSchema`]
pub trait StorageSingleton<T>:
    StorageSingletonReads<T> + StorageSingletonImmutableWrites<T>
where
    T: Clone,
{
}

/// Defines storage reader interface
pub trait StorageReader<ParentKey, ParentValue> {
    /// Return multiple values from storage, in the same order as the input keys
    fn get_many(&self, keys: &[ParentKey]) -> Vec<Option<ParentValue>>;

    /// Return a single value from storage
    fn get(&self, key: ParentKey) -> Option<ParentValue>;
}

/// Defines storage writer interface
pub trait StorageWriter<ParentKey, ParentValue> {
    /// Write data to storage
    fn persist(&self);
    /// restore, or new
    fn restore_or_new(&self);
}
