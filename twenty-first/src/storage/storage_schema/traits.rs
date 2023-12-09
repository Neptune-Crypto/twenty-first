//! Traits that define the StorageSchema interface
//!
//! It is recommended to wildcard import these with
//! `use twenty_first::storage::storage_vec::traits::*`

use super::{RustyKey, RustyValue, WriteOperation};
pub use crate::leveldb::database::key::IntoLevelDBKey;

/// Defines table interface for types used by [`super::DbtSchema`]
pub trait DbTable {
    /// Retrieve all unwritten operations and empty write-queue
    fn pull_queue(&self) -> Vec<WriteOperation>;
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
pub trait StorageReader {
    /// Return multiple values from storage, in the same order as the input keys
    fn get_many(&self, keys: &[RustyKey]) -> Vec<Option<RustyValue>>;

    /// Return a single value from storage
    fn get(&self, key: RustyKey) -> Option<RustyValue>;
}

/// Defines storage writer interface
pub trait StorageWriter {
    /// Write data to storage
    fn persist(&mut self);
    /// restore, or new
    fn restore_or_new(&mut self);
}
