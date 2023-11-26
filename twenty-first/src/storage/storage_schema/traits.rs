use super::WriteOperation;

/// Defines table interface for types used by [`DbtSchema`]
pub trait DbTable<ParentKey, ParentValue> {
    /// Retrieve all unwritten operations and empty write-queue
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>>;
    /// Restore existing table if present, else create a new one
    fn restore_or_new(&mut self);
}

/// Defines storage singleton for types created by [`DbtSchema`]
pub trait StorageSingleton<T>
where
    T: Clone,
{
    /// Retrieve value
    fn get(&self) -> T;
    /// Set value
    fn set(&mut self, t: T);
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
    fn persist(&mut self);
    /// restore, or new
    fn restore_or_new(&mut self);
}
