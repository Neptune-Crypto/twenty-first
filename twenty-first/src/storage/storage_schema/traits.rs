use super::WriteOperation;

pub trait DbTable<ParentKey, ParentValue> {
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>>;
    fn restore_or_new(&mut self);
}

pub trait StorageReader<ParentKey, ParentValue> {
    /// Return multiple values from storage, in the same order as the input keys
    fn get_many(&self, keys: &[ParentKey]) -> Vec<Option<ParentValue>>;

    /// Return a single value from storage
    fn get(&self, key: ParentKey) -> Option<ParentValue>;
}

pub trait StorageSingleton<T>
where
    T: Clone,
{
    fn get(&self) -> T;
    fn set(&mut self, t: T);
}

pub trait StorageWriter<ParentKey, ParentValue> {
    fn persist(&mut self);
    fn restore_or_new(&mut self);
}
