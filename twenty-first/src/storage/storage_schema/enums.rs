/// Database write operations
#[derive(Debug, Clone)]
pub enum WriteOperation<K, V> {
    /// write operation
    Write(K, V),
    /// delete operation
    Delete(K),
}

/// Vector write operations
#[derive(Debug, Clone)]
pub enum VecWriteOperation<Index, T> {
    /// overwrite, aka set operation
    OverWrite((Index, T)),
    /// push to end operation
    Push(T),
    /// pop from end operation
    Pop,
}
