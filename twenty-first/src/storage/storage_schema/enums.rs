/// Database write operations
pub enum WriteOperation<K, V> {
    /// write operation
    Write(K, V),
    /// delete operation
    Delete(K),
}

/// Vector write operations
pub enum VecWriteOperation<Index, T> {
    /// overwrite, aka set operation
    OverWrite((Index, T)),
    /// push to end operation
    Push(T),
    /// pop from end operation
    Pop,
}
