pub enum WriteOperation<K, V> {
    Write(K, V),
    Delete(K),
}

pub enum VecWriteOperation<Index, T> {
    OverWrite((Index, T)),
    Push(T),
    Pop,
}
