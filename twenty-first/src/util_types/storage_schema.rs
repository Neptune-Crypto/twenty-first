use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    fmt::Debug,
    sync::Arc,
};

use rusty_leveldb::{WriteBatch, DB};

use super::storage_vec::{IndexType, StorageVec};

pub enum WriteOperation<ParentKey, ParentValue> {
    Write(ParentKey, ParentValue),
    Delete(ParentKey),
}

pub trait DbTable<ParentKey, ParentValue> {
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>>;
    fn restore_or_new(&mut self);
}

pub trait StorageReader<ParentKey, ParentValue> {
    fn get(&mut self, key: ParentKey) -> Option<ParentValue>;
}

pub enum VecWriteOperation<Index, T> {
    OverWrite((Index, T)),
    Push(T),
    Pop,
}

pub struct DbtVec<ParentKey, ParentValue, Index, T> {
    reader: Arc<RefCell<dyn StorageReader<ParentKey, ParentValue>>>,
    current_length: Index,
    key_prefix: u8,
    write_queue: VecDeque<VecWriteOperation<Index, T>>,
    cache: HashMap<Index, T>,
    name: String,
}

impl<ParentKey, ParentValue, Index, T> DbtVec<ParentKey, ParentValue, Index, T>
where
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    ParentKey: From<Index>,
    Index: From<ParentValue> + From<u64> + Clone,
{
    // Return the key of ParentKey type used to store the length of the vector
    fn get_length_key(key_prefix: u8) -> ParentKey {
        let const_length_key: ParentKey = 0u8.into();
        let key_prefix_key: ParentKey = key_prefix.into();
        (key_prefix_key, const_length_key).into()
    }

    /// Return the length at the last write to disk
    fn persisted_length(&self) -> Option<Index> {
        self.reader
            .as_ref()
            .borrow_mut()
            .get(Self::get_length_key(self.key_prefix))
            .map(|v| v.into())
    }

    /// Return the key of ParentKey type used to store the element at a given index of Index type
    fn get_index_key(&self, index: Index) -> ParentKey {
        let key_prefix_key: ParentKey = self.key_prefix.into();
        let index_key: ParentKey = index.into();
        (key_prefix_key, index_key).into()
    }

    pub fn new(
        reader: Arc<RefCell<dyn StorageReader<ParentKey, ParentValue>>>,
        key_prefix: u8,
        name: &str,
    ) -> Self {
        let length: Index = 0.into();
        let cache = HashMap::new();
        Self {
            key_prefix,
            reader,
            write_queue: VecDeque::default(),
            current_length: length,
            cache,
            name: name.to_string(),
        }
    }
}

impl<ParentKey, ParentValue, T> StorageVec<T> for DbtVec<ParentKey, ParentValue, IndexType, T>
where
    ParentKey: From<IndexType>,
    ParentValue: From<T>,
    T: Clone,
    T: From<ParentValue>,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    IndexType: From<ParentValue>,
{
    fn is_empty(&self) -> bool {
        self.current_length == 0
    }

    fn len(&self) -> IndexType {
        self.current_length
    }

    fn get(&self, index: IndexType) -> T {
        // Disallow getting values out-of-bounds
        assert!(
            index < self.len(),
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self.current_length,
            self.name
        );

        // try cache first
        if self.cache.contains_key(&index) {
            return self.cache.get(&index).unwrap().clone();
        }

        // then try persistent storage
        let key: ParentKey = self.get_index_key(index);
        let val = self
            .reader
            .as_ref()
            .borrow_mut()
            .get(key)
            .unwrap_or_else(|| {
                panic!(
                    "Element with index {index} does not exist in {}. This should not happen",
                    self.name
                )
            });
        val.into()
    }

    fn set(&mut self, index: IndexType, value: T) {
        // Disallow setting values out-of-bounds
        assert!(
            index < self.len(),
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self.current_length,
            self.name
        );

        let _old_value = self.cache.insert(index, value.clone());

        // TODO: If `old_value` is Some(*) use it to remove the corresponding
        // element in the `write_queue` to reduce disk IO.

        self.write_queue
            .push_back(VecWriteOperation::OverWrite((index, value)));
    }

    fn pop(&mut self) -> Option<T> {
        // add to write queue
        self.write_queue.push_back(VecWriteOperation::Pop);

        // If vector is empty, return None
        if self.current_length == 0 {
            return None;
        }

        // Update length
        self.current_length -= 1;

        // try cache first
        if self.cache.contains_key(&self.current_length) {
            self.cache.remove(&self.current_length)
        } else {
            // then try persistent storage
            let key = self.get_index_key(self.current_length);
            self.reader
                .as_ref()
                .borrow_mut()
                .get(key)
                .map(|value| value.into())
        }
    }

    fn push(&mut self, value: T) {
        // add to write queue
        self.write_queue
            .push_back(VecWriteOperation::Push(value.clone()));

        // record in cache
        let _old_value = self.cache.insert(self.current_length, value);

        // TODO: if `old_value` is Some(_) then use it to remove the corresponding
        // element from the `write_queue` to reduce disk operations

        // update length
        self.current_length += 1;
    }
}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtVec<ParentKey, ParentValue, IndexType, T>
where
    ParentKey: From<IndexType>,
    ParentValue: From<T>,
    T: Clone,
    T: From<ParentValue>,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    IndexType: From<ParentValue>,
    ParentValue: From<IndexType>,
{
    /// Collect all added elements that have not yet bit persisted
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        let maybe_original_length = self.persisted_length();
        // necessary because we need maybe_original_length.is_none() later
        #[allow(clippy::unnecessary_unwrap)]
        let original_length = if maybe_original_length.is_some() {
            maybe_original_length.unwrap()
        } else {
            0
        };
        let mut length = original_length;
        let mut queue = vec![];
        while let Some(write_element) = self.write_queue.pop_front() {
            match write_element {
                VecWriteOperation::OverWrite((i, t)) => {
                    let key = self.get_index_key(i);
                    queue.push(WriteOperation::Write(key, Into::<ParentValue>::into(t)));
                }
                VecWriteOperation::Push(t) => {
                    let key = self.get_index_key(length);
                    length += 1;
                    queue.push(WriteOperation::Write(key, Into::<ParentValue>::into(t)));
                }
                VecWriteOperation::Pop => {
                    let key = self.get_index_key(length - 1);
                    length -= 1;
                    queue.push(WriteOperation::Delete(key));
                }
            };
        }

        if original_length != length || maybe_original_length.is_none() {
            let key = Self::get_length_key(self.key_prefix);
            queue.push(WriteOperation::Write(
                key,
                Into::<ParentValue>::into(length),
            ));
        }

        self.cache.clear();

        queue
    }

    fn restore_or_new(&mut self) {
        if let Some(length) = self
            .reader
            .as_ref()
            .borrow_mut()
            .get(Self::get_length_key(self.key_prefix))
        {
            self.current_length = length.into();
        } else {
            self.current_length = 0;
        }
    }
}

// possible future extension
// pub struct DbtHashMap<Key, Value, K, V> {
//     parent: Arc<Mutex<DbtSchema<Key, Value>>>,
// }

pub trait StorageSingleton<T>
where
    T: Clone,
{
    fn get(&self) -> T;
    fn set(&mut self, t: T);
}

pub struct DbtSingleton<ParentKey, ParentValue, T> {
    current_value: T,
    old_value: T,
    key: ParentKey,
    reader: Arc<RefCell<dyn StorageReader<ParentKey, ParentValue>>>,
}

impl<ParentKey, ParentValue, T> StorageSingleton<T> for DbtSingleton<ParentKey, ParentValue, T>
where
    T: Clone + From<ParentValue>,
{
    fn get(&self) -> T {
        self.current_value.clone()
    }

    fn set(&mut self, t: T) {
        self.current_value = t;
    }
}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtSingleton<ParentKey, ParentValue, T>
where
    T: Eq + Clone + Default + From<ParentValue>,
    ParentValue: From<T> + Debug,
    ParentKey: Clone,
{
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        if self.current_value == self.old_value {
            vec![]
        } else {
            self.old_value = self.current_value.clone();
            vec![WriteOperation::Write(
                self.key.clone(),
                self.current_value.clone().into(),
            )]
        }
    }

    fn restore_or_new(&mut self) {
        self.current_value = match self.reader.as_ref().borrow_mut().get(self.key.clone()) {
            Some(value) => value.into(),
            None => T::default(),
        }
    }
}

pub struct DbtSchema<ParentKey, ParentValue, Reader: StorageReader<ParentKey, ParentValue>> {
    tables: Vec<Arc<RefCell<dyn DbTable<ParentKey, ParentValue>>>>,
    reader: Arc<RefCell<Reader>>,
}

impl<ParentKey, ParentValue, Reader: StorageReader<ParentKey, ParentValue> + 'static>
    DbtSchema<ParentKey, ParentValue, Reader>
{
    pub fn new_vec<Index, T>(
        &mut self,
        name: &str,
    ) -> Arc<RefCell<DbtVec<ParentKey, ParentValue, Index, T>>>
    where
        ParentKey: From<IndexType> + 'static,
        ParentValue: From<T> + 'static,
        T: Clone + From<ParentValue> + 'static,
        ParentKey: From<(ParentKey, ParentKey)>,
        ParentKey: From<u8>,
        Index: From<ParentValue>,
        ParentValue: From<IndexType>,
        Index: From<u64> + 'static,
        DbtVec<ParentKey, ParentValue, Index, T>: DbTable<ParentKey, ParentValue>,
    {
        assert!(self.tables.len() < 255);
        let reader = self.reader.clone();
        let vector = DbtVec::<ParentKey, ParentValue, Index, T> {
            reader,
            current_length: 0.into(),
            key_prefix: self.tables.len() as u8,
            write_queue: VecDeque::new(),
            cache: HashMap::new(),
            name: name.to_string(),
        };
        let arc_refcell_vector = Arc::new(RefCell::new(vector));
        self.tables.push(arc_refcell_vector.clone());
        arc_refcell_vector
    }

    // possible future extension
    // fn new_hashmap<K, V>(&self) -> Arc<RefCell<DbtHashMap<K, V>>> { }

    pub fn new_singleton<S>(
        &mut self,
        key: ParentKey,
    ) -> Arc<RefCell<DbtSingleton<ParentKey, ParentValue, S>>>
    where
        S: Default + Eq + Clone + 'static,
        ParentKey: 'static,
        ParentValue: From<S> + 'static,
        ParentKey: From<(ParentKey, ParentKey)> + From<u8>,
        DbtSingleton<ParentKey, ParentValue, S>: DbTable<ParentKey, ParentValue>,
    {
        let reader = self.reader.clone();
        let singleton = DbtSingleton::<ParentKey, ParentValue, S> {
            current_value: S::default(),
            old_value: S::default(),
            key,
            reader,
        };
        let arc_refcell_singleton = Arc::new(RefCell::new(singleton));
        self.tables.push(arc_refcell_singleton.clone());
        arc_refcell_singleton
    }
}

pub trait StorageWriter<ParentKey, ParentValue> {
    fn persist(&mut self);
    fn restore_or_new(&mut self);
}

#[derive(Clone, PartialEq, Eq)]
struct RustyKey(Vec<u8>);
impl From<u8> for RustyKey {
    fn from(value: u8) -> Self {
        Self([value].to_vec())
    }
}
impl From<(RustyKey, RustyKey)> for RustyKey {
    fn from(value: (RustyKey, RustyKey)) -> Self {
        let v0 = value.0 .0;
        let v1 = value.1 .0;
        RustyKey([v0, v1].concat())
    }
}
#[derive(Debug)]
struct RustyValue(Vec<u8>);

/// Database schema and tables logic for RustyLevelDB. You probably
/// want to implement your own storage class after this example so
/// that you can hardcode the schema in new(). But it is nevertheless
/// possible to use this struct and add to the scheme after calling
/// new() (that's what the tests do).
pub struct SimpleRustyStorage {
    db: Arc<RefCell<DB>>,
    schema: DbtSchema<RustyKey, RustyValue, SimpleRustyReader>,
}

impl StorageWriter<RustyKey, RustyValue> for SimpleRustyStorage {
    fn persist(&mut self) {
        let mut write_batch = WriteBatch::new();
        for table in &self.schema.tables {
            let operations = table.as_ref().borrow_mut().pull_queue();
            for op in operations {
                match op {
                    WriteOperation::Write(key, value) => write_batch.put(&key.0, &value.0),
                    WriteOperation::Delete(key) => write_batch.delete(&key.0),
                }
            }
        }

        self.db
            .as_ref()
            .borrow_mut()
            .write(write_batch, true)
            .expect("Could not persist to database.");
    }

    fn restore_or_new(&mut self) {
        for table in &self.schema.tables {
            table.as_ref().borrow_mut().restore_or_new();
        }
    }
}

impl SimpleRustyStorage {
    pub fn new(db: DB) -> Self {
        let db_pointer = Arc::new(RefCell::new(db));
        let reader = SimpleRustyReader {
            db: db_pointer.clone(),
        };
        let schema = DbtSchema::<RustyKey, RustyValue, SimpleRustyReader> {
            tables: Vec::new(),
            reader: Arc::new(RefCell::new(reader)),
        };
        Self {
            db: db_pointer,
            schema,
        }
    }

    pub fn close(&mut self) {
        self.db
            .as_ref()
            .borrow_mut()
            .close()
            .expect("Could not close database.");
    }
}

struct SimpleRustyReader {
    db: Arc<RefCell<DB>>,
}

impl StorageReader<RustyKey, RustyValue> for SimpleRustyReader {
    fn get(&mut self, key: RustyKey) -> Option<RustyValue> {
        self.db.as_ref().borrow_mut().get(&key.0).map(RustyValue)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[derive(Default, PartialEq, Eq, Clone, Debug)]
    struct S(Vec<u8>);
    impl From<Vec<u8>> for S {
        fn from(value: Vec<u8>) -> Self {
            S(value)
        }
    }
    impl From<S> for Vec<u8> {
        fn from(value: S) -> Self {
            value.0
        }
    }
    impl From<(S, S)> for S {
        fn from(value: (S, S)) -> Self {
            let vector0: Vec<u8> = value.0.into();
            let vector1: Vec<u8> = value.1.into();
            S([vector0, vector1].concat())
        }
    }
    impl From<RustyValue> for S {
        fn from(value: RustyValue) -> Self {
            Self(value.0)
        }
    }
    impl From<S> for RustyValue {
        fn from(value: S) -> Self {
            Self(value.0)
        }
    }

    #[test]
    fn test_simple_singleton() {
        let singleton_value = S([1u8, 3u8, 3u8, 7u8].to_vec());
        // let opt = rusty_leveldb::Options::default();
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let singleton = rusty_storage
            .schema
            .new_singleton::<S>(RustyKey([1u8; 1].to_vec()));

        // initialize
        rusty_storage.restore_or_new();

        // test
        assert_eq!(singleton.as_ref().borrow_mut().get(), S([].to_vec()));

        // set
        singleton.as_ref().borrow_mut().set(singleton_value.clone());

        // test
        assert_eq!(singleton.as_ref().borrow_mut().get(), singleton_value);

        // persist
        rusty_storage.persist();

        // test
        assert_eq!(singleton.as_ref().borrow_mut().get(), singleton_value);

        // drop
        rusty_storage.close();

        // restore
        let new_db = DB::open("test-database", opt).unwrap();
        let mut new_rusty_storage = SimpleRustyStorage::new(new_db);
        let new_singleton = new_rusty_storage
            .schema
            .new_singleton::<S>(RustyKey([1u8; 1].to_vec()));
        new_rusty_storage.restore_or_new();

        // test
        assert_eq!(new_singleton.as_ref().borrow_mut().get(), singleton_value);
    }
}
