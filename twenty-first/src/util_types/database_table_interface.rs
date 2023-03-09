use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
};

use rusty_leveldb::{WriteBatch, DB};
use serde::{de::DeserializeOwned, Serialize};

type IndexType = u64;

pub trait DbtVec<T> {
    fn is_empty(&mut self) -> bool;
    fn len(&mut self) -> IndexType;
    fn get(&mut self, index: IndexType) -> T;
    fn set(&mut self, index: IndexType, value: T);
    fn pop(&mut self) -> Option<T>;
    fn push(&mut self, value: T);
}

pub enum WriteElement<T: Serialize + DeserializeOwned> {
    OverWrite((IndexType, T)),
    Push(T),
    Pop,
    // NewWrite((IndexType, T)),
    // Delete(IndexType),
}

impl<T: Serialize + DeserializeOwned + Clone> DbtVec<T> for RustyLevelDbVec<T> {
    fn is_empty(&mut self) -> bool {
        self.length == 0
    }

    fn len(&mut self) -> IndexType {
        self.length
    }

    fn get(&mut self, index: IndexType) -> T {
        // try cache first
        if self.cache.contains_key(&index) {
            return match &self.cache[&index] {
                Some(value) => value.clone(),
                None => panic!("Element with index {index} was marked for deletion in cache."),
            };
        }

        // then try persistent storage
        let db_key = self.get_index_key(index);
        let db_val = self
            .db
            .borrow_mut()
            .get(&db_key)
            .expect("Element with index {index} does not exist in ");
        bincode::deserialize(&db_val).unwrap()
    }

    fn set(&mut self, index: IndexType, value: T) {
        let _old_value = self.cache.insert(index, Some(value.clone()));

        // TODO: If `old_value` is Some(*) use it to remove the corresponding
        // element in the `write_queue` to reduce disk IO.

        self.write_queue
            .push_back(WriteElement::OverWrite((index, value)));
    }

    fn pop(&mut self) -> Option<T> {
        // add to write queue
        self.write_queue.push_back(WriteElement::Pop);

        // update length
        self.length -= 1;

        // try cache first
        if self.cache.contains_key(&self.length) {
            self.cache.insert(self.length, None).unwrap()
        }
        // then try persistent storage
        else {
            let db_key = self.get_index_key(self.length);
            self.db
                .borrow_mut()
                .get(&db_key)
                .map(|bytes| bincode::deserialize(&bytes).unwrap())
        }
    }

    fn push(&mut self, value: T) {
        // add to write queue
        self.write_queue
            .push_back(WriteElement::Push(value.clone()));

        // record in cache
        let _old_value = self.cache.insert(self.length, Some(value));

        // TODO: if `old_value` is Some(_) then use it to remove the corresponding
        // element from the `write_queue` to reduce disk operations

        // update length
        self.length += 1;
    }
}

pub struct RustyLevelDbVec<T: Serialize + DeserializeOwned> {
    key_prefix: u8,
    db: Rc<RefCell<DB>>,
    write_queue: VecDeque<WriteElement<T>>,
    length: IndexType,
    cache: HashMap<IndexType, Option<T>>,
    pub name: String,
}

impl<T: Serialize + DeserializeOwned> RustyLevelDbVec<T> {
    fn get_length_key(key_prefix: u8) -> [u8; 2] {
        const LENGTH_KEY: u8 = 0u8;
        [key_prefix, LENGTH_KEY]
    }

    fn get_persisted_length(&self) -> IndexType {
        let key = Self::get_length_key(self.key_prefix);
        match self.db.borrow_mut().get(&key) {
            Some(value) => bincode::deserialize(&value).unwrap(),
            None => 0,
        }
    }

    fn get_index_key(&self, index: IndexType) -> [u8; 7] {
        vec![vec![self.key_prefix], bincode::serialize(&index).unwrap()]
            .concat()
            .try_into()
            .unwrap()
    }

    pub fn new(db: Rc<RefCell<DB>>, key_prefix: u8, name: &str) -> Self {
        let length_key = Self::get_length_key(key_prefix);
        let length = match db.borrow_mut().get(&length_key) {
            Some(length_bytes) => bincode::deserialize(&length_bytes).unwrap(),
            None => 0,
        };
        let cache = HashMap::new();
        Self {
            key_prefix,
            db,
            write_queue: VecDeque::default(),
            length,
            cache,
            name: name.to_string(),
        }
    }

    pub fn pull_queue(&mut self, batch_write: &mut WriteBatch) {
        let original_length = self.get_persisted_length();
        let mut length = original_length;
        while let Some(write_element) = self.write_queue.pop_front() {
            match write_element {
                WriteElement::OverWrite((i, t)) => {
                    let key = self.get_index_key(i);
                    let value = bincode::serialize(&t).unwrap();
                    batch_write.put(&key, &value);
                }
                WriteElement::Push(t) => {
                    let key =
                        vec![vec![self.key_prefix], bincode::serialize(&length).unwrap()].concat();
                    length += 1;
                    let value = bincode::serialize(&t).unwrap();
                    batch_write.put(&key, &value);
                }
                WriteElement::Pop => {
                    let key = vec![
                        vec![self.key_prefix],
                        bincode::serialize(&(length - 1)).unwrap(),
                    ]
                    .concat();
                    length -= 1;
                    batch_write.delete(&key);
                }
            };
        }

        if original_length != length {
            let key = Self::get_length_key(self.key_prefix);
            batch_write.put(&key, &bincode::serialize(&self.length).unwrap());
        }

        self.cache.clear();
    }
}
