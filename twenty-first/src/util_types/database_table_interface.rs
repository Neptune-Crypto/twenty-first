use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use rusty_leveldb::{WriteBatch, DB};
use serde::{de::DeserializeOwned, Serialize};

type IndexType = u64;

pub trait DbtVec<T> {
    fn is_empty(&mut self) -> bool;
    fn len(&mut self) -> IndexType;
    fn overwrite_with_vec(&mut self, new_vector: Vec<T>);
    fn get(&mut self, index: IndexType) -> T;
    fn set(&mut self, index: IndexType, value: T);
    fn batch_set(&mut self, indices_and_vals: &[(IndexType, T)]);
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

pub struct RustyLevelDbVec<T: Serialize + DeserializeOwned> {
    key_prefix: u8,
    db: Rc<RefCell<DB>>,
    write_queue: VecDeque<WriteElement<T>>,
    length: u64,
}

impl<T: Serialize + DeserializeOwned> RustyLevelDbVec<T> {
    fn get_length_key(&self) -> [u8; 2] {
        const LENGTH_KEY: u8 = 0u8;
        [self.key_prefix, LENGTH_KEY]
    }

    pub fn pull_queue(&mut self, batch_write: &mut WriteBatch) {
        let original_length = self.length;
        while let Some(write_element) = self.write_queue.pop_front() {
            match write_element {
                WriteElement::OverWrite((i, t)) => {
                    let key = vec![vec![self.key_prefix], bincode::serialize(&i).unwrap()].concat();
                    let value = bincode::serialize(&t).unwrap();
                    batch_write.put(&key, &value);
                }
                WriteElement::Push(t) => {
                    let key = vec![
                        vec![self.key_prefix],
                        bincode::serialize(&self.length).unwrap(),
                    ]
                    .concat();
                    self.length += 1;
                    let value = bincode::serialize(&t).unwrap();
                    batch_write.put(&key, &value);
                }
                WriteElement::Pop => {
                    let key = vec![
                        vec![self.key_prefix],
                        bincode::serialize(&(self.length - 1)).unwrap(),
                    ]
                    .concat();
                    self.length -= 1;
                    batch_write.delete(&key);
                }
            };
        }

        if original_length != self.length {
            let key = self.get_length_key();
            batch_write.put(&key, &bincode::serialize(&self.length).unwrap());
        }
    }
}
