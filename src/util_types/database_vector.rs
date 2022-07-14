use rusty_leveldb::DB;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;

/// This is the key for the storage of the length of the vector
const LENGTH_KEY: Vec<u8> = vec![];
const INDEX_ZERO: u128 = 0u128;

pub struct DatabaseVector<T: Serialize + DeserializeOwned> {
    db: DB,
    _type: PhantomData<T>,
}

impl<T: Serialize + DeserializeOwned> DatabaseVector<T> {
    fn set_length(&mut self, length: u128) {
        let length_as_bytes = bincode::serialize(&length).unwrap();
        self.db
            .put(&LENGTH_KEY, &length_as_bytes)
            .expect("Length write must succeed")
    }

    fn delete(&mut self, index: u128) {
        let index_as_bytes = bincode::serialize(&index).unwrap();
        self.db
            .delete(&index_as_bytes)
            .expect("Deleting element must succeed");
    }

    /// Return true if the database vector looks empty. Used for sanity check when creating
    /// a new database vector.
    fn attempt_verify_empty(&mut self) -> bool {
        let index_bytes: Vec<u8> = bincode::serialize(&INDEX_ZERO).unwrap();
        self.db.get(&index_bytes).is_none()
    }

    pub fn is_empty(&mut self) -> bool {
        self.len() == 0
    }

    pub fn len(&mut self) -> u128 {
        let length_as_bytes = self.db.get(&LENGTH_KEY).expect("Length must exist");
        bincode::deserialize(&length_as_bytes).unwrap()
    }

    /// given a database containing a database vector, restore it into a database vector struct
    pub fn restore(db: DB) -> Self {
        let mut ret = Self {
            _type: PhantomData,
            db,
        };

        // sanity check to verify that the length is set
        let _dummy_res = ret.len();
        ret
    }

    /// Create a new, empty database vector
    pub fn new(db: DB) -> Self {
        let mut ret = DatabaseVector {
            db,
            _type: PhantomData,
        };
        assert!(
            ret.attempt_verify_empty(),
            "Database must be empty when instantiating database vector with `new`"
        );
        ret.set_length(0);

        ret
    }

    pub fn get(&mut self, index: u128) -> T {
        debug_assert!(
            self.len() > index,
            "Cannot get outside of length. Length: {}, index: {}",
            self.len(),
            index
        );
        let index_bytes: Vec<u8> = bincode::serialize(&index).unwrap();
        let elem_as_bytes = self.db.get(&index_bytes).unwrap();
        bincode::deserialize(&elem_as_bytes).unwrap()
    }

    pub fn set(&mut self, index: u128, value: T) {
        debug_assert!(
            self.len() > index,
            "Cannot set outside of length. Length: {}, index: {}",
            self.len(),
            index
        );
        let index_bytes: Vec<u8> = bincode::serialize(&index).unwrap();
        let value_bytes: Vec<u8> = bincode::serialize(&value).unwrap();
        self.db.put(&index_bytes, &value_bytes).unwrap();
    }

    pub fn pop(&mut self) -> Option<T> {
        match self.len() {
            0 => None,
            length => {
                let element = self.get(length - 1);
                self.delete(length - 1);
                self.set_length(length - 1);
                Some(element)
            }
        }
    }

    pub fn push(&mut self, value: T) {
        let length = self.len();
        let index_bytes = bincode::serialize(&length).unwrap();
        let value_bytes = bincode::serialize(&value).unwrap();
        self.db.put(&index_bytes, &value_bytes).unwrap();
        self.set_length(length + 1);
    }

    /// Dispose of the vector and return the database. You should probably only use this for testing.
    pub fn extract_db(self) -> DB {
        self.db
    }
}

#[cfg(test)]
mod database_vector_tests {
    use super::*;
    use rusty_leveldb::DB;

    #[test]
    fn push_pop_test() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        let mut db_vector: DatabaseVector<u64> = DatabaseVector::new(db);
        assert_eq!(0, db_vector.len());
        assert!(db_vector.is_empty());

        // pop an element and verify that `None` is returns
        assert!(db_vector.pop().is_none());
        assert_eq!(0, db_vector.len());
        assert!(db_vector.is_empty());

        // push two elements and check length and values
        db_vector.push(14442);
        db_vector.push(5558999);
        assert_eq!(14442, db_vector.get(0));
        assert_eq!(5558999, db_vector.get(1));
        assert_eq!(2, db_vector.len());

        // Set a value to verify that `set` works
        db_vector.set(1, 4);
        assert_eq!(4, db_vector.get(1));

        // Verify that `pop` works
        assert_eq!(Some(4), db_vector.pop());
        assert_eq!(1, db_vector.len());
        assert_eq!(Some(14442), db_vector.pop());
        assert_eq!(0, db_vector.len());
        assert!(db_vector.is_empty());
    }

    #[test]
    fn push_many_test() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        let mut db_vector: DatabaseVector<u64> = DatabaseVector::new(db);
        for _ in 0..1000 {
            db_vector.push(17);
        }

        assert_eq!(1000, db_vector.len());
    }

    #[should_panic = "Cannot get outside of length. Length: 0, index: 0"]
    #[test]
    fn panic_on_index_out_of_range_empty_test() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        let mut db_vector: DatabaseVector<u64> = DatabaseVector::new(db);
        db_vector.get(0);
    }

    #[should_panic = "Cannot get outside of length. Length: 1, index: 1"]
    #[test]
    fn panic_on_index_out_of_range_length_one_test() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        let mut db_vector: DatabaseVector<u64> = DatabaseVector::new(db);
        db_vector.push(5558999);
        db_vector.get(1);
    }

    #[should_panic = "Cannot set outside of length. Length: 1, index: 1"]
    #[test]
    fn panic_on_index_out_of_range_length_one_set_test() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        let mut db_vector: DatabaseVector<u64> = DatabaseVector::new(db);
        db_vector.push(5558999);
        db_vector.set(1, 14);
    }

    #[test]
    fn restore_test() {
        // Verify that we can restore a database vector object from a database object
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        let mut db_vector: DatabaseVector<u64> = DatabaseVector::new(db);
        assert!(db_vector.is_empty());
        let extracted_db = db_vector.db;
        let mut new_db_vector: DatabaseVector<u64> = DatabaseVector::restore(extracted_db);
        assert!(new_db_vector.is_empty());
    }
}
