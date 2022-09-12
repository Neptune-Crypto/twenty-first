use rusty_leveldb::{WriteBatch, DB};
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;

/// Permanent storage of a fixed-length array with elements of type `T`.
pub struct DatabaseArray<const N: u128, T: Serialize + DeserializeOwned + Default> {
    db: DB,
    _type: PhantomData<T>,
}

impl<const N: u128, T: Serialize + DeserializeOwned + Default> DatabaseArray<N, T> {
    /// Return the element at position index. Returns `T::defeault()` if value is unset
    pub fn get(&mut self, index: u128) -> T {
        assert!(
            N > index,
            "Cannot get outside of length. Length: {}, index: {}",
            N,
            index
        );
        let index_bytes: Vec<u8> = bincode::serialize(&index).unwrap();
        let elem_as_bytes_res = self.db.get(&index_bytes);
        match elem_as_bytes_res {
            Some(bytes) => bincode::deserialize(&bytes).unwrap(),
            None => T::default(),
        }
    }

    pub fn batch_set(&mut self, indices_and_vals: &[(u128, T)]) {
        let indices: Vec<u128> = indices_and_vals.iter().map(|(index, _)| *index).collect();
        assert!(
            indices.iter().all(|index| *index < N),
            "All indices must be lower than length of array. Got: {:?}",
            indices
        );
        let mut batch_write = WriteBatch::new();
        for (index, val) in indices_and_vals.iter() {
            let index_bytes: Vec<u8> = bincode::serialize(index).unwrap();
            let value_bytes: Vec<u8> = bincode::serialize(val).unwrap();
            batch_write.put(&index_bytes, &value_bytes);
        }

        self.db
            .write(batch_write, true)
            .expect("Failed to batch-write to database");
    }

    /// Set the value at index
    pub fn set(&mut self, index: u128, value: T) {
        assert!(
            N > index,
            "Cannot set outside of length. Length: {}, index: {}",
            N,
            index
        );
        let index_bytes: Vec<u8> = bincode::serialize(&index).unwrap();
        let value_bytes: Vec<u8> = bincode::serialize(&value).unwrap();
        self.db.put(&index_bytes, &value_bytes).unwrap();
        self.db.flush().expect("set must succeed flushing");
    }

    // Flush database
    pub fn flush(&mut self) {
        self.db.flush().expect("Flush must succeed.")
    }

    /// Create a new, default-initialized database array. Input database must be empty.
    pub fn new(db: DB) -> Self {
        Self {
            db,
            _type: PhantomData,
        }
    }

    /// Restore a database array object from a database.
    pub fn restore(db: DB) -> Self {
        Self::new(db)
    }

    /// Drop the database array and return the database in which its values are stored
    pub fn extract_db(self) -> DB {
        self.db
    }
}

#[cfg(test)]
mod database_array_tests {
    use super::*;
    use rusty_leveldb::DB;

    #[test]
    fn init_and_default_values_test() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        assert_eq!(0u64, u64::default());
        let mut db_array: DatabaseArray<101, u64> = DatabaseArray::new(db);
        assert_eq!(0u64, db_array.get(0));
        assert_eq!(0u64, db_array.get(100));
        assert_eq!(0u64, db_array.get(42));
        db_array.set(0, 17);
        db_array.set(59, 16);
        db_array.set(80, 0);
        assert_eq!(17u64, db_array.get(0));
        assert_eq!(16u64, db_array.get(59));
        assert_eq!(0u64, db_array.get(80));
        assert_eq!(0u64, db_array.get(81));
        assert_eq!(0u64, db_array.get(79));

        // Test batch-set, verify that writes worked and that other values were unchanged
        db_array.batch_set(&vec![(100, 10000), (11, 1100), (12, 1200)]);
        assert_eq!(10000u64, db_array.get(100));
        assert_eq!(1100u64, db_array.get(11));
        assert_eq!(1200u64, db_array.get(12));
        assert_eq!(0u64, db_array.get(80));
        assert_eq!(0u64, db_array.get(81));
        assert_eq!(0u64, db_array.get(79));
    }

    #[should_panic = "Cannot get outside of length. Length: 101, index: 101"]
    #[test]
    fn panic_on_index_out_of_range_empty_test() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        let mut db_array: DatabaseArray<101, u64> = DatabaseArray::new(db);
        db_array.get(101);
    }

    #[should_panic = "Cannot set outside of length. Length: 50, index: 90"]
    #[test]
    fn panic_on_index_out_of_range_length_one_set_test() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("mydatabase", opt).unwrap();
        let mut db_array: DatabaseArray<50, u64> = DatabaseArray::new(db);
        db_array.set(90, 17);
    }
}
