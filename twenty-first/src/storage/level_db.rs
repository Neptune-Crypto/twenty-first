//! [`DB`] wraps [`crate::leveldb::database::Database`] and provides
//! functionality for reading the database on-disk path
//! as well as destroying the on-disk database manually
//! or automatically upon drop.
//!
//! auto-destroy-on-drop is needed for unit tests that use the DB.

use leveldb::{
    batch::{Batch, WriteBatch},
    compaction::Compaction,
    database::comparator::Comparator,
    database::Database,
    error::Error as DbError,
    iterator::{Iterable, Iterator, KeyIterator, ValueIterator},
    key::IntoLevelDBKey,
    options::{Options, ReadOptions, WriteOptions},
    snapshots::{Snapshot, Snapshots},
};
use std::path::Path;
use std::sync::Arc;

/// DB provides thread-safe access to LevelDB API.
//
//  This also provides an abstraction layer which enables
//  us to provide an API that is somewhat backwards compatible
//  with rusty-leveldb.  For example, our get() and put()
//  do not require ReadOptions and WriteOptions param.
//
//  Do not add any public (mutable) fields to this struct.
#[derive(Debug, Clone)]
pub struct DB {
    // note: these must be private and unchanged after creation.
    db: Arc<Database>, // Send + Sync.  Arc is so we can derive Clone.
    path: std::path::PathBuf,
    destroy_db_on_drop: bool,
    read_options: ReadOptions,
    write_options: WriteOptions,
}

impl DB {
    /// Open a new database
    ///
    /// If the database is missing, the behaviour depends on `options.create_if_missing`.
    /// The database will be created using the settings given in `options`.
    #[inline]
    pub fn open(name: &Path, options: &Options) -> Result<Self, DbError> {
        let db = Database::open(name, options)?;
        Ok(Self {
            db: Arc::new(db),
            path: name.into(),
            destroy_db_on_drop: false,
            read_options: ReadOptions::new(),
            write_options: WriteOptions::new(),
        })
    }

    /// Open a new database
    ///
    /// If the database is missing, the behaviour depends on `options.create_if_missing`.
    /// The database will be created using the settings given in `options`.
    #[inline]
    pub fn open_with_options(
        name: &Path,
        options: &Options,
        read_options: ReadOptions,
        write_options: WriteOptions,
    ) -> Result<Self, DbError> {
        let db = Database::open(name, options)?;
        Ok(Self {
            db: Arc::new(db),
            path: name.into(),
            destroy_db_on_drop: false,
            read_options,
            write_options,
        })
    }

    /// Open a new database with a custom comparator
    ///
    /// If the database is missing, the behaviour depends on `options.create_if_missing`.
    /// The database will be created using the settings given in `options`.
    ///
    /// The comparator must implement a total ordering over the keyspace.
    ///
    /// For keys that implement Ord, consider the `OrdComparator`.
    #[inline]
    pub fn open_with_comparator<C: Comparator>(
        name: &Path,
        options: &Options,
        comparator: C,
    ) -> Result<Self, DbError> {
        let db = Database::open_with_comparator(name, options, comparator)?;
        Ok(Self {
            db: Arc::new(db),
            path: name.into(),
            destroy_db_on_drop: false,
            read_options: ReadOptions::new(),
            write_options: WriteOptions::new(),
        })
    }

    /// Creates and opens a test database
    ///
    /// The database will be created in the system
    /// temp directory with prefix "test-db-" followed
    /// by a random string.
    ///
    /// if destroy_db_on_drop is true, the database on-disk
    /// files will be wiped when the DB struct is dropped.
    pub fn open_new_test_database(
        destroy_db_on_drop: bool,
        options: Option<Options>,
        read_options: Option<ReadOptions>,
        write_options: Option<WriteOptions>,
    ) -> Result<Self, DbError> {
        use rand::distributions::DistString;
        use rand_distr::Alphanumeric;

        let path = std::env::temp_dir().join(format!(
            "test-db-{}",
            Alphanumeric.sample_string(&mut rand::thread_rng(), 10)
        ));
        Self::open_test_database(
            &path,
            destroy_db_on_drop,
            options,
            read_options,
            write_options,
        )
    }

    /// Opens an existing (test?) database, with auto-destroy option.
    ///
    /// if destroy_db_on_drop is true, the database on-disk
    /// files will be wiped when the DB struct is dropped.
    /// This is usually useful only for unit-test purposes.
    pub fn open_test_database(
        path: &std::path::Path,
        destroy_db_on_drop: bool,
        options: Option<Options>,
        read_options: Option<ReadOptions>,
        write_options: Option<WriteOptions>,
    ) -> Result<Self, DbError> {
        let mut opt = options.unwrap_or_else(Options::new);
        let read_opt = read_options.unwrap_or_else(ReadOptions::new);
        let write_opt = write_options.unwrap_or_else(WriteOptions::new);

        opt.create_if_missing = true;
        opt.error_if_exists = false;

        let mut db = DB::open_with_options(path, &opt, read_opt, write_opt)?;
        db.destroy_db_on_drop = destroy_db_on_drop;
        Ok(db)
    }

    /// Set a key/val in the database
    #[inline]
    pub fn put(&self, key: &dyn IntoLevelDBKey, value: &[u8]) -> Result<(), DbError> {
        self.db.put(&self.write_options, key, value)
    }

    /// Set a key/val in the database, with key as bytes.
    #[inline]
    pub fn put_u8(&self, key: &[u8], value: &[u8]) -> Result<(), DbError> {
        self.db.put_u8(&self.write_options, key, value)
    }

    /// Get a value matching key from the database
    #[inline]
    pub fn get(&self, key: &dyn IntoLevelDBKey) -> Result<Option<Vec<u8>>, DbError> {
        self.db.get(&self.read_options, key)
    }

    /// Get a value matching key from the database, with key as bytes
    #[inline]
    pub fn get_u8(&self, key: &[u8]) -> Result<Option<Vec<u8>>, DbError> {
        self.db.get_u8(&self.read_options, key)
    }

    /// Delete an entry matching key from the database
    #[inline]
    pub fn delete(&self, key: &dyn IntoLevelDBKey) -> Result<(), DbError> {
        self.db.delete(&self.write_options, key)
    }

    /// Delete an entry matching key from the database, with key as bytes
    #[inline]
    pub fn delete_u8(&self, key: &[u8]) -> Result<(), DbError> {
        self.db.delete_u8(&self.write_options, key)
    }

    /// Write the WriteBatch to database atomically
    ///
    /// The sync flag forces filesystem sync operation eg fsync
    /// which will be slower than async writes, which are not
    /// guaranteed to complete. See leveldb Docs.
    pub fn write(&self, batch: &WriteBatch, sync: bool) -> Result<(), DbError> {
        const WO_SYNC: WriteOptions = WriteOptions { sync: true };
        const WO_NOSYNC: WriteOptions = WriteOptions { sync: false };

        self.db
            .write(if sync { &WO_SYNC } else { &WO_NOSYNC }, batch)
    }

    /// Write [`WriteBatch`] to database atomically
    ///
    /// Sync behavior will be determined by the WriteOptions
    /// supplied at `DB` creation.
    pub fn write_auto(&self, batch: &WriteBatch) -> Result<(), DbError> {
        self.db.write(&self.write_options, batch)
    }

    /// returns the directory path of the database files on disk.
    #[inline]
    pub fn path(&self) -> &std::path::PathBuf {
        &self.path
    }

    /// returns `destroy_db_on_drop` setting
    #[inline]
    pub fn destroy_db_on_drop(&self) -> bool {
        self.destroy_db_on_drop
    }

    /// Wipe the database files, if existing.
    fn destroy_db(&mut self) -> Result<(), std::io::Error> {
        match self.path.exists() {
            true => std::fs::remove_dir_all(&self.path),
            false => Ok(()),
        }
    }
}

impl Drop for DB {
    #[inline]
    fn drop(&mut self) {
        if self.destroy_db_on_drop {
            // note: we do not panic if the database directory
            // cannot be removed.  Perhaps revisit later.
            let _ = self.destroy_db();
        }
    }
}

// impl Batch for DB {
//     #[inline]
//     fn write(&self, options: &WriteOptions, batch: &WriteBatch) -> Result<(), DbError> {
//         self.db.write(options, batch)
//     }
// }

impl<'a> Compaction<'a> for DB {
    #[inline]
    fn compact(&self, start: &'a [u8], limit: &'a [u8]) {
        self.db.compact(start, limit)
    }
}

impl<'a> Iterable<'a> for DB {
    #[inline]
    fn iter(&'a self, options: &ReadOptions) -> Iterator<'a> {
        self.db.iter(options)
    }

    #[inline]
    fn keys_iter(&'a self, options: &ReadOptions) -> KeyIterator<'a> {
        self.db.keys_iter(options)
    }

    #[inline]
    fn value_iter(&'a self, options: &ReadOptions) -> ValueIterator<'a> {
        self.db.value_iter(options)
    }
}

impl Snapshots for DB {
    fn snapshot(&self) -> Snapshot {
        self.db.snapshot()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn level_db_close_and_reload() {
        // open new test database that will not be destroyed on close.
        let db = DB::open_new_test_database(false, None, None, None).unwrap();
        let db_path = db.path.clone();

        let key = "answer-to-everything";
        let val = vec![42];

        let _ = db.put(&key, &val);

        drop(db); // close the DB.

        assert!(db_path.exists());

        // open existing database that will be destroyed on close.
        let db2 = DB::open_test_database(&db_path, true, None, None, None).unwrap();

        let val2 = db2.get(&key).unwrap().unwrap();
        assert_eq!(val, val2);

        drop(db2); // close the DB.  db_path dir is auto removed.

        assert!(!db_path.exists());
    }
}
