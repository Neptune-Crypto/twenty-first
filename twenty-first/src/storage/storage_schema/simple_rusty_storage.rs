use super::super::level_db::DB;
use super::enums::WriteOperation;
use super::{traits::StorageWriter, DbtSchema, RustyKey, RustyValue, SimpleRustyReader};
use leveldb::batch::WriteBatch;
use std::sync::Arc;

/// Database schema and tables logic for RustyLevelDB. You probably
/// want to implement your own storage class after this example so
/// that you can hardcode the schema in new(). But it is nevertheless
/// possible to use this struct and add to the schema.
pub struct SimpleRustyStorage {
    /// dynamic DB Schema.  (new tables may be added)
    pub schema: DbtSchema<RustyKey, RustyValue, SimpleRustyReader>,
}

impl StorageWriter<RustyKey, RustyValue> for SimpleRustyStorage {
    #[inline]
    fn persist(&self) {
        let write_batch = WriteBatch::new();
        for table in self.schema.tables.iter() {
            let operations = table.pull_queue();
            for op in operations {
                match op {
                    WriteOperation::Write(key, value) => write_batch.put(&key.0, &value.0),
                    WriteOperation::Delete(key) => write_batch.delete(&key.0),
                }
            }
        }

        self.db()
            .write(&write_batch, true)
            .expect("Could not persist to database.");
    }

    #[inline]
    fn restore_or_new(&self) {
        for table in self.schema.tables.iter() {
            table.restore_or_new();
        }
    }
}

impl SimpleRustyStorage {
    /// Create a new SimpleRustyStorage
    #[inline]
    pub fn new(db: DB) -> Self {
        let schema = DbtSchema::<RustyKey, RustyValue, SimpleRustyReader> {
            tables: Vec::new(),
            reader: Arc::new(SimpleRustyReader { db }),
        };
        Self { schema }
    }

    #[inline]
    pub(crate) fn db(&self) -> &DB {
        &self.schema.reader.db
    }
}
