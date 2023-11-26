//! Thread-safe collection types backed by levelDB.
//!
//! In particular:
//!  - `RustyLeveldbVec` provides a database-backed Vec with
//!    read/write cache and atomic writes.
//!  - `SimpleRustyStorage provides atomic DB writes across
//!    any number of `DbtVec` or `DbtSingleton` "tables"
//!  - `DatabaseArray` and `DatabaseVec` provide uncached
//!    and non-atomic writes.
//!  - `DB` provides direct access to the LevelDB API.
pub mod database_array;
pub mod database_vector;
pub mod level_db;
pub mod storage_schema;
pub mod storage_vec;
