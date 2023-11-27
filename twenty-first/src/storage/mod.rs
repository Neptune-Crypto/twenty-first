#![warn(missing_docs)]
#![warn(rustdoc::unescaped_backticks)]
#![warn(rustdoc::broken_intra_doc_links)]

//! Thread-safe collection types backed by levelDB.
//!
//! In particular:
//!  - [`storage_vec::RustyLevelDbVec`] provides a database-backed Vec with
//!    read/write cache and atomic writes.
//!  - [`storage_schema::SimpleRustyStorage`] provides atomic DB writes across
//!    any number of `DbtVec` or `DbtSingleton` "tables"
//!  - [`database_array::DatabaseArray`] and [`database_vector::DatabaseVector`] provide uncached
//!    and non-atomic writes.
//!  - [`level_db::DB`] provides direct access to the LevelDB API.

pub mod database_array;
pub mod database_vector;
pub mod level_db;
pub mod storage_schema;
pub mod storage_vec;

mod utils;
