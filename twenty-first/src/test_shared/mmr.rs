use std::sync::{Arc, Mutex};

use rusty_leveldb::DB;

use crate::shared_math::digest::Digest;
use crate::util_types::storage_vec::RustyLevelDbVec;
use crate::util_types::{algebraic_hasher::AlgebraicHasher, mmr::archival_mmr::ArchivalMmr};

/// Return an empty in-memory archival MMR for testing purposes.
/// Does *not* have a unique ID, so you can't expect multiple of these
/// instances to behave independently unless you understand the
/// underlying data structure.
pub fn get_empty_rustyleveldb_ammr<H: AlgebraicHasher>() -> ArchivalMmr<H, RustyLevelDbVec<Digest>>
{
    let opt = rusty_leveldb::in_memory();
    let db = DB::open("mydatabase", opt).unwrap();
    let db = Arc::new(Mutex::new(db));
    let pv = RustyLevelDbVec::new(db, 0, "in-memory AMMR for unit tests");
    ArchivalMmr::new(pv)
}

pub fn get_rustyleveldb_ammr_from_digests<H>(
    digests: Vec<Digest>,
) -> ArchivalMmr<H, RustyLevelDbVec<Digest>>
where
    H: AlgebraicHasher,
{
    let mut ammr = get_empty_rustyleveldb_ammr();
    for digest in digests {
        ammr.append_raw(digest);
    }

    ammr
}
