use rusty_leveldb::DB;

use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::{algebraic_hasher::AlgebraicHasher, mmr::archival_mmr::ArchivalMmr};

pub fn get_empty_archival_mmr<H: AlgebraicHasher>() -> ArchivalMmr<H> {
    let opt = rusty_leveldb::in_memory();
    let db = DB::open("mydatabase", opt).unwrap();
    ArchivalMmr::new(db)
}

pub fn get_archival_mmr_from_digests<H>(digests: Vec<Digest>) -> ArchivalMmr<H>
where
    H: AlgebraicHasher,
{
    let mut ammr = get_empty_archival_mmr();
    for digest in digests {
        ammr.append_raw(digest);
    }

    ammr
}
