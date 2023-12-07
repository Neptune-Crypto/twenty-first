use divan::Bencher;
use leveldb::options::{ReadOptions, WriteOptions};
use leveldb_sys::Compression;
// use twenty_first::leveldb::database::cache::Cache;
use twenty_first::leveldb::options::Options;
use twenty_first::storage::level_db::DB;
use twenty_first::util_types::storage_schema::{traits::*, DbtVec, SimpleRustyStorage};
use twenty_first::util_types::storage_vec::traits::*;

// These database bench tests are made with divan.
//
// See:
//  https://nikolaivazquez.com/blog/divan/
//  https://docs.rs/divan/0.1.0/divan/attr.bench.html
//  https://github.com/nvzqz/divan
//
//  Options for #[bench] attr:
//   https://docs.rs/divan/0.1.0/divan/attr.bench.html#options
//
//   name, crate, consts, types, sample_count, sample_size, threads
//   counters, min_time, max_time, skip_ext_time, ignore

fn main() {
    divan::main();
}

/// These settings affect DB performance and correctness.
///
/// Adjust and re-run the benchmarks to see effects.
///
/// Rust docs:  (basic)
///   https://docs.rs/rs-leveldb/0.1.5/leveldb/database/options/struct.Options.html
///
/// C++ docs:  (complete)
///   https://github.com/google/leveldb/blob/068d5ee1a3ac40dabd00d211d5013af44be55bea/include/leveldb/options.h
fn db_options() -> Option<Options> {
    Some(Options {
        // default: false
        create_if_missing: true,

        // default: false
        error_if_exists: true,

        // default: false
        paranoid_checks: false,

        // default: None  --> (4 * 1024 * 1024)
        write_buffer_size: None,

        // default: None   --> 1000
        max_open_files: None,

        // default: None   -->  4 * 1024
        block_size: None,

        // default: None   -->  16
        block_restart_interval: None,

        // default: Compression::No
        //      or: Compression::Snappy
        compression: Compression::No,

        // default: None   --> 8MB
        cache: None,
        // cache: Some(Cache::new(1024)),
        // note: tests put 128 bytes in each entry.
        // 100 entries = 12,800 bytes.
        // So Cache of 1024 bytes is 8% of total data set.
        // that seems reasonably realistic to get some
        // hits/misses.
    })
}

fn value() -> Vec<u8> {
    (0..127).collect()
}

fn create_test_dbtvec() -> (SimpleRustyStorage, DbtVec<Vec<u8>>) {
    let db = DB::open_new_test_database(
        true,
        db_options(),
        Some(ReadOptions {
            verify_checksums: false,
            fill_cache: false,
        }),
        Some(WriteOptions { sync: true }),
    )
    .unwrap();
    let mut storage = SimpleRustyStorage::new(db);
    let vec = storage.schema.new_vec::<Vec<u8>>("test-vector");
    (storage, vec)
}

mod write_100_entries {
    use super::*;

    // note: numbers > 100 make the sync_on_write::put() test really slow.
    const NUM_WRITE_ITEMS: u64 = 100;

    mod push {
        use super::*;

        fn push_impl(bencher: Bencher, persist: bool) {
            let (storage, vector) = create_test_dbtvec();

            bencher.bench_local(|| {
                for _i in 0..NUM_WRITE_ITEMS {
                    vector.push(value());
                }
                if persist {
                    storage.persist();
                }
            });
        }

        #[divan::bench]
        fn push(bencher: Bencher) {
            push_impl(bencher, false);
        }

        #[divan::bench]
        fn push_and_persist(bencher: Bencher) {
            push_impl(bencher, true);
        }
    }

    mod set {
        use super::*;

        fn set_impl(bencher: Bencher, persist: bool) {
            let (storage, vector) = create_test_dbtvec();

            for _i in 0..NUM_WRITE_ITEMS {
                vector.push(value());
            }

            bencher.bench_local(|| {
                for i in 0..NUM_WRITE_ITEMS {
                    vector.set(i, value());
                }

                if persist {
                    storage.persist();
                }
            });
        }

        #[divan::bench]
        fn set(bencher: Bencher) {
            set_impl(bencher, false);
        }

        #[divan::bench]
        fn set_and_persist(bencher: Bencher) {
            set_impl(bencher, true);
        }
    }

    mod set_many {
        use super::*;

        fn set_many_impl(bencher: Bencher, persist: bool) {
            let (storage, vector) = create_test_dbtvec();

            for _ in 0..NUM_WRITE_ITEMS {
                vector.push(vec![42]);
            }

            bencher.bench_local(|| {
                let values: Vec<_> = (0..NUM_WRITE_ITEMS).map(|i| (i, value())).collect();
                vector.set_many(values);
                if persist {
                    storage.persist();
                }
            });
        }

        #[divan::bench]
        fn set_many(bencher: Bencher) {
            set_many_impl(bencher, false);
        }

        #[divan::bench]
        fn set_many_and_persist(bencher: Bencher) {
            set_many_impl(bencher, true);
        }
    }

    mod pop {
        use super::*;

        fn pop_impl(bencher: Bencher, persist: bool) {
            let (storage, vector) = create_test_dbtvec();

            for _i in 0..NUM_WRITE_ITEMS {
                vector.push(value());
            }

            bencher.bench_local(|| {
                for _i in 0..NUM_WRITE_ITEMS {
                    vector.pop();
                }

                if persist {
                    storage.persist();
                }
            });
        }

        #[divan::bench]
        fn pop(bencher: Bencher) {
            pop_impl(bencher, false);
        }

        #[divan::bench]
        fn pop_and_persist(bencher: Bencher) {
            pop_impl(bencher, true);
        }
    }
}

mod read_100_entries {
    use super::*;

    const NUM_READ_ITEMS: u64 = 100;

    fn get_impl(bencher: Bencher, num_each: usize, persisted: bool) {
        let (storage, vector) = create_test_dbtvec();

        for _i in 0..NUM_READ_ITEMS {
            vector.push(value());
        }
        if persisted {
            storage.persist();
        }

        bencher.bench_local(|| {
            for i in 0..NUM_READ_ITEMS {
                for _j in 0..num_each {
                    let _ = vector.get(i);
                }
            }
        });
    }

    fn get_many_impl(bencher: Bencher, num_each: usize, persisted: bool) {
        let (storage, vector) = create_test_dbtvec();

        for _i in 0..NUM_READ_ITEMS {
            vector.push(value());
        }
        if persisted {
            storage.persist();
        }

        let indices: Vec<u64> = (0..NUM_READ_ITEMS).collect();
        bencher.bench_local(|| {
            for _j in 0..num_each {
                let _ = vector.get_many(&indices);
            }
        });
    }

    mod get_each_entry_1_time {
        use super::*;

        #[divan::bench]
        fn get_unpersisted(bencher: Bencher) {
            get_impl(bencher, 1, false);
        }

        #[divan::bench]
        fn get_persisted(bencher: Bencher) {
            get_impl(bencher, 1, true);
        }

        #[divan::bench]
        fn get_many_unpersisted(bencher: Bencher) {
            get_many_impl(bencher, 1, false);
        }

        #[divan::bench]
        fn get_many_persisted(bencher: Bencher) {
            get_many_impl(bencher, 1, true);
        }
    }

    mod get_each_entry_20_times {
        use super::*;

        #[divan::bench]
        fn get_unpersisted(bencher: Bencher) {
            get_impl(bencher, 20, false);
        }

        #[divan::bench]
        fn get_persisted(bencher: Bencher) {
            get_impl(bencher, 20, true);
        }

        #[divan::bench]
        fn get_many_unpersisted(bencher: Bencher) {
            get_many_impl(bencher, 20, false);
        }

        #[divan::bench]
        fn get_many_persisted(bencher: Bencher) {
            get_many_impl(bencher, 20, true);
        }
    }
}
