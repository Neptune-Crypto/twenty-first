//! This module contains various configuration options. In general, the
//! configuration options impact performance only. The default configuration is
//! sane and should provide good performance for most users.
//!
//! Most configuration options can also be set via environment variables.
//! Generally, the environment variables take precedence over the options set
//! in this module.

use std::cell::RefCell;

use arbitrary::Arbitrary;

thread_local! {
    static CONFIG: RefCell<Config> = RefCell::new(Config::new());
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
struct Config {
    pub merkle_tree_parallelization_cutoff: MerkleTreeParallelizationCutoff,
}

impl Config {
    fn new() -> Self {
        let merkle_tree_parallelization_cutoff = MerkleTreeParallelizationCutoff::new(None);

        Self {
            merkle_tree_parallelization_cutoff,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
struct MerkleTreeParallelizationCutoff(usize);

impl MerkleTreeParallelizationCutoff {
    const ENV_VAR: &'static str = "TWENTY_FIRST_MERKLE_TREE_PARALLELIZATION_CUTOFF";
    const DEFAULT: usize = 512;
    const MINIMUM: usize = 2;

    /// Creates a new `MerkleTreeParallelizationCutoff` with the given value.
    /// Respects the precedence of the environment variable if set. Uses the
    /// default if no value is provided.
    fn new(config_value: Option<usize>) -> Self {
        let cutoff = std::env::var(Self::ENV_VAR)
            .ok()
            .and_then(|s| s.parse().ok())
            .or(config_value)
            .unwrap_or(Self::DEFAULT)
            .max(Self::MINIMUM);

        Self(cutoff)
    }
}

/// Sets the cutoff for parallelizing Merkle tree operations.
///
/// For example, if the cutoff is set to 512, then building a Merkle tree with
/// fewer than 512 leaves will be done sequentially. If the number of leaves is
/// 512 or greater, the tree will initially be built in parallel, until the
/// number of missing internal nodes is less than or equal to the cutoff, at
/// which point the remaining nodes will be built sequentially.
///
/// Can also be set via the environment variable
/// `TWENTY_FIRST_MERKLE_TREE_PARALLELIZATION_CUTOFF`. The environment variable
/// has higher precedence than this function.
///
/// The default is 512. The minimum is always 2.
pub fn set_merkle_tree_parallelization_cutoff(cutoff: usize) {
    let cutoff = MerkleTreeParallelizationCutoff::new(Some(cutoff));
    CONFIG.with(|c| c.borrow_mut().merkle_tree_parallelization_cutoff = cutoff);
}

pub(crate) fn merkle_tree_parallelization_cutoff() -> usize {
    CONFIG
        .with(|c| c.borrow().merkle_tree_parallelization_cutoff)
        .0
}
