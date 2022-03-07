use crate::util_types::simple_hasher::ToDigest;
use std::fmt::Debug;

#[derive(Debug, Clone, PartialEq)]
pub struct AppendProof<HashDigest>
where
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
{
    pub old_leaf_count: u128,
    pub old_peaks: Vec<HashDigest>,
    pub new_peaks: Vec<HashDigest>,
    // TODO: Add a verify method
}
