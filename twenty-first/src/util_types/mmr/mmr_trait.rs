use super::mmr_accumulator::MmrAccumulator;
use super::mmr_membership_proof::MmrMembershipProof;
use crate::math::digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

/// A definition of a mutation of a leaf in an MMR
#[derive(Debug, Clone)]
pub struct LeafMutation<'a, H: AlgebraicHasher> {
    /// The leaf-index of the leaf being mutated. If the MMR is viewed as a
    /// commitment to a list, then this is simply the (0-indexed) list-index
    ///  into that list.
    pub leaf_index: u64,

    /// The new leaf value, after the mutation has been applied.
    pub new_leaf: Digest,

    /// MMR membership proof (authentication path) both before *and* after the
    /// leaf has been mutated. An authentication path is a commitment to all
    /// other leafs in the Merkle tree than the one it is a membership proof
    /// for.
    pub membership_proof: &'a MmrMembershipProof<H>,
}

impl<'a, H: AlgebraicHasher> LeafMutation<'a, H> {
    pub fn new(
        leaf_index: u64,
        new_leaf: Digest,
        membership_proof: &'a MmrMembershipProof<H>,
    ) -> Self {
        Self {
            leaf_index,
            new_leaf,
            membership_proof,
        }
    }

    /// Returns the node indices into the MMR of the nodes that are mutated by
    /// this leaf-mutation.
    pub fn affected_node_indices(&self) -> Vec<u64> {
        self.membership_proof
            .get_direct_path_indices(self.leaf_index)
    }
}

pub trait Mmr<H: AlgebraicHasher> {
    /// Create a new MMR instanc from a list of hash digests. The supplied digests
    /// are the leaves of the MMR.

    // constructors cannot be part of the interface since the archival version requires a
    // database which we want the caller to create, and the accumulator does not need a
    // constructor.

    /// Calculate a single hash digest committing to the entire MMR.
    fn bag_peaks(&self) -> Digest;

    /// Returns the peaks of the MMR, which are roots of the Merkle trees that constitute
    /// the MMR
    fn peaks(&self) -> Vec<Digest>;

    /// Returns `true` iff the MMR has no leaves
    fn is_empty(&self) -> bool;

    /// Returns the number of leaves in the MMR
    fn num_leaves(&self) -> u64;

    /// Append a hash digest to the MMR
    fn append(&mut self, new_leaf: Digest) -> MmrMembershipProof<H>;

    /// Mutate an existing leaf. It is the caller's responsibility that the
    /// membership proof is valid. If the membership proof is wrong, the MMR
    /// will end up in a broken state.
    fn mutate_leaf(&mut self, leaf_mutation: LeafMutation<H>);

    /// Batch mutate an MMR while updating a list of membership proofs. Returns the indices of the
    /// membership proofs that have changed as a result of this operation.
    fn batch_mutate_leaf_and_update_mps(
        &mut self,
        membership_proofs: &mut [&mut MmrMembershipProof<H>],
        membership_proof_leaf_indices: &[u64],
        mutation_data: Vec<LeafMutation<H>>,
    ) -> Vec<usize>;

    /// Returns true if a list of leaf mutations and a list of appends results in the expected
    /// `new_peaks`.
    fn verify_batch_update(
        &self,
        new_peaks: &[Digest],
        appended_leafs: &[Digest],
        leaf_mutations: Vec<LeafMutation<H>>,
    ) -> bool;

    /// Return an MMR accumulator containing only peaks and leaf count
    fn to_accumulator(&self) -> MmrAccumulator<H>;
}
