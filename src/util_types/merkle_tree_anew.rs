use super::simple_hasher::Hasher;
use std::marker::PhantomData;

pub struct AuthenticationPath<H: Hasher> {
    digests: Vec<H::Digest>,
    _hasher: PhantomData<H>,
}

impl<H: Hasher> AuthenticationPath<H> {
    pub fn new(digests: Vec<H::Digest>) -> Self {
        todo!()
    }

    pub fn verify(&self, root_digest: H::Digest, leaf_digest: H::Digest) -> bool {
        todo!()
    }
}

pub struct MerkleMultiProof<H: Hasher> {
    partial_auth_paths: Vec<Vec<Option<H::Digest>>>,
}

impl<H: Hasher> MerkleMultiProof<H> {
    pub fn new(partial_auth_paths: Vec<Vec<Option<H::Digest>>>) -> Self {
        todo!()
    }

    pub fn verify(
        &self,
        root: H::Digest,
        leaf_digests: Vec<H::Digest>,
        leaf_indices: Vec<usize>,
    ) -> bool {
        todo!()
    }
}

pub struct MerkleTree<H: Hasher> {
    digests: Vec<H::Digest>,
    _hasher: PhantomData<H>,
}

pub trait MerkleTreeTrait<H: Hasher>
where
    Self: Sized,
{
    fn from_vec(digests: &[H::Digest]) -> Self;
    fn to_vec(&self) -> Vec<H::Digest>;

    fn get_leaf(&self, leaf_index: usize) -> Option<H::Digest>;
    fn get_root(&self) -> H::Digest;

    fn get_authentication_path(&self, leaf_index: usize) -> Option<AuthenticationPath<H>>;
    fn get_authentication_paths(&self, leaf_indices: &[usize]) -> Option<MerkleMultiProof<H>>;
}

impl<H: Hasher> MerkleTreeTrait<H> for MerkleTree<H> {
    fn from_vec(digests: &[H::Digest]) -> Self {
        todo!()
    }

    fn to_vec(&self) -> Vec<H::Digest> {
        todo!()
    }

    fn get_leaf(&self, leaf_index: usize) -> Option<H::Digest> {
        todo!()
    }

    fn get_root(&self) -> H::Digest {
        todo!()
    }

    fn get_authentication_path(&self, leaf_index: usize) -> Option<AuthenticationPath<H>> {
        todo!()
    }

    fn get_authentication_paths(&self, leaf_indices: &[usize]) -> Option<MerkleMultiProof<H>> {
        todo!()
    }
}
