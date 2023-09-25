use crate::shared_math::digest::Digest;

pub mod mmr;

pub fn corrupt_digest(digest: Digest) -> Digest {
    let mut bad_elements = digest.values();
    bad_elements[0].increment();
    Digest::new(bad_elements)
}
