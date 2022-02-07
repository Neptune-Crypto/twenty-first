pub trait Hasher<Value: AsBytes + Sized> {
    type Digest;

    fn new() -> Self;
    fn hash_one(&mut self, one: &Value) -> Self::Digest;
    fn hash_two(&mut self, one: &Value, two: &Value) -> Self::Digest;
    fn hash_many(&mut self, input: &[Value]) -> Self::Digest;
}

pub trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

impl AsBytes for blake3::Hash {
    fn as_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<Value> Hasher<Value> for blake3::Hasher
where
    Value: AsBytes,
{
    type Digest = blake3::Hash;

    fn new() -> Self {
        todo!()
    }

    fn hash_one(&mut self, one: &Value) -> Self::Digest {
        self.reset();
        self.update(one.as_bytes());
        self.finalize()
    }

    fn hash_two(&mut self, one: &Value, two: &Value) -> Self::Digest {
        self.reset();
        self.update(one.as_bytes());
        self.update(two.as_bytes());
        self.finalize()
    }

    fn hash_many(&mut self, input: &[Value]) -> blake3::Hash {
        self.reset();

        for value in input {
            self.update(value.as_bytes());
        }

        self.finalize()
    }
}
