use serde::{de::DeserializeOwned, Serialize};

pub type Index = u64;

pub trait StorageVec<T> {
    /// check if collection is empty
    fn is_empty(&self) -> bool;

    /// get collection length
    fn len(&self) -> Index;

    /// get single element at index
    fn get(&self, index: Index) -> T;

    /// get multiple elements matching indices
    fn get_many(&self, indices: &[Index]) -> Vec<T>;

    /// get all elements
    fn get_all(&self) -> Vec<T>;

    /// set a single element.
    fn set(&mut self, index: Index, value: T);

    /// set multiple elements.
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>);

    /// set all elements with a simple list of values in an array or Vec.
    ///
    /// calls ::set_many() internally.
    ///
    /// panics if input length does not match target length.
    ///
    /// note: casts the array's indexes from usize to Index.
    fn set_all(&mut self, vals: &[T])
    where
        T: Clone,
    {
        assert!(
            vals.len() as Index == self.len(),
            "size-mismatch.  input has {} elements and target has {} elements.",
            vals.len(),
            self.len(),
        );

        self.set_many(
            vals.iter()
                .enumerate()
                .map(|(i, v)| (i as Index, v.clone())),
        );
    }

    /// pop an element from end of collection
    fn pop(&mut self) -> Option<T>;

    /// push an element to end of collection
    fn push(&mut self, value: T);
}

pub enum WriteElement<T: Serialize + DeserializeOwned> {
    OverWrite((Index, T)),
    Push(T),
    Pop,
}
