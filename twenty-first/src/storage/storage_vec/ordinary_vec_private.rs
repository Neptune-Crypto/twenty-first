use super::{traits::*, Index};

#[derive(Debug, Clone, Default)]
pub(crate) struct OrdinaryVecPrivate<T>(pub(super) Vec<T>);

impl<T: Clone> StorageVecReads<T> for OrdinaryVecPrivate<T> {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn len(&self) -> Index {
        self.0.len() as Index
    }

    #[inline]
    fn get(&self, index: Index) -> T {
        self.0.get(index as usize).unwrap().clone()
    }

    fn many_iter(
        &self,
        _indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        unreachable!()
    }

    fn many_iter_values(
        &self,
        _indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = T> + '_> {
        unreachable!()
    }
}

impl<T: Clone> StorageVecMutableWrites<T> for OrdinaryVecPrivate<T> {
    #[inline]
    fn set(&mut self, index: Index, value: T) {
        self.0[index as usize] = value;
    }

    #[inline]
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        for (key, val) in key_vals.into_iter() {
            self.set(key, val);
        }
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    #[inline]
    fn push(&mut self, value: T) {
        self.0.push(value);
    }

    #[inline]
    fn clear(&mut self) {
        self.0.clear();
    }
}
