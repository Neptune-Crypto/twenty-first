use super::super::storage_vec::{traits::*, Index};
use super::{traits::StorageReader, VecWriteOperation};
use itertools::Itertools;
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    sync::Arc,
};

// note: no locking is required in `DbtVecPrivate` because locking
// is performed in the `DbtVec` public wrapper.
pub struct DbtVecPrivate<ParentKey, ParentValue, Index, T> {
    pub(super) reader: Arc<dyn StorageReader<ParentKey, ParentValue> + Send + Sync>,
    pub(super) current_length: Option<Index>,
    pub(super) key_prefix: u8,
    pub(super) write_queue: VecDeque<VecWriteOperation<Index, T>>,
    pub(super) cache: HashMap<Index, T>,
    pub(super) name: String,
}

impl<ParentKey, ParentValue, T> DbtVecPrivate<ParentKey, ParentValue, Index, T>
where
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    ParentKey: From<Index>,
    Index: From<ParentValue> + From<u64> + Clone,
    T: Clone,
{
    // Return the key of ParentKey type used to store the length of the vector
    #[inline]
    pub(super) fn get_length_key(key_prefix: u8) -> ParentKey {
        let const_length_key: ParentKey = 0u8.into();
        let key_prefix_key: ParentKey = key_prefix.into();
        (key_prefix_key, const_length_key).into()
    }

    /// Return the length at the last write to disk
    #[inline]
    pub(super) fn persisted_length(&self) -> Option<Index> {
        self.reader
            .get(Self::get_length_key(self.key_prefix))
            .map(|v| v.into())
    }

    /// Return the key of ParentKey type used to store the element at a given index of Index type
    #[inline]
    pub(super) fn get_index_key(&self, index: Index) -> ParentKey {
        let key_prefix_key: ParentKey = self.key_prefix.into();
        let index_key: ParentKey = index.into();
        (key_prefix_key, index_key).into()
    }

    #[inline]
    pub(crate) fn new(
        reader: Arc<dyn StorageReader<ParentKey, ParentValue> + Send + Sync>,
        key_prefix: u8,
        name: &str,
    ) -> Self {
        let length = None;
        let cache = HashMap::new();
        Self {
            key_prefix,
            reader,
            write_queue: VecDeque::default(),
            current_length: length,
            cache,
            name: name.to_string(),
        }
    }

    #[inline]
    fn write_op_overwrite(&mut self, index: Index, value: T) {
        self.cache.insert(index, value.clone());

        // note: benchmarks have revealed this code to slow down
        //       set operations by about 7x, eg 10us to 70us.
        //       Disabling for now.
        //
        // if let Some(_old_val) = self.cache.insert(index, value.clone()) {
        // If cache entry exists, we remove any corresponding
        // OverWrite ops in the `write_queue` to reduce disk IO.

        // logic: retain all ops that are not overwrite, and
        // overwrite ops that do not have an index matching cache_index.
        // self.write_queue.retain(|op| match op {
        //     VecWriteOperation::OverWrite((i, _)) => *i != index,
        //     _ => true,
        // })
        // }

        self.write_queue
            .push_back(VecWriteOperation::OverWrite((index, value)));
    }
}

impl<ParentKey, ParentValue, T> StorageVecReads<T>
    for DbtVecPrivate<ParentKey, ParentValue, Index, T>
where
    ParentKey: From<Index>,
    ParentValue: From<T>,
    T: Clone + From<ParentValue> + Debug,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    Index: From<ParentValue> + From<u64>,
{
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn len(&self) -> Index {
        self.current_length
            .unwrap_or_else(|| self.persisted_length().unwrap_or(0))
    }

    #[inline]
    fn get(&self, index: Index) -> T {
        // Disallow getting values out-of-bounds

        assert!(
            index < self.len(),
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self.len(),
            self.name
        );

        // try cache first
        if self.cache.contains_key(&index) {
            return self
                .cache
                .get(&index)
                .expect("there should be some value")
                .clone();
        }

        // then try persistent storage
        let key: ParentKey = self.get_index_key(index);
        let val = self.reader.get(key).unwrap_or_else(|| {
            panic!(
                "Element with index {index} does not exist in {}. This should not happen",
                self.name
            )
        });
        val.into()
    }

    // this fn is here to satisfy the trait but is actually
    // implemented by DbtVec
    // todo
    // fn iter_keys<'a>(&'a self) -> Box<dyn Iterator<Item = Index> + '_> {
    //     unreachable!()
    // }

    // this fn is here to satisfy the trait but is actually
    // implemented by DbtVec
    fn many_iter<'a>(
        &'a self,
        _indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = (Index, T)> + '_> {
        unreachable!()
    }

    // this fn is here to satisfy the trait but is actually
    // implemented by DbtVec
    fn many_iter_values<'a>(
        &'a self,
        _indices: impl IntoIterator<Item = Index> + 'static,
    ) -> Box<dyn Iterator<Item = T> + '_> {
        unreachable!()
    }

    /// Fetch multiple elements from a `DbtVec` and return the elements matching the order
    /// of the input indices.
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        fn sort_to_match_requested_index_order<T>(indexed_elements: HashMap<usize, T>) -> Vec<T> {
            let mut elements = indexed_elements.into_iter().collect_vec();
            elements.sort_unstable_by_key(|&(index_position, _)| index_position);
            elements.into_iter().map(|(_, element)| element).collect()
        }

        let max_index = match indices.iter().max() {
            Some(i) => i,
            None => return vec![],
        };

        assert!(
            *max_index < self.len(),
            "Out-of-bounds. Got index {max_index} but length was {}. persisted vector name: {}",
            self.len(),
            self.name
        );

        let (indices_of_elements_in_cache, indices_of_elements_not_in_cache): (Vec<_>, Vec<_>) =
            indices
                .iter()
                .copied()
                .enumerate()
                .partition(|&(_, index)| self.cache.contains_key(&index));

        let mut fetched_elements = HashMap::with_capacity(indices.len());
        for (index_position, index) in indices_of_elements_in_cache {
            let value = self
                .cache
                .get(&index)
                .expect("there should be some value")
                .clone();
            fetched_elements.insert(index_position, value);
        }

        let no_need_to_lock_database = indices_of_elements_not_in_cache.is_empty();
        if no_need_to_lock_database {
            return sort_to_match_requested_index_order(fetched_elements);
        }

        let keys_for_indices_not_in_cache = indices_of_elements_not_in_cache
            .iter()
            .map(|&(_, index)| self.get_index_key(index))
            .collect_vec();
        let elements_fetched_from_db = self
            .reader
            .get_many(&keys_for_indices_not_in_cache)
            .into_iter()
            .map(|x| x.expect("there should be some value").into());

        let indexed_fetched_elements_from_db = indices_of_elements_not_in_cache
            .iter()
            .map(|&(index_position, _)| index_position)
            .zip_eq(elements_fetched_from_db);
        fetched_elements.extend(indexed_fetched_elements_from_db);

        sort_to_match_requested_index_order(fetched_elements)
    }

    /// Return all stored elements in a vector, whose index matches the StorageVec's.
    /// It's the caller's responsibility that there is enough memory to store all elements.
    fn get_all(&self) -> Vec<T> {
        let (indices_of_elements_in_cache, indices_of_elements_not_in_cache): (Vec<_>, Vec<_>) =
            (0..self.len()).partition(|index| self.cache.contains_key(index));

        let mut fetched_elements: Vec<Option<T>> = vec![None; self.len() as usize];
        for index in indices_of_elements_in_cache {
            let element = self.cache[&index].clone();
            fetched_elements[index as usize] = Some(element);
        }

        let no_need_to_lock_database = indices_of_elements_not_in_cache.is_empty();
        if no_need_to_lock_database {
            return fetched_elements
                .into_iter()
                .map(|x| x.expect("there should be some value"))
                .collect_vec();
        }

        let keys = indices_of_elements_not_in_cache
            .iter()
            .map(|x| self.get_index_key(*x))
            .collect_vec();
        let elements_fetched_from_db = self
            .reader
            .get_many(&keys)
            .into_iter()
            .map(|x| x.expect("there should be some value").into());
        let indexed_fetched_elements_from_db = indices_of_elements_not_in_cache
            .into_iter()
            .zip_eq(elements_fetched_from_db);
        for (index, element) in indexed_fetched_elements_from_db {
            fetched_elements[index as usize] = Some(element);
        }

        fetched_elements
            .into_iter()
            .map(|x| x.expect("there should be some value"))
            .collect_vec()
    }
}

impl<ParentKey, ParentValue, T> StorageVecMutableWrites<T>
    for DbtVecPrivate<ParentKey, ParentValue, Index, T>
where
    ParentKey: From<Index>,
    ParentValue: From<T>,
    T: Clone + From<ParentValue> + Debug,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    Index: From<ParentValue> + From<u64>,
{
    #[inline]
    fn set(&mut self, index: Index, value: T) {
        // Disallow setting values out-of-bounds

        assert!(
            index < self.len(),
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self.len(),
            self.name
        );

        self.write_op_overwrite(index, value);
    }

    /// set multiple elements.
    ///
    /// panics if key_vals contains an index not in the collection
    ///
    /// It is the caller's responsibility to ensure that index values are
    /// unique.  If not, the last value with the same index will win.
    /// For unordered collections such as HashMap, the behavior is undefined.
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        let self_len = self.len();

        for (index, value) in key_vals.into_iter() {
            assert!(
                index < self_len,
                "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
                self_len,
                self.name
            );

            self.write_op_overwrite(index, value);
        }
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        // If vector is empty, return None
        if self.is_empty() {
            return None;
        }

        // add to write queue
        self.write_queue.push_back(VecWriteOperation::Pop);

        // Update length
        *self
            .current_length
            .as_mut()
            .expect("there should be some value") -= 1;

        // try cache first
        let current_length = self.len();
        if self.cache.contains_key(&current_length) {
            self.cache.remove(&current_length)
        } else {
            // then try persistent storage
            let key = self.get_index_key(current_length);
            self.reader.get(key).map(|value| value.into())
        }
    }

    #[inline]
    fn push(&mut self, value: T) {
        // add to write queue
        self.write_queue
            .push_back(VecWriteOperation::Push(value.clone()));

        // record in cache
        let current_length = self.len();
        let _old_val = self.cache.insert(current_length, value);

        // note: we cannot naively remove any previous `Push` ops with
        // this value from the write_queue (to reduce disk i/o) because
        // there might be corresponding `Pop` op(s).

        // update length
        self.current_length = Some(current_length + 1);
    }
}
