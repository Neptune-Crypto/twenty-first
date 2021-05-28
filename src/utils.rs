use std::collections::HashSet;
use std::hash::Hash;

pub fn has_unique_elements<T>(iter: T) -> bool
where
    T: IntoIterator,
    T::Item: Eq + Hash,
{
    let mut uniq = HashSet::new();
    iter.into_iter().all(move |x| uniq.insert(x))
}

#[cfg(test)]
mod test_utils {
    use super::*;

    #[test]
    fn has_unique_elements_test() {
        let v = vec![10, 20, 30, 10, 50];
        assert!(!has_unique_elements(v));

        let v = vec![10, 20, 30, 40, 50];
        assert!(has_unique_elements(v));
    }
}
