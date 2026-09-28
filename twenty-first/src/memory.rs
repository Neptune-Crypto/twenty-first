//! Helpers for dealing with large amounts of memory.

/// Advise the operating system to back the given memory with huge pages,
/// where supported. On other systems, this does nothing.
///
/// Large arrays of field elements are often accessed in patterns that touch
/// many pages at once. With the default page size of 4 KiB, both faulting in
/// freshly allocated pages and the subsequent address translations can become
/// a significant part of the runtime, especially on machines with many cores.
/// Huge pages reduce both by orders of magnitude. Advising the operating
/// system is cheap, and a no-op for regions smaller than a huge page.
///
/// Typical use is on the [spare capacity](Vec::spare_capacity_mut) of a
/// freshly allocated vector, before it is filled.
pub fn advise_huge_pages<T>(memory: &mut [T]) {
    #[cfg(target_os = "linux")]
    {
        const HUGE_PAGE_SIZE: usize = 2 << 20;
        const PAGE_SIZE: usize = 4 << 10;

        let start = memory.as_mut_ptr() as usize;
        let end = start + std::mem::size_of_val(memory);
        let aligned_start = start.next_multiple_of(PAGE_SIZE);
        let aligned_end = end & !(PAGE_SIZE - 1);
        if aligned_end < aligned_start + HUGE_PAGE_SIZE {
            return;
        }

        let region = aligned_start as *mut libc::c_void;
        let region_len = aligned_end - aligned_start;
        // SAFETY: The region lies within `memory`, which is exclusively
        // borrowed. The advice does not alter the memory's contents or its
        // mapping; it only informs the kernel's paging decisions. Failure is
        // harmless and hence ignored.
        let _ = unsafe { libc::madvise(region, region_len, libc::MADV_HUGEPAGE) };
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = memory;
    }
}

/// A vector with the given capacity, [advised](advise_huge_pages) to be backed
/// by huge pages.
pub fn vec_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut vec = Vec::with_capacity(capacity);
    advise_huge_pages(vec.spare_capacity_mut());
    vec
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::tests::test;

    #[macro_rules_attr::apply(test)]
    fn advising_huge_pages_is_harmless() {
        for len in [0, 1, 4 << 10, (2 << 20) + 17, 8 << 20] {
            let mut memory = vec_with_capacity::<u8>(len);
            advise_huge_pages(memory.spare_capacity_mut());
            memory.resize(len, 42);
            assert!(memory.iter().all(|&x| x == 42));
        }
    }
}
