//! A naïve implementation of [Tip5] for testing. Contains no performance
//! optimizations and is therefore easier to read and understand.
//!
//! Since we (currently) use conditional compilation to get _either_ the
//! non-AVX-512 version _or_ the AVX-512 version, there's no (easy) way to
//! compare the two implementations for equivalence. Hence, this naïve
//! implementation: it always exists, and compares to whichever of the two
//! optimized versions is part of the binary.
//!
//! In the future, we might choose to use [dynamic dispatch][dd]. Then the
//! different versions can coexist on AVX-512 capable machines, which might
//! render this module obsolete.
//!
//! [dd]: https://doc.rust-lang.org/std/arch/#dynamic-cpu-feature-detection

// This is a test-only module; it should not influence code coverage.
#![cfg_attr(coverage_nightly, coverage(off))]

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaiveTip5 {
    pub state: [BFieldElement; STATE_SIZE],
}

impl NaiveTip5 {
    pub fn permutation(&mut self) {
        for i in 0..NUM_ROUNDS {
            self.round(i);
        }
    }

    fn round(&mut self, round_index: usize) {
        self.sbox_layer();
        self.mds_matrix_mul();
        self.add_constants(round_index);
    }

    fn sbox_layer(&mut self) {
        for i in 0..NUM_SPLIT_AND_LOOKUP {
            Self::split_and_lookup(&mut self.state[i]);
        }

        for i in NUM_SPLIT_AND_LOOKUP..STATE_SIZE {
            self.state[i] = self.state[i].mod_pow(7);
        }
    }

    fn split_and_lookup(element: &mut BFieldElement) {
        let bytes = element.raw_bytes().map(|byte| LOOKUP_TABLE[byte as usize]);
        *element = BFieldElement::from_raw_bytes(&bytes);
    }

    fn mds_matrix_mul(&mut self) {
        let mut new_state = [BFieldElement::ZERO; STATE_SIZE];

        for (row_idx, new_elem) in new_state.iter_mut().enumerate() {
            for col_idx in 0..STATE_SIZE {
                // See <https://en.wikipedia.org/wiki/Circulant_matrix>
                // The initial summand `STATE_SIZE` only prevents overflows.
                let mds_matrix_idx = (STATE_SIZE + row_idx - col_idx) % STATE_SIZE;
                let matrix_element = BFieldElement::from(MDS_MATRIX_FIRST_COLUMN[mds_matrix_idx]);
                *new_elem += matrix_element * self.state[col_idx];
            }
        }

        self.state = new_state;
    }

    fn add_constants(&mut self, round_index: usize) {
        let round_offset = round_index * STATE_SIZE;
        for i in 0..STATE_SIZE {
            self.state[i] += ROUND_CONSTANTS[round_offset + i];
        }
    }
}

impl PartialEq<Tip5> for NaiveTip5 {
    fn eq(&self, other: &Tip5) -> bool {
        self.state == other.state
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::tests::proptest;

    /// The entire point of this module: test the naïve against the optimized
    /// Tip5 implementation.
    #[macro_rules_attr::apply(proptest)]
    fn tip5_corresponds_to_naive_tip5(
        state: [BFieldElement; STATE_SIZE],
        #[strategy(0..NUM_ROUNDS)] round_no: usize,
    ) {
        let mut naive = NaiveTip5 { state };
        let mut real = Tip5 { state };

        naive.round(round_no);
        real.round(round_no);

        prop_assert_eq!(naive, real);
    }
}
