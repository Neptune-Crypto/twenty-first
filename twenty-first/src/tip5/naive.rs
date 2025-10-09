//! A naïve implementation of [Tip5] for testing. Contains no performance
//! optimizations and is therefore easier to read and understand.

use proptest::prelude::*;
use test_strategy::proptest;

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NaiveTip5 {
    pub state: [BFieldElement; STATE_SIZE],
}

impl NaiveTip5 {
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
            let s_i = self.state[i];
            let s_i_to_the_3 = s_i * s_i * s_i;
            self.state[i] *= s_i_to_the_3 * s_i_to_the_3;
        }
    }

    fn split_and_lookup(element: &mut BFieldElement) {
        let mut bytes = element.raw_bytes();
        for i in 0..8 {
            bytes[i] = LOOKUP_TABLE[bytes[i] as usize];
        }

        *element = BFieldElement::from_raw_bytes(&bytes);
    }

    fn mds_matrix_mul(&mut self) {
        let mut new_state = [BFieldElement::ZERO; STATE_SIZE];

        for (row_idx, new_elem) in new_state.iter_mut().enumerate() {
            for col_idx in 0..STATE_SIZE {
                // See <https://en.wikipedia.org/wiki/Circulant_matrix>
                let first_col_index = (col_idx * (STATE_SIZE - 1) + row_idx) % STATE_SIZE;
                let matrix_element = BFieldElement::from(MDS_MATRIX_FIRST_COLUMN[first_col_index]);
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

/// The entire point of this module: test the naïve against the optimized Tip5
/// implementation.
#[proptest]
fn tip5_corresponds_to_naive_tip5(
    state: [BFieldElement; STATE_SIZE],
    #[strategy(0_usize..5)] round_no: usize,
) {
    let mut naive = NaiveTip5 { state };
    let mut real = Tip5 { state };

    naive.round(round_no);
    real.round(round_no);

    prop_assert_eq!(naive, real);
}
