//! The inverse of [`Tip5`].
//!
//! That is, [`InverseTip5::inv_permutation`] computes the inverse of
//! [`Tip5::permutation`], and each of its steps computes the
//! inverse of the corresponding step in `Tip5`.
//!
//! Computing (partial) inverses is useful for constructing initial states
//! that lead to (in some way) “interesting” internal states before some step.

// This is a test-only module; it should not influence code coverage.
#![cfg_attr(coverage_nightly, coverage(off))]

use super::*;

/// The inverse of the [`LOOKUP_TABLE`].
const INV_LOOKUP_TABLE: [u8; 256] = [
    0, 248, 146, 63, 209, 108, 39, 1, 20, 118, 139, 155, 99, 193, 29, 240, 222, 88, 170, 75, 225,
    164, 94, 36, 152, 227, 2, 246, 51, 16, 117, 127, 19, 14, 175, 58, 153, 173, 50, 162, 250, 247,
    223, 221, 40, 10, 90, 217, 57, 60, 141, 231, 163, 232, 101, 79, 207, 97, 132, 120, 142, 83, 68,
    3, 182, 96, 210, 136, 86, 133, 244, 41, 66, 52, 131, 149, 202, 105, 238, 237, 183, 47, 242, 71,
    55, 6, 190, 22, 61, 185, 144, 168, 126, 42, 186, 54, 104, 49, 112, 25, 76, 148, 81, 199, 171,
    229, 27, 191, 166, 211, 21, 130, 78, 134, 160, 243, 43, 220, 181, 59, 67, 140, 145, 98, 4, 218,
    178, 224, 31, 77, 37, 251, 157, 110, 115, 188, 196, 74, 35, 212, 12, 95, 121, 177, 125, 234,
    44, 89, 64, 228, 26, 84, 56, 174, 107, 179, 230, 143, 206, 151, 201, 69, 213, 129, 87, 111, 70,
    194, 233, 65, 249, 200, 184, 13, 208, 72, 18, 17, 150, 53, 106, 124, 203, 189, 214, 11, 122,
    169, 119, 45, 159, 73, 252, 187, 172, 113, 135, 123, 158, 48, 176, 154, 23, 92, 24, 114, 195,
    198, 38, 165, 245, 215, 34, 32, 8, 5, 93, 205, 82, 102, 197, 80, 241, 236, 128, 138, 239, 204,
    9, 253, 28, 103, 219, 161, 91, 30, 180, 85, 167, 33, 15, 226, 62, 156, 100, 116, 137, 235, 254,
    216, 147, 46, 192, 109, 7, 255,
];

/// The exponent to compute the 7th root of a [`BFieldElement`].
///
/// In particular, `INV_POWER_MAP_EXPONENT`·7 == 1 (mod p - 1).
const INV_POWER_MAP_EXPONENT: u64 = 10_540_996_611_094_048_183;

/// The defining, first column of the inverse of the (circulant)
/// [MDS matrix](MDS_MATRIX_FIRST_COLUMN).
pub const INV_MDS_MATRIX_FIRST_COLUMN: [BFieldElement; STATE_SIZE] = [
    BFieldElement::new(0xdcd4bbcc7abbbdc8),
    BFieldElement::new(0x322fc8fee5105727),
    BFieldElement::new(0xcd0e9d3bc1c39e5d),
    BFieldElement::new(0x60387df95dfa27a9),
    BFieldElement::new(0xdb7fea1eb517bee0),
    BFieldElement::new(0x45fd375ce8aed794),
    BFieldElement::new(0xaa4e2f867bca82cf),
    BFieldElement::new(0x20066ac8bcaa222c),
    BFieldElement::new(0xf514d7d7271c3511),
    BFieldElement::new(0x93184445c1c397cf),
    BFieldElement::new(0xb75a4aaeb010289f),
    BFieldElement::new(0xb618d7b53a4f0b93),
    BFieldElement::new(0xce0e2726ff0a50f6),
    BFieldElement::new(0x351828e9b5eb5dbe),
    BFieldElement::new(0xf375941d3eef26e8),
    BFieldElement::new(0x1c158a0f5c11fe81),
];

pub struct InverseTip5 {
    pub state: [BFieldElement; STATE_SIZE],
}

impl InverseTip5 {
    pub fn inv_permutation(&mut self) {
        for i in (0..NUM_ROUNDS).rev() {
            self.inv_round(i);
        }
    }

    pub fn inv_round(&mut self, round_index: usize) {
        self.subtract_constants(round_index);
        self.inv_mds_matrix_mul();
        self.inv_sbox_layer();
    }

    pub fn subtract_constants(&mut self, round_index: usize) {
        let round_offset = round_index * STATE_SIZE;
        for i in 0..STATE_SIZE {
            self.state[i] -= ROUND_CONSTANTS[round_offset + i];
        }
    }

    pub fn inv_mds_matrix_mul(&mut self) {
        let mut new_state = [BFieldElement::ZERO; STATE_SIZE];

        for (row_idx, new_elem) in new_state.iter_mut().enumerate() {
            for col_idx in 0..STATE_SIZE {
                // see `NaiveTip5::mds_matrix_mul` for details
                let mds_matrix_idx = (STATE_SIZE + row_idx - col_idx) % STATE_SIZE;
                let matrix_element = INV_MDS_MATRIX_FIRST_COLUMN[mds_matrix_idx];
                *new_elem += matrix_element * self.state[col_idx];
            }
        }

        self.state = new_state;
    }

    pub fn inv_sbox_layer(&mut self) {
        for i in 0..NUM_SPLIT_AND_LOOKUP {
            Self::split_and_inv_lookup(&mut self.state[i]);
        }

        for i in NUM_SPLIT_AND_LOOKUP..STATE_SIZE {
            self.state[i] = self.state[i].mod_pow(INV_POWER_MAP_EXPONENT);
        }
    }

    fn split_and_inv_lookup(element: &mut BFieldElement) {
        let bytes = element.raw_bytes().map(|b| INV_LOOKUP_TABLE[b as usize]);
        *element = BFieldElement::from_raw_bytes(&bytes);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::tests::proptest;
    use crate::tests::test;

    #[macro_rules_attr::apply(test)]
    fn inv_lookup_table_is_inv_of_lookup_table() {
        for (idx, looked_up) in LOOKUP_TABLE.into_iter().enumerate() {
            let idx_again = INV_LOOKUP_TABLE[looked_up as usize] as usize;
            assert_eq!(idx, idx_again);
        }
    }

    #[macro_rules_attr::apply(test)]
    fn inv_power_map_exponent_is_bezout_coefficient_of_7() {
        let one = (INV_POWER_MAP_EXPONENT as u128 * 7) % (BFieldElement::P as u128 - 1);
        assert_eq!(1, one);
    }

    #[macro_rules_attr::apply(proptest)]
    fn inv_power_map_exponent_computes_the_correct_root(bfe: BFieldElement) {
        let bfe_again = bfe.mod_pow(7).mod_pow(INV_POWER_MAP_EXPONENT);
        prop_assert_eq!(bfe, bfe_again);
    }

    #[macro_rules_attr::apply(proptest)]
    fn sbox_layer(mut tip5: Tip5) {
        let orig_state = tip5.state;
        tip5.sbox_layer();
        let mut inverse_tip5 = InverseTip5 { state: tip5.state };
        inverse_tip5.inv_sbox_layer();

        prop_assert_eq!(orig_state, inverse_tip5.state);
    }

    #[macro_rules_attr::apply(proptest)]
    fn mds_matrix_mul(mut tip5: Tip5) {
        let orig_state = tip5.state;
        tip5.mds_generated();
        let mut inverse_tip5 = InverseTip5 { state: tip5.state };
        inverse_tip5.inv_mds_matrix_mul();

        prop_assert_eq!(orig_state, inverse_tip5.state);
    }

    #[macro_rules_attr::apply(proptest)]
    fn round(mut tip5: Tip5, #[strategy(0..NUM_ROUNDS)] round_idx: usize) {
        let orig_state = tip5.state;
        tip5.round(round_idx);
        let mut inverse_tip5 = InverseTip5 { state: tip5.state };
        inverse_tip5.inv_round(round_idx);

        prop_assert_eq!(orig_state, inverse_tip5.state);
    }

    #[macro_rules_attr::apply(proptest)]
    fn permutation(mut tip5: Tip5) {
        let orig_state = tip5.state;
        tip5.permutation();
        let mut inverse_tip5 = InverseTip5 { state: tip5.state };
        inverse_tip5.inv_permutation();

        prop_assert_eq!(orig_state, inverse_tip5.state);
    }
}
