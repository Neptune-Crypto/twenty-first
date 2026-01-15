//! The arithmetization-oriented, cryptographic hash function Tip5.
//!
//! This module contains the reference implementation of [“The Tip5 Hash
//! Function for Recursive STARKs”](https://eprint.iacr.org/2023/107.pdf), as
//! well as an [AVX-512](https://en.wikipedia.org/wiki/AVX-512)-accelerated
//! implementation, which subject to conditional compilation and compatible
//! hardware.

use std::hash::Hasher;

use arbitrary::Arbitrary;
pub use digest::Digest;
use get_size2::GetSize;
use itertools::Itertools;
use num_traits::ConstOne;
use num_traits::ConstZero;
use serde::Deserialize;
use serde::Serialize;

use crate::math::x_field_element::EXTENSION_DEGREE;
use crate::prelude::BFieldCodec;
use crate::prelude::BFieldElement;
use crate::prelude::XFieldElement;
use crate::util_types::sponge::Domain;
use crate::util_types::sponge::Sponge;

pub const STATE_SIZE: usize = 16;
pub const NUM_SPLIT_AND_LOOKUP: usize = 4;
pub const LOG2_STATE_SIZE: usize = 4;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;
pub const NUM_ROUNDS: usize = 5;

pub mod digest;

#[cfg(all(
    target_feature = "avx512ifma",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
mod avx512;
#[cfg(test)]
mod inverse;
#[cfg(test)]
mod naive;

/// The lookup table with a high algebraic degree used in the TIP-5 permutation. To verify its
/// correctness, see the test “lookup_table_is_correct.”
pub const LOOKUP_TABLE: [u8; 256] = [
    0, 7, 26, 63, 124, 215, 85, 254, 214, 228, 45, 185, 140, 173, 33, 240, 29, 177, 176, 32, 8,
    110, 87, 202, 204, 99, 150, 106, 230, 14, 235, 128, 213, 239, 212, 138, 23, 130, 208, 6, 44,
    71, 93, 116, 146, 189, 251, 81, 199, 97, 38, 28, 73, 179, 95, 84, 152, 48, 35, 119, 49, 88,
    242, 3, 148, 169, 72, 120, 62, 161, 166, 83, 175, 191, 137, 19, 100, 129, 112, 55, 221, 102,
    218, 61, 151, 237, 68, 164, 17, 147, 46, 234, 203, 216, 22, 141, 65, 57, 123, 12, 244, 54, 219,
    231, 96, 77, 180, 154, 5, 253, 133, 165, 98, 195, 205, 134, 245, 30, 9, 188, 59, 142, 186, 197,
    181, 144, 92, 31, 224, 163, 111, 74, 58, 69, 113, 196, 67, 246, 225, 10, 121, 50, 60, 157, 90,
    122, 2, 250, 101, 75, 178, 159, 24, 36, 201, 11, 243, 132, 198, 190, 114, 233, 39, 52, 21, 209,
    108, 238, 91, 187, 18, 104, 194, 37, 153, 34, 200, 143, 126, 155, 236, 118, 64, 80, 172, 89,
    94, 193, 135, 183, 86, 107, 252, 13, 167, 206, 136, 220, 207, 103, 171, 160, 76, 182, 227, 217,
    158, 56, 174, 4, 66, 109, 139, 162, 184, 211, 249, 47, 125, 232, 117, 43, 16, 42, 127, 20, 241,
    25, 149, 105, 156, 51, 53, 168, 145, 247, 223, 79, 78, 226, 15, 222, 82, 115, 70, 210, 27, 41,
    1, 170, 40, 131, 192, 229, 248, 255,
];

/// The round constants used in the Tip5 permutation. To verify their correctness, see the test
/// “round_constants_are_correct.”
pub const ROUND_CONSTANTS: [BFieldElement; NUM_ROUNDS * STATE_SIZE] = [
    BFieldElement::new(13630775303355457758),
    BFieldElement::new(16896927574093233874),
    BFieldElement::new(10379449653650130495),
    BFieldElement::new(1965408364413093495),
    BFieldElement::new(15232538947090185111),
    BFieldElement::new(15892634398091747074),
    BFieldElement::new(3989134140024871768),
    BFieldElement::new(2851411912127730865),
    BFieldElement::new(8709136439293758776),
    BFieldElement::new(3694858669662939734),
    BFieldElement::new(12692440244315327141),
    BFieldElement::new(10722316166358076749),
    BFieldElement::new(12745429320441639448),
    BFieldElement::new(17932424223723990421),
    BFieldElement::new(7558102534867937463),
    BFieldElement::new(15551047435855531404),
    BFieldElement::new(17532528648579384106),
    BFieldElement::new(5216785850422679555),
    BFieldElement::new(15418071332095031847),
    BFieldElement::new(11921929762955146258),
    BFieldElement::new(9738718993677019874),
    BFieldElement::new(3464580399432997147),
    BFieldElement::new(13408434769117164050),
    BFieldElement::new(264428218649616431),
    BFieldElement::new(4436247869008081381),
    BFieldElement::new(4063129435850804221),
    BFieldElement::new(2865073155741120117),
    BFieldElement::new(5749834437609765994),
    BFieldElement::new(6804196764189408435),
    BFieldElement::new(17060469201292988508),
    BFieldElement::new(9475383556737206708),
    BFieldElement::new(12876344085611465020),
    BFieldElement::new(13835756199368269249),
    BFieldElement::new(1648753455944344172),
    BFieldElement::new(9836124473569258483),
    BFieldElement::new(12867641597107932229),
    BFieldElement::new(11254152636692960595),
    BFieldElement::new(16550832737139861108),
    BFieldElement::new(11861573970480733262),
    BFieldElement::new(1256660473588673495),
    BFieldElement::new(13879506000676455136),
    BFieldElement::new(10564103842682358721),
    BFieldElement::new(16142842524796397521),
    BFieldElement::new(3287098591948630584),
    BFieldElement::new(685911471061284805),
    BFieldElement::new(5285298776918878023),
    BFieldElement::new(18310953571768047354),
    BFieldElement::new(3142266350630002035),
    BFieldElement::new(549990724933663297),
    BFieldElement::new(4901984846118077401),
    BFieldElement::new(11458643033696775769),
    BFieldElement::new(8706785264119212710),
    BFieldElement::new(12521758138015724072),
    BFieldElement::new(11877914062416978196),
    BFieldElement::new(11333318251134523752),
    BFieldElement::new(3933899631278608623),
    BFieldElement::new(16635128972021157924),
    BFieldElement::new(10291337173108950450),
    BFieldElement::new(4142107155024199350),
    BFieldElement::new(16973934533787743537),
    BFieldElement::new(11068111539125175221),
    BFieldElement::new(17546769694830203606),
    BFieldElement::new(5315217744825068993),
    BFieldElement::new(4609594252909613081),
    BFieldElement::new(3350107164315270407),
    BFieldElement::new(17715942834299349177),
    BFieldElement::new(9600609149219873996),
    BFieldElement::new(12894357635820003949),
    BFieldElement::new(4597649658040514631),
    BFieldElement::new(7735563950920491847),
    BFieldElement::new(1663379455870887181),
    BFieldElement::new(13889298103638829706),
    BFieldElement::new(7375530351220884434),
    BFieldElement::new(3502022433285269151),
    BFieldElement::new(9231805330431056952),
    BFieldElement::new(9252272755288523725),
    BFieldElement::new(10014268662326746219),
    BFieldElement::new(15565031632950843234),
    BFieldElement::new(1209725273521819323),
    BFieldElement::new(6024642864597845108),
];

/// The defining, first column of the (circulant) MDS matrix.
/// Derived from the SHA-256 hash of the ASCII string “Tip5” by dividing the digest into 16-bit
/// chunks.
pub const MDS_MATRIX_FIRST_COLUMN: [i64; STATE_SIZE] = [
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034, 56951, 27521, 41351, 40901, 12021, 59689,
    26798, 17845,
];

#[derive(
    Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq, GetSize, BFieldCodec, Arbitrary,
)]
#[repr(align(64))] // for SIMD, align along BFieldElement
pub struct Tip5 {
    pub state: [BFieldElement; STATE_SIZE],
}

#[cfg(not(all(
    target_feature = "avx512ifma",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
)))]
impl Tip5 {
    #[inline(always)]
    fn round(&mut self, round_index: usize) {
        self.sbox_layer();
        self.mds_generated();
        for i in 0..STATE_SIZE {
            self.state[i] += ROUND_CONSTANTS[round_index * STATE_SIZE + i];
        }
    }

    #[inline(always)]
    fn sbox_layer(&mut self) {
        for i in 0..NUM_SPLIT_AND_LOOKUP {
            Self::split_and_lookup(&mut self.state[i]);
        }

        for i in NUM_SPLIT_AND_LOOKUP..STATE_SIZE {
            let sq = self.state[i] * self.state[i];
            let qu = sq * sq;
            self.state[i] *= sq * qu;
        }
    }

    #[inline]
    fn split_and_lookup(element: &mut BFieldElement) {
        // let value = element.value();
        let mut bytes = element.raw_bytes();

        for i in 0..8 {
            // bytes[i] = Self::offset_fermat_cube_map(bytes[i].into()) as u8;
            bytes[i] = LOOKUP_TABLE[bytes[i] as usize];
        }

        *element = BFieldElement::from_raw_bytes(&bytes);
    }

    #[inline(always)]
    fn mds_generated(&mut self) {
        let mut lo: [u64; STATE_SIZE] = [0; STATE_SIZE];
        let mut hi: [u64; STATE_SIZE] = [0; STATE_SIZE];
        for i in 0..STATE_SIZE {
            let b = self.state[i].raw_u64();
            hi[i] = b >> 32;
            lo[i] = b & 0xffff_ffff;
        }

        lo = Self::generated_function(lo);
        hi = Self::generated_function(hi);

        // In isolation, the following reduction modulo BFieldElement::P is
        // buggy. Concretely, it can produce degenerate Montgomery
        // representations, that is, `state[r].0` can be greater than or equal
        // to BFieldElement::P. While there are many inputs for which this can
        // happen, the easiest with which to trace the behavior manually is:
        // lo[r] = 0x10;
        // hi[r] = 0xf_ffff_fff0;
        //
        // These starting values lead to `s_hi = 0`, `s_lo = BFieldElement::P`.
        // Since `s_hi` is 0, the `overflowing_add` does nothing, and the
        // degenerate representation from `s_lo` is directly transferred to the
        // state element.
        //
        // All that said, due to the specific context this method is always (!)
        // used in, the bug does not propagate. In particular, this method is
        // followed up with round-constant addition. Due to a quirk in the
        // implementation of `BFieldElement::Add` and a property of the round
        // constants, any degenerate representations are corrected.
        //
        // Below, you can find tests for the specific properties claimed. The
        // doc-string of those tests mention the name of this method.
        for r in 0..STATE_SIZE {
            let s = (lo[r] >> 4) as u128 + ((hi[r] as u128) << 28);

            let s_hi = (s >> 64) as u64;
            let s_lo = s as u64;

            let (res, over) = s_lo.overflowing_add(s_hi * 0xffff_ffff);

            self.state[r] = BFieldElement::from_raw_u64(if over { res + 0xffff_ffff } else { res });
        }
    }

    #[inline(always)]
    fn generated_function(input: [u64; STATE_SIZE]) -> [u64; STATE_SIZE] {
        let node_34 = input[0].wrapping_add(input[8]);
        let node_38 = input[4].wrapping_add(input[12]);
        let node_36 = input[2].wrapping_add(input[10]);
        let node_40 = input[6].wrapping_add(input[14]);
        let node_35 = input[1].wrapping_add(input[9]);
        let node_39 = input[5].wrapping_add(input[13]);
        let node_37 = input[3].wrapping_add(input[11]);
        let node_41 = input[7].wrapping_add(input[15]);
        let node_50 = node_34.wrapping_add(node_38);
        let node_52 = node_36.wrapping_add(node_40);
        let node_51 = node_35.wrapping_add(node_39);
        let node_53 = node_37.wrapping_add(node_41);
        let node_160 = input[0].wrapping_sub(input[8]);
        let node_161 = input[1].wrapping_sub(input[9]);
        let node_165 = input[5].wrapping_sub(input[13]);
        let node_163 = input[3].wrapping_sub(input[11]);
        let node_167 = input[7].wrapping_sub(input[15]);
        let node_162 = input[2].wrapping_sub(input[10]);
        let node_166 = input[6].wrapping_sub(input[14]);
        let node_164 = input[4].wrapping_sub(input[12]);
        let node_58 = node_50.wrapping_add(node_52);
        let node_59 = node_51.wrapping_add(node_53);
        let node_90 = node_34.wrapping_sub(node_38);
        let node_91 = node_35.wrapping_sub(node_39);
        let node_93 = node_37.wrapping_sub(node_41);
        let node_92 = node_36.wrapping_sub(node_40);
        let node_64 = node_58.wrapping_add(node_59).wrapping_mul(524757);
        let node_67 = node_58.wrapping_sub(node_59).wrapping_mul(52427);
        let node_71 = node_50.wrapping_sub(node_52);
        let node_72 = node_51.wrapping_sub(node_53);
        let node_177 = node_161.wrapping_add(node_165);
        let node_179 = node_163.wrapping_add(node_167);
        let node_178 = node_162.wrapping_add(node_166);
        let node_176 = node_160.wrapping_add(node_164);
        let node_69 = node_64.wrapping_add(node_67);
        let node_397 = node_71
            .wrapping_mul(18446744073709525744)
            .wrapping_sub(node_72.wrapping_mul(53918));
        let node_1857 = node_90.wrapping_mul(395512);
        let node_99 = node_91.wrapping_add(node_93);
        let node_1865 = node_91.wrapping_mul(18446744073709254400);
        let node_1869 = node_93.wrapping_mul(179380);
        let node_1873 = node_92.wrapping_mul(18446744073709509368);
        let node_1879 = node_160.wrapping_mul(35608);
        let node_185 = node_161.wrapping_add(node_163);
        let node_1915 = node_161.wrapping_mul(18446744073709340312);
        let node_1921 = node_163.wrapping_mul(18446744073709494992);
        let node_1927 = node_162.wrapping_mul(18446744073709450808);
        let node_228 = node_165.wrapping_add(node_167);
        let node_1939 = node_165.wrapping_mul(18446744073709420056);
        let node_1945 = node_167.wrapping_mul(18446744073709505128);
        let node_1951 = node_166.wrapping_mul(216536);
        let node_1957 = node_164.wrapping_mul(18446744073709515080);
        let node_70 = node_64.wrapping_sub(node_67);
        let node_702 = node_71
            .wrapping_mul(53918)
            .wrapping_add(node_72.wrapping_mul(18446744073709525744));
        let node_1961 = node_90.wrapping_mul(18446744073709254400);
        let node_1963 = node_91.wrapping_mul(395512);
        let node_1965 = node_92.wrapping_mul(179380);
        let node_1967 = node_93.wrapping_mul(18446744073709509368);
        let node_1970 = node_160.wrapping_mul(18446744073709340312);
        let node_1973 = node_161.wrapping_mul(35608);
        let node_1982 = node_162.wrapping_mul(18446744073709494992);
        let node_1985 = node_163.wrapping_mul(18446744073709450808);
        let node_1988 = node_166.wrapping_mul(18446744073709505128);
        let node_1991 = node_167.wrapping_mul(216536);
        let node_1994 = node_164.wrapping_mul(18446744073709420056);
        let node_1997 = node_165.wrapping_mul(18446744073709515080);
        let node_98 = node_90.wrapping_add(node_92);
        let node_184 = node_160.wrapping_add(node_162);
        let node_227 = node_164.wrapping_add(node_166);
        let node_86 = node_69.wrapping_add(node_397);
        let node_403 = node_1857.wrapping_sub(
            node_99
                .wrapping_mul(18446744073709433780)
                .wrapping_sub(node_1865)
                .wrapping_sub(node_1869)
                .wrapping_add(node_1873),
        );
        let node_271 = node_177.wrapping_add(node_179);
        let node_1891 = node_177.wrapping_mul(18446744073709208752);
        let node_1897 = node_179.wrapping_mul(18446744073709448504);
        let node_1903 = node_178.wrapping_mul(115728);
        let node_1909 = node_185.wrapping_mul(18446744073709283688);
        let node_1933 = node_228.wrapping_mul(18446744073709373568);
        let node_88 = node_70.wrapping_add(node_702);
        let node_708 = node_1961
            .wrapping_add(node_1963)
            .wrapping_sub(node_1965.wrapping_add(node_1967));
        let node_1976 = node_178.wrapping_mul(18446744073709448504);
        let node_1979 = node_179.wrapping_mul(115728);
        let node_87 = node_69.wrapping_sub(node_397);
        let node_897 = node_1865
            .wrapping_add(node_98.wrapping_mul(353264))
            .wrapping_sub(node_1857)
            .wrapping_sub(node_1873)
            .wrapping_sub(node_1869);
        let node_2007 = node_184.wrapping_mul(18446744073709486416);
        let node_2013 = node_227.wrapping_mul(180000);
        let node_89 = node_70.wrapping_sub(node_702);
        let node_1077 = node_98
            .wrapping_mul(18446744073709433780)
            .wrapping_add(node_99.wrapping_mul(353264))
            .wrapping_sub(node_1961.wrapping_add(node_1963))
            .wrapping_sub(node_1965.wrapping_add(node_1967));
        let node_2020 = node_184.wrapping_mul(18446744073709283688);
        let node_2023 = node_185.wrapping_mul(18446744073709486416);
        let node_2026 = node_227.wrapping_mul(18446744073709373568);
        let node_2029 = node_228.wrapping_mul(180000);
        let node_2035 = node_176.wrapping_mul(18446744073709550688);
        let node_2038 = node_176.wrapping_mul(18446744073709208752);
        let node_2041 = node_177.wrapping_mul(18446744073709550688);
        let node_270 = node_176.wrapping_add(node_178);
        let node_152 = node_86.wrapping_add(node_403);
        let node_412 = node_1879.wrapping_sub(
            node_271
                .wrapping_mul(18446744073709105640)
                .wrapping_sub(node_1891)
                .wrapping_sub(node_1897)
                .wrapping_add(node_1903)
                .wrapping_sub(
                    node_1909
                        .wrapping_sub(node_1915)
                        .wrapping_sub(node_1921)
                        .wrapping_add(node_1927),
                )
                .wrapping_sub(
                    node_1933
                        .wrapping_sub(node_1939)
                        .wrapping_sub(node_1945)
                        .wrapping_add(node_1951),
                )
                .wrapping_add(node_1957),
        );
        let node_154 = node_88.wrapping_add(node_708);
        let node_717 = node_1970.wrapping_add(node_1973).wrapping_sub(
            node_1976
                .wrapping_add(node_1979)
                .wrapping_sub(node_1982.wrapping_add(node_1985))
                .wrapping_sub(node_1988.wrapping_add(node_1991))
                .wrapping_add(node_1994.wrapping_add(node_1997)),
        );
        let node_156 = node_87.wrapping_add(node_897);
        let node_906 = node_1915
            .wrapping_add(node_2007)
            .wrapping_sub(node_1879)
            .wrapping_sub(node_1927)
            .wrapping_sub(
                node_1897
                    .wrapping_sub(node_1921)
                    .wrapping_sub(node_1945)
                    .wrapping_add(
                        node_1939
                            .wrapping_add(node_2013)
                            .wrapping_sub(node_1957)
                            .wrapping_sub(node_1951),
                    ),
            );
        let node_158 = node_89.wrapping_add(node_1077);
        let node_1086 = node_2020
            .wrapping_add(node_2023)
            .wrapping_sub(node_1970.wrapping_add(node_1973))
            .wrapping_sub(node_1982.wrapping_add(node_1985))
            .wrapping_sub(
                node_2026
                    .wrapping_add(node_2029)
                    .wrapping_sub(node_1994.wrapping_add(node_1997))
                    .wrapping_sub(node_1988.wrapping_add(node_1991)),
            );
        let node_153 = node_86.wrapping_sub(node_403);
        let node_1237 = node_1909
            .wrapping_sub(node_1915)
            .wrapping_sub(node_1921)
            .wrapping_add(node_1927)
            .wrapping_add(node_2035)
            .wrapping_sub(node_1879)
            .wrapping_sub(node_1957)
            .wrapping_sub(
                node_1933
                    .wrapping_sub(node_1939)
                    .wrapping_sub(node_1945)
                    .wrapping_add(node_1951),
            );
        let node_155 = node_88.wrapping_sub(node_708);
        let node_1375 = node_1982
            .wrapping_add(node_1985)
            .wrapping_add(node_2038.wrapping_add(node_2041))
            .wrapping_sub(node_1970.wrapping_add(node_1973))
            .wrapping_sub(node_1994.wrapping_add(node_1997))
            .wrapping_sub(node_1988.wrapping_add(node_1991));
        let node_157 = node_87.wrapping_sub(node_897);
        let node_1492 = node_1921
            .wrapping_add(
                node_1891
                    .wrapping_add(node_270.wrapping_mul(114800))
                    .wrapping_sub(node_2035)
                    .wrapping_sub(node_1903),
            )
            .wrapping_sub(
                node_1915
                    .wrapping_add(node_2007)
                    .wrapping_sub(node_1879)
                    .wrapping_sub(node_1927),
            )
            .wrapping_sub(
                node_1939
                    .wrapping_add(node_2013)
                    .wrapping_sub(node_1957)
                    .wrapping_sub(node_1951),
            )
            .wrapping_sub(node_1945);
        let node_159 = node_89.wrapping_sub(node_1077);
        let node_1657 = node_270
            .wrapping_mul(18446744073709105640)
            .wrapping_add(node_271.wrapping_mul(114800))
            .wrapping_sub(node_2038.wrapping_add(node_2041))
            .wrapping_sub(node_1976.wrapping_add(node_1979))
            .wrapping_sub(
                node_2020
                    .wrapping_add(node_2023)
                    .wrapping_sub(node_1970.wrapping_add(node_1973))
                    .wrapping_sub(node_1982.wrapping_add(node_1985)),
            )
            .wrapping_sub(
                node_2026
                    .wrapping_add(node_2029)
                    .wrapping_sub(node_1994.wrapping_add(node_1997))
                    .wrapping_sub(node_1988.wrapping_add(node_1991)),
            );

        [
            node_152.wrapping_add(node_412),
            node_154.wrapping_add(node_717),
            node_156.wrapping_add(node_906),
            node_158.wrapping_add(node_1086),
            node_153.wrapping_add(node_1237),
            node_155.wrapping_add(node_1375),
            node_157.wrapping_add(node_1492),
            node_159.wrapping_add(node_1657),
            node_152.wrapping_sub(node_412),
            node_154.wrapping_sub(node_717),
            node_156.wrapping_sub(node_906),
            node_158.wrapping_sub(node_1086),
            node_153.wrapping_sub(node_1237),
            node_155.wrapping_sub(node_1375),
            node_157.wrapping_sub(node_1492),
            node_159.wrapping_sub(node_1657),
        ]
    }
}

impl Tip5 {
    #[inline]
    pub const fn new(domain: Domain) -> Self {
        let mut state = const { [BFieldElement::ZERO; STATE_SIZE] };

        match domain {
            Domain::VariableLength => (),
            Domain::FixedLength => {
                let mut i = RATE;
                while i < STATE_SIZE {
                    state[i] = BFieldElement::ONE;
                    i += 1;
                }
            }
        }

        Self { state }
    }

    #[inline(always)]
    pub fn permutation(&mut self) {
        for i in 0..NUM_ROUNDS {
            self.round(i);
        }
    }

    /// Functionally equivalent to [`permutation`](Self::permutation). Returns the trace of
    /// applying the permutation; that is, the initial state of the sponge as well as its state
    /// after each round.
    pub fn trace(&mut self) -> [[BFieldElement; STATE_SIZE]; 1 + NUM_ROUNDS] {
        let mut trace = [[BFieldElement::ZERO; STATE_SIZE]; 1 + NUM_ROUNDS];

        trace[0] = self.state;
        for i in 0..NUM_ROUNDS {
            self.round(i);
            trace[1 + i] = self.state;
        }

        trace
    }

    /// Hash 10 [`BFieldElement`]s.
    ///
    /// There is no input-padding because the input length is fixed.
    ///
    /// When you want to hash together two [`Digest`]s, use [`Self::hash_pair`]
    /// instead. In some rare cases you do want to hash a fixed-length string
    /// of individual [`BFieldElement`]s, which is why this function is exposed.
    ///
    /// See also: [`Self::hash_pair`], [`Self::hash`], [`Self::hash_varlen`].
    pub fn hash_10(input: &[BFieldElement; RATE]) -> [BFieldElement; Digest::LEN] {
        let mut sponge = Self::new(Domain::FixedLength);

        // absorb once
        sponge.state[..RATE].copy_from_slice(input);

        sponge.permutation();

        // squeeze once
        sponge.state[..Digest::LEN].try_into().unwrap()
    }

    /// Hash two [`Digest`]s together.
    ///
    /// This function is syntax sugar for calling [`Self::hash_10`] on the
    /// concatenation of the digests' values.
    ///
    /// See also: [`Self::hash_10`], [`Self::hash`], [`Self::hash_varlen`].
    pub fn hash_pair(left: Digest, right: Digest) -> Digest {
        let mut sponge = Self::new(Domain::FixedLength);
        sponge.state[..Digest::LEN].copy_from_slice(&left.values());
        sponge.state[Digest::LEN..2 * Digest::LEN].copy_from_slice(&right.values());

        sponge.permutation();

        let digest_values = sponge.state[..Digest::LEN].try_into().unwrap();
        Digest::new(digest_values)
    }

    /// Hash an object based on its [`BFieldCodec`]-encoding.
    ///
    /// Thin wrapper around [`hash_varlen`](Self::hash_varlen).
    ///
    /// See also: [`Self::hash_10`], [`Self::hash_pair`], [`Self::hash_varlen`].
    pub fn hash<T: BFieldCodec>(value: &T) -> Digest {
        Self::hash_varlen(&value.encode())
    }

    /// Hash a variable-length sequence of [`BFieldElement`].
    ///
    /// This function pads the input as its length is variable.
    ///
    /// Note that [`Self::hash_varlen`] and [`Self::hash_10`] are different
    /// functions, even when the input to the former, after padding, agrees with
    /// the input to the latter. The difference comes from the initial value of
    /// the capacity-part of the state, which in the case of variable-length
    /// hashing is all-ones but in the case of fixed-length hashing is
    /// all-zeroes.
    ///
    /// Prefer [`Self::hash`] whenever an object is being hashed whose type
    /// implements [`BFieldCodec`]. However, such an object is not always
    /// available, which is why this function is exposed.
    ///
    /// See also: [`Self::hash_10`], [`Self::hash_pair`], [`Self::hash`].
    //
    // - Apply the correct padding
    // - [Sponge::pad_and_absorb_all()]
    // - Read the digest from the resulting state.
    pub fn hash_varlen(input: &[BFieldElement]) -> Digest {
        let mut sponge = Self::init();
        sponge.pad_and_absorb_all(input);
        let produce = (&sponge.state[..Digest::LEN]).try_into().unwrap();

        Digest::new(produce)
    }

    /// Produce `num_indices` random integer values in the range `[0, upper_bound)`. The
    /// `upper_bound` must be a power of 2.
    ///
    /// This method uses von Neumann rejection sampling.
    /// Specifically, if the top 32 bits of a BFieldElement are all ones, then the bottom 32 bits
    /// are not uniformly distributed, and so they are dropped. This method invokes squeeze until
    /// enough uniform u32s have been sampled.
    ///
    /// # Panics
    ///
    /// Panics if `upper_bound` is not a power of two.
    pub fn sample_indices(&mut self, upper_bound: u32, num_indices: usize) -> Vec<u32> {
        assert!(upper_bound.is_power_of_two());

        let mut indices = Vec::with_capacity(num_indices);
        let mut buffer = const { [BFieldElement::ZERO; RATE] };
        let mut next_in_buffer = RATE;
        while indices.len() < num_indices {
            if next_in_buffer == RATE {
                buffer = self.squeeze();
                next_in_buffer = 0;
            }
            let element = buffer[next_in_buffer];
            next_in_buffer += 1;

            if element != const { BFieldElement::new(BFieldElement::MAX) } {
                indices.push(element.value() as u32 % upper_bound);
            }
        }

        indices
    }

    /// Produce `num_elements` random [`XFieldElement`] values.
    ///
    /// If `num_elements` is not divisible by [`RATE`][rate], spill the remaining elements of the
    /// last [`squeeze`][Sponge::squeeze].
    ///
    /// [rate]: Sponge::RATE
    pub fn sample_scalars(&mut self, num_elements: usize) -> Vec<XFieldElement> {
        let num_squeezes = (num_elements * EXTENSION_DEGREE).div_ceil(Self::RATE);
        debug_assert!(num_elements * EXTENSION_DEGREE <= num_squeezes * Self::RATE);

        (0..num_squeezes)
            .flat_map(|_| self.squeeze())
            .tuples()
            .take(num_elements)
            .map(|(x0, x1, x2)| XFieldElement::new([x0, x1, x2]))
            .collect()
    }
}

impl Sponge for Tip5 {
    const RATE: usize = RATE;

    fn init() -> Self {
        Self::new(Domain::VariableLength)
    }

    fn absorb(&mut self, input: [BFieldElement; RATE]) {
        self.state[..RATE]
            .iter_mut()
            .zip_eq(&input)
            .for_each(|(a, &b)| *a = b);

        self.permutation();
    }

    fn squeeze(&mut self) -> [BFieldElement; RATE] {
        let produce: [BFieldElement; RATE] = (&self.state[..RATE]).try_into().unwrap();
        self.permutation();

        produce
    }
}

impl Hasher for Tip5 {
    fn finish(&self) -> u64 {
        self.state[0].value()
    }

    fn write(&mut self, bytes: &[u8]) {
        let bfield_elements = bytes.chunks(BFieldElement::BYTES).map(|chunk| {
            let mut buffer = [0u8; BFieldElement::BYTES];
            buffer[..chunk.len()].copy_from_slice(chunk);
            BFieldElement::new(u64::from_le_bytes(buffer))
        });

        for chunk in bfield_elements.chunks(Tip5::RATE).into_iter() {
            let mut buffer = [BFieldElement::ZERO; Tip5::RATE];
            for (buffer_elem, chunk_elem) in buffer.iter_mut().zip(chunk) {
                *buffer_elem = chunk_elem;
            }
            self.absorb(buffer)
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use std::hash::Hash;
    use std::ops::Mul;

    use prop::sample::size_range;
    use proptest::prelude::*;
    use rand::RngCore;
    use rayon::prelude::IntoParallelIterator;
    use rayon::prelude::ParallelIterator;

    use super::*;
    use crate::math::other::random_elements;
    use crate::math::x_field_element::XFieldElement;
    use crate::proptest_arbitrary_interop::arb;
    use crate::tests::proptest;
    use crate::tests::test;

    impl proptest::arbitrary::Arbitrary for Tip5 {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            arb().boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    impl Tip5 {
        fn mds_cyclomul(&mut self) {
            let mut result = [BFieldElement::ZERO; STATE_SIZE];

            let mut lo: [i64; STATE_SIZE] = [0; STATE_SIZE];
            let mut hi: [i64; STATE_SIZE] = [0; STATE_SIZE];
            for (i, b) in self.state.iter().enumerate() {
                hi[i] = (b.raw_u64() >> 32) as i64;
                lo[i] = (b.raw_u64() as u32) as i64;
            }

            lo = Self::fast_cyclomul16(lo, MDS_MATRIX_FIRST_COLUMN);
            hi = Self::fast_cyclomul16(hi, MDS_MATRIX_FIRST_COLUMN);

            for r in 0..STATE_SIZE {
                let s = lo[r] as u128 + ((hi[r] as u128) << 32);
                let s_hi = (s >> 64) as u64;
                let s_lo = s as u64;
                let z = (s_hi << 32) - s_hi;
                let (res, over) = s_lo.overflowing_add(z);

                result[r] = BFieldElement::from_raw_u64(
                    res.wrapping_add(0u32.wrapping_sub(over as u32) as u64),
                );
            }
            self.state = result;
        }

        fn fast_cyclomul16(f: [i64; 16], g: [i64; 16]) -> [i64; 16] {
            const N: usize = 8;
            let mut ff_lo = [0i64; N];
            let mut gg_lo = [0i64; N];
            let mut ff_hi = [0i64; N];
            let mut gg_hi = [0i64; N];
            for i in 0..N {
                ff_lo[i] = f[i] + f[i + N];
                ff_hi[i] = f[i] - f[i + N];
                gg_lo[i] = g[i] + g[i + N];
                gg_hi[i] = g[i] - g[i + N];
            }

            let hh_lo = Self::fast_cyclomul8(ff_lo, gg_lo);
            let hh_hi = Self::complex_negacyclomul8(ff_hi, gg_hi);

            let mut hh = [0i64; 2 * N];
            for i in 0..N {
                hh[i] = (hh_lo[i] + hh_hi[i]) >> 1;
                hh[i + N] = (hh_lo[i] - hh_hi[i]) >> 1;
            }

            hh
        }

        fn fast_cyclomul8(f: [i64; 8], g: [i64; 8]) -> [i64; 8] {
            const N: usize = 4;
            let mut ff_lo = [0i64; N];
            let mut gg_lo = [0i64; N];
            let mut ff_hi = [0i64; N];
            let mut gg_hi = [0i64; N];
            for i in 0..N {
                ff_lo[i] = f[i] + f[i + N];
                ff_hi[i] = f[i] - f[i + N];
                gg_lo[i] = g[i] + g[i + N];
                gg_hi[i] = g[i] - g[i + N];
            }

            let hh_lo = Self::fast_cyclomul4(ff_lo, gg_lo);
            let hh_hi = Self::complex_negacyclomul4(ff_hi, gg_hi);

            let mut hh = [0i64; 2 * N];
            for i in 0..N {
                hh[i] = (hh_lo[i] + hh_hi[i]) >> 1;
                hh[i + N] = (hh_lo[i] - hh_hi[i]) >> 1;
            }

            hh
        }

        fn fast_cyclomul4(f: [i64; 4], g: [i64; 4]) -> [i64; 4] {
            const N: usize = 2;
            let mut ff_lo = [0i64; N];
            let mut gg_lo = [0i64; N];
            let mut ff_hi = [0i64; N];
            let mut gg_hi = [0i64; N];
            for i in 0..N {
                ff_lo[i] = f[i] + f[i + N];
                ff_hi[i] = f[i] - f[i + N];
                gg_lo[i] = g[i] + g[i + N];
                gg_hi[i] = g[i] - g[i + N];
            }

            let hh_lo = Self::fast_cyclomul2(ff_lo, gg_lo);
            let hh_hi = Self::complex_negacyclomul2(ff_hi, gg_hi);

            let mut hh = [0i64; 2 * N];
            for i in 0..N {
                hh[i] = (hh_lo[i] + hh_hi[i]) >> 1;
                hh[i + N] = (hh_lo[i] - hh_hi[i]) >> 1;
            }

            hh
        }

        fn fast_cyclomul2(f: [i64; 2], g: [i64; 2]) -> [i64; 2] {
            let ff_lo = f[0] + f[1];
            let ff_hi = f[0] - f[1];
            let gg_lo = g[0] + g[1];
            let gg_hi = g[0] - g[1];

            let hh_lo = ff_lo * gg_lo;
            let hh_hi = ff_hi * gg_hi;

            let mut hh = [0i64; 2];
            hh[0] = (hh_lo + hh_hi) >> 1;
            hh[1] = (hh_lo - hh_hi) >> 1;

            hh
        }

        fn complex_negacyclomul8(f: [i64; 8], g: [i64; 8]) -> [i64; 8] {
            const N: usize = 4;

            let mut f0 = [(0i64, 0i64); N];
            let mut g0 = [(0i64, 0i64); N];

            for i in 0..N {
                f0[i] = (f[i], -f[N + i]);
                g0[i] = (g[i], -g[N + i]);
            }

            let h0 = Self::complex_karatsuba4(f0, g0);

            let mut h = [0i64; 3 * N - 1];
            for i in 0..(2 * N - 1) {
                h[i] += h0[i].0;
                h[i + N] -= h0[i].1;
            }

            let mut hh = [0i64; 2 * N];
            for i in 0..(2 * N) {
                hh[i] += h[i];
            }
            for i in (2 * N)..(3 * N - 1) {
                hh[i - 2 * N] -= h[i];
            }

            hh
        }

        fn complex_negacyclomul4(f: [i64; 4], g: [i64; 4]) -> [i64; 4] {
            const N: usize = 2;

            let mut f0 = [(0i64, 0i64); N];
            let mut g0 = [(0i64, 0i64); N];

            for i in 0..N {
                f0[i] = (f[i], -f[N + i]);
                g0[i] = (g[i], -g[N + i]);
            }

            let h0 = Self::complex_karatsuba2(f0, g0);

            let mut h = [0i64; 4 * N - 1];
            for i in 0..(2 * N - 1) {
                h[i] += h0[i].0;
                h[i + N] -= h0[i].1;
            }

            let mut hh = [0i64; 2 * N];
            for i in 0..(2 * N) {
                hh[i] += h[i];
            }
            for i in (2 * N)..(4 * N - 1) {
                hh[i - 2 * N] -= h[i];
            }

            hh
        }

        fn complex_negacyclomul2(f: [i64; 2], g: [i64; 2]) -> [i64; 2] {
            let f0 = (f[0], -f[1]);
            let g0 = (g[0], -g[1]);

            let h0 = Self::complex_product(f0, g0);

            [h0.0, -h0.1]
        }

        #[inline(always)]
        fn complex_karatsuba4(f: [(i64, i64); 4], g: [(i64, i64); 4]) -> [(i64, i64); 7] {
            const N: usize = 2;

            let ff = Self::complex_sum::<2>(f[..N].try_into().unwrap(), f[N..].try_into().unwrap());
            let gg = Self::complex_sum::<2>(g[..N].try_into().unwrap(), g[N..].try_into().unwrap());

            let lo =
                Self::complex_karatsuba2(f[..N].try_into().unwrap(), g[..N].try_into().unwrap());
            let hi =
                Self::complex_karatsuba2(f[N..].try_into().unwrap(), g[N..].try_into().unwrap());

            let li = Self::complex_diff::<3>(
                Self::complex_karatsuba2(ff, gg),
                Self::complex_sum::<3>(lo, hi),
            );

            let mut result = [(0i64, 0i64); 4 * N - 1];
            for i in 0..(2 * N - 1) {
                result[i].0 = lo[i].0;
                result[i].1 = lo[i].1;
            }
            for i in 0..(2 * N - 1) {
                result[N + i].0 += li[i].0;
                result[N + i].1 += li[i].1;
            }
            for i in 0..(2 * N - 1) {
                result[2 * N + i].0 += hi[i].0;
                result[2 * N + i].1 += hi[i].1;
            }

            result
        }

        fn complex_karatsuba2(f: [(i64, i64); 2], g: [(i64, i64); 2]) -> [(i64, i64); 3] {
            const N: usize = 1;

            let ff = (f[0].0 + f[1].0, f[0].1 + f[1].1);
            let gg = (g[0].0 + g[1].0, g[0].1 + g[1].1);

            let lo = Self::complex_product(f[0], g[0]);
            let hi = Self::complex_product(f[1], g[1]);

            let ff_times_gg = Self::complex_product(ff, gg);
            let lo_plus_hi = (lo.0 + hi.0, lo.1 + hi.1);

            let li = (ff_times_gg.0 - lo_plus_hi.0, ff_times_gg.1 - lo_plus_hi.1);

            let mut result = [(0i64, 0i64); 4 * N - 1];
            result[0].0 += lo.0;
            result[0].1 += lo.1;
            result[N].0 += li.0;
            result[N].1 += li.1;
            result[2 * N].0 += hi.0;
            result[2 * N].1 += hi.1;

            result
        }

        fn complex_product(f: (i64, i64), g: (i64, i64)) -> (i64, i64) {
            // don't karatsuba; this is faster
            (f.0 * g.0 - f.1 * g.1, f.0 * g.1 + f.1 * g.0)
        }

        fn complex_sum<const N: usize>(f: [(i64, i64); N], g: [(i64, i64); N]) -> [(i64, i64); N] {
            let mut h = [(0i64, 0i64); N];
            for i in 0..N {
                h[i].0 = f[i].0 + g[i].0;
                h[i].1 = f[i].1 + g[i].1;
            }
            h
        }

        fn complex_diff<const N: usize>(f: [(i64, i64); N], g: [(i64, i64); N]) -> [(i64, i64); N] {
            let mut h = [(0i64, 0i64); N];
            for i in 0..N {
                h[i].0 = f[i].0 - g[i].0;
                h[i].1 = f[i].1 - g[i].1;
            }
            h
        }

        fn offset_fermat_cube_map(x: u16) -> u16 {
            let xx = (x + 1) as u64;
            let xxx = xx * xx * xx;
            ((xxx + 256) % 257) as u16
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn get_size(tip5: Tip5) {
        assert_eq!(STATE_SIZE * BFieldElement::ZERO.get_size(), tip5.get_size());
    }

    #[macro_rules_attr::apply(test)]
    fn lookup_table_is_correct() {
        let table: [u8; 256] = (0..256)
            .map(|t| Tip5::offset_fermat_cube_map(t as u16) as u8)
            .collect_vec()
            .try_into()
            .unwrap();

        println!(
            "Entire lookup table:\n{}",
            table.iter().map(|t| format!("{t:02x}")).join(", ")
        );

        (0_usize..256).for_each(|i| {
            assert_eq!(
                LOOKUP_TABLE[i], table[i],
                "Lookup tables must agree at every index, including index {i}."
            )
        });
    }

    #[macro_rules_attr::apply(test)]
    fn round_constants_are_correct() {
        let to_int = |bytes: &[u8]| {
            bytes
                .iter()
                .take(16)
                .enumerate()
                .map(|(i, b)| (*b as u128) << (8 * i))
                .sum::<u128>()
        };
        let round_constants = (0..NUM_ROUNDS * STATE_SIZE)
            .map(|i| ["Tip5".to_string().as_bytes(), &[i as u8]].concat())
            .map(|bytes| blake3::hash(&bytes))
            .map(|hash| *hash.as_bytes())
            .map(|bytes| to_int(&bytes))
            .map(|i| (i % BFieldElement::P as u128) as u64)
            .map(BFieldElement::from_raw_u64)
            .collect_vec();

        println!(
            "In case you changed something, here are all round constants:\n{}",
            round_constants.iter().map(|c| format!("{c}")).join(", ")
        );

        (0_usize..NUM_ROUNDS * STATE_SIZE).for_each(|i| {
            assert_eq!(
                ROUND_CONSTANTS[i], round_constants[i],
                "Round constants must agree at every index, including index {i}."
            )
        });
    }

    #[macro_rules_attr::apply(test)]
    fn test_fermat_cube_map_is_permutation() {
        let mut touched = [false; 256];
        for i in 0..256 {
            touched[Tip5::offset_fermat_cube_map(i) as usize] = true;
        }
        assert!(touched.iter().all(|t| *t));
        assert_eq!(Tip5::offset_fermat_cube_map(0), 0);
        assert_eq!(Tip5::offset_fermat_cube_map(255), 255);
    }

    /// Ensure that the claims made in [`Tip5::mds_generated`] are true.
    ///
    /// In particular, `BFieldElement::Add` internally uses the equality
    /// `a + b = a - (p - b)`. If `a` is a degenerate representation (i.e., is
    /// larger than or equal to [`BFieldElement::P`]), then a small enough `b`
    /// makes the result of the addition non-degenerate by removing the
    /// “surplus” from `a`. In particular, this correction will happen if the
    /// following inequality holds:
    ///
    ///      p - b > u64::MAX - p
    ///   ↔  p - b > 2^64 - 1 - (2^64 - 2^32 + 1)
    ///   ↔  p - b > 2^32 - 2
    ///   ↔     -b > 2^32 - 2 - p
    ///   ↔      b < p + 2 - 2^32
    ///
    /// While it’s not particularly beautiful to depend on such implementation
    /// details, Ferdinand is too scared to change the implementation of Tip5.
    /// If you change the behavior of `BFieldElement::Add`, please make sure
    /// that [`Tip5`] is not breaking.
    ///
    /// The test [`round_constants_correct_degenerate_lhs_when_adding`] makes
    /// sure that all round constants have the required property.
    ///
    /// See also: https://www.hyrumslaw.com/. Sorry about that.
    #[macro_rules_attr::apply(proptest)]
    fn adding_degenerate_lhs_and_small_enough_rhs_makes_sum_non_degenerate(
        #[strategy(BFieldElement::P..)] raw_a: u64,
        #[strategy(0..BFieldElement::P + 2 - (1 << 32))] raw_b: u64,
    ) {
        let a = BFieldElement::from_raw_u64(raw_a);
        let b = BFieldElement::from_raw_u64(raw_b);
        let raw_sum = (a + b).raw_u64();
        prop_assert!(raw_sum < BFieldElement::P);
    }

    /// Ensure that the claims made in [`Tip5::mds_generated`] are true.
    ///
    /// [`adding_degenerate_lhs_and_small_enough_rhs_makes_sum_non_degenerate`]
    /// explains the requirement in greater detail.
    #[macro_rules_attr::apply(test)]
    fn round_constants_correct_degenerate_lhs_when_adding() {
        for constant in ROUND_CONSTANTS {
            assert!(constant.raw_u64() < BFieldElement::P + 2 - (1 << 32));
        }
    }

    /// Ensure that the claims made in [`Tip5::mds_generated`] are true.
    #[macro_rules_attr::apply(test)]
    fn tip5_recovers_from_degenerate_field_element_representations() {
        let state = [
            0x1063_c4bf_5d8b_b0dd,
            0xdb62_75d3_71fe_05d0,
            0xde58_cae3_0144_cdae,
            0xc774_e646_81d3_622e,
            0xc4a9_47d1_0a5a_a466,
            0xda55_77a0_0a91_3151,
            0xe80e_978b_3836_dcd0,
            0x8dd1_61f0_a3ac_00c2,
            0x6857_f251_a9c0_f693,
            0x4923_a368_3046_178e,
            0x6e6f_c54a_9b81_010b,
            0xcb84_fa5b_b9fa_ec36,
            0x93cb_f9db_4c5c_b1ea,
            0xf215_d9b9_2dc8_7266,
            0x88f0_9783_d2ae_3c57,
            0x6d29_f9ce_94a9_0b71,
        ]
        .map(BFieldElement::new);
        let expected = [
            0xa5d3_2d62_9e60_d72e,
            0x5516_ef90_d277_3d74,
            0x65d3_fa1c_de45_f6cb,
            0x7bf0_e725_dfa5_906b,
            0x67a2_db4b_141b_90e9,
            0x91db_162d_3230_9083,
            0xefec_1d00_146a_05c9,
            0xcca0_d656_6bca_8186,
            0x405b_aeb5_b3f8_7f02,
            0xd897_0158_7027_8f76,
            0xd4b2_ee48_10aa_c7d1,
            0x27b4_51e7_06a5_c2fc,
            0xe9b4_177f_0a0e_ffe4,
            0x0c60_def0_f2c5_287f,
            0x703a_a06d_327c_cc34,
            0x536f_2355_0ebf_98f1,
        ]
        .map(BFieldElement::new);

        let mut dbg_tip5 = Tip5 { state };
        dbg_tip5.sbox_layer();
        dbg_tip5.mds_generated();

        // If this assertion fails, you might have improved the internals of
        // Tip5 in a way that makes the properties tested for in
        // `adding_degenerate_lhs_and_small_enough_rhs_makes_sum_non_degenerate`
        // and
        // `round_constants_correct_degenerate_lhs_when_adding`
        // superfluous. If that is the case, feel free to remove those tests
        // as well as this one.
        debug_assert!(dbg_tip5.state[1].raw_u64() >= BFieldElement::P);

        let mut tip5 = Tip5 { state };
        tip5.permutation();
        assert_eq!(expected, tip5.state);

        let mut naive = naive::NaiveTip5 { state };
        naive.permutation();
        assert_eq!(expected, naive.state);
    }

    #[macro_rules_attr::apply(test)]
    fn calculate_differential_uniformity() {
        // cargo test calculate_differential_uniformity -- --include-ignored --nocapture
        // addition-differential
        let count_satisfiers_fermat = |a, b| {
            (0..(1 << 8))
                .map(|x| {
                    u16::from(
                        (256 + Tip5::offset_fermat_cube_map((x + a) & 0xff)
                            - Tip5::offset_fermat_cube_map(x))
                            & 0xff
                            == b,
                    )
                })
                .sum()
        };
        let du_fermat: u16 = (1..256)
            .into_par_iter()
            .map(|a| {
                (1..256)
                    .map(|b| count_satisfiers_fermat(a, b))
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        println!("additive differential uniformity for fermat cube map: {du_fermat}");

        // bitwise-differential
        let count_satisfiers_fermat_bitwise = |a: u16, b: u16| {
            (0..(1 << 8))
                .map(|x| {
                    u16::from(
                        (Tip5::offset_fermat_cube_map(x ^ a) ^ Tip5::offset_fermat_cube_map(x))
                            == b,
                    )
                })
                .sum::<u16>()
        };
        for a in 1..256 {
            for b in 1..256 {
                let num_satisfiers = count_satisfiers_fermat_bitwise(a, b);
                if num_satisfiers == 256 {
                    println!("a: {a}, b: {b} -> 256 satisfiers");
                }
            }
        }
        let du_fermat_bitwise: u16 = (1..256)
            .into_par_iter()
            .map(|a| {
                (1..256)
                    .map(|b| count_satisfiers_fermat_bitwise(a, b))
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        println!("bitwise differential uniformity for fermat cube map: {du_fermat_bitwise}");
    }

    #[macro_rules_attr::apply(test)]
    fn calculate_approximation_quality() {
        let mut fermat_cubed = [0u16; 65536];
        let mut bfield_cubed = [0u16; 65536];
        for i in 0..65536 {
            let cubed = (i as u64) * (i as u64) * (i as u64);
            fermat_cubed[i] = (cubed % 65537) as u16;
            bfield_cubed[i] = (cubed & 0xffff) as u16;
        }
        let equal_count = fermat_cubed
            .iter()
            .zip(bfield_cubed.iter())
            .filter(|(a, b)| a == b)
            .count();
        println!("agreement with low-degree function: {equal_count}");
    }

    /// A snapshot of a [`Digest`], in hexadecimal form. The exact procedure to
    /// arrive at the `Digest` in question is involved and best read from the
    /// code below.
    ///
    /// It is paramount that Tip5 has at least some snapshot tests. The reason
    /// is that (with the current implementation), conditional compilation
    /// changes the concrete instructions that make up Tip5. Testing different
    /// binaries for equivalent behavior is easiest when that behavior is pinned
    /// through snapshots.
    const MAGIC_SNAPSHOT_HEX: &str =
        "109cc2fe453bd9962f754b96d8f5b919b60af030940a275f5540da195fef65ee651c1b6fa19b2c6a";

    #[macro_rules_attr::apply(test)]
    fn hash10_test_vectors_snapshot() {
        let mut preimage = [BFieldElement::ZERO; RATE];
        for i in 0..6 {
            let digest = Tip5::hash_10(&preimage);
            preimage[i..i + Digest::LEN].copy_from_slice(&digest);
        }
        let final_digest = Digest::new(Tip5::hash_10(&preimage)).to_hex();
        assert_eq!(MAGIC_SNAPSHOT_HEX, final_digest);
    }

    #[macro_rules_attr::apply(test)]
    fn hash_varlen_test_vectors() {
        let mut digest_sum = [BFieldElement::ZERO; Digest::LEN];
        for i in 0..20 {
            let preimage = (0..i).map(BFieldElement::new).collect_vec();
            let digest = Tip5::hash_varlen(&preimage);
            digest_sum
                .iter_mut()
                .zip(digest.values().iter())
                .for_each(|(s, d)| *s += *d);
        }

        let final_digest = Digest::new(digest_sum).to_hex();
        assert_eq!(
            "efbafa86622a9c69652f8a1c4ffd734f021ad23a0a8085412a877de0f9170b18ea4ff69b6fff9a03",
            final_digest,
        );
    }

    #[macro_rules_attr::apply(test)]
    fn snapshot() {
        let state = [
            0x0000_000f_ffff_fff0,
            0x0000_0000_ffff_ffff,
            0x0000_0000_ffff_ffff,
            0x0000_0028_ffff_ffd7,
            0x0000_0006_ffff_fff9,
            0x0000_0002_ffff_fffd,
            0x0000_0000_ffff_ffff,
            0x0000_0030_ffff_ffcf,
            0x0000_0397_ffff_fc68,
            0x0000_000f_ffff_fff0,
            0x316b_fb72_3638_2123,
            0x216f_521b_66ef_83f5,
            0x5689_d7b3_63f5_2df0,
            0xeb2f_59e3_aeae_25fc,
            0xb082_99d2_77cb_b4dc,
            0xcbe3_d9fd_c534_9140,
        ]
        .map(BFieldElement::from_raw_u64);

        let mut tip5 = Tip5 { state };
        tip5.permutation();

        let expected = [
            0x15d3_8ea9_29f6_632a,
            0xf988_e509_ff73_8bb4,
            0x48bc_dfae_88a2_e9f3,
            0x8733_9e83_2daa_c02a,
            0x511e_4126_8150_fdac,
        ]
        .map(BFieldElement::from_raw_u64);

        assert_eq!(&expected, &tip5.state[0..5]);
    }

    fn manual_hash_varlen(preimage: &[BFieldElement]) -> Digest {
        let mut sponge = Tip5::init();
        sponge.pad_and_absorb_all(preimage);
        let squeeze_result = sponge.squeeze();

        Digest::new((&squeeze_result[..Digest::LEN]).try_into().unwrap())
    }

    #[macro_rules_attr::apply(test)]
    fn hash_var_len_equivalence_edge_cases() {
        for preimage_length in 0..=11 {
            let preimage = vec![BFieldElement::new(42); preimage_length];
            let hash_varlen_digest = Tip5::hash_varlen(&preimage);

            let digest_through_pad_squeeze_absorb = manual_hash_varlen(&preimage);
            assert_eq!(digest_through_pad_squeeze_absorb, hash_varlen_digest);
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn hash_var_len_equivalence(#[strategy(arb())] preimage: Vec<BFieldElement>) {
        let hash_varlen_digest = Tip5::hash_varlen(&preimage);
        let digest_through_pad_squeeze_absorb = manual_hash_varlen(&preimage);
        prop_assert_eq!(digest_through_pad_squeeze_absorb, hash_varlen_digest);
    }

    #[macro_rules_attr::apply(test)]
    fn test_linearity_of_mds() {
        type SpongeState = [BFieldElement; STATE_SIZE];

        let mds_procedure = |state| {
            let mut sponge = Tip5 { state };
            sponge.mds_cyclomul();
            sponge.state
        };

        let a: BFieldElement = random_elements(1)[0];
        let b: BFieldElement = random_elements(1)[0];

        let mul_procedure = |u: SpongeState, v: SpongeState| -> SpongeState {
            let mul_result = u.iter().zip(&v).map(|(&uu, &vv)| a * uu + b * vv);
            mul_result.collect_vec().try_into().unwrap()
        };

        let u: SpongeState = random_elements(STATE_SIZE).try_into().unwrap();
        let v: SpongeState = random_elements(STATE_SIZE).try_into().unwrap();
        let w = mul_procedure(u, v);

        let u = mds_procedure(u);
        let v = mds_procedure(v);
        let w = mds_procedure(w);

        let w_ = mul_procedure(u, v);

        assert_eq!(w, w_);
    }

    #[macro_rules_attr::apply(test)]
    fn test_mds_circulancy() {
        let mut sponge = Tip5::init();
        sponge.state = [BFieldElement::ZERO; STATE_SIZE];
        sponge.state[0] = BFieldElement::ONE;

        sponge.mds_cyclomul();

        let mut mat_first_row = [BFieldElement::ZERO; STATE_SIZE];
        mat_first_row[0] = sponge.state[0];
        for (i, first_row_elem) in mat_first_row.iter_mut().enumerate().skip(1) {
            *first_row_elem = sponge.state[STATE_SIZE - i];
        }

        println!(
            "mds matrix first row: {:?}",
            mat_first_row.map(|b| b.value())
        );

        let initial_state: [BFieldElement; STATE_SIZE] =
            random_elements(STATE_SIZE).try_into().unwrap();

        let mut mv = [BFieldElement::ZERO; STATE_SIZE];
        for i in 0..STATE_SIZE {
            for j in 0..STATE_SIZE {
                mv[i] += mat_first_row[(STATE_SIZE - i + j) % STATE_SIZE] * initial_state[j];
            }
        }

        let mut sponge_2 = Tip5::init();
        sponge_2.state = initial_state;
        sponge_2.mds_cyclomul();

        assert_eq!(sponge_2.state, mv);
    }

    #[macro_rules_attr::apply(test)]
    fn test_complex_karatsuba() {
        const N: usize = 4;
        let mut f = [(0i64, 0i64); N];
        let mut g = [(0i64, 0i64); N];
        for i in 0..N {
            f[i].0 = i as i64;
            g[i].0 = 1;
            f[i].1 = 1;
            g[i].1 = i as i64;
        }

        let h0 = Tip5::complex_karatsuba4(f, g);
        let h1 = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 13), (0, 14), (0, 10)];

        assert_eq!(h0, h1);
    }

    #[macro_rules_attr::apply(test)]
    fn test_complex_product() {
        let mut rng = rand::rng();
        let mut random_small_i64 = || (rng.next_u32() % (1 << 16)) as i64;
        for _ in 0..1000 {
            let f = (random_small_i64(), random_small_i64());
            let g = (random_small_i64(), random_small_i64());
            let h0 = Tip5::complex_product(f, g);
            let h1 = (f.0 * g.0 - f.1 * g.1, f.0 * g.1 + f.1 * g.0);
            assert_eq!(h1, h0);
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn sample_scalars(mut tip5: Tip5, #[strategy(0_usize..=100)] num_scalars: usize) {
        tip5.permutation(); // remove any 0s that exist due to shrinking

        let scalars = tip5.sample_scalars(num_scalars);
        prop_assert_eq!(num_scalars, scalars.len());

        let product = scalars
            .into_iter()
            .fold(XFieldElement::ONE, XFieldElement::mul);
        prop_assert_ne!(product, XFieldElement::ZERO);
    }

    // Function `mds_generated` is not available if the AVX-512 functions are.
    #[cfg(not(all(
        target_feature = "avx512ifma",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    )))]
    #[macro_rules_attr::apply(proptest)]
    fn test_mds_matrix_mul_methods_agree(state: [BFieldElement; STATE_SIZE]) {
        let mut sponge_cyclomut = Tip5 { state };
        let mut sponge_generated = Tip5 { state };

        sponge_cyclomut.mds_cyclomul();
        sponge_generated.mds_generated();

        prop_assert_eq!(
            &sponge_cyclomut,
            &sponge_generated,
            "cyclomul =/= generated\n{}\n{}",
            sponge_cyclomut.state.into_iter().join(","),
            sponge_generated.state.into_iter().join(",")
        );
    }

    #[macro_rules_attr::apply(test)]
    fn tip5_hasher_trait_snapshot_test() {
        let mut hasher = Tip5::init();
        let data = b"hello world";
        hasher.write(data);
        assert_eq!(hasher.finish(), 2267905471610932299);
    }

    #[macro_rules_attr::apply(proptest)]
    fn tip5_hasher_consumes_small_data(#[filter(!#bytes.is_empty())] bytes: Vec<u8>) {
        let mut hasher = Tip5::init();
        bytes.hash(&mut hasher);

        prop_assert_ne!(Tip5::init().finish(), hasher.finish());
    }

    #[macro_rules_attr::apply(proptest)]
    fn appending_small_data_to_big_data_changes_tip5_hash(
        #[any(size_range(2_000..8_000).lift())] big_data: Vec<u8>,
        #[filter(!#small_data.is_empty())] small_data: Vec<u8>,
    ) {
        let mut hasher = Tip5::init();
        big_data.hash(&mut hasher);
        let big_data_hash = hasher.finish();

        // finish doesn't terminate the hasher; see it's documentation
        small_data.hash(&mut hasher);
        let all_data_hash = hasher.finish();

        prop_assert_ne!(big_data_hash, all_data_hash);
    }

    #[macro_rules_attr::apply(proptest)]
    fn tip5_trace_starts_with_initial_state_and_is_equivalent_to_permutation(mut tip5: Tip5) {
        let [first, .., last] = tip5.clone().trace();
        prop_assert_eq!(first, tip5.state);

        tip5.permutation();
        prop_assert_eq!(last, tip5.state);
    }
}
