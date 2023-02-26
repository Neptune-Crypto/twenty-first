//! Mock (!) implementations for Poseidon over the Oxfoi field with state size 16. This is used
//! for benchmarking only. DO NOT USE IN PRODUCTION!
//!
//! The matrices `MDS_MATRIX_CIRC` as well as `MDS_MATRIX_DIAG` are incorrect.
//!
//! Adapted from Plonky2: <https://github.com/mir-protocol/plonky2/tree/main/plonky2/src/hash>

use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use unroll::unroll_for_loops;

use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::b_field_element::BFIELD_ONE;
use crate::shared_math::b_field_element::BFIELD_ZERO;
use crate::util_types::algebraic_hasher::Domain;
use crate::util_types::algebraic_hasher::SpongeHasher;

pub const STATE_SIZE: usize = 16;
pub const RATE: usize = 10;
pub const CAPACITY: usize = 6;
pub const DIGEST_LENGTH: usize = 5;
pub(crate) const HALF_N_FULL_ROUNDS: usize = 4;
pub const N_FULL_ROUNDS_TOTAL: usize = 2 * HALF_N_FULL_ROUNDS;
pub const N_PARTIAL_ROUNDS: usize = 22;
pub const N_ROUNDS: usize = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poseidon {}

impl Poseidon {
    // The MDS matrix we use is the circulant matrix with first row given by the vector
    // [ 2^x for x in MDS_MATRIX_EXPS] =
    //      [1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 512, 8388608, 268435456, 128, 8192]
    //
    // WARNING: If the MDS matrix is changed, then the following
    // constants need to be updated accordingly:
    //  - FAST_PARTIAL_ROUND_CONSTANTS
    //  - FAST_PARTIAL_ROUND_VS
    //  - FAST_PARTIAL_ROUND_W_HATS
    //  - FAST_PARTIAL_ROUND_INITIAL_MATRIX

    // const MDS_MATRIX_EXPS: [u64; STATE_SIZE] =
    //     [0, 0, 1, 0, 3, 5, 1, 8, 12, 3, 16, 9, 23, 28, 7, 13];

    // todo this is not correct – how to derive from `MDS_MATRIX_EXPS`?
    const MDS_MATRIX_CIRC: [BFieldElement; STATE_SIZE] = [
        BFieldElement::new(0xf0),
        BFieldElement::new(0xf1),
        BFieldElement::new(0xf2),
        BFieldElement::new(0xf3),
        BFieldElement::new(0xf4),
        BFieldElement::new(0xf5),
        BFieldElement::new(0xf6),
        BFieldElement::new(0xf7),
        BFieldElement::new(0xf8),
        BFieldElement::new(0xf9),
        BFieldElement::new(0xfa),
        BFieldElement::new(0xfb),
        BFieldElement::new(0xfc),
        BFieldElement::new(0xfd),
        BFieldElement::new(0xfe),
        BFieldElement::new(0xff),
    ];
    const MDS_MATRIX_DIAG: [BFieldElement; STATE_SIZE] = [
        BFieldElement::new(42),
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
        BFIELD_ZERO,
    ];

    const FAST_PARTIAL_FIRST_ROUND_CONSTANT: [BFieldElement; STATE_SIZE] = [
        BFieldElement::new(0x645cf794e4da78a9),
        BFieldElement::new(0x392be9bdd5903f18),
        BFieldElement::new(0x1409c3739b635615),
        BFieldElement::new(0x2a9f3fed156b5755),
        BFieldElement::new(0xfdf39b2e242b1dc7),
        BFieldElement::new(0x5c9cc85808664209),
        BFieldElement::new(0x1a6f7d30f8b46734),
        BFieldElement::new(0xe9e5feca5f7b40d6),
        BFieldElement::new(0x84dc6a24e66abbcd),
        BFieldElement::new(0x27021af3acf36b04),
        BFieldElement::new(0x158848f07a40fc3a),
        BFieldElement::new(0x5d0c3d6af25c81c9),
        BFieldElement::new(0x077c237e4bd1cc19),
        BFieldElement::new(0x0cb4d8fda4c80f3b),
        BFieldElement::new(0x60e76018cdac7f8d),
        BFieldElement::new(0x0539751cf45a7ad0),
    ];

    const FAST_PARTIAL_ROUND_CONSTANTS: [BFieldElement; N_PARTIAL_ROUNDS] = [
        BFieldElement::new(0x4c9f7046464c3559),
        BFieldElement::new(0x6fc0fd2090f45d91),
        BFieldElement::new(0x9c3658cdf757b8bb),
        BFieldElement::new(0xe7a3b322dedb3045),
        BFieldElement::new(0x12bceac3cb27ac23),
        BFieldElement::new(0x0e5186c836e1c629),
        BFieldElement::new(0xfc5c824956dbec57),
        BFieldElement::new(0xbfb3184414ade4aa),
        BFieldElement::new(0x5c071800f1c5158b),
        BFieldElement::new(0x37c921bdfda75092),
        BFieldElement::new(0x199d62605d6eb4fc),
        BFieldElement::new(0x6a70e766c461101f),
        BFieldElement::new(0xb2d0c6fb7ac87f00),
        BFieldElement::new(0x3b8d77cb4d4d668a),
        BFieldElement::new(0xa444b110faebf240),
        BFieldElement::new(0x038b641fa8907103),
        BFieldElement::new(0x9aa8885bc0ccd024),
        BFieldElement::new(0x5130b8ac47cba283),
        BFieldElement::new(0x592b25f2410b18aa),
        BFieldElement::new(0xa354b2aa76732058),
        BFieldElement::new(0x8b69b52fa05b7a83),
        BFieldElement::new(0x0000000000000000),
    ];

    const FAST_PARTIAL_ROUND_VS: [[BFieldElement; STATE_SIZE - 1]; N_PARTIAL_ROUNDS] = [
        [
            BFieldElement::new(0xa60354da96f812b5),
            BFieldElement::new(0x01a6a651583700e0),
            BFieldElement::new(0x61fd1cba617fa90f),
            BFieldElement::new(0x8af038fbd1cf7439),
            BFieldElement::new(0x920f515a1e0dbd20),
            BFieldElement::new(0x1786eca3f42b31db),
            BFieldElement::new(0x7f2336653be83a6c),
            BFieldElement::new(0xb8c9c35f0f4e3d54),
            BFieldElement::new(0x5ee8a75f25bdfc27),
            BFieldElement::new(0x3134aca9ca4e4818),
            BFieldElement::new(0x41e45b99ac544f67),
            BFieldElement::new(0x951b5f24b9a90ea5),
            BFieldElement::new(0xef8dadc6fe1a886f),
            BFieldElement::new(0xc8cd89e10b39fc10),
            BFieldElement::new(0x927ac55f25294586),
        ],
        [
            BFieldElement::new(0x38bff2ef2a41a8f7),
            BFieldElement::new(0x2118ad3603c7c22c),
            BFieldElement::new(0x7680123b91586920),
            BFieldElement::new(0x6e26303b9f4af049),
            BFieldElement::new(0x8595716b0f13bfc5),
            BFieldElement::new(0xb6c188401ae6064c),
            BFieldElement::new(0xbff08bbdbc88da41),
            BFieldElement::new(0x77ae43703e4cac45),
            BFieldElement::new(0xb592f9d15324f6d1),
            BFieldElement::new(0x97f27fa48d5f71f7),
            BFieldElement::new(0x72d02c12d282b87f),
            BFieldElement::new(0xf322cd9a8247b275),
            BFieldElement::new(0x2ebfc30690ef7e81),
            BFieldElement::new(0x94daafe87e2492c8),
            BFieldElement::new(0x9e9127bc94a4de4c),
        ],
        [
            BFieldElement::new(0x18f6691436012dc1),
            BFieldElement::new(0x0eaf41161519175e),
            BFieldElement::new(0xd30cbf138949ea7d),
            BFieldElement::new(0x6106cebc905b457a),
            BFieldElement::new(0x33ef6d848899eb7f),
            BFieldElement::new(0x246fa2cb1ba28aa0),
            BFieldElement::new(0xd600ca91764452be),
            BFieldElement::new(0x3903cb162d15d7bd),
            BFieldElement::new(0x8839259c35f34a6c),
            BFieldElement::new(0x376c54b3aa918df1),
            BFieldElement::new(0xe8428e6fbe6f3dde),
            BFieldElement::new(0xcfbcb817525b16ce),
            BFieldElement::new(0x077e76aef6e8ccea),
            BFieldElement::new(0xc8f2799d209e4024),
            BFieldElement::new(0x31421005f0b6188a),
        ],
        [
            BFieldElement::new(0xc40659cef0dcd597),
            BFieldElement::new(0xb6a21d982d27c545),
            BFieldElement::new(0x4c3833d34d2a02da),
            BFieldElement::new(0xdcefeeba14a03e3a),
            BFieldElement::new(0x21d04bbb11b34175),
            BFieldElement::new(0xb0f39bff6773170f),
            BFieldElement::new(0xae6a754e3278edcc),
            BFieldElement::new(0x2063a8497e048e22),
            BFieldElement::new(0xc8d58ddaeced9c67),
            BFieldElement::new(0x1617a03e39577cdc),
            BFieldElement::new(0xfca9ac4db311a41d),
            BFieldElement::new(0xfb644f5352deffc1),
            BFieldElement::new(0xabf4a8b253a22d2e),
            BFieldElement::new(0x7573b0eaea02db08),
            BFieldElement::new(0x775af67e48db3598),
        ],
        [
            BFieldElement::new(0xe406cc1c10084db1),
            BFieldElement::new(0xb0cd5ac3ec340cff),
            BFieldElement::new(0x5ac221a4339910f3),
            BFieldElement::new(0x05bb2cda01400af0),
            BFieldElement::new(0x1b33bb116c9ce157),
            BFieldElement::new(0xbf420f7663b42f43),
            BFieldElement::new(0x1a24164a0b98338a),
            BFieldElement::new(0xa5554af6e01ef10d),
            BFieldElement::new(0x50b83c070e738941),
            BFieldElement::new(0xf5515b9c7c1f83d2),
            BFieldElement::new(0xdcfd683ba349ed3e),
            BFieldElement::new(0x374077ca695b3587),
            BFieldElement::new(0xa470ff0c5c0a7055),
            BFieldElement::new(0xe078f2523f84ab99),
            BFieldElement::new(0x8eb110adf82282ab),
        ],
        [
            BFieldElement::new(0xbf6ef82c00135dda),
            BFieldElement::new(0x640c9685ffa88839),
            BFieldElement::new(0x57676999ce82fcc3),
            BFieldElement::new(0x1aa9f7eae143d8fb),
            BFieldElement::new(0xe9ad55c4075bde4c),
            BFieldElement::new(0x26e9625eb76b55a4),
            BFieldElement::new(0x7c4c55be1dea11b1),
            BFieldElement::new(0xdfc51b16ff7ca899),
            BFieldElement::new(0x2879c711bb6a94f4),
            BFieldElement::new(0x200ca2086c14a180),
            BFieldElement::new(0xccb4baf5fc2f6600),
            BFieldElement::new(0xf701c0fa1faa6124),
            BFieldElement::new(0x640ca9f52afc160b),
            BFieldElement::new(0x6c1285b451481d5a),
            BFieldElement::new(0xb6d9b730d49bf33e),
        ],
        [
            BFieldElement::new(0x8397aa104cbd0e68),
            BFieldElement::new(0xccad525617141b3e),
            BFieldElement::new(0x2a163e08dc82274d),
            BFieldElement::new(0x6e26f9f5dcaee1de),
            BFieldElement::new(0x9f316aea1288c832),
            BFieldElement::new(0xe71ad82e6edb81be),
            BFieldElement::new(0x80152c0ad94f9585),
            BFieldElement::new(0xd59f37872ba4fe41),
            BFieldElement::new(0x36d66846e2ffdea1),
            BFieldElement::new(0x1fc73750d8f7a11b),
            BFieldElement::new(0xfe521f1be30d3f19),
            BFieldElement::new(0xdf57e4331ae02030),
            BFieldElement::new(0x6d80093c26b7be78),
            BFieldElement::new(0x426dadb61efa35d0),
            BFieldElement::new(0x39f114716485328a),
        ],
        [
            BFieldElement::new(0x5771110c58f0e10a),
            BFieldElement::new(0xd8424fdb042de2a4),
            BFieldElement::new(0x93df7ba314e542e3),
            BFieldElement::new(0x63560040bf13163c),
            BFieldElement::new(0x7800f478a3dcd464),
            BFieldElement::new(0x9a7d1bb0f812c6eb),
            BFieldElement::new(0x7ef8168f0eefb440),
            BFieldElement::new(0x52e4a36da5f1d4c5),
            BFieldElement::new(0xdbc59f72cd299d1e),
            BFieldElement::new(0xc15b111586c5976d),
            BFieldElement::new(0xbae596cd7019072c),
            BFieldElement::new(0xad96a0689e493cf9),
            BFieldElement::new(0x6fca05f7312465a1),
            BFieldElement::new(0x52ae62c5cb86845c),
            BFieldElement::new(0x209ffc2bf9c21812),
        ],
        [
            BFieldElement::new(0x01a6c97947026698),
            BFieldElement::new(0x22bae46405d979ed),
            BFieldElement::new(0x52f9a4baf4bfa60c),
            BFieldElement::new(0xe783fe3f4e11703c),
            BFieldElement::new(0x271efb306cb0537f),
            BFieldElement::new(0xbf54e0662e15ef9f),
            BFieldElement::new(0xcba96579dfa448d0),
            BFieldElement::new(0x23c0102ba863536d),
            BFieldElement::new(0x671b82230a0a647b),
            BFieldElement::new(0xd9a12f4e92610880),
            BFieldElement::new(0x307ed9d4f33fa5d2),
            BFieldElement::new(0x89207736aef5063a),
            BFieldElement::new(0xf34be0ba4e74bf1c),
            BFieldElement::new(0x232c02427d3f25ad),
            BFieldElement::new(0xd20614cec0ca4bd9),
        ],
        [
            BFieldElement::new(0xd73d316d171dccfa),
            BFieldElement::new(0xfef971e4815a594b),
            BFieldElement::new(0xad5d0723f177d6e8),
            BFieldElement::new(0x9e8f834a382426af),
            BFieldElement::new(0x8a7bfc71d594e590),
            BFieldElement::new(0x9eeb763ca352da7d),
            BFieldElement::new(0x1e4f5b9a8d9f6c5b),
            BFieldElement::new(0xe2a31c8af22f9129),
            BFieldElement::new(0x50954e84ab0e5a36),
            BFieldElement::new(0x878468911f4e8d65),
            BFieldElement::new(0x4ed1e2784cc7af1d),
            BFieldElement::new(0x70a7720f5aa3157d),
            BFieldElement::new(0x5eaab03e68deef9f),
            BFieldElement::new(0xffeba511c6be8bd1),
            BFieldElement::new(0x8225cbdd6b19e51d),
        ],
        [
            BFieldElement::new(0x60694521bf724c63),
            BFieldElement::new(0x93628a748f157650),
            BFieldElement::new(0xa99c8e8cb8ca3730),
            BFieldElement::new(0x091241fa6bc3fbf8),
            BFieldElement::new(0x8155e2156cac26f9),
            BFieldElement::new(0x22368e7d6740be91),
            BFieldElement::new(0xf86bb1ff62178fb9),
            BFieldElement::new(0x5f08a9a1e4c7d130),
            BFieldElement::new(0xd01c9395d851cbd2),
            BFieldElement::new(0xef2498cb53843534),
            BFieldElement::new(0xbeef1dfa9c185a0b),
            BFieldElement::new(0xcf47ab6dd99c5f5b),
            BFieldElement::new(0x35eb8f6853aea4f7),
            BFieldElement::new(0x2c60dc3acbdaea45),
            BFieldElement::new(0xfbd15651b3fafb32),
        ],
        [
            BFieldElement::new(0x5c83e326655518ad),
            BFieldElement::new(0x652b692833619d89),
            BFieldElement::new(0x84dcee6f713df9c7),
            BFieldElement::new(0x16773c91c16ed1f0),
            BFieldElement::new(0x22a8c2319247a526),
            BFieldElement::new(0x4e1d95cc81f32120),
            BFieldElement::new(0x5824257a89c43f05),
            BFieldElement::new(0x127e53c893c040b0),
            BFieldElement::new(0x0d82a3a1a982cc93),
            BFieldElement::new(0x070c5001f54a446e),
            BFieldElement::new(0x3c1732c56187d025),
            BFieldElement::new(0x0266e361ef354bf9),
            BFieldElement::new(0xf58d7fc305282e81),
            BFieldElement::new(0xc8c3789d5796ad0e),
            BFieldElement::new(0x3a9f900c97722a97),
        ],
        [
            BFieldElement::new(0xe9b321c4639ca431),
            BFieldElement::new(0xfa08af9ae75a9ccd),
            BFieldElement::new(0x6982789e1504d58c),
            BFieldElement::new(0xfac55b5209080b85),
            BFieldElement::new(0x7a07f57b7f2b70e7),
            BFieldElement::new(0x22252da66afaf8e7),
            BFieldElement::new(0x26ddf56827f76003),
            BFieldElement::new(0x2b514dd7f16eed8b),
            BFieldElement::new(0x056efc3622e58b7c),
            BFieldElement::new(0x5a027b41cca55eed),
            BFieldElement::new(0x99996e4885764ef0),
            BFieldElement::new(0xc4330aa68ea0dc45),
            BFieldElement::new(0x940ebad50d2b6841),
            BFieldElement::new(0x6899a147ffffa50f),
            BFieldElement::new(0x9ee2fd8b08eefc5c),
        ],
        [
            BFieldElement::new(0x6b33600744bd1b58),
            BFieldElement::new(0x5f5cde8c774bc3cb),
            BFieldElement::new(0xd3bc3c50e9d4dff9),
            BFieldElement::new(0x0d65494588231750),
            BFieldElement::new(0xa3a34cdc7ddb7bf3),
            BFieldElement::new(0x64b64fe38b97bad1),
            BFieldElement::new(0x5fbad1cde7660497),
            BFieldElement::new(0xb5f5c0b24bd08b7f),
            BFieldElement::new(0x56070b8a2002304d),
            BFieldElement::new(0x4c5552c1f163aa1d),
            BFieldElement::new(0x8f1e4c61a3fa8077),
            BFieldElement::new(0x21b15b8b490f15a2),
            BFieldElement::new(0x6f5950301719ff4c),
            BFieldElement::new(0x113bb49e499bbe7b),
            BFieldElement::new(0x03017a5e0d524789),
        ],
        [
            BFieldElement::new(0xb8b0830b3c79e4cd),
            BFieldElement::new(0x4ddc9964222d03bf),
            BFieldElement::new(0x49ea4445749112d6),
            BFieldElement::new(0xb04f76d2acdb28e6),
            BFieldElement::new(0xad9916138bf08b9f),
            BFieldElement::new(0x2437cabf6bade384),
            BFieldElement::new(0x4d589e0bbe298f22),
            BFieldElement::new(0xa55a60186b28a173),
            BFieldElement::new(0x4f49aa4e83909f1f),
            BFieldElement::new(0xe600b0dca83f635c),
            BFieldElement::new(0x15c4ba5fc4c4100f),
            BFieldElement::new(0x251d71019b1462f9),
            BFieldElement::new(0x5f72842d385b3cd0),
            BFieldElement::new(0x036b49f08330ed6f),
            BFieldElement::new(0x88b119d37bf31025),
        ],
        [
            BFieldElement::new(0x8f4f7841f9787cb1),
            BFieldElement::new(0x5e9f02fcd12a8209),
            BFieldElement::new(0xecbe6e1c199771d1),
            BFieldElement::new(0x2ab8facd72b32ad8),
            BFieldElement::new(0xba2af71b7ee68836),
            BFieldElement::new(0x32133df94440445f),
            BFieldElement::new(0xf6e1c4364b18acca),
            BFieldElement::new(0xfb985a12db3bf841),
            BFieldElement::new(0xb7eaad3675ae8d1e),
            BFieldElement::new(0xe9ead5244f0f5028),
            BFieldElement::new(0xaf5f5b6fea2989d2),
            BFieldElement::new(0x3f38a7f9cb0e53f4),
            BFieldElement::new(0x069a88dd310e3480),
            BFieldElement::new(0xf69ec30c104e8f40),
            BFieldElement::new(0xcc4d7391c8846c39),
        ],
        [
            BFieldElement::new(0x3e5da896b57de77e),
            BFieldElement::new(0x0dab84746f0a766f),
            BFieldElement::new(0xba37115ccb7baffd),
            BFieldElement::new(0xbb4adf5b8aa12cae),
            BFieldElement::new(0xc1d463b6902de770),
            BFieldElement::new(0x77f3ce5060eb8722),
            BFieldElement::new(0xd6beb0a88c3262a9),
            BFieldElement::new(0xf32c5915fec26a82),
            BFieldElement::new(0xe3070dcc88387ceb),
            BFieldElement::new(0xb7055e82bd313436),
            BFieldElement::new(0xac1a698bff99fe94),
            BFieldElement::new(0x840a4e889ccd375c),
            BFieldElement::new(0xf3238286ae444e90),
            BFieldElement::new(0x9285edaf418ad7ab),
            BFieldElement::new(0x4a88ad3fb881df9d),
        ],
        [
            BFieldElement::new(0xd4e8ddb13aced683),
            BFieldElement::new(0x6a70c04347807c2f),
            BFieldElement::new(0xcc020581445551b9),
            BFieldElement::new(0xf7a7fcc4f61ef7a3),
            BFieldElement::new(0x6660bdf6cd5a896e),
            BFieldElement::new(0x482f8da20482718d),
            BFieldElement::new(0x62837bb4a6888b95),
            BFieldElement::new(0x429fd6b05f1bdb60),
            BFieldElement::new(0x5c45d9d31ce54b20),
            BFieldElement::new(0x926c654109823f1c),
            BFieldElement::new(0x09d3dc574302b296),
            BFieldElement::new(0x45d98b1596d8dac9),
            BFieldElement::new(0xb9e036cf92c81cb9),
            BFieldElement::new(0x660d86e4fbaa6536),
            BFieldElement::new(0xbb25a6129f2b08a8),
        ],
        [
            BFieldElement::new(0xc5310c37b00c8479),
            BFieldElement::new(0x25d46f9c49150f53),
            BFieldElement::new(0xc6cf47fb4abe2dd9),
            BFieldElement::new(0x576478ecf8cb6d28),
            BFieldElement::new(0x164d563250f8a77c),
            BFieldElement::new(0x5daeb3b77d188b94),
            BFieldElement::new(0x689582ecc9d13c2f),
            BFieldElement::new(0xb0add9d9db9b2315),
            BFieldElement::new(0xf29d0b5918269748),
            BFieldElement::new(0xfb5931e6d95dfb34),
            BFieldElement::new(0xa2e8e67a2889a41e),
            BFieldElement::new(0xda7e3903fa3325e2),
            BFieldElement::new(0x22f8ede1e1566ceb),
            BFieldElement::new(0xfc605cb4fb5ca3ff),
            BFieldElement::new(0x5e7d60a6278aa6af),
        ],
        [
            BFieldElement::new(0x67214a00e38a3364),
            BFieldElement::new(0x1e92d02edea57578),
            BFieldElement::new(0x023b9deca8cf99fc),
            BFieldElement::new(0x0635431c660086ce),
            BFieldElement::new(0x0424906adadb90f0),
            BFieldElement::new(0x023e964679b0af9c),
            BFieldElement::new(0x0022030e55717229),
            BFieldElement::new(0x8000fb878ec1e0a5),
            BFieldElement::new(0x302850bf82f4a8f7),
            BFieldElement::new(0x0c62aa567d984ef0),
            BFieldElement::new(0x009073caf2f71371),
            BFieldElement::new(0x60c189a1c57127fa),
            BFieldElement::new(0x199620596bbb9b9a),
            BFieldElement::new(0xc240c7dabe907dbc),
            BFieldElement::new(0x300d94ab6e1d4a3b),
        ],
        [
            BFieldElement::new(0x0000000041604120),
            BFieldElement::new(0x0000000026114680),
            BFieldElement::new(0x0000000011260c80),
            BFieldElement::new(0x0000040000858414),
            BFieldElement::new(0x00000030000232a0),
            BFieldElement::new(0x0100000080814630),
            BFieldElement::new(0x001000004002260c),
            BFieldElement::new(0x0000404001021288),
            BFieldElement::new(0x0000200204000994),
            BFieldElement::new(0x0000010100540063),
            BFieldElement::new(0x000002000c018034),
            BFieldElement::new(0x000000310008220e),
            BFieldElement::new(0x0000000140522005),
            BFieldElement::new(0x0000000422044843),
            BFieldElement::new(0x0000000122018901),
        ],
        [
            BFieldElement::new(0x0000000000002000),
            BFieldElement::new(0x0000000000000080),
            BFieldElement::new(0x0000000010000000),
            BFieldElement::new(0x0000000000800000),
            BFieldElement::new(0x0000000000000200),
            BFieldElement::new(0x0000000000010000),
            BFieldElement::new(0x0000000000000008),
            BFieldElement::new(0x0000000000001000),
            BFieldElement::new(0x0000000000000100),
            BFieldElement::new(0x0000000000000002),
            BFieldElement::new(0x0000000000000020),
            BFieldElement::new(0x0000000000000008),
            BFieldElement::new(0x0000000000000001),
            BFieldElement::new(0x0000000000000002),
            BFieldElement::new(0x0000000000000001),
        ],
    ];

    const FAST_PARTIAL_ROUND_W_HATS: [[BFieldElement; STATE_SIZE - 1]; N_PARTIAL_ROUNDS] = [
        [
            BFieldElement::new(0x2d60148c4cd5082a),
            BFieldElement::new(0x667741791cd1260a),
            BFieldElement::new(0xf713b90d037ea1a9),
            BFieldElement::new(0x85ecf17b242cbe89),
            BFieldElement::new(0xdb574e64d38a6fb9),
            BFieldElement::new(0xac86337d8307f4c2),
            BFieldElement::new(0xa739822dca96116e),
            BFieldElement::new(0x209e80c2f58b38b4),
            BFieldElement::new(0x289d186a71301383),
            BFieldElement::new(0x726e1dabbaa941cd),
            BFieldElement::new(0x20a383613efa5c1f),
            BFieldElement::new(0x0a8e09845cceb541),
            BFieldElement::new(0x3fba2767af78fcac),
            BFieldElement::new(0xa7b85fdda403fb51),
            BFieldElement::new(0x450169794edc2c36),
        ],
        [
            BFieldElement::new(0x22a74743e214c927),
            BFieldElement::new(0x68ab7928cbb80023),
            BFieldElement::new(0x49fbd42aacef04e3),
            BFieldElement::new(0xd4347fbcd134def1),
            BFieldElement::new(0x83aa2b7f91c70ab4),
            BFieldElement::new(0x3f850d2300be08fb),
            BFieldElement::new(0x9c9cd1d7bd06050e),
            BFieldElement::new(0xe99f926b834d15f6),
            BFieldElement::new(0x2d5d1334084bdbdc),
            BFieldElement::new(0x79a541ba7a7ede6b),
            BFieldElement::new(0x36be475fe7677a7c),
            BFieldElement::new(0x18dcdef04721c91a),
            BFieldElement::new(0x4a2928d32bad4b42),
            BFieldElement::new(0x47798d0cc771f3b1),
            BFieldElement::new(0xad30254e83e193b6),
        ],
        [
            BFieldElement::new(0x45b78eba6de22df6),
            BFieldElement::new(0x5c465fb45b1b92bc),
            BFieldElement::new(0x4b117fc037028834),
            BFieldElement::new(0xb3d8fa958f164d56),
            BFieldElement::new(0x79f814ab1c5b4e72),
            BFieldElement::new(0x50702775d218f924),
            BFieldElement::new(0x8322396ae7be79e4),
            BFieldElement::new(0x1aa703a0191d28f5),
            BFieldElement::new(0x7700eb7d601c3328),
            BFieldElement::new(0x4ee113edde68be34),
            BFieldElement::new(0x78f5bdd75b592a93),
            BFieldElement::new(0xae936eb2462e0eb2),
            BFieldElement::new(0x82797674d21752ab),
            BFieldElement::new(0xa21190f33035c6c3),
            BFieldElement::new(0xe0eab69b527c8dad),
        ],
        [
            BFieldElement::new(0x6f3267c372108b77),
            BFieldElement::new(0xd7cdd0117eac7231),
            BFieldElement::new(0x5456c4ecf22eeeba),
            BFieldElement::new(0xadee95668c5bcbd9),
            BFieldElement::new(0x05930c4141890f1e),
            BFieldElement::new(0xf7486bbff3ba07a2),
            BFieldElement::new(0x71142628a9cdcebd),
            BFieldElement::new(0x19a45a9477faf592),
            BFieldElement::new(0x69bc0ddbb4d57e7f),
            BFieldElement::new(0xf3d144ddfc1dae59),
            BFieldElement::new(0x833aa1dbdbf1da42),
            BFieldElement::new(0xc095ac91b9d8fbbc),
            BFieldElement::new(0x26933c8f994c394b),
            BFieldElement::new(0x758bc327e1813735),
            BFieldElement::new(0xe48fc29daaf0bc62),
        ],
        [
            BFieldElement::new(0x8b95727898ed9fac),
            BFieldElement::new(0x1001bc460573f6d0),
            BFieldElement::new(0x21fcbedd080e3c54),
            BFieldElement::new(0x44ada324b4b07a6f),
            BFieldElement::new(0x4e03d49cc6134798),
            BFieldElement::new(0x7c839a5bf25c854b),
            BFieldElement::new(0xf33f25d3662178f1),
            BFieldElement::new(0xe2da8f49dca680a3),
            BFieldElement::new(0xf3ce90556c5a8630),
            BFieldElement::new(0xc6861ff50a118cd8),
            BFieldElement::new(0x35fce37c0cd1a20c),
            BFieldElement::new(0xddcda5fbe7e86c15),
            BFieldElement::new(0x316c66eb50cfefa0),
            BFieldElement::new(0xf3d87f4e72f8916c),
            BFieldElement::new(0x8ef8d4b7e9d8f1c5),
        ],
        [
            BFieldElement::new(0xd57c60388b6d12e2),
            BFieldElement::new(0xab9005205acec1db),
            BFieldElement::new(0x1bc613e5ee505f32),
            BFieldElement::new(0x3429cefb69d56531),
            BFieldElement::new(0xc867f26f02258ba9),
            BFieldElement::new(0xc77957bd49d0fc8c),
            BFieldElement::new(0xfa63d42d184dc7c8),
            BFieldElement::new(0xdf2e12da2db6b4c6),
            BFieldElement::new(0x669f7465bbb5ac10),
            BFieldElement::new(0xeb2d996ef6694922),
            BFieldElement::new(0xb7235d883d01e338),
            BFieldElement::new(0xe227e308f210aaa0),
            BFieldElement::new(0x938c33374f78f440),
            BFieldElement::new(0xe00c2db8245b974b),
            BFieldElement::new(0xa75261a8010c9e6e),
        ],
        [
            BFieldElement::new(0x0e734cd5668692cc),
            BFieldElement::new(0x1e1fb141618afbb0),
            BFieldElement::new(0xc32aeb0ae4dbb284),
            BFieldElement::new(0xcc73b1aa907f4e85),
            BFieldElement::new(0xb453271cd2657f8e),
            BFieldElement::new(0x2710326360be0fad),
            BFieldElement::new(0x0023e7fe142d169c),
            BFieldElement::new(0x38b62f37570e4211),
            BFieldElement::new(0x5032bfb13d88028c),
            BFieldElement::new(0xad0b24ef1e74028b),
            BFieldElement::new(0x10b3d92c488f3ff7),
            BFieldElement::new(0xa31bbece8eeb9616),
            BFieldElement::new(0xf803fa0655fbc4b4),
            BFieldElement::new(0xe53ceae022b11d04),
            BFieldElement::new(0x2b752c36effc9f0f),
        ],
        [
            BFieldElement::new(0x64fe8c819e26deaf),
            BFieldElement::new(0xc8d5aea363ec771f),
            BFieldElement::new(0xb68aa800bfd89809),
            BFieldElement::new(0x4d6f6fddc7780041),
            BFieldElement::new(0xdea438efd9e35848),
            BFieldElement::new(0x8a313935b3259d71),
            BFieldElement::new(0x76dd486677c2fe33),
            BFieldElement::new(0x0d0920feb09de869),
            BFieldElement::new(0x1783b0d2f7308520),
            BFieldElement::new(0x3cf6e3685a9bebd2),
            BFieldElement::new(0xe8bf6f819a6d3629),
            BFieldElement::new(0x45494d129a044f38),
            BFieldElement::new(0x9592f2687542a45a),
            BFieldElement::new(0x8308f25103e26dad),
            BFieldElement::new(0x7e6552f0c267a956),
        ],
        [
            BFieldElement::new(0xa5f8771062d614aa),
            BFieldElement::new(0x7a2468590d1a28d0),
            BFieldElement::new(0x8b42914359fa9f3d),
            BFieldElement::new(0xb3ed1f5644cb37be),
            BFieldElement::new(0x31971662fa259395),
            BFieldElement::new(0x2093ea8bb8723727),
            BFieldElement::new(0xc7fb6ca12af52d9e),
            BFieldElement::new(0xf284eb031037de75),
            BFieldElement::new(0x6e89701ead4ed25d),
            BFieldElement::new(0xca8e19ea05b8ac3a),
            BFieldElement::new(0x8d6acc1f07f6e04f),
            BFieldElement::new(0x09eb1837894c7be7),
            BFieldElement::new(0xf9e5f4cd3c34a3c5),
            BFieldElement::new(0x05fff0046f1dd5d1),
            BFieldElement::new(0x8f8c02fd0a62bb7f),
        ],
        [
            BFieldElement::new(0x0fa828a51d0a2a39),
            BFieldElement::new(0x943136cadb32864a),
            BFieldElement::new(0x65c1f909bcaeafff),
            BFieldElement::new(0xd47d026bc56fb120),
            BFieldElement::new(0x26b41887602f3b97),
            BFieldElement::new(0xb4ac1b778c76f73e),
            BFieldElement::new(0x4452d2de946e835b),
            BFieldElement::new(0xe43900fb1fa218e2),
            BFieldElement::new(0x3a23dcc1bf6536f3),
            BFieldElement::new(0x8b06031cb4c13dee),
            BFieldElement::new(0x579fa2c33e0da984),
            BFieldElement::new(0x245104ce239481ad),
            BFieldElement::new(0x0969403ba8a170d8),
            BFieldElement::new(0xd4c75904298d5d31),
            BFieldElement::new(0x9507d4d8469139b7),
        ],
        [
            BFieldElement::new(0x390e4a65f0be23ec),
            BFieldElement::new(0x7eac8e18933ca127),
            BFieldElement::new(0x4d3d43f2a98b04c0),
            BFieldElement::new(0x82e088cc9c5b058d),
            BFieldElement::new(0xfcf8d7247df98d42),
            BFieldElement::new(0xb282c488bcb5e88f),
            BFieldElement::new(0x5c0de586b54b4dd1),
            BFieldElement::new(0x58271c979c1d6a3e),
            BFieldElement::new(0x964c0e2b4f0bfda7),
            BFieldElement::new(0xb3980d80b52160c0),
            BFieldElement::new(0x38f787fda142214e),
            BFieldElement::new(0x347482173432ffaa),
            BFieldElement::new(0xf7895bc0d150cb61),
            BFieldElement::new(0x6737c099f23ce185),
            BFieldElement::new(0x612fd2911073f868),
        ],
        [
            BFieldElement::new(0xb9bf779f95106c83),
            BFieldElement::new(0xc50a3c7c6fad9b98),
            BFieldElement::new(0xebb8aa99ed5de673),
            BFieldElement::new(0x531ce9100957568e),
            BFieldElement::new(0x41c6f8d6072a2fe2),
            BFieldElement::new(0x423f33d65dce2603),
            BFieldElement::new(0xd7bb347a717556c7),
            BFieldElement::new(0xf47846c10a1a9f0d),
            BFieldElement::new(0xdf8e48498ba91fbf),
            BFieldElement::new(0x526a72e1b2c0eaec),
            BFieldElement::new(0xd3dbbb411c7cafbb),
            BFieldElement::new(0x07520a3357fb0b21),
            BFieldElement::new(0x48409553df744959),
            BFieldElement::new(0xa5f3d94b13f131bf),
            BFieldElement::new(0x96c645ed481dc4b6),
        ],
        [
            BFieldElement::new(0x9baaaa23321b9836),
            BFieldElement::new(0x8c1319998301a122),
            BFieldElement::new(0x2b9e1726c50d747b),
            BFieldElement::new(0x4b29f95bf2613480),
            BFieldElement::new(0xfe022c60da0df202),
            BFieldElement::new(0x382b8efcc588da9b),
            BFieldElement::new(0xc1de94a5ed575c44),
            BFieldElement::new(0x33a20e2f3780b431),
            BFieldElement::new(0x5a9befa8888d6369),
            BFieldElement::new(0xa376a7d9c44d0059),
            BFieldElement::new(0xbeccdf2fbff94c21),
            BFieldElement::new(0x18793264400e8f01),
            BFieldElement::new(0x39d639c71a7687df),
            BFieldElement::new(0x6215c0572b3400fa),
            BFieldElement::new(0xf07a72db3842e849),
        ],
        [
            BFieldElement::new(0x1fa23ce9272fe439),
            BFieldElement::new(0x60164c1bc341b949),
            BFieldElement::new(0x235321b12ffa14d5),
            BFieldElement::new(0x1b9033ea97a9bdcf),
            BFieldElement::new(0x20b9ebe4e61c9e86),
            BFieldElement::new(0x5e5d17e6df930898),
            BFieldElement::new(0x0a62f8396676957f),
            BFieldElement::new(0x57bd56f5e7e036b0),
            BFieldElement::new(0xb769a978acf1cb99),
            BFieldElement::new(0x5aa1c3b5d42ba457),
            BFieldElement::new(0x6a032181509cee61),
            BFieldElement::new(0xc44f96d8111e4ba1),
            BFieldElement::new(0x2c688bf1dcfbf1fd),
            BFieldElement::new(0x7782670671bd4338),
            BFieldElement::new(0xcff20372ac960fd9),
        ],
        [
            BFieldElement::new(0xb5c0d3764755fb94),
            BFieldElement::new(0x895bd40e8a88218b),
            BFieldElement::new(0xfa4b16c1321ef0f7),
            BFieldElement::new(0xd349105a7d97edf1),
            BFieldElement::new(0x07cdadcee1ac4f35),
            BFieldElement::new(0x96686d206f4eb17b),
            BFieldElement::new(0xad52fcf4182c854b),
            BFieldElement::new(0xaeb86a622f208bd1),
            BFieldElement::new(0x43857ab356e55a9b),
            BFieldElement::new(0xd6328443b80bcdd2),
            BFieldElement::new(0x3ac21f994e3a0a4f),
            BFieldElement::new(0xd4d7b16eb53e074a),
            BFieldElement::new(0xc845a71556677a6e),
            BFieldElement::new(0xffac9cf44f371069),
            BFieldElement::new(0xcfc7b356123086a0),
        ],
        [
            BFieldElement::new(0xb1f6663b60795f56),
            BFieldElement::new(0x5c14f62cdeed4915),
            BFieldElement::new(0x1ded26eccaedb163),
            BFieldElement::new(0x1c1c9aca01749422),
            BFieldElement::new(0x17da16d96691b0c4),
            BFieldElement::new(0x632c9a0642bcc926),
            BFieldElement::new(0x44ba213a6fd881cc),
            BFieldElement::new(0x074ee28b5644e9f4),
            BFieldElement::new(0x2a574faa86f42a19),
            BFieldElement::new(0x4a2a3e08551e1892),
            BFieldElement::new(0xc02bb9f7ecd966a3),
            BFieldElement::new(0xcc8b1e3a2f282f06),
            BFieldElement::new(0x15a6f59df14d80e3),
            BFieldElement::new(0x58a397ddb5242b74),
            BFieldElement::new(0x666eb2e2945b15b0),
        ],
        [
            BFieldElement::new(0x8e7a3cb535e56f91),
            BFieldElement::new(0x4f1344bcd4336824),
            BFieldElement::new(0x5ec275b4cb44f9f2),
            BFieldElement::new(0x3446d9d262b15ce1),
            BFieldElement::new(0x29b95085f2fa1428),
            BFieldElement::new(0x56b5dc2a90aad746),
            BFieldElement::new(0x141601d326c4c8ab),
            BFieldElement::new(0x37f7921e76f1e0c1),
            BFieldElement::new(0xee4ed130ce593acd),
            BFieldElement::new(0x4aa16bd5091832b9),
            BFieldElement::new(0xe0f55e56aef2dfb4),
            BFieldElement::new(0xb9c9c60229e1f3fc),
            BFieldElement::new(0x3ae2fa50fc458aaa),
            BFieldElement::new(0xd8315eafb0aa2437),
            BFieldElement::new(0xf2014d5228f196e2),
        ],
        [
            BFieldElement::new(0x0f00f47a66b37162),
            BFieldElement::new(0x87239b60001db4c3),
            BFieldElement::new(0x5bc94c46308eaf34),
            BFieldElement::new(0x08b7a148f46ca633),
            BFieldElement::new(0xdd413b335344e434),
            BFieldElement::new(0xf128a7ebf1d297b3),
            BFieldElement::new(0xe6e3787c0760b41e),
            BFieldElement::new(0x1017732597e5c83c),
            BFieldElement::new(0x9f73a89ea69d3cda),
            BFieldElement::new(0x7bef0599a841751b),
            BFieldElement::new(0x8ae7f9e3439513bc),
            BFieldElement::new(0x0e19ea5b1f3052ac),
            BFieldElement::new(0xab86a3f673a67baa),
            BFieldElement::new(0xfb40f524ee227161),
            BFieldElement::new(0x07b83570949382f2),
        ],
        [
            BFieldElement::new(0x45095ed9fd1880c8),
            BFieldElement::new(0xd1c78499a208a931),
            BFieldElement::new(0xefd04edf8100e261),
            BFieldElement::new(0xf6394ae1e7dea88c),
            BFieldElement::new(0xd0f6c399510282b1),
            BFieldElement::new(0x6f261750372e113f),
            BFieldElement::new(0xed93c2a4fcef8c52),
            BFieldElement::new(0x9bf469f377ac0f2b),
            BFieldElement::new(0x145db0c87185b180),
            BFieldElement::new(0x61def4a0ac9d0af8),
            BFieldElement::new(0xfe07bbc60d9d961e),
            BFieldElement::new(0x2571657b9863f28a),
            BFieldElement::new(0x6541f9a3fa69f97f),
            BFieldElement::new(0x262f384801cfd6b9),
            BFieldElement::new(0xab13a52f537bdf8c),
        ],
        [
            BFieldElement::new(0x11d21514dd2b84fa),
            BFieldElement::new(0x82e1461815b2458b),
            BFieldElement::new(0xcc81638bfbacd3af),
            BFieldElement::new(0x3a37e81dd745d6de),
            BFieldElement::new(0x1d60dad00f8d0bb6),
            BFieldElement::new(0x3d63b6b948b01050),
            BFieldElement::new(0x2f2706bf5c4ab8e9),
            BFieldElement::new(0x7a9270bd4feb0354),
            BFieldElement::new(0xfab1cc8a7d38939d),
            BFieldElement::new(0xd29d3c104ab6c600),
            BFieldElement::new(0x2b5fd5e308a6b0c7),
            BFieldElement::new(0x64029dfce61d9cf5),
            BFieldElement::new(0xe545b2e57bbcffc4),
            BFieldElement::new(0x7562bca95251fbee),
            BFieldElement::new(0x9229d7b4238f1cb0),
        ],
        [
            BFieldElement::new(0xbc31fd2bd84efacc),
            BFieldElement::new(0xb296e44a45df1c52),
            BFieldElement::new(0xa723b0a363de88d4),
            BFieldElement::new(0xdd5bb85035846b74),
            BFieldElement::new(0x417adb2dc350f749),
            BFieldElement::new(0xb49931f7d91c7150),
            BFieldElement::new(0x707d338afbada578),
            BFieldElement::new(0x0f18559abbaa122b),
            BFieldElement::new(0x1b01adfcad87e87e),
            BFieldElement::new(0x2c4863f68fd7e527),
            BFieldElement::new(0xb6132d64bd4899d3),
            BFieldElement::new(0x444bc447bfc222f9),
            BFieldElement::new(0x59044b1254d00abb),
            BFieldElement::new(0xfce6e87a7ff25122),
            BFieldElement::new(0x4bb71228721c4b8c),
        ],
        [
            BFieldElement::new(0x66d02cf5113faa91),
            BFieldElement::new(0x0cd886fb10e216d3),
            BFieldElement::new(0xc14bac3080b1e91b),
            BFieldElement::new(0x1b31c4364de77ff6),
            BFieldElement::new(0xb4c152ea3aca17f7),
            BFieldElement::new(0x45c7e2e33ba91e9f),
            BFieldElement::new(0x6ec0f4d0cc4c2120),
            BFieldElement::new(0xb4b51fca3ba6d78b),
            BFieldElement::new(0x7ba0fd6b55737244),
            BFieldElement::new(0x3aa869b54906c553),
            BFieldElement::new(0xb0fe247604a4a01a),
            BFieldElement::new(0xa003c912762eac32),
            BFieldElement::new(0x30e3593ed4d91a3e),
            BFieldElement::new(0x62251eefd6bb940f),
            BFieldElement::new(0xec1d9ac18f2f5237),
        ],
    ];

    // NB: This is in ROW-major order to support cache-friendly pre-multiplication.
    const FAST_PARTIAL_ROUND_INITIAL_MATRIX: [[BFieldElement; STATE_SIZE - 1]; STATE_SIZE - 1] = [
        [
            BFieldElement::new(0xd756fafac8a3a7aa),
            BFieldElement::new(0x713d815f3bafde52),
            BFieldElement::new(0x92ebdd13a760bc25),
            BFieldElement::new(0x7858308eb5c308a9),
            BFieldElement::new(0x3b1c6b633070b180),
            BFieldElement::new(0x6e38265c7fa42f3c),
            BFieldElement::new(0x22ba9842585260eb),
            BFieldElement::new(0xfd8bbeda359fbe73),
            BFieldElement::new(0x00e8678f34ab75f5),
            BFieldElement::new(0xb4fb6ec8fbb050e1),
            BFieldElement::new(0xb56e63be5b7ce802),
            BFieldElement::new(0x0eb6af006f9eca7c),
            BFieldElement::new(0x6d81695e2816a204),
            BFieldElement::new(0x2145f290a89bd153),
            BFieldElement::new(0xd243c0ed57dc7fa3),
        ],
        [
            BFieldElement::new(0x03b50a498cec5034),
            BFieldElement::new(0xad488af67f375ea4),
            BFieldElement::new(0xfdd57a2b15705051),
            BFieldElement::new(0xf1fdb44479f0960e),
            BFieldElement::new(0x6bfadb2ead07efc8),
            BFieldElement::new(0xd1b05a2d115dafd9),
            BFieldElement::new(0x6fde2d137959faf0),
            BFieldElement::new(0x07e7b57830140488),
            BFieldElement::new(0x7c6bc4d292b5fdc8),
            BFieldElement::new(0xd8e521a6be2f187f),
            BFieldElement::new(0xddbf0afe7560201c),
            BFieldElement::new(0x77545f125ec817bf),
            BFieldElement::new(0xa4ecb852c2f0597a),
            BFieldElement::new(0x347b23de024f2590),
            BFieldElement::new(0x2145f290a89bd153),
        ],
        [
            BFieldElement::new(0x657087520afb6f88),
            BFieldElement::new(0x6c31b813a7fd17d1),
            BFieldElement::new(0x9321b7824174bbdf),
            BFieldElement::new(0x482b42a968d1683e),
            BFieldElement::new(0x9df98d7bbf195c5e),
            BFieldElement::new(0x00f1766b40c1a85e),
            BFieldElement::new(0x39177ef5c385005b),
            BFieldElement::new(0x91365e253af7a930),
            BFieldElement::new(0x7b871daaaa1818c5),
            BFieldElement::new(0xa366987e611fda51),
            BFieldElement::new(0x31dcd6976985be96),
            BFieldElement::new(0xcc0df32e8eb6e65e),
            BFieldElement::new(0x3c6c6f26112f322e),
            BFieldElement::new(0xa4ecb852c2f0597a),
            BFieldElement::new(0x6d81695e2816a204),
        ],
        [
            BFieldElement::new(0xfd3107339673574c),
            BFieldElement::new(0xb2de73088a0f2d73),
            BFieldElement::new(0xab49c439984e6aa6),
            BFieldElement::new(0xd6803e24ba7da994),
            BFieldElement::new(0x2f1d1f9b383682ed),
            BFieldElement::new(0xf3ea5daa49881fd0),
            BFieldElement::new(0xc9834b60252d38b3),
            BFieldElement::new(0xa4bcddc1024d6d09),
            BFieldElement::new(0x53758663681cbd36),
            BFieldElement::new(0xd430bf465e4cd51d),
            BFieldElement::new(0xaa3bc7ed1a368194),
            BFieldElement::new(0x2b3273aef05750d2),
            BFieldElement::new(0xcc0df32e8eb6e65e),
            BFieldElement::new(0x77545f125ec817bf),
            BFieldElement::new(0x0eb6af006f9eca7c),
        ],
        [
            BFieldElement::new(0x588ebaa5e80bf02c),
            BFieldElement::new(0x42947c78fb668a0d),
            BFieldElement::new(0x2fcd63b4bc0a2ad9),
            BFieldElement::new(0xacd68ac4c6a18b0b),
            BFieldElement::new(0x7382e2036ffc227e),
            BFieldElement::new(0xdaf63f8e0c3db058),
            BFieldElement::new(0xe13572c8337230be),
            BFieldElement::new(0x78eb39fafdb4eea1),
            BFieldElement::new(0xb66fd6e505128426),
            BFieldElement::new(0xfb14967766b9e394),
            BFieldElement::new(0xe26e89cd7940cedf),
            BFieldElement::new(0xaa3bc7ed1a368194),
            BFieldElement::new(0x31dcd6976985be96),
            BFieldElement::new(0xddbf0afe7560201c),
            BFieldElement::new(0xb56e63be5b7ce802),
        ],
        [
            BFieldElement::new(0x0aade53dbbf4b21d),
            BFieldElement::new(0xe316dd22f616e0cb),
            BFieldElement::new(0xdb9a2350898d3611),
            BFieldElement::new(0x93688668f5e92e0b),
            BFieldElement::new(0xc4dbda101e3c11fb),
            BFieldElement::new(0x924e561dbce3dbb2),
            BFieldElement::new(0xa9e01e71795b9ffa),
            BFieldElement::new(0x659ba620ebd18225),
            BFieldElement::new(0x57c96b901f7a9bf6),
            BFieldElement::new(0x1a18215205d43bd5),
            BFieldElement::new(0xfb14967766b9e394),
            BFieldElement::new(0xd430bf465e4cd51d),
            BFieldElement::new(0xa366987e611fda51),
            BFieldElement::new(0xd8e521a6be2f187f),
            BFieldElement::new(0xb4fb6ec8fbb050e1),
        ],
        [
            BFieldElement::new(0x2aec1142c6ef0c0c),
            BFieldElement::new(0xc47bdc5f484f2f63),
            BFieldElement::new(0x66a47c85f44d4fb4),
            BFieldElement::new(0x45ee87cb99584b31),
            BFieldElement::new(0x335fa58dae6b7776),
            BFieldElement::new(0xf5478845e264c519),
            BFieldElement::new(0x58cb472e90ab672c),
            BFieldElement::new(0x7d8a5127b11b69c9),
            BFieldElement::new(0xdb176abe361e4769),
            BFieldElement::new(0x57c96b901f7a9bf6),
            BFieldElement::new(0xb66fd6e505128426),
            BFieldElement::new(0x53758663681cbd36),
            BFieldElement::new(0x7b871daaaa1818c5),
            BFieldElement::new(0x7c6bc4d292b5fdc8),
            BFieldElement::new(0x00e8678f34ab75f5),
        ],
        [
            BFieldElement::new(0xd39e37653d275b9e),
            BFieldElement::new(0x3f1cfe4a3e2b00bb),
            BFieldElement::new(0xa264760c42358471),
            BFieldElement::new(0x5c0f52d30ea26b25),
            BFieldElement::new(0x0a37a357283a737e),
            BFieldElement::new(0xabf6fa3871f04fc4),
            BFieldElement::new(0xe9cc65038c753215),
            BFieldElement::new(0xc9c34d6eaee6b538),
            BFieldElement::new(0x7d8a5127b11b69c9),
            BFieldElement::new(0x659ba620ebd18225),
            BFieldElement::new(0x78eb39fafdb4eea1),
            BFieldElement::new(0xa4bcddc1024d6d09),
            BFieldElement::new(0x91365e253af7a930),
            BFieldElement::new(0x07e7b57830140488),
            BFieldElement::new(0xfd8bbeda359fbe73),
        ],
        [
            BFieldElement::new(0xcdd4ecd232a5160b),
            BFieldElement::new(0xb6c8ca789dc7f9b7),
            BFieldElement::new(0x7ef48920e3132097),
            BFieldElement::new(0x9da73c8dd9cb28be),
            BFieldElement::new(0x843901405c263281),
            BFieldElement::new(0x00adbdb561a20293),
            BFieldElement::new(0xb3449cf3d6fc96da),
            BFieldElement::new(0xe9cc65038c753215),
            BFieldElement::new(0x58cb472e90ab672c),
            BFieldElement::new(0xa9e01e71795b9ffa),
            BFieldElement::new(0xe13572c8337230be),
            BFieldElement::new(0xc9834b60252d38b3),
            BFieldElement::new(0x39177ef5c385005b),
            BFieldElement::new(0x6fde2d137959faf0),
            BFieldElement::new(0x22ba9842585260eb),
        ],
        [
            BFieldElement::new(0x085cf385d8e98d34),
            BFieldElement::new(0x1daaa15d577a6fd6),
            BFieldElement::new(0x4e9434037d1946d5),
            BFieldElement::new(0x6d6dfbe10f93903c),
            BFieldElement::new(0xa3e0a11d7e843d91),
            BFieldElement::new(0xd2671d511914a16f),
            BFieldElement::new(0x00adbdb561a20293),
            BFieldElement::new(0xabf6fa3871f04fc4),
            BFieldElement::new(0xf5478845e264c519),
            BFieldElement::new(0x924e561dbce3dbb2),
            BFieldElement::new(0xdaf63f8e0c3db058),
            BFieldElement::new(0xf3ea5daa49881fd0),
            BFieldElement::new(0x00f1766b40c1a85e),
            BFieldElement::new(0xd1b05a2d115dafd9),
            BFieldElement::new(0x6e38265c7fa42f3c),
        ],
        [
            BFieldElement::new(0xc24f53bd665e9902),
            BFieldElement::new(0x42d83c2c151777a6),
            BFieldElement::new(0x84877a08c57c13ab),
            BFieldElement::new(0x4724bf4e6bfdef2e),
            BFieldElement::new(0xa79b2daf63f2d452),
            BFieldElement::new(0xa3e0a11d7e843d91),
            BFieldElement::new(0x843901405c263281),
            BFieldElement::new(0x0a37a357283a737e),
            BFieldElement::new(0x335fa58dae6b7776),
            BFieldElement::new(0xc4dbda101e3c11fb),
            BFieldElement::new(0x7382e2036ffc227e),
            BFieldElement::new(0x2f1d1f9b383682ed),
            BFieldElement::new(0x9df98d7bbf195c5e),
            BFieldElement::new(0x6bfadb2ead07efc8),
            BFieldElement::new(0x3b1c6b633070b180),
        ],
        [
            BFieldElement::new(0xab6f71abdc6a97e7),
            BFieldElement::new(0xeb6e01da906998bb),
            BFieldElement::new(0x957885c6054eb7ed),
            BFieldElement::new(0x2110bb7a7106506d),
            BFieldElement::new(0x4724bf4e6bfdef2e),
            BFieldElement::new(0x6d6dfbe10f93903c),
            BFieldElement::new(0x9da73c8dd9cb28be),
            BFieldElement::new(0x5c0f52d30ea26b25),
            BFieldElement::new(0x45ee87cb99584b31),
            BFieldElement::new(0x93688668f5e92e0b),
            BFieldElement::new(0xacd68ac4c6a18b0b),
            BFieldElement::new(0xd6803e24ba7da994),
            BFieldElement::new(0x482b42a968d1683e),
            BFieldElement::new(0xf1fdb44479f0960e),
            BFieldElement::new(0x7858308eb5c308a9),
        ],
        [
            BFieldElement::new(0xf719a31143815f95),
            BFieldElement::new(0x5f1908823b0a2b83),
            BFieldElement::new(0x7c0ab23404c72581),
            BFieldElement::new(0x957885c6054eb7ed),
            BFieldElement::new(0x84877a08c57c13ab),
            BFieldElement::new(0x4e9434037d1946d5),
            BFieldElement::new(0x7ef48920e3132097),
            BFieldElement::new(0xa264760c42358471),
            BFieldElement::new(0x66a47c85f44d4fb4),
            BFieldElement::new(0xdb9a2350898d3611),
            BFieldElement::new(0x2fcd63b4bc0a2ad9),
            BFieldElement::new(0xab49c439984e6aa6),
            BFieldElement::new(0x9321b7824174bbdf),
            BFieldElement::new(0xfdd57a2b15705051),
            BFieldElement::new(0x92ebdd13a760bc25),
        ],
        [
            BFieldElement::new(0xef2883fa2e946996),
            BFieldElement::new(0x8a674e9eb07ba43e),
            BFieldElement::new(0x5f1908823b0a2b83),
            BFieldElement::new(0xeb6e01da906998bb),
            BFieldElement::new(0x42d83c2c151777a6),
            BFieldElement::new(0x1daaa15d577a6fd6),
            BFieldElement::new(0xb6c8ca789dc7f9b7),
            BFieldElement::new(0x3f1cfe4a3e2b00bb),
            BFieldElement::new(0xc47bdc5f484f2f63),
            BFieldElement::new(0xe316dd22f616e0cb),
            BFieldElement::new(0x42947c78fb668a0d),
            BFieldElement::new(0xb2de73088a0f2d73),
            BFieldElement::new(0x6c31b813a7fd17d1),
            BFieldElement::new(0xad488af67f375ea4),
            BFieldElement::new(0x713d815f3bafde52),
        ],
        [
            BFieldElement::new(0x0bcd59652a8446d1),
            BFieldElement::new(0xef2883fa2e946996),
            BFieldElement::new(0xf719a31143815f95),
            BFieldElement::new(0xab6f71abdc6a97e7),
            BFieldElement::new(0xc24f53bd665e9902),
            BFieldElement::new(0x085cf385d8e98d34),
            BFieldElement::new(0xcdd4ecd232a5160b),
            BFieldElement::new(0xd39e37653d275b9e),
            BFieldElement::new(0x2aec1142c6ef0c0c),
            BFieldElement::new(0x0aade53dbbf4b21d),
            BFieldElement::new(0x588ebaa5e80bf02c),
            BFieldElement::new(0xfd3107339673574c),
            BFieldElement::new(0x657087520afb6f88),
            BFieldElement::new(0x03b50a498cec5034),
            BFieldElement::new(0xd756fafac8a3a7aa),
        ],
    ];

    const ALL_ROUND_CONSTANTS: [BFieldElement; STATE_SIZE * N_ROUNDS] = [
        BFieldElement::new(0xb585f767417ee042),
        BFieldElement::new(0x7746a55f77c10331),
        BFieldElement::new(0xb2fb0d321d356f7a),
        BFieldElement::new(0x0f6760a486f1621f),
        BFieldElement::new(0xe10d6666b36abcdf),
        BFieldElement::new(0x8cae14cb455cc50b),
        BFieldElement::new(0xd438539cf2cee334),
        BFieldElement::new(0xef781c7d4c1fd8b4),
        BFieldElement::new(0xcdc4a23a0aca4b1f),
        BFieldElement::new(0x277fa208d07b52e3),
        BFieldElement::new(0xe17653a300493d38),
        BFieldElement::new(0xc54302f27c287dc1),
        BFieldElement::new(0x8628782231d47d10),
        BFieldElement::new(0x59cd1a8a690b49f2),
        BFieldElement::new(0xc3b919ad9efec0b0),
        BFieldElement::new(0xa484c4c637641d97),
        BFieldElement::new(0x308bbd23f191398b),
        BFieldElement::new(0x6e4a40c1bf713cf1),
        BFieldElement::new(0x9a2eedb7510414fb),
        BFieldElement::new(0xe360c6e111c2c63b),
        BFieldElement::new(0xd5c771901d4d89aa),
        BFieldElement::new(0xc35eae076e7d6b2f),
        BFieldElement::new(0x849c2656d0a09cad),
        BFieldElement::new(0xc0572c8c5cf1df2b),
        BFieldElement::new(0xe9fa634a883b8bf3),
        BFieldElement::new(0xf56f6d4900fb1fdd),
        BFieldElement::new(0xf7d713e872a72a1b),
        BFieldElement::new(0x8297132b6ba47612),
        BFieldElement::new(0xad6805e12ee8af1c),
        BFieldElement::new(0xac51d9f6485c22b9),
        BFieldElement::new(0x502ad7dc3bd56bf8),
        BFieldElement::new(0x57a1550c3761c577),
        BFieldElement::new(0x66bbd30e99d311da),
        BFieldElement::new(0x0da2abef5e948f87),
        BFieldElement::new(0xf0612750443f8e94),
        BFieldElement::new(0x28b8ec3afb937d8c),
        BFieldElement::new(0x92a756e6be54ca18),
        BFieldElement::new(0x70e741ec304e925d),
        BFieldElement::new(0x019d5ee2b037c59f),
        BFieldElement::new(0x6f6f2ed7a30707d1),
        BFieldElement::new(0x7cf416d01e8c169c),
        BFieldElement::new(0x61df517bb17617df),
        BFieldElement::new(0x85dc499b4c67dbaa),
        BFieldElement::new(0x4b959b48dad27b23),
        BFieldElement::new(0xe8be3e5e0dd779a0),
        BFieldElement::new(0xf5c0bc1e525ed8e6),
        BFieldElement::new(0x40b12cbf263cf853),
        BFieldElement::new(0xa637093f13e2ea3c),
        BFieldElement::new(0x3cc3f89232e3b0c8),
        BFieldElement::new(0x2e479dc16bfe86c0),
        BFieldElement::new(0x6f49de07d6d39469),
        BFieldElement::new(0x213ce7beecc232de),
        BFieldElement::new(0x5b043134851fc00a),
        BFieldElement::new(0xa2de45784a861506),
        BFieldElement::new(0x7103aaf97bed8dd5),
        BFieldElement::new(0x5326fc0dbb88a147),
        BFieldElement::new(0xa9ceb750364cb77a),
        BFieldElement::new(0x27f8ec88cc9e991f),
        BFieldElement::new(0xfceb4fda8c93fb83),
        BFieldElement::new(0xfac6ff13b45b260e),
        BFieldElement::new(0x7131aa455813380b),
        BFieldElement::new(0x93510360d5d68119),
        BFieldElement::new(0xad535b24fb96e3db),
        BFieldElement::new(0x4627f5c6b7efc045),
        BFieldElement::new(0x645cf794e4da78a9),
        BFieldElement::new(0x241c70ed1ac2877f),
        BFieldElement::new(0xacb8e076b009e825),
        BFieldElement::new(0x3737e9db6477bd9d),
        BFieldElement::new(0xe7ea5e344cd688ed),
        BFieldElement::new(0x90dee4a009214640),
        BFieldElement::new(0xd1b1edf7c77e74af),
        BFieldElement::new(0x0b65481bab42158e),
        BFieldElement::new(0x99ad1aab4b4fe3e7),
        BFieldElement::new(0x438a7c91f1a360cd),
        BFieldElement::new(0xb60de3bd159088bf),
        BFieldElement::new(0xc99cab6b47a3e3bb),
        BFieldElement::new(0x69a5ed92d5677cef),
        BFieldElement::new(0x5e7b329c482a9396),
        BFieldElement::new(0x5fc0ac0829f893c9),
        BFieldElement::new(0x32db82924fb757ea),
        BFieldElement::new(0x0ade699c5cf24145),
        BFieldElement::new(0x7cc5583b46d7b5bb),
        BFieldElement::new(0x85df9ed31bf8abcb),
        BFieldElement::new(0x6604df501ad4de64),
        BFieldElement::new(0xeb84f60941611aec),
        BFieldElement::new(0xda60883523989bd4),
        BFieldElement::new(0x8f97fe40bf3470bf),
        BFieldElement::new(0xa93f485ce0ff2b32),
        BFieldElement::new(0x6704e8eebc2afb4b),
        BFieldElement::new(0xcee3e9ac788ad755),
        BFieldElement::new(0x510d0e66062a270d),
        BFieldElement::new(0xf6323f48d74634a0),
        BFieldElement::new(0x0b508cdf04990c90),
        BFieldElement::new(0xf241708a4ef7ddf9),
        BFieldElement::new(0x60e75c28bb368f82),
        BFieldElement::new(0xa6217d8c3f0f9989),
        BFieldElement::new(0x7159cd30f5435b53),
        BFieldElement::new(0x839b4e8fe97ec79f),
        BFieldElement::new(0x0d3f3e5e885db625),
        BFieldElement::new(0x8f7d83be1daea54b),
        BFieldElement::new(0x780f22441e8dbc04),
        BFieldElement::new(0xeb9158465aedacd3),
        BFieldElement::new(0xd19e120d826c1b6c),
        BFieldElement::new(0x016ee53a7f007110),
        BFieldElement::new(0xcb5fd54ed22dd1ca),
        BFieldElement::new(0xacb84178c58de144),
        BFieldElement::new(0x9c22190c2c463227),
        BFieldElement::new(0x5d693c1bcc98406d),
        BFieldElement::new(0xdcef0798235f321a),
        BFieldElement::new(0x3d639263f55e0b1e),
        BFieldElement::new(0xe273fd977edb8fda),
        BFieldElement::new(0x418f027049d10fe7),
        BFieldElement::new(0x8c25fda3f253a284),
        BFieldElement::new(0x2cbaed4dc25a884e),
        BFieldElement::new(0x5f58e6aff78dc2af),
        BFieldElement::new(0x284650ac6fb9d206),
        BFieldElement::new(0x635b337f1391c13c),
        BFieldElement::new(0x9f9a036f1ac6361f),
        BFieldElement::new(0xb93e260cff6747b4),
        BFieldElement::new(0xb0a7eae8c7272e33),
        BFieldElement::new(0xd0762cbce7da0a9f),
        BFieldElement::new(0x34c6efb829c754d6),
        BFieldElement::new(0x40bf0ab6166855c1),
        BFieldElement::new(0xb6b570fccc46a242),
        BFieldElement::new(0x5a27b90055549545),
        BFieldElement::new(0xb1a5b166048b306f),
        BFieldElement::new(0x8722e0ad24f1006d),
        BFieldElement::new(0x788ee3b3b315049a),
        BFieldElement::new(0x14a726661e5b0351),
        BFieldElement::new(0x98b7672fe1c3f13e),
        BFieldElement::new(0xbb93ae77bdc3aa8f),
        BFieldElement::new(0x28fd3b04756fc222),
        BFieldElement::new(0x30a46805a86d7109),
        BFieldElement::new(0x337dc00c7844a0e7),
        BFieldElement::new(0xd5eca245253c861b),
        BFieldElement::new(0x77626382990d8546),
        BFieldElement::new(0xc1e434bf33c3ae7a),
        BFieldElement::new(0x0299351a54dbf35e),
        BFieldElement::new(0xb2d456e4fb620184),
        BFieldElement::new(0x3e9ed1fdc00265ea),
        BFieldElement::new(0x2972a92bb672e8db),
        BFieldElement::new(0x20216dd789f333ec),
        BFieldElement::new(0xadffe8cf746494a1),
        BFieldElement::new(0x1c4dbb1c5889d420),
        BFieldElement::new(0x15a16a8a8c9972f5),
        BFieldElement::new(0x388a128b98960e26),
        BFieldElement::new(0x2300e5d6ca3e5589),
        BFieldElement::new(0x2f63aa865c9ceb9f),
        BFieldElement::new(0xf1c36ce8d894420f),
        BFieldElement::new(0x271811252953f84a),
        BFieldElement::new(0xe5840293d5466a8e),
        BFieldElement::new(0x4d9bbc3e24e5f20e),
        BFieldElement::new(0xea35bc29cfa2794b),
        BFieldElement::new(0x18e21b4bf59e2d28),
        BFieldElement::new(0x1e3b9fc632ef6adb),
        BFieldElement::new(0x25d643627a05e678),
        BFieldElement::new(0x5a3f1bb1ecb63263),
        BFieldElement::new(0xdb7f0238ca031e31),
        BFieldElement::new(0xb462065960bfc4c4),
        BFieldElement::new(0x49c24ae463c280f4),
        BFieldElement::new(0xd793862c6f7b901a),
        BFieldElement::new(0xaadd1106bdce475e),
        BFieldElement::new(0xc43b6e0eed8ad58f),
        BFieldElement::new(0xe29024c1f2060cb7),
        BFieldElement::new(0x5e50c2755efbe17a),
        BFieldElement::new(0x10383f20ac183625),
        BFieldElement::new(0x38e8ee9d8a8a435d),
        BFieldElement::new(0xdd511837bcc52452),
        BFieldElement::new(0x7750059861a7da6a),
        BFieldElement::new(0x86ab99b518d1dbef),
        BFieldElement::new(0xb1204f608ccfe33b),
        BFieldElement::new(0xef61ac84d8dfca49),
        BFieldElement::new(0x1bbcd90f1f4eff36),
        BFieldElement::new(0x0cd1dabd9be9850a),
        BFieldElement::new(0x11a3ae5bf354bb11),
        BFieldElement::new(0xf755bfef11bb5516),
        BFieldElement::new(0xa3b832506e2f3adb),
        BFieldElement::new(0x516306f4b617e6ba),
        BFieldElement::new(0xddb4ac4a2aeead3a),
        BFieldElement::new(0x64bb6dec62af4430),
        BFieldElement::new(0xf9cc95c29895a152),
        BFieldElement::new(0x08d37f75632771b9),
        BFieldElement::new(0xeec49b619cee6b56),
        BFieldElement::new(0xf143933b56b3711a),
        BFieldElement::new(0xe4c5dd82b9f6570c),
        BFieldElement::new(0xe7ad775756eefdc4),
        BFieldElement::new(0x92c2318bc834ef78),
        BFieldElement::new(0x739c25f93007aa0a),
        BFieldElement::new(0x5636caca1725f788),
        BFieldElement::new(0xdd8f909af47cd0b6),
        BFieldElement::new(0xc6401fe16bc24d4e),
        BFieldElement::new(0x8ad97b342e6b3a3c),
        BFieldElement::new(0x0c49366bb7be8ce2),
        BFieldElement::new(0x0784d3d2f4b39fb5),
        BFieldElement::new(0x530fb67ec5d77a58),
        BFieldElement::new(0x41049229b8221f3b),
        BFieldElement::new(0x139542347cb606a3),
        BFieldElement::new(0x9cb0bd5ee62e6438),
        BFieldElement::new(0x02e3f615c4d3054a),
        BFieldElement::new(0x985d4f4adefb64a0),
        BFieldElement::new(0x775b9feb32053cde),
        BFieldElement::new(0x304265a64d6c1ba6),
        BFieldElement::new(0x593664c3be7acd42),
        BFieldElement::new(0x4f0a2e5fd2bd6718),
        BFieldElement::new(0xdd611f10619bf1da),
        BFieldElement::new(0xd8185f9b3e74f9a4),
        BFieldElement::new(0xef87139d126ec3b3),
        BFieldElement::new(0x3ba71336dd67f99b),
        BFieldElement::new(0x7d3a455d8d808091),
        BFieldElement::new(0x660d32e15cbdecc7),
        BFieldElement::new(0x297a863f5af2b9ff),
        BFieldElement::new(0x90e0a736e6b434df),
        BFieldElement::new(0x549f80ce7a12182e),
        BFieldElement::new(0x0f73b29235fb5b84),
        BFieldElement::new(0x16bf1f74056e3a01),
        BFieldElement::new(0x6d1f5a593019a39f),
        BFieldElement::new(0x02ff876fa73f6305),
        BFieldElement::new(0xc5cb72a2fb9a5bd7),
        BFieldElement::new(0x8470f39d674dfaa3),
        BFieldElement::new(0x25abb3f1e41aea30),
        BFieldElement::new(0x23eb8cc9c32951c7),
        BFieldElement::new(0xd687ba56242ac4ea),
        BFieldElement::new(0xda8d9e915d2de6b7),
        BFieldElement::new(0xe3cbdc7d938d8f1e),
        BFieldElement::new(0xb9a8c9b4001efad6),
        BFieldElement::new(0xc0d28a5c64f2285c),
        BFieldElement::new(0x45d7ac9b878575b8),
        BFieldElement::new(0xeeb76e39d8da283e),
        BFieldElement::new(0x3d06c8bd2fc7daac),
        BFieldElement::new(0x9c9c9820c13589f5),
        BFieldElement::new(0x65700b51db40bae3),
        BFieldElement::new(0x911f451579044242),
        BFieldElement::new(0x7ae6849ff1fee8cc),
        BFieldElement::new(0x3bb340ebba896ae5),
        BFieldElement::new(0xb46e9d8bb71f0b4b),
        BFieldElement::new(0x8dcf22f9e1bde2a3),
        BFieldElement::new(0x77bdaeda8cc55427),
        BFieldElement::new(0xf19e400ababa0e12),
        BFieldElement::new(0xc368a34939eb5c7f),
        BFieldElement::new(0x9ef1cd612c03bc5e),
        BFieldElement::new(0xe89cd8553b94bbd8),
        BFieldElement::new(0x5cd377dcb4550713),
        BFieldElement::new(0xa7b0fb78cd4c5665),
        BFieldElement::new(0x7684403ef76c7128),
        BFieldElement::new(0x5fa3f06f79c4f483),
        BFieldElement::new(0x8df57ac159dbade6),
        BFieldElement::new(0x2db01efa321b2625),
        BFieldElement::new(0x54846de4cfd58cb6),
        BFieldElement::new(0xba674538aa20f5cd),
        BFieldElement::new(0x541d4963699f9777),
        BFieldElement::new(0xe9096784dadaa548),
        BFieldElement::new(0xdfe8992458bf85ff),
        BFieldElement::new(0xece5a71e74a35593),
        BFieldElement::new(0x5ff98fd5ff1d14fd),
        BFieldElement::new(0x83e89419524c06e1),
        BFieldElement::new(0x5922040b6ef03286),
        BFieldElement::new(0xf97d750eab002858),
        BFieldElement::new(0x5080d4c2dba7b3ec),
        BFieldElement::new(0xa7de115ba038b508),
        BFieldElement::new(0x6a9242acb5f37ec0),
        BFieldElement::new(0xf7856ef865619ed0),
        BFieldElement::new(0x2265fc930dbd7a89),
        BFieldElement::new(0x17dfc8e5022c723b),
        BFieldElement::new(0x9001a64248f2d676),
        BFieldElement::new(0x90004c13b0b8b50e),
        BFieldElement::new(0xb932b7cfc63485b0),
        BFieldElement::new(0xa0b1df81fd4c2bc5),
        BFieldElement::new(0x8ef1dd26b594c383),
        BFieldElement::new(0x0541a4f9d20ba562),
        BFieldElement::new(0x9e611061be0a3c5b),
        BFieldElement::new(0xb3767e80e1e1624a),
        BFieldElement::new(0x0098d57820a88c6b),
        BFieldElement::new(0x31d191cd71e01691),
        BFieldElement::new(0x410fefafbf90a57a),
        BFieldElement::new(0xbdf8f2433633aea8),
        BFieldElement::new(0x9e8cd55b9cc11c28),
        BFieldElement::new(0xde122bec4acb869f),
        BFieldElement::new(0x4d001fd5b0b03314),
        BFieldElement::new(0xca66370067416209),
        BFieldElement::new(0x2f2339d6399888c6),
        BFieldElement::new(0x6d1a7918f7c98a13),
        BFieldElement::new(0xdf9a493995f688f3),
        BFieldElement::new(0xebc2151f4ded22ca),
        BFieldElement::new(0x03cc2ba8a2bab82f),
        BFieldElement::new(0xd341d03844ad9a9b),
        BFieldElement::new(0x387cb5d273ab3f58),
        BFieldElement::new(0xbba2515f74a7a221),
        BFieldElement::new(0x7248fe7737f37d9c),
        BFieldElement::new(0x4d61e56a7437f6b9),
        BFieldElement::new(0x262e963c9e54bef8),
        BFieldElement::new(0x59e89b097477d296),
        BFieldElement::new(0x055d5b52b9e47452),
        BFieldElement::new(0x82b27eb36e430708),
        BFieldElement::new(0xd30094caf3080f94),
        BFieldElement::new(0xcf5cb38227c2a3be),
        BFieldElement::new(0xfeed4db701262c7c),
        BFieldElement::new(0x41703f5391dd0154),
        BFieldElement::new(0x5eeea9412666f57b),
        BFieldElement::new(0x4cd1f1b196abdbc4),
        BFieldElement::new(0x4a20358594b3662b),
        BFieldElement::new(0x1478d361e4b47c26),
        BFieldElement::new(0x6f02dc0801d2c79f),
        BFieldElement::new(0x296a202eeb03c4b6),
        BFieldElement::new(0x2afd6799aec20c38),
        BFieldElement::new(0x7acfd96f3050383d),
        BFieldElement::new(0x6798ba0c380dfdd3),
        BFieldElement::new(0x34c6f57b3de02c88),
        BFieldElement::new(0x5736e1baf82eb8a0),
        BFieldElement::new(0x20057d2a0e58b8de),
        BFieldElement::new(0x3dea5bd5eb6e1404),
        BFieldElement::new(0x16e50d89874a6a98),
        BFieldElement::new(0x29bff3eccbfba19a),
        BFieldElement::new(0x475cd3207974793c),
        BFieldElement::new(0x18a42105cde34cfa),
        BFieldElement::new(0x023e7414b0618331),
        BFieldElement::new(0x151471081b52594b),
        BFieldElement::new(0xe4a3dff23bdeb0f3),
        BFieldElement::new(0x01a8d1a588c232ef),
        BFieldElement::new(0x11b4c74ee221d621),
        BFieldElement::new(0xe587cc0dce129c8c),
        BFieldElement::new(0x1ff7327025a65080),
        BFieldElement::new(0x594e29c44b8602b1),
        BFieldElement::new(0xf6f31db1f5a56fd3),
        BFieldElement::new(0xc02ac5e4c7258a5e),
        BFieldElement::new(0xe70201e9c5dc598f),
        BFieldElement::new(0x6f90ff3b9b3560b2),
        BFieldElement::new(0x42747a7262faf016),
        BFieldElement::new(0xd1f507e496927d26),
        BFieldElement::new(0x1c86d265fdd24cd9),
        BFieldElement::new(0x3996ce73f6b5266e),
        BFieldElement::new(0x8e7fba02d68a061e),
        BFieldElement::new(0xba0dec71548b7546),
        BFieldElement::new(0x9e9cbd785b8d8f40),
        BFieldElement::new(0xdae86459f6b3828c),
        BFieldElement::new(0xdebe08541314f71d),
        BFieldElement::new(0xa49229d29501358f),
        BFieldElement::new(0x7be5ba0010c4df7c),
        BFieldElement::new(0xa3c95eaf09ecc39c),
        BFieldElement::new(0x0230bca8f5d457cd),
        BFieldElement::new(0x4135c2bedc68cdf9),
        BFieldElement::new(0x166fc0cc4d5b20cc),
        BFieldElement::new(0x3762b59aa3236e6e),
        BFieldElement::new(0xe8928a4ceed163d2),
        BFieldElement::new(0x2a440b51b71223d9),
        BFieldElement::new(0x80cefd2bb5f48e46),
        BFieldElement::new(0xbb9879c738328b71),
        BFieldElement::new(0x6e7c8f1ab47cced0),
        BFieldElement::new(0x164bb2de257ffc0a),
        BFieldElement::new(0xf3c12fe5b800ea30),
        BFieldElement::new(0x40b9e92309e8c7e1),
        BFieldElement::new(0x551f5b0fe3b8d017),
        BFieldElement::new(0x25032aa7d4fc7aba),
        BFieldElement::new(0xaaed340795de0a0a),
        BFieldElement::new(0x8ffd96bc38c8ba0f),
        BFieldElement::new(0x70fc91eb8aa58833),
        BFieldElement::new(0x7f795e2a97566d73),
        BFieldElement::new(0x4543d9df72c4831d),
        BFieldElement::new(0xf172d73e69f20739),
        BFieldElement::new(0xdfd1c4ff1eb3d868),
        BFieldElement::new(0xbc8dfb62d26376f7),
        BFieldElement::new(0x3f3b0b53ae4624f0),
        BFieldElement::new(0x8aff6a4012784bf9),
        BFieldElement::new(0xa788db8140374349),
        BFieldElement::new(0x7519463b9ca12e3f),
        BFieldElement::new(0x916257fce69a385e),
        BFieldElement::new(0x1e2333a9f193f20a),
        BFieldElement::new(0x7e218f76de8e895d),
        BFieldElement::new(0x3a679f3e1277d39c),
        BFieldElement::new(0x8291da1124f8da28),
        BFieldElement::new(0xa6915ec6c00589aa),
        BFieldElement::new(0xbd7f43cbef9842e6),
        BFieldElement::new(0x396046345ece5f81),
        BFieldElement::new(0x9c0f589411f8f696),
        BFieldElement::new(0x97bf800069ca178d),
        BFieldElement::new(0xb70dadd0f6d9b373),
        BFieldElement::new(0xd3a8a413f5b0947a),
        BFieldElement::new(0x710dd9eaee2506b7),
        BFieldElement::new(0x03a826cc0d051fb6),
        BFieldElement::new(0x78c126112bf8cb54),
        BFieldElement::new(0x5d13d03c9d6032e3),
        BFieldElement::new(0x502a9452275bac54),
        BFieldElement::new(0xa399ed365a014c85),
        BFieldElement::new(0xd91d7776476759aa),
        BFieldElement::new(0x3388ea2010484f20),
        BFieldElement::new(0x8bc8d8e0335c157b),
        BFieldElement::new(0x967830d0be15dfe6),
        BFieldElement::new(0x6b69b32c4e880930),
        BFieldElement::new(0xd730fbf87d6e70b5),
        BFieldElement::new(0x85223b3b3b7bed07),
        BFieldElement::new(0xfa0d85221f3d72cf),
        BFieldElement::new(0xa6ded7c7f5d6d272),
        BFieldElement::new(0x9fbddba3c86edbad),
        BFieldElement::new(0x0eac3c714b886b3d),
        BFieldElement::new(0x694d74d6b0a32710),
        BFieldElement::new(0x6b8ee6b99cd2ac11),
        BFieldElement::new(0x3501d69447d6f2b9),
        BFieldElement::new(0x0290abe42804865e),
        BFieldElement::new(0x0331884444ffdab9),
        BFieldElement::new(0xf29c228bf96bc677),
        BFieldElement::new(0xf1d3df53724b6a6e),
        BFieldElement::new(0xc23624ec8fd59daf),
        BFieldElement::new(0xb32c718470b115c2),
        BFieldElement::new(0x522741569915690a),
        BFieldElement::new(0xf4e786b5d6d87ecf),
        BFieldElement::new(0x2c0d326a4e938885),
        BFieldElement::new(0xaff56278b67a71c2),
        BFieldElement::new(0x2ca8e42395fb3398),
        BFieldElement::new(0x8091f2239b333ebd),
        BFieldElement::new(0xea4505d1a9901ff3),
        BFieldElement::new(0xdc48db966aac2a52),
        BFieldElement::new(0x48c7c93ff2349004),
        BFieldElement::new(0xa567c362fbb799da),
        BFieldElement::new(0xd390081f9c257d4b),
        BFieldElement::new(0x9384c1070eb42745),
        BFieldElement::new(0xda4080d5e8af4bb3),
        BFieldElement::new(0x1929674becefc8e3),
        BFieldElement::new(0xdce4ee6ff917b599),
        BFieldElement::new(0x613df420085e3c40),
        BFieldElement::new(0xe35ffbe87d486774),
        BFieldElement::new(0x19c8fe60f374b898),
        BFieldElement::new(0x4ee06e1a5d0b2b59),
        BFieldElement::new(0xeb11f532da72e497),
        BFieldElement::new(0xa9f8b9b3b64cebcb),
        BFieldElement::new(0xd4e2607ab772baf5),
        BFieldElement::new(0x082e4328a115b4bb),
        BFieldElement::new(0x571b0bc42e1d7204),
        BFieldElement::new(0xe5b2f2ab010ba6e8),
        BFieldElement::new(0xf95fca5d0b2e0dfd),
        BFieldElement::new(0xa1515d97346e9a08),
        BFieldElement::new(0x25a98f6ef97ccefc),
        BFieldElement::new(0x0db6fa5ec2d790d0),
        BFieldElement::new(0x3df9019d524ffcf6),
        BFieldElement::new(0x08ea1368d4262d01),
        BFieldElement::new(0xa09d948b08f421cd),
        BFieldElement::new(0x0ad5e966870973eb),
        BFieldElement::new(0x326ccef45586a421),
        BFieldElement::new(0xc90f035d2f195b89),
        BFieldElement::new(0x17501e67f48218fc),
        BFieldElement::new(0xe18ee2416be34c37),
        BFieldElement::new(0xced93f36a645442c),
        BFieldElement::new(0x754264af3522104f),
        BFieldElement::new(0x67eb06be66266452),
        BFieldElement::new(0xf37bfb01de53d2a6),
        BFieldElement::new(0x84dc8b4909daed76),
        BFieldElement::new(0xfdab302f7750570f),
        BFieldElement::new(0x63e856e68d12f481),
        BFieldElement::new(0x957712a1717c53f8),
        BFieldElement::new(0x0eaf6bdd23ca146c),
        BFieldElement::new(0x2c2f6e88585050a8),
        BFieldElement::new(0x1734b689a1096dbd),
        BFieldElement::new(0x435b37fe097ebbb4),
        BFieldElement::new(0xdb8ff81f76f13ea2),
        BFieldElement::new(0xf26e6bf5ec48653b),
        BFieldElement::new(0xb3ef69ddf738923b),
        BFieldElement::new(0x0e5deaf37c2e7f7c),
        BFieldElement::new(0x537d810ff2ff5c7f),
        BFieldElement::new(0x8b64439d81824d8f),
        BFieldElement::new(0x05cd0f2210bef923),
        BFieldElement::new(0x620236e39bcf6e15),
        BFieldElement::new(0x2f11cf1501e35f97),
        BFieldElement::new(0x88752b7e6ff8173c),
        BFieldElement::new(0x2545e7e8883df01f),
        BFieldElement::new(0xdf7617a579ea6f69),
        BFieldElement::new(0x311b2fcdd633def4),
        BFieldElement::new(0x179e1422f5d0c83f),
        BFieldElement::new(0x68d91273e79d1c5b),
        BFieldElement::new(0x3dd74da013b28e0e),
        BFieldElement::new(0x6d996a910a4a189d),
        BFieldElement::new(0x5ebf824774f28933),
        BFieldElement::new(0x2b89e07840b58f52),
        BFieldElement::new(0x96397a640fa9b9c4),
        BFieldElement::new(0x90524d7159139759),
        BFieldElement::new(0x258aa8b9125dabdb),
        BFieldElement::new(0x08f6bcce0819ce31),
        BFieldElement::new(0xe9469f8e3423540c),
        BFieldElement::new(0xde5164d3c42081e7),
        BFieldElement::new(0x7d5c9ef70889cadd),
        BFieldElement::new(0xf9507747dd71c6c7),
        BFieldElement::new(0x121fd36677688512),
        BFieldElement::new(0x416e819aff900513),
    ];

    #[inline(always)]
    #[unroll_for_loops]
    fn mds_row_shf(r: usize, v: &[BFieldElement; STATE_SIZE]) -> BFieldElement {
        debug_assert!(r < STATE_SIZE);
        // The values of `MDS_MATRIX_CIRC` and `MDS_MATRIX_DIAG` are
        // known to be small, so we can accumulate all the products for
        // each row and reduce just once at the end (done by the
        // caller).

        // NB: Unrolling this, calculating each term independently, and
        // summing at the end, didn't improve performance for me.
        let mut res = BFIELD_ZERO;

        // This is a hacky way of fully unrolling the loop.
        for i in 0..STATE_SIZE {
            res += v[(i + r) % STATE_SIZE] * Self::MDS_MATRIX_CIRC[i];
        }
        res += v[r] * Self::MDS_MATRIX_DIAG[r];

        res
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn mds_layer(state: &[BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        let mut result = [BFIELD_ZERO; STATE_SIZE];

        // This is a hacky way of fully unrolling the loop.
        #[allow(clippy::needless_range_loop)]
        for r in 0..STATE_SIZE {
            result[r] = Self::mds_row_shf(r, state);
        }

        result
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn partial_first_constant_layer(state: &mut [BFieldElement; STATE_SIZE]) {
        #[allow(clippy::needless_range_loop)]
        for i in 0..STATE_SIZE {
            state[i] += Self::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i];
        }
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn mds_partial_layer_init(state: &[BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        let mut result = [BFIELD_ZERO; STATE_SIZE];

        // Initial matrix has first row/column = [1, 0, ..., 0];

        // c = 0
        result[0] = state[0];

        #[allow(clippy::needless_range_loop)]
        for r in 1..STATE_SIZE {
            for c in 1..STATE_SIZE {
                // NB: FAST_PARTIAL_ROUND_INITIAL_MATRIX is stored in
                // row-major order so that this dot product is cache
                // friendly.
                let t = Self::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1];
                result[c] += state[r] * t;
            }
        }
        result
    }

    /// Computes s*A where s is the state row vector and A is the matrix
    ///
    ///    [ M_00  | v  ]
    ///    [ ------+--- ]
    ///    [ w_hat | Id ]
    ///
    /// M_00 is a scalar, v is 1x(t-1), w_hat is (t-1)x1 and Id is the
    /// (t-1)x(t-1) identity matrix.
    #[inline(always)]
    #[unroll_for_loops]
    fn mds_partial_layer_fast(
        state: &[BFieldElement; STATE_SIZE],
        r: usize,
    ) -> [BFieldElement; STATE_SIZE] {
        // Set d = [M_00 | w^] dot [state]

        let mut d_sum = BFIELD_ZERO;
        #[allow(clippy::needless_range_loop)]
        for i in 1..STATE_SIZE {
            d_sum += state[i] * Self::FAST_PARTIAL_ROUND_W_HATS[r][i - 1];
        }
        let mds0to0 = Self::MDS_MATRIX_CIRC[0] + Self::MDS_MATRIX_DIAG[0];
        d_sum += state[0] * mds0to0;
        let d = d_sum;

        // result = [d] concat [state[0] * v + state[shift up by 1]]
        let mut result = [BFIELD_ZERO; STATE_SIZE];
        result[0] = d;
        for i in 1..STATE_SIZE {
            result[i] = state[i] + state[0] * Self::FAST_PARTIAL_ROUND_VS[r][i - 1];
        }
        result
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn constant_layer(state: &mut [BFieldElement; STATE_SIZE], round_ctr: usize) {
        #[allow(clippy::needless_range_loop)]
        for i in 0..STATE_SIZE {
            let round_constant = Self::ALL_ROUND_CONSTANTS[i + STATE_SIZE * round_ctr];
            state[i] += round_constant;
        }
    }

    #[inline(always)]
    fn sbox_monomial(x: BFieldElement) -> BFieldElement {
        // x |--> x^7
        let x2 = x * x;
        let x4 = x2 * x2;
        let x3 = x * x2;
        x3 * x4
    }

    #[inline(always)]
    #[unroll_for_loops]
    fn sbox_layer(state: &mut [BFieldElement; STATE_SIZE]) {
        #[allow(clippy::needless_range_loop)]
        for i in 0..STATE_SIZE {
            state[i] = Self::sbox_monomial(state[i]);
        }
    }

    #[inline]
    fn full_rounds(state: &mut [BFieldElement; STATE_SIZE], round_ctr: &mut usize) {
        for _ in 0..HALF_N_FULL_ROUNDS {
            Self::constant_layer(state, *round_ctr);
            Self::sbox_layer(state);
            *state = Self::mds_layer(state);
            *round_ctr += 1;
        }
    }

    #[inline]
    fn partial_rounds(state: &mut [BFieldElement; STATE_SIZE], round_ctr: &mut usize) {
        Self::partial_first_constant_layer(state);
        *state = Self::mds_partial_layer_init(state);

        for i in 0..N_PARTIAL_ROUNDS {
            state[0] = Self::sbox_monomial(state[0]);
            state[0] += Self::FAST_PARTIAL_ROUND_CONSTANTS[i];
            *state = Self::mds_partial_layer_fast(state, i);
        }
        *round_ctr += N_PARTIAL_ROUNDS;
    }

    #[inline]
    pub fn poseidon(input: [BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        let mut state = input;
        let mut round_ctr = 0;

        Self::full_rounds(&mut state, &mut round_ctr);
        Self::partial_rounds(&mut state, &mut round_ctr);
        Self::full_rounds(&mut state, &mut round_ctr);
        debug_assert_eq!(round_ctr, N_ROUNDS);

        state
    }

    // For testing only, to ensure that various tricks are correct.
    #[inline]
    fn partial_rounds_naive(state: &mut [BFieldElement; STATE_SIZE], round_ctr: &mut usize) {
        for _ in 0..N_PARTIAL_ROUNDS {
            Self::constant_layer(state, *round_ctr);
            state[0] = Self::sbox_monomial(state[0]);
            *state = Self::mds_layer(state);
            *round_ctr += 1;
        }
    }

    #[inline]
    pub fn poseidon_naive(input: [BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        let mut state = input;
        let mut round_ctr = 0;

        Self::full_rounds(&mut state, &mut round_ctr);
        Self::partial_rounds_naive(&mut state, &mut round_ctr);
        Self::full_rounds(&mut state, &mut round_ctr);
        debug_assert_eq!(round_ctr, N_ROUNDS);

        state
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoseidonState {
    pub state: [BFieldElement; STATE_SIZE],
}

impl PoseidonState {
    #[inline]
    pub const fn new(domain: Domain) -> Self {
        use Domain::*;

        let mut state = [BFIELD_ZERO; STATE_SIZE];

        match domain {
            VariableLength => (),
            FixedLength => state[RATE] = BFIELD_ONE,
        }

        Self { state }
    }
}

impl SpongeHasher for Poseidon {
    const RATE: usize = RATE;
    type SpongeState = PoseidonState;

    fn init() -> Self::SpongeState {
        PoseidonState::new(Domain::VariableLength)
    }

    fn absorb(sponge: &mut Self::SpongeState, input: &[BFieldElement; RATE]) {
        // absorb
        sponge.state[..RATE]
            .iter_mut()
            .zip_eq(input.iter())
            .for_each(|(a, &b)| *a += b);

        sponge.state = Poseidon::poseidon(sponge.state);
    }

    fn squeeze(sponge: &mut Self::SpongeState) -> [BFieldElement; RATE] {
        // squeeze
        let produce: [BFieldElement; RATE] = (&sponge.state[..RATE]).try_into().unwrap();

        sponge.state = Poseidon::poseidon(sponge.state);

        produce
    }
}

#[cfg(test)]
mod tests {
    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::b_field_element::BFIELD_ZERO;
    use crate::shared_math::poseidon::{Poseidon, STATE_SIZE};

    pub(crate) fn check_test_vectors(test_vectors: Vec<([u64; STATE_SIZE], [u64; STATE_SIZE])>) {
        for (input_, expected_output_) in test_vectors.into_iter() {
            let mut input = [BFIELD_ZERO; STATE_SIZE];
            for i in 0..STATE_SIZE {
                input[i] = BFieldElement::new(input_[i]);
            }
            let output = Poseidon::poseidon(input);
            for i in 0..STATE_SIZE {
                let ex_output = BFieldElement::new(expected_output_[i]);
                assert_eq!(output[i], ex_output);
            }
        }
    }

    pub(crate) fn check_consistency() {
        let mut input = [BFIELD_ZERO; STATE_SIZE];
        #[allow(clippy::needless_range_loop)]
        for i in 0..STATE_SIZE {
            input[i] = BFieldElement::new(i as u64);
        }
        let output = Poseidon::poseidon(input);
        let output_naive = Poseidon::poseidon_naive(input);
        for i in 0..STATE_SIZE {
            assert_eq!(output[i], output_naive[i]);
        }
    }

    #[test]
    #[ignore = "the MDS matrices `MDS_MATRIX_CIRC` and `MDS_MATRIX_DIAG` are incorrect"]
    fn test_vectors() {
        // Test inputs are:
        // 1. all zeros
        // 2. all -1's
        // 3. range 0..STATE_SIZE
        // 4. random elements of Oxfoi Field.
        // expected output calculated with (modified) hadeshash reference implementation.

        #[rustfmt::skip]
            let test_vectors: Vec<([u64; STATE_SIZE], [u64; STATE_SIZE])> = vec![
            ([0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
                 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
                 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
                 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, ],
             [0xdafb6ec6b16908ac, 0x25cb76e3dad238c5, 0xa08ce3541b3b6eed, 0x56f394375038f54c,
                 0xd5794029640f47da, 0xde38b2b23cf6ffa4, 0x4fef02edc1fc54a3, 0xc9b1dd3b3aab3b66,
                 0xa22c01e95cda0722, 0xe7ec4d06ce4c5b10, 0xe242983b312d6d40, 0xe3e0977163feb58b,
                 0x2af1e3de0e94a508, 0xc671b6842ffd4406, 0x1c5b7bc9eebbf151, 0x1b76a8a5a1ce2046, ]),
            ([0xffffffff00000000, 0xffffffff00000000, 0xffffffff00000000, 0xffffffff00000000,
                 0xffffffff00000000, 0xffffffff00000000, 0xffffffff00000000, 0xffffffff00000000,
                 0xffffffff00000000, 0xffffffff00000000, 0xffffffff00000000, 0xffffffff00000000,
                 0xffffffff00000000, 0xffffffff00000000, 0xffffffff00000000, 0xffffffff00000000, ],
             [0x5ce4a828d177c432, 0x922b49fba5acc2fb, 0x1ca2938d5b1680ab, 0x51486314f32444ce,
                 0x3efa3c3374c4f4fc, 0xbde7dc24c8e8c98e, 0x13b0edd2057c2724, 0x0234df0bffb7f5a1,
                 0xe7e361f7278d8e3c, 0x7b3090f1f3d30d63, 0xbdc1b8ed2c133e08, 0x360586235695aa60,
                 0xec2dd4c9b4a913b0, 0x81931dbd001a3bbd, 0x9ff888e959ae8fc4, 0xab0eefad194925f2, ]),
            ([0x0000000000000000, 0x0000000000000001, 0x0000000000000002, 0x0000000000000003,
                 0x0000000000000004, 0x0000000000000005, 0x0000000000000006, 0x0000000000000007,
                 0x0000000000000008, 0x0000000000000009, 0x000000000000000a, 0x000000000000000b,
                 0x000000000000000c, 0x000000000000000d, 0x000000000000000e, 0x000000000000000f, ],
             [0xa0ebee93aba018ba, 0xa323beed4370782e, 0xe1316dcd39966899, 0xa07ee998d8a043c0,
                 0x5afcf21de9cec0d7, 0x3c8c37f8c8e03b43, 0x0d52eeb8a39bee47, 0xc1a411bf95b7fefe,
                 0x6560a38561c87046, 0x2453f036da9bc2c3, 0xd8b3e70480a49355, 0xd5e0716166174f86,
                 0xc10c6fe8b55ff72f, 0xd711fb4de209a00d, 0xca538324a158b405, 0xceeddb78bdc13da7, ]),
            ([0x071ed9ca1d6639ab, 0x03e537c6de1df024, 0x8f5d5f493f547a62, 0x157f82ccbdfbd59d,
                 0x314a6508cd53d03d, 0xafbadea1279d6a77, 0x2be1a4c3c7e6b950, 0x02d1ed4b4f355f99,
                 0xbde4682fcd76b8c0, 0x9199032b416a449c, 0x4b4c19f89e268d8e, 0x2948b8d82f8194d2,
                 0x772097a5026d019a, 0x835c7f44a5ef953e, 0x0f909c4868716a10, 0x470a44c696f840b5, ],
             [0x8cdad7ead2ce1aee, 0xec6fffab538c6dc7, 0x6bdf4aa29227141e, 0x18972a31baae3399,
                 0x7678815bdb72f9ed, 0xb81fdfa7e8de4ced, 0x8d6b9528e61df048, 0x001eeedf16035643,
                 0x58889781be9d9819, 0x43db1d10379bc79d, 0x55480ef5260869bb, 0x9a643eab82c091bf,
                 0x4858958d9bc0c47b, 0x4f2b56aca3e649ca, 0xb330b85e3383eeca, 0x389b62e02db18ea6, ]),
        ];

        check_test_vectors(test_vectors);
    }

    #[test]
    #[ignore = "the MDS matrices `MDS_MATRIX_CIRC` and `MDS_MATRIX_DIAG` are incorrect"]
    fn consistency() {
        check_consistency();
    }
}
