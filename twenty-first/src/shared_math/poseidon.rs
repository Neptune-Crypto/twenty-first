use itertools::Itertools;
use num_traits::One;
use serde::{Deserialize, Serialize};

use crate::shared_math::b_field_element::{BFieldElement, BFIELD_ONE, BFIELD_ZERO};
use crate::shared_math::traits::FiniteField;
use crate::util_types::algebraic_hasher::AlgebraicHasher;

use super::rescue_prime_digest::Digest;

pub const DIGEST_LENGTH: usize = 5;
pub const STATE_SIZE: usize = 16;
pub const CAPACITY: usize = 6;
pub const RATE: usize = 10;

// parameters generated from
// $> sage generate_params_poseidon.sage 1 0 64 16 7 160 0xffffffff00000001

pub const NUM_PARTIAL: usize = 22;
pub const NUM_FULL: usize = 8;
pub const NUM_ROUNDS: usize = NUM_PARTIAL + NUM_FULL;

pub const ALPHA: u64 = 7;

pub const MDS: [BFieldElement; STATE_SIZE * STATE_SIZE] = [
    BFieldElement::new(0x5988b85748e01d64),
    BFieldElement::new(0x6ce5b2a51c028cd5),
    BFieldElement::new(0x97151726a74103eb),
    BFieldElement::new(0x48da805da22db860),
    BFieldElement::new(0xab527bddf567ce5d),
    BFieldElement::new(0xa62c7a8802d557bf),
    BFieldElement::new(0x682f001eff55d953),
    BFieldElement::new(0x739f833269079c91),
    BFieldElement::new(0x52f36cc427b3c2d1),
    BFieldElement::new(0x207f3edd31fec0c1),
    BFieldElement::new(0xe861a14803653f31),
    BFieldElement::new(0x7f75391531d1351a),
    BFieldElement::new(0x9c1ff22b7b6519c0),
    BFieldElement::new(0x33b1b3b9f6243b4d),
    BFieldElement::new(0x39a21454890d9e9f),
    BFieldElement::new(0x5153a55d83d32ad0),
    BFieldElement::new(0x2c7fa8fb1e89be4a),
    BFieldElement::new(0x53aaf9e104e4e906),
    BFieldElement::new(0x5466ca73c94286db),
    BFieldElement::new(0x01533ab42def6283),
    BFieldElement::new(0xa1d91119905de8f2),
    BFieldElement::new(0xc4a70e83b3da8175),
    BFieldElement::new(0x9c4fa46f80a875b7),
    BFieldElement::new(0x2ab64e023d7a4574),
    BFieldElement::new(0x95870ac3b43221ad),
    BFieldElement::new(0x8ec4c7dfe0b1e529),
    BFieldElement::new(0x29af3c3d9caab861),
    BFieldElement::new(0x0f5e03a41985622f),
    BFieldElement::new(0x6e351f8b80d25233),
    BFieldElement::new(0x34c392b28480b011),
    BFieldElement::new(0xa2e21b3c21902702),
    BFieldElement::new(0x98c40b93bfa7d635),
    BFieldElement::new(0x3669f4cbc4539d7e),
    BFieldElement::new(0xf8fee0879daec673),
    BFieldElement::new(0xb732c1f23b761aee),
    BFieldElement::new(0xceee88ee79daf441),
    BFieldElement::new(0x86c99724e2cef313),
    BFieldElement::new(0x1cf1f3ab213a0d3c),
    BFieldElement::new(0xd5ea051783a3122a),
    BFieldElement::new(0x22352d4ea73f6045),
    BFieldElement::new(0xf383c05e03cd7c62),
    BFieldElement::new(0x81dfe9c8a0051bbe),
    BFieldElement::new(0x3f07ffcdd9266723),
    BFieldElement::new(0xc93603d00f5dc893),
    BFieldElement::new(0xe69d2645321274ab),
    BFieldElement::new(0xd4d06b40d3b2458d),
    BFieldElement::new(0x3c3b9bf934fc9fd7),
    BFieldElement::new(0x6c3ed734df9cf691),
    BFieldElement::new(0x3c3edb0d8e367cbe),
    BFieldElement::new(0x4d141f636fb94d24),
    BFieldElement::new(0x593dc28f61908b6d),
    BFieldElement::new(0x51429b67ced5b2ea),
    BFieldElement::new(0xb8ad92e65afe1948),
    BFieldElement::new(0x7de771ac1136a634),
    BFieldElement::new(0xf2bc3d4d9f8c12b4),
    BFieldElement::new(0xa00899fee1163f9a),
    BFieldElement::new(0xefd9361aba69c5f9),
    BFieldElement::new(0x692af10ed750d360),
    BFieldElement::new(0x203a34814f5fdba1),
    BFieldElement::new(0x22aa4b0fc8e4f127),
    BFieldElement::new(0x8404c6ef28881b6d),
    BFieldElement::new(0x332c23191cb610fd),
    BFieldElement::new(0xe03b41f9d0a5d63f),
    BFieldElement::new(0xf6bc920a03f7752b),
    BFieldElement::new(0x35accf0a901fcb8d),
    BFieldElement::new(0x8dcff51324521386),
    BFieldElement::new(0x57e239c58d0c8e89),
    BFieldElement::new(0xfaed102b92f35770),
    BFieldElement::new(0x3e4bfd2caf5c21f7),
    BFieldElement::new(0x3cd53e002583952a),
    BFieldElement::new(0xea2a4a22eb681b2c),
    BFieldElement::new(0x670a2ee21d7014eb),
    BFieldElement::new(0xdde0a9aa6df4b771),
    BFieldElement::new(0x95b1fbc56e1a6c39),
    BFieldElement::new(0x6346cd06cc929757),
    BFieldElement::new(0x6ae4bda639af92e0),
    BFieldElement::new(0x77314dfd51e7d92c),
    BFieldElement::new(0x0391e3978ff24b17),
    BFieldElement::new(0xe27611f117d6d97f),
    BFieldElement::new(0x04883758be9f54e3),
    BFieldElement::new(0xe6d3dda85e6395c2),
    BFieldElement::new(0x0cd42c05f66b3c1c),
    BFieldElement::new(0x698359374266b60b),
    BFieldElement::new(0x9e9eaa8893e52d50),
    BFieldElement::new(0x034432c8f57c7559),
    BFieldElement::new(0x4674cb4bc8006eae),
    BFieldElement::new(0x92010504468f2681),
    BFieldElement::new(0x182baea6c583b9c6),
    BFieldElement::new(0xfd348e1d8e95d693),
    BFieldElement::new(0xc2f2fd9b9078ece9),
    BFieldElement::new(0x1aba1a45a6419bf9),
    BFieldElement::new(0xbcc2b6584c1af690),
    BFieldElement::new(0xad0b58758270f4ca),
    BFieldElement::new(0xd4e24b7e35851406),
    BFieldElement::new(0x9572342e0aec74bd),
    BFieldElement::new(0xf681bb067f7ac76e),
    BFieldElement::new(0x30f21444e981732e),
    BFieldElement::new(0x0a4b4c356cf81323),
    BFieldElement::new(0xe0ddde574b7bf32b),
    BFieldElement::new(0xc3f99cc853e76784),
    BFieldElement::new(0xe1aa2dd912c46532),
    BFieldElement::new(0x54531f9f67108f80),
    BFieldElement::new(0x8639b004a91e6a28),
    BFieldElement::new(0x8cd1ddb584f7a9ac),
    BFieldElement::new(0xd78dad0f1337bba4),
    BFieldElement::new(0xc481f8f3a4b87f91),
    BFieldElement::new(0x4c24f9db5e94ab02),
    BFieldElement::new(0x1c62f47a8c19eddb),
    BFieldElement::new(0xbf16cec4fba38913),
    BFieldElement::new(0x4ed589b4fd6bbaf0),
    BFieldElement::new(0x86cff751dac92c05),
    BFieldElement::new(0x86249af3a8eff0f6),
    BFieldElement::new(0xe1033c0d774c285d),
    BFieldElement::new(0x2f8586c1921d8fb4),
    BFieldElement::new(0x7441a10af45e5808),
    BFieldElement::new(0x8d8feb70aded8628),
    BFieldElement::new(0x7da45cc544d9ae35),
    BFieldElement::new(0x767f12ba8a5a0c14),
    BFieldElement::new(0x620efed7f8d3a459),
    BFieldElement::new(0xdd7ff646c99f70ad),
    BFieldElement::new(0x3678cc601846b44a),
    BFieldElement::new(0x6755b4209c3401cc),
    BFieldElement::new(0x8e0ae4aa7035b0eb),
    BFieldElement::new(0xba9b1267dd354322),
    BFieldElement::new(0xfadd3d5f0683e9b6),
    BFieldElement::new(0xbc5c35ae68f2dfe2),
    BFieldElement::new(0x73ae2e3c9c090886),
    BFieldElement::new(0xcc0c1c1e8d9bef49),
    BFieldElement::new(0x8671921f7a0a1794),
    BFieldElement::new(0x9cd5f1eac6037499),
    BFieldElement::new(0x3e527bd0148ade83),
    BFieldElement::new(0xaa717d8659a564c3),
    BFieldElement::new(0x548583860ccd6913),
    BFieldElement::new(0x0df5a873cf6cef5f),
    BFieldElement::new(0x6b1a27e8227f6df8),
    BFieldElement::new(0xdb3e44aa830df1c7),
    BFieldElement::new(0x71353fc324da6e34),
    BFieldElement::new(0x519ed83e6503d741),
    BFieldElement::new(0x81d74e3914f76769),
    BFieldElement::new(0x6c8a3d26ef84b973),
    BFieldElement::new(0x48033d71b77fe8c1),
    BFieldElement::new(0x0b845bc4926fb555),
    BFieldElement::new(0x9e07dbc3ba174490),
    BFieldElement::new(0x0ec39acc7f68d8ce),
    BFieldElement::new(0x7677a79600252463),
    BFieldElement::new(0xf3e2c73584829d44),
    BFieldElement::new(0x554b023d8621a4de),
    BFieldElement::new(0xf8afcc7e5c120419),
    BFieldElement::new(0x385e05151f117f7c),
    BFieldElement::new(0x6f349729b45b5be6),
    BFieldElement::new(0x3a98b59f73759ee1),
    BFieldElement::new(0xd98152e584082edb),
    BFieldElement::new(0x795f0382526cb465),
    BFieldElement::new(0x3d7631494b438675),
    BFieldElement::new(0x1003b04cc5497839),
    BFieldElement::new(0x0f224853d28542fe),
    BFieldElement::new(0xf83d612bda7234b5),
    BFieldElement::new(0xdbfc370b88104851),
    BFieldElement::new(0xc1dd2e22be7c3a16),
    BFieldElement::new(0x17ca5948341a933b),
    BFieldElement::new(0xa9d70e4e8c01ad89),
    BFieldElement::new(0x47bb5a0c5a8da9b0),
    BFieldElement::new(0xc7b5e1da10d895f7),
    BFieldElement::new(0x5f0cc2c2aaa7b03e),
    BFieldElement::new(0xbc8097f44e63863f),
    BFieldElement::new(0x70c8e880581bd43e),
    BFieldElement::new(0x758cfe2ffa079581),
    BFieldElement::new(0x839395f285d3ece8),
    BFieldElement::new(0xa2f27080ac482121),
    BFieldElement::new(0x3f62d1fcb8f9d784),
    BFieldElement::new(0x4b2c662ed6ae100c),
    BFieldElement::new(0x14f3fbce248827f0),
    BFieldElement::new(0xbda2a9225eda8792),
    BFieldElement::new(0x1ed0e69117ef3c36),
    BFieldElement::new(0x3af8405a89b5c1a5),
    BFieldElement::new(0x3429a7904ea79c9d),
    BFieldElement::new(0x9d8119220744a23b),
    BFieldElement::new(0x730bf95139f665b1),
    BFieldElement::new(0xf71f347cf06eb11b),
    BFieldElement::new(0x1e16d46ef71b8e86),
    BFieldElement::new(0xffb5517c741183a7),
    BFieldElement::new(0x9de7bdf6e7763c76),
    BFieldElement::new(0x641535f22ebbb8b5),
    BFieldElement::new(0xcc9a2ac6e3525ff2),
    BFieldElement::new(0x6ca74aadfd1870b2),
    BFieldElement::new(0x26c7b78d1a569b93),
    BFieldElement::new(0x6771174eb61e5363),
    BFieldElement::new(0xfece43af74186a6c),
    BFieldElement::new(0xe2d04ce9d92aac6c),
    BFieldElement::new(0x069bce14795ab3ee),
    BFieldElement::new(0x454663cdef4ad361),
    BFieldElement::new(0xc9d1d1b13aa366ba),
    BFieldElement::new(0x2a14bb5529f46c5e),
    BFieldElement::new(0x9284dc3ec5a3298e),
    BFieldElement::new(0xc32a357d30aa5298),
    BFieldElement::new(0x6378633155977074),
    BFieldElement::new(0xd42fbe0d1987c1d4),
    BFieldElement::new(0x812a7402da911fc3),
    BFieldElement::new(0x0aa377c56139e86d),
    BFieldElement::new(0xf738501437323e35),
    BFieldElement::new(0x716814c20125dd1a),
    BFieldElement::new(0x4b8f884ea190466e),
    BFieldElement::new(0x73af1afa89a8b0f8),
    BFieldElement::new(0x581257e0a6514156),
    BFieldElement::new(0x2a3e9650920f7c21),
    BFieldElement::new(0x3a4b286082ce7d23),
    BFieldElement::new(0xaa89fb9040aed6fa),
    BFieldElement::new(0xcbcd108576e6d1ba),
    BFieldElement::new(0xa0e6af946333f9cc),
    BFieldElement::new(0x5b9fa025d5f91de4),
    BFieldElement::new(0xa37f3398a3ff534c),
    BFieldElement::new(0x5d807c5491b334cc),
    BFieldElement::new(0xdde0491f392f54b3),
    BFieldElement::new(0xd31f9255564584e5),
    BFieldElement::new(0xb682b5175a329122),
    BFieldElement::new(0x36ac21cacdf0a097),
    BFieldElement::new(0xbd69c3e422624ed0),
    BFieldElement::new(0xfd03026489b74c0b),
    BFieldElement::new(0xb2b862a95c6f5fc9),
    BFieldElement::new(0x7a6ec2a715549e03),
    BFieldElement::new(0x6a3cf97409861804),
    BFieldElement::new(0xdd31032efb7843cc),
    BFieldElement::new(0x080d82ec2393f0c3),
    BFieldElement::new(0x5ca5e139dcf07d41),
    BFieldElement::new(0xf57bae2bb3999ef3),
    BFieldElement::new(0x1b3241817e6f40c5),
    BFieldElement::new(0x12f1d5f21156e7b5),
    BFieldElement::new(0x44261d736bf454a8),
    BFieldElement::new(0xe1194d1a1cd17a51),
    BFieldElement::new(0x9394f656c1246dc8),
    BFieldElement::new(0x050fe6e2bb26bab4),
    BFieldElement::new(0xcb184bfb067f33a3),
    BFieldElement::new(0x87b5f94c9d2c7907),
    BFieldElement::new(0x456daf135d96f358),
    BFieldElement::new(0x116eba266dcc0cb3),
    BFieldElement::new(0xa2dbf3c480582332),
    BFieldElement::new(0x1302f5ef13049a4d),
    BFieldElement::new(0xd2ac7ba5adc5269a),
    BFieldElement::new(0x43b32aedb14c08cc),
    BFieldElement::new(0xe00837b7287de52d),
    BFieldElement::new(0xd9553f671aad6e14),
    BFieldElement::new(0x4bb5271bb83f98a0),
    BFieldElement::new(0x2c84ca7932ff0d9e),
    BFieldElement::new(0x0ac13c2ec615ff05),
    BFieldElement::new(0xfd066e3f8f82e427),
    BFieldElement::new(0x6cddcb7116f97320),
    BFieldElement::new(0xce298f1a321e7087),
    BFieldElement::new(0x74c9ceb2944dc1b5),
    BFieldElement::new(0x6afbdadac658ec67),
    BFieldElement::new(0x8c32f317e710b3bd),
    BFieldElement::new(0xc380701ec43dc56d),
    BFieldElement::new(0x96b0a79e387441a4),
    BFieldElement::new(0xcac77034b3035048),
    BFieldElement::new(0x015fd2a7e0a193ac),
    BFieldElement::new(0x8f848885347161d7),
    BFieldElement::new(0x110276a5157b853a),
];

pub const ROUND_CONSTANTS: [BFieldElement; NUM_ROUNDS * STATE_SIZE] = [
    BFieldElement::new(0x15ebea3fc73397c3),
    BFieldElement::new(0xd73cd9fbfe8e275c),
    BFieldElement::new(0x8c096bfce77f6c26),
    BFieldElement::new(0x4e128f68b53d8fea),
    BFieldElement::new(0x29b779a36b2763f6),
    BFieldElement::new(0xfe2adc6fb65acd08),
    BFieldElement::new(0x8d2520e725ad0955),
    BFieldElement::new(0x1c2392b214624d2a),
    BFieldElement::new(0x37482118206dcc6e),
    BFieldElement::new(0x2f829bed19be019a),
    BFieldElement::new(0x2fe298cb6f8159b0),
    BFieldElement::new(0x2bbad982deccdbbf),
    BFieldElement::new(0xbad568b8cc60a81e),
    BFieldElement::new(0xb86a814265baad10),
    BFieldElement::new(0xbec2005513b3acb3),
    BFieldElement::new(0x6bf89b59a07c2a94),
    BFieldElement::new(0xa25deeb835e230f5),
    BFieldElement::new(0x3c5bad8512b8b12a),
    BFieldElement::new(0x7230f73c3cb7a4f2),
    BFieldElement::new(0xa70c87f095c74d0f),
    BFieldElement::new(0x6b7606b830bb2e80),
    BFieldElement::new(0x6cd467cfc4f24274),
    BFieldElement::new(0xfeed794df42a9b0a),
    BFieldElement::new(0x8cf7cf6163b7dbd3),
    BFieldElement::new(0x9a6e9dda597175a0),
    BFieldElement::new(0xaa52295a684faf7b),
    BFieldElement::new(0x017b811cc3589d8d),
    BFieldElement::new(0x55bfb699b6181648),
    BFieldElement::new(0xc2ccaf71501c2421),
    BFieldElement::new(0x1707950327596402),
    BFieldElement::new(0xdd2fcdcd42a8229f),
    BFieldElement::new(0x8b9d7d5b27778a21),
    BFieldElement::new(0xac9a05525f9cf512),
    BFieldElement::new(0x2ba125c58627b5e8),
    BFieldElement::new(0xc74e91250a8147a5),
    BFieldElement::new(0xa3e64b640d5bb384),
    BFieldElement::new(0xf53047d18d1f9292),
    BFieldElement::new(0xbaaeddacae3a6374),
    BFieldElement::new(0xf2d0914a808b3db1),
    BFieldElement::new(0x18af1a3742bfa3b0),
    BFieldElement::new(0x9a621ef50c55bdb8),
    BFieldElement::new(0xc615f4d1cc5466f3),
    BFieldElement::new(0xb7fbac19a35cf793),
    BFieldElement::new(0xd2b1a15ba517e46d),
    BFieldElement::new(0x4a290c4d7fd26f6f),
    BFieldElement::new(0x4f0cf1bb1770c4c4),
    BFieldElement::new(0x548345386cd377f5),
    BFieldElement::new(0x33978d2789fddd42),
    BFieldElement::new(0xab78c59deb77e211),
    BFieldElement::new(0xc485b2a933d2be7f),
    BFieldElement::new(0xbde3792c00c03c53),
    BFieldElement::new(0xab4cefe8f893d247),
    BFieldElement::new(0xc5c0e752eab7f85f),
    BFieldElement::new(0xdbf5a76f893bafea),
    BFieldElement::new(0xa91f6003e3d984de),
    BFieldElement::new(0x099539077f311e87),
    BFieldElement::new(0x097ec52232f9559e),
    BFieldElement::new(0x53641bdf8991e48c),
    BFieldElement::new(0x2afe9711d5ed9d7c),
    BFieldElement::new(0xa7b13d3661b5d117),
    BFieldElement::new(0x5a0e243fe7af6556),
    BFieldElement::new(0x1076fae8932d5f00),
    BFieldElement::new(0x9b53a83d434934e3),
    BFieldElement::new(0xed3fd595a3c0344a),
    BFieldElement::new(0x28eff4b01103d100),
    BFieldElement::new(0x60400ca3e2685a45),
    BFieldElement::new(0x1c8636beb3389b84),
    BFieldElement::new(0xac1332b60e13eff0),
    BFieldElement::new(0x2adafcc364e20f87),
    BFieldElement::new(0x79ffc2b14054ea0b),
    BFieldElement::new(0x3f98e4c0908f0a05),
    BFieldElement::new(0xcdb230bc4e8a06c4),
    BFieldElement::new(0x1bcaf7705b152a74),
    BFieldElement::new(0xd9bca249a82a7470),
    BFieldElement::new(0x91e24af19bf82551),
    BFieldElement::new(0xa62b43ba5cb78858),
    BFieldElement::new(0xb4898117472e797f),
    BFieldElement::new(0xb3228bca606cdaa0),
    BFieldElement::new(0x844461051bca39c9),
    BFieldElement::new(0xf3411581f6617d68),
    BFieldElement::new(0xf7fd50646782b533),
    BFieldElement::new(0x6ca664253c18fb48),
    BFieldElement::new(0x2d2fcdec0886a08f),
    BFieldElement::new(0x29da00dd799b575e),
    BFieldElement::new(0x47d966cc3b6e1e93),
    BFieldElement::new(0xde884e9a17ced59e),
    BFieldElement::new(0xdacf46dc1c31a045),
    BFieldElement::new(0x5d2e3c121eb387f2),
    BFieldElement::new(0x51f8b0658b124499),
    BFieldElement::new(0x1e7dbd1daa72167d),
    BFieldElement::new(0x8275015a25c55b88),
    BFieldElement::new(0xe8521c24ac7a70b3),
    BFieldElement::new(0x6521d121c40b3f67),
    BFieldElement::new(0xac12de797de135b0),
    BFieldElement::new(0xafa28ead79f6ed6a),
    BFieldElement::new(0x685174a7a8d26f0b),
    BFieldElement::new(0xeff92a08d35d9874),
    BFieldElement::new(0x3058734b76dd123a),
    BFieldElement::new(0xfa55dcfba429f79c),
    BFieldElement::new(0x559294d4324c7728),
    BFieldElement::new(0x7a770f53012dc178),
    BFieldElement::new(0xedd8f7c408f3883b),
    BFieldElement::new(0x39b533cf8d795fa5),
    BFieldElement::new(0x160ef9de243a8c0a),
    BFieldElement::new(0x431d52da6215fe3f),
    BFieldElement::new(0x54c51a2a2ef6d528),
    BFieldElement::new(0x9b13892b46ff9d16),
    BFieldElement::new(0x263c46fcee210289),
    BFieldElement::new(0xb738c96d25aabdc4),
    BFieldElement::new(0x5c33a5203996d38f),
    BFieldElement::new(0x2626496e7c98d8dd),
    BFieldElement::new(0xc669e0a52785903a),
    BFieldElement::new(0xaecde726c8ae1f47),
    BFieldElement::new(0x039343ef3a81e999),
    BFieldElement::new(0x2615ceaf044a54f9),
    BFieldElement::new(0x7e41e834662b66e1),
    BFieldElement::new(0x4ca5fd4895335783),
    BFieldElement::new(0x64b334d02916f2b0),
    BFieldElement::new(0x87268837389a6981),
    BFieldElement::new(0x034b75bcb20a6274),
    BFieldElement::new(0x58e658296cc2cd6e),
    BFieldElement::new(0xe2d0f759acc31df4),
    BFieldElement::new(0x81a652e435093e20),
    BFieldElement::new(0x0b72b6e0172eaf47),
    BFieldElement::new(0x4aec43cec577d66d),
    BFieldElement::new(0xde78365b028a84e6),
    BFieldElement::new(0x444e19569adc0ee4),
    BFieldElement::new(0x942b2451fa40d1da),
    BFieldElement::new(0xe24506623ea5bd6c),
    BFieldElement::new(0x082854bf2ef7c743),
    BFieldElement::new(0x69dbbc566f59d62e),
    BFieldElement::new(0x248c38d02a7b5cb2),
    BFieldElement::new(0x4f4e8f8c09d15edb),
    BFieldElement::new(0xd96682f188d310cf),
    BFieldElement::new(0x6f9a25d56818b54c),
    BFieldElement::new(0xb6cefed606546cd9),
    BFieldElement::new(0x5bc07523da38a67b),
    BFieldElement::new(0x7df5a3c35b8111cf),
    BFieldElement::new(0xaaa2cc5d4db34bb0),
    BFieldElement::new(0x9e673ff22a4653f8),
    BFieldElement::new(0xbd8b278d60739c62),
    BFieldElement::new(0xe10d20f6925b8815),
    BFieldElement::new(0xf6c87b91dd4da2bf),
    BFieldElement::new(0xfed623e2f71b6f1a),
    BFieldElement::new(0xa0f02fa52a94d0d3),
    BFieldElement::new(0xbb5794711b39fa16),
    BFieldElement::new(0xd3b94fba9d005c7f),
    BFieldElement::new(0x15a26e89fad946c9),
    BFieldElement::new(0xf3cb87db8a67cf49),
    BFieldElement::new(0x400d2bf56aa2a577),
    BFieldElement::new(0x56045ba28fd1dbb1),
    BFieldElement::new(0x06851d67ed5acacd),
    BFieldElement::new(0xf7019bc57980d178),
    BFieldElement::new(0x0b56dd650d860d28),
    BFieldElement::new(0xcd095759d842636d),
    BFieldElement::new(0x17180482b913fb6d),
    BFieldElement::new(0xf085ee1e1911b728),
    BFieldElement::new(0x44549837566a0bfd),
    BFieldElement::new(0xadd0271c8a246c59),
    BFieldElement::new(0x4292bcfb6e12c8ce),
    BFieldElement::new(0xb9335e2407759a82),
    BFieldElement::new(0x08eb37a7111d5618),
    BFieldElement::new(0x2b7c80b429d54278),
    BFieldElement::new(0x53ea5d3e1999f4ce),
    BFieldElement::new(0xedee2a45b06fa613),
    BFieldElement::new(0xacaf4772a36459c2),
    BFieldElement::new(0xcc0b4aa757746b2f),
    BFieldElement::new(0x52536a16f47129c0),
    BFieldElement::new(0x9b0abd0b07ccc510),
    BFieldElement::new(0x3d3b86ee7806a3ee),
    BFieldElement::new(0x533072e923c588e5),
    BFieldElement::new(0x14555d5ea03b4428),
    BFieldElement::new(0xd9e7702629648bd3),
    BFieldElement::new(0x273c5dc3b8b6ed83),
    BFieldElement::new(0x26315afb96c27395),
    BFieldElement::new(0xbabc1eb9506c3767),
    BFieldElement::new(0x0d0281a2fb57b4c7),
    BFieldElement::new(0x2c709a3b40cce2bd),
    BFieldElement::new(0x9ab9a14cafb72112),
    BFieldElement::new(0x189aed3cd14d0f5e),
    BFieldElement::new(0xc941490f292f3566),
    BFieldElement::new(0x29d339bc4c4c5ed1),
    BFieldElement::new(0x51886b29d3b0d26e),
    BFieldElement::new(0x26ecf673ad01666f),
    BFieldElement::new(0x58f20c05df6136e4),
    BFieldElement::new(0xcaae02f891d9a693),
    BFieldElement::new(0x99ea2de13df560c7),
    BFieldElement::new(0xfb7e604032a6c89b),
    BFieldElement::new(0x9484e6a84078ffab),
    BFieldElement::new(0x33eb1f78ea2bf882),
    BFieldElement::new(0x305fba6c763df8a4),
    BFieldElement::new(0x855a940575a2f5cc),
    BFieldElement::new(0x3bbe96637669d8a9),
    BFieldElement::new(0xe0bf92ec5aa783b5),
    BFieldElement::new(0xd2d8f7be3c581792),
    BFieldElement::new(0x25d1dbdc95699756),
    BFieldElement::new(0x6811562dd7d380e7),
    BFieldElement::new(0xd2d0c07c8cb96126),
    BFieldElement::new(0xed7de2464aab8093),
    BFieldElement::new(0x28e68f8ff765cddd),
    BFieldElement::new(0x000f4302468decf6),
    BFieldElement::new(0xf408a8da2ec86476),
    BFieldElement::new(0xa910d923652cbc51),
    BFieldElement::new(0x61ff0fdd666285f9),
    BFieldElement::new(0x0d35de1fb1d92032),
    BFieldElement::new(0xc33f743297168e6b),
    BFieldElement::new(0x5280166634968f2f),
    BFieldElement::new(0x8029701a169b8c58),
    BFieldElement::new(0x442b1659f53fc92c),
    BFieldElement::new(0xb80d1df09583aae7),
    BFieldElement::new(0x3f68f6f41e84d283),
    BFieldElement::new(0x2dfd848230eeab8e),
    BFieldElement::new(0x464724542108bcce),
    BFieldElement::new(0x9cbd9af019eb40d8),
    BFieldElement::new(0x9cd056128537dca6),
    BFieldElement::new(0x0fdc70699871ec1f),
    BFieldElement::new(0x91b4ed0840036aa9),
    BFieldElement::new(0x0426a03ab3b941ae),
    BFieldElement::new(0x3620ce1d1da9d9ea),
    BFieldElement::new(0xa909c10f133b12df),
    BFieldElement::new(0xdff78cb9ce5e72af),
    BFieldElement::new(0x355a0507ea3d39e9),
    BFieldElement::new(0x33ab02012ebb8e90),
    BFieldElement::new(0x66e681125ad48d09),
    BFieldElement::new(0x754e7875a4cba209),
    BFieldElement::new(0xb4f4092cd96fe585),
    BFieldElement::new(0xe0beb61c71cd590b),
    BFieldElement::new(0x289d6191348815af),
    BFieldElement::new(0xc7f9310eebd89190),
    BFieldElement::new(0xc404c95833496b5d),
    BFieldElement::new(0x67cd987a2e477666),
    BFieldElement::new(0x83a966eaaba6f2d0),
    BFieldElement::new(0xa843095304ddb411),
    BFieldElement::new(0xab9bd2ffaf5ce005),
    BFieldElement::new(0x4b8b976e23ed8d5e),
    BFieldElement::new(0xe853009d6815ce32),
    BFieldElement::new(0x30b462f67decb8bd),
    BFieldElement::new(0x4fbe35dfc8561dde),
    BFieldElement::new(0xb182e2119eb521e0),
    BFieldElement::new(0x9a37c025983b008b),
    BFieldElement::new(0x9a8036d5c0b0ddff),
    BFieldElement::new(0x530e4069d0629e2f),
    BFieldElement::new(0xb5d7149987ef9062),
    BFieldElement::new(0x6a7120ccc56ca201),
    BFieldElement::new(0xd4fb060e7aed5a56),
    BFieldElement::new(0x9d01fca4ebb8856b),
    BFieldElement::new(0xe4bafe3340849d38),
    BFieldElement::new(0x0632a3df994f0c09),
    BFieldElement::new(0xf084664f0b9c8659),
    BFieldElement::new(0xbed5c89cfd8f9a70),
    BFieldElement::new(0x2732cb3f2103948c),
    BFieldElement::new(0x56b9259a2307cb16),
    BFieldElement::new(0xe9c504996e7d3273),
    BFieldElement::new(0x1c8a11e45261ef10),
    BFieldElement::new(0xdd694710e5643a96),
    BFieldElement::new(0xf84538050c75ba9c),
    BFieldElement::new(0x6d21207c2dd2ffdd),
    BFieldElement::new(0x398156c24d3ee292),
    BFieldElement::new(0x5dfe3b4a26dfdb31),
    BFieldElement::new(0x5264c0dce0fc508d),
    BFieldElement::new(0x68db82e30df8c0b6),
    BFieldElement::new(0x0bb69f951e7dae8f),
    BFieldElement::new(0xa4a00a693a6a9dc6),
    BFieldElement::new(0x7d2ed77fb67e1edb),
    BFieldElement::new(0x3a29153665b5deb6),
    BFieldElement::new(0xde05f000f7734b79),
    BFieldElement::new(0x2d3298b50e9b8ee3),
    BFieldElement::new(0x0a2801c5da77fe8d),
    BFieldElement::new(0xe4dae42bd80dc286),
    BFieldElement::new(0x2a92a223212d9bb1),
    BFieldElement::new(0x4a9e772dc0e7ae65),
    BFieldElement::new(0xf91ad425467514a6),
    BFieldElement::new(0xc947db43432f981c),
    BFieldElement::new(0xfab35dc94da6ff86),
    BFieldElement::new(0x30e86b98128ca1d8),
    BFieldElement::new(0x73045f05de00acdb),
    BFieldElement::new(0x877a90860890a505),
    BFieldElement::new(0x904b699fae24adf6),
    BFieldElement::new(0x2fd36112b8db92ea),
    BFieldElement::new(0x37c8e7b709efcfee),
    BFieldElement::new(0x117825ee6efb194d),
    BFieldElement::new(0xef084994ba70e740),
    BFieldElement::new(0x3bbbb3e35ddff9ff),
    BFieldElement::new(0x94599a930c7d397e),
    BFieldElement::new(0x3c962b6062d37a0b),
    BFieldElement::new(0x8cedf279060e80cf),
    BFieldElement::new(0xe3d6fee148661835),
    BFieldElement::new(0xac6dc576dcf69a1b),
    BFieldElement::new(0x4395a68ba78162dc),
    BFieldElement::new(0x1509b345505d0cfc),
    BFieldElement::new(0x78ea2a2c0ff44da3),
    BFieldElement::new(0x3b8453ac216d9dcc),
    BFieldElement::new(0xeeea1ecb6e9739da),
    BFieldElement::new(0x998a5a4578f042a1),
    BFieldElement::new(0x85b4d1176041379e),
    BFieldElement::new(0xf1c548623858b53e),
    BFieldElement::new(0x84f76ecd462db219),
    BFieldElement::new(0x7d478acf677e40c9),
    BFieldElement::new(0x78bd49e6248a3f8b),
    BFieldElement::new(0xbaa176467eac8dc0),
    BFieldElement::new(0xa1d87308ce5f171b),
    BFieldElement::new(0x6ecdc783e58e6ded),
    BFieldElement::new(0x0a27470d8a54c073),
    BFieldElement::new(0x303dedb68be27aff),
    BFieldElement::new(0x1283fa6b9fab0360),
    BFieldElement::new(0x09b9238f8990d3c6),
    BFieldElement::new(0x783fe452c827ce70),
    BFieldElement::new(0x51053b0a0b311026),
    BFieldElement::new(0x913af1029780fd02),
    BFieldElement::new(0x2768db16e909ab50),
    BFieldElement::new(0xe2711e4680cd49d4),
    BFieldElement::new(0xae38b891ffd47d23),
    BFieldElement::new(0x747a4123904482ef),
    BFieldElement::new(0x7590a5536989e07f),
    BFieldElement::new(0x597bcf699cb8f029),
    BFieldElement::new(0xcf1022c720e9c768),
    BFieldElement::new(0x6be4d569436457d0),
    BFieldElement::new(0x1ea4854422964af6),
    BFieldElement::new(0x9ae378da59b64449),
    BFieldElement::new(0x01111bdcbee5f71a),
    BFieldElement::new(0x3e5b8b050fbb6ffe),
    BFieldElement::new(0x5a14f21fc4493541),
    BFieldElement::new(0x1ce4f54dfef557a1),
    BFieldElement::new(0x5cd4599991cc8d4f),
    BFieldElement::new(0xc564b336f4eb42a2),
    BFieldElement::new(0xdcb6c46e887ac22e),
    BFieldElement::new(0x8a75e51054cd014f),
    BFieldElement::new(0xb0ff306616755d5e),
    BFieldElement::new(0x4dfee54c8dd19e47),
    BFieldElement::new(0x7ca27c634f5d6f20),
    BFieldElement::new(0x377a3f686a605c43),
    BFieldElement::new(0x942f96fe69242828),
    BFieldElement::new(0x63363826c868c690),
    BFieldElement::new(0xf622c7bbff21a41e),
    BFieldElement::new(0x7413fe06ae65a228),
    BFieldElement::new(0xf27e12c8f2b63a12),
    BFieldElement::new(0xcb9cdc2782a601b6),
    BFieldElement::new(0x4c355b11ae9d1501),
    BFieldElement::new(0x97b3caa2ca1983c8),
    BFieldElement::new(0x6f253eb18a69a3d6),
    BFieldElement::new(0x9daaad35dbe278d0),
    BFieldElement::new(0x9b154395fc9479de),
    BFieldElement::new(0xb88a4c10b121859b),
    BFieldElement::new(0x23f574023668b146),
    BFieldElement::new(0xe014dd76f0daeb2a),
    BFieldElement::new(0x47ca3455c99abbfc),
    BFieldElement::new(0xe15bcce7d6626f93),
    BFieldElement::new(0xe785e1d29b697a2a),
    BFieldElement::new(0x72f6e6335f2779b2),
    BFieldElement::new(0x63354be802a8d806),
    BFieldElement::new(0xe8680400a8b7da56),
    BFieldElement::new(0xa57878f6cd7dc1f5),
    BFieldElement::new(0xa3242e1666842e56),
    BFieldElement::new(0x9d4763931e5c80b8),
    BFieldElement::new(0xf5ad9042f9c2a2ea),
    BFieldElement::new(0xfde9819dcf522ccf),
    BFieldElement::new(0x2c1b7126142f99b2),
    BFieldElement::new(0x512a49e97331bcdf),
    BFieldElement::new(0xdc36f6f22179f9a4),
    BFieldElement::new(0x1287293218e2af1d),
    BFieldElement::new(0x9547747ddf7cd847),
    BFieldElement::new(0x27e9d7e39b288667),
    BFieldElement::new(0xf53ed68c44662e1c),
    BFieldElement::new(0xccd3913a837bf56d),
    BFieldElement::new(0xa54b0c67cee88c45),
    BFieldElement::new(0xc9204bc4f678e860),
    BFieldElement::new(0x3b2b3332135618af),
    BFieldElement::new(0x3bfe057360941901),
    BFieldElement::new(0x7ea1c40408898d03),
    BFieldElement::new(0x65735bc04a989af5),
    BFieldElement::new(0xc0d0f9741e308f84),
    BFieldElement::new(0x274505af11e8561c),
    BFieldElement::new(0x00c01a31a2a8adf7),
    BFieldElement::new(0xae16a9b5d19bd9eb),
    BFieldElement::new(0xeb47bf1cf2f8c536),
    BFieldElement::new(0x6281eff11a965684),
    BFieldElement::new(0x16fbec6cb544e37b),
    BFieldElement::new(0x43661c50ecbaf61d),
    BFieldElement::new(0x48ededeefe678215),
    BFieldElement::new(0x41454e2ac92e020f),
    BFieldElement::new(0x677ab8b3c9ba50e2),
    BFieldElement::new(0x7fe3981c7c556ff9),
    BFieldElement::new(0x35c8b8589248bedd),
    BFieldElement::new(0x8466ae507b53ba99),
    BFieldElement::new(0x3c1ce7a6557cd52c),
    BFieldElement::new(0xb547ce487a0afa54),
    BFieldElement::new(0x05edb6bc9e4c7961),
    BFieldElement::new(0x50573b9cb7beeefe),
    BFieldElement::new(0xb26c40ecb1535571),
    BFieldElement::new(0x4a9d2fb483061eb1),
    BFieldElement::new(0x9738a691d84dd72c),
    BFieldElement::new(0xedfb4e4efb79d5f8),
    BFieldElement::new(0xbfdb2e3c4d600b54),
    BFieldElement::new(0xa44a4620bb87c023),
    BFieldElement::new(0xedfd03c8933b6509),
    BFieldElement::new(0x6409ecdd7113b38f),
    BFieldElement::new(0xbc7dd042ff7eb60b),
    BFieldElement::new(0x1c41de22a36b7a75),
    BFieldElement::new(0x177fdf40e9646ded),
    BFieldElement::new(0x8eeff87cd83ea10d),
    BFieldElement::new(0xf4ba0b7ed08bafeb),
    BFieldElement::new(0xe5b3ede57adb7f0d),
    BFieldElement::new(0xd8bf37d380f76ba9),
    BFieldElement::new(0xb5e07f4e8f36d2f7),
    BFieldElement::new(0xf2014366c2e22b88),
    BFieldElement::new(0x876f5726bb600720),
    BFieldElement::new(0x2fdd356780329918),
    BFieldElement::new(0xa60294429a5d0408),
    BFieldElement::new(0x0fd714f70b67a879),
    BFieldElement::new(0xd3efd294503511e7),
    BFieldElement::new(0x9ae10ed6c78b28c6),
    BFieldElement::new(0xae221f0e399fa644),
    BFieldElement::new(0x7d75772d7d8bf011),
    BFieldElement::new(0x0a6321da6233b471),
    BFieldElement::new(0x9bb7b0e6658281be),
    BFieldElement::new(0x7b16cc63c27fe8f4),
    BFieldElement::new(0xbf53f47f87f56a6d),
    BFieldElement::new(0xe635bf90172d71e6),
    BFieldElement::new(0xccf02f48250d3fb5),
    BFieldElement::new(0x3d36aa7d16141aaf),
    BFieldElement::new(0xae1eb04558b772b4),
    BFieldElement::new(0x1d58ab28e8083c4f),
    BFieldElement::new(0x1d9bda2e12d3e534),
    BFieldElement::new(0x423dd55f154482ef),
    BFieldElement::new(0x72b65b610654133b),
    BFieldElement::new(0x487d3548f56a85f0),
    BFieldElement::new(0xd8c52e8889a97c5a),
    BFieldElement::new(0x505ec42c4db39362),
    BFieldElement::new(0xc217b41e7d0ef6e7),
    BFieldElement::new(0xa6fd6c3cda5c8473),
    BFieldElement::new(0xff5dd9dfcd788211),
    BFieldElement::new(0x303ff3284e6f7308),
    BFieldElement::new(0xa1aa11255463d565),
    BFieldElement::new(0x78e87d336ccc6981),
    BFieldElement::new(0xcbe6f13680f9259e),
    BFieldElement::new(0x43ed60b3f0605305),
    BFieldElement::new(0x285bb9012ad02b65),
    BFieldElement::new(0x5e3be4930ba77f91),
    BFieldElement::new(0x7d0b7f76259514de),
    BFieldElement::new(0x196f1021318d4d14),
    BFieldElement::new(0x32aeee488c159236),
    BFieldElement::new(0x7042b7b5a3393989),
    BFieldElement::new(0xefe7a9290c7ff77c),
    BFieldElement::new(0x429dc22cfde71457),
    BFieldElement::new(0x70a0a8b0cd2e4e92),
    BFieldElement::new(0x2a66e414a2e69a2d),
    BFieldElement::new(0xeb6d94e92221051c),
    BFieldElement::new(0x704157837371e9be),
    BFieldElement::new(0x8a085f6f2d59f8bb),
    BFieldElement::new(0xd195560e9297a989),
    BFieldElement::new(0x8ef6701b17a6930c),
    BFieldElement::new(0xbdb996211667ba57),
    BFieldElement::new(0xe240f4c1c4b2ad06),
    BFieldElement::new(0xfc2f101d588e0907),
    BFieldElement::new(0x791d9d059d498e68),
    BFieldElement::new(0xd21b9dfad1421e14),
    BFieldElement::new(0x3f40701396692e53),
    BFieldElement::new(0x837e836ab5a382ca),
    BFieldElement::new(0x6b7358c4631537d0),
    BFieldElement::new(0x7e3b633af145a310),
    BFieldElement::new(0xbac53ee64fd8b153),
    BFieldElement::new(0xb5e07a6e5b2b5242),
    BFieldElement::new(0xe754f45f540247cb),
    BFieldElement::new(0x23ada0ccd4a9e8e9),
    BFieldElement::new(0x5bc4109d42b4c5c8),
    BFieldElement::new(0x754fa15d43d39e06),
    BFieldElement::new(0x9a99178df6fb2088),
    BFieldElement::new(0xe8fca9a16c030c7a),
    BFieldElement::new(0xbb9f10f3f43aef47),
    BFieldElement::new(0x11350f1b54ae9adc),
    BFieldElement::new(0x665e172a588ee1db),
    BFieldElement::new(0x08302ce9f0ee5277),
    BFieldElement::new(0x494e7e049554756c),
    BFieldElement::new(0xded40cca27f6c2ae),
    BFieldElement::new(0xfa62692d0d41fc3c),
    BFieldElement::new(0x7b2f0fdcde013e9b),
    BFieldElement::new(0x0998315134e1d6a9),
    BFieldElement::new(0x5e59c08deb844be8),
    BFieldElement::new(0x3a37c9734e09cfb2),
    BFieldElement::new(0x31719e2d681d7414),
];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoseidonState {
    pub state: [BFieldElement; STATE_SIZE],
}

impl PoseidonState {
    #[inline]
    const fn new() -> PoseidonState {
        PoseidonState {
            state: [BFIELD_ZERO; STATE_SIZE],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct Poseidon {}

impl Poseidon {
    #[inline]
    fn batch_square(array: &mut [BFieldElement; STATE_SIZE]) {
        for a in array.iter_mut() {
            *a = a.square();
        }
    }

    #[inline]
    fn batch_mul_into(
        array: &mut [BFieldElement; STATE_SIZE],
        operand: [BFieldElement; STATE_SIZE],
    ) {
        for (a, b) in array.iter_mut().zip_eq(operand.iter()) {
            *a *= *b;
        }
    }

    #[inline]
    fn batch_mod_pow_alpha(array: [BFieldElement; STATE_SIZE]) -> [BFieldElement; STATE_SIZE] {
        let mut result = array;
        Self::batch_square(&mut result);
        Self::batch_mul_into(&mut result, array);
        Self::batch_square(&mut result);
        Self::batch_mul_into(&mut result, array);
        result
    }

    #[allow(dead_code)]
    fn batch_mod_pow(
        array: [BFieldElement; STATE_SIZE],
        power: u64,
    ) -> [BFieldElement; STATE_SIZE] {
        let mut acc = [BFieldElement::one(); STATE_SIZE];
        for i in (0..64).rev() {
            if i != 63 {
                Self::batch_square(&mut acc);
            }
            if power & (1 << i) != 0 {
                Self::batch_mul_into(&mut acc, array);
            }
        }

        acc
    }

    /// Apply one round of Poseidon
    #[inline]
    fn poseidon_round(sponge: &mut PoseidonState, round_index: usize) {
        debug_assert!(
            round_index < NUM_ROUNDS,
            "Cannot apply {}th round; only have {} in total.",
            round_index,
            NUM_ROUNDS
        );

        // S-box
        if !(NUM_FULL / 2..NUM_FULL / 2 + NUM_PARTIAL).contains(&round_index) {
            sponge.state = Self::batch_mod_pow_alpha(sponge.state);
        } else {
            sponge.state[STATE_SIZE - 1] = sponge.state[STATE_SIZE - 1].mod_pow(ALPHA);
        }

        // MDS matrix
        let mut v: [BFieldElement; STATE_SIZE] = [BFIELD_ZERO; STATE_SIZE];
        for i in 0..STATE_SIZE {
            for j in 0..STATE_SIZE {
                v[i] += MDS[i * STATE_SIZE + j] * sponge.state[j];
            }
        }
        sponge.state = v;

        // round constants A
        for i in 0..STATE_SIZE {
            sponge.state[i] += ROUND_CONSTANTS[round_index * STATE_SIZE + i];
        }
    }

    /// Apply the permutation to the state of a sponge.
    #[inline]
    fn permutation(sponge: &mut PoseidonState) {
        for i in 0..NUM_ROUNDS {
            Self::poseidon_round(sponge, i);
        }
    }

    /// hash_10
    /// Hash 10 elements, or two digests. There is no padding because
    /// the input length is fixed.
    pub fn hash_10(input: &[BFieldElement; 10]) -> [BFieldElement; 5] {
        let mut sponge = PoseidonState::new();

        // absorb once
        sponge.state[..10].copy_from_slice(input);

        // apply domain separation for fixed-length input
        sponge.state[10] = BFIELD_ONE;

        // apply permutation
        Self::permutation(&mut sponge);

        // squeeze once
        sponge.state[..5].try_into().unwrap()
    }

    /// hash_varlen hashes an arbitrary number of field elements.
    ///
    /// Takes care of padding by applying the padding rule: append a single 1 ∈ Fp
    /// and as many 0 ∈ Fp elements as required to make the number of input elements
    /// a multiple of `RATE`.
    pub fn hash_varlen(input: &[BFieldElement]) -> [BFieldElement; 5] {
        let mut sponge = PoseidonState::new();

        // pad input
        let mut padded_input = input.to_vec();
        padded_input.push(BFIELD_ONE);
        while padded_input.len() % RATE != 0 {
            padded_input.push(BFIELD_ZERO);
        }

        // absorb
        while !padded_input.is_empty() {
            for (sponge_state_element, &input_element) in sponge
                .state
                .iter_mut()
                .take(RATE)
                .zip_eq(padded_input.iter().take(RATE))
            {
                *sponge_state_element += input_element;
            }
            padded_input.drain(..RATE);
            Self::permutation(&mut sponge);
        }

        // squeeze once
        sponge.state[..5].try_into().unwrap()
    }
}

impl AlgebraicHasher for Poseidon {
    fn hash_slice(elements: &[BFieldElement]) -> Digest {
        Digest::new(Poseidon::hash_varlen(elements))
    }

    fn hash_pair(left: &Digest, right: &Digest) -> Digest {
        let mut input = [BFIELD_ZERO; 10];
        input[..DIGEST_LENGTH].copy_from_slice(&left.values());
        input[DIGEST_LENGTH..].copy_from_slice(&right.values());
        Digest::new(Poseidon::hash_10(&input))
    }
}

#[cfg(test)]
mod poseidon_tests {}
