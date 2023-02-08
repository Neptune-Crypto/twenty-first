# https://extgit.iaik.tugraz.at/krypto/hadeshash/-/raw/master/code/poseidonperm_x3_64_24_optimized.sage

import time

N = 1024
t = 16
n = int(N / t)
R_F = 8
R_P = 22
prime = 0xffffffff00000001
F = GF(prime)

timer_start = 0
timer_end = 0

round_constants = ['0x15ebea3fc73397c3', '0xd73cd9fbfe8e275c', '0x8c096bfce77f6c26', '0x4e128f68b53d8fea', '0x29b779a36b2763f6', '0xfe2adc6fb65acd08', '0x8d2520e725ad0955', '0x1c2392b214624d2a', '0x37482118206dcc6e', '0x2f829bed19be019a', '0x2fe298cb6f8159b0', '0x2bbad982deccdbbf', '0xbad568b8cc60a81e', '0xb86a814265baad10', '0xbec2005513b3acb3', '0x6bf89b59a07c2a94', '0xa25deeb835e230f5', '0x3c5bad8512b8b12a', '0x7230f73c3cb7a4f2', '0xa70c87f095c74d0f', '0x6b7606b830bb2e80', '0x6cd467cfc4f24274', '0xfeed794df42a9b0a', '0x8cf7cf6163b7dbd3', '0x9a6e9dda597175a0', '0xaa52295a684faf7b', '0x017b811cc3589d8d', '0x55bfb699b6181648', '0xc2ccaf71501c2421', '0x1707950327596402', '0xdd2fcdcd42a8229f', '0x8b9d7d5b27778a21', '0xac9a05525f9cf512', '0x2ba125c58627b5e8', '0xc74e91250a8147a5', '0xa3e64b640d5bb384', '0xf53047d18d1f9292', '0xbaaeddacae3a6374', '0xf2d0914a808b3db1', '0x18af1a3742bfa3b0', '0x9a621ef50c55bdb8', '0xc615f4d1cc5466f3', '0xb7fbac19a35cf793', '0xd2b1a15ba517e46d', '0x4a290c4d7fd26f6f', '0x4f0cf1bb1770c4c4', '0x548345386cd377f5', '0x33978d2789fddd42', '0xab78c59deb77e211', '0xc485b2a933d2be7f', '0xbde3792c00c03c53', '0xab4cefe8f893d247', '0xc5c0e752eab7f85f', '0xdbf5a76f893bafea', '0xa91f6003e3d984de', '0x099539077f311e87', '0x097ec52232f9559e', '0x53641bdf8991e48c', '0x2afe9711d5ed9d7c', '0xa7b13d3661b5d117', '0x5a0e243fe7af6556', '0x1076fae8932d5f00', '0x9b53a83d434934e3', '0xed3fd595a3c0344a', '0x28eff4b01103d100', '0x60400ca3e2685a45', '0x1c8636beb3389b84', '0xac1332b60e13eff0', '0x2adafcc364e20f87', '0x79ffc2b14054ea0b', '0x3f98e4c0908f0a05', '0xcdb230bc4e8a06c4', '0x1bcaf7705b152a74', '0xd9bca249a82a7470', '0x91e24af19bf82551', '0xa62b43ba5cb78858', '0xb4898117472e797f', '0xb3228bca606cdaa0', '0x844461051bca39c9', '0xf3411581f6617d68', '0xf7fd50646782b533', '0x6ca664253c18fb48', '0x2d2fcdec0886a08f', '0x29da00dd799b575e', '0x47d966cc3b6e1e93', '0xde884e9a17ced59e', '0xdacf46dc1c31a045', '0x5d2e3c121eb387f2', '0x51f8b0658b124499', '0x1e7dbd1daa72167d', '0x8275015a25c55b88', '0xe8521c24ac7a70b3', '0x6521d121c40b3f67', '0xac12de797de135b0', '0xafa28ead79f6ed6a', '0x685174a7a8d26f0b', '0xeff92a08d35d9874', '0x3058734b76dd123a', '0xfa55dcfba429f79c', '0x559294d4324c7728', '0x7a770f53012dc178', '0xedd8f7c408f3883b', '0x39b533cf8d795fa5', '0x160ef9de243a8c0a', '0x431d52da6215fe3f', '0x54c51a2a2ef6d528', '0x9b13892b46ff9d16', '0x263c46fcee210289', '0xb738c96d25aabdc4', '0x5c33a5203996d38f', '0x2626496e7c98d8dd', '0xc669e0a52785903a', '0xaecde726c8ae1f47', '0x039343ef3a81e999', '0x2615ceaf044a54f9', '0x7e41e834662b66e1', '0x4ca5fd4895335783', '0x64b334d02916f2b0', '0x87268837389a6981', '0x034b75bcb20a6274', '0x58e658296cc2cd6e', '0xe2d0f759acc31df4', '0x81a652e435093e20', '0x0b72b6e0172eaf47', '0x4aec43cec577d66d', '0xde78365b028a84e6', '0x444e19569adc0ee4', '0x942b2451fa40d1da', '0xe24506623ea5bd6c', '0x082854bf2ef7c743', '0x69dbbc566f59d62e', '0x248c38d02a7b5cb2', '0x4f4e8f8c09d15edb', '0xd96682f188d310cf', '0x6f9a25d56818b54c', '0xb6cefed606546cd9', '0x5bc07523da38a67b', '0x7df5a3c35b8111cf', '0xaaa2cc5d4db34bb0', '0x9e673ff22a4653f8', '0xbd8b278d60739c62', '0xe10d20f6925b8815', '0xf6c87b91dd4da2bf', '0xfed623e2f71b6f1a', '0xa0f02fa52a94d0d3', '0xbb5794711b39fa16', '0xd3b94fba9d005c7f', '0x15a26e89fad946c9', '0xf3cb87db8a67cf49', '0x400d2bf56aa2a577', '0x56045ba28fd1dbb1', '0x06851d67ed5acacd', '0xf7019bc57980d178', '0x0b56dd650d860d28', '0xcd095759d842636d', '0x17180482b913fb6d', '0xf085ee1e1911b728', '0x44549837566a0bfd', '0xadd0271c8a246c59', '0x4292bcfb6e12c8ce', '0xb9335e2407759a82', '0x08eb37a7111d5618', '0x2b7c80b429d54278', '0x53ea5d3e1999f4ce', '0xedee2a45b06fa613', '0xacaf4772a36459c2', '0xcc0b4aa757746b2f', '0x52536a16f47129c0', '0x9b0abd0b07ccc510', '0x3d3b86ee7806a3ee', '0x533072e923c588e5', '0x14555d5ea03b4428', '0xd9e7702629648bd3', '0x273c5dc3b8b6ed83', '0x26315afb96c27395', '0xbabc1eb9506c3767', '0x0d0281a2fb57b4c7', '0x2c709a3b40cce2bd', '0x9ab9a14cafb72112', '0x189aed3cd14d0f5e', '0xc941490f292f3566', '0x29d339bc4c4c5ed1', '0x51886b29d3b0d26e', '0x26ecf673ad01666f', '0x58f20c05df6136e4', '0xcaae02f891d9a693', '0x99ea2de13df560c7', '0xfb7e604032a6c89b', '0x9484e6a84078ffab', '0x33eb1f78ea2bf882', '0x305fba6c763df8a4', '0x855a940575a2f5cc', '0x3bbe96637669d8a9', '0xe0bf92ec5aa783b5', '0xd2d8f7be3c581792', '0x25d1dbdc95699756', '0x6811562dd7d380e7', '0xd2d0c07c8cb96126', '0xed7de2464aab8093', '0x28e68f8ff765cddd', '0x000f4302468decf6', '0xf408a8da2ec86476', '0xa910d923652cbc51', '0x61ff0fdd666285f9', '0x0d35de1fb1d92032', '0xc33f743297168e6b', '0x5280166634968f2f', '0x8029701a169b8c58', '0x442b1659f53fc92c', '0xb80d1df09583aae7', '0x3f68f6f41e84d283', '0x2dfd848230eeab8e', '0x464724542108bcce', '0x9cbd9af019eb40d8', '0x9cd056128537dca6', '0x0fdc70699871ec1f', '0x91b4ed0840036aa9', '0x0426a03ab3b941ae', '0x3620ce1d1da9d9ea', '0xa909c10f133b12df', '0xdff78cb9ce5e72af', '0x355a0507ea3d39e9', '0x33ab02012ebb8e90', '0x66e681125ad48d09', '0x754e7875a4cba209', '0xb4f4092cd96fe585', '0xe0beb61c71cd590b', '0x289d6191348815af', '0xc7f9310eebd89190', '0xc404c95833496b5d', '0x67cd987a2e477666', '0x83a966eaaba6f2d0', '0xa843095304ddb411', '0xab9bd2ffaf5ce005', '0x4b8b976e23ed8d5e', '0xe853009d6815ce32', '0x30b462f67decb8bd', '0x4fbe35dfc8561dde', '0xb182e2119eb521e0', '0x9a37c025983b008b', '0x9a8036d5c0b0ddff', '0x530e4069d0629e2f', '0xb5d7149987ef9062', '0x6a7120ccc56ca201', '0xd4fb060e7aed5a56', '0x9d01fca4ebb8856b', '0xe4bafe3340849d38', '0x0632a3df994f0c09', '0xf084664f0b9c8659', '0xbed5c89cfd8f9a70', '0x2732cb3f2103948c', '0x56b9259a2307cb16', '0xe9c504996e7d3273', '0x1c8a11e45261ef10', '0xdd694710e5643a96', '0xf84538050c75ba9c', '0x6d21207c2dd2ffdd', '0x398156c24d3ee292', '0x5dfe3b4a26dfdb31', '0x5264c0dce0fc508d', '0x68db82e30df8c0b6', '0x0bb69f951e7dae8f', '0xa4a00a693a6a9dc6', '0x7d2ed77fb67e1edb', '0x3a29153665b5deb6', '0xde05f000f7734b79', '0x2d3298b50e9b8ee3', '0x0a2801c5da77fe8d', '0xe4dae42bd80dc286', '0x2a92a223212d9bb1', '0x4a9e772dc0e7ae65', '0xf91ad425467514a6', '0xc947db43432f981c', '0xfab35dc94da6ff86', '0x30e86b98128ca1d8', '0x73045f05de00acdb', '0x877a90860890a505', '0x904b699fae24adf6', '0x2fd36112b8db92ea', '0x37c8e7b709efcfee', '0x117825ee6efb194d', '0xef084994ba70e740', '0x3bbbb3e35ddff9ff', '0x94599a930c7d397e', '0x3c962b6062d37a0b', '0x8cedf279060e80cf', '0xe3d6fee148661835', '0xac6dc576dcf69a1b', '0x4395a68ba78162dc', '0x1509b345505d0cfc', '0x78ea2a2c0ff44da3', '0x3b8453ac216d9dcc', '0xeeea1ecb6e9739da', '0x998a5a4578f042a1', '0x85b4d1176041379e', '0xf1c548623858b53e', '0x84f76ecd462db219', '0x7d478acf677e40c9', '0x78bd49e6248a3f8b', '0xbaa176467eac8dc0', '0xa1d87308ce5f171b', '0x6ecdc783e58e6ded', '0x0a27470d8a54c073', '0x303dedb68be27aff', '0x1283fa6b9fab0360', '0x09b9238f8990d3c6', '0x783fe452c827ce70', '0x51053b0a0b311026', '0x913af1029780fd02', '0x2768db16e909ab50', '0xe2711e4680cd49d4', '0xae38b891ffd47d23', '0x747a4123904482ef', '0x7590a5536989e07f', '0x597bcf699cb8f029', '0xcf1022c720e9c768', '0x6be4d569436457d0', '0x1ea4854422964af6', '0x9ae378da59b64449', '0x01111bdcbee5f71a', '0x3e5b8b050fbb6ffe', '0x5a14f21fc4493541', '0x1ce4f54dfef557a1', '0x5cd4599991cc8d4f', '0xc564b336f4eb42a2', '0xdcb6c46e887ac22e', '0x8a75e51054cd014f', '0xb0ff306616755d5e', '0x4dfee54c8dd19e47', '0x7ca27c634f5d6f20', '0x377a3f686a605c43', '0x942f96fe69242828', '0x63363826c868c690', '0xf622c7bbff21a41e', '0x7413fe06ae65a228', '0xf27e12c8f2b63a12', '0xcb9cdc2782a601b6', '0x4c355b11ae9d1501', '0x97b3caa2ca1983c8', '0x6f253eb18a69a3d6', '0x9daaad35dbe278d0', '0x9b154395fc9479de', '0xb88a4c10b121859b', '0x23f574023668b146', '0xe014dd76f0daeb2a', '0x47ca3455c99abbfc', '0xe15bcce7d6626f93', '0xe785e1d29b697a2a', '0x72f6e6335f2779b2', '0x63354be802a8d806', '0xe8680400a8b7da56', '0xa57878f6cd7dc1f5', '0xa3242e1666842e56', '0x9d4763931e5c80b8', '0xf5ad9042f9c2a2ea', '0xfde9819dcf522ccf', '0x2c1b7126142f99b2', '0x512a49e97331bcdf', '0xdc36f6f22179f9a4', '0x1287293218e2af1d', '0x9547747ddf7cd847', '0x27e9d7e39b288667', '0xf53ed68c44662e1c', '0xccd3913a837bf56d', '0xa54b0c67cee88c45', '0xc9204bc4f678e860', '0x3b2b3332135618af', '0x3bfe057360941901', '0x7ea1c40408898d03', '0x65735bc04a989af5', '0xc0d0f9741e308f84', '0x274505af11e8561c', '0x00c01a31a2a8adf7', '0xae16a9b5d19bd9eb', '0xeb47bf1cf2f8c536', '0x6281eff11a965684', '0x16fbec6cb544e37b', '0x43661c50ecbaf61d', '0x48ededeefe678215', '0x41454e2ac92e020f', '0x677ab8b3c9ba50e2', '0x7fe3981c7c556ff9', '0x35c8b8589248bedd', '0x8466ae507b53ba99', '0x3c1ce7a6557cd52c', '0xb547ce487a0afa54', '0x05edb6bc9e4c7961', '0x50573b9cb7beeefe', '0xb26c40ecb1535571', '0x4a9d2fb483061eb1', '0x9738a691d84dd72c', '0xedfb4e4efb79d5f8', '0xbfdb2e3c4d600b54', '0xa44a4620bb87c023', '0xedfd03c8933b6509', '0x6409ecdd7113b38f', '0xbc7dd042ff7eb60b', '0x1c41de22a36b7a75', '0x177fdf40e9646ded', '0x8eeff87cd83ea10d', '0xf4ba0b7ed08bafeb', '0xe5b3ede57adb7f0d', '0xd8bf37d380f76ba9', '0xb5e07f4e8f36d2f7', '0xf2014366c2e22b88', '0x876f5726bb600720', '0x2fdd356780329918', '0xa60294429a5d0408', '0x0fd714f70b67a879', '0xd3efd294503511e7', '0x9ae10ed6c78b28c6', '0xae221f0e399fa644', '0x7d75772d7d8bf011', '0x0a6321da6233b471', '0x9bb7b0e6658281be', '0x7b16cc63c27fe8f4', '0xbf53f47f87f56a6d', '0xe635bf90172d71e6', '0xccf02f48250d3fb5', '0x3d36aa7d16141aaf', '0xae1eb04558b772b4', '0x1d58ab28e8083c4f', '0x1d9bda2e12d3e534', '0x423dd55f154482ef', '0x72b65b610654133b', '0x487d3548f56a85f0', '0xd8c52e8889a97c5a', '0x505ec42c4db39362', '0xc217b41e7d0ef6e7', '0xa6fd6c3cda5c8473', '0xff5dd9dfcd788211', '0x303ff3284e6f7308', '0xa1aa11255463d565', '0x78e87d336ccc6981', '0xcbe6f13680f9259e', '0x43ed60b3f0605305', '0x285bb9012ad02b65', '0x5e3be4930ba77f91', '0x7d0b7f76259514de', '0x196f1021318d4d14', '0x32aeee488c159236', '0x7042b7b5a3393989', '0xefe7a9290c7ff77c', '0x429dc22cfde71457', '0x70a0a8b0cd2e4e92', '0x2a66e414a2e69a2d', '0xeb6d94e92221051c', '0x704157837371e9be', '0x8a085f6f2d59f8bb', '0xd195560e9297a989', '0x8ef6701b17a6930c', '0xbdb996211667ba57', '0xe240f4c1c4b2ad06', '0xfc2f101d588e0907', '0x791d9d059d498e68', '0xd21b9dfad1421e14', '0x3f40701396692e53', '0x837e836ab5a382ca', '0x6b7358c4631537d0', '0x7e3b633af145a310', '0xbac53ee64fd8b153', '0xb5e07a6e5b2b5242', '0xe754f45f540247cb', '0x23ada0ccd4a9e8e9', '0x5bc4109d42b4c5c8', '0x754fa15d43d39e06', '0x9a99178df6fb2088', '0xe8fca9a16c030c7a', '0xbb9f10f3f43aef47', '0x11350f1b54ae9adc', '0x665e172a588ee1db', '0x08302ce9f0ee5277', '0x494e7e049554756c', '0xded40cca27f6c2ae', '0xfa62692d0d41fc3c', '0x7b2f0fdcde013e9b', '0x0998315134e1d6a9', '0x5e59c08deb844be8', '0x3a37c9734e09cfb2', '0x31719e2d681d7414']
MDS_matrix = [['0x5988b85748e01d64', '0x6ce5b2a51c028cd5', '0x97151726a74103eb', '0x48da805da22db860', '0xab527bddf567ce5d', '0xa62c7a8802d557bf', '0x682f001eff55d953', '0x739f833269079c91', '0x52f36cc427b3c2d1', '0x207f3edd31fec0c1', '0xe861a14803653f31', '0x7f75391531d1351a', '0x9c1ff22b7b6519c0', '0x33b1b3b9f6243b4d', '0x39a21454890d9e9f', '0x5153a55d83d32ad0'],['0x2c7fa8fb1e89be4a', '0x53aaf9e104e4e906', '0x5466ca73c94286db', '0x01533ab42def6283', '0xa1d91119905de8f2', '0xc4a70e83b3da8175', '0x9c4fa46f80a875b7', '0x2ab64e023d7a4574', '0x95870ac3b43221ad', '0x8ec4c7dfe0b1e529', '0x29af3c3d9caab861', '0x0f5e03a41985622f', '0x6e351f8b80d25233', '0x34c392b28480b011', '0xa2e21b3c21902702', '0x98c40b93bfa7d635'],['0x3669f4cbc4539d7e', '0xf8fee0879daec673', '0xb732c1f23b761aee', '0xceee88ee79daf441', '0x86c99724e2cef313', '0x1cf1f3ab213a0d3c', '0xd5ea051783a3122a', '0x22352d4ea73f6045', '0xf383c05e03cd7c62', '0x81dfe9c8a0051bbe', '0x3f07ffcdd9266723', '0xc93603d00f5dc893', '0xe69d2645321274ab', '0xd4d06b40d3b2458d', '0x3c3b9bf934fc9fd7', '0x6c3ed734df9cf691'],['0x3c3edb0d8e367cbe', '0x4d141f636fb94d24', '0x593dc28f61908b6d', '0x51429b67ced5b2ea', '0xb8ad92e65afe1948', '0x7de771ac1136a634', '0xf2bc3d4d9f8c12b4', '0xa00899fee1163f9a', '0xefd9361aba69c5f9', '0x692af10ed750d360', '0x203a34814f5fdba1', '0x22aa4b0fc8e4f127', '0x8404c6ef28881b6d', '0x332c23191cb610fd', '0xe03b41f9d0a5d63f', '0xf6bc920a03f7752b'],['0x35accf0a901fcb8d', '0x8dcff51324521386', '0x57e239c58d0c8e89', '0xfaed102b92f35770', '0x3e4bfd2caf5c21f7', '0x3cd53e002583952a', '0xea2a4a22eb681b2c', '0x670a2ee21d7014eb', '0xdde0a9aa6df4b771', '0x95b1fbc56e1a6c39', '0x6346cd06cc929757', '0x6ae4bda639af92e0', '0x77314dfd51e7d92c', '0x0391e3978ff24b17', '0xe27611f117d6d97f', '0x04883758be9f54e3'],['0xe6d3dda85e6395c2', '0x0cd42c05f66b3c1c', '0x698359374266b60b', '0x9e9eaa8893e52d50', '0x034432c8f57c7559', '0x4674cb4bc8006eae', '0x92010504468f2681', '0x182baea6c583b9c6', '0xfd348e1d8e95d693', '0xc2f2fd9b9078ece9', '0x1aba1a45a6419bf9', '0xbcc2b6584c1af690', '0xad0b58758270f4ca', '0xd4e24b7e35851406', '0x9572342e0aec74bd', '0xf681bb067f7ac76e'],['0x30f21444e981732e', '0x0a4b4c356cf81323', '0xe0ddde574b7bf32b', '0xc3f99cc853e76784', '0xe1aa2dd912c46532', '0x54531f9f67108f80', '0x8639b004a91e6a28', '0x8cd1ddb584f7a9ac', '0xd78dad0f1337bba4', '0xc481f8f3a4b87f91', '0x4c24f9db5e94ab02', '0x1c62f47a8c19eddb', '0xbf16cec4fba38913', '0x4ed589b4fd6bbaf0', '0x86cff751dac92c05', '0x86249af3a8eff0f6'],['0xe1033c0d774c285d', '0x2f8586c1921d8fb4', '0x7441a10af45e5808', '0x8d8feb70aded8628', '0x7da45cc544d9ae35', '0x767f12ba8a5a0c14', '0x620efed7f8d3a459', '0xdd7ff646c99f70ad', '0x3678cc601846b44a', '0x6755b4209c3401cc', '0x8e0ae4aa7035b0eb', '0xba9b1267dd354322', '0xfadd3d5f0683e9b6', '0xbc5c35ae68f2dfe2', '0x73ae2e3c9c090886', '0xcc0c1c1e8d9bef49'],['0x8671921f7a0a1794', '0x9cd5f1eac6037499', '0x3e527bd0148ade83', '0xaa717d8659a564c3', '0x548583860ccd6913', '0x0df5a873cf6cef5f', '0x6b1a27e8227f6df8', '0xdb3e44aa830df1c7', '0x71353fc324da6e34', '0x519ed83e6503d741', '0x81d74e3914f76769', '0x6c8a3d26ef84b973', '0x48033d71b77fe8c1', '0x0b845bc4926fb555', '0x9e07dbc3ba174490', '0x0ec39acc7f68d8ce'],['0x7677a79600252463', '0xf3e2c73584829d44', '0x554b023d8621a4de', '0xf8afcc7e5c120419', '0x385e05151f117f7c', '0x6f349729b45b5be6', '0x3a98b59f73759ee1', '0xd98152e584082edb', '0x795f0382526cb465', '0x3d7631494b438675', '0x1003b04cc5497839', '0x0f224853d28542fe', '0xf83d612bda7234b5', '0xdbfc370b88104851', '0xc1dd2e22be7c3a16', '0x17ca5948341a933b'],['0xa9d70e4e8c01ad89', '0x47bb5a0c5a8da9b0', '0xc7b5e1da10d895f7', '0x5f0cc2c2aaa7b03e', '0xbc8097f44e63863f', '0x70c8e880581bd43e', '0x758cfe2ffa079581', '0x839395f285d3ece8', '0xa2f27080ac482121', '0x3f62d1fcb8f9d784', '0x4b2c662ed6ae100c', '0x14f3fbce248827f0', '0xbda2a9225eda8792', '0x1ed0e69117ef3c36', '0x3af8405a89b5c1a5', '0x3429a7904ea79c9d'],['0x9d8119220744a23b', '0x730bf95139f665b1', '0xf71f347cf06eb11b', '0x1e16d46ef71b8e86', '0xffb5517c741183a7', '0x9de7bdf6e7763c76', '0x641535f22ebbb8b5', '0xcc9a2ac6e3525ff2', '0x6ca74aadfd1870b2', '0x26c7b78d1a569b93', '0x6771174eb61e5363', '0xfece43af74186a6c', '0xe2d04ce9d92aac6c', '0x069bce14795ab3ee', '0x454663cdef4ad361', '0xc9d1d1b13aa366ba'],['0x2a14bb5529f46c5e', '0x9284dc3ec5a3298e', '0xc32a357d30aa5298', '0x6378633155977074', '0xd42fbe0d1987c1d4', '0x812a7402da911fc3', '0x0aa377c56139e86d', '0xf738501437323e35', '0x716814c20125dd1a', '0x4b8f884ea190466e', '0x73af1afa89a8b0f8', '0x581257e0a6514156', '0x2a3e9650920f7c21', '0x3a4b286082ce7d23', '0xaa89fb9040aed6fa', '0xcbcd108576e6d1ba'],['0xa0e6af946333f9cc', '0x5b9fa025d5f91de4', '0xa37f3398a3ff534c', '0x5d807c5491b334cc', '0xdde0491f392f54b3', '0xd31f9255564584e5', '0xb682b5175a329122', '0x36ac21cacdf0a097', '0xbd69c3e422624ed0', '0xfd03026489b74c0b', '0xb2b862a95c6f5fc9', '0x7a6ec2a715549e03', '0x6a3cf97409861804', '0xdd31032efb7843cc', '0x080d82ec2393f0c3', '0x5ca5e139dcf07d41'],['0xf57bae2bb3999ef3', '0x1b3241817e6f40c5', '0x12f1d5f21156e7b5', '0x44261d736bf454a8', '0xe1194d1a1cd17a51', '0x9394f656c1246dc8', '0x050fe6e2bb26bab4', '0xcb184bfb067f33a3', '0x87b5f94c9d2c7907', '0x456daf135d96f358', '0x116eba266dcc0cb3', '0xa2dbf3c480582332', '0x1302f5ef13049a4d', '0xd2ac7ba5adc5269a', '0x43b32aedb14c08cc', '0xe00837b7287de52d'],['0xd9553f671aad6e14', '0x4bb5271bb83f98a0', '0x2c84ca7932ff0d9e', '0x0ac13c2ec615ff05', '0xfd066e3f8f82e427', '0x6cddcb7116f97320', '0xce298f1a321e7087', '0x74c9ceb2944dc1b5', '0x6afbdadac658ec67', '0x8c32f317e710b3bd', '0xc380701ec43dc56d', '0x96b0a79e387441a4', '0xcac77034b3035048', '0x015fd2a7e0a193ac', '0x8f848885347161d7', '0x110276a5157b853a']]

# convert hexadecimal strings to field elements
MDS_matrix_field = matrix(F, t, t)
for i in range(0, t):
    for j in range(0, t):
        MDS_matrix_field[i, j] = F(int(MDS_matrix[i][j], 16))
round_constants_field = []
for i in range(0, (R_F + R_P) * t):
    round_constants_field.append(F(int(round_constants[i], 16)))

def print_words_to_hex(words):
    hex_length = int(ceil(float(n) / 4)) + 2 # +2 for "0x"
    print(["{0:#0{1}x}".format(int(entry), hex_length) for entry in words])

def print_concat_words_to_large(words):
    hex_length = int(ceil(float(n) / 4))
    nums = ["{0:0{1}x}".format(int(entry), hex_length) for entry in words]
    final_string = "0x" + ''.join(nums)
    print(final_string)

def calc_equivalent_constants(constants):
    constants_temp = [constants[index:index+t] for index in range(0, len(constants), t)]

    MDS_matrix_field_transpose = MDS_matrix_field.transpose()

    # Start moving round constants up
    # Calculate c_i' = M^(-1) * c_(i+1)
    # Split c_i': Add c_i'[0] AFTER the S-box, add the rest to c_i
    # I.e.: Store c_i'[0] for each of the partial rounds, and make c_i = c_i + c_i' (where now c_i'[0] = 0)
    num_rounds = R_F + R_P
    R_f = R_F / 2
    for i in range(num_rounds - 2 - R_f, R_f - 1, -1):
        inv_cip1 = list(vector(constants_temp[i+1]) * MDS_matrix_field_transpose.inverse())
        constants_temp[i] = list(vector(constants_temp[i]) + vector([0] + inv_cip1[1:]))
        constants_temp[i+1] = [inv_cip1[0]] + [0] * (t-1)

    return constants_temp

def calc_equivalent_matrices():
    # Following idea: Split M into M' * M'', where M'' is "cheap" and M' can move before the partial nonlinear layer
    # The "previous" matrix layer is then M * M'. Due to the construction of M', the M[0,0] and v values will be the same for the new M' (and I also, obviously)
    # Thus: Compute the matrices, store the w_hat and v_hat values

    MDS_matrix_field_transpose = MDS_matrix_field.transpose()

    w_hat_collection = []
    v_collection = []
    v = MDS_matrix_field_transpose[[0], list(range(1,t))]
    # print "M:", MDS_matrix_field_transpose
    # print "v:", v
    M_mul = MDS_matrix_field_transpose
    M_i = matrix(F, t, t)
    for i in range(R_P - 1, -1, -1):
        M_hat = M_mul[list(range(1,t)), list(range(1,t))]
        w = M_mul[list(range(1,t)), [0]]
        v = M_mul[[0], list(range(1,t))]
        v_collection.append(v.list())
        w_hat = M_hat.inverse() * w
        w_hat_collection.append(w_hat.list())

        # Generate new M_i, and multiplication M * M_i for "previous" round
        M_i = matrix.identity(t)
        M_i[list(range(1,t)), list(range(1,t))] = M_hat
        #M_mul = MDS_matrix_field_transpose * M_i

        test_mat = matrix(F, t, t)
        test_mat[[0], list(range(0, t))] = MDS_matrix_field_transpose[[0], list(range(0, t))]
        test_mat[[0], list(range(1, t))] = v
        test_mat[list(range(1, t)), [0]] = w_hat
        test_mat[list(range(1,t)), list(range(1,t))] = matrix.identity(t-1)

        # print M_mul == M_i * test_mat
        M_mul = MDS_matrix_field_transpose * M_i
        #return[M_i, test_mat]


        #M_mul = MDS_matrix_field_transpose * M_i
        #exit()
    #exit()
        

    # print [M_i, w_hat_collection, MDS_matrix_field_transpose[0, 0], v.list()]
    return [M_i, v_collection, w_hat_collection, MDS_matrix_field_transpose[0, 0]]

def cheap_matrix_mul(state_words, v, w_hat, M_0_0):
    state_words_new = [0] * t
    column_1 = [M_0_0] + w_hat
    state_words_new[0] = sum([column_1[i] * state_words[i] for i in range(0, t)])
    mul_row = [(state_words[0] * v[i]) for i in range(0, t-1)]
    add_row = [(mul_row[i] + state_words[i+1]) for i in range(0, t-1)]
    state_words_new = [state_words_new[0]] + add_row

    return state_words_new

def perm(input_words):
    round_constants_field_new = calc_equivalent_constants(round_constants_field)
    [M_i, v_collection, w_hat_collection, M_0_0] = calc_equivalent_matrices()
    #[M_i, test_mat] = calc_equivalent_matrices()
    
    global timer_start, timer_end

    timer_start = time.time()

    R_f = int(R_F / 2)

    round_constants_round_counter = 0

    state_words = list(input_words)

    # First full rounds
    for r in range(0, R_f):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants_field_new[round_constants_round_counter][i]
        for i in range(0, t):
            state_words[i] = (state_words[i])^3
        state_words = list(MDS_matrix_field * vector(state_words))
        round_constants_round_counter += 1

    # Middle partial rounds
    # Initial constants addition
    for i in range(0, t):
        state_words[i] = state_words[i] + round_constants_field_new[round_constants_round_counter][i]
    # First full matrix multiplication
    state_words = list(vector(state_words) * M_i)
    for r in range(0, R_P):
        # Round constants, nonlinear layer, matrix multiplication
        #state_words = list(vector(state_words) * M_i)
        state_words[0] = (state_words[0])^3
        # Moved constants addition
        if r < (R_P - 1):
            round_constants_round_counter += 1
            state_words[0] = state_words[0] + round_constants_field_new[round_constants_round_counter][0]
        # Optimized multiplication with cheap matrices
        state_words = cheap_matrix_mul(state_words, v_collection[R_P - r - 1], w_hat_collection[R_P - r - 1], M_0_0)
    round_constants_round_counter += 1

    # Last full rounds
    for r in range(0, R_f):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants_field_new[round_constants_round_counter][i]
        for i in range(0, t):
            state_words[i] = (state_words[i])^3
        state_words = list(MDS_matrix_field * vector(state_words))
        round_constants_round_counter += 1

    timer_end = time.time()
    
    return state_words

def perm_original(input_words):
    round_constants_field_new = [round_constants_field[index:index+t] for index in range(0, len(round_constants_field), t)]

    global timer_start, timer_end
    
    timer_start = time.time()

    R_f = int(R_F / 2)

    round_constants_round_counter = 0

    state_words = list(input_words)

    # First full rounds
    for r in range(0, R_f):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants_field_new[round_constants_round_counter][i]
        for i in range(0, t):
            state_words[i] = (state_words[i])^3
        state_words = list(MDS_matrix_field * vector(state_words))
        round_constants_round_counter += 1

    # Middle partial rounds
    for r in range(0, R_P):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants_field_new[round_constants_round_counter][i]
        state_words[0] = (state_words[0])^3
        state_words = list(MDS_matrix_field * vector(state_words))
        round_constants_round_counter += 1

    # Last full rounds
    for r in range(0, R_f):
        # Round constants, nonlinear layer, matrix multiplication
        for i in range(0, t):
            state_words[i] = state_words[i] + round_constants_field_new[round_constants_round_counter][i]
        for i in range(0, t):
            state_words[i] = (state_words[i])^3
        state_words = list(MDS_matrix_field * vector(state_words))
        round_constants_round_counter += 1
    
    timer_end = time.time()

    return state_words


# compute round constants and MDS matrix components for efficient evaluation of the permutation
fast_constants = calc_equivalent_constants(round_constants_field)
[M_i, v_collection, w_hat_collection, M_0_0] = calc_equivalent_matrices()
print(f"\n === Constants ===\n{fast_constants}")
print(f"\n === M_i ===\n{M_i}")
print(f"\n === v_collection ===\n{v_collection}")
print(f"\n === w_hat_collection ===\n{w_hat_collection}")
print(f"\n === M_0_0 ===\n{M_0_0}")
