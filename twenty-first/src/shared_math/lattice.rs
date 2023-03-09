use std::ops::{Add, AddAssign, Mul, Sub};

use itertools::Itertools;
use num_traits::Zero;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use super::b_field_element::BFieldElement;

pub fn coset_intt_noswap_64(array: &mut [BFieldElement; 64]) {
    const N: usize = 64;
    const N_INV: BFieldElement = BFieldElement::new(18158513693329981441);
    let powers_of_psi_inv_bitreversed = [
        BFieldElement::new(1),
        BFieldElement::new(18446462594437873665),
        BFieldElement::new(18446742969902956801),
        BFieldElement::new(18446744069397807105),
        BFieldElement::new(18442240469788262401),
        BFieldElement::new(18446744000695107585),
        BFieldElement::new(17293822564807737345),
        BFieldElement::new(18446744069414580225),
        BFieldElement::new(18158513693329981441),
        BFieldElement::new(18446739671368073217),
        BFieldElement::new(18446744052234715141),
        BFieldElement::new(18446744069414322177),
        BFieldElement::new(18446673700670423041),
        BFieldElement::new(18446744068340842497),
        BFieldElement::new(18428729670905102337),
        BFieldElement::new(18446744069414584257),
        BFieldElement::new(16140901060737761281),
        BFieldElement::new(18446708885042495489),
        BFieldElement::new(18446743931975630881),
        BFieldElement::new(18446744069412487169),
        BFieldElement::new(18446181119461294081),
        BFieldElement::new(18446744060824649729),
        BFieldElement::new(18302628881338728449),
        BFieldElement::new(18446744069414583809),
        BFieldElement::new(18410715272404008961),
        BFieldElement::new(18446743519658770433),
        BFieldElement::new(9223372032559808513),
        BFieldElement::new(18446744069414551553),
        BFieldElement::new(18446735273321564161),
        BFieldElement::new(18446744069280366593),
        BFieldElement::new(18444492269600899073),
        BFieldElement::new(18446744069414584313),
        BFieldElement::new(274873712576),
        BFieldElement::new(274882101184),
        BFieldElement::new(4611756386097823744),
        BFieldElement::new(13835128420805115905),
        BFieldElement::new(288230376151710720),
        BFieldElement::new(288230376151712768),
        BFieldElement::new(1125917086449664),
        BFieldElement::new(18445618186687873025),
        BFieldElement::new(4294901759),
        BFieldElement::new(4295032831),
        BFieldElement::new(72058693532778496),
        BFieldElement::new(18374687574905061377),
        BFieldElement::new(4503599627370480),
        BFieldElement::new(4503599627370512),
        BFieldElement::new(17592454475776),
        BFieldElement::new(18446726477496979457),
        BFieldElement::new(34359214072),
        BFieldElement::new(34360262648),
        BFieldElement::new(576469548262227968),
        BFieldElement::new(17870292113338400769),
        BFieldElement::new(36028797018963840),
        BFieldElement::new(36028797018964096),
        BFieldElement::new(140739635806208),
        BFieldElement::new(18446603334073745409),
        BFieldElement::new(2305843009213685760),
        BFieldElement::new(2305843009213702144),
        BFieldElement::new(9007336691597312),
        BFieldElement::new(18437737007600893953),
        BFieldElement::new(562949953421310),
        BFieldElement::new(562949953421314),
        BFieldElement::new(2199056809472),
        BFieldElement::new(18446741870424883713),
    ];
    const LOGN: usize = 6;

    let mut t = 1;
    let mut h = N / 2;
    for _ in 0..LOGN {
        let mut k = 0;
        for i in 0..h {
            let zeta = powers_of_psi_inv_bitreversed[h + i];
            for j in k..(k + t) {
                let u = array[j];
                let v = array[j + t];
                array[j] = u + v;
                array[j + t] = (u - v) * zeta;
            }

            k += 2 * t;
        }

        t *= 2;
        h >>= 1;
    }

    for a in array.iter_mut() {
        *a *= N_INV;
    }
}

pub fn coset_ntt_noswap_64(array: &mut [BFieldElement; 64]) {
    const N: usize = 64;

    let powers_of_psi_bitreversed = [
        BFieldElement::new(1),
        BFieldElement::new(281474976710656),
        BFieldElement::new(16777216),
        BFieldElement::new(1099511627520),
        BFieldElement::new(4096),
        BFieldElement::new(1152921504606846976),
        BFieldElement::new(68719476736),
        BFieldElement::new(4503599626321920),
        BFieldElement::new(64),
        BFieldElement::new(18014398509481984),
        BFieldElement::new(1073741824),
        BFieldElement::new(70368744161280),
        BFieldElement::new(262144),
        BFieldElement::new(17179869180),
        BFieldElement::new(4398046511104),
        BFieldElement::new(288230376084602880),
        BFieldElement::new(8),
        BFieldElement::new(2251799813685248),
        BFieldElement::new(134217728),
        BFieldElement::new(8796093020160),
        BFieldElement::new(32768),
        BFieldElement::new(9223372036854775808),
        BFieldElement::new(549755813888),
        BFieldElement::new(36028797010575360),
        BFieldElement::new(512),
        BFieldElement::new(144115188075855872),
        BFieldElement::new(8589934592),
        BFieldElement::new(562949953290240),
        BFieldElement::new(2097152),
        BFieldElement::new(137438953440),
        BFieldElement::new(35184372088832),
        BFieldElement::new(2305843008676823040),
        BFieldElement::new(2198989700608),
        BFieldElement::new(18446741870357774849),
        BFieldElement::new(18446181119461163007),
        BFieldElement::new(18446181119461163011),
        BFieldElement::new(9007061813690368),
        BFieldElement::new(18437736732722987009),
        BFieldElement::new(16140901060200882177),
        BFieldElement::new(16140901060200898561),
        BFieldElement::new(140735340838912),
        BFieldElement::new(18446603329778778113),
        BFieldElement::new(18410715272395620225),
        BFieldElement::new(18410715272395620481),
        BFieldElement::new(576451956076183552),
        BFieldElement::new(17870274521152356353),
        BFieldElement::new(18446744035054321673),
        BFieldElement::new(18446744035055370249),
        BFieldElement::new(17591917604864),
        BFieldElement::new(18446726476960108545),
        BFieldElement::new(18442240469787213809),
        BFieldElement::new(18442240469787213841),
        BFieldElement::new(72056494509522944),
        BFieldElement::new(18374685375881805825),
        BFieldElement::new(18446744065119551490),
        BFieldElement::new(18446744065119682562),
        BFieldElement::new(1125882726711296),
        BFieldElement::new(18445618152328134657),
        BFieldElement::new(18158513693262871553),
        BFieldElement::new(18158513693262873601),
        BFieldElement::new(4611615648609468416),
        BFieldElement::new(13834987683316760577),
        BFieldElement::new(18446743794532483137),
        BFieldElement::new(18446743794540871745),
    ];

    let mut m: usize = 1;
    let mut t: usize = N;
    while m < N {
        t >>= 1;

        for i in 0..m {
            let s = i * t * 2;
            let zeta = powers_of_psi_bitreversed[m + i];
            for j in s..(s + t) {
                let u = array[j];
                let v = array[j + t] * zeta;
                array[j] = u + v;
                array[j + t] = u - v;
            }
        }

        m *= 2;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CyclotomicRingElement {
    coefficients: [BFieldElement; 64],
}

impl CyclotomicRingElement {
    pub fn sample_short(randomness: &[u8]) -> CyclotomicRingElement {
        debug_assert!(randomness.len() >= 8 * 64);
        CyclotomicRingElement {
            coefficients: randomness
                .chunks(8)
                .into_iter()
                .map(|r| TryInto::<[u8; 8]>::try_into(r).unwrap())
                .map(|r| sample_short_bfield_element(&r))
                .collect_vec()
                .try_into()
                .unwrap(),
        }
    }

    pub fn sample_uniform(randomness: &[u8]) -> CyclotomicRingElement {
        debug_assert!(randomness.len() >= 9 * 64);
        let mut coefficients = [BFieldElement::zero(); 64];
        for i in 0..64 {
            let mut acc = 0u128;
            for j in 0..9 {
                acc = acc * 256 + randomness[i * 9 + j] as u128;
            }
            acc %= BFieldElement::P as u128;
            coefficients[i] = BFieldElement::new(acc as u64);
        }
        CyclotomicRingElement { coefficients }
    }

    pub fn hadamard(a: CyclotomicRingElement, b: CyclotomicRingElement) -> CyclotomicRingElement {
        let mut c = CyclotomicRingElement::zero();
        for i in 0..64 {
            c.coefficients[i] = a.coefficients[i] * b.coefficients[i];
        }
        c
    }
}

impl Add for CyclotomicRingElement {
    type Output = CyclotomicRingElement;

    fn add(self, rhs: Self) -> Self::Output {
        CyclotomicRingElement {
            coefficients: (0..64)
                .into_iter()
                .map(|i| self.coefficients[i] + rhs.coefficients[i])
                .collect_vec()
                .try_into()
                .unwrap(),
        }
    }
}

impl AddAssign for CyclotomicRingElement {
    fn add_assign(&mut self, rhs: Self) {
        self.coefficients
            .iter_mut()
            .zip(rhs.coefficients.iter())
            .for_each(|(l, r)| *l += *r);
    }
}

impl Sub for CyclotomicRingElement {
    type Output = CyclotomicRingElement;

    fn sub(self, rhs: Self) -> Self::Output {
        CyclotomicRingElement {
            coefficients: (0..64)
                .into_iter()
                .map(|i| self.coefficients[i] - rhs.coefficients[i])
                .collect_vec()
                .try_into()
                .unwrap(),
        }
    }
}

impl Mul for CyclotomicRingElement {
    type Output = CyclotomicRingElement;

    /// Multiply two polynomials in the ring
    /// Fp[X] / (X^64 + 1)
    /// using coset-NTT.
    fn mul(self, rhs: Self) -> Self::Output {
        let mut lhs_coeffs = self.coefficients;
        let mut rhs_coeffs = rhs.coefficients;
        coset_ntt_noswap_64(&mut lhs_coeffs);
        coset_ntt_noswap_64(&mut rhs_coeffs);
        let mut out_coeffs = [BFieldElement::zero(); 64];
        for i in 0..64 {
            out_coeffs[i] = lhs_coeffs[i] * rhs_coeffs[i];
        }
        coset_intt_noswap_64(&mut out_coeffs);
        CyclotomicRingElement {
            coefficients: out_coeffs,
        }
    }
}

impl Zero for CyclotomicRingElement {
    fn zero() -> Self {
        CyclotomicRingElement {
            coefficients: [BFieldElement::zero(); 64],
        }
    }

    fn is_zero(&self) -> bool {
        self.coefficients == [BFieldElement::zero(); 64]
    }
}

pub fn embed_msg(msg: [u8; 32]) -> CyclotomicRingElement {
    let mut embedding: [BFieldElement; 64] = [BFieldElement::zero(); 64];
    for i in 0..msg.len() {
        let mut integer = 0u64;
        for j in 0..4 {
            let bit = (msg[i] >> j) & 1;
            integer += (bit as u64) << (15 + 16 * j);
        }
        embedding[2 * i] = BFieldElement::new(integer);

        integer = 0;
        for j in 0..4 {
            let bit = (msg[i] >> (4 + j)) & 1;
            integer += (bit as u64) << (15 + 16 * j);
        }
        embedding[2 * i + 1] = BFieldElement::new(integer);
    }
    CyclotomicRingElement {
        coefficients: embedding,
    }
}

pub fn extract_msg(embedding: CyclotomicRingElement) -> [u8; 32] {
    let mut msg = [0u8; 32];
    for (ctr, pair) in embedding.coefficients.chunks(2).enumerate() {
        let mut byte = 0u8;
        let mut value = pair[0].value();
        for j in 0..4 {
            let chunk = value & 0xffff;
            value >>= 16;

            let bit = if chunk < (1 << 14) || (1 << 16) - chunk < (1 << 14) {
                0
            } else {
                1
            };
            byte |= bit << j;
        }

        value = pair[1].value();
        for j in 0..4 {
            let chunk = value & 0xffff;
            value >>= 16;

            let bit = if chunk < (1 << 14) || (1 << 16) - chunk < (1 << 14) {
                0
            } else {
                1
            };
            byte |= bit << (4 + j);
        }
        msg[ctr] = byte;
    }
    msg
}

const fn num_set_bits(a: u8) -> u8 {
    let mut sum = 0;
    let mut i = 0;
    while i < 8 {
        let bit = if a & (1 << i) != 0 { 1 } else { 0 };
        sum += bit;
        i += 1;
    }
    sum
}

const fn num_set_bits_table() -> [u8; 256] {
    let mut table: [u8; 256] = [0u8; 256];
    let mut i = 1;
    while i < 256 {
        table[i] = num_set_bits(i as u8);
        i += 1;
    }
    table
}

pub fn sample_short_bfield_element(randomness: &[u8; 8]) -> BFieldElement {
    const NUM_SET_BITS: [u8; 256] = num_set_bits_table();
    let left = ((NUM_SET_BITS[randomness[0] as usize] as u64) << (3 * 16))
        + ((NUM_SET_BITS[randomness[1] as usize] as u64) << (2 * 16))
        + ((NUM_SET_BITS[randomness[2] as usize] as u64) << 16)
        + (NUM_SET_BITS[randomness[3] as usize] as u64);
    let right = ((NUM_SET_BITS[randomness[4] as usize] as u64) << (3 * 16))
        + ((NUM_SET_BITS[randomness[5] as usize] as u64) << (2 * 16))
        + ((NUM_SET_BITS[randomness[6] as usize] as u64) << 16)
        + (NUM_SET_BITS[randomness[7] as usize] as u64);
    BFieldElement::new(left) - BFieldElement::new(right)
}

/// The Module is a matrix over the cyclotomic ring (i.e., the ring
/// of residue classes of polynomials modulo X^64+1). The matrix
/// contains N cyclotomic ring elements in total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleElement<const N: usize> {
    elements: [CyclotomicRingElement; N],
}

impl<const N: usize> ModuleElement<N> {
    pub fn sample_short(randomness: &[u8]) -> Self {
        debug_assert!(randomness.len() >= 8 * 64 * N);
        let mut elements = [CyclotomicRingElement::zero(); N];
        for n in 0..N {
            elements[n] =
                CyclotomicRingElement::sample_short(&randomness[8 * 64 * n..8 * 64 * (n + 1)]);
        }
        Self { elements }
    }

    pub fn sample_uniform(randomness: &[u8]) -> Self {
        debug_assert!(randomness.len() >= N * 9 * 64);
        ModuleElement {
            elements: (0..N)
                .into_iter()
                .map(|i| {
                    CyclotomicRingElement::sample_uniform(&randomness[i * 9 * 64..(i + 1) * 9 * 64])
                })
                .collect_vec()
                .try_into()
                .unwrap(),
        }
    }

    pub fn ntt(&self) -> Self {
        let mut copy = *self;
        for n in 0..N {
            coset_ntt_noswap_64(&mut copy.elements[n].coefficients);
        }
        copy
    }

    pub fn intt(&self) -> Self {
        let mut copy = *self;
        for n in 0..N {
            coset_intt_noswap_64(&mut copy.elements[n].coefficients);
        }
        copy
    }

    /// Multiply two module elements from a pair of matrix-
    /// multiplication-compatible modules. This method uses
    /// hadamard multiplication for cyclotomic ring elements, which
    /// is useful for avoiding the repeated conversion to and from
    /// NTT domain.
    ///  - `N` counts the total number of elements in the matrix;
    ///  - `H` counts the number of rows of the left hand side (and of
    ///  the output) matrix;
    ///  - `W` counts the number of columns of the right hand side (and
    ///  of the output) matrix;
    ///  - `INNER` counts the number of columns of the left hand side,
    ///  as well as the number of rows of the right hand side.
    pub fn multiply_hadamard<
        const LHS_H: usize,
        const LHS_N: usize,
        const RHS_W: usize,
        const RHS_N: usize,
        const INNER: usize,
        const OUT_N: usize,
    >(
        lhs: ModuleElement<LHS_N>,
        rhs: ModuleElement<RHS_N>,
    ) -> ModuleElement<OUT_N> {
        debug_assert_eq!(LHS_H * INNER, LHS_N);
        debug_assert_eq!(INNER * RHS_W, RHS_N);
        debug_assert_eq!(LHS_H * RHS_W, OUT_N);

        let mut elements = [CyclotomicRingElement::zero(); OUT_N];
        for h in 0..LHS_H {
            for w in 0..RHS_W {
                for i in 0..INNER {
                    elements[h * RHS_W + w] += CyclotomicRingElement::hadamard(
                        lhs.elements[h * INNER + i],
                        rhs.elements[i * RHS_W + w],
                    );
                }
            }
        }

        ModuleElement { elements }
    }

    /// Multiply two module elements from a pair of matrix-
    /// multiplication-compatible modules. This method uses the
    /// multiplication defined for cyclotomic ring elements
    /// abstractly. For a faster method that computes the entire
    /// matrix multiplication in the NTT domain, use `fast_multiply`.
    ///  - `N` counts the total number of elements in the matrix;
    ///  - `H` counts the number of rows of the left hand side (and of
    ///  the output) matrix;
    ///  - `W` counts the number of columns of the right hand side (and
    ///  of the output) matrix;
    ///  - `INNER` counts the number of columns of the left hand side,
    ///  as well as the number of rows of the right hand side.
    pub fn multiply<
        const LHS_H: usize,
        const LHS_N: usize,
        const RHS_W: usize,
        const RHS_N: usize,
        const INNER: usize,
        const OUT_N: usize,
    >(
        lhs: ModuleElement<LHS_N>,
        rhs: ModuleElement<RHS_N>,
    ) -> ModuleElement<OUT_N> {
        debug_assert_eq!(LHS_H * INNER, LHS_N);
        debug_assert_eq!(INNER * RHS_W, RHS_N);
        debug_assert_eq!(LHS_H * RHS_W, OUT_N);

        let mut out = ModuleElement {
            elements: [CyclotomicRingElement::zero(); OUT_N],
        };
        for h in 0..LHS_H {
            for w in 0..RHS_W {
                for i in 0..INNER {
                    out.elements[h * RHS_W + w] +=
                        lhs.elements[h * INNER + i] * rhs.elements[i * RHS_W + w];
                }
            }
        }

        out
    }

    /// Multiply two module elements from a pair of matrix-
    /// multiplication-compatible modules, by converting everything
    /// into the NTT domain, performing the matrix multiplication,
    /// and converting back.
    ///  - `N` counts the total number of elements in the matrix;
    ///  - `H` counts the number of rows of the left hand side (and of
    ///  the output) matrix;
    ///  - `W` counts the number of columns of the right hand side (and
    ///  of the output) matrix;
    ///  - `INNER` counts the number of columns of the left hand side,
    ///  as well as the number of rows of the right hand side.
    pub fn fast_multiply<
        const LHS_H: usize,
        const LHS_N: usize,
        const RHS_W: usize,
        const RHS_N: usize,
        const INNER: usize,
        const OUT_N: usize,
    >(
        lhs: ModuleElement<LHS_N>,
        rhs: ModuleElement<RHS_N>,
    ) -> ModuleElement<OUT_N> {
        debug_assert_eq!(LHS_H * INNER, LHS_N);
        debug_assert_eq!(INNER * RHS_W, RHS_N);
        debug_assert_eq!(LHS_H * RHS_W, OUT_N);

        let lhs_ntt = lhs.ntt();
        let rhs_ntt = rhs.ntt();

        let out_ntt =
            Self::multiply_hadamard::<LHS_H, LHS_N, RHS_W, RHS_N, INNER, OUT_N>(lhs_ntt, rhs_ntt);

        out_ntt.intt()
    }
}

impl<const N: usize> Add for ModuleElement<N> {
    type Output = ModuleElement<N>;

    fn add(self, rhs: Self) -> Self::Output {
        let elements: [CyclotomicRingElement; N] = (0..N)
            .into_par_iter()
            .map(|i| self.elements[i] + rhs.elements[i])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        ModuleElement::<N> { elements }
    }
}

impl<const N: usize> Sub for ModuleElement<N> {
    type Output = ModuleElement<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let elements: [CyclotomicRingElement; N] = (0..N)
            .into_par_iter()
            .map(|i| self.elements[i] - rhs.elements[i])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        ModuleElement::<N> { elements }
    }
}

pub mod kem {
    use rand::{thread_rng, RngCore};

    use super::{embed_msg, extract_msg, ModuleElement};
    use crate::shared_math::fips202::{sha3_256, shake256};

    #[derive(PartialEq, Eq, Clone)]
    pub struct SecretKey {
        key: [u8; 32],
        seed: [u8; 32],
    }

    #[derive(PartialEq, Eq, Clone)]
    pub struct PublicKey {
        seed: [u8; 32],
        ga: ModuleElement<4>,
    }

    #[derive(PartialEq, Eq, Clone)]
    pub struct Ciphertext {
        bg: ModuleElement<4>,
        bga_m: ModuleElement<1>,
    }

    fn derive_public_matrix(seed: &[u8; 32]) -> ModuleElement<16> {
        const NUM_BYTES: usize = 9 * 64 * 16;
        let randomness = shake256(seed, NUM_BYTES);
        ModuleElement::<16>::sample_uniform(&randomness)
    }

    fn derive_secret_vectors(seed: &[u8; 32]) -> (ModuleElement<4>, ModuleElement<4>) {
        const NUM_BYTES: usize = 2 * 4 * 64 * 8;
        let randomness = shake256(seed, NUM_BYTES);
        let a = ModuleElement::<4>::sample_short(&randomness[0..(NUM_BYTES / 2)]);
        let b = ModuleElement::<4>::sample_short(&randomness[(NUM_BYTES / 2)..]);
        (a, b)
    }

    /// Generate a public-secret key pair for key encapsulation.
    pub fn keygen() -> (SecretKey, PublicKey) {
        let mut rng = thread_rng();
        let mut seed: [u8; 32] = [0u8; 32];
        rng.fill_bytes(&mut seed);

        let mut key: [u8; 32] = [0u8; 32];
        rng.fill_bytes(&mut key);
        let sk = SecretKey { key, seed };

        let pk = derive_public_key(&key, &seed);
        (sk, pk)
    }

    fn derive_public_key(key: &[u8; 32], seed: &[u8; 32]) -> PublicKey {
        let (a, c) = derive_secret_vectors(key);
        let g = derive_public_matrix(seed);
        let ga = ModuleElement::<16>::multiply_hadamard::<4, 16, 1, 4, 4, 4>(g, a.ntt()) + c.ntt();

        PublicKey { seed: *seed, ga }
    }

    /// Generate a ciphertext with the given seed (`payload`) from
    /// which to derive all randomness.
    fn generate_ciphertext_derandomized(pk: PublicKey, payload: [u8; 32]) -> Ciphertext {
        let (b, d) = derive_secret_vectors(&payload);
        let b_ntt = b.ntt();
        let d_ntt = d.ntt();
        let g = derive_public_matrix(&pk.seed);
        let bg = ModuleElement::<9>::multiply_hadamard::<1, 4, 4, 16, 4, 4>(b_ntt, g) + d_ntt;

        let m = embed_msg(payload);
        let bga_m = ModuleElement::<3>::multiply_hadamard::<1, 4, 1, 4, 4, 1>(b_ntt, pk.ga)
            + ModuleElement::<1> { elements: [m] }.ntt();

        Ciphertext { bg, bga_m }
    }

    /// Encapsulate: generate a ciphertext and an associated shared
    /// symmetric key.
    pub fn enc(pk: PublicKey) -> ([u8; 32], Ciphertext) {
        let mut rng = thread_rng();
        let mut payload = [0u8; 32];
        rng.fill_bytes(&mut payload);
        let ciphertext = generate_ciphertext_derandomized(pk, payload);
        let shared_key: [u8; 32] = sha3_256(&payload);

        (shared_key, ciphertext)
    }

    /// Decapsulate: use the secret key to extract the corresponding
    /// shared symmetric key from a ciphertext (if successful).
    pub fn dec(sk: SecretKey, ctxt: Ciphertext) -> Option<[u8; 32]> {
        let (a, _) = derive_secret_vectors(&sk.key);
        let bga = ModuleElement::<3>::multiply_hadamard::<1, 4, 1, 4, 4, 1>(ctxt.bg, a.ntt());
        let m = (ctxt.bga_m - bga).intt();
        let payload = extract_msg(m.elements[0]);

        let pk = derive_public_key(&sk.key, &sk.seed);
        let regenerated_ciphertext = generate_ciphertext_derandomized(pk, payload);

        if regenerated_ciphertext != ctxt {
            return None;
        }

        let shared_key = sha3_256(&payload);
        Some(shared_key)
    }
}

#[cfg(test)]
mod lattice_test {
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::{thread_rng, RngCore};

    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::lattice::*;
    use crate::shared_math::other::random_elements_array;

    #[test]
    fn test_fast_mul() {
        let a: [BFieldElement; 64] = random_elements_array();
        let b: [BFieldElement; 64] = random_elements_array();

        let mut c_schoolbook = [BFieldElement::zero(); 64];
        for i in 0..64 {
            for j in 0..64 {
                if i + j >= 64 {
                    c_schoolbook[i + j - 64] -= a[i] * b[j];
                } else {
                    c_schoolbook[i + j] += a[i] * b[j];
                }
            }
        }

        let c_fast = (CyclotomicRingElement { coefficients: a }
            * CyclotomicRingElement { coefficients: b })
        .coefficients;

        assert_eq!(c_fast, c_schoolbook);
    }

    #[test]
    fn test_embedding() {
        let mut rng = thread_rng();
        let msg: [u8; 32] = (0..32)
            .into_iter()
            .map(|_| (rng.next_u32() % 256) as u8)
            .collect_vec()
            .try_into()
            .unwrap();
        let embedding = embed_msg(msg);
        let extracted = extract_msg(embedding);

        assert_eq!(msg, extracted);
    }

    #[test]
    fn test_module_distributivity() {
        let mut rng = thread_rng();
        let randomness = (0..(2 * 3 + 2 * 3 + 3) * 64 * 9)
            .into_iter()
            .map(|_| (rng.next_u32() % 256) as u8)
            .collect_vec();
        let mut start = 0;
        let mut stop = 2 * 3 * 9 * 64;
        let a = ModuleElement::<{ 2 * 3 }>::sample_uniform(&randomness[start..stop]);
        start = stop;
        stop += 2 * 3 * 9 * 64;
        let b = ModuleElement::<{ 2 * 3 }>::sample_uniform(&randomness[start..stop]);
        start = stop;
        stop += 3 * 9 * 64;
        let c = ModuleElement::<3>::sample_uniform(&randomness[start..stop]);

        let sumprod = ModuleElement::<1>::multiply::<2, 6, 1, 3, 3, 2>(a + b, c);
        let prodsum = ModuleElement::<1>::multiply::<2, 6, 1, 3, 3, 2>(a, c)
            + ModuleElement::<1>::multiply::<2, 6, 1, 3, 3, 2>(b, c);

        assert_eq!(sumprod, prodsum);
    }

    #[test]
    fn test_module_multiply() {
        let mut rng = thread_rng();
        let randomness = (0..(2 * 3 + 2 * 3 + 3) * 64 * 9)
            .into_iter()
            .map(|_| (rng.next_u32() % 256) as u8)
            .collect_vec();
        let mut start = 0;
        let mut stop = 2 * 3 * 9 * 64;
        let a = ModuleElement::<{ 2 * 3 }>::sample_uniform(&randomness[start..stop]);
        start = stop;
        stop += 3 * 2 * 9 * 64;
        let b = ModuleElement::<{ 2 * 3 }>::sample_uniform(&randomness[start..stop]);

        assert_eq!(
            ModuleElement::<1>::fast_multiply::<2, 6, 2, 6, 3, 4>(a, b),
            ModuleElement::<1>::multiply::<2, 6, 2, 6, 3, 4>(a, b)
        );
    }

    #[test]
    fn test_kem() {
        // correctness
        let (sk, pk) = kem::keygen();
        let (alice_key, ctxt) = kem::enc(pk);
        if let Some(bob_key) = kem::dec(sk, ctxt.clone()) {
            assert_eq!(alice_key, bob_key);
        } else {
            panic!()
        }

        // sanity
        let (other_sk, _) = kem::keygen();
        assert!(kem::dec(other_sk, ctxt).is_none());
    }
}