use itertools::Itertools;
use num_traits::Zero;

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

pub fn had64(a: &[BFieldElement; 64], b: &[BFieldElement; 64]) -> [BFieldElement; 64] {
    let mut c = [BFieldElement::zero(); 64];
    for i in 0..64 {
        c[i] = a[i] * b[i];
    }
    c
}

pub fn add64(a: &[BFieldElement; 64], b: &[BFieldElement; 64]) -> [BFieldElement; 64] {
    let mut c = [BFieldElement::zero(); 64];
    for i in 0..64 {
        c[i] = a[i] + b[i];
    }
    c
}

/// Multiply two polynomials in the ring
/// Fp[X] / (X^64 + 1)
/// using coset-NTT.
pub fn cycloring64_mul(a: &[BFieldElement; 64], b: &[BFieldElement; 64]) -> [BFieldElement; 64] {
    let mut a_copy = *a;
    let mut b_copy = *b;

    coset_ntt_noswap_64(&mut a_copy);
    coset_ntt_noswap_64(&mut b_copy);

    let mut c = had64(&a_copy, &b_copy);

    coset_intt_noswap_64(&mut c);

    c
}

pub fn embed_msg(msg: [u8; 32]) -> [BFieldElement; 64] {
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
    embedding
}

pub fn extract_msg(embedding: [BFieldElement; 64]) -> [u8; 32] {
    let mut msg = [0u8; 32];
    for (ctr, pair) in embedding.chunks(2).enumerate() {
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

fn sample_short_bfield_element(randomness: &[u8; 8]) -> BFieldElement {
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

fn sample_short_cycloring_element(randomness: &[u8; 8 * 64]) -> [BFieldElement; 64] {
    randomness
        .chunks(8)
        .into_iter()
        .map(|r| TryInto::<[u8; 8]>::try_into(r).unwrap())
        .map(|r| sample_short_bfield_element(&r))
        .collect_vec()
        .try_into()
        .unwrap()
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

        let c_fast = cycloring64_mul(&a, &b);

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
}
