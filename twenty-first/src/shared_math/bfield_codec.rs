use std::marker::PhantomData;

use anyhow::bail;
use anyhow::Result;
use bfieldcodec_derive::BFieldCodec;
use itertools::Itertools;
use num_traits::One;
use num_traits::Zero;

use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::merkle_tree::PartialAuthenticationPath;

use super::b_field_element::BFieldElement;
use super::tip5::Digest;
use super::tip5::Tip5;
use super::tip5::DIGEST_LENGTH;
use super::x_field_element::XFieldElement;
use super::x_field_element::EXTENSION_DEGREE;

/// BFieldCodec
///
/// This trait provides functions for encoding to and decoding from a
/// Vec of BFieldElements. This encoding records the length of
/// variable-size structures, whether implicitly or explicitly via
/// length-prepending. It does not record type information; this is
/// the responsibility of the decoder.
pub trait BFieldCodec {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>>;
    fn encode(&self) -> Vec<BFieldElement>;
}

impl BFieldCodec for BFieldElement {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 1 {
            bail!("trying to decode more or less than one BFieldElements as one BFieldElement");
        }
        let element_zero = sequence[0];
        Ok(Box::new(element_zero))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        [*self].to_vec()
    }
}

impl BFieldCodec for XFieldElement {
    // FIXME: Use `XFieldElement::try_into()`.
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != EXTENSION_DEGREE {
            bail!(
                "trying to decode slice of not EXTENSION_DEGREE BFieldElements into XFieldElement"
            );
        }

        Ok(Box::new(XFieldElement {
            coefficients: sequence.try_into().unwrap(),
        }))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.coefficients.to_vec()
    }
}

impl BFieldCodec for Digest {
    // FIXME: Use `Digest::try_from()`
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != DIGEST_LENGTH {
            bail!("trying to decode slice of not DIGEST_LENGTH BFieldElements into Digest");
        }

        Ok(Box::new(Digest::new(sequence.try_into().unwrap())))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.values().to_vec()
    }
}

impl<T: BFieldCodec, S: BFieldCodec> BFieldCodec for (T, S) {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        // decode T
        let maybe_element_zero = str.get(0);
        if matches!(maybe_element_zero, None) {
            bail!("trying to decode empty slice as tuple",);
        }

        let len_t = maybe_element_zero.unwrap().value() as usize;
        if str.len() < 1 + len_t {
            bail!("prepended length of tuple element does not match with remaining string length");
        }
        let maybe_t = T::decode(&str[1..(1 + len_t)]);

        // decode S
        let maybe_next_element = str.get(1 + len_t);
        if matches!(maybe_next_element, None) {
            bail!("trying to decode singleton as tuple");
        }

        let len_s = maybe_next_element.unwrap().value() as usize;
        if str.len() != 1 + len_t + 1 + len_s {
            bail!(
                "prepended length of second tuple element does not match with remaining string length",
            );
        }
        let maybe_s = S::decode(&str[(1 + len_t + 1)..]);

        if let Ok(t) = maybe_t {
            if let Ok(s) = maybe_s {
                Ok(Box::new((*t, *s)))
            } else {
                bail!("could not decode s")
            }
        } else {
            bail!("could not decode t")
        }
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        let mut encoding_of_t = self.0.encode();
        let mut encoding_of_s = self.1.encode();
        str.push(BFieldElement::new(encoding_of_t.len().try_into().expect(
            "encoding of t has length that does not fit in BFieldElement",
        )));
        str.append(&mut encoding_of_t);
        str.push(BFieldElement::new(encoding_of_s.len().try_into().expect(
            "encoding of s has length that does not fit in BFieldElement",
        )));
        str.append(&mut encoding_of_s);
        str
    }
}

impl BFieldCodec for PartialAuthenticationPath<Digest> {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.is_empty() {
            bail!("cannot decode empty string into PartialAuthenticationPath");
        }
        let mut vect: Vec<Option<Digest>> = vec![];
        let mut index = 0;
        while index < sequence.len() {
            let len = sequence[index].value();
            if sequence.len() < index + 1 + len as usize {
                bail!(
                    "cannot decode vec of optional digests because of improper length prepending"
                );
            }
            let substr = &sequence[(index + 1)..(index + 1 + len as usize)];
            let decoded = Option::<Digest>::decode(substr);
            if let Ok(optional_digest) = decoded {
                vect.push(*optional_digest);
            } else {
                bail!("cannot decode optional digest in vec");
            }

            index += 1 + len as usize;
        }
        Ok(Box::new(PartialAuthenticationPath::<Digest>(vect)))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut vect = vec![];
        for optional_authpath in self.0.iter() {
            let mut encoded = optional_authpath.encode();
            vect.push(BFieldElement::new(encoded.len().try_into().expect(
                "encoded optional authpath has length greater than what fits into BFieldElement",
            )));
            vect.append(&mut encoded);
        }
        vect
    }
}

impl<T: BFieldCodec> BFieldCodec for Option<T> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let maybe_element_zero = str.get(0);
        if matches!(maybe_element_zero, None) {
            bail!("trying to decode empty slice into option of elements");
        }

        if maybe_element_zero.unwrap().is_zero() {
            Ok(Box::new(None))
        } else {
            let maybe_t = T::decode(&str[1..]);
            match maybe_t {
                Ok(t) => Ok(Box::new(Some(*t))),
                Err(e) => Err(e),
            }
        }
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        match self {
            None => {
                str.push(BFieldElement::zero());
            }
            Some(t) => {
                str.push(BFieldElement::one());
                str.append(&mut t.encode());
            }
        }
        str
    }
}

impl BFieldCodec for Vec<BFieldElement> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        Ok(Box::new(str.to_vec()))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.to_vec()
    }
}

impl BFieldCodec for Vec<XFieldElement> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        if str.len() % EXTENSION_DEGREE != 0 {
            bail!(
                "cannot decode string of BFieldElements into XFieldElements \
                when string length is not a multiple of EXTENSION_DEGREE",
            );
        }
        let mut vector = vec![];
        for chunk in str.chunks(EXTENSION_DEGREE) {
            let coefficients: [BFieldElement; EXTENSION_DEGREE] = chunk.try_into().unwrap();
            vector.push(XFieldElement::new(coefficients));
        }
        Ok(Box::new(vector))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.iter().map(|xfe| xfe.coefficients.to_vec()).concat()
    }
}

impl BFieldCodec for Vec<Digest> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        if str.len() % DIGEST_LENGTH != 0 {
            bail!(
                "cannot decode string of BFieldElements into Digests \
                when string length is not a multiple of DIGEST_LENGTH",
            );
        }
        let mut vector: Vec<Digest> = vec![];
        for chunk in str.chunks(DIGEST_LENGTH) {
            let digest: [BFieldElement; DIGEST_LENGTH] = chunk.try_into().unwrap();
            vector.push(Digest::new(digest));
        }
        Ok(Box::new(vector))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.iter().map(|d| d.encode()).concat()
    }
}

impl<T> BFieldCodec for Vec<Vec<T>>
where
    Vec<T>: BFieldCodec,
{
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let mut index = 0;
        let mut outer_vec: Vec<Vec<T>> = vec![];
        while index < str.len() {
            let len = str[index].value() as usize;
            index += 1;

            if let Some(inner_vec) = str.get(index..(index + len)) {
                outer_vec.push(*Vec::<T>::decode(inner_vec)?);
            } else {
                bail!("cannot decode string BFieldElements into Vec<Vec<T>>; length mismatch");
            }
            index += len;
        }
        Ok(Box::new(outer_vec))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        for inner_vec in self {
            let mut encoding = inner_vec.encode();
            str.push(BFieldElement::new(encoding.len().try_into().unwrap()));
            str.append(&mut encoding);
        }
        str
    }
}

impl BFieldCodec for Vec<PartialAuthenticationPath<Digest>> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let mut index = 0;
        let mut vector = vec![];

        // while there is at least one partial auth path left, parse it
        while index < str.len() {
            let len_remaining = str[index].value() as usize;
            index += 1;

            if len_remaining < 2 || index + len_remaining > str.len() {
                bail!(
                    "cannot decode string of BFieldElements as Vec of PartialAuthenticationPaths \
                    due to length mismatch (1)",
                );
            }

            let vec_len = str[index].value() as usize;
            let mask = str[index + 1].value() as u32;
            index += 2;

            // if the vector length and mask indicates some digests are following
            // and we are already at the end of the buffer
            if vec_len != 0 && mask != 0 && index == str.len() {
                bail!(
                    "Cannot decode string of BFieldElements as Vec of PartialAuthenticationPaths \
                    due to length mismatch (2).\n\
                    vec_len: {}\n\
                    mask: {}\n\
                    index: {}\n\
                    str.len(): {}\n\
                    str[0]: {}",
                    vec_len,
                    mask,
                    index,
                    str.len(),
                    str[0]
                );
            }

            if (len_remaining - 2) % DIGEST_LENGTH != 0 {
                bail!(
                    "cannot decode string of BFieldElements as Vec of PartialAuthenticationPaths \
                    due to length mismatch (3)",
                );
            }

            let mut pap = vec![];

            for i in (0..vec_len).rev() {
                if mask & (1 << i) == 0 {
                    pap.push(None);
                } else if let Some(chunk) = str.get(index..(index + DIGEST_LENGTH)) {
                    pap.push(Some(*Digest::decode(chunk)?));
                    index += DIGEST_LENGTH;
                } else {
                    bail!(
                        "cannot decode string of BFieldElements as Vec of \
                        PartialAuthenticationPaths due to length mismatch (4)",
                    );
                }
            }

            vector.push(PartialAuthenticationPath(pap));
        }

        Ok(Box::new(vector))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        for pap in self.iter() {
            let len = pap.len();
            let mut mask = 0u32;
            for maybe_digest in pap.iter() {
                mask <<= 1;
                if maybe_digest.is_some() {
                    mask |= 1;
                }
            }
            let mut vector = pap.iter().flatten().map(|d| d.encode()).concat();

            str.push(BFieldElement::new(
                2u64 + std::convert::TryInto::<u64>::try_into(vector.len()).unwrap(),
            ));
            str.push(BFieldElement::new(len.try_into().unwrap()));
            str.push(BFieldElement::new(mask.try_into().unwrap()));
            str.append(&mut vector);
        }
        str
    }
}

/// Decode a string of `BFieldElement`s into a `Vec` for `T`s. This
/// function exists because it is not a good idea to implement
/// `BFieldCodec` for `Vec<T>`.
pub fn decode_vec<T: BFieldCodec>(sequence: &[BFieldElement]) -> Result<Box<Vec<T>>> {
    let total_length = match sequence.get(0) {
        Some(result) => result.value() as usize,
        None => bail!("Cannot decode empty Vec of BFieldElements."),
    };
    if sequence.len() < total_length + 1 {
        bail!("Cannot decode Vec of BFieldElements because of improper length prepending.");
    }

    let mut vector: Vec<T> = vec![];
    let mut read_index = 1;
    while read_index < sequence.len() {
        let inner_len = match sequence.get(read_index) {
            Some(result) => result.value() as usize,
            None => bail!(
                "Cannot decode Vec of BFieldElements because element is not length-prepended."
            ),
        };
        read_index += 1;

        if sequence.len() < read_index + inner_len {
            bail!("Cannot decode Vec of BFieldElements because of improper element length prepending.");
        }

        let inner_sequence = &sequence[read_index..read_index + inner_len];
        read_index += inner_len;
        vector.push(*T::decode(inner_sequence)?);
    }

    Ok(Box::new(vector))
}

/// Encode a `Vec` of `T`s into a `Vec` of `BFieldElement`s in such a
/// way that the matching `decode_vec` function recovers the original
/// This function exists because it is not a good idea to implement
/// `BFieldCodec` for `Vec<T>`.
///
/// This function should not be used when `Vec<T>` already implements
/// `BFieldCodec`. In that case, just use `.encode()` instead.
///
/// Todo: investigate whether a smarter encoding makes sense
pub fn encode_vec<T: BFieldCodec>(vector: &[T]) -> Vec<BFieldElement> {
    let mut sequence: Vec<BFieldElement> = vec![BFieldElement::zero()];
    for v in vector.iter() {
        let mut element = v.encode();
        sequence.push(BFieldElement::new(element.len() as u64));
        sequence.append(&mut element);
    }
    sequence[0] = BFieldElement::new(sequence.len() as u64 - 1);
    sequence
}

impl<T> BFieldCodec for PhantomData<T> {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if !sequence.is_empty() {
            bail!("Cannot decode non-empty BFE slice as phantom data")
        }
        Ok(Box::new(PhantomData))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        vec![]
    }
}

impl BFieldCodec for u128 {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 4 {
            bail!(
                "Cannot decode sequence of length {} =/= 4 as u128.",
                sequence.len()
            );
        }
        if !sequence.iter().all(|s| s.value() <= u32::MAX as u64) {
            bail!(
                "Could not parse sequence of BFieldElements {:?} as u128.",
                sequence
            );
        }
        return Ok(Box::new(
            sequence
                .iter()
                .enumerate()
                .map(|(i, s)| (s.value() as u128) << (i * 32))
                .sum(),
        ));
    }

    fn encode(&self) -> Vec<BFieldElement> {
        (0..4)
            .map(|i| (*self >> (i * 32)) as u64 & u32::MAX as u64)
            .map(BFieldElement::new)
            .collect_vec()
    }
}

impl BFieldCodec for u64 {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 2 {
            bail!(
                "Cannot decode sequence of length {} =/= 2 as u64.",
                sequence.len()
            );
        }
        if !sequence.iter().all(|s| s.value() <= u32::MAX as u64) {
            bail!(
                "Could not parse sequence of BFieldElements {:?} as u64.",
                sequence
            );
        }
        return Ok(Box::new(
            sequence
                .iter()
                .enumerate()
                .map(|(i, s)| s.value() << (i * 32))
                .sum(),
        ));
    }

    fn encode(&self) -> Vec<BFieldElement> {
        (0..2)
            .map(|i| (*self >> (i * 32)) & u32::MAX as u64)
            .map(BFieldElement::new)
            .collect_vec()
    }
}

impl BFieldCodec for bool {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 1 {
            bail!(
                "Cannot decode sequence of {} =/= 1 BFieldElements as bool.",
                sequence.len()
            );
        }
        Ok(Box::new(match sequence[0].value() {
            0 => false,
            1 => true,
            n => bail!("Failed to parse BFieldElement {n} as bool."),
        }))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }
}

impl BFieldCodec for u32 {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 1 {
            bail!(
                "Cannot decode sequence of {} =/= 1 BFieldElements as bool.",
                sequence.len()
            );
        }
        let value = sequence[0].value();
        if value > u32::MAX as u64 {
            bail!("Cannot decode BFieldElement {value} as u32.");
        }
        Ok(Box::new(value as u32))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }
}

/// Decode a single field of a struct, assuming its length is prepended.
/// Return the remaining sequence.
pub fn decode_field_length_prepended<T: BFieldCodec>(
    sequence: &[BFieldElement],
) -> Result<(T, Vec<BFieldElement>)> {
    if sequence.is_empty() {
        bail!("Cannot decode field: sequence is empty.");
    }
    let len = sequence[0].value() as usize;
    if sequence.len() < 1 + len {
        bail!("Cannot decode field: sequence too short.");
    }
    let decoded = *T::decode(&sequence[1..1 + len])?;
    Ok((decoded, sequence[1 + len..].to_vec()))
}

/// Decode a vector of some struct, assuming the length of the encoding is prepended. Return the remaining sequence.
pub fn decode_vec_length_prepended<T: BFieldCodec>(
    sequence: &[BFieldElement],
) -> Result<(Vec<T>, Vec<BFieldElement>)> {
    if sequence.is_empty() {
        bail!("Cannot decode vec: sequence is empty.");
    }
    let len = sequence[0].value() as usize;
    if sequence.len() < 1 + len {
        bail!("Cannot decode vec: sequence too short.");
    }
    let decoded: Vec<T> = *decode_vec(&sequence[1..1 + len])?;
    Ok((decoded, sequence[1 + len..].to_vec()))
}

#[cfg(test)]
mod bfield_codec_tests {
    use itertools::Itertools;
    use rand::thread_rng;
    use rand::Rng;
    use rand::RngCore;

    use crate::amount::u32s::U32s;
    use crate::shared_math::tip5::Tip5;

    use super::*;

    fn random_bool() -> bool {
        let mut rng = thread_rng();
        rng.next_u32() % 2 == 0
    }

    fn random_length(max: usize) -> usize {
        let mut rng = thread_rng();
        rng.next_u32() as usize % max
    }

    fn random_bfieldelement() -> BFieldElement {
        let mut rng = thread_rng();
        BFieldElement::new(rng.next_u64())
    }

    fn random_xfieldelement() -> XFieldElement {
        XFieldElement {
            coefficients: [
                random_bfieldelement(),
                random_bfieldelement(),
                random_bfieldelement(),
            ],
        }
    }

    fn random_digest() -> Digest {
        Digest::new([
            random_bfieldelement(),
            random_bfieldelement(),
            random_bfieldelement(),
            random_bfieldelement(),
            random_bfieldelement(),
        ])
    }

    fn random_partial_authentication_path(len: usize) -> PartialAuthenticationPath<Digest> {
        PartialAuthenticationPath(
            (0..len)
                .map(|_| {
                    if random_bool() {
                        Some(random_digest())
                    } else {
                        None
                    }
                })
                .collect_vec(),
        )
    }

    #[test]
    fn test_encode_decode_random_bfieldelement() {
        for _ in 1..=10 {
            let bfe = random_bfieldelement();
            let str = bfe.encode();
            let bfe_ = *BFieldElement::decode(&str).unwrap();
            assert_eq!(bfe, bfe_);
        }
    }

    #[test]
    fn test_encode_decode_random_xfieldelement() {
        for _ in 1..=10 {
            let xfe = random_xfieldelement();
            let str = xfe.encode();
            let xfe_ = *XFieldElement::decode(&str).unwrap();
            assert_eq!(xfe, xfe_);
        }
    }

    #[test]
    fn test_encode_decode_random_digest() {
        for _ in 1..=10 {
            let dig = random_digest();
            let str = dig.encode();
            let dig_ = *Digest::decode(&str).unwrap();
            assert_eq!(dig, dig_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_bfieldelement() {
        for _ in 1..=10 {
            let len = random_length(100);
            let bfe_vec = (0..len).map(|_| random_bfieldelement()).collect_vec();
            let str = bfe_vec.encode();
            let bfe_vec_ = *Vec::<BFieldElement>::decode(&str).unwrap();
            assert_eq!(bfe_vec, bfe_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_xfieldelement() {
        for _ in 1..=10 {
            let len = random_length(100);
            let xfe_vec = (0..len).map(|_| random_xfieldelement()).collect_vec();
            let str = xfe_vec.encode();
            let xfe_vec_ = *Vec::<XFieldElement>::decode(&str).unwrap();
            assert_eq!(xfe_vec, xfe_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_digest() {
        for _ in 1..=10 {
            let len = random_length(100);
            let digest_vec = (0..len).map(|_| random_digest()).collect_vec();
            let str = digest_vec.encode();
            let digest_vec_ = *Vec::<Digest>::decode(&str).unwrap();
            assert_eq!(digest_vec, digest_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_vec_of_bfieldelement() {
        for _ in 1..=10 {
            let len = random_length(10);
            let bfe_vec_vec = (0..len)
                .map(|_| {
                    let inner_len = random_length(20);
                    (0..inner_len).map(|_| random_bfieldelement()).collect_vec()
                })
                .collect_vec();
            let str = bfe_vec_vec.encode();
            let bfe_vec_vec_ = *Vec::<Vec<BFieldElement>>::decode(&str).unwrap();
            assert_eq!(bfe_vec_vec, bfe_vec_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_vec_of_xfieldelement() {
        for _ in 1..=10 {
            let len = random_length(10);
            let xfe_vec_vec = (0..len)
                .map(|_| {
                    let inner_len = random_length(20);
                    (0..inner_len).map(|_| random_xfieldelement()).collect_vec()
                })
                .collect_vec();
            let str = xfe_vec_vec.encode();
            let xfe_vec_vec_ = *Vec::<Vec<XFieldElement>>::decode(&str).unwrap();
            assert_eq!(xfe_vec_vec, xfe_vec_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_partial_authentication_path() {
        for _ in 1..=10 {
            let len = 1 + random_length(10);
            let pap = random_partial_authentication_path(len);
            let str = pap.encode();
            let pap_ = *PartialAuthenticationPath::decode(&str).unwrap();
            assert_eq!(pap, pap_);
        }
    }

    #[test]
    fn test_decode_random_negative() {
        for _ in 1..=10000 {
            let len = random_length(100);
            let str = (0..len).map(|_| random_bfieldelement()).collect_vec();

            // Some of the following cases can be triggered by false
            // positives. This should occur with probability roughly
            // 2^-60.

            if let Ok(_sth) = BFieldElement::decode(&str) {
                if str.len() != 1 {
                    panic!();
                }
            }

            if let Ok(_sth) = XFieldElement::decode(&str) {
                if str.len() != EXTENSION_DEGREE {
                    panic!();
                }
            }

            if let Ok(_sth) = Digest::decode(&str) {
                if str.len() != DIGEST_LENGTH {
                    panic!();
                }
            }

            // if let Ok(sth) = Vec::<BFieldElement>::decode(&str) {
            //     (should work)
            // }

            if str.len() % EXTENSION_DEGREE != 0 {
                if let Ok(sth) = Vec::<XFieldElement>::decode(&str) {
                    panic!("{sth:?}");
                }
            }

            if str.len() % DIGEST_LENGTH != 0 {
                if let Ok(sth) = Vec::<Digest>::decode(&str) {
                    panic!("{sth:?}");
                }
            }

            if let Ok(sth) = Vec::<Vec<BFieldElement>>::decode(&str) {
                if !sth.is_empty() {
                    panic!("{sth:?}");
                }
            }

            if let Ok(sth) = Vec::<Vec<XFieldElement>>::decode(&str) {
                if !sth.is_empty() {
                    panic!("{sth:?}");
                }
            }

            if let Ok(_sth) = PartialAuthenticationPath::decode(&str) {
                panic!();
            }
        }
    }

    #[test]
    fn test_encode_decode_random_vec_option_xfieldelement() {
        let mut rng = thread_rng();
        let n = 10 + (rng.next_u32() % 50);
        let mut vector: Vec<Option<XFieldElement>> = vec![];
        for _ in 0..n {
            if rng.next_u32() % 2 == 0 {
                vector.push(None);
            } else {
                vector.push(Some(rng.gen()));
            }
        }

        let encoded = encode_vec(&vector);
        let decoded = *decode_vec(&encoded).unwrap();

        assert_eq!(vector, decoded);
    }

    #[test]
    fn test_decode_u128() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let num = (rng.next_u64() as u128) << 64 | rng.next_u64() as u128;
            let encoded = num.encode();
            let decoded = *u128::decode(&encoded).unwrap();
            assert_eq!(num, decoded);

            // Verify same encoding as U32<4>
            let u32_4: U32s<4> = num.try_into().unwrap();
            let u32_4_encoded = u32_4.encode();
            assert_eq!(encoded, u32_4_encoded);
        }

        for v in [
            0u64,
            1u64,
            u64::MAX - 1,
            u64::MAX,
            BFieldElement::MAX,
            BFieldElement::P,
        ] {
            let encoded = v.encode();
            assert_eq!(*u64::decode(&encoded).unwrap(), v);

            // Verify same encoding as U32<2>
            let u32_2: U32s<2> = v.try_into().unwrap();
            let u32_2_encoded = u32_2.encode();
            assert_eq!(encoded, u32_2_encoded);
        }
    }

    #[test]
    fn test_decode_u64() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let num = rng.next_u64();
            let encoded = num.encode();
            let decoded = *u64::decode(&encoded).unwrap();
            assert_eq!(num, decoded);

            // Verify same encoding as U32<2>
            let u32_2: U32s<2> = num.try_into().unwrap();
            let u32_2_encoded = u32_2.encode();
            assert_eq!(encoded, u32_2_encoded);
        }

        for v in [
            0u64,
            1u64,
            u64::MAX - 1,
            u64::MAX,
            BFieldElement::MAX,
            BFieldElement::P,
        ] {
            let encoded = v.encode();
            assert_eq!(*u64::decode(&encoded).unwrap(), v);

            // Verify same encoding as U32<2>
            let u32_2: U32s<2> = v.try_into().unwrap();
            let u32_2_encoded = u32_2.encode();
            assert_eq!(encoded, u32_2_encoded);
        }
    }

    #[test]
    fn test_decode_u32() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let num = rng.next_u32();
            let encoded = num.encode();
            let decoded = *u32::decode(&encoded).unwrap();
            assert_eq!(num, decoded);

            // Verify same encoding as U32<1>
            let u32_1: U32s<1> = num.try_into().unwrap();
            let u32_1_encoded = u32_1.encode();
            assert_eq!(encoded, u32_1_encoded);
        }

        for v in [0u32, 1u32, u32::MAX - 1, u32::MAX] {
            assert_eq!(*u32::decode(&v.encode()).unwrap(), v);
        }
    }

    #[test]
    fn test_decode_bool() {
        let trew = true;
        let true_encoded = trew.encode();
        let true_decoded = *bool::decode(&true_encoded).unwrap();
        assert_eq!(true_decoded, trew);

        let fallse = false;
        let false_encoded = fallse.encode();
        let false_decoded = *bool::decode(&false_encoded).unwrap();
        assert_eq!(false_decoded, fallse);
    }

    #[test]
    fn test_phantom_data() {
        let pd = PhantomData::<Tip5>;
        let encoded = pd.encode();
        let decoded = *PhantomData::decode(&encoded).unwrap();
        assert_eq!(decoded, pd);

        let mut sequence = vec![BFieldElement::new(encoded.len() as u64)];
        sequence.append(&mut pd.encode());

        let (field_value, sequence) =
            decode_field_length_prepended::<PhantomData<Tip5>>(&sequence).unwrap();
        assert_eq!(pd, field_value);
        assert!(sequence.is_empty());
    }
}

#[derive(BFieldCodec, PartialEq, Eq, Debug)]
struct DeriveTestStructA {
    field_a: u64,
    field_b: u64,
    field_c: u64,
}

#[derive(BFieldCodec, PartialEq, Eq, Debug)]
struct DeriveTestStructB(u128);

#[derive(BFieldCodec, PartialEq, Eq, Debug)]
struct DeriveTestStructC(u128, u64, u32);

// Struct containing Vec<T> where T is BFieldCodec
#[derive(BFieldCodec, PartialEq, Eq, Debug)]
struct DeriveTestStructD(Vec<u128>);

#[derive(BFieldCodec, PartialEq, Eq, Debug, Default)]
struct DeriveTestStructE(Vec<u128>, u128, u64, Vec<bool>, u32, Vec<u128>);

#[derive(BFieldCodec, PartialEq, Eq, Debug, Default)]
struct DeriveTestStructF {
    field_a: Vec<u64>,
    field_b: bool,
    field_c: u32,
    field_d: Vec<bool>,
    field_e: Vec<BFieldElement>,
    field_f: Vec<BFieldElement>,
}

#[derive(BFieldCodec, PartialEq, Eq, Debug, Default)]
struct WithPhantomData<H: AlgebraicHasher> {
    a_field: u128,
    #[bfield_codec(ignore)]
    _phantom_data: PhantomData<H>,
    another_field: Vec<u64>,
}

#[derive(BFieldCodec, PartialEq, Eq, Debug, Default)]
struct WithNestedPhantomData<H: AlgebraicHasher> {
    a_field: u128,
    #[bfield_codec(ignore)]
    _phantom_data: PhantomData<H>,
    another_field: Vec<u64>,
    a_third_field: Vec<WithPhantomData<H>>,
    a_fourth_field: WithPhantomData<Tip5>,
}

#[cfg(test)]
pub mod derive_tests {
    // Since we cannot use the derive macro in the same crate where it is defined,
    // we test the macro here instead.

    use rand::{random, thread_rng, Rng};

    use crate::{
        shared_math::{other::random_elements, tip5::Tip5},
        util_types::algebraic_hasher::AlgebraicHasher,
    };

    use super::*;

    fn prop<T: BFieldCodec + PartialEq + Eq + std::fmt::Debug>(value: T) {
        let encoded = value.encode();
        let decoded = T::decode(&encoded);
        let decoded = decoded.unwrap();
        assert_eq!(value, *decoded);

        let encoded_too_long = vec![encoded, vec![BFieldElement::new(5)]].concat();
        assert!(T::decode(&encoded_too_long).is_err());

        let encoded_too_short = encoded_too_long[..encoded_too_long.len() - 2].to_vec();
        assert!(T::decode(&encoded_too_short).is_err());
    }

    #[test]
    fn simple_struct_with_named_fields() {
        prop(DeriveTestStructA {
            field_a: 14,
            field_b: 555558,
            field_c: 1337,
        });
    }

    #[test]
    fn simple_struct_with_one_unnamed_field() {
        prop(DeriveTestStructB(127));
    }

    #[test]
    fn simple_struct_with_unnamed_fields() {
        prop(DeriveTestStructC(127 << 100, 14, 1000));
    }

    #[test]
    fn struct_with_unnamed_vec_field() {
        prop(DeriveTestStructD(vec![
            1 << 99,
            99,
            1 << 120,
            120,
            u64::MAX as u128,
        ]));

        // Test the empty struct
        prop(DeriveTestStructD(vec![]));
    }

    #[test]
    fn struct_with_unnamed_vec_fields() {
        fn random_struct() -> DeriveTestStructE {
            let mut rng = thread_rng();
            let length_0: usize = rng.gen_range(0..10);
            let length_3: usize = rng.gen_range(0..20);
            let length_5: usize = rng.gen_range(0..20);
            DeriveTestStructE(
                random_elements(length_0),
                random(),
                random(),
                random_elements(length_3),
                random(),
                random_elements(length_5),
            )
        }
        for _ in 0..20 {
            prop(random_struct());
        }

        // Also test the Default/empty struct
        prop(DeriveTestStructE::default());
    }

    #[test]
    fn struct_with_named_vec_fields() {
        fn random_struct() -> DeriveTestStructF {
            let mut rng = thread_rng();
            let length_a: usize = rng.gen_range(0..10);
            let length_d: usize = rng.gen_range(0..20);
            let length_e: usize = rng.gen_range(0..20);
            let length_f: usize = rng.gen_range(0..20);
            DeriveTestStructF {
                field_a: random_elements(length_a),
                field_b: random(),
                field_c: random(),
                field_d: random_elements(length_d),
                field_e: random_elements(length_e),
                field_f: random_elements(length_f),
            }
        }
        for _ in 0..20 {
            prop(random_struct());
        }

        // Also test the Default/empty struct
        prop(DeriveTestStructF::default());
    }

    #[test]
    fn struct_with_phantom_data() {
        fn random_struct() -> WithPhantomData<Tip5> {
            let mut rng = thread_rng();
            let length_another_field: usize = rng.gen_range(0..10);
            WithPhantomData {
                a_field: random(),
                _phantom_data: PhantomData,
                another_field: random_elements(length_another_field),
            }
        }
        for _ in 0..20 {
            prop(random_struct());
        }

        // Also test the Default/empty struct
        prop(WithPhantomData::<Tip5>::default());
    }

    #[test]
    fn struct_with_nested_phantom_data() {
        fn random_struct() -> WithPhantomData<Tip5> {
            let mut rng = thread_rng();
            let length_another_field: usize = rng.gen_range(0..10);
            WithNestedPhantomData {
                a_field: random(),
                _phantom_data: PhantomData,
                another_field: random_elements(length_another_field),
            }
        }
        for _ in 0..20 {
            prop(random_struct());
        }

        // Also test the Default/empty struct
        prop(WithNestedPhantomData::<Tip5>::default());
    }
}
