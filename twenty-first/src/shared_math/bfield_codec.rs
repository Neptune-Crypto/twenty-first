use std::marker::PhantomData;
use std::slice::Iter;

use anyhow::bail;
use anyhow::Result;
use itertools::Itertools;
use num_traits::One;
use num_traits::Zero;

// Re-export the derive macro so that it can be used in other crates without having to add
// an explicit dependency on `bfieldcodec_derive` to their Cargo.toml.
pub use bfieldcodec_derive::BFieldCodec;

use super::b_field_element::BFieldElement;
use super::tip5::Digest;
use super::tip5::DIGEST_LENGTH;
use super::x_field_element::XFieldElement;
use super::x_field_element::EXTENSION_DEGREE;

/// BFieldCodec
///
/// This trait provides functions for encoding to and decoding from a
/// Vec of BFieldElements. This encoding does not record the size of
/// objects nor their type information; this is
/// the responsibility of the decoder.
pub trait BFieldCodec {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>>;
    fn encode(&self) -> Vec<BFieldElement>;

    /// Returns the length in number of BFieldElements if it is known at compile-time.
    /// Otherwise, None.
    fn static_length() -> Option<usize>;
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

    fn static_length() -> Option<usize> {
        Some(1)
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

    fn static_length() -> Option<usize> {
        Some(EXTENSION_DEGREE)
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

    fn static_length() -> Option<usize> {
        Some(DIGEST_LENGTH)
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

    fn static_length() -> Option<usize> {
        Some(4)
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

    fn static_length() -> Option<usize> {
        Some(2)
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

    fn static_length() -> Option<usize> {
        Some(1)
    }
}

impl BFieldCodec for u32 {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.len() != 1 {
            bail!(
                "Cannot decode sequence of length {} =/= 1 BFieldElements as u32.",
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

    fn static_length() -> Option<usize> {
        Some(1)
    }
}

impl<T: BFieldCodec, S: BFieldCodec> BFieldCodec for (T, S) {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let mut index = 0;
        // decode T
        let len_t = match T::static_length() {
            Some(len) => len,
            None => {
                let length = match str.get(index) {
                    Some(bfe) => bfe.value() as usize,
                    None => bail!(
                        "Prepended length of type T satisfying unsized BFieldCodec does not exist"
                    ),
                };
                index += 1;
                length
            }
        };

        if str.len() < index + len_t {
            bail!("Sequence too short to decode tuple, element 0");
        }
        let t = *T::decode(&str[index..index + len_t])?;
        index += len_t;

        // decode S
        let len_s = match S::static_length() {
            Some(len) => len,
            None => {
                let length = match str.get(index) {
                    Some(bfe) => bfe.value() as usize,
                    None => bail!(
                        "Prepended length of type S satisfying unsized BFieldCodec does not exist"
                    ),
                };
                index += 1;
                length
            }
        };

        if str.len() < index + len_s {
            bail!("Sequence too short to decode tuple, element 1");
        }
        let s = *S::decode(&str[index..index + len_s])?;
        index += len_s;

        if index != str.len() {
            bail!("Error decoding (T,S): length mismatch");
        }

        Ok(Box::new((t, s)))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        let mut encoding_of_t = self.0.encode();
        let mut encoding_of_s = self.1.encode();
        if T::static_length().is_none() {
            str.push(BFieldElement::new(encoding_of_t.len() as u64));
        }
        str.append(&mut encoding_of_t);
        if S::static_length().is_none() {
            str.push(BFieldElement::new(encoding_of_s.len() as u64));
        }
        str.append(&mut encoding_of_s);
        str
    }

    fn static_length() -> Option<usize> {
        match (T::static_length(), S::static_length()) {
            (Some(sl_t), Some(sl_s)) => Some(sl_t + sl_s),
            _ => None,
        }
    }
}

impl<T: BFieldCodec> BFieldCodec for Option<T> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let is_some = match str.get(0) {
            Some(e) => {
                if e.is_one() || e.is_zero() {
                    e.is_one()
                } else {
                    bail!("Invalid option indicator: {e}")
                }
            }
            None => bail!("Cannot decode Option of T: empty sequence"),
        };

        if is_some {
            Ok(Box::new(Some(*T::decode(&str[1..])?)))
        } else {
            Ok(Box::new(None))
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

    fn static_length() -> Option<usize> {
        None
    }
}

impl<T: BFieldCodec, const N: usize> BFieldCodec for [T; N] {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if N > 0 && sequence.is_empty() {
            bail!("Cannot decode empty sequence into [T; {N}]");
        }

        let vec_t = bfield_codec_decode_list(N, sequence)?;
        let array = match vec_t.try_into() {
            Ok(array) => array,
            Err(_) => bail!("Cannot convert Vec<T> into [T; {N}]."),
        };
        Ok(Box::new(array))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        bfield_codec_encode_list(self.iter())
    }

    fn static_length() -> Option<usize> {
        T::static_length().map(|len| len * N)
    }
}

impl<T: BFieldCodec> BFieldCodec for Vec<T> {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if sequence.is_empty() {
            bail!("Cannot decode empty sequence into Vec<T>");
        }

        let indicated_num_elements = sequence[0].value() as usize;
        let vec_t = bfield_codec_decode_list(indicated_num_elements, &sequence[1..])?;
        Ok(Box::new(vec_t))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let num_elements = (self.len() as u64).into();
        let mut encoding = vec![num_elements];
        let mut encoded_items = bfield_codec_encode_list(self.iter());
        encoding.append(&mut encoded_items);
        encoding
    }

    fn static_length() -> Option<usize> {
        None
    }
}

/// The core of the [`BFieldCodec`] decoding logic for `Vec<T>` and `[T; N]`.
/// Decoding the length-prepending must be handled by the caller (if necessary).
fn bfield_codec_decode_list<T: BFieldCodec>(
    indicated_num_items: usize,
    sequence: &[BFieldElement],
) -> Result<Vec<T>> {
    // Initializing the vector with the indicated capacity potentially allows a DOS.
    let mut vec_t = vec![];

    let sequence_len = sequence.len();
    if T::static_length().is_some() {
        let item_length = T::static_length().unwrap();
        let maybe_vector_size = indicated_num_items.checked_mul(item_length);
        let Some(vector_size) = maybe_vector_size else {
            bail!("Length indication too large: {indicated_num_items} * {item_length}");
        };

        if sequence_len != vector_size {
            bail!(
                "Indicator for number of items must match sequence length. \
                List claims to contain {indicated_num_items} items. \
                Item size is {item_length}. Sequence length is {sequence_len}.",
            );
        }
        let raw_item_iter = sequence.chunks_exact(item_length);
        if !raw_item_iter.remainder().is_empty() {
            bail!("Could not chunk sequence into equal parts of size {item_length}.");
        }
        if raw_item_iter.len() != indicated_num_items {
            bail!(
                "List contains wrong number of items. Expected {indicated_num_items}, found {}.",
                raw_item_iter.len()
            );
        }
        for raw_item in raw_item_iter {
            let item = *T::decode(raw_item)?;
            vec_t.push(item);
        }
    } else {
        let mut sequence_index = 0;
        for item_idx in 1..=indicated_num_items {
            let item_length = match sequence.get(sequence_index) {
                Some(len) => len.value() as usize,
                None => bail!(
                    "Index count mismatch while decoding List of T. \
                    Attempted to decode item {item_idx} of {indicated_num_items}.",
                ),
            };
            sequence_index += 1;
            if sequence_len < sequence_index + item_length {
                bail!(
                    "Sequence too short to decode List of T: \
                    {sequence_len} < {sequence_index} + {item_length}. \
                    Attempted to decode item {item_idx} of {indicated_num_items}.",
                );
            }
            let item = *T::decode(&sequence[sequence_index..sequence_index + item_length])?;
            sequence_index += item_length;
            vec_t.push(item);
        }
        if sequence_len != sequence_index {
            bail!("List contains too many items: {sequence_len} != {sequence_index}.");
        }
    }
    Ok(vec_t)
}

/// The core of the [`BFieldCodec`] encoding logic for `Vec<T>` and `[T; N]`.
/// Encoding the length-prepending must be handled by the caller (if necessary).
fn bfield_codec_encode_list<T: BFieldCodec>(item_iter: Iter<T>) -> Vec<BFieldElement> {
    if T::static_length().is_some() {
        item_iter.flat_map(|elem| elem.encode()).collect()
    } else {
        let mut encoding = vec![];
        for elem in item_iter {
            let mut element_encoded = elem.encode();
            let element_len = (element_encoded.len() as u64).into();
            encoding.push(element_len);
            encoding.append(&mut element_encoded);
        }
        encoding
    }
}

impl<T> BFieldCodec for PhantomData<T> {
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        if !sequence.is_empty() {
            bail!(
                "Cannot decode non-empty BFE slice as phantom data; sequence length: {}",
                sequence.len()
            )
        }
        Ok(Box::new(PhantomData))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        vec![]
    }

    fn static_length() -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use crate::shared_math::other::random_elements;

    use super::*;
    use rand::{random, thread_rng};

    fn prop<T: BFieldCodec + PartialEq + Eq + std::fmt::Debug>(value: &T) {
        fn expensive_encoding_pbt<T: BFieldCodec + PartialEq + Eq + std::fmt::Debug>(
            original_encoded: &[BFieldElement],
            original_decoded: Box<T>,
        ) {
            const PBT_SIZE: usize = 50;

            // Go over every single element in the sequence, to see if mutating anyone
            // of the elements causes undesired behavior
            let mut mutated_encoding = original_encoded.to_vec();
            for i in 0..mutated_encoding.len() {
                let original_value = mutated_encoding[i];

                // Ensure that random values do not cause problems
                for _ in 0..PBT_SIZE {
                    mutated_encoding[i] = random();
                    let decoded_mutated = T::decode(&mutated_encoding);
                    assert!(
                        decoded_mutated.is_err()
                            || *decoded_mutated.unwrap() != *original_decoded
                            || mutated_encoding[i] == original_value
                    );
                }

                // Ensure that "nearby" values do not cause problems
                mutated_encoding[i] = original_value - BFieldElement::new((PBT_SIZE / 2) as u64);
                for _ in 0..PBT_SIZE {
                    mutated_encoding[i].increment();
                    let decoded_mutated = T::decode(&mutated_encoding);
                    assert!(
                        decoded_mutated.is_err()
                            || *decoded_mutated.unwrap() != *original_decoded
                            || mutated_encoding[i] == original_value
                    );
                }

                // Reset word
                mutated_encoding[i] = original_value;
            }

            // Ensure no crash while decoding very short sequences
            let upper_length_bound = std::cmp::min(mutated_encoding.len(), PBT_SIZE / 2);
            for i in 0..upper_length_bound {
                let random_bfes = random_elements(i);

                // Ensure `decode` does not crash
                let dec_res = T::decode(&random_bfes);
                assert!(
                    dec_res.is_err() || *dec_res.unwrap() != *original_decoded,
                    "random BFEs of shorter length than original encoding should not decode to the original struct"
                );
            }

            // Ensure no crash while decoding a sequence *one* shorter than the original sequence
            assert_eq!(
                original_encoded, mutated_encoding,
                "Sanity check of test integrity"
            );
            match mutated_encoding.pop() {
                Some(_) => {
                    let dec_res = T::decode(&mutated_encoding);
                    assert!(
                        dec_res.is_err() || *dec_res.unwrap() != *original_decoded,
                        "random BFEs of shorter length than original encoding should not decode to the original struct"
                    );
                }
                None => (),
            }
        }

        let encoded = value.encode();
        let decoded = T::decode(&encoded);
        let decoded = decoded.unwrap();
        assert_eq!(*value, *decoded);

        // Set this environment variable to run an expensive PBT of encoding/decoding
        // uniqueness and `decode`'s inability to panic
        if std::env::var("EXPENSIVE_ENCODING_PBT").is_ok() {
            expensive_encoding_pbt(&encoded, decoded);
        }

        let encoded_too_long = vec![encoded, vec![BFieldElement::new(5)]].concat();
        assert!(T::decode(&encoded_too_long).is_err());

        let encoded_too_short = encoded_too_long[..encoded_too_long.len() - 2].to_vec();
        assert!(T::decode(&encoded_too_short).is_err());
    }

    mod bfield_codec_tests {

        use rand::{Rng, RngCore};

        use crate::{
            shared_math::{other::random_elements, tip5::Tip5},
            util_types::merkle_tree::PartialAuthenticationPath,
        };

        use super::*;

        fn random_bool() -> bool {
            let mut rng = thread_rng();
            rng.next_u32() % 2 == 0
        }

        fn random_length(max: usize) -> usize {
            let mut rng = thread_rng();
            rng.next_u32() as usize % max
        }

        fn random_partial_authentication_paths(
            inner_length: usize,
            count: usize,
        ) -> Vec<PartialAuthenticationPath<Digest>> {
            let mut ret = vec![];

            for _ in 0..count {
                ret.push(
                    (0..inner_length)
                        .map(|_| if random_bool() { Some(random()) } else { None })
                        .collect_vec(),
                )
            }

            ret
        }

        #[test]
        fn test_encode_decode_random_bfieldelement() {
            for _ in 1..=10 {
                let bfe: BFieldElement = random();
                prop(&bfe);
            }
        }

        #[test]
        fn test_encode_decode_random_xfieldelement() {
            for _ in 1..=10 {
                let xfe: XFieldElement = random();
                prop(&xfe);
            }
        }

        #[test]
        fn test_encode_decode_random_digest() {
            for _ in 1..=10 {
                let dig: Digest = random();
                prop(&dig);
            }
        }

        #[test]
        fn test_encode_decode_random_vec_of_bfieldelement() {
            for _ in 1..=10 {
                let len = random_length(100);
                let bfe_vec: Vec<BFieldElement> = (0..len).map(|_| random()).collect_vec();
                prop(&bfe_vec);
            }
        }

        #[test]
        fn test_encode_decode_random_vec_of_xfieldelement() {
            for _ in 1..=10 {
                let len = random_length(100);
                let xfe_vec: Vec<XFieldElement> = (0..len).map(|_| random()).collect_vec();
                prop(&xfe_vec);
            }
        }

        #[test]
        fn test_encode_decode_random_vec_of_digest() {
            for _ in 1..=10 {
                let len = random_length(100);
                let digest_vec: Vec<Digest> = (0..len).map(|_| random()).collect_vec();
                prop(&digest_vec);
            }
        }

        #[test]
        fn test_encode_decode_random_vec_of_vec_of_bfieldelement() {
            for _ in 1..=10 {
                let len = random_length(10);
                let bfe_vec_vec: Vec<Vec<BFieldElement>> = (0..len)
                    .map(|_| {
                        let inner_len = random_length(20);
                        (0..inner_len).map(|_| random()).collect_vec()
                    })
                    .collect_vec();
                prop(&bfe_vec_vec);
            }
        }

        #[test]
        fn test_encode_decode_random_vec_of_vec_of_xfieldelement() {
            for _ in 1..=10 {
                let len = random_length(10);
                let xfe_vec_vec: Vec<Vec<XFieldElement>> = (0..len)
                    .map(|_| {
                        let inner_len = random_length(20);
                        (0..inner_len).map(|_| random()).collect_vec()
                    })
                    .collect_vec();
                prop(&xfe_vec_vec);
            }
        }

        #[test]
        fn test_encode_decode_random_partial_authentication_path() {
            for _ in 1..=10 {
                let len = 1 + random_length(10);
                let count = random_length(10);
                let pap = random_partial_authentication_paths(len, count);
                prop(&pap);
            }
        }

        #[test]
        fn test_encode_decode_tuples_static_static_size_0() {
            let static_static: (Digest, u128) = (random(), random());
            prop(&static_static);
        }

        #[test]
        fn test_encode_decode_tuples_static_static_size_1() {
            let static_static: (Digest, u64) = (random(), random());
            prop(&static_static);
        }

        #[test]
        fn test_encode_decode_tuples_static_static_size_2() {
            let static_static: (BFieldElement, BFieldElement) = (random(), random());
            prop(&static_static);
        }

        #[test]
        fn test_encode_decode_tuples_static_static_size_3() {
            let static_static: (BFieldElement, XFieldElement) = (random(), random());
            prop(&static_static);
        }

        #[test]
        fn test_encode_decode_tuples_static_static_size_4() {
            let static_static: (XFieldElement, BFieldElement) = (random(), random());
            prop(&static_static);
        }

        #[test]
        fn test_encode_decode_tuples_static_static_size_5() {
            let static_static: (XFieldElement, Digest) = (random(), random());
            prop(&static_static);
        }

        #[test]
        fn test_encode_decode_tuples_static_dynamic_size() {
            for vec_length in 0..10 {
                let static_dynamic: (Digest, Vec<BFieldElement>) =
                    (random(), random_elements(vec_length));
                prop(&static_dynamic);
            }
        }

        #[test]
        fn test_encode_decode_tuples_dynamic_static_size() {
            for vec_length in 0..10 {
                let dynamic_static: (Vec<XFieldElement>, Digest) =
                    (random_elements(vec_length), random());
                prop(&dynamic_static);
            }
        }

        #[test]
        fn test_encode_decode_tuples_dynamic_dynamic_size() {
            for vec_length_left in 0..4 {
                for vec_length_right in 0..4 {
                    let dynamic_dynamic: (Vec<XFieldElement>, Vec<Digest>) = (
                        random_elements(vec_length_left),
                        random_elements(vec_length_right),
                    );
                    prop(&dynamic_dynamic);
                }
            }
        }

        #[test]
        fn test_decode_random_negative() {
            for _ in 1..=10000 {
                let len = random_length(100);
                let str: Vec<BFieldElement> = random_elements(len);

                // Some of the following cases can be triggered by false
                // positives. This should occur with probability roughly
                // 2^-60.

                if let Ok(sth) = Vec::<BFieldElement>::decode(&str) {
                    panic!("{sth:?}");
                }

                if str.len() % EXTENSION_DEGREE != 1 {
                    if let Ok(sth) = Vec::<XFieldElement>::decode(&str) {
                        panic!("{sth:?}");
                    }
                }

                if str.len() % DIGEST_LENGTH != 1 {
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

                // if let Ok(_sth) = Vec::<PartialAuthenticationPath<Digest>>::decode(&str) {
                //     (will work quite often)
                // }
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

            prop(&vector);
        }

        #[test]
        fn test_phantom_data() {
            let pd = PhantomData::<Tip5>;
            let encoded = pd.encode();
            let decoded = *PhantomData::decode(&encoded).unwrap();
            assert_eq!(decoded, pd);

            assert!(encoded.is_empty());
        }

        #[test]
        fn decode_encode_array_with_static_element_size() {
            const N: usize = 14;
            type Array = [u64; N];

            let an_array: Array = [14u64; N];
            prop(&an_array);

            assert_eq!(
                Some(N * 2),
                <Array>::static_length(),
                "Static length must be set and correct for array with static element lengths"
            );
        }

        #[test]
        fn decode_encode_array_with_dynamic_element_size() {
            const N: usize = 19;
            type Array = [Vec<Digest>; N];

            // Initialize an array with empty vectors in all elements
            let mut an_array: Array = Default::default();
            prop(&an_array);

            // Should also work if the elements are not just the empty vectors
            let mut rng = thread_rng();
            for elem in an_array.iter_mut() {
                *elem = random_elements(rng.gen_range(0..7));
            }

            prop(&an_array);

            assert!(
                <Array>::static_length().is_none(),
                "Static length must be none for array with dynamic element lengths"
            );
        }
    }

    #[cfg(test)]
    pub mod derive_tests {
        // Since we cannot use the derive macro in the same crate where it is defined,
        // we test the macro here instead.

        use rand::{random, thread_rng, Rng, RngCore};

        use crate::{
            shared_math::{other::random_elements, tip5::Tip5},
            util_types::{
                algebraic_hasher::AlgebraicHasher, mmr::mmr_membership_proof::MmrMembershipProof,
            },
        };

        use super::*;

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

        #[derive(BFieldCodec, PartialEq, Eq, Debug, Default)]
        struct WithNestedVec {
            a_field: Vec<Vec<u64>>,
        }

        #[test]
        fn simple_struct_with_named_fields() {
            prop(&DeriveTestStructA {
                field_a: 14,
                field_b: 555558,
                field_c: 1337,
            });

            assert_eq!(Some(6), DeriveTestStructA::static_length());
        }

        #[test]
        fn simple_struct_with_one_unnamed_field() {
            prop(&DeriveTestStructB(127));

            assert_eq!(Some(4), DeriveTestStructB::static_length());
        }

        #[test]
        fn simple_struct_with_unnamed_fields() {
            prop(&DeriveTestStructC(127 << 100, 14, 1000));

            assert_eq!(Some(7), DeriveTestStructC::static_length());
        }

        #[test]
        fn struct_with_unnamed_vec_field() {
            prop(&DeriveTestStructD(vec![
                1 << 99,
                99,
                1 << 120,
                120,
                u64::MAX as u128,
            ]));

            // Test the empty struct
            prop(&DeriveTestStructD(vec![]));

            assert!(DeriveTestStructD::static_length().is_none());
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
                prop(&random_struct());
            }

            // Also test the Default/empty struct
            prop(&DeriveTestStructE::default());

            assert!(DeriveTestStructE::static_length().is_none());
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
                prop(&random_struct());
            }

            // Also test the Default/empty struct
            prop(&DeriveTestStructF::default());

            assert!(DeriveTestStructF::static_length().is_none());
        }

        fn random_with_phantomdata_struct() -> WithPhantomData<Tip5> {
            let mut rng = thread_rng();
            let length_another_field: usize = rng.gen_range(0..10);
            WithPhantomData {
                a_field: random(),
                _phantom_data: PhantomData,
                another_field: random_elements(length_another_field),
            }
        }

        #[test]
        fn struct_with_phantom_data() {
            for _ in 0..20 {
                prop(&random_with_phantomdata_struct());
            }

            // Also test the Default/empty struct
            prop(&WithPhantomData::<Tip5>::default());
        }

        #[test]
        fn struct_with_nested_phantom_data() {
            fn random_struct() -> WithNestedPhantomData<Tip5> {
                let mut rng = thread_rng();
                let length_a_fourth_field: usize = rng.gen_range(0..10);
                let a_third_field = (0..length_a_fourth_field)
                    .map(|_| random_with_phantomdata_struct())
                    .collect_vec();
                WithNestedPhantomData {
                    _phantom_data: PhantomData,
                    a_field: random(),
                    another_field: random_elements(rng.gen_range(0..15)),
                    a_fourth_field: random_with_phantomdata_struct(),
                    a_third_field,
                }
            }

            for _ in 0..10 {
                prop(&random_struct());
            }

            // Also test the Default/empty struct
            prop(&WithNestedPhantomData::<Tip5>::default());
        }

        #[test]
        fn struct_with_nested_vec() {
            fn random_struct() -> WithNestedVec {
                let mut rng = thread_rng();
                let outer_length = rng.gen_range(0..10);
                let mut ret = WithNestedVec {
                    a_field: Vec::with_capacity(outer_length),
                };
                for _ in 0..outer_length {
                    let inner_length = rng.gen_range(0..12);
                    let inner_vec: Vec<u64> = random_elements(inner_length);
                    ret.a_field.push(inner_vec);
                }

                ret
            }

            for _ in 0..10 {
                prop(&random_struct());
            }

            // Also test the Default/empty struct
            prop(&WithNestedVec::default());

            assert!(WithNestedVec::static_length().is_none());
        }

        #[test]
        fn deeply_nested_vec() {
            #[derive(BFieldCodec, PartialEq, Eq, Debug, Default)]
            struct MuchNesting {
                a: Vec<Vec<Vec<Vec<BFieldElement>>>>,
                #[bfield_codec(ignore)]
                b: PhantomData<Tip5>,
                #[bfield_codec(ignore)]
                c: PhantomData<Tip5>,
            }

            fn random_struct() -> MuchNesting {
                let mut rng = thread_rng();
                let outer_length = rng.gen_range(0..10);
                let mut ret = MuchNesting {
                    a: Vec::with_capacity(outer_length),
                    b: PhantomData,
                    c: PhantomData,
                };
                for i in 0..outer_length {
                    ret.a.push(vec![]);
                    let second_length = rng.gen_range(0..10);
                    for j in 0..second_length {
                        ret.a[i].push(vec![]);
                        let third_length = rng.gen_range(0..10);
                        for k in 0..third_length {
                            ret.a[i][j].push(vec![]);
                            ret.a[i][j][k] = random_elements(rng.gen_range(0..15));
                        }
                    }
                }

                ret
            }

            for _ in 0..5 {
                prop(&random_struct());
            }

            assert!(MuchNesting::static_length().is_none());
        }

        #[test]
        fn struct_with_small_array() {
            #[derive(BFieldCodec, PartialEq, Eq, Debug, Default)]
            struct SmallArrayStructUnnamedFields([u128; 1]);

            #[derive(BFieldCodec, PartialEq, Eq, Debug, Default)]
            struct SmallArrayStructNamedFields {
                a: [u128; 1],
            }

            fn random_struct_unnamed_fields() -> SmallArrayStructUnnamedFields {
                SmallArrayStructUnnamedFields([random(); 1])
            }

            fn random_struct_named_fields() -> SmallArrayStructNamedFields {
                SmallArrayStructNamedFields { a: [random(); 1] }
            }

            for _ in 0..5 {
                prop(&random_struct_unnamed_fields());
                prop(&random_struct_named_fields());
            }

            assert_eq!(Some(4), SmallArrayStructUnnamedFields::static_length());
            assert_eq!(Some(4), SmallArrayStructNamedFields::static_length());
        }

        #[test]
        fn struct_with_big_array() {
            const BIG_ARRAY_LENGTH: usize = 300;

            #[derive(BFieldCodec, PartialEq, Eq, Debug)]
            struct BigArrayStructUnnamedFields([u128; BIG_ARRAY_LENGTH]);

            #[derive(BFieldCodec, PartialEq, Eq, Debug)]
            struct BigArrayStructNamedFields {
                a: [XFieldElement; BIG_ARRAY_LENGTH],
                b: XFieldElement,
                c: u64,
            }

            fn random_struct_unnamed_fields() -> BigArrayStructUnnamedFields {
                BigArrayStructUnnamedFields(
                    random_elements::<u128>(BIG_ARRAY_LENGTH)
                        .try_into()
                        .unwrap(),
                )
            }

            fn random_struct_named_fields() -> BigArrayStructNamedFields {
                BigArrayStructNamedFields {
                    a: random_elements::<XFieldElement>(BIG_ARRAY_LENGTH)
                        .try_into()
                        .unwrap(),
                    b: random(),
                    c: random(),
                }
            }

            for _ in 0..3 {
                prop(&random_struct_unnamed_fields());
                prop(&random_struct_named_fields());
            }

            assert_eq!(
                Some(BIG_ARRAY_LENGTH * 4),
                BigArrayStructUnnamedFields::static_length()
            );
            assert_eq!(
                Some(BIG_ARRAY_LENGTH * 3 + 3 + 2),
                BigArrayStructNamedFields::static_length()
            );
        }

        #[test]
        fn struct_with_array_with_dynamically_sized_elements() {
            const ARRAY_LENGTH: usize = 7;

            #[derive(BFieldCodec, PartialEq, Eq, Debug)]
            struct ArrayStructDynamicallySizedElementsUnnamedFields([Vec<u128>; ARRAY_LENGTH]);

            #[derive(BFieldCodec, PartialEq, Eq, Debug)]
            struct ArrayStructDynamicallySizedElementsNamedFields {
                a: [Vec<u128>; ARRAY_LENGTH],
            }

            fn random_struct_unnamed_fields() -> ArrayStructDynamicallySizedElementsUnnamedFields {
                let mut rng = thread_rng();
                ArrayStructDynamicallySizedElementsUnnamedFields(std::array::from_fn(|_| {
                    random_elements(rng.gen_range(0..10))
                }))
            }

            fn random_struct_named_fields() -> ArrayStructDynamicallySizedElementsNamedFields {
                let mut rng = thread_rng();
                ArrayStructDynamicallySizedElementsNamedFields {
                    a: std::array::from_fn(|_| random_elements(rng.gen_range(0..10))),
                }
            }

            for _ in 0..5 {
                prop(&random_struct_unnamed_fields());
                prop(&random_struct_named_fields());
            }

            assert!(ArrayStructDynamicallySizedElementsUnnamedFields::static_length().is_none());
            assert!(ArrayStructDynamicallySizedElementsNamedFields::static_length().is_none());
        }

        #[test]
        fn struct_with_tuple_field_dynamic_size() {
            #[derive(BFieldCodec, PartialEq, Eq, Debug)]
            struct StructWithTupleField {
                a: (Digest, Vec<Digest>),
            }

            fn random_struct() -> StructWithTupleField {
                let mut rng = thread_rng();
                StructWithTupleField {
                    a: (random(), random_elements(rng.gen_range(0..10))),
                }
            }

            for _ in 0..5 {
                prop(&random_struct());
            }

            assert!(StructWithTupleField::static_length().is_none());
        }

        #[test]
        fn struct_with_tuple_field_static_size() {
            #[derive(BFieldCodec, PartialEq, Eq, Debug)]
            struct StructWithTupleField {
                a: ([Digest; 2], XFieldElement),
            }

            fn random_struct() -> StructWithTupleField {
                StructWithTupleField {
                    a: ([random(), random()], random()),
                }
            }

            for _ in 0..5 {
                prop(&random_struct());
            }

            assert_eq!(Some(13), StructWithTupleField::static_length());
        }

        #[test]
        fn struct_with_nested_tuple_field() {
            #[derive(BFieldCodec, PartialEq, Eq, Debug)]
            struct StructWithTupleField {
                a: (([Digest; 2], Vec<XFieldElement>), (XFieldElement, u64)),
            }

            fn random_struct() -> StructWithTupleField {
                let mut rng = thread_rng();
                StructWithTupleField {
                    a: (
                        ([random(), random()], random_elements(rng.gen_range(0..10))),
                        (random(), random()),
                    ),
                }
            }

            for _ in 0..5 {
                prop(&random_struct());
            }

            assert!(StructWithTupleField::static_length().is_none());
        }

        #[test]
        fn ms_membership_proof_derive_test() {
            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            struct MsMembershipProof<H: AlgebraicHasher> {
                sender_randomness: Digest,
                receiver_preimage: Digest,
                auth_path_aocl: MmrMembershipProof<H>,
            }

            fn random_mmr_membership_proof<H: AlgebraicHasher>() -> MmrMembershipProof<H> {
                let leaf_index: u64 = random();
                let authentication_path: Vec<Digest> =
                    random_elements((thread_rng().next_u32() % 15) as usize);
                MmrMembershipProof {
                    leaf_index,
                    authentication_path,
                    _hasher: PhantomData,
                }
            }

            fn random_struct() -> MsMembershipProof<Tip5> {
                let sender_randomness: Digest = random();
                let receiver_preimage: Digest = random();
                let auth_path_aocl: MmrMembershipProof<Tip5> = random_mmr_membership_proof();
                MsMembershipProof {
                    sender_randomness,
                    receiver_preimage,
                    auth_path_aocl,
                }
            }

            for _ in 0..5 {
                prop(&random_struct());
            }

            assert!(MsMembershipProof::<Tip5>::static_length().is_none());
        }

        #[test]
        fn mmr_bfieldcodec_derive_test() {
            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            struct MmrAccumulator<H: BFieldCodec> {
                leaf_count: u64,
                peaks: Vec<Digest>,
                #[bfield_codec(ignore)]
                _hasher: PhantomData<H>,
            }

            fn random_struct() -> MmrAccumulator<Tip5> {
                let leaf_count: u64 = random();
                let mut rng = thread_rng();
                let peaks = random_elements(rng.gen_range(0..63));
                MmrAccumulator {
                    leaf_count,
                    peaks,
                    _hasher: PhantomData,
                }
            }

            for _ in 0..5 {
                prop(&random_struct());
            }
        }

        #[test]
        fn unsupported_fields_can_be_ignored_test() {
            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            struct UnsupportedFields {
                a: u64,
                #[bfield_codec(ignore)]
                b: usize,
            }
            let my_struct = UnsupportedFields {
                a: random(),
                b: random(),
            };
            let encoded = my_struct.encode();
            let decoded = UnsupportedFields::decode(&encoded).unwrap();
            assert_eq!(
                my_struct.a, decoded.a,
                "Non-ignored fields must be preserved under encoding"
            );
            assert_eq!(
                usize::default(),
                decoded.b,
                "Ignored field must decode to default value"
            )
        }

        #[test]
        fn vec_of_struct_with_one_fix_len_field_test() {
            // This is a regression test addressing #124.
            // https://github.com/Neptune-Crypto/twenty-first/issues/124

            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            struct OneFixedLenField {
                some_digest: Digest,
            }

            let rand_struct = || OneFixedLenField {
                some_digest: random(),
            };

            for num_elements in 0..5 {
                dbg!(num_elements);
                prop(&vec![rand_struct(); num_elements]);
            }
        }

        #[test]
        fn vec_of_struct_with_two_fix_len_fields_test() {
            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            pub struct TwoFixedLenFields {
                pub some_digest: Digest,
                pub some_u64: u64,
            }

            let rand_struct = || TwoFixedLenFields {
                some_digest: random(),
                some_u64: random(),
            };

            for num_elements in 0..5 {
                dbg!(num_elements);
                prop(&vec![rand_struct(); num_elements]);
            }
        }

        #[test]
        fn vec_of_struct_with_two_fix_len_unnamed_fields_test() {
            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            pub struct TwoFixedLenUnnamedFields(Digest, u64);

            let rand_struct = || TwoFixedLenUnnamedFields(random(), random());

            for num_elements in 0..5 {
                dbg!(num_elements);
                prop(&vec![rand_struct(); num_elements]);
            }
        }

        #[test]
        fn vec_of_struct_with_fix_and_var_len_fields_test() {
            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            pub struct FixAndVarLenFields {
                pub some_digest: Digest,
                pub some_vec: Vec<u64>,
            }

            let rand_struct = || {
                let num_elements: usize = thread_rng().gen_range(0..7);
                FixAndVarLenFields {
                    some_digest: random(),
                    some_vec: random_elements(num_elements),
                }
            };

            for num_elements in 0..5 {
                dbg!(num_elements);
                prop(&vec![rand_struct(); num_elements]);
            }
        }

        #[test]
        fn vec_of_struct_with_fix_and_var_len_unnamed_fields_test() {
            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            struct FixAndVarLenUnnamedFields(Digest, Vec<u64>);

            let rand_struct = || {
                let num_elements: usize = thread_rng().gen_range(0..7);
                FixAndVarLenUnnamedFields(random(), random_elements(num_elements))
            };

            for num_elements in 0..5 {
                dbg!(num_elements);
                prop(&vec![rand_struct(); num_elements]);
            }
        }

        #[test]
        fn vec_of_struct_with_quite_a_few_fix_and_var_len_fields_test() {
            #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
            struct QuiteAFewFixAndVarLenFields {
                some_digest: Digest,
                some_vec: Vec<u64>,
                some_u64: u64,
                other_vec: Vec<u32>,
                other_digest: Digest,
                yet_another_vec: Vec<u32>,
                and_another_vec: Vec<u64>,
                more_fixed_len: u64,
                even_more_fixed_len: u64,
            }

            let rand_struct = || {
                let num_elements_0: usize = thread_rng().gen_range(0..7);
                let num_elements_1: usize = thread_rng().gen_range(0..7);
                let num_elements_2: usize = thread_rng().gen_range(0..7);
                let num_elements_3: usize = thread_rng().gen_range(0..7);
                QuiteAFewFixAndVarLenFields {
                    some_digest: random(),
                    some_vec: random_elements(num_elements_0),
                    some_u64: random(),
                    other_vec: random_elements(num_elements_1),
                    other_digest: random(),
                    yet_another_vec: random_elements(num_elements_2),
                    and_another_vec: random_elements(num_elements_3),
                    more_fixed_len: random(),
                    even_more_fixed_len: random(),
                }
            };

            for num_elements in 0..5 {
                dbg!(num_elements);
                prop(&vec![rand_struct(); num_elements]);
            }
        }
    }
}
