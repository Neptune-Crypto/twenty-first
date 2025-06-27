use std::cmp::Ordering;
use std::error::Error;
use std::fmt::Debug;
use std::fmt::Display;
use std::marker::PhantomData;
use std::slice::Iter;

// Re-export the derive macro so that it can be used in other crates without having to add
// an explicit dependency on `bfieldcodec_derive` to their Cargo.toml.
pub use bfieldcodec_derive::BFieldCodec;
use num_traits::ConstOne;
use num_traits::ConstZero;
use thiserror::Error;

use super::b_field_element::BFieldElement;
use super::polynomial::Polynomial;
use super::traits::FiniteField;
use crate::bfe;
use crate::bfe_vec;

/// This trait provides functions for encoding to and decoding from a Vec of
/// [BFieldElement]s. This encoding does not record the size of objects nor
/// their type information; this is the responsibility of the decoder.
///
/// ### Dyn-Compatibility
///
/// This trait is _not_ [dyn-compatible] (previously known as “object safe”).
///
/// [dyn-compatible]: https://doc.rust-lang.org/reference/items/traits.html#dyn-compatibility
pub trait BFieldCodec {
    type Error: Into<Box<dyn Error + Send + Sync>> + Debug + Display;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error>;
    fn encode(&self) -> Vec<BFieldElement>;

    /// Returns the length in number of [BFieldElement]s if it is known at compile-time.
    /// Otherwise, None.
    fn static_length() -> Option<usize>;
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BFieldCodecError {
    #[error("empty sequence")]
    EmptySequence,

    #[error("sequence too short")]
    SequenceTooShort,

    #[error("sequence too long")]
    SequenceTooLong,

    #[error("element out of range")]
    ElementOutOfRange,

    #[error("missing length indicator")]
    MissingLengthIndicator,

    #[error("invalid length indicator")]
    InvalidLengthIndicator,

    #[error(transparent)]
    TryFromIntError(#[from] std::num::TryFromIntError),

    #[error("inner decoding error: {0}")]
    InnerDecodingFailure(#[from] Box<dyn Error + Send + Sync>),
}

// The type underlying BFieldElement is u64. A single u64 does not fit in one BFieldElement.
// Therefore, deriving the BFieldCodec for BFieldElement using the derive macro will result in a
// BFieldCodec implementation that encodes a single BFieldElement as two BFieldElements.
// This is not desired. Hence, BFieldCodec is implemented manually for BFieldElement.
impl BFieldCodec for BFieldElement {
    type Error = BFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        if sequence.is_empty() {
            return Err(Self::Error::EmptySequence);
        }
        if sequence.len() > 1 {
            return Err(Self::Error::SequenceTooLong);
        }
        Ok(Box::new(sequence[0]))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        [*self].to_vec()
    }

    fn static_length() -> Option<usize> {
        Some(1)
    }
}

macro_rules! impl_bfield_codec_for_big_primitive_uint {
    ($ty:ty, $size:literal) => {
        impl BFieldCodec for $ty {
            type Error = BFieldCodecError;

            fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
                if sequence.is_empty() {
                    return Err(Self::Error::EmptySequence);
                }
                if sequence.len() < $size {
                    return Err(Self::Error::SequenceTooShort);
                }
                if sequence.len() > $size {
                    return Err(Self::Error::SequenceTooLong);
                }
                if sequence.iter().any(|s| s.value() > u32::MAX.into()) {
                    return Err(Self::Error::ElementOutOfRange);
                }

                let element = sequence
                    .iter()
                    .enumerate()
                    .map(|(i, s)| Self::from(s) << (i * 32))
                    .sum();
                Ok(Box::new(element))
            }

            fn encode(&self) -> Vec<BFieldElement> {
                const LOW_BIT_MASK: $ty = u32::MAX as $ty;

                (0..$size)
                    .map(|i| bfe!((*self >> (i * 32)) & LOW_BIT_MASK))
                    .collect()
            }

            fn static_length() -> Option<usize> {
                Some($size)
            }
        }
    };
}

impl_bfield_codec_for_big_primitive_uint!(u64, 2);
impl_bfield_codec_for_big_primitive_uint!(u128, 4);

macro_rules! impl_bfield_codec_for_signed_int {
    ($iint:ty, using $uint:ty) => {
        impl BFieldCodec for $iint {
            type Error = <$uint as BFieldCodec>::Error;

            fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
                <$uint>::decode(sequence).map(|v| Box::new(*v as Self))
            }

            fn encode(&self) -> Vec<BFieldElement> {
                (*self as $uint).encode()
            }

            fn static_length() -> Option<usize> {
                <$uint>::static_length()
            }
        }
    };
}

impl_bfield_codec_for_signed_int!(i8, using u8);
impl_bfield_codec_for_signed_int!(i16, using u16);
impl_bfield_codec_for_signed_int!(i32, using u32);
impl_bfield_codec_for_signed_int!(i64, using u64);
impl_bfield_codec_for_signed_int!(i128, using u128);

impl BFieldCodec for bool {
    type Error = BFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        if sequence.is_empty() {
            return Err(Self::Error::EmptySequence);
        }
        if sequence.len() > 1 {
            return Err(Self::Error::SequenceTooLong);
        }

        let element = match sequence[0].value() {
            0 => false,
            1 => true,
            _ => return Err(Self::Error::ElementOutOfRange),
        };
        Ok(Box::new(element))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        vec![BFieldElement::new(*self as u64)]
    }

    fn static_length() -> Option<usize> {
        Some(1)
    }
}

macro_rules! impl_bfield_codec_for_small_primitive_uint {
    ($($t:ident),+ $(,)?) => {$(
        impl BFieldCodec for $t {
            type Error = BFieldCodecError;

            fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
                if sequence.is_empty() {
                    return Err(Self::Error::EmptySequence);
                }
                let [first] = sequence[..] else {
                    return Err(Self::Error::SequenceTooLong);
                };
                let element = $t::try_from(first.value())
                    .map_err(|_| Self::Error::ElementOutOfRange)?;

                Ok(Box::new(element))
            }

            fn encode(&self) -> Vec<BFieldElement> {
                bfe_vec![*self]
            }

            fn static_length() -> Option<usize> {
                Some(1)
            }
        }
    )+};
}

impl_bfield_codec_for_small_primitive_uint!(u8, u16, u32);

impl<T: BFieldCodec> BFieldCodec for Box<T> {
    type Error = T::Error;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        T::decode(sequence).map(Box::new)
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.as_ref().encode()
    }

    fn static_length() -> Option<usize> {
        T::static_length()
    }
}

macro_rules! impl_bfield_codec_for_tuple {
    // entrypoint: introduce marker tokens to start reversing token sequence
    ($($forward:ident),+) => {
        impl_bfield_codec_for_tuple!(@>@ $($forward)+ @rev@);
    };

    // general case: reversing in progress, move marker token `@>@`
    ($($a:ident)* @>@ $b:ident $($c:ident)* @rev@ $($reverse:ident)*) => {
        impl_bfield_codec_for_tuple!($($a)* $b @>@ $($c)* @rev@ $b $($reverse)*);
    };

    // base case: reversing is done
    //
    // Use token sequence in “forward” mode to
    //  - declare the tuple type,
    //  - construct the tuple after successful decoding, and
    //  - pattern-match-deconstruct `self` to prepare encoding.
    // Use token sequence in “reverse” mode to en/decode in reverse order from type
    // declaration.
    //
    // Because it is impossible to construct tuple accessors (like the 0 in
    // `my_tuple.0`) in a declarative macro, pattern matching is used to
    // deconstruct `self`. This requires identifiers. The current design uses the
    // type identifiers for this purpose. While these identifiers don't follow the
    // guidelines for variable identifiers, this is deemed acceptable in macro code.
    ($($f:ident)+ @>@ @rev@ $($r:ident)+) => {
        impl <$($f),+> BFieldCodec for ($($f),+ ,)
        where $($f: BFieldCodec),+
        {
            type Error = BFieldCodecError;

            fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
                $(
                if $r::static_length().is_none() && sequence.first().is_none() {
                    return Err(Self::Error::MissingLengthIndicator);
                }
                let (length, sequence) = $r::static_length()
                    .map(|length| (Ok(length), sequence))
                    .unwrap_or_else(|| (sequence[0].try_into(), &sequence[1..]));
                let length = length?;
                if sequence.len() < length {
                    return Err(Self::Error::SequenceTooShort);
                }
                let (sequence_for_ty, sequence) = sequence.split_at(length);
                #[allow(non_snake_case)]
                let $r = *$r::decode(sequence_for_ty).map_err(|err| err.into())?;
                )+

                if !sequence.is_empty() {
                    return Err(Self::Error::SequenceTooLong);
                }
                Ok(Box::new(($($f),+ ,)))
            }

            fn encode(&self) -> Vec<BFieldElement> {
                #[allow(non_snake_case)]
                let ($($f),+ ,) = self;
                let mut sequence = vec![];

                $(
                let encoding = $r.encode();
                if $r::static_length().is_none() {
                    sequence.push(encoding.len().into());
                }
                sequence.extend(encoding);
                )+

                sequence
            }

            fn static_length() -> Option<usize> {
                // Token `+` cannot be used as macro repetition separator. As a workaround, use
                // it inside macro repetition, and make last emitted `+` syntactic by adding 0.
                Some($($f::static_length()? +)+ 0)
            }
        }
    };
}

impl_bfield_codec_for_tuple!(A);
impl_bfield_codec_for_tuple!(A, B);
impl_bfield_codec_for_tuple!(A, B, C);
impl_bfield_codec_for_tuple!(A, B, C, D);
impl_bfield_codec_for_tuple!(A, B, C, D, E);
impl_bfield_codec_for_tuple!(A, B, C, D, E, F);
impl_bfield_codec_for_tuple!(A, B, C, D, E, F, G);
impl_bfield_codec_for_tuple!(A, B, C, D, E, F, G, H);
impl_bfield_codec_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_bfield_codec_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_bfield_codec_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_bfield_codec_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

impl<T: BFieldCodec> BFieldCodec for Option<T> {
    type Error = BFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        if sequence.is_empty() {
            return Err(Self::Error::EmptySequence);
        }
        let is_some = *bool::decode(&sequence[0..1])?;
        let sequence = &sequence[1..];
        let maybe_result = is_some.then(|| T::decode(sequence).map_err(|e| e.into()));
        let element = maybe_result.transpose()?.map(|boxed_self| *boxed_self);

        if !is_some && !sequence.is_empty() {
            return Err(Self::Error::SequenceTooLong);
        }
        Ok(Box::new(element))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        match self {
            None => vec![BFieldElement::ZERO],
            Some(t) => [vec![BFieldElement::ONE], t.encode()].concat(),
        }
    }

    fn static_length() -> Option<usize> {
        None
    }
}

impl<T: BFieldCodec, const N: usize> BFieldCodec for [T; N] {
    type Error = BFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        if N > 0 && sequence.is_empty() {
            return Err(Self::Error::EmptySequence);
        }

        let vec_t = bfield_codec_decode_list(N, sequence)?;
        let array = vec_t.try_into().map_err(|_| {
            Self::Error::InnerDecodingFailure(format!("cannot convert Vec<T> into [T; {N}]").into())
        })?;
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
    type Error = BFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        if sequence.is_empty() {
            return Err(Self::Error::EmptySequence);
        }

        let vec_length = sequence[0].try_into()?;
        let vec = bfield_codec_decode_list(vec_length, &sequence[1..])?;
        Ok(Box::new(vec))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut encoding = bfe_vec![self.len()];
        encoding.extend(bfield_codec_encode_list(self.iter()));
        encoding
    }

    fn static_length() -> Option<usize> {
        None
    }
}

#[derive(Debug, Error)]
pub enum PolynomialBFieldCodecError {
    /// A polynomial with leading zero-coefficients, corresponding to an encoding
    /// with trailing zeros, is a waste of space. More importantly, it allows for
    /// non-canonical encodings of the same underlying polynomial, which is
    /// confusing and can lead to errors.
    ///
    /// A note on “leading” vs “trailing”:
    /// The polynomial 0·x³ + 2x² + 42 has a spuriuous _leading_ zero, which
    /// translates into a spurious _trailing_ zero in its encoding.
    #[error("trailing zeros in polynomial-encoding")]
    TrailingZerosInPolynomialEncoding,

    #[error(transparent)]
    Other(#[from] BFieldCodecError),
}

impl<T> BFieldCodec for Polynomial<'_, T>
where
    T: BFieldCodec + FiniteField + 'static,
{
    type Error = PolynomialBFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        if sequence.is_empty() {
            return Err(Self::Error::Other(BFieldCodecError::EmptySequence));
        }

        let coefficients_field_length_indicator: Result<usize, _> = sequence[0].value().try_into();
        let Ok(coefficients_field_length_indicator) = coefficients_field_length_indicator else {
            return Err(Self::Error::Other(BFieldCodecError::InvalidLengthIndicator));
        };

        // Indicated sequence length is 1 + field size, as the size indicator takes up 1 word.
        let indicated_sequence_length = coefficients_field_length_indicator + 1;
        let decoded_vec = match sequence.len().cmp(&indicated_sequence_length) {
            Ordering::Equal => Vec::<T>::decode(&sequence[1..]),
            Ordering::Less => Err(BFieldCodecError::SequenceTooShort),
            Ordering::Greater => Err(BFieldCodecError::SequenceTooLong),
        }?;

        let encoding_contains_trailing_zeros = decoded_vec
            .last()
            .is_some_and(|last_coeff| last_coeff.is_zero());
        if encoding_contains_trailing_zeros {
            return Err(PolynomialBFieldCodecError::TrailingZerosInPolynomialEncoding);
        }

        Ok(Box::new(Polynomial::new(*decoded_vec)))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        // The length of the coefficients field is prepended to the encoding to make the
        // encoding consistent with a hypothetical derived implementation.
        let coefficients_encoded = self.coefficients().to_owned().encode();
        [bfe_vec!(coefficients_encoded.len()), coefficients_encoded].concat()
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
) -> Result<Vec<T>, BFieldCodecError> {
    let vec = if T::static_length().is_some() {
        bfield_codec_decode_list_with_statically_sized_items(indicated_num_items, sequence)?
    } else {
        bfield_codec_decode_list_with_dynamically_sized_items(indicated_num_items, sequence)?
    };
    Ok(vec)
}

fn bfield_codec_decode_list_with_statically_sized_items<T: BFieldCodec>(
    num_items: usize,
    sequence: &[BFieldElement],
) -> Result<Vec<T>, BFieldCodecError> {
    // Initializing the vector with the indicated capacity potentially allows a DOS.
    let mut vec = vec![];

    let item_length = T::static_length().unwrap();
    let maybe_vector_size = num_items.checked_mul(item_length);
    let Some(vector_size) = maybe_vector_size else {
        return Err(BFieldCodecError::InvalidLengthIndicator);
    };
    if sequence.len() < vector_size {
        return Err(BFieldCodecError::SequenceTooShort);
    }
    if sequence.len() > vector_size {
        return Err(BFieldCodecError::SequenceTooLong);
    }

    for raw_item in sequence.chunks_exact(item_length) {
        let item = *T::decode(raw_item).map_err(|e| e.into())?;
        vec.push(item);
    }
    Ok(vec)
}

fn bfield_codec_decode_list_with_dynamically_sized_items<T: BFieldCodec>(
    num_items: usize,
    sequence: &[BFieldElement],
) -> Result<Vec<T>, BFieldCodecError> {
    // Initializing the vector with the indicated capacity potentially allows a DOS.
    let mut vec = vec![];
    let mut sequence_index = 0;
    for _ in 0..num_items {
        let item_length = sequence
            .get(sequence_index)
            .ok_or(BFieldCodecError::MissingLengthIndicator)?;
        let item_length = usize::try_from(item_length)?;
        sequence_index += 1;
        if sequence.len() < sequence_index + item_length {
            return Err(BFieldCodecError::SequenceTooShort);
        }
        let item = *T::decode(&sequence[sequence_index..sequence_index + item_length])
            .map_err(|e| e.into())?;
        sequence_index += item_length;
        vec.push(item);
    }
    if sequence.len() != sequence_index {
        return Err(BFieldCodecError::SequenceTooLong);
    }
    Ok(vec)
}

/// The core of the [`BFieldCodec`] encoding logic for `Vec<T>` and `[T; N]`.
/// Encoding the length-prepending must be handled by the caller (if necessary).
fn bfield_codec_encode_list<T: BFieldCodec>(elements: Iter<T>) -> Vec<BFieldElement> {
    if T::static_length().is_some() {
        return elements.flat_map(|elem| elem.encode()).collect();
    }

    let mut encoding = vec![];
    for element in elements {
        let element_encoded = element.encode();
        encoding.push(element_encoded.len().into());
        encoding.extend(element_encoded);
    }
    encoding
}

impl BFieldCodec for () {
    type Error = BFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        sequence
            .is_empty()
            .then(|| Box::new(()))
            .ok_or_else(|| BFieldCodecError::SequenceTooLong)
    }

    fn encode(&self) -> Vec<BFieldElement> {
        Vec::new()
    }

    fn static_length() -> Option<usize> {
        Some(0)
    }
}

impl<T> BFieldCodec for PhantomData<T> {
    type Error = BFieldCodecError;

    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>, Self::Error> {
        sequence
            .is_empty()
            .then(|| Box::new(PhantomData))
            .ok_or_else(|| BFieldCodecError::SequenceTooLong)
    }

    fn encode(&self) -> Vec<BFieldElement> {
        Vec::new()
    }

    fn static_length() -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use num_traits::ConstZero;
    use num_traits::Zero;
    use proptest::collection::size_range;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;
    use crate::prelude::*;

    #[derive(Debug, PartialEq, Eq, test_strategy::Arbitrary)]
    struct BFieldCodecPropertyTestData<T>
    where
        T: 'static + BFieldCodec + Eq + Debug + Clone + for<'a> arbitrary::Arbitrary<'a>,
    {
        #[strategy(arb())]
        value: T,

        #[strategy(Just(#value.encode()))]
        encoding: Vec<BFieldElement>,

        #[strategy(vec(arb(), #encoding.len()))]
        #[filter(#encoding.iter().zip(#random_encoding.iter()).all(|(a, b)| a != b))]
        random_encoding: Vec<BFieldElement>,

        #[any(size_range(1..128).lift())]
        encoding_lengthener: Vec<BFieldElement>,

        #[strategy(0..=#encoding.len())]
        length_of_too_short_sequence: usize,
    }

    impl<T> BFieldCodecPropertyTestData<T>
    where
        T: 'static + BFieldCodec + Eq + Debug + Clone + for<'a> arbitrary::Arbitrary<'a>,
    {
        fn assert_bfield_codec_properties(&self) -> TestCaseResult {
            self.assert_static_length_is_equal_to_encoded_length()?;
            self.assert_decoded_encoding_is_self()?;
            self.assert_decoding_too_long_encoding_fails()?;
            self.assert_decoding_too_short_encoding_fails()?;
            self.modify_each_element_and_assert_decoding_failure()?;
            self.assert_decoding_random_too_short_encoding_fails_gracefully()?;
            Ok(())
        }

        fn assert_static_length_is_equal_to_encoded_length(&self) -> TestCaseResult {
            if let Some(static_len) = T::static_length() {
                prop_assert_eq!(static_len, self.encoding.len());
            }
            Ok(())
        }

        fn assert_decoded_encoding_is_self(&self) -> TestCaseResult {
            let Ok(decoding) = T::decode(&self.encoding) else {
                let err = TestCaseError::Fail("decoding canonical encoding must not fail".into());
                return Err(err);
            };
            prop_assert_eq!(&self.value, &*decoding);
            Ok(())
        }

        fn assert_decoding_too_long_encoding_fails(&self) -> TestCaseResult {
            let mut too_long_encoding = self.encoding.to_owned();
            too_long_encoding.extend(self.encoding_lengthener.to_owned());
            prop_assert!(T::decode(&too_long_encoding).is_err());
            Ok(())
        }

        fn assert_decoding_too_short_encoding_fails(&self) -> TestCaseResult {
            if self.failure_assertions_for_decoding_too_short_sequence_is_not_meaningful() {
                return Ok(());
            }
            let mut encoded = self.encoding.to_owned();
            encoded.pop();
            prop_assert!(T::decode(&encoded).is_err());
            Ok(())
        }

        fn modify_each_element_and_assert_decoding_failure(&self) -> TestCaseResult {
            let mut encoding = self.encoding.to_owned();
            for i in 0..encoding.len() {
                let original_value = encoding[i];
                encoding[i] = self.random_encoding[i];
                let decoding = T::decode(&encoding);
                if decoding.is_ok_and(|d| *d == self.value) {
                    return Err(TestCaseError::Fail(format!("failing index: {i}").into()));
                }
                encoding[i] = original_value;
            }
            Ok(())
        }

        fn assert_decoding_random_too_short_encoding_fails_gracefully(&self) -> TestCaseResult {
            if self.failure_assertions_for_decoding_too_short_sequence_is_not_meaningful() {
                return Ok(());
            }

            let random_encoding =
                self.random_encoding[..self.length_of_too_short_sequence].to_vec();
            let decoding_result = T::decode(&random_encoding);
            if decoding_result.is_ok_and(|d| *d == self.value) {
                return Err(TestCaseError::Fail("randomness mustn't be `self`".into()));
            }
            Ok(())
        }

        fn failure_assertions_for_decoding_too_short_sequence_is_not_meaningful(&self) -> bool {
            self.encoding.is_empty()
        }
    }

    macro_rules! test_case {
        (fn $fn_name:ident for $t:ty: $static_len:expr) => {
            #[proptest]
            fn $fn_name(test_data: BFieldCodecPropertyTestData<$t>) {
                prop_assert_eq!($static_len, <$t as BFieldCodec>::static_length());
                test_data.assert_bfield_codec_properties()?;
            }
        };
    }

    macro_rules! neg_test_case {
        (fn $fn_name:ident for $t:ty) => {
            #[proptest]
            fn $fn_name(random_encoding: Vec<BFieldElement>) {
                let decoding = <$t as BFieldCodec>::decode(&random_encoding);
                prop_assert!(decoding.is_err());
            }
        };
    }

    test_case! { fn test_u8 for u8: Some(1) }
    test_case! { fn test_i8 for i8: Some(1) }
    test_case! { fn test_u16 for u16: Some(1) }
    test_case! { fn test_i16 for i16: Some(1) }
    test_case! { fn test_u32 for u32: Some(1) }
    test_case! { fn test_i32 for i32: Some(1) }
    test_case! { fn test_u64 for u64: Some(2) }
    test_case! { fn test_i64 for i64: Some(2) }
    test_case! { fn test_u128 for u128: Some(4) }
    test_case! { fn test_i128 for i128: Some(4) }
    test_case! { fn bfieldelement for BFieldElement: Some(1) }
    test_case! { fn xfieldelement for XFieldElement: Some(3) }
    test_case! { fn digest for Digest: Some(5) }
    test_case! { fn vec_of_bfieldelement for Vec<BFieldElement>: None }
    test_case! { fn vec_of_xfieldelement for Vec<XFieldElement>: None }
    test_case! { fn vec_of_digest for Vec<Digest>: None }
    test_case! { fn vec_of_vec_of_bfieldelement for Vec<Vec<BFieldElement>>: None }
    test_case! { fn vec_of_vec_of_xfieldelement for Vec<Vec<XFieldElement>>: None }
    test_case! { fn vec_of_vec_of_digest for Vec<Vec<Digest>>: None }
    test_case! { fn poly_bfe for Polynomial<'static, BFieldElement>: None }
    test_case! { fn poly_xfe for Polynomial<'static, XFieldElement>: None }
    test_case! { fn tuples_static_static_size_0 for (Digest, u128): Some(9) }
    test_case! { fn tuples_static_static_size_1 for (Digest, u64): Some(7) }
    test_case! { fn tuples_static_static_size_2 for (BFieldElement, BFieldElement): Some(2) }
    test_case! { fn tuples_static_static_size_3 for (BFieldElement, XFieldElement): Some(4) }
    test_case! { fn tuples_static_static_size_4 for (XFieldElement, BFieldElement): Some(4) }
    test_case! { fn tuples_static_static_size_5 for (XFieldElement, Digest): Some(8) }
    test_case! { fn tuples_static_static_size_6 for (u8, u16, u32, u64): Some(5) }
    test_case! { fn tuples_static_static_size_7 for (u8,): Some(1) }
    test_case! { fn tuples_static_dynamic_size_0 for (u32, Vec<u32>): None }
    test_case! { fn tuples_static_dynamic_size_1 for (u32, Vec<u64>): None }
    test_case! { fn tuples_static_dynamic_size_2 for (Digest, Vec<BFieldElement>): None }
    test_case! { fn tuples_static_dynamic_size_3 for (Vec<BFieldElement>,): None }
    test_case! { fn tuples_dynamic_static_size for (Vec<XFieldElement>, Digest): None }
    test_case! { fn tuples_dynamic_dynamic_size for (Vec<XFieldElement>, Vec<Digest>): None }
    test_case! { fn unit for (): Some(0) }
    test_case! { fn phantom_data for PhantomData<Tip5>: Some(0) }
    test_case! { fn boxed_u32s for Box<u32>: Some(1) }
    test_case! { fn tuple_with_boxed_bfe for (u64, Box<BFieldElement>): Some(3) }
    test_case! { fn tuple_with_boxed_digest for (u128, Box<Digest>): Some(9) }
    test_case! { fn vec_of_boxed_tuple_of_u128_and_bfe for Vec<(u128, Box<BFieldElement>)>: None }
    test_case! { fn vec_option_xfieldelement for Vec<Option<XFieldElement>>: None }
    test_case! { fn array_with_static_element_size for [u64; 14]: Some(28) }
    test_case! { fn array_with_dynamic_element_size for [Vec<Digest>; 19]: None }

    neg_test_case! { fn vec_of_bfield_element_neg for Vec<BFieldElement> }
    neg_test_case! { fn vec_of_xfield_elements_neg for Vec<XFieldElement> }
    neg_test_case! { fn vec_of_digests_neg for Vec<Digest> }
    neg_test_case! { fn vec_of_vec_of_bfield_elements_neg for Vec<Vec<BFieldElement>> }
    neg_test_case! { fn vec_of_vec_of_xfield_elements_neg for Vec<Vec<XFieldElement>> }
    neg_test_case! { fn poly_of_bfe_neg for Polynomial<BFieldElement> }
    neg_test_case! { fn poly_of_xfe_neg for Polynomial<XFieldElement> }

    #[test]
    fn encoding_tuple_puts_fields_in_expected_order() {
        let encoding = (1_u8, 2_u16, 3_u32, 4_u64, true).encode();
        let expected = bfe_vec![1, 4, 0, 3, 2, 1];
        assert_eq!(expected, encoding);
    }

    #[test]
    fn leading_zero_coefficient_have_no_effect_on_encoding_empty_poly_bfe() {
        let empty_poly = Polynomial::<BFieldElement>::new(vec![]);
        assert_eq!(empty_poly.encode(), Polynomial::new(bfe_vec![0]).encode());
    }

    #[test]
    fn leading_zero_coefficients_have_no_effect_on_encoding_empty_poly_xfe() {
        let empty_poly = Polynomial::<XFieldElement>::new(vec![]);
        assert_eq!(empty_poly.encode(), Polynomial::new(xfe_vec![0]).encode());
    }

    #[proptest]
    fn leading_zero_coefficients_have_no_effect_on_encoding_poly_bfe_pbt(
        polynomial: Polynomial<'static, BFieldElement>,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let encoded = polynomial.encode();

        let mut coefficients = polynomial.into_coefficients();
        coefficients.extend(vec![BFieldElement::ZERO; num_leading_zeros]);
        let poly_w_leading_zeros = Polynomial::new(coefficients);

        prop_assert_eq!(encoded, poly_w_leading_zeros.encode());
    }

    #[proptest]
    fn leading_zero_coefficients_have_no_effect_on_encoding_poly_xfe_pbt(
        polynomial: Polynomial<'static, XFieldElement>,
        #[strategy(0usize..30)] num_leading_zeros: usize,
    ) {
        let encoded = polynomial.encode();

        let mut coefficients = polynomial.into_coefficients();
        coefficients.extend(vec![XFieldElement::ZERO; num_leading_zeros]);
        let poly_w_leading_zeros = Polynomial::new(coefficients);

        prop_assert_eq!(encoded, poly_w_leading_zeros.encode());
    }

    fn disallow_trailing_zeros_in_poly_encoding_prop<FF>(
        polynomial: Polynomial<FF>,
        leading_coefficient: FF,
        num_leading_zeros: usize,
    ) -> TestCaseResult
    where
        FF: FiniteField + BFieldCodec + 'static,
    {
        let mut polynomial_coefficients = polynomial.into_coefficients();
        polynomial_coefficients.push(leading_coefficient);
        let actual_polynomial = Polynomial::new(polynomial_coefficients.clone());
        let vec_encoding = actual_polynomial.coefficients().to_vec().encode();
        let poly_encoding = actual_polynomial.encode();
        prop_assert_eq!(
            [bfe_vec!(vec_encoding.len()), vec_encoding].concat(),
            poly_encoding,
            "This test expects similarity of Vec and Polynomial encoding"
        );

        polynomial_coefficients.extend(vec![FF::ZERO; num_leading_zeros]);
        let coefficient_encoding = polynomial_coefficients.encode();
        let poly_encoding_with_leading_zeros =
            [bfe_vec!(coefficient_encoding.len()), coefficient_encoding].concat();
        let decoding_result = Polynomial::<FF>::decode(&poly_encoding_with_leading_zeros);
        prop_assert!(matches!(
            decoding_result.unwrap_err(),
            PolynomialBFieldCodecError::TrailingZerosInPolynomialEncoding
        ));

        Ok(())
    }

    #[proptest]
    fn disallow_trailing_zeros_in_poly_encoding_bfe(
        polynomial: Polynomial<'static, BFieldElement>,
        #[filter(!#leading_coefficient.is_zero())] leading_coefficient: BFieldElement,
        #[strategy(1usize..30)] num_leading_zeros: usize,
    ) {
        disallow_trailing_zeros_in_poly_encoding_prop(
            polynomial,
            leading_coefficient,
            num_leading_zeros,
        )?
    }

    #[proptest]
    fn disallow_trailing_zeros_in_poly_encoding_xfe(
        polynomial: Polynomial<'static, XFieldElement>,
        #[filter(!#leading_coefficient.is_zero())] leading_coefficient: XFieldElement,
        #[strategy(1usize..30)] num_leading_zeros: usize,
    ) {
        disallow_trailing_zeros_in_poly_encoding_prop(
            polynomial,
            leading_coefficient,
            num_leading_zeros,
        )?
    }

    /// Depending on the test helper [`BFieldCodecPropertyTestData`] in the bfieldcodec_derive crate
    /// would introduce an almost-cyclic dependency.[^1] This would make publishing to crates.io
    /// quite difficult. Hence, integration tests in the bfieldcodec_derive crate are also off the
    /// table. Consequently, we test the derive macro here.
    ///
    /// See also: https://github.com/rust-lang/cargo/issues/8379#issuecomment-1261970561
    ///
    /// [^1]: almost-cyclic because the dependency would be a dev-dependency
    pub mod derive {
        use arbitrary::Arbitrary;
        use num_traits::Zero;

        use super::*;
        use crate::math::x_field_element::XFieldElement;
        use crate::tip5::Digest;
        use crate::tip5::Tip5;
        use crate::util_types::mmr::mmr_accumulator::MmrAccumulator;
        use crate::util_types::mmr::mmr_membership_proof::MmrMembershipProof;

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct UnitStruct;

        test_case! { fn unit_struct for UnitStruct: Some(0) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithUnitStruct {
            a: UnitStruct,
        }

        test_case! { fn struct_with_unit_struct for StructWithUnitStruct: Some(0) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithUnitStruct {
            A(UnitStruct),
            B,
        }

        test_case! { fn enum_with_unit_struct for EnumWithUnitStruct: Some(1) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructA {
            field_a: u64,
            field_b: u64,
            field_c: u64,
        }

        test_case! { fn struct_a for DeriveTestStructA: Some(6) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructB(u128);

        test_case! { fn struct_b for DeriveTestStructB: Some(4) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructC(u128, u64, u32);

        test_case! { fn struct_c for DeriveTestStructC: Some(7) }

        // Struct containing Vec<T> where T is BFieldCodec
        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructD(Vec<u128>);

        test_case! { fn struct_d for DeriveTestStructD: None }

        // Structs containing Polynomial<T> where T is BFieldCodec
        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructWithBfePolynomialField(Polynomial<'static, BFieldElement>);

        test_case! {
            fn struct_with_bfe_poly_field for DeriveTestStructWithBfePolynomialField: None
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructWithXfePolynomialField(Polynomial<'static, XFieldElement>);

        test_case! {
            fn struct_with_xfe_poly_field for DeriveTestStructWithXfePolynomialField: None
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithVariantWithPolyField {
            Bee,
            BooBoo(Polynomial<'static, BFieldElement>),
            Bop(Polynomial<'static, XFieldElement>),
        }

        test_case! { fn enum_with_variant_with_poly_field for EnumWithVariantWithPolyField: None }
        neg_test_case! { fn enum_with_variant_with_poly_field_neg for EnumWithVariantWithPolyField }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructE(Vec<u128>, u128, u64, Vec<bool>, u32, Vec<u128>);

        test_case! { fn struct_e for DeriveTestStructE: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DeriveTestStructF {
            field_a: Vec<u64>,
            field_b: bool,
            field_c: u32,
            field_d: Vec<bool>,
            field_e: Vec<BFieldElement>,
            field_f: Vec<BFieldElement>,
        }

        test_case! { fn struct_f for DeriveTestStructF: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct WithPhantomData<FF: FiniteField> {
            a_field: u128,
            #[bfield_codec(ignore)]
            _phantom_data: PhantomData<FF>,
            another_field: Vec<u64>,
        }

        test_case! { fn with_phantom_data for WithPhantomData<BFieldElement>: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct WithNestedPhantomData<FF: FiniteField> {
            a_field: u128,
            #[bfield_codec(ignore)]
            _phantom_data: PhantomData<FF>,
            another_field: Vec<u64>,
            a_third_field: Vec<WithPhantomData<FF>>,
            a_fourth_field: WithPhantomData<BFieldElement>,
        }

        test_case! { fn with_nested_phantom_data for WithNestedPhantomData<XFieldElement>: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct WithNestedVec {
            a_field: Vec<Vec<u64>>,
        }

        test_case! { fn with_nested_vec for WithNestedVec: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct EmptyStruct {}

        test_case! { fn empty_struct for EmptyStruct: Some(0) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithEmptyStruct {
            a: EmptyStruct,
        }

        test_case! { fn struct_with_empty_struct for StructWithEmptyStruct: Some(0) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithTwoEmptyStructs {
            a: EmptyStruct,
            b: EmptyStruct,
        }

        test_case! { fn struct_with_two_empty_structs for StructWithTwoEmptyStructs: Some(0) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct BigStructWithEmptyStructs {
            a: EmptyStruct,
            b: EmptyStruct,
            c: StructWithTwoEmptyStructs,
            d: StructWithEmptyStruct,
            e: EmptyStruct,
        }

        test_case! { fn big_struct_with_empty_structs for BigStructWithEmptyStructs: Some(0) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithEmptyStruct {
            A(EmptyStruct),
            B,
            C(EmptyStruct),
        }

        test_case! { fn enum_with_empty_struct for EnumWithEmptyStruct: Some(1) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct MuchNesting {
            a: Vec<Vec<Vec<Vec<BFieldElement>>>>,
            #[bfield_codec(ignore)]
            b: PhantomData<Tip5>,
            #[bfield_codec(ignore)]
            c: PhantomData<Tip5>,
        }

        test_case! { fn much_nesting for MuchNesting: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct SmallArrayStructUnnamedFields([u128; 1]);

        test_case! {
            fn small_array_struct_unnamed_fields for SmallArrayStructUnnamedFields: Some(4)
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct SmallArrayStructNamedFields {
            a: [u128; 1],
        }

        test_case! { fn small_array_struct_named_fields for SmallArrayStructNamedFields: Some(4) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct BigArrayStructUnnamedFields([u128; 150]);

        test_case! {
            fn big_array_struct_unnamed_fields for BigArrayStructUnnamedFields: Some(4 * 150)
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct BigArrayStructNamedFields {
            a: [XFieldElement; 150],
            b: XFieldElement,
            c: u64,
        }

        test_case! {
            fn big_array_struct_named_fields for BigArrayStructNamedFields: Some(3 * 150 + 3 + 2)
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct ArrayStructDynamicallySizedElementsUnnamedFields([Vec<u128>; 7]);

        test_case! {
            fn array_struct_dyn_sized_elements_unnamed_fields
            for ArrayStructDynamicallySizedElementsUnnamedFields: None
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct ArrayStructDynamicallySizedElementsNamedFields {
            a: [Vec<u128>; 7],
        }

        test_case! {
            fn array_struct_dyn_sized_elements_named_fields
            for ArrayStructDynamicallySizedElementsNamedFields: None
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithTupleField {
            a: (Digest, Vec<Digest>),
        }

        test_case! { fn struct_with_tuple_field for StructWithTupleField: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithTupleFieldTwoElements {
            a: ([Digest; 2], XFieldElement),
        }

        test_case! {
            fn struct_with_tuple_field_two_elements for StructWithTupleFieldTwoElements: Some(13)
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithNestedTupleField {
            a: (([Digest; 2], Vec<XFieldElement>), (XFieldElement, u64)),
        }

        test_case! { fn struct_with_nested_tuple_field for StructWithNestedTupleField: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct MsMembershipProof {
            sender_randomness: Digest,
            receiver_preimage: Digest,
            auth_path_aocl: MmrMembershipProof,
        }

        test_case! { fn ms_membership_proof for MsMembershipProof: None }
        test_case! { fn mmr_accumulator for MmrAccumulator: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct UnsupportedFields {
            a: u64,
            #[bfield_codec(ignore)]
            b: usize,
        }

        #[proptest]
        fn unsupported_fields_can_be_ignored_test(#[strategy(arb())] my_struct: UnsupportedFields) {
            let encoded = my_struct.encode();
            let decoded = UnsupportedFields::decode(&encoded).unwrap();
            prop_assert_eq!(my_struct.a, decoded.a);
            prop_assert_eq!(usize::default(), decoded.b);
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct OneFixedLenField {
            some_digest: Digest,
        }

        test_case! { fn one_fixed_len_field for OneFixedLenField: Some(5) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        pub struct TwoFixedLenFields {
            pub some_digest: Digest,
            pub some_u64: u64,
        }

        test_case! { fn two_fixed_len_fields for TwoFixedLenFields: Some(7) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        pub struct TwoFixedLenUnnamedFields(Digest, u64);

        test_case! { fn two_fixed_len_unnamed_fields for TwoFixedLenUnnamedFields: Some(7) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        pub struct FixAndVarLenFields {
            pub some_digest: Digest,
            pub some_vec: Vec<u64>,
        }

        test_case! { fn fix_and_var_len_fields for FixAndVarLenFields: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct FixAndVarLenUnnamedFields(Digest, Vec<u64>);

        test_case! { fn fix_and_var_len_unnamed_fields for FixAndVarLenUnnamedFields: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
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

        test_case! { fn quite_a_few_fix_and_var_len_fields for QuiteAFewFixAndVarLenFields: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct SimpleStructA {
            a: BFieldElement,
            b: XFieldElement,
        }

        test_case! { fn simple_struct_a for SimpleStructA: Some(4) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct SimpleStructB(u32, u64);

        test_case! { fn simple_struct_b for SimpleStructB: Some(3) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithBox {
            a: Box<u64>,
        }

        test_case! { fn struct_with_box for StructWithBox: Some(2) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct TupleStructWithBox(Box<u64>);

        test_case! { fn tuple_struct_with_box for TupleStructWithBox: Some(2) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct StructWithBoxContainingStructWithBox {
            a: Box<StructWithBox>,
            b: Box<TupleStructWithBox>,
        }

        test_case! {
            fn struct_with_box_containing_struct_with_box
            for StructWithBoxContainingStructWithBox: Some(4)
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum SimpleEnum {
            A,
            B(u32),
            C(BFieldElement, u32, u32),
        }

        test_case! { fn simple_enum for SimpleEnum: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum ComplexEnum {
            A,
            B(u32),
            C(BFieldElement, XFieldElement, u32),
            D(Vec<BFieldElement>, Digest),
            E(Option<bool>),
        }

        test_case! { fn complex_enum for ComplexEnum: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithUniformDataSize {
            A(Digest),
            B(Digest),
            C(Digest),
        }

        test_case! { fn enum_with_uniform_data_size for EnumWithUniformDataSize: Some(6) }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithIrregularDataSize {
            A(u32),
            B,
            C(Digest),
            D(Vec<XFieldElement>),
        }

        test_case! { fn enum_with_irregular_data_size for EnumWithIrregularDataSize: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithGenerics<I: Into<u64>> {
            A(I),
            B(I, I),
        }

        test_case! { fn enum_with_generics for EnumWithGenerics<u32>: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithGenericsAndWhereClause<I: Into<u64>>
        where
            I: Debug + Copy + Eq,
        {
            A,
            B(I),
            C(I, I),
        }

        test_case! {
            fn enum_with_generics_and_where_clause for EnumWithGenericsAndWhereClause<u32>: None
        }

        #[test]
        fn enums_bfield_codec_discriminant_can_be_accessed() {
            let a = EnumWithGenericsAndWhereClause::<u32>::A;
            let b = EnumWithGenericsAndWhereClause::<u32>::B(1);
            let c = EnumWithGenericsAndWhereClause::<u32>::C(1, 2);
            assert_eq!(0, a.bfield_codec_discriminant());
            assert_eq!(1, b.bfield_codec_discriminant());
            assert_eq!(2, c.bfield_codec_discriminant());
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithBoxedVariant {
            A(Box<u64>),
            B(Box<u64>, Box<u64>),
        }

        test_case! { fn enum_with_boxed_variant for EnumWithBoxedVariant: None }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        enum EnumWithBoxedVariantAndBoxedStruct {
            A(Box<StructWithBox>),
            B(Box<StructWithBox>, Box<TupleStructWithBox>),
        }

        test_case! {
            fn enum_with_boxed_variant_and_boxed_struct for EnumWithBoxedVariantAndBoxedStruct: None
        }

        #[derive(Debug, Clone, PartialEq, Eq, BFieldCodec, Arbitrary)]
        struct DummyPolynomial<T: FiniteField + BFieldCodec> {
            coefficients: Vec<T>,
        }

        #[proptest]
        fn manual_poly_encoding_implementation_is_consistent_with_derived_bfe(
            #[strategy(arb())] coefficients: Vec<BFieldElement>,
        ) {
            manual_poly_encoding_implementation_is_consistent_with_derived(coefficients)?;
        }

        #[proptest]
        fn manual_poly_encoding_implementation_is_consistent_with_derived_xfe(
            #[strategy(arb())] coefficients: Vec<XFieldElement>,
        ) {
            manual_poly_encoding_implementation_is_consistent_with_derived(coefficients)?;
        }

        fn manual_poly_encoding_implementation_is_consistent_with_derived<FF>(
            coefficients: Vec<FF>,
        ) -> TestCaseResult
        where
            FF: FiniteField + BFieldCodec + 'static,
        {
            if coefficients.last().is_some_and(Zero::is_zero) {
                let rsn = "`DummyPolynomial::encode` only works for non-zero leading coefficients";
                return Err(TestCaseError::Reject(rsn.into()));
            }

            let polynomial = Polynomial::new(coefficients.clone());
            let dummy_polynomial = DummyPolynomial { coefficients };
            prop_assert_eq!(dummy_polynomial.encode(), polynomial.encode());
            Ok(())
        }
    }
}
