use crate::shared_math::b_field_element::BFieldElement;
use crate::util_types::simple_hasher::Hasher;
use std::error::Error;
use std::fmt::Display;
use std::marker::PhantomData;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ProofStream<Item, H> {
    items: Vec<(Item, usize)>,
    items_index: usize,
    transcript: Vec<BFieldElement>,
    transcript_index: usize,
    _hasher: PhantomData<H>,
}

#[derive(Debug, Clone)]
pub struct ProofStreamError {
    pub message: String,
}

impl ProofStreamError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }

    pub fn boxed(message: &str) -> Box<dyn Error> {
        Box::new(Self::new(message))
    }
}

impl Display for ProofStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.message)
    }
}

impl Error for ProofStreamError {}

impl<Item, H> ProofStream<Item, H>
where
    Item: IntoIterator<Item = BFieldElement> + Clone,
    H: Hasher<Digest = Vec<BFieldElement>>,
{
    pub fn default() -> Self {
        ProofStream {
            items: vec![],
            items_index: 0,
            transcript: vec![],
            transcript_index: 0,
            _hasher: PhantomData,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn transcript_length(&self) -> usize {
        self.transcript.len()
    }

    pub fn enqueue(&mut self, item: &Item) {
        let mut elems: Vec<BFieldElement> = item.clone().into_iter().collect();
        self.items.push((item.clone(), elems.len()));
        self.transcript.append(&mut elems);
    }

    pub fn dequeue(&mut self) -> Result<Item, Box<dyn Error>> {
        let (item, elems_len) = self
            .items
            .get(self.items_index)
            .ok_or_else(|| ProofStreamError::boxed("Could not dequeue, queue empty"))?;

        self.items_index += 1;
        self.transcript_index += elems_len;
        Ok(item.clone())
    }

    pub fn prover_fiat_shamir(&self) -> H::Digest {
        let hasher = H::new();
        hasher.fiat_shamir(&self.transcript)
    }

    pub fn verifier_fiat_shamir(&self) -> H::Digest {
        let hasher = H::new();
        hasher.fiat_shamir(&self.transcript[0..self.transcript_index])
    }
}

#[cfg(test)]
mod proof_stream_typed_tests {
    use itertools::Itertools;

    use super::*;
    use crate::shared_math::x_field_element::XFieldElement;
    use crate::util_types::simple_hasher::RescuePrimeProduction;

    #[derive(Clone, Debug, PartialEq)]
    enum TestItem {
        ManyB(Vec<BFieldElement>),
        ManyX(Vec<XFieldElement>),
    }

    impl TestItem {
        pub fn as_bs(&self) -> Option<Vec<BFieldElement>> {
            match self {
                Self::ManyB(bs) => Some(bs.clone()),
                _ => None,
            }
        }

        pub fn as_xs(&self) -> Option<Vec<XFieldElement>> {
            match self {
                Self::ManyX(xs) => Some(xs.clone()),
                _ => None,
            }
        }
    }

    impl IntoIterator for TestItem {
        type Item = BFieldElement;

        type IntoIter = std::vec::IntoIter<BFieldElement>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                TestItem::ManyB(bs) => bs.into_iter(),
                TestItem::ManyX(xs) => xs
                    .into_iter()
                    .map(|x| x.coefficients.to_vec())
                    .concat()
                    .into_iter(),
            }
        }
    }

    #[test]
    fn enqueue_dequeue_test() {
        let mut proof_stream = ProofStream::<TestItem, RescuePrimeProduction>::default();
        let ps: &mut ProofStream<TestItem, RescuePrimeProduction> = &mut proof_stream;

        // Empty

        assert!(ps.dequeue().is_err(), "cannot dequeue empty");

        // B

        let b_one = BFieldElement::ring_one();
        let bs_expected = vec![b_one; 3];
        let item_1 = TestItem::ManyB(bs_expected.clone());
        ps.enqueue(&item_1);

        let item_1_option = ps.dequeue();
        assert!(item_1_option.is_ok(), "item 1 exists in queue");

        let item_1_actual: TestItem = item_1_option.unwrap();
        assert!(item_1_actual.as_xs().is_none(), "wrong type of item 1");
        let bs_option: Option<Vec<BFieldElement>> = item_1_actual.as_bs();
        assert!(bs_option.is_some(), "item 1 decodes to the right type");

        let bs_actual: Vec<BFieldElement> = bs_option.unwrap();
        assert_eq!(bs_expected, bs_actual, "enqueue/dequeue item 2");

        // Empty

        assert!(ps.dequeue().is_err(), "queue has become empty");

        // X

        let x_one = XFieldElement::ring_one();

        let xs_expected = vec![x_one; 3];
        let item_2 = TestItem::ManyX(xs_expected.clone());
        ps.enqueue(&item_2);

        let item_2_option = ps.dequeue();
        assert!(item_2_option.is_ok(), "item 2 exists in queue");

        let item_2_actual: TestItem = item_2_option.unwrap();
        assert!(item_2_actual.as_bs().is_none(), "wrong type of item 2");
        let xs_option: Option<Vec<XFieldElement>> = item_2_actual.as_xs();
        assert!(xs_option.is_some(), "item 2 decodes to the right type");

        let xs_actual: Vec<XFieldElement> = xs_option.unwrap();
        assert_eq!(xs_expected, xs_actual, "enqueue/dequeue item 2");
    }

    // Property: prover_fiat_shamir() is equivalent to verifier_fiat_shamir() when the entire stream has been read.
    #[test]
    fn prover_verifier_fiat_shamir_test() {
        let mut proof_stream = ProofStream::<TestItem, RescuePrimeProduction>::default();
        let ps: &mut ProofStream<TestItem, RescuePrimeProduction> = &mut proof_stream;

        let hasher = RescuePrimeProduction::new();
        let digest_1 = hasher.hash(&BFieldElement::ring_one());
        ps.enqueue(&TestItem::ManyB(digest_1));
        let _result = ps.dequeue();

        assert_eq!(
            ps.prover_fiat_shamir(),
            ps.verifier_fiat_shamir(),
            "prover_fiat_shamir() and verifier_fiat_shamir() are equivalent when the entire stream is read"
        );

        let digest_2 = hasher.hash(&BFieldElement::ring_one());
        ps.enqueue(&TestItem::ManyB(digest_2));

        assert_ne!(
            ps.prover_fiat_shamir(),
            ps.verifier_fiat_shamir(),
            "prover_fiat_shamir() and verifier_fiat_shamir() are different when the stream isn't fully read"
        );

        let _result = ps.dequeue();

        assert_eq!(
            ps.prover_fiat_shamir(),
            ps.verifier_fiat_shamir(),
            "prover_fiat_shamir() and verifier_fiat_shamir() are equivalent when the entire stream is read again",
        );
    }
}
