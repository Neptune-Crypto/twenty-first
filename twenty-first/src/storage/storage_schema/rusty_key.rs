#[derive(Clone, PartialEq, Eq)]
pub struct RustyKey(pub Vec<u8>);
impl From<u8> for RustyKey {
    #[inline]
    fn from(value: u8) -> Self {
        Self([value].to_vec())
    }
}
impl From<(RustyKey, RustyKey)> for RustyKey {
    #[inline]
    fn from(value: (RustyKey, RustyKey)) -> Self {
        let v0 = value.0 .0;
        let v1 = value.1 .0;
        RustyKey([v0, v1].concat())
    }
}
impl From<u64> for RustyKey {
    #[inline]
    fn from(value: u64) -> Self {
        RustyKey(value.to_be_bytes().to_vec())
    }
}
