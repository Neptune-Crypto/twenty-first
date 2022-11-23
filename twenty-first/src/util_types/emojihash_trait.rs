use itertools::Itertools;

pub const EMOJI_PER_ELEMENT: usize = 3;

pub trait Emojihash {
    fn emojihash(&self) -> String;
}

impl<T: Emojihash> Emojihash for &[T] {
    fn emojihash(&self) -> String {
        let mut emojis = self.iter().map(|elem: &T| elem.emojihash());
        format!("[{}]", emojis.join("|"))
    }
}

impl<T: Emojihash, const N: usize> Emojihash for &[T; N] {
    fn emojihash(&self) -> String {
        let slef: &[T] = self.as_ref();
        slef.emojihash()
    }
}

impl<T: Emojihash> Emojihash for &Vec<T> {
    fn emojihash(&self) -> String {
        let slef: &[T] = self.as_ref();
        slef.emojihash()
    }
}

impl<T: Emojihash, const N: usize> Emojihash for [T; N] {
    fn emojihash(&self) -> String {
        let slef: &[T] = self.as_ref();
        slef.emojihash()
    }
}

impl<T: Emojihash> Emojihash for Vec<T> {
    fn emojihash(&self) -> String {
        let slef: &[T] = self.as_ref();
        slef.emojihash()
    }
}
