use anyhow::{bail, Result};

pub fn my_library_function() -> Result<()> {
    my_private_function(42)?;
    Ok(())
}

fn my_private_function(n: i32) -> Result<()> {
    if n > 1000 {
        bail!("`n` cannot be larger than 1000!")
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal() {
        assert!(my_private_function(42).is_ok());
        assert!(my_private_function(9001).is_err());
    }
}
