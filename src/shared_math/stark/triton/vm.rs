use super::instruction::Instruction;
use super::state::VMState;
use crate::shared_math::rescue_prime_xlix;
use std::error::Error;

#[allow(clippy::needless_lifetimes)]
pub fn run<'pgm>(program: &'pgm [Instruction]) -> Result<Vec<VMState<'pgm>>, Box<dyn Error>> {
    let mut rng = rand::thread_rng();
    let rescue_prime = rescue_prime_xlix::neptune_params();
    let mut stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut trace = vec![VMState::new(program)];
    while !trace.last().unwrap().is_final() {
        trace.push(
            trace
                .last()
                .unwrap()
                .step(&mut rng, &rescue_prime, &mut stdin, &mut stdout)?,
        );
    }

    Ok(trace)
}

#[cfg(test)]
mod triton_vm_tests {
    use super::super::instruction::push;
    use super::Instruction::*;
    use super::*;

    #[test]
    fn vm_run_test() {
        let program = vec![push(2), push(2), Add];
        let empty_run = run(&vec![]);
    }
}
