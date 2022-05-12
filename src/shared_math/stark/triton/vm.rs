use super::instruction::{Instruction, Program};
use super::state::VMState;
use crate::shared_math::rescue_prime_xlix;
use std::error::Error;

#[allow(clippy::needless_lifetimes)]
pub fn run<'pgm>(program: &'pgm Program) -> Result<Vec<VMState<'pgm>>, Box<dyn Error>> {
    let mut rng = rand::thread_rng();
    let rescue_prime = rescue_prime_xlix::neptune_params();
    let mut stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut trace = vec![VMState::new(&program.instructions)];
    while !trace.last().unwrap().is_final() {
        let derp1 = trace.last().unwrap();
        let derp2 = derp1.step(&mut rng, &rescue_prime, &mut stdin, &mut stdout);
        if derp2.is_err() {
            for x in trace.iter() {
                println!("{:?}", x);
            }
        }
        trace.push(derp2?);
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
        let instructions = vec![push(2), push(2), Add];
        let program = Program { instructions };
        let _empty_run = run(&program);
    }
}
