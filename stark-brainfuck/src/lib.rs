pub mod evaluation_argument;
pub mod instruction_table;
pub mod io_table;
pub mod memory_table;
pub mod permutation_argument;
pub mod processor_table;
pub mod stark;
pub mod stark_proof_stream;
pub mod table;
pub mod table_collection;
pub mod vm;
pub mod xfri;

use std::collections::HashMap;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

pub fn lift_coefficients_to_xfield(
    mpolynomial: &MPolynomial<BFieldElement>,
) -> MPolynomial<XFieldElement> {
    let mut new_coefficients: HashMap<Vec<u64>, XFieldElement> = HashMap::new();
    mpolynomial.coefficients.iter().for_each(|(key, value)| {
        new_coefficients.insert(key.to_owned(), value.lift());
    });

    MPolynomial {
        variable_count: mpolynomial.variable_count,
        coefficients: new_coefficients,
    }
}
