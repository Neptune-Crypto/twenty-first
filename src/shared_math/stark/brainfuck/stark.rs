use crate::shared_math::{
    b_field_element::BFieldElement, fri::Fri, other::is_power_of_two,
    stark::brainfuck::processor_table::ProcessorTable, traits::GetPrimitiveRootOfUnity,
    x_field_element::XFieldElement,
};

pub struct Stark {
    trace_length: usize,
    program: Vec<BFieldElement>,
    input_symbols: Vec<BFieldElement>,
    output_symbols: Vec<BFieldElement>,
    expansion_factor: usize,
    security_level: usize,
    num_colinearity_checks: usize,
    num_randomizers: usize,
    // base_tables: [BaseTable; 5],
    // permutation_arguments: [PermuationArgument; 2],
    // evaluation_arguments: [EvaluationArgument; 3],
    max_degree: usize,
    fri: Fri<XFieldElement, blake3::Hasher>,
}

impl Stark {
    pub fn new(
        trace_length: usize,
        program: Vec<BFieldElement>,
        input_symbols: Vec<BFieldElement>,
        output_symbols: Vec<BFieldElement>,
    ) -> Self {
        let log_expansion_factor = 2; // TODO: For speed
        let expansion_factor: usize = 1 << log_expansion_factor;
        let security_level = 2; // TODO: Consider increasing this
        let num_colinearity_checks = security_level / log_expansion_factor;
        assert!(
            num_colinearity_checks > 0,
            "At least one colinearity check is required"
        );
        assert!(
            is_power_of_two(expansion_factor),
            "expansion factor must be power of two."
        );
        assert!(
            expansion_factor >= 4,
            "expansion factor must be at least 4."
        );

        let num_randomizers = 1;
        let order = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order)
            .0
            .unwrap();

        // instantiate table objects
        let processor_table = ProcessorTable::new(
            trace_length,
            num_randomizers,
            smooth_generator,
            order as usize,
        );

        todo!()
    }
}
