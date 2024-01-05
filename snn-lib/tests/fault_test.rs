#[cfg(test)]
mod fault_test {
    use snn_lib::snn::faults::{Component, FaultConfiguration, FaultType, OuterComponent};
    use snn_lib::snn::lif::LifSpecificComponent;

    #[test]
    fn create_fault_conf() {
        let f_c = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
            ],
            8,
            FaultType::TransientBitFlip,
            100,
        );

        assert_eq!(f_c.components_contain_inner_weights(), false);
    }

    #[test]
    #[should_panic(expected = "Invalid param, expected number of buses at least 1, but got 0")]
    fn create_fault_conf_panic_bus() {
        let _ = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
            ],
            0,
            FaultType::TransientBitFlip,
            100,
        );
    }

    #[test]
    #[should_panic(
        expected = "Invalid param, expected number of occurrences at least 1, but got 0"
    )]
    fn create_fault_conf_panic_occurrences() {
        let _ = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Outside(OuterComponent::Weights),
            ],
            1,
            FaultType::TransientBitFlip,
            0,
        );
    }
}
