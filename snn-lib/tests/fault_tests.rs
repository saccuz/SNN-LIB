#[cfg(test)]
mod fault_tests {
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

    #[test]
    fn generate_actual_fault() {
        let f_c = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
            ],
            8,
            FaultType::TransientBitFlip,
            100,
        );

        let layers_info = vec![(3, false)];
        let res = f_c.get_actual_fault(layers_info, 5, None);
        assert!(matches!(res.component, Component::Inside(LifSpecificComponent::Adder)));
        assert_eq!(res.layer_id, 0);
        assert!(res.neuron_id.0 < 3);
        assert!(matches!(res.fault_type, FaultType::TransientBitFlip));
        assert!(res.time_tbf.unwrap() < 5);
        assert!(res.bus.is_none());
        assert!(res.offset < 64);

        let f_c = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
            ],
            8,
            FaultType::StuckAtOne,
            100,
        );

        let layers_info = vec![(3, false)];
        let res = f_c.get_actual_fault(layers_info, 5, None);
        assert!(matches!(res.component, Component::Inside(LifSpecificComponent::Adder)));
        assert_eq!(res.layer_id, 0);
        assert!(res.neuron_id.0 < 3);
        assert!(matches!(res.fault_type, FaultType::StuckAtOne));
        assert!(res.time_tbf.is_none());
        assert!(res.bus.is_none());
        assert!(res.offset < 64);
    }

    #[test]
    #[should_panic(expected = "Invalid component for fault configurations: if Inner Weights is selected, be sure to initialize it in the Snn model.")]
    fn generate_actual_fault_panic_inner_weight() {
        let f_c = FaultConfiguration::<LifSpecificComponent>::new(
            vec![
                Component::Outside(OuterComponent::InnerWeights),
            ],
            8,
            FaultType::TransientBitFlip,
            100,
        );

        let layers_info = vec![(3, false)];
        let res = f_c.get_actual_fault(layers_info, 5, None);

    }

    #[test]
    #[should_panic(expected = "Invalid param, expected total time at least 1, but got 0")]
    fn generate_actual_fault_panic_if_time_zero() {
        let f_c = FaultConfiguration::<LifSpecificComponent>::new(
            vec![
                Component::Outside(OuterComponent::InnerWeights),
            ],
            8,
            FaultType::TransientBitFlip,
            100,
        );

        let layers_info = vec![(3, true)];
        let res = f_c.get_actual_fault(layers_info, 0, None);

    }
}
