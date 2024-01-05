#[cfg(test)]
mod fault_test {
    use snn_lib::snn::faults::{FaultConfiguration, FaultType, Component, OuterComponent};
    use snn_lib::snn::lif::LifSpecificComponent;

    #[test]
    fn create_fault() {
        let fault_configuration = FaultConfiguration::new(
            vec![
                Component::Inside(LifSpecificComponent::Adder),
                Component::Inside(LifSpecificComponent::Divider),
                Component::Inside(LifSpecificComponent::Multiplier),
                Component::Inside(LifSpecificComponent::Comparator),
                Component::Inside(LifSpecificComponent::Membrane),
                Component::Inside(LifSpecificComponent::Rest),
                Component::Inside(LifSpecificComponent::Threshold),
                Component::Outside(OuterComponent::Weights),
                Component::Outside(OuterComponent::InnerWeights),
                Component::Outside(OuterComponent::Connections),
            ],
            8,
            //FaultType::StuckAtZero,
            //FaultType::StuckAtOne,
            FaultType::TransientBitFlip,
            100,
        );
    }

}