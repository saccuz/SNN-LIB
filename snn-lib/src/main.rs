mod snn;

use crate::snn::faults::{
    Component, FaultConfiguration, FaultType, InnerComponent, OuterComponent,
};
use crate::snn::lif::{LifNeuron, LifNeuronParameters, ResetMode};
use crate::snn::matrix::{Input, Matrix};
use crate::snn::snn::Snn;

fn main() {
    // 1 - First way to set specific neuron parameters, different for each layer
    let mut arr = Vec::<LifNeuronParameters>::new();
    for _ in 0..3 {
        arr.push(LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.0,
            r_type: ResetMode::Zero,
            tau: 0.5,
        })
    }

    let mut snn = Snn::<LifNeuron>::new(6, vec![10, 5, 3], vec![true, true, true], Some(arr));

    // 2 - Alternative way to set the same parameters for the whole network
    let parameters_for_lif = LifNeuronParameters {
        v_rest: 0.0,
        v_th: 0.8,
        r_type: ResetMode::Zero,
        tau: 0.35,
    };
    snn.set_parameters(&parameters_for_lif);

    //let input_matrix =vec![
    //    vec![0, 1, 0, 0, 0, 0],
    //    vec![1, 0, 1, 0, 1, 0],
    //    vec![0, 1, 1, 1, 1, 1],
    //    vec![1, 0, 0, 1, 0, 0],
    //    vec![0, 0, 1, 0, 1, 0],
    //    vec![0, 1, 1, 1, 1, 1],
    //    vec![1, 0, 0, 1, 0, 0],
    //    vec![0, 0, 1, 0, 1, 0],
    //    vec![0, 1, 1, 1, 1, 1],
    //    vec![1, 1, 0, 0, 1, 1]
    //];

    let input_matrix = Input::random(17, 6, false);

    //println!("{}", input_matrix);

    //println!("{:?}", snn.forward(input_matrix, Option::None));

    println!("{}", snn);

    // Fault injection
    let fault_configuration = FaultConfiguration::new(
        vec![
            Component::Inside(InnerComponent::Adder),
            Component::Outside(OuterComponent::Connections),
            Component::Outside(OuterComponent::Weights),
        ],
        FaultType::StuckAtZero,
        100,
    );

    snn.emulate_fault(&input_matrix, &fault_configuration);
}
