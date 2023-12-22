mod snn;

use crate::snn::faults::{Component, FaultConfiguration, FaultType, OuterComponent};
use crate::snn::lif::{LifNeuron, LifNeuronParameters, LifSpecificComponent, ResetMode};
use crate::snn::matrix::{Input, Matrix};
use crate::snn::snn::Snn;
//use std::time::{Instant, Duration};

fn main() {
    // 1 - First way to set specific neuron parameters, different for each layer
    let mut arr = Vec::<LifNeuronParameters>::new();

    for _ in 0..17 {
        arr.push(LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.0,
            r_type: ResetMode::SubThreshold,
            tau: 0.5,
        })
    }

    //let mut snn = Snn::<LifNeuron>::new(6, vec![50, 40, 30, 20, 10, 5, 3], vec![true, true, true, true, true, true, true], Some(arr));
    let mut snn = Snn::<LifNeuron>::new(6, vec![10, 5, 3], vec![true, true, true], Some(arr));
    //let mut snn = Snn::<LifNeuron>::new(6, vec![200, 190, 180, 170, 150, 140, 120, 100, 80, 70, 50, 40, 30, 20, 10, 5, 3], vec![true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true], Some(arr));

    // 2 - Alternative way to set the same parameters for the whole network
    let parameters_for_lif = LifNeuronParameters {
        v_rest: 0.0,
        v_th: 0.8,
        r_type: ResetMode::Zero,
        tau: 0.35,
    };
    snn.set_parameters(&parameters_for_lif);

    let input_matrix = Input::random(17, 6, false);
    //let input_matrix = <Input as Matrix>::from(vec![
    //    vec![0, 0, 0, 0, 0, 1],
    //    vec![0, 0, 0, 0, 1, 0],
    //    vec![0, 0, 0, 1, 0, 0],
    //    vec![0, 0, 1, 0, 0, 0],
    //    vec![0, 1, 0, 0, 0, 0],
    //    vec![1, 0, 0, 0, 0, 0],
    //    vec![1, 0, 0, 0, 0, 1],
    //    vec![1, 0, 0, 0, 1, 0],
    //    vec![1, 0, 0, 1, 0, 0],
    //    vec![1, 0, 1, 0, 0, 0],
    //]);

    println!("{}", snn);

    // Fault injection
    let fault_configuration = FaultConfiguration::new(
        vec![
            Component::Inside(LifSpecificComponent::Membrane),
            Component::Outside(OuterComponent::Connections),
            Component::Outside(OuterComponent::Weights),
        ],
        8,
        FaultType::StuckAtZero,
        100,
    );

    snn.emulate_fault(&input_matrix, &fault_configuration);
    //println!("\nSo the final result is: {:?}", snn.forward(&input_matrix, None));

    /*
    let mut times = Vec::new();
    let num_rep_for_statistics = 10;
    for _ in 0..num_rep_for_statistics {
        let start = Instant::now();
        snn.emulate_fault(&input_matrix, &fault_configuration);
        times.push(start.elapsed());
    }

    println!("Mean time elapsed in expensive_function() is: {:?}", times.iter().sum::<Duration>()/num_rep_for_statistics);
    */
}

