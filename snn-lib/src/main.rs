mod snn;

use crate::snn::faults::{Component, FaultConfiguration, FaultType, OuterComponent};
use crate::snn::lif::{LifNeuron, LifNeuronParameters, LifSpecificComponent, ResetMode};
use crate::snn::matrix::{Input, Matrix};
use crate::snn::snn::Snn;
//use std::time::{Instant, Duration};

fn main() {
    // 1 - First way to set specific neuron parameters, different for each layer
    let mut arr = Vec::<LifNeuronParameters>::new();

    for _ in 0..3 {
        arr.push(LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.0,
            r_type: ResetMode::SubThreshold,
            tau: 0.5,
        })
    }

    //If defined, at snn creation the personalized inner weights are set
    let personalized_weights = vec![
        vec![vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03]],
        vec![vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]],
        vec![vec![0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03]],
    ];
    //Else, if None, random inner weights are set
    //let personalized_inner_weights = None;

    //If defined, at snn creation the personalized weights are set
    let personalized_inner_weights = vec![
        Some(vec![vec![0.0,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.0,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.0,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.0,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.0,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.0,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.0,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.0,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.0,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.0]]),
        Some(vec![vec![0.0,0.03,0.03,0.03,0.03], vec![0.03,0.0,0.03,0.03,0.03], vec![0.03,0.03,0.0,0.03,0.03], vec![0.03,0.03,0.03,0.0,0.03], vec![0.03,0.03,0.03,0.03,0.0]]),
        Some(vec![vec![0.0,0.03,0.03], vec![0.03,0.0,0.03], vec![0.03,0.03,0.0]]),
    ];
    /* This is the alternative if no inner weights are given for the layer 2 (NOTE THAT intra_conn in Snn::new should be consistent with this)
    let personalized_inner_weights = vec![
        Some(vec![vec![0.0,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.0,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.0,0.03,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.0,0.03,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.0,0.03,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.0,0.03,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.0,0.03,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.0,0.03,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.0,0.03], vec![0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.0]]),
        None,
        Some(vec![vec![0.0,0.03,0.03], vec![0.03,0.0,0.03], vec![0.03,0.03,0.0]]),
    ];
    */
    //Else, if None, random weights are set
    //let personalized_weights = None;

    //let mut snn = Snn::<LifNeuron>::new(6, vec![50, 40, 30, 20, 10, 5, 3], vec![true, true, true, true, true, true, true], Some(arr));
    let mut snn = Snn::<LifNeuron>::new(6, vec![10, 5, 3], vec![true, true, true], Some(arr), Some(personalized_weights), Some(personalized_inner_weights));
    //let mut snn = Snn::<LifNeuron>::new(6, vec![200, 190, 180, 170, 150, 140, 120, 100, 80, 70, 50, 40, 30, 20, 10, 5, 3], vec![true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true], Some(arr));

    // 2 - Alternative way to set the same parameters for the whole network (same neuron parameters for all layers)
    let parameters_for_lif = LifNeuronParameters {
        v_rest: 0.0,
        v_th: 0.8,
        r_type: ResetMode::Zero,
        tau: 0.35,
    };

    // If None: parameters_for_lif are applied to all layers,
    // If Some([idx1, idx2, ...]): parameters_for_lif are applied to index specified layers
    snn.set_neuron_parameters(&parameters_for_lif, None);
    //snn.set_neuron_parameters(&parameters_for_lif, Some(vec![0]));
    //snn.set_neuron_parameters(&parameters_for_lif, Some(vec![0,2]));

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
            Component::Outside(OuterComponent::InnerWeights),
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

