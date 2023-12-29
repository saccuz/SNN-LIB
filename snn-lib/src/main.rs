mod snn;

use crate::snn::faults::{Component, FaultConfiguration, FaultType, OuterComponent};
use crate::snn::lif::{LifNeuron, LifNeuronParameters, LifSpecificComponent, ResetMode};
use crate::snn::matrix::{Input, Matrix};
use crate::snn::generic_matrix::MatrixG;
use crate::snn::snn::Snn;

fn main() {

    // Set the seed for weights, inner weights and input matrix
    let seed = Some(21);


    // Configuring the Snn
    let n_inputs: usize = 10;
    let layers = vec![10,5,3];

    // Randomly creating an input matrix
    let input_matrix = Input::random(17, n_inputs , false, seed);
    //println!("{}", input_matrix);

    // TODO: Drop these lines, but first check the correctness of MatrixG
    //let inputttt = MatrixG::random(17, n_inputs, false, seed, 0, 1);
    //let weightsss = MatrixG::random(10, 5 , false, seed, 0.01, 0.99);
    //let inner_weightsss = MatrixG::random(5, 5 , true, seed, 0.01, 0.99);
    //println!("{}", inputttt);
    //println!("{}", weightsss);
    //println!("{}", inner_weightsss);

    // 1 - First way to set specific neuron parameters, different for each layer
    let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
    for _ in 0..layers.len() {
        neuron_parameters_per_layer.push(LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.0,
            r_type: ResetMode::SubThreshold,
            tau: 0.5,
        })
    }

    // TODO: drop these two mega-for when the correctness tests will be done
    // Setting personalized weights (all equal) - ONLY FOR DEBUG PURPOSES
    let mut personalized_weights= Vec::new();
    for (idx, l) in layers.iter().enumerate() {
        let mut v = Vec::new();
        for _ in 0..*l {
            if idx == 0 {
                v.push(vec![0.20; n_inputs]);
            }
            else {
                v.push(vec![0.20; layers[idx-1] as usize]);
            }
        }
        personalized_weights.push(v);
    }

    // Setting personalized inner weights (all equal) - ONLY FOR DEBUG PURPOSES
    let mut personalized_inner_weights= Vec::new();
    for l in layers.iter() {
        let mut v = Vec::new();
        for i in 0..*l {
            let mut x  = Vec::new();
            for j in 0..*l {
                if i == j {
                    x.push(0.0);
                }
                else {
                    x.push( -0.20);
                }
            }
            v.push(x)
        }
        personalized_inner_weights.push(Some(v));
    }

    // Snn creation
    let mut snn = Snn::<LifNeuron>::new(n_inputs as u32, layers, vec![true, true, true], Some(neuron_parameters_per_layer), Some(personalized_weights), Some(personalized_inner_weights), seed);

    // Fault injection
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

    // To start the n_occurrences faults emulations
    snn.emulate_fault(&input_matrix, &fault_configuration, 1);
}


// DO NOT DELETE THE FOLLOWING LINES => TODO: put important things in README and expalin

// To start the inference without faults
//println!("\The final result is: {:?}", snn.forward(&input_matrix, None));


// 2 - Alternative way to set the same parameters for the whole network (same neuron parameters for all layers)
/*let parameters_for_lif = LifNeuronParameters {
    v_rest: 0.0,
    v_th: 0.8,
    r_type: ResetMode::Zero,
    tau: 0.35,
};*/

// If None: parameters_for_lif are applied to all layers
/*snn.set_neuron_parameters(&parameters_for_lif, None);*/

// If Some([idx1, idx2, ...]): parameters_for_lif are applied to layers specified by the indexes (layer 0 and layer 2 in this case)
//snn.set_neuron_parameters(&parameters_for_lif, Some(vec![0,2]));



//If defined, at snn creation this personalized weights are set
/*let personalized_weights = vec![
    vec![vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20]],
    vec![vec![0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20]],
    vec![vec![0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20], vec![0.20,0.20,0.20,0.20,0.20]],
];*/
//Else, if None, random inner weights are set
//let personalized_inner_weights = None;

//If defined, at snn creation this personalized inner weights are set
/*let personalized_inner_weights = vec![
    Some(vec![vec![0.0,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20], vec![-0.20,0.0,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20], vec![-0.20,-0.20,0.0,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20], vec![-0.20,-0.20,-0.20,0.0,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20], vec![-0.20,-0.20,-0.20,-0.20,0.0,-0.20,-0.20,-0.20,-0.20,-0.20], vec![-0.20,-0.20,-0.20,-0.20,-0.20,0.0,-0.20,-0.20,-0.20,-0.20], vec![-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,0.0,-0.20,-0.20,-0.20], vec![-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,0.0,-0.20,-0.20], vec![-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,0.0,-0.20], vec![-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,0.0]]),
    Some(vec![vec![0.0,-0.20,-0.20,-0.20,-0.20], vec![-0.20,0.0,-0.20,-0.20,-0.20], vec![-0.20,-0.20,0.0,-0.20,-0.20], vec![-0.20,-0.20,-0.20,0.0,-0.20], vec![-0.20,-0.20,-0.20,-0.20,0.0]]),
    Some(vec![vec![0.0,-0.20,-0.20], vec![-0.20,0.0,-0.20], vec![-0.20,-0.20,0.0]]),
];*/
//Else, if None, random weights are set




/* Time statistics to compare parallelization vs non-parallelization
use std::time::{Instant, Duration};
let mut times = Vec::new();
let num_rep_for_statistics = 10;
for _ in 0..num_rep_for_statistics {
    let start = Instant::now();
    snn.emulate_fault(&input_matrix, &fault_configuration);
    times.push(start.elapsed());
}
println!("Mean time elapsed in expensive_function() is: {:?}", times.iter().sum::<Duration>()/num_rep_for_statistics);
*/

