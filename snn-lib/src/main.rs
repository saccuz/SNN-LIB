mod snn;

use snn::{neuron, lif};


fn main() {

    let mut n = neuron::LifNeuron::new(0.0, 0.0, 0.0, neuron::ResetMode::Zero, 0.0);
    let mut input = [0.0, 0.0, 0.0, 0.0];
    let mut output = 0.0;

    for i in 0..4 {
        input[i] = i as f64;
    }

    output = n.forward(&input);

    println!("Output: {}", output);

}
