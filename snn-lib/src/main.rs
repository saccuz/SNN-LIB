mod snn;

//use rand::Rng;
use crate::snn::lif::NeuronType;
use snn::lif;

fn main() {
    let snn = lif::Snn::new(vec![3, 2, 2], vec![true, true, true],NeuronType::LifNeuron);
    //snn.set_parameters();


    //let mut rng = rand::thread_rng();
    //let mut weights = Vec::<Vec<f64>>::new();
    //println!("{}", rng.gen_range(0.01..1.0));

    //let mut n = neuron::LifNeuron::new(0.0, 0.0, 0.0, neuron::ResetMode::Zero, 0.0);
    //let mut input = [0.0, 0.0, 0.0, 0.0];
    //let mut output = 0.0;
    //
    //for i in 0..4 {
    //    input[i] = i as f64;
    //}
    //
    //output = n.forward(&input);
    //
    //println!("Output: {}", output);
}
