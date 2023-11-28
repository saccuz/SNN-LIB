mod snn;

//use rand::Rng;
use crate::snn::lif::NeuronType;
use snn::lif;

fn main() {
    let mut snn = lif::Snn::new(6,vec![10,5,3], vec![true, true, true],NeuronType::LifNeuron);
    //println!("{:?}", );
    for i in 0..30 {
        println!("{:?}", snn.forward(vec![0,1,0,0,0,0]));
    }
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
