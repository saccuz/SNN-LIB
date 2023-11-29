mod snn;

use crate::snn::lif::{LifNeuron, LifNeuronParameters, ResetMode};
use crate::snn::snn::Snn;

fn main() {
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

    let parameters_for_lif = LifNeuronParameters {
        v_rest: 0.0,
        v_th: 0.8,
        r_type: ResetMode::Zero,
        tau: 0.35,
    };

    snn.set_parameters(&parameters_for_lif);

    for _ in 0..30 {
        println!("{:?}", snn.forward(vec![0, 1, 0, 0, 0, 0]));
    }

    println!("{}", snn);
    //snn.set_parameters();
}
