//Make snn from snn.rs usable in this file

use crate::snn;
use snn::neuron;

struct Snn {
    layers : Vec<u8>
}

impl Snn {
    fn new<N>(layers : Vec<N>)  -> Self where N : Neuron {
        Snn { layers }
    }


}

pub enum ResetMode {
    Zero,
    RestingPotential(f64),
    Subthreshold
}

//enum OutputWeight {
//    None,
//    Weight(f64)
//}

struct LifNeuron {
    v_mem : f64,
    v_rest : f64,
    v_th : f64,
    r_type : ResetMode,
    t_s_last : u64,  //??? serve? Forse sì, salvi in t_s_last l'ultimo "istante" (forward) in cui si è ricevuto un impulso
    tau : f64,
    // errore => w_out : OutputWeight   //output weight, if None is a terminal neuron
    // ...ogni collegamento ha il suo peso...non si può fare così, come gestiamo i vari collegamenti?
}

impl LifNeuron {
    fn new(v_mem : f64, v_rest : f64, v_th : f64, r_type : ResetMode, tau : f64) -> Self {
        LifNeuron { v_mem, v_rest, v_th, r_type, t_s_last : 0, tau }
    }
}

impl neuron::Neuron for LifNeuron {
    // Creates a new LifNeuron
    // (t_s_last is set to 0 by default at the beginning, no previous impulse received from the beginning of the neuron existence)
    fn new(v_mem : f64, v_rest : f64, v_th : f64, r_type : ResetMode, v_rest : f64, tau : f64) -> Self {
        LifNeuron { v_mem, v_rest, v_th, r_type, v_rest, 0, tau }
    }

    // implements the forward pass of the snn
    fn forward(&mut self, input : &[f64]) -> f64 {
        
        //TODO: aggiungere calcolo della membrana e aggiornare parametri

        spike = (self.v_mem > self.v_th) as u8;   //if v_mem>v_th then spike=1 else spike=0
        
        if spike == 1 {
            self.v_mem = match self.r_type {
                ResetMode::Zero => 0.0,
                ResetMode::RestingPotential(v) => v,
                ResetMode::Subthreshold => self.v_mem - self.v_th
            }
        }

        spike as f64
    }
}
