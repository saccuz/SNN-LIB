//Make snn from snn.rs usable in this file

use rand::Rng;
use crate::snn::neuron::Neuron;

#[derive(Clone,Copy)]
pub enum NeuronType {
    LifNeuron
}

struct Layer<N : Neuron> {
    id : String,
    neurons : Vec<N>,
    inputs : Vec<u8>,
    states : Vec<u8>,
    states_weights : Option<Vec<Vec<f64>>>,
    weights : Option<Vec<Vec<f64>>>,
}

impl<N> Layer<N> where N : Neuron {
    fn new(id : u32, neurons : u32, neuron_type: NeuronType, states_weights : Option<Vec<Vec<f64>>>, weights : Option<Vec<Vec<f64>>>)  -> Self {
        let mut neurons_vec = match neuron_type {
            NeuronType::LifNeuron => Vec::<LifNeuron>::new()
        };
        for i in 0..neurons {
            neurons_vec.push(LifNeuron::new( format!("{}-{}", id.to_string(), i.to_string()), i))
        }
        Layer { id : id.to_string(), neurons : neurons_vec, inputs : Vec::<u8>::new(), states : Vec::<u8>::new(), states_weights, weights }
    }
}

pub struct Snn<N> {
    layers : Vec<Layer<N>>
}

impl<N> Snn<N> where N : Neuron {
    // Generates random weights matrix
    fn random_weights(r : u32, c : u32, diag : bool) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::<Vec<f64>>::new();
        for row in 0..r {
            let mut row = Vec::<f64>::new();
            for column in 0..c {
                if diag && row == column {
                    row.push(0.0);
                }
                else {
                    row.push(rng.gen_range(0.01..1.0));
                }
            }
            weights.push(row);
        }
        weights
    }

    pub fn new(layers : Vec<u32>, neuron_type : NeuronType)  -> Self {
        let mut layers_vec = Vec::<Layer<N>>::new();

        layers_vec.push(Layer::new(0, layers[0], neuron_type, None, None));

        for (idx, l) in layers.iter().skip(1).enumerate() {
            layers_vec.push(Layer::new(*l, *l, neuron_type, Option::Some(Snn::random_weights(*l, *l, true)),
                                       Option::Some(Snn::random_weights(layers[idx], *l, false))));
        }
        Snn { layers : layers_vec }
    }


}

pub enum ResetMode {
    Zero,
    RestingPotential(f64),
    Subthreshold
}

pub struct LifNeuron {
    id : String,
    v_mem : f64,
    v_rest : f64,
    v_th : f64,
    r_type : ResetMode,
    t_s_last : u64,  //??? serve? Forse sì, salvi in t_s_last l'ultimo "istante" (forward) in cui si è ricevuto un impulso
    tau : f64,
    // errore => w_out : OutputWeight   //output weight, if None is a terminal snn
    // ...ogni collegamento ha il suo peso...non si può fare così, come gestiamo i vari collegamenti (sinapsi)?
}

impl LifNeuron {
    //TODO: uncomment this and make it right
    //fn new(v_mem : f64, v_rest : f64, v_th : f64, r_type : ResetMode, tau : f64) -> Self {
    //    LifNeuron { v_mem, v_rest, v_th, r_type, t_s_last : 0, tau }
    //}
    fn new(id : String) -> Self {
        LifNeuron { id,  v_mem : 0.0, v_rest : 0.0, v_th : 0.0, r_type: ResetMode::Zero, t_s_last : 0, tau : 0.0 }
    }
}

impl Neuron for LifNeuron {
    // Creates a new LifNeuron
    // (t_s_last is set to 0 by default at the beginning, no previous impulse received from the beginning of the snn existence)

    // implements the forward pass of the snn
    fn forward(&mut self, input : &[f64]) -> f64 {
        
        //TODO: aggiungere calcolo della membrana e aggiornare parametri

        let spike = (self.v_mem > self.v_th) as u8;   //if v_mem>v_th then spike=1 else spike=0
        
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
