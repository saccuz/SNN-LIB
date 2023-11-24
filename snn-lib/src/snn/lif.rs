//Make snn from snn.rs usable in this file

use crate::snn::neuron::Neuron;
use rand::Rng;
//use crate::snn::lif::NeuronType::LifNeuron;
//use crate::snn::lif::NeuronType::LifNeuron;

#[derive(Clone, Copy)]
pub enum NeuronType {
    LifNeuron,
}

pub enum ResetMode {
    Zero,
    RestingPotential(f64),
    Subthreshold,
}

pub struct Snn {
    layers: Vec<Layer>,
}

impl Snn {
    // Generates random weights matrix
    fn random_weights(h: u32, w: u32, diag: bool) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::<Vec<f64>>::new();
        for r in 0..h {
            let mut row = Vec::<f64>::new();
            for c in 0..w {
                if diag && r == c {
                    row.push(0.0);
                } else {
                    row.push(rng.gen_range(0.01..1.0));
                }
            }
            weights.push(row);
        }
        weights
    }

    pub fn new(layers: Vec<u32>, neuron_type: NeuronType) -> Self {
        let mut layers_vec = Vec::<Layer>::new();

        layers_vec.push(Layer::new(0, layers[0], neuron_type, None, None));

        for (idx, l) in layers.iter().skip(1).enumerate() {
            layers_vec.push(Layer::new(
                idx+1,
                *l,
                neuron_type,
                Option::Some(Snn::random_weights(*l, *l, true)),
                Option::Some(Snn::random_weights(layers[idx], *l, false)),
            ));
        }
        Snn { layers: layers_vec }
    }
}

pub enum ResetMode {
    Zero,
    RestingPotential(f64),
    Subthreshold,
}

pub struct LifNeuron {
    id: String,
    v_mem: f64,
    v_rest: f64,
    v_th: f64,
    r_type: ResetMode,
    t_s_last: u64, //??? serve? Forse sì, salvi in t_s_last l'ultimo "istante" (forward) in cui si è ricevuto un impulso
    tau: f64,
    // errore => w_out : OutputWeight   //output weight, if None is a terminal snn
    // ...ogni collegamento ha il suo peso...non si può fare così, come gestiamo i vari collegamenti (sinapsi)?
}

impl LifNeuron {
    //TODO: uncomment this and make it right
    //fn new(v_mem : f64, v_rest : f64, v_th : f64, r_type : ResetMode, tau : f64) -> Self {
    //    LifNeuron { v_mem, v_rest, v_th, r_type, t_s_last : 0, tau }
    //}
    fn new(id: String) -> Self {
        LifNeuron {
            id,
            v_mem: 0.0,
            v_rest: 0.0,
            v_th: 0.0,
            r_type: ResetMode::Zero,
            t_s_last: 0,
            tau: 0.0,
        }
    }



    //fn get_weights(&self, weights : &Option<Vec<Vec<f64>>>) -> &Vec<f64> {
    //    //questo
    //    let idx = ;
    //    //Capire come risolvere
    //
//
    //}

    fn y(inputs : &Vec<u8>, states : &Vec<u8> ,  weights : &Vec<f64>, states_weights : &Vec<f64>) -> f64 {
        let mut out = 0.0;
        for (idx, x) in inputs.iter().enumerate(){
            out = LifNeuron::add( //somma per ogni neurone
                out, LifNeuron::add( //somma le due moltiplicazioni
                               LifNeuron::mul(x,weights[idx]), // moltiplica la spike per il peso dell'input
                               LifNeuron::mul(&states[idx], states_weights[idx]) // moltiplica lo stato interno per il peso dello stato
                )
            )
        }

        //Out è la combinazione lineare tra il vettore degli input e il vettore dei pesi associati
        out

    }

    // da spostare in un altra libreria (es: simhw)
    fn add(x: f64, y : f64) -> f64{
        //fare controllo su guasto
        x + y
    }
    fn mul(x: &u8, y: f64) -> f64{
        //fare controllo su guasto
        *x as f64 * y
    }

}

impl Neuron for LifNeuron {
    // Creates a new LifNeuron
    // (t_s_last is set to 0 by default at the beginning, no previous impulse received from the beginning of the snn existence)

    // implements the forward pass of the snn
    fn forward(&mut self, input: &Vec<u8>, states_weights : &Option<Vec<Vec<f64>>>, weights : &Option<Vec<Vec<f64>>>, states : &Vec<u8>) -> u8 {


        let exponent : f64 = (3 - self.t_s_last) as f64 / self.tau;
        let n_neuron = self.id.split("-").collect::<Vec<&str>>()[1].parse::<u32>().unwrap() as usize;
        let out = LifNeuron::y(input, states, &weights.as_ref().unwrap()[n_neuron],
                               &states_weights.as_ref().unwrap()[n_neuron]);

        self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * exponent.exp() + out;

        let spike = (self.v_mem > self.v_th) as u8; //if v_mem>v_th then spike=1 else spike=0

        if spike == 1 {
            self.v_mem = match self.r_type {
                ResetMode::Zero => 0.0,
                ResetMode::RestingPotential(v) => v,
                ResetMode::Subthreshold => self.v_mem - self.v_th,
            }
        }

        spike
    }
}
