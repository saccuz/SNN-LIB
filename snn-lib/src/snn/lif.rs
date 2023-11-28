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
                let rnd_number = rng.gen_range(0.01..1.0);
                if diag {
                    if r == c {
                        row.push(0.0);
                    }
                    else {
                        row.push(-rnd_number);
                    }
                }
                else {
                    row.push(rnd_number);
                }
            }
            weights.push(row);
        }

        weights
    }

    pub fn new(n_inputs: u32, layers: Vec<u32>, intra_conn: Vec<bool>, neuron_type: NeuronType) -> Self {
        let mut layers_vec = Vec::<Layer>::new();

        layers_vec.push(Layer::new(0, layers[0], neuron_type, None, None));

        for (idx, l) in layers.iter().skip(1).enumerate() {
            layers_vec.push(Layer::new(
                idx,
                *l,
                neuron_type,
                match intra_conn[idx] {
                    true => Option::Some(Snn::random_weights(*l, *l, true)),
                    false => None
                },
                Option::Some(Snn::random_weights(*l, layers[idx], false)),
            ));
        }
        Snn { layers: layers_vec }
    }

    pub fn forward(&mut self, x: Vec<u8>) -> Vec<u8> {
        let mut out = x;
        //TODO: AGGIUNGERE CONTROLLI SUI VARI PARAMETRI
        for l in self.layers.iter_mut() {
            out = l.forward(&out);
        }

        out
    }
}

struct Layer {
    id: String,
    neurons: Vec<Box<dyn Neuron + 'static>>,
    inputs: Vec<u8>,
    states: Vec<u8>,
    states_weights: Option<Vec<Vec<f64>>>,
    weights: Vec<Vec<f64>>,
}

impl Layer {
    fn new(
        id: usize,
        neurons: u32,
        neuron_type: NeuronType,
        states_weights: Option<Vec<Vec<f64>>>,
        weights: Vec<Vec<f64>>
    ) -> Self {
        let mut neurons_vec = match neuron_type {
            NeuronType::LifNeuron => Vec::<Box<dyn Neuron>>::new(),
        };
        for i in 0..neurons {
            neurons_vec.push(Box::new(LifNeuron::new(format!(
                "{}-{}",
                id.to_string(),
                i.to_string()
            ))))
        }
        Layer {
            id: id.to_string(),
            neurons: neurons_vec,
            inputs: Vec::<u8>::new(),
            states: Vec::<u8>::new(),
            states_weights,
            weights,
        }
    }

    fn forward(self, states_vector: &Vec<u8>) -> Vec<u8> {
        let mut spikes = Vec::<u8>::new();

        for n in self.neurons.iter_mut() {
            spikes.push(n.forward(&inputs, &self.states_weights, &self.weights, &self.states));
        }

        spikes
    }
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
            v_th: 0.8,
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
        //TODO: AGGIUNGERE CONTROLLI SUI VARI PARAMETRI
        //TODO: CAPIRE SE TRASFERIRE I FOR IN UNA FUNZIONE A PARTE

        //prodotto scalare tra input e pesi degli input
        for (idx, x) in inputs.iter().enumerate(){
            //somma per ogni neurone
            out = LifNeuron::add(out, LifNeuron::mul(x,weights[idx])); // moltiplica la spike per il peso dell'input
        }

        match states_weights {
            //se è presente il vettore dei pesi degli stati
            Some(states_weights) => {
                let mut sum1 = 0.0;
                //prodotto scalare tra stati e pesi degli stati
                for (idx, x) in states_weights.iter().enumerate(){
                    sum1 = LifNeuron::add(sum1, LifNeuron::mul(&states[idx], *x));
                }
                LifNeuron::add(out, sum1)
            },
            //se non è presente il vettore dei pesi degli stati
            None => {
                   out
            },
        }
        //Out è la combinazione lineare tra il vettore degli input e il vettore dei pesi associati
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
