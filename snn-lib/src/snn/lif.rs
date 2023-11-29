//Make snn from snn.rs usable in this file
use crate::snn::neuron::Neuron;
use crate::snn::neuron::NeuronParameters;

#[derive(Clone, Copy)]

pub enum ResetMode {
    Zero,
    RestingPotential(f64),
    Subthreshold,
}
#[derive(Clone)]
pub struct LifNeuronParameters {
    pub v_rest: f64,
    pub v_th: f64,
    pub r_type: ResetMode,
    pub tau: f64,
}

impl NeuronParameters for LifNeuronParameters {}

pub struct LifNeuron {
    id: String,
    v_mem: f64,
    v_rest: f64,
    v_th: f64,
    r_type: ResetMode,
    t_s_last: u64,
    tau: f64,
    timer: u64, //conta il tempo dopo ogni forward
}

impl LifNeuron {
    //fn get_weights(&self, weights : &Option<Vec<Vec<f64>>>) -> &Vec<f64> {
    //    //questo
    //    let idx = ;
    //    //Capire come risolvere
    //
    //
    //}

    fn y(
        inputs: &Vec<u8>,
        states: &Vec<u8>,
        weights: &Vec<f64>,
        states_weights: Option<&Vec<f64>>,
    ) -> f64 {
        let mut out = 0.0;
        //TODO: AGGIUNGERE CONTROLLI SUI VARI PARAMETRI
        //TODO: CAPIRE SE TRASFERIRE I FOR IN UNA FUNZIONE A PARTE

        //prodotto scalare tra input e pesi degli input
        for (idx, x) in inputs.iter().enumerate() {
            //somma per ogni neurone
            out = LifNeuron::add(out, LifNeuron::mul(x, weights[idx])); // moltiplica la spike per il peso dell'input
        }

        match states_weights {
            //se è presente il vettore dei pesi degli stati
            Some(states_weights) => {
                let mut sum1 = 0.0;
                //prodotto scalare tra stati e pesi degli stati
                for (idx, x) in states_weights.iter().enumerate() {
                    sum1 = LifNeuron::add(sum1, LifNeuron::mul(&states[idx], *x));
                }
                LifNeuron::add(out, sum1)
            }
            //se non è presente il vettore dei pesi degli stati
            None => out,
        }
        //Out è la combinazione lineare tra il vettore degli input e il vettore dei pesi associati
    }

    // da spostare in un altra libreria (es: simhw)
    fn add(x: f64, y: f64) -> f64 {
        //fare controllo su guasto
        x + y
    }
    fn mul(x: &u8, y: f64) -> f64 {
        //fare controllo su guasto
        *x as f64 * y
    }
}

impl Neuron for LifNeuron {
    type T = LifNeuronParameters;
    // Creates a new LifNeuron
    // (t_s_last is set to 0 by default at the beginning, no previous impulse received from the beginning of the snn existence)
    fn new(id: String, parameters: Option<&LifNeuronParameters>) -> Self {
        match parameters {
            Some(p) => LifNeuron {
                id,
                v_mem: 0.0,
                v_rest: p.v_rest,
                v_th: p.v_th,
                r_type: ResetMode::Zero,
                t_s_last: 0,
                tau: p.tau,
                timer: 0,
            },
            None => LifNeuron {
                id,
                v_mem: 0.0,
                v_rest: 0.0,
                v_th: 0.8,
                r_type: ResetMode::Zero,
                t_s_last: 0,
                tau: 0.0,
                timer: 0,
            },
        }
    }

    fn set_parameters(&mut self, parameters: &LifNeuronParameters) {
        self.v_rest = parameters.v_rest;
        self.v_th = parameters.v_th;
        self.r_type = parameters.r_type;
        self.tau = parameters.tau;
    }

    fn get_parameters(&self) -> Self::T {
        LifNeuronParameters {
            v_rest: self.v_rest,
            v_th: self.v_th,
            r_type: self.r_type,
            tau: self.tau,
        }
    }

    // implements the forward pass of the snn
    fn forward(
        &mut self,
        input: &Vec<u8>,
        states_weights: &Option<Vec<Vec<f64>>>,
        weights: &Vec<Vec<f64>>,
        states: &Vec<u8>,
    ) -> u8 {
        self.timer += 1;
        // Input impulses summation
        let n_neuron = self.id.split("-").collect::<Vec<&str>>()[1]
            .parse::<u32>()
            .unwrap() as usize;
        let summation = match states_weights {
            Some(states_weights) => LifNeuron::y(
                input,
                states,
                &weights[n_neuron],
                Option::Some(&states_weights[n_neuron]),
            ),
            None => LifNeuron::y(input, states, &weights[n_neuron], Option::None),
        };

        let exponent: f64 = -((self.timer - self.t_s_last) as f64) / self.tau;
        self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * exponent.exp() + summation;

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
