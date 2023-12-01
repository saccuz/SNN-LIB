use crate::snn::faults::{ActualFault, Component, FaultConfiguration, OuterComponent};
use crate::snn::matrix::Input;
use crate::snn::neuron::Neuron;
use rand::Rng;
use std::fmt::{Display, Formatter, Result};

#[derive(Clone, Copy)]
pub enum NeuronType {
    LifNeuron,
}

pub struct Snn<N: Neuron> {
    layers: Vec<Layer<N>>,
}

impl<N: Neuron> Snn<N> {
    pub fn new(
        n_inputs: u32,
        layers: Vec<u32>,
        intra_conn: Vec<bool>,
        parameters_array: Option<Vec<N::T>>,
    ) -> Self {
        let mut layers_vec = Vec::<Layer<N>>::new();
        if layers.len() != intra_conn.len() {
            panic!(
                "Invalid params, expected Boolean vector shape to be [{}], but got [{}] instead",
                layers.len(),
                intra_conn.len()
            );
        }
        match parameters_array {
            Some(ref v) => {
                if v.len() != layers.len() {
                    panic!("Invalid params, expected Parameters vector shape to be [{}], but got [{}] instead", layers.len(), v.len())
                }
            }
            None => (),
        }

        for (idx, l) in layers.iter().enumerate() {
            layers_vec.push(Layer::new(
                idx as u32,
                *l,
                match intra_conn[idx] {
                    true => Some(Snn::<N>::random_weights(*l, *l, true)),
                    false => None,
                },
                match idx {
                    0 => Snn::<N>::random_weights(*l, n_inputs, false),
                    _ => Snn::<N>::random_weights(*l, layers[idx - 1], false),
                },
                match parameters_array {
                    Some(ref pr) => pr.get(idx),
                    None => None,
                },
            ));
        }
        Snn { layers: layers_vec }
    }

    // To set the inner parameters of layers and neurons.
    pub fn set_parameters(&mut self, parameters: &N::T) {
        for l in self.layers.iter_mut() {
            l.set_parameters(parameters);
        }
    }

    pub fn forward(
        &mut self,
        input_matrix: &Input,
        fault_configuration: Option<&FaultConfiguration>,
    ) -> Vec<Vec<u8>> {
        let mut out = Vec::new();

        match fault_configuration {
            Some(fault_configuration) => {
                let mut layers_info = Vec::new();
                for l in self.layers.iter() {
                    layers_info.push(l.neurons.len());
                }
                let actual_faults = fault_configuration.get_actual_faults(layers_info);
                for input_array in &input_matrix.data {
                    let mut y = self.layers[0].forward(input_array, Some(&actual_faults));
                    for l in self.layers.iter_mut().skip(1) {
                        if l.id == actual_faults.layer_id {
                            y = l.forward(&y, Some(&actual_faults));
                        } else {
                            y = l.forward(&y, None);
                        }
                    }
                    out.push(y.clone());
                }
                out
            }
            None => {
                for input_array in &input_matrix.data {
                    let mut y = self.layers[0].forward(input_array, None);
                    for l in self.layers.iter_mut().skip(1) {
                        y = l.forward(&y, None);
                    }
                    out.push(y.clone());
                }
                out
            }
        }
    }

    pub fn emulate_fault(
        &mut self,
        input_matrix: &Input,
        fault_configuration: &FaultConfiguration,
    ) -> () {
        //TODO salvare le matrici dei pesi dei singoli layer per ripristinarli ad ogni iterazione
        for i in 0..fault_configuration.get_n_occurrences() {
            let result = self.forward(input_matrix, Some(fault_configuration));

            // Print results
            println!("The result for the {} repetition is {:?}", i, result);
        }
    }

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
                    } else {
                        row.push(-rnd_number);
                    }
                } else {
                    row.push(rnd_number);
                }
            }
            weights.push(row);
        }
        weights
    }
}

impl<N: Neuron> From<Vec<Layer<N>>> for Snn<N> {
    fn from(layers_vec: Vec<Layer<N>>) -> Self {
        Snn { layers: layers_vec }
    }
}

impl<N: Neuron> Display for Snn<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        _ = write!(
            f,
            "\n\nSnn Model \n| {:<16} | {:<16} | {:<16} | {:<16} |\n",
            "Layer id", "n_neurons", "n_weights", "n_inner_weights"
        );
        _ = write!(f, "|{:-<18}|{:-<18}|{:-<18}|{:-<18}|\n", "", "", "", "");
        for layer in &self.layers {
            _ = write!(
                f,
                "| {:<16} | {:<16} | {:<16} | {:<16} |\n",
                layer.id,
                layer.neurons.len(),
                (layer.weights.len() * layer.weights[0].len()),
                match &layer.states_weights {
                    Some(a) => a.len() * a.len() - a.len(),
                    None => 0,
                }
            );
        }
        Ok(())
    }
}

struct Layer<N: Neuron> {
    id: u32,
    neurons: Vec<N>,
    states: Vec<u8>,
    states_weights: Option<Vec<Vec<f64>>>,
    weights: Vec<Vec<f64>>,
}

impl<N: Neuron> Layer<N> {
    fn new(
        id: u32,
        neurons: u32,
        states_weights: Option<Vec<Vec<f64>>>,
        weights: Vec<Vec<f64>>,
        parameters: Option<&N::T>,
    ) -> Self {
        let mut neurons_vec = Vec::<N>::new();
        for i in 0..neurons {
            neurons_vec.push(N::new(
                format!("{}-{}", id.to_string(), i.to_string()),
                parameters,
            ));
        }
        Layer {
            id,
            neurons: neurons_vec,
            states: vec![0; neurons as usize],
            states_weights,
            weights,
        }
    }

    fn forward(&mut self, inputs: &Vec<u8>, actual_faults: Option<&ActualFault>) -> Vec<u8> {
        let mut spikes = Vec::<u8>::new();
        match actual_faults {
            None => {
                for n in self.neurons.iter_mut() {
                    spikes.push(n.forward(
                        &inputs,
                        &self.states_weights,
                        &self.weights,
                        &self.states,
                        None,
                    ));
                }
            }
            Some(a_f) => {
                match a_f.component {
                    Component::Inside(_) => {
                        for n in self.neurons.iter_mut() {
                            spikes.push(n.forward(
                                &inputs,
                                &self.states_weights,
                                &self.weights,
                                &self.states,
                                actual_faults,
                            ));
                        }
                    }
                    Component::Outside(ref c) => {
                        match c {
                            OuterComponent::Weights => {
                                //TODO modificare 1 bit del self.weights[a_f.neuron_id.0][a_f.neuron_id.1]
                            }
                            OuterComponent::Connections => {
                                //TODO chiedere se stuck_at deve significare connessione persa sia se 1 o 0
                                //TODO eventualmente mettere a 0 il self.weights[a_f.neuron_id.0][a_f.neuron_id.1]
                            }
                            OuterComponent::InnerWeights => {
                                //TODO modificare 1 bit del self.state_weights[a_f.neuron_id.0][a_f.neuron_id.1]
                            }
                            OuterComponent::InnerConnections => {
                                //TODO chiedere se stuck_at deve significare connessione persa sia se 1 o 0
                                //TODO eventualmente mettere a 0 il self.state_weights[a_f.neuron_id.0][a_f.neuron_id.1]
                            }
                        }
                    }
                }
            }
        }

        self.states = spikes.clone();

        spikes
    }

    fn set_parameters(&mut self, parameters: &N::T) {
        for n in self.neurons.iter_mut() {
            n.set_parameters(parameters);
        }
    }
}
