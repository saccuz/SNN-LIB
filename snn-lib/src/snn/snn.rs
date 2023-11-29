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
        //TODO: AGGIUNGERE CONTROLLI SUI VARI PARAMETRI
        for (idx, l) in layers.iter().enumerate() {
            layers_vec.push(Layer::new(
                idx,
                *l,
                match intra_conn[idx] {
                    true => Option::Some(Snn::<N>::random_weights(*l, *l, true)),
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

    pub fn forward(&mut self, x: Vec<u8>) -> Vec<u8> {
        let mut out = x;
        //TODO: AGGIUNGERE CONTROLLI SUI VARI PARAMETRI
        for l in self.layers.iter_mut() {
            out = l.forward(&out);
        }
        out
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
            "Layer id", "n_neurons", "n_weights", "n_inter_weights"
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
                    Some(a) => (a.len() * a.len()),
                    None => 0,
                }
            );
        }
        Result::Ok(())
    }
}

struct Layer<N: Neuron> {
    id: String,
    neurons: Vec<N>,
    states: Vec<u8>,
    states_weights: Option<Vec<Vec<f64>>>,
    weights: Vec<Vec<f64>>,
}

impl<N: Neuron> Layer<N> {
    fn new(
        id: usize,
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
            id: id.to_string(),
            neurons: neurons_vec,
            states: vec![0; neurons as usize],
            states_weights,
            weights,
        }
    }

    fn forward(&mut self, inputs: &Vec<u8>) -> Vec<u8> {
        let mut spikes = Vec::<u8>::new();

        for n in self.neurons.iter_mut() {
            spikes.push(n.forward(&inputs, &self.states_weights, &self.weights, &self.states));
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
