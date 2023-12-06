use crate::snn::faults::{
    bit_flip, stuck_at_one, stuck_at_zero, ActualFault, Component, FaultConfiguration, FaultType,
    OuterComponent,
};
use crate::snn::lif::LifNeuronParameters;
use crate::snn::matrix::Input;
use crate::snn::neuron::{Neuron, NeuronParameters};
use rand::Rng;
use std::fmt::{Display, Formatter, Result};

#[derive(Clone, Copy)]
pub enum NeuronType {
    LifNeuron,
}

#[derive(Clone)]
pub struct Snn<N: Neuron> {
    layers: Vec<Layer<N>>,
}

impl<N: Neuron + Clone> Snn<N> {
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

    fn get_layer_nneurons(&self) -> Vec<usize> {
        let mut layers_info = Vec::new();
        for l in self.layers.iter() {
            layers_info.push(l.neurons.len());
        }
        layers_info
    }

    pub fn forward(
        &mut self,
        input_matrix: &Input,
        fault_configuration: Option<&FaultConfiguration>,
    ) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        match fault_configuration {
            Some(fault_configuration) => {
                let actual_faults = fault_configuration
                    .get_actual_faults(self.get_layer_nneurons(), input_matrix.rows);
                for (idx, input_array) in input_matrix.data.iter().enumerate() {
                    let mut y = Vec::new();
                    for (layer_idx, l) in self.layers.iter_mut().enumerate() {
                        //La prima condizione equivale a let y = input per il primo layer
                        //La seconda condizione è un check sul fault
                        y = l.forward(
                            if layer_idx == 0 { input_array } else { &y },
                            if l.id == actual_faults.layer_id {
                                Some(&actual_faults)
                            } else {
                                None
                            },
                            idx,
                        )
                    }
                    out.push(y.clone());
                }
                out
            }
            None => {
                for input_array in &input_matrix.data {
                    let mut y = Vec::new();
                    for (layer_idx, l) in self.layers.iter_mut().enumerate() {
                        y = l.forward(if layer_idx == 0 { input_array } else { &y }, None, 0);
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
        let saved_w = self
            .layers
            .iter()
            .map(|x| (x.weights.clone(), x.states_weights.clone()))
            .collect::<Vec<(Vec<Vec<f64>>, Option<Vec<Vec<f64>>>)>>();
        for i in 0..fault_configuration.get_n_occurrences() {
            let ll = self
                .layers
                .iter()
                .map(|x| (*x).clone())
                .collect::<Vec<Layer<N>>>();

            let mut cloned = Snn::from(ll);

            let result = cloned.forward(input_matrix, Some(fault_configuration));

            println!("The result for the {} repetition is {:?}", i, result);

            // restore original weights
            for (idx, l) in self.layers.iter_mut().enumerate() {
                l.weights = saved_w[idx].0.clone();
                l.states_weights = saved_w[idx].1.clone();
            }
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
#[derive(Clone)]
struct Layer<N: Neuron> {
    id: u32,
    neurons: Vec<N>,
    states: Vec<u8>,
    states_weights: Option<Vec<Vec<f64>>>,
    weights: Vec<Vec<f64>>,
}

impl<N: Neuron + Clone> Layer<N> {
    fn new(
        id: u32,
        neurons: u32,
        states_weights: Option<Vec<Vec<f64>>>,
        weights: Vec<Vec<f64>>,
        parameters: Option<&N::T>,
    ) -> Self {
        let mut neurons_vec = Vec::<N>::new();
        for i in 0..neurons {
            neurons_vec.push(N::new(i, parameters));
        }
        Layer {
            id,
            neurons: neurons_vec,
            states: vec![0; neurons as usize],
            states_weights,
            weights,
        }
    }

    fn forward(
        &mut self,
        inputs: &Vec<u8>,
        actual_faults: Option<&ActualFault>,
        time: usize,
    ) -> Vec<u8> {
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
                        let mut save = self.neurons[0].get_parameters();
                        let fault = match a_f.fault_type {
                            FaultType::TransientBitFlip if (a_f.time_tbf.unwrap() != time) => None,
                            _ => actual_faults,
                        };
                        for n in self.neurons.iter_mut() {
                            if a_f.neuron_id.0 == n.get_id() {
                                if a_f.time_tbf.is_some() {
                                    save = n.get_parameters();
                                }
                                spikes.push(n.forward(
                                    &inputs,
                                    &self.states_weights,
                                    &self.weights,
                                    &self.states,
                                    fault,
                                ));
                            } else {
                                spikes.push(n.forward(
                                    &inputs,
                                    &self.states_weights,
                                    &self.weights,
                                    &self.states,
                                    None,
                                ));
                            }
                        }
                        match a_f.fault_type {
                            FaultType::TransientBitFlip if (a_f.time_tbf.unwrap() == time) => {
                                self.neurons[a_f.neuron_id.0 as usize].set_parameters(&save);
                            }
                            _ => {}
                        }
                    }
                    Component::Outside(ref c) => {
                        let mut save = 0.0;
                        match c {
                            OuterComponent::Weights => match a_f.fault_type {
                                FaultType::StuckAtZero => {
                                    stuck_at_zero(
                                        &mut self.weights[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize],
                                        a_f.offset,
                                    );
                                }
                                FaultType::StuckAtOne => {
                                    stuck_at_one(
                                        &mut self.weights[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize],
                                        a_f.offset,
                                    );
                                }
                                FaultType::TransientBitFlip => {
                                    if time == a_f.time_tbf.unwrap() {
                                        save = self.weights[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize]
                                            .clone();
                                        bit_flip(
                                            &mut self.weights[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize]
                                                .clone(),
                                            a_f.offset,
                                        );
                                    }
                                }
                            },
                            OuterComponent::Connections => {
                                //TODO eventualmente mettere a 0 il self.weights[a_f.neuron_id.0][a_f.neuron_id.1]
                                match a_f.fault_type {
                                    FaultType::StuckAtZero => {}
                                    FaultType::StuckAtOne => {}
                                    FaultType::TransientBitFlip => {
                                        //      let save =  (self.weights[a_f.neuron_id.0][a_f.neuron_id.1])
                                        //      self.weights[a_f.neuron_id.0][a_f.neuron_id.1] = 0;
                                        //      l.forward(cazzivari)
                                        //      self.weights[a_f.neuron_id.0][a_f.neuron_id.1] = save;
                                    }
                                }
                            }
                            OuterComponent::InnerWeights => match self.states_weights {
                                Some(ref mut sw) => match a_f.fault_type {
                                    FaultType::StuckAtZero => {
                                        stuck_at_zero(
                                            &mut sw[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                    FaultType::StuckAtOne => {
                                        stuck_at_one(
                                            &mut sw[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                    FaultType::TransientBitFlip => {}
                                },
                                None => {}
                            },
                            OuterComponent::InnerConnections => {
                                //TODO eventualmente mettere a 0 il self.state_weights[a_f.neuron_id.0][a_f.neuron_id.1]
                                match a_f.fault_type {
                                    FaultType::StuckAtZero => {}
                                    FaultType::StuckAtOne => {}
                                    FaultType::TransientBitFlip => {}
                                }
                            }
                        }
                        for n in self.neurons.iter_mut() {
                            spikes.push(n.forward(
                                &inputs,
                                &self.states_weights,
                                &self.weights,
                                &self.states,
                                None,
                            ));
                        }
                        match a_f.fault_type {
                            FaultType::TransientBitFlip if (a_f.time_tbf.unwrap() == time) => {
                                self.weights[a_f.neuron_id.0 as usize]
                                    [a_f.neuron_id.1.unwrap() as usize] = save
                            }
                            _ => {}
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
