use crate::snn::faults::{
    bit_flip, fault_iter, stuck_at_one, stuck_at_zero, ActualFault, Component, FaultConfiguration,
    FaultType, OuterComponent,
};
use crate::snn::matrix::Input;
use crate::snn::neuron::Neuron;
use rand::Rng;
use std::fmt::{Display, Formatter, Result};
use crossbeam::channel::{unbounded, Receiver, Sender};
use std::thread;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
pub enum NeuronType {
    LifNeuron,
}

#[derive(Clone)]
pub struct Snn<N: Neuron> {
    layers: Vec<Layer<N>>,
}

impl<N: Neuron + Clone + Send> Snn<N> {
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

    //
    fn get_layer_n_neurons(&self) -> Vec<usize> {
        let mut layers_info = Vec::new();
        for l in self.layers.iter() {
            layers_info.push(l.neurons.len());
        }
        layers_info
    }

    pub fn forward(
        &mut self,
        input_matrix: &Input,
        fault_configuration: Option<&FaultConfiguration<N::D>>,
    ) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        match fault_configuration {
            Some(fault_configuration) => {
                let actual_faults = fault_configuration
                    .get_actual_faults(self.get_layer_n_neurons(), input_matrix.rows);
                thread::scope(|scope|{
                    let n_layers = self.layers.len();
                    for (layer_idx, l ) in self.layers.iter_mut().enumerate() {
                        let rx_clone_to_send1 = layers_channel_receivers[layer_idx].clone();
                        let tx_clone_to_send2 = layers_channel_senders[layer_idx + 1].clone();

                        scope.spawn(|| {
                            Snn::forward_parallel(l,
                                                  rx_clone_to_send1,
                                                  tx_clone_to_send2,
                                                  if l.id == actual_faults.layer_id {
                                                      Arc::new(Some(&actual_faults))
                                                  } else {
                                                      Arc::new(None)
                                                  })
                        });
                    }
                    for (idx, input_array) in input_matrix.data.iter().enumerate() {
                        layers_channel_senders[0].send((idx , input_array.clone())).unwrap();
                    }
                    // Actually just the first sender should be dropped..... CHECK THIS
                    drop(layers_channel_senders);

                    // Receiving the final result for the current input_array
                    while let Ok(result) = layers_channel_receivers[n_layers].recv() {
                        out.push(result.1.clone());
                    }
                });
                out
            },
            None => {
                //We create n+1 channel:
                //input (layer 0)
                //layer i
                //...
                //output (n+1 | output receiver)
                thread::scope(|scope|{
                    let n_layers = self.layers.len();
                    for (layer_idx, l ) in self.layers.iter_mut().enumerate() {
                        let rx_clone_to_send1 = layers_channel_receivers[layer_idx].clone();
                        let tx_clone_to_send2 = layers_channel_senders[layer_idx + 1].clone();
                        scope.spawn(|| {
                            Snn::forward_parallel(l, rx_clone_to_send1, tx_clone_to_send2, Arc::new(None));
                        });
                    }
                    for (idx, input_array) in input_matrix.data.iter().enumerate() {
                        layers_channel_senders[0].send((idx,input_array.clone())).unwrap();
                    }
                    // Actually just the first sender should be dropped..... CHECK THIS
                    drop(layers_channel_senders);

                    // Receiving the final result for the current input_array
                    while let Ok(result) = layers_channel_receivers[n_layers].recv() {
                        out.push(result.1.clone());
                    }
                });
                out
            }
        }
    }

    fn forward_parallel(l :&mut Layer<N>, rx: Receiver<(usize, Vec<u8>)>, tx: Sender<(usize, Vec<u8>)>, actual_fault: Arc<Option<&ActualFault<N::D>>>) {
        let mut out = Vec::new();
        let fault = *actual_fault;
        while let Ok(value) = rx.recv() {
            //Do neuron stuff here
            out = l.forward(&value.1, fault, value.0);
            tx.send((value.0, out)).unwrap();
        }
    }

    pub fn emulate_fault(
        &mut self,
        input_matrix: &Input,
        fault_configuration: &FaultConfiguration<N::D>,
    ) -> () {
        let saved_w = self
            .layers
            .iter()
            .map(|x| (x.weights.clone(), x.states_weights.clone()))
            .collect::<Vec<(Vec<Vec<f64>>, Option<Vec<Vec<f64>>>)>>();

        for i in 0..fault_configuration.get_n_occurrences() {

            let result = self.clone().forward(input_matrix, Some(fault_configuration));

            // First version: nice output printing, Second version: debug speed of light
            //println!("and the result for the {} repetition is {:?}\n", i, result);
            println!("Result for rep {:02}: {:?}", i, result);

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
        actual_faults: Option<&ActualFault<N::D>>,
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
                        // true is for weights, false is for states_weights
                        let mut save = (true, 0.0);
                        let mut saved_weights = (Vec::<f64>::new(), Vec::<f64>::new());
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
                                        save = (
                                            true,
                                            self.weights[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize]
                                                .clone(),
                                        );
                                        bit_flip(
                                            &mut self.weights[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                }
                            },
                            OuterComponent::Connections => {
                                //##### We suppose that both weights and internal weights are passed through the same buses ######//
                                match a_f.fault_type {
                                    FaultType::StuckAtZero => {
                                        fault_iter(
                                            &mut self.weights[a_f.neuron_id.0 as usize],
                                            a_f,
                                            &stuck_at_zero,
                                        );
                                        match self.states_weights {
                                            Some(ref mut v) => {
                                                fault_iter(
                                                    &mut v[a_f.neuron_id.0 as usize],
                                                    a_f,
                                                    &stuck_at_zero,
                                                );
                                            }
                                            _ => {}
                                        }
                                    }
                                    FaultType::StuckAtOne => {
                                        fault_iter(
                                            &mut self.weights[a_f.neuron_id.0 as usize],
                                            a_f,
                                            &stuck_at_one,
                                        );
                                        match self.states_weights {
                                            Some(ref mut v) => {
                                                fault_iter(
                                                    &mut v[a_f.neuron_id.0 as usize],
                                                    a_f,
                                                    &stuck_at_one,
                                                );
                                            }
                                            _ => {}
                                        }
                                    }
                                    FaultType::TransientBitFlip => {
                                        if time == a_f.time_tbf.unwrap() {
                                            saved_weights.0 =
                                                self.weights[a_f.neuron_id.0 as usize].clone();
                                            fault_iter(
                                                &mut self.weights[a_f.neuron_id.0 as usize],
                                                a_f,
                                                &bit_flip,
                                            );
                                            match self.states_weights {
                                                Some(ref mut v) => {
                                                    saved_weights.1 =
                                                        v[a_f.neuron_id.0 as usize].clone();
                                                    fault_iter(
                                                        &mut v[a_f.neuron_id.0 as usize],
                                                        a_f,
                                                        &bit_flip,
                                                    );
                                                }
                                                _ => {}
                                            }
                                        }
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
                                    FaultType::TransientBitFlip => {
                                        if time == a_f.time_tbf.unwrap() {
                                            save = (
                                                false,
                                                sw[a_f.neuron_id.0 as usize]
                                                    [a_f.neuron_id.1.unwrap() as usize]
                                                    .clone(),
                                            );
                                            bit_flip(
                                                &mut sw[a_f.neuron_id.0 as usize]
                                                    [a_f.neuron_id.1.unwrap() as usize],
                                                a_f.offset,
                                            );
                                        }
                                    }
                                },
                                None => {}
                            },
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
                                if a_f.bus.is_none() {
                                    if save.0 {
                                        self.weights[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize] = save.1
                                    }
                                    else {
                                        match self.states_weights {
                                            Some(ref mut v) => {
                                                v[a_f.neuron_id.0 as usize]
                                                    [a_f.neuron_id.1.unwrap() as usize] = save.1
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                else {
                                    self.weights[a_f.neuron_id.0 as usize] = saved_weights.0;
                                    match self.states_weights {
                                        Some(ref mut v) => {
                                            v[a_f.neuron_id.0 as usize] = saved_weights.1;
                                        }
                                        _ => {}
                                    }
                                }
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
