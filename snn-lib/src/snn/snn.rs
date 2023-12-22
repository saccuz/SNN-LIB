use std::cmp::min;
use crate::snn::faults::{
    bit_flip, fault_iter, stuck_at_one, stuck_at_zero, ActualFault, Component, FaultConfiguration,
    FaultType, OuterComponent,
};
use crate::snn::matrix::Input;
use crate::snn::neuron::Neuron;
use crossbeam::channel::{unbounded, Receiver, Sender};
use rand::Rng;
use std::fmt::{Display, Formatter, Result};
use std::sync::Arc;
use std::thread;
use std::fs::OpenOptions;
use gag::{Redirect};
use std::env::current_dir;
use chrono::{Datelike, DateTime, Timelike, Utc};
use dirs::data_local_dir;

// Setting the format for the log file
fn get_temp_filepath() -> String {
    //take the path of the working directory and add the name of the log file
    let date = Local::now();
    #[cfg(windows)]
    return format!("{}\\Logs\\{}-{:02}-{:02}-{:02}.{:02}.{:02}.log",
                   current_dir().unwrap().display(),
                   date.year_ce().1,
                   date.month(),
                   date.day(),
                   date.hour(),
                   date.minute(),
                   date.second()
    );
    #[cfg(unix)]
    return format!("{}/Logs/{}-{:02}-{:02}-{:02}.{:02}.{:02}.log",
                   current_dir().unwrap().display(),
                   date.year_ce().1,
                   date.month(),
                   date.year(),
                   date.hour(),
                   date.minute(),
                   date.second()
    );
}

// Snn structure
#[derive(Clone)]
pub struct Snn<N: Neuron> {
    layers: Vec<Layer<N>>,
}

// Implementation of Snn structure
impl<N: Neuron + Clone + Send> Snn<N> {

    // Returns a new Snn, with or without setting parameters
    pub fn new(
        n_inputs: u32,
        layers: Vec<u32>,
        intra_conn: Vec<bool>,
        parameters_array: Option<Vec<N::T>>,
    ) -> Self {
        let mut layers_vec = Vec::<Layer<N>>::new();

        // Intra connections (between neurons of same layer) parameter check
        if layers.len() != intra_conn.len() {
            panic!(
                "Invalid params, expected Boolean vector shape to be [{}], but got [{}] instead",
                layers.len(),
                intra_conn.len()
            );
        }

        // Neuron parameters (per layer) parameter check
        match parameters_array {
            Some(ref v) => {
                if v.len() != layers.len() {
                    panic!("Invalid params, expected Parameters vector shape to be [{}], but got [{}] instead", layers.len(), v.len())
                }
            }
            None => (),
        }

        // Building the actual network layers
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

    // To furtherly set the inner parameters of layers' neurons.
    pub fn set_parameters(&mut self, parameters: &N::T) {
        for l in self.layers.iter_mut() {
            l.set_parameters(parameters);
        }
    }

    // Returns the number of neurons for each layer.
    fn get_layer_n_neurons(&self) -> Vec<usize> {
        let mut layers_info = Vec::new();
        for l in self.layers.iter() {
            layers_info.push(l.neurons.len());
        }
        layers_info
    }

    //Compute the number of layers to give to each thread, to better parallelize the network
    fn layer_per_thread(&mut self) -> Vec<Vec<&mut Layer<N>>> {
        //That's because if the number of layers outnumbers the number of threads
        //it would cause a growth in the overall overhead

        let n_th = num_cpus::get();
        let count = self.layers.len() % n_th;
        let mut l_per_t = Vec::new();
        for i in 0..n_th {
            if i < count {
                l_per_t.push(self.layers.len() / n_th + 1);
            } else {
                if (self.layers.len() / n_th) != 0 {
                    l_per_t.push(self.layers.len() / n_th)
                }
            }
        }

        let mut s = Vec::new();
        for _ in 0..l_per_t.len() {
            s.push(Vec::new())
        }

        let mut pos = 0;
        for l in self.layers.iter_mut() {
            if l_per_t[pos] > s[pos].len() {
                s[pos].push(l);
            } else {
                pos += 1;
                s[pos].push(l);
            }
        }

        s
    }

    // Snn forward function
    pub fn forward(
        &mut self,
        input_matrix: &Input,
        fault_configuration: Option<&FaultConfiguration<N::D>>,
    ) -> Vec<Vec<u8>> {

        // Creating channels for threads communication
        let mut layers_channel_senders = Vec::new();
        let mut layers_channel_receivers = Vec::new();

        let n_channels = min(self.layers.len() + 1, num_cpus::get() + 1);

        for _ in 0..n_channels {
            let channel = unbounded::<(usize, Vec<u8>)>();
            layers_channel_senders.push(channel.0);
            layers_channel_receivers.push(channel.1);
        }

        // Getting the actual fault, if present
        let actual_fault = match fault_configuration {
            Some(f_c) => {
                let a_f = f_c.get_actual_faults(self.get_layer_n_neurons(), input_matrix.rows);
                //TODO: This print has to be made conditional for debugging purposes
                println!("{}", a_f);
                Some(a_f)
            },
            None => { None }
        };

        let mut out = Vec::new();
        let mut layer_distribution = self.layer_per_thread();
        let th_len = layer_distribution.len();

        // Creating the scope in which threads will be launched
        thread::scope(|scope| {
            for (thread_idx, v_l) in layer_distribution.iter_mut().enumerate() {
                let rx_clone_to_send1 = layers_channel_receivers[thread_idx].clone();
                let tx_clone_to_send2 = layers_channel_senders[thread_idx + 1].clone();

                scope.spawn(|| {
                    Snn::forward_parallel(
                        v_l,
                        rx_clone_to_send1,
                        tx_clone_to_send2,
                        Arc::new(match actual_fault {
                            Some(ref a_f) => {
                                Some(a_f)
                            },
                            None => None
                        }),
                    )
                });
            }
            // Starting the pipeline sending the inputs: each column of the input matrix
            for (idx, input_array) in input_matrix.data.iter().enumerate() {
                layers_channel_senders[0]
                    .send((idx, input_array.clone()))
                    .unwrap();
            }
            // Actually just the first sender should be dropped..... TODO: CHECK THIS
            drop(layers_channel_senders);

            // Receiving the final results
            while let Ok(result) = layers_channel_receivers[th_len].recv() {
                out.push(result.1.clone());
            }
        });
        out
    }

    // Function that each thread will run
    fn forward_parallel(
        layers: &mut Vec<&mut Layer<N>>,
        rx: Receiver<(usize, Vec<u8>)>,
        tx: Sender<(usize, Vec<u8>)>,
        actual_fault: Arc<Option<&ActualFault<N::D>>>,
    ) {
        // Waits until the previous layer output is sent as input
        while let Ok(value) = rx.recv() {
            let mut y = value.1;
            for l in layers.iter_mut() {
                //Do neuron stuff here
                match *actual_fault {
                    Some(a_f) if (*l).id == a_f.layer_id => {
                            y = (*l).forward(&y, Some(a_f), value.0);
                    }
                    _ => {
                        y = (*l).forward(&y, None, value.0);
                    }
                }
            }
            tx.send((value.0, y)).unwrap();
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

        println!("Output logs can be found in: {}", get_temp_filepath());

        let log = OpenOptions::new()
            .truncate(true)
            .read(true)
            .write(true)
            .create(true)
            .open(get_temp_filepath())
            .unwrap();

        let _print_redirect = Redirect::stdout(log).unwrap();

        println!("{}", fault_configuration);
        for i in 0..fault_configuration.get_n_occurrences() {
            let result = self
                .clone()
                .forward(input_matrix, Some(fault_configuration));

            println!("Result for rep {:02}: {:?}\n", i, result);

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

    fn set_parameters(&mut self, parameters: &N::T) {
        for n in self.neurons.iter_mut() {
            n.set_parameters(parameters);
        }
    }

    //set_parameters??
    //Should we add other utility function for debugging?

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
                        let fault = match a_f.fault_type {
                            FaultType::TransientBitFlip if (a_f.time_tbf.unwrap() != time) => None,
                            _ => actual_faults,
                        };
                        for n in self.neurons.iter_mut() {

                            spikes.push(n.forward(
                                &inputs,
                                &self.states_weights,
                                &self.weights,
                                &self.states,
                                if a_f.neuron_id.0 == n.get_id() { fault } else { None },
                            ));
                        }
                    },
                    Component::Outside(ref c) => {
                        // true is for weights, false is for states_weights
                        let mut save = (true, 0.0);
                        let mut saved_weights = (Vec::<f64>::new(), Vec::<f64>::new());
                        match c {
                            OuterComponent::Weights => match a_f.fault_type {
                                FaultType::StuckAtZero if time == 0 => {
                                    stuck_at_zero(
                                        &mut self.weights[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize],
                                        a_f.offset,
                                    );
                                }
                                FaultType::StuckAtOne if time == 0 => {
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
                                    FaultType::StuckAtZero if time == 0 => {
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
                                    FaultType::StuckAtOne if time == 0 => {
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
                                    _ => { /* in if time != 0 we don't need to apply Stuck At-X because it was already applied */ }
                                }
                            }
                            OuterComponent::InnerWeights => match self.states_weights {
                                Some(ref mut sw) => match a_f.fault_type {
                                    FaultType::StuckAtZero if time == 0 => {
                                        stuck_at_zero(
                                            &mut sw[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                    FaultType::StuckAtOne if time == 0 => {
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
                                    _ => { /* in if time != 0 we don't need to apply Stuck At-X because it was already applied */ }
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
                                    } else {
                                        match self.states_weights {
                                            Some(ref mut v) => {
                                                v[a_f.neuron_id.0 as usize]
                                                    [a_f.neuron_id.1.unwrap() as usize] = save.1
                                            }
                                            _ => {}
                                        }
                                    }
                                } else {
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
}
