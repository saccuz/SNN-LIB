use crate::snn::faults::{
    bit_flip, fault_iter, stuck_at_one, stuck_at_zero, ActualFault, Component, FaultConfiguration,
    FaultType, OuterComponent,
};
use crate::snn::generic_matrix::MatrixG;
use crate::snn::neuron::Neuron;
use chrono::{Datelike, Local, Timelike};
use crossbeam::channel::{unbounded, Receiver, Sender};
use gag::Redirect;
use std::cmp::min;
use std::env::current_dir;
use std::fmt::{Display, Formatter, Result};
use std::fs::OpenOptions;
use std::sync::Arc;
use std::thread;

// Setting the format for the log file
fn get_temp_filepath() -> String {
    // Take the path of the working directory and add the name of the log file
    let date = Local::now();
    #[cfg(windows)]
    return format!(
        "{}\\Logs\\{}-{:02}-{:02}-{:02}.{:02}.{:02}.log",
        current_dir().unwrap().display(),
        date.year_ce().1,
        date.month(),
        date.day(),
        date.hour(),
        date.minute(),
        date.second()
    );

    #[cfg(unix)]
    return format!(
        "{}/Logs/{}-{:02}-{:02}-{:02}.{:02}.{:02}.log",
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
        personalized_weights: Option<Vec<MatrixG<f64>>>,
        personalized_inner_weights: Option<Vec<Option<MatrixG<f64>>>>,
        seed: Option<u64>,
    ) -> Self {
        let mut layers_vec = Vec::<Layer<N>>::new();

        // The network should have at least 1 input.
        if n_inputs == 0 {
            panic!("Invalid params, n_inputs expected to be greater than 0")
        }

        // The network can't be empty, so at least one layers has to exists.
        if layers.len() == 0 {
            panic!("Invalid params, layers len expected to be greater than 0. Can't create an empty network")
        }

        // The number of neuron for each layer can't be 0.
        if !layers.iter().all(|&x| x > 0) {
            panic!("Invalid params, the number of neurons for each layers has to be greater than zero. Consider to change layers vector.")
        }

        // Intra connections (between neurons of same layer) parameter check.
        if layers.len() != intra_conn.len() {
            panic!(
                "Invalid params, expected Boolean vector shape to be [{}], but got [{}] instead",
                layers.len(),
                intra_conn.len()
            );
        }

        // Neuron parameters (per layer) parameter check
        if let Some(ref v) = parameters_array {
            if v.len() != layers.len() {
                panic!("Invalid params, expected Parameters vector shape to be [{}], but got [{}] instead", layers.len(), v.len())
            }
        }

        // Check weights
        if let Some(ref weights) = personalized_weights {
            // Check that the number of Weights vectors it's equal to the number of layers
            if weights.len() != layers.len() {
                panic!("Invalid params, expected Parameters vector shape to be [{}], but got [{}] instead", layers.len(), weights.len())
            }
            // Checking Weights matrices shapes for each layer.
            for (idx, w) in weights.iter().enumerate() {
                // Check if the number of weights of the first layer for each neuron it's equal to the number of inputs (FC Layer)
                if idx == 0 && (w.cols != n_inputs as usize || w.rows != layers[idx] as usize) {
                    panic!("Invalid params, expected weights shape at index {} to be [{}, {}], but got [{}, {}] instead",
                           idx ,
                           layers[idx],
                           n_inputs,
                           w.rows,
                           w.cols)
                }
                // Check if the number of weights of the idx layer for each neuron it's equal to the number of neuron of the previous layer (FC Layer)
                else if idx != 0
                    && (w.cols != layers[idx - 1] as usize || w.rows != layers[idx] as usize)
                {
                    panic!("Invalid params, expected weights shape at index {} to be [{}, {}], but got [{}, {}] instead",
                           idx ,
                           layers[idx],
                           n_inputs,
                           w.rows,
                           w.cols)
                }
            }
        }

        if let Some(ref inners) = personalized_inner_weights {
            if inners.len() != layers.len() {
                panic!("Invalid params, expected Inner Weights vector shape to be [{}], but got [{}] instead", layers.len(), inners.len())
            }
            for (idx, iw) in inners.iter().enumerate() {
                match iw {
                    Some(i_w) => {
                        if !intra_conn[idx] {
                            panic!("Invalid params, found inner weights vector for layer {} with intra_conn value of {}. Consider to change it to {} ", idx, intra_conn[idx], !intra_conn[idx])
                        }
                        if i_w.rows != layers[idx] as usize || i_w.cols != layers[idx] as usize {
                            panic!("Invalid params, expected inner weights shape at index {} to be [{}, {}], but got [{}, {}] instead",idx, layers[idx], layers[idx], i_w.rows, i_w.cols)
                        }
                    }
                    None => {
                        if intra_conn[idx] {
                            panic!("Invalid params, inner weights vector not found for layer {} with intra_conn value of {}. Consider to change it to {}", idx, intra_conn[idx], !intra_conn[idx])
                        }
                    }
                }
            }
        }

        // Building the actual network layers
        for (idx, l) in layers.iter().enumerate() {
            layers_vec.push(Layer::new(
                idx as u32,
                *l,
                match personalized_inner_weights {
                    Some(ref i_weights) => match i_weights[idx] {
                        Some(ref iw) => Some(iw.clone()),
                        None => None,
                    },
                    None => match intra_conn[idx] {
                        true => Some(MatrixG::random(
                            *l as usize,
                            *l as usize,
                            true,
                            seed,
                            -0.99,
                            -0.01,
                        )),
                        false => None,
                    },
                },
                match personalized_weights {
                    Some(ref w) => w[idx].clone(),
                    None => match idx {
                        0 => {
                            MatrixG::random(*l as usize, n_inputs as usize, false, seed, 0.01, 0.99)
                        }
                        _ => MatrixG::random(
                            *l as usize,
                            layers[idx - 1] as usize,
                            false,
                            seed,
                            0.01,
                            0.99,
                        ),
                    },
                },
                match parameters_array {
                    Some(ref pr) => pr.get(idx),
                    None => None,
                },
            ));
        }
        Snn { layers: layers_vec }
    }

    // To further set the inner parameters of layers' neurons.
    pub fn set_neuron_parameters(&mut self, parameters: &N::T, indexes: Option<Vec<usize>>) {
        match indexes {
            None => {
                for l in self.layers.iter_mut() {
                    l.set_neuron_parameters(parameters);
                }
            }
            Some(v) => {
                if v.len() > self.layers.len() {
                    panic!("Invalid params, too many layers, {} requested but network has only {} layers.", v.len(), self.layers.len())
                }
                for idx in v {
                    if idx > self.layers.len() {
                        panic!("Invalid params, layer of index {} requested but network has only {} layers.", idx, self.layers.len());
                    }
                    self.layers[idx].set_neuron_parameters(parameters);
                }
            }
        }
    }

    pub fn set_layer_weights(&mut self, weights: MatrixG<f64>, l_idx: usize) {
        self.layers[l_idx].set_weights(weights)
    }

    pub fn set_layer_inner_weights(&mut self, inner_weights: MatrixG<f64>, l_idx: usize) {
        self.layers[l_idx].set_inner_weights(inner_weights)
    }

    pub fn get_neuron_parameters(&mut self, l_idx: usize, n_idx: usize) -> N::T {
        self.layers[l_idx].get_neuron_parameters(n_idx)
    }

    pub fn get_layer_weights(&mut self, l_idx: usize) -> &MatrixG<f64> {
        self.layers[l_idx].get_weights()
    }

    pub fn get_layer_inner_weights(&mut self, l_idx: usize) -> &Option<MatrixG<f64>> {
        self.layers[l_idx].get_inner_weights()
    }

    // Returns the number of neurons and the presence/absence of inner weights for each layer.
    fn get_layers_info(&self) -> Vec<(usize, bool)> {
        let mut layers_info = Vec::new();
        for l in self.layers.iter() {
            layers_info.push((
                l.neurons.len(),
                if l.states_weights.is_none() {
                    false
                } else {
                    true
                },
            ));
        }
        layers_info
    }

    // Compute the number of layers to give to each thread, to better parallelize the network
    fn layer_per_thread(&mut self) -> Vec<Vec<&mut Layer<N>>> {
        // That's because if the number of layers outnumbers the number of threads
        // it would cause a growth in the overall overhead

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
        input_matrix: &MatrixG<u8>,
        fault_configuration: Option<&FaultConfiguration<N::D>>,
        log_level: u8,
        seed: Option<u64>,
    ) -> Vec<Vec<u8>> {
        // Check if the input shape it's the same as defined in the network.
        if input_matrix.cols != self.layers[0].weights.cols {
            panic!(
                "Invalid input, expected Input shape to be [{}, {}], but got [{}, {}] instead",
                input_matrix.rows,
                self.layers[0].weights.cols,
                input_matrix.rows,
                input_matrix.cols
            )
        }

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
                let a_f = f_c.get_actual_faults(self.get_layers_info(), input_matrix.rows, seed);
                if log_level == 1 || log_level == 3 {
                    println!("{}", a_f);
                }
                Some(a_f)
            }
            None => None,
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
                            Some(ref a_f) => Some(a_f),
                            None => None,
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
                // Do neuron stuff here
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

    // Run Snn start to end multiple time to fault emulation
    pub fn emulate_fault(
        &mut self,
        input_matrix: &MatrixG<u8>,
        fault_configuration: &FaultConfiguration<N::D>,
        log_level: u8, // 0: only result on stdout, 1: verbose on stdout, 2: only result on file, 3: verbose on file
        seed: Option<u64>,
    ) -> () {
        // Check if, between all possible components that can fail, the inner weights component is present
        // and if no layer has inner weights => panic
        if fault_configuration.components_contain_inner_weights()
            && self.layers.iter().all(|l| l.states_weights.is_none())
        {
            panic!("Invalid component for fault configurations: if Inner Weights is selected, be sure to initialize it in the Snn model.")
        }

        // Save components that have to be reset every run
        let saved_w = self
            .layers
            .iter()
            .map(|x| (x.weights.clone(), x.states_weights.clone()))
            .collect::<Vec<(MatrixG<f64>, Option<MatrixG<f64>>)>>();

        let mut _print_redirect;
        if log_level == 2 || log_level == 3 {
            println!("Output logs can be found in: {}", get_temp_filepath());

            let log = OpenOptions::new()
                .truncate(true)
                .read(true)
                .write(true)
                .create(true)
                .open(get_temp_filepath())
                .unwrap();

            _print_redirect = Redirect::stdout(log).unwrap();
        }

        if log_level == 1 || log_level == 3 {
            println!("{}", self);
            println!("{}", fault_configuration);
            println!("\nInput matrix:\n{}\n\n", input_matrix);
        }

        for i in 0..fault_configuration.get_n_occurrences() {
            let result =
                self.clone()
                    .forward(input_matrix, Some(fault_configuration), log_level, seed);

            println!("Result for rep {:02}: {:?}\n", i, result);

            // Restore original weights
            for (idx, l) in self.layers.iter_mut().enumerate() {
                l.weights = saved_w[idx].0.clone();
                l.states_weights = saved_w[idx].1.clone();
            }
        }
    }

    // Generates random weights matrix
    /* fn random_weights(h: u32, w: u32, diag: bool, seed: Option<u64>) -> Vec<Vec<f64>> {
        // If a seed is given, it is set for the PRNG, otherwise it
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy()
        };

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
    } */
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
                layer.weights.rows * layer.weights.cols,
                match &layer.states_weights {
                    Some(a) => a.rows * a.rows - a.rows,
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
    states_weights: Option<MatrixG<f64>>,
    weights: MatrixG<f64>,
}

impl<N: Neuron + Clone> Layer<N> {
    fn new(
        id: u32,
        neurons: u32,
        states_weights: Option<MatrixG<f64>>,
        weights: MatrixG<f64>,
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

    //################# Utility Functions ##################
    fn set_neuron_parameters(&mut self, parameters: &N::T) {
        for n in self.neurons.iter_mut() {
            n.set_parameters(parameters);
        }
    }

    fn get_neuron_parameters(&mut self, idx: usize) -> N::T {
        return self.neurons[idx].get_parameters();
    }

    fn set_weights(&mut self, weights: MatrixG<f64>) {
        if weights.rows != self.weights.rows || weights.cols != self.weights.cols {
            panic!("Invalid params, expected Weights vector shape to be [{} , {}], but got [{}, {}] instead",
                self.weights.rows,
                self.weights.cols,
                weights.rows,
                weights.cols,
            )
        }

        self.weights = weights;
    }

    fn get_weights(&mut self) -> &MatrixG<f64> {
        return &self.weights;
    }

    fn set_inner_weights(&mut self, weights: MatrixG<f64>) {
        if weights.rows != self.neurons.len() || weights.cols != self.neurons.len() {
            panic!("Invalid params, expected Weights vector shape to be [{} , {}], but got [{}, {}] instead",
                   self.neurons.len(),
                   self.neurons.len(),
                   weights.rows,
                   weights.cols,
            )
        }

        self.states_weights = Some(weights);
    }

    fn get_inner_weights(&mut self) -> &Option<MatrixG<f64>> {
        return &self.states_weights;
    }
    //#########################################################

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
                            FaultType::TransientBitFlip if a_f.time_tbf.unwrap() != time => None,
                            _ => actual_faults,
                        };
                        for n in self.neurons.iter_mut() {
                            spikes.push(n.forward(
                                &inputs,
                                &self.states_weights,
                                &self.weights,
                                &self.states,
                                if a_f.neuron_id.0 == n.get_id() {
                                    fault
                                } else {
                                    None
                                },
                            ));
                        }
                    }
                    Component::Outside(ref c) => {
                        // true is for weights, false is for states_weights
                        let mut save = (true, 0.0);
                        let mut saved_weights = (Vec::<f64>::new(), Vec::<f64>::new());
                        match c {
                            OuterComponent::Weights => match a_f.fault_type {
                                FaultType::StuckAtZero if time == 0 => {
                                    stuck_at_zero(
                                        &mut self.weights.data[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize],
                                        a_f.offset,
                                    );
                                }
                                FaultType::StuckAtOne if time == 0 => {
                                    stuck_at_one(
                                        &mut self.weights.data[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize],
                                        a_f.offset,
                                    );
                                }
                                FaultType::TransientBitFlip => {
                                    if time == a_f.time_tbf.unwrap() {
                                        save = (
                                            true,
                                            self.weights.data[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize]
                                                .clone(),
                                        );
                                        bit_flip(
                                            &mut self.weights.data[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                }
                                _ => { /* in if time != 0 we don't need to apply Stuck At-X because it was already applied */
                                }
                            },
                            OuterComponent::Connections => {
                                //##### We suppose that both weights and internal weights are passed through the same buses ######//
                                match a_f.fault_type {
                                    FaultType::StuckAtZero if time == 0 => {
                                        fault_iter(
                                            &mut self.weights.data[a_f.neuron_id.0 as usize],
                                            a_f,
                                            &stuck_at_zero,
                                        );
                                        if let Some(ref mut v) = self.states_weights {
                                            fault_iter(
                                                &mut v.data[a_f.neuron_id.0 as usize],
                                                a_f,
                                                &stuck_at_zero,
                                            );
                                        }
                                    }
                                    FaultType::StuckAtOne if time == 0 => {
                                        fault_iter(
                                            &mut self.weights.data[a_f.neuron_id.0 as usize],
                                            a_f,
                                            &stuck_at_one,
                                        );
                                        if let Some(ref mut v) = self.states_weights {
                                            fault_iter(
                                                &mut v.data[a_f.neuron_id.0 as usize],
                                                a_f,
                                                &stuck_at_one,
                                            );
                                        }
                                    }
                                    FaultType::TransientBitFlip => {
                                        if time == a_f.time_tbf.unwrap() {
                                            saved_weights.0 =
                                                self.weights.data[a_f.neuron_id.0 as usize].clone();
                                            fault_iter(
                                                &mut self.weights.data[a_f.neuron_id.0 as usize],
                                                a_f,
                                                &bit_flip,
                                            );
                                            if let Some(ref mut v) = self.states_weights {
                                                saved_weights.1 =
                                                    v.data[a_f.neuron_id.0 as usize].clone();
                                                fault_iter(
                                                    &mut v.data[a_f.neuron_id.0 as usize],
                                                    a_f,
                                                    &bit_flip,
                                                );
                                            }
                                        }
                                    }
                                    _ => { /* in if time != 0 we don't need to apply Stuck At-X because it was already applied */
                                    }
                                }
                            }
                            OuterComponent::InnerWeights => match self.states_weights {
                                Some(ref mut sw) => match a_f.fault_type {
                                    FaultType::StuckAtZero if time == 0 => {
                                        stuck_at_zero(
                                            &mut sw.data[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                    FaultType::StuckAtOne if time == 0 => {
                                        stuck_at_one(
                                            &mut sw.data[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize],
                                            a_f.offset,
                                        );
                                    }
                                    FaultType::TransientBitFlip => {
                                        if time == a_f.time_tbf.unwrap() {
                                            save = (
                                                false,
                                                sw.data[a_f.neuron_id.0 as usize]
                                                    [a_f.neuron_id.1.unwrap() as usize]
                                                    .clone(),
                                            );
                                            bit_flip(
                                                &mut sw.data[a_f.neuron_id.0 as usize]
                                                    [a_f.neuron_id.1.unwrap() as usize],
                                                a_f.offset,
                                            );
                                        }
                                    }
                                    _ => { /* in if time != 0 we don't need to apply Stuck At-X because it was already applied */
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
                            FaultType::TransientBitFlip if a_f.time_tbf.unwrap() == time => {
                                if a_f.bus.is_none() {
                                    if save.0 {
                                        self.weights.data[a_f.neuron_id.0 as usize]
                                            [a_f.neuron_id.1.unwrap() as usize] = save.1
                                    } else {
                                        if let Some(ref mut v) = self.states_weights {
                                            v.data[a_f.neuron_id.0 as usize]
                                                [a_f.neuron_id.1.unwrap() as usize] = save.1
                                        }
                                    }
                                } else {
                                    self.weights.data[a_f.neuron_id.0 as usize] = saved_weights.0;
                                    if let Some(ref mut v) = self.states_weights {
                                        v.data[a_f.neuron_id.0 as usize] = saved_weights.1;
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
