use crate::snn::faults::{apply_fault, ActualFault, Component};
use crate::snn::neuron::NeuronParameters;
use crate::snn::neuron::{Neuron, SpecificComponent};

#[derive(Clone, Copy)]
pub enum ResetMode {
    Zero,
    RestingPotential,
    SubThreshold,
}

#[derive(Clone)]
pub struct LifNeuronParameters {
    pub v_rest: f64,
    pub v_th: f64,
    pub r_type: ResetMode,
    pub tau: f64,
}

impl NeuronParameters for LifNeuronParameters {}

#[derive(Clone, PartialEq, Debug)]
pub enum LifSpecificComponent {
    //TODO: Capire se conviene generalizzare queste operazioni ed includerle di default nella libreria
    Adder,      //add
    Multiplier, //mul
    Divider,    //div
    Comparator, //compare
    //################################################################################################
    Threshold,  //v_th
    Membrane,   //v_mem
    Rest,       //v_rest
}

impl SpecificComponent for LifSpecificComponent {}

#[derive(Clone)]
pub struct LifNeuron {
    id: u32,
    v_mem: f64,
    v_rest: f64,
    v_th: f64,
    r_type: ResetMode,
    t_s_last: u64, //t_s_last nel nostro caso funge come variazione di tempo dall'ultima spike. Quindi è uguale a t-t_last
    tau: f64,
}

impl LifNeuron {
    fn scalar_product(
        inputs: &Vec<u8>,
        weights: &Vec<f64>,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
        ops: &Vec<bool>,
    ) -> f64 {
        let mut scalar = 0.0;

        for (idx, x) in inputs.iter().enumerate() {
            //somma per ogni neurone
            scalar = LifNeuron::add(
                scalar,
                LifNeuron::mul(*x as f64, weights[idx], actual_fault, ops[1]),
                actual_fault,
                ops[0],
            ); // moltiplica la spike per il peso dell'input
        }

        scalar
    }

    fn y(
        inputs: &Vec<u8>,
        states: &Vec<u8>,
        weights: &Vec<f64>,
        states_weights: Option<&Vec<f64>>,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
        ops: &Vec<bool>,
    ) -> f64 {
        //TODO: AGGIUNGERE CONTROLLI SUI VARI PARAMETRI

        //prodotto scalare tra input e pesi degli input
        let out = LifNeuron::scalar_product(inputs, weights, actual_fault, ops);

        match states_weights {
            //se è presente il vettore dei pesi degli stati
            Some(states_weights) => {
                let sum1 = LifNeuron::scalar_product(states, states_weights, actual_fault, ops);
                LifNeuron::add(out, sum1, actual_fault, ops[0])
            }
            //se non è presente il vettore dei pesi degli stati
            None => out,
        }
        //Out è la combinazione lineare tra il vettore degli input e il vettore dei pesi associati
    }

    //################################# OPERAZIONI SIMULATE #################################//
    //#region Operazioni
    fn add(
        x: f64,
        y: f64,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
        its_me: bool,
    ) -> f64 {
        apply_fault(x + y, actual_fault, its_me)
    }
    fn mul(
        x: f64,
        y: f64,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
        its_me: bool,
    ) -> f64 {
        apply_fault(x * y, actual_fault, its_me)
    }

    fn div(
        x: f64,
        y: f64,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
        its_me: bool,
    ) -> f64 {
        apply_fault(x / y, actual_fault, its_me)
    }

    fn compare(
        x: f64,
        y: f64,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
        its_me: bool,
    ) -> u8 {
        apply_fault(((x > y) as u8) as f64, actual_fault, its_me) as u8
    }

    //#endregion
    //########################################################################################//
}

impl Neuron for LifNeuron {
    type T = LifNeuronParameters;
    type D = LifSpecificComponent;
    // Creates a new LifNeuron
    // (t_s_last is set to 0 by default at the beginning, no previous impulse received from the beginning of the snn existence)
    fn new(id: u32, parameters: Option<&LifNeuronParameters>) -> Self {
        match parameters {
            Some(p) => LifNeuron {
                id,
                v_mem: 0.0,
                v_rest: p.v_rest,
                v_th: p.v_th,
                r_type: ResetMode::Zero,
                t_s_last: 0,
                tau: p.tau,
            },
            None => LifNeuron {
                id,
                v_mem: 0.0,
                v_rest: 0.0,
                v_th: 0.8,
                r_type: ResetMode::Zero,
                t_s_last: 0,
                tau: 0.0,
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

    fn get_id(&self) -> u32 {
        self.id
    }

    // implements the forward pass of the snn
    fn forward(
        &mut self,
        input: &Vec<u8>,
        states_weights: &Option<Vec<Vec<f64>>>,
        weights: &Vec<Vec<f64>>,
        states: &Vec<u8>,
        actual_fault: Option<&ActualFault<LifSpecificComponent>>,
    ) -> u8 {
        self.t_s_last += 1;

        let mut ops = vec![false; 7];
        match actual_fault {
            Some(a_f) => match &a_f.component {
                Component::Inside(real_comp) => match real_comp {
                    LifSpecificComponent::Adder => ops[0] = true,
                    LifSpecificComponent::Multiplier => ops[1] = true,
                    LifSpecificComponent::Comparator => ops[2] = true,
                    LifSpecificComponent::Threshold => ops[3] = true,
                    LifSpecificComponent::Membrane => ops[4] = true,
                    LifSpecificComponent::Rest => ops[5] = true,
                    LifSpecificComponent::Divider => ops[6] = true,
                },
                _ => {}
            },
            None => {}
        }
        // Salviamo i valori effettivi v_mem, v_th, v_rest
        self.v_th = apply_fault(self.v_th, actual_fault, ops[3]);
        self.v_mem = apply_fault(self.v_mem, actual_fault, ops[4]);
        self.v_rest = apply_fault(self.v_rest, actual_fault, ops[5]);

        // Input impulses summation
        let n_neuron = self.id as usize;
        let summation = match states_weights {
            Some(states_weights) => LifNeuron::y(
                input,
                states,
                &weights[n_neuron],
                Some(&states_weights[n_neuron]),
                actual_fault,
                &ops,
            ),
            None => LifNeuron::y(input, states, &weights[n_neuron], None, actual_fault, &ops),
        };

        let exponent: f64 =
            LifNeuron::div(-((self.t_s_last) as f64), self.tau, actual_fault, ops[6]);

        // rest + (mem - rest) * exp(dt/tau) + sum(w*x -wi*xi)
        // Questa operazione:
        // self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * exponent.exp() + summation;
        // è stata divisa in più parti ed ogni operazione è stata simulata con l'eventuale fault
        self.v_mem = LifNeuron::add(
            LifNeuron::add(
                LifNeuron::mul(
                    LifNeuron::add(self.v_mem, -self.v_rest, actual_fault, ops[0]),
                    exponent.exp(),
                    actual_fault,
                    ops[1],
                ),
                self.v_rest,
                actual_fault,
                ops[0],
            ),
            summation,
            actual_fault,
            ops[0],
        );

        let spike = LifNeuron::compare(self.v_mem, self.v_th, actual_fault, ops[2]); //if v_mem>v_th then spike=1 else spike=0

        if spike == 1 {
            self.t_s_last = 0;
            self.v_mem = match self.r_type {
                ResetMode::Zero => 0.0,
                ResetMode::RestingPotential => self.v_rest,
                ResetMode::SubThreshold => {
                    LifNeuron::add(self.v_mem, -self.v_th, actual_fault, ops[0])
                }
            }
        }
        // Corrupting v_mem memory when the value is written back to memory
        self.v_mem = apply_fault(self.v_mem, actual_fault, ops[4]);

        spike
    }
}
