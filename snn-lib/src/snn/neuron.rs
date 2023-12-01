//use crate::snn;
pub trait Neuron {
    type T: NeuronParameters;
    fn new(id: String, parameters: Option<&Self::T>) -> Self;
    fn set_parameters(&mut self, parameters: &Self::T) -> ();
    fn get_parameters(&self) -> Self::T;
    fn forward(
        &mut self,
        input: &Vec<u8>,
        states_weights: &Option<Vec<Vec<f64>>>,
        weights: &Vec<Vec<f64>>,
        states: &Vec<u8>,
    ) -> u8;

    //Non è sicuro che useremo tutti questi metodi, sono messi qui nel caso in cui servissero.
    //fn get_output(&self) -> f64;
    //fn set_input(&mut self, input: f64);
    //fn get_input(&self) -> f64;
    //fn get_weights(&self) -> Vec<f64>;
    //fn set_weights(&mut self, weights: Vec<f64>);
    //fn get_bias(&self) -> f64;
    //fn set_bias(&mut self, bias: f64);
    //fn get_delta(&self) -> f64;
    //fn set_delta(&mut self, delta: f64);
    //fn get_activation(&self) -> Activation;
    //fn set_activation(&mut self, activation: Activation);
}

pub trait NeuronParameters {}
