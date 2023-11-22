use crate::snn;
pub trait Neuron {
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
    fn forward(&mut self, input : &[f64]) -> u8;
}







