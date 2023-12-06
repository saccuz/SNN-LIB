use crate::snn::faults::{ActualFault, InnerComponent};

//use crate::snn;
pub trait Neuron {
    type T: NeuronParameters;
    fn new(id: u32, parameters: Option<&Self::T>) -> Self;
    fn set_parameters(&mut self, parameters: &Self::T) -> ();
    fn get_parameters(&self) -> Self::T;
    fn get_id(&self) -> u32;

    //fn restore_param(comp: InnerComponent) -> ();
    fn forward(
        &mut self,
        input: &Vec<u8>,
        states_weights: &Option<Vec<Vec<f64>>>,
        weights: &Vec<Vec<f64>>,
        states: &Vec<u8>,
        actual_fault: Option<&ActualFault>,
    ) -> u8;
}

pub trait NeuronParameters {}
