#[cfg(test)]
mod neuron_tests {
    use snn_lib::snn::generic_matrix::MatrixG;
    use snn_lib::snn::lif::{LifNeuronParameters, ResetMode, LifNeuron};
    use snn_lib::snn::neuron::Neuron;

    #[test]
    fn neuron_creation() {
        let p = LifNeuronParameters{v_rest: 0.0,v_th: 0.01, r_type: ResetMode::RestingPotential, tau:1.0};
        let neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(neuron.get_id(), 0);
        assert_eq!(neuron.get_parameters().v_rest, p.v_rest);
        assert_eq!(neuron.get_parameters().v_th, p.v_th);
        assert_eq!(neuron.get_parameters().tau, p.tau);
        assert!(matches!(neuron.get_parameters().r_type, ResetMode::RestingPotential));
    }

    #[test]
    fn neuron_creation_default() {
        let neuron = LifNeuron::new(3, None);
        assert_eq!(neuron.get_id(), 3);
        assert_eq!(neuron.get_parameters().v_rest, 0.0);
        assert_eq!(neuron.get_parameters().v_th, 0.8);
        assert_eq!(neuron.get_parameters().tau, 5e-3);
        assert!(matches!(neuron.get_parameters().r_type, ResetMode::Zero));
    }

    #[test]
    #[should_panic(expected = "Invalid neuron params, v_th must be strictly positive")]
    fn neuron_creation_panic_threshold() {
        let p = LifNeuronParameters{v_rest: 0.0,v_th: 0.0, r_type: ResetMode::RestingPotential, tau:1.0};
        let neuron = LifNeuron::new(0, Some(&p));
    }

    #[test]
    #[should_panic(expected = "Invalid neuron params, tau must be strictly positive")]
    fn neuron_creation_panic_tau() {
        let p = LifNeuronParameters{v_rest: 0.0,v_th: 0.01, r_type: ResetMode::RestingPotential, tau:0.0};
        let neuron = LifNeuron::new(0, Some(&p));
    }

    #[test]
    fn neuron_setter() {
        let p = LifNeuronParameters{v_rest: 0.0,v_th: 0.01, r_type: ResetMode::RestingPotential, tau:1.0};
        let mut neuron = LifNeuron::new(0, Some(&p));
        let p1 = LifNeuronParameters{v_rest: 1.0,v_th: 1.01, r_type: ResetMode::SubThreshold, tau:1.1};
        neuron.set_parameters(&p1);
        assert_eq!(neuron.get_parameters().v_rest, p1.v_rest);
        assert_eq!(neuron.get_parameters().v_th, p1.v_th);
        assert_eq!(neuron.get_parameters().tau, p1.tau);
        assert!(matches!(neuron.get_parameters().r_type, ResetMode::SubThreshold))
    }

    #[test]
    #[should_panic(expected = "Invalid neuron params, v_th must be strictly positive")]
    fn neuron_setter_wrong_v_th() {
        let p = LifNeuronParameters{v_rest: 0.0,v_th: 0.01, r_type: ResetMode::RestingPotential, tau:1.0};
        let mut neuron = LifNeuron::new(0, Some(&p));
        let p1 = LifNeuronParameters{v_rest: 1.0,v_th: -0.2, r_type: ResetMode::SubThreshold, tau:1.1};
        neuron.set_parameters(&p1);
    }

    #[test]
    #[should_panic(expected = "Invalid neuron params, tau must be strictly positive")]
    fn neuron_setter_wrong_tau() {
        let p = LifNeuronParameters{v_rest: 0.0,v_th: 0.01, r_type: ResetMode::RestingPotential, tau: 1.0};
        let mut neuron = LifNeuron::new(0, Some(&p));
        let p1 = LifNeuronParameters{v_rest: 1.0,v_th: 0.2, r_type: ResetMode::SubThreshold, tau: 0.0};
        neuron.set_parameters(&p1);
    }

    #[test]
    fn neuron_forward_zero_input() {
        let w = MatrixG::from(vec![vec![1.0,1.0,1.0], vec![1.0,1.0,1.0]]);
        let p = LifNeuronParameters{v_rest: 0.9,v_th: 0.01, r_type: ResetMode::RestingPotential, tau:1.0};
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(neuron.forward(&vec![0,0,0], &None, &w , &vec![0,0,0], None), 0);
    }

    #[test]
    fn neuron_forward() {
        let w = MatrixG::from(vec![vec![1.0,1.0,1.0], vec![1.0,1.0,1.0]]);
        let p = LifNeuronParameters{v_rest: 0.9,v_th: 0.01, r_type: ResetMode::RestingPotential, tau:1.0};
        let mut neuron = LifNeuron::new(0, Some(&p));
        assert_eq!(neuron.forward(&vec![1,1,1], &None, &w , &vec![0,0,0], None), 1);
    }

}