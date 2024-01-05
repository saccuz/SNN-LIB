#[cfg(test)]
mod layer_tests {
    use snn_lib::snn::generic_matrix::MatrixG;
    use snn_lib::snn::layer::Layer;
    use snn_lib::snn::lif::{LifNeuron, LifNeuronParameters, ResetMode};

    #[test]
    fn layer_creation_and_getters() {
        let weights = MatrixG::random(3,2, false, None, 0.01, 0.99);
        let states_weights = MatrixG::random(3,3, true, None, 0.01, 0.99);
        let mut l: Layer<LifNeuron> = Layer::new(0, 3, Some(states_weights.clone()), weights.clone(), Some(&LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.8,
            r_type: ResetMode::Zero,
            tau: 0.35,
        }));

        // Getters tests
        assert_eq!(l.get_id(), 0);
        let neuron_params = l.get_neuron_parameters(0);
        assert_eq!(neuron_params.v_rest, 0.0);
        assert_eq!(neuron_params.v_th, 0.8);
        assert!(matches!(neuron_params.r_type, ResetMode::Zero));
        assert_eq!(neuron_params.tau, 0.35);
        assert_eq!(l.get_weights().data, weights.data);
        assert_eq!(l.get_states_weights().clone().unwrap().data, states_weights.data);
        assert_eq!(l.get_n_neurons(), 3);
        assert!(l.has_states_weights());


    }

    #[test]
    fn layer_setters() {
        let weights = MatrixG::random(3,2, false, None, 0.01, 0.99);
        let states_weights = MatrixG::random(3,3, true, None, 0.01, 0.99);
        let mut l: Layer<LifNeuron> = Layer::new(0, 3, Some(states_weights.clone()), weights.clone(), Some(&LifNeuronParameters {
            v_rest: 0.0,
            v_th: 0.8,
            r_type: ResetMode::Zero,
            tau: 0.35,
        }));

        // Setters tests
        let new_weights = MatrixG::random(3,4, false, None, 0.20, 0.40);
        let new_states_weights = MatrixG::random(3,3, true, None, 0.30, 0.60);
        l.set_weights(new_weights.clone());
        assert_eq!(l.get_weights().data, new_weights.data);
        l.set_states_weights(Some(new_states_weights.clone()));
        assert_eq!(l.get_states_weights().clone().unwrap().data, new_states_weights.data);
        l.set_states_weights(None);
        assert!(l.get_states_weights().is_none());
        let new_lif_params = LifNeuronParameters {
            v_rest: 0.2,
            v_th: 0.6,
            r_type: ResetMode::RestingPotential,
            tau: 0.44,
        };
        l.set_neuron_parameters(&new_lif_params);
        let neuron_params = l.get_neuron_parameters(0);
        assert_eq!(neuron_params.v_rest, 0.2);
        assert_eq!(neuron_params.v_th, 0.6);
        assert!(matches!(neuron_params.r_type, ResetMode::RestingPotential));
        assert_eq!(neuron_params.tau, 0.44);
    }

}