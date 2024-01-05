#[cfg(test)]
mod snn_tests {
    use snn_lib::snn::faults::{Component, FaultConfiguration, FaultType, OuterComponent};
    use snn_lib::snn::generic_matrix::MatrixG;
    use snn_lib::snn::layer::Layer;
    use snn_lib::snn::lif::{LifNeuron, LifNeuronParameters, LifSpecificComponent, ResetMode};
    use snn_lib::snn::snn::Snn;

    #[test]
    fn input_matrix_creation() {
        // Randomly creating an input matrix
        let input_matrix = MatrixG::random(2, 3, false, None, 0, 1);
        // Checks matrix dimensions
        assert_eq!(input_matrix.rows, 2);
        assert_eq!(input_matrix.cols, 3);
        assert_eq!(input_matrix.cols, input_matrix[0].len());
        assert_eq!(input_matrix.rows, input_matrix.data.len());

        // Checks that the generated values are inside the given range
        for i in 0..input_matrix.rows {
            for j in 0..input_matrix.cols {
                assert!(input_matrix[i][j] <= 1);
                assert!(input_matrix[i][j] >= 0);
            }
        }
    }

    #[test]
    fn diag_matrix_creation() {
        let matrix = MatrixG::random(2, 3, true, None, 1, 2);
        // Checks the diagonal, should be 0
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                if i == j {
                    assert_eq!(matrix[i][j], 0);
                }
            }
        }
    }

    #[test]
    fn weights_matrix_creation() {
        // Randomly creating an input matrix
        let weights_matrix = MatrixG::random(2, 3, false, None, 0.01, 0.99);

        // Checks that the generated values are inside the given range
        for i in 0..weights_matrix.rows {
            for j in 0..weights_matrix.cols {
                assert!(weights_matrix[i][j] <= 0.99);
                assert!(weights_matrix[i][j] >= 0.01);
            }
        }
    }

    #[test]
    fn matrix_from() {
        let v = vec![vec![0,1], vec![1,0]];
        let m = MatrixG::from(v);
        // Checks the generation of a MatrixG from a vec of vecs
        assert_eq!(m.cols, 2);
        assert_eq!(m.rows, 2);
        assert_eq!(m[0][0], 0);
        assert_eq!(m[0][1], 1);
        assert_eq!(m[1][0], 1);
        assert_eq!(m[1][1], 0);
    }

//Test on input matrix
//Consider to test with seed

    //Test the neuron param creation

//Test on snn creation
//Consider to test with Seed

    #[test]
    fn snn_creation() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        // 1 - First way to set specific neuron parameters, different for each layer
        let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
        for _ in 0..layers.len() {
            neuron_parameters_per_layer.push(LifNeuronParameters {
                v_rest: 0.0,
                v_th: 1.5,
                r_type: ResetMode::Zero,
                tau: 0.5,
            })
        }

        // Setting personalized weights (all equal) - ONLY FOR DEBUG PURPOSES
        let mut personalized_weights = Vec::new();
        for (idx, l) in layers.iter().enumerate() {
            let mut v = Vec::new();
            for _ in 0..*l {
                if idx == 0 {
                    v.push(vec![0.40; n_inputs]);
                } else {
                    v.push(vec![0.40; layers[idx - 1] as usize]);
                }
            }
            personalized_weights.push(MatrixG::from(v));
        }

        // Setting personalized inner weights (all equal) - ONLY FOR DEBUG PURPOSES
        let mut personalized_inner_weights = Vec::new();
        for l in layers.iter() {
            personalized_inner_weights.push(Some(MatrixG::random(
                *l as usize,
                *l as usize,
                true,
                seed,
                -0.40,
                -0.20,
            )));
        }

        // Snn creation
        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers,
            layers_inner_connections,
            Some(neuron_parameters_per_layer.clone()),
            Some(personalized_weights.clone()),
            Some(personalized_inner_weights.clone()),
            seed,
        );

        assert_eq!(snn.get_neuron_parameters(0,0).v_th, neuron_parameters_per_layer[0].v_th);
        assert_eq!(snn.get_neuron_parameters(0,0).v_rest, neuron_parameters_per_layer[0].v_rest);
        assert_eq!(snn.get_neuron_parameters(0,0).tau, neuron_parameters_per_layer[0].tau);
        if let ResetMode::Zero = snn.get_neuron_parameters(0,0).r_type {
            assert!(true);
        }
        else { assert!(false) }
        //assert_eq!(snn.get_neuron_parameters(0,0).r_type, neuron_parameters_per_layer[0].r_type);


        assert_eq!(snn.get_layer_weights(0).data, personalized_weights[0].data);
        assert_eq!(snn.get_layer_states_weights(0).clone().unwrap().data, personalized_inner_weights[0].clone().unwrap().data);
    }

    #[test]
    fn snn_creation_with_neuron_per_layer() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );

        //We check the shape and if the weights are between 0.0 and 0.1 (hardcoded random function)
        for (i,l ) in layers.iter().enumerate() {
            if i == 0 {
                assert_eq!(snn.get_layer_weights(i).cols, n_inputs);
                assert_eq!(snn.get_layer_weights(i).rows, *l as usize);

            }
            else {
                assert_eq!(snn.get_layer_weights(i).cols, layers[i-1] as usize);
                assert_eq!(snn.get_layer_weights(i).rows, *l as usize);
            }
            assert_eq!(snn.get_layer_states_weights(i).clone().unwrap().rows, *l as usize);
            assert_eq!(snn.get_layer_states_weights(i).clone().unwrap().cols, *l as usize);
        }

    }
/*
    #[test]
    fn snn_creation_with_trait_from() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];
        let mut personalized_weights = Vec::new();
        for (idx, l) in layers.iter().enumerate() {
            let mut v = Vec::new();
            for _ in 0..*l {
                if idx == 0 {
                    v.push(vec![0.40; n_inputs]);
                } else {
                    v.push(vec![0.40; layers[idx - 1] as usize]);
                }
            }
            personalized_weights.push(MatrixG::from(v));
        }



        let vec = Vec::with_capacity(layers.len());
        for (i,v) in layers.iter().enumerate() {

        }

        let snn = Snn::from(vec)
    }*/

    //-- panicking test
    #[test]
    #[should_panic]
    fn snn_creation_with_wrong_weight_shape_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        // 1 - First way to set specific neuron parameters, different for each layer
        let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
        for _ in 0..layers.len() {
            neuron_parameters_per_layer.push(LifNeuronParameters {
                v_rest: 0.0,
                v_th: 1.5,
                r_type: ResetMode::Zero,
                tau: 0.5,
            })
        }

        // Setting personalized weights (all equal) - ONLY FOR DEBUG PURPOSES
        let mut personalized_weights = Vec::new();
        for (idx, l) in layers.iter().enumerate() {
            let mut v = Vec::new();
            for _ in 0..5 {
                if idx == 0 {
                    v.push(vec![0.40; n_inputs]);
                } else {
                    v.push(vec![0.40; layers[idx - 1] as usize]);
                }
            }
            personalized_weights.push(MatrixG::from(v));
        }

        // Setting personalized inner weights (all equal) - ONLY FOR DEBUG PURPOSES
        let mut personalized_inner_weights = Vec::new();
        for l in layers.iter() {
            personalized_inner_weights.push(Some(MatrixG::random(
                *l as usize,
                *l as usize,
                true,
                seed,
                -0.40,
                -0.20,
            )));
        }

        // Snn creation
        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers,
            layers_inner_connections,
            Some(neuron_parameters_per_layer.clone()),
            Some(personalized_weights.clone()),
            Some(personalized_inner_weights.clone()),
            seed,
        );

        assert_eq!(snn.get_neuron_parameters(0,0).v_th, neuron_parameters_per_layer[0].v_th);
        assert_eq!(snn.get_neuron_parameters(0,0).v_rest, neuron_parameters_per_layer[0].v_rest);
        assert_eq!(snn.get_neuron_parameters(0,0).tau, neuron_parameters_per_layer[0].tau);
        if let ResetMode::Zero = snn.get_neuron_parameters(0,0).r_type {
            assert!(true);
        }
        else { assert!(false) }
        //assert_eq!(snn.get_neuron_parameters(0,0).r_type, neuron_parameters_per_layer[0].r_type);


        assert_eq!(snn.get_layer_weights(0).data, personalized_weights[0].data);
        assert_eq!(snn.get_layer_states_weights(0).clone().unwrap().data, personalized_inner_weights[0].clone().unwrap().data);
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_wrong_state_weight_shape_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 10, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        // 1 - First way to set specific neuron parameters, different for each layer
        let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
        for _ in 0..layers.len() {
            neuron_parameters_per_layer.push(LifNeuronParameters {
                v_rest: 0.0,
                v_th: 1.5,
                r_type: ResetMode::Zero,
                tau: 0.5,
            })
        }

        // Setting personalized weights (all equal) - ONLY FOR DEBUG PURPOSES
        let mut personalized_weights = Vec::new();
        for (idx, l) in layers.iter().enumerate() {
            let mut v = Vec::new();
            for _ in 0..*l {
                if idx == 0 {
                    v.push(vec![0.40; n_inputs]);
                } else {
                    v.push(vec![0.40; layers[idx - 1] as usize]);
                }
            }
            personalized_weights.push(MatrixG::from(v));
        }

        // Setting personalized inner weights (all equal) - ONLY FOR DEBUG PURPOSES
        let mut personalized_inner_weights = Vec::new();
        for l in layers.iter() {
            personalized_inner_weights.push(Some(MatrixG::random(
                3 as usize,
                5 as usize,
                true,
                seed,
                -0.40,
                -0.20,
            )));
        }

        // Snn creation
        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers,
            layers_inner_connections,
            Some(neuron_parameters_per_layer.clone()),
            Some(personalized_weights.clone()),
            Some(personalized_inner_weights.clone()),
            seed,
        );

        assert_eq!(snn.get_neuron_parameters(0,0).v_th, neuron_parameters_per_layer[0].v_th);
        assert_eq!(snn.get_neuron_parameters(0,0).v_rest, neuron_parameters_per_layer[0].v_rest);
        assert_eq!(snn.get_neuron_parameters(0,0).tau, neuron_parameters_per_layer[0].tau);
        if let ResetMode::Zero = snn.get_neuron_parameters(0,0).r_type {
            assert!(true);
        }
        else { assert!(false) }
        //assert_eq!(snn.get_neuron_parameters(0,0).r_type, neuron_parameters_per_layer[0].r_type);


        assert_eq!(snn.get_layer_weights(0).data, personalized_weights[0].data);
        assert_eq!(snn.get_layer_states_weights(0).clone().unwrap().data, personalized_inner_weights[0].clone().unwrap().data);
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_layer_with_0_neurons_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 0, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_with_0_inputs_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 0;
        let layers = vec![20, 0, 5, 3];
        let layers_inner_connections = vec![true; layers.len()];

        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_0_layers_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![];
        let layers_inner_connections = vec![true; layers.len()];

        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );
    }

    #[test]
    #[should_panic]
    fn snn_creation_with_wrong_inner_connection_vector_shape_should_panic() {
        let seed = Some(21);
        let n_inputs: usize = 10;
        let layers = vec![20, 0, 5, 3];
        let layers_inner_connections = vec![true; layers.len()-1];

        let snn = Snn::<LifNeuron>::new(
            n_inputs as u32,
            layers.clone(),
            layers_inner_connections,
            None,
            None,
            None,
            seed,
        );
    }


//Test on a single forward

//Test on a Actual fault

//Test on a Stuckat0

//Test on a Stuckat1

//Test on a Bitflip

//Consider to test all the components

}