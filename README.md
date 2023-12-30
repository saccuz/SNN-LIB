# Group45

# Spiking Neural Network + Resilience Library
## snn-lib

- [Group45](#group45)
- [Description](#description)
- [Framework](#framework)
- [Main structs](#main-structures)
- [Main methods](#main-methods)
- [Network parameters setting](#network-parameters-setting)
- [Usage examples](#usage-examples)
- [Licence](#licence)

## Group45
- Francesco Sorrentino (s301665)
- Marco Sacchet (s295033)
- Simone Mascali (s305001)

## Description
This repository contains a Rust library for the study of the resilience of a Spiking Neural Network in terms of hardware faults that can happen during inference.

## Network parameters setting
### Seed setting
If you want to reproduce a certain behavior or check the difference of outputs with a specific network setting, 
you can set a specific seed and pass it to the proper function for the random generations used inside the network parameters setting.\
You can set the seed for the generation of: input matrix, weights and inner weights.\
Note that it should be an ```Option<u64>```, if None is given then the default setting for the PRNG (Pseudo Random Number Generator) is used.

```rust
// Set the seed for input matrix, weights, and inner weights 
let seed = Some(21);
...
// Passing the seed here set that seed for the input matrix generation
let input_matrix = MatrixG::random(rows, cols, false, seed, 0, 1);
// AND/OR
// Passing the seed here set that seed for weights and inner weights generation.
let mut snn = Snn::<LifNeuron>::new(..., seed);
```

### Log level setting
You can set the log level by setting this value:
- 0: print only the results of emulation on `stdout`, without any details about the network's structure or the injected faults
- 1: print on `stdout` network structure, requested injected fault, input matrix and the applied fault for every result.
- 2: print only the results of emulation on a log file, without any details about the network's structure or the injected faults
- 3: print on a log file network structure, requested injected fault, input matrix and the applied fault for every result.
```rust
// Setting the log level
let log_level = 0;
```

### Basic configuration of the SNN
You must set the ```n_inputs``` as the number of neurons in the input layer, ```layers``` vector as vector of neurons per inner layer, and ```layers_inner_connection``` as parallel information about each layer:
```true``` if inner connections are present, ```false``` otherwise.
```rust
// Configuring the Snn
let n_inputs: usize = 20;
let layers = vec![15,10,5,2];
let layers_inner_connections = vec![true, false, true, true];
```

### Weights setting
If you want to set specific weights for the network you can create a ```Vec<MatrixG>```.\
You must build each MatrixG using ```MatrixG::from(Vec<Vec<T>>```) where ```vec[i][j]``` represents the weight connection between neuron i of layer n and neuron j of layer n-1.
```rust
let mut personalized_weights = Vec::new();
personalized_weights.push(MatrixG::from(vec![vec![...], vec![...], ...]));
...
let mut snn = Snn::<LifNeuron>::new(..., Some(personalized_weights), ...);
```

### Inner weights setting
If you want to set specific inner weights for the network you can create a ```Vec<MatrixG>```.\
You must build each MatrixG using ```MatrixG::from(Vec<Vec<T>>```) where ```vec[i][j]``` represents the inner weight connection between neuron i and neuron j of the same layer,
the matrix is a square matrix and the diagonal should be 0 otherwise it would represent the inner weights of a neuron with itself.
```rust
let mut personalized_inner_weights = Vec::new();
personalized_inner_weights.push(MatrixG::from(vec![vec![...], vec![...], ...]));
...
let mut snn = Snn::<LifNeuron>::new(..., Some(personalized_inner_weights), ...);
```

### Neuron parameters setting
There is a specific struct to define neuron specific parameters (Lif in this case):

```rust
let parameters_for_lif = LifNeuronParameters {
    v_rest: 0.0,
    v_th: 0.8,
    r_type: ResetMode::Zero,
    tau: 0.35,
};
```

And then you have multiple ways to set parameters of neurons inside the network layers:

```rust
// 1 - First way: set different neuron parameters for each layer
let mut neuron_parameters_per_layer = Vec::<LifNeuronParameters>::new();
for _ in 0..layers.len() {
    neuron_parameters_per_layer.push(LifNeuronParameters { ... })
}
let mut snn = Snn::<LifNeuron>::new(..., Some(neuron_parameters_per_layer), ...);

// 2 - Second way: set the same parameters for the whole network (same neuron parameters for all layers)
let parameters_for_all_lif = LifNeuronParameters { ... };
snn.set_neuron_parameters(parameters_for_all_lif, None);

// 3 - Third way: set the same parameters for specific layers in the network (specified by a vector of layer indexes)
let parameters_for_lif_of_some_layers = LifNeuronParameters { ... };
snn.set_neuron_parameters(parameters_for_lif_of_some_layers, Some(vec![...]));
```
### Fault configuration setting
The FaultConfiguration struct is used to specify the fault configuration that has to be emulated.\
The vector of Component represents the component that could present a fault during the emulation.\
The bus number is the number of bus and one of these bus could present a fault during the emulation.\
The fault type is an enum that specifies which type of fault has to occur during the emulation.\
The last number is the desired number of repetitions of the fault emulation. 

```rust
let fault_configuration = FaultConfiguration::new(
    vec![
        Component::Inside(LifSpecificComponent::Adder),
        Component::Inside(LifSpecificComponent::Divider),
        Component::Inside(LifSpecificComponent::Multiplier),
        Component::Inside(LifSpecificComponent::Comparator),
        Component::Inside(LifSpecificComponent::Membrane),
        Component::Inside(LifSpecificComponent::Rest),
        Component::Inside(LifSpecificComponent::Threshold),
        Component::Outside(OuterComponent::Weights),
        Component::Outside(OuterComponent::InnerWeights),
        Component::Outside(OuterComponent::Connections),
    ],
    8, //Bus number
    FaultType::TransientBitFlip, //or FaultType::StuckAtZero or FaultType::StuckAtOne 
    100,
);
```

## Usage examples
See ```main.rs``` file for a complete example of usage.\
To create a SNN configured with all parameters explained before:

```rust
// Snn creation
let mut snn = Snn::<LifNeuron>::new(n_inputs, layers, layers_inner_connections, Some(neuron_parameters_per_layer), Some(personalized_weights), Some(personalized_inner_weights), seed);
```

To randomly create an input matrix:
```rust
 // Randomly creating an input matrix
let input_matrix = MatrixG::random(length, n_inputs, false, seed, 0, 1);
```

To run a simple network inference over the input matrix:

```rust
// To start the inference without faults
println!("The final result is: {:?}", snn.forward(&input_matrix, None, log_level, None));
```

For emulation of faults:

```rust
// Fault injection
let fault_configuration = FaultConfiguration::new(
    vec![
        Component::Inside(LifSpecificComponent::Adder),
        Component::Inside(LifSpecificComponent::Divider),
        Component::Inside(LifSpecificComponent::Threshold),
        Component::Outside(OuterComponent::Weights),
    ],
    8,
    FaultType::TransientBitFlip,
    100,
);

// To start the n_occurrences faults emulations
snn.emulate_fault(&input_matrix, &fault_configuration, 1);
```

###

## Licence
Released under MIT Licence.\
Note: this is a university project in the partial fulfillment of a course exam, no guarantees on the absence of errors or bugs that may not have been found and considered yet.