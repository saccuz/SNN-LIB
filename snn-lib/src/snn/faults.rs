use crate::snn::neuron::SpecificComponent;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::fmt::Debug;
use crate::snn::lif::LifSpecificComponent;

// Which of the three basic types of fault is going to happen
#[derive(Clone, Debug)]
pub enum FaultType {
    StuckAtZero,
    StuckAtOne,
    TransientBitFlip,
}

//Components that can generate faults
#[derive(Clone, Debug)]
pub enum Component<D: SpecificComponent> {
    Inside(D),
    Outside(OuterComponent),
}

#[derive(Clone, Debug)]
pub enum OuterComponent {
    Weights,      //weights
    InnerWeights, //state_weights
    Connections,  //weight bus
}

pub struct ActualFault<D: SpecificComponent> {
    pub component: Component<D>,
    pub layer_id: u32,
    pub neuron_id: (u32, Option<u32>),
    pub fault_type: FaultType,
    pub time_tbf: Option<usize>,
    pub bus: Option<usize>,
    pub offset: u8,
}

impl<D: SpecificComponent + Clone + Debug> std::fmt::Display for ActualFault<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut component = String::new();
        match self.component.clone() {
            Component::Inside(c) => component.push_str(&format!("Inside({:?})", c)),
            Component::Outside(c) => component.push_str(&format!("Outside({:?})", c)),
        }
        write!(
            f,
            "Starting experiment with Fault: {{ \n\tcomponent: {},\n\t layer_id: {},\n\t neuron_id: {:?},\n\t fault_type: {:?},\n\t time_tbf: {:?},\n\t bus: {:?},\n\t offset: {}\n\t }}",
            component,
            self.layer_id,
            self.neuron_id,
            self.fault_type,
            self.time_tbf,
            self.bus,
            self.offset
        )
    }
}

pub struct FaultConfiguration<D: SpecificComponent + Clone + Debug> {
    components: Vec<Component<D>>,
    n_bus: usize,
    fault_type: FaultType,
    n_occurrences: u32,
}

impl<D: SpecificComponent + Clone + Debug> FaultConfiguration<D> {
    pub fn new(
        components: Vec<Component<D>>,
        n_bus: usize,
        fault_type: FaultType,
        n_occurrences: u32,
    ) -> Self {
        FaultConfiguration {
            components,
            n_bus,
            fault_type,
            n_occurrences,
        }
    }

    // Check if between faultable components there is "inner_weights"
    pub fn components_contain_inner_weights(&self) -> bool {
        let mut ret = false;
        for c in self.components.iter() {
            match c {
                Component::Outside(o) => match o {
                    OuterComponent::InnerWeights => {
                        ret = true;
                        break;
                    },
                    _ => {}
                },
                _ => {}
            }
        }
        ret
    }

    // Get number of occurrences the whole fault emulation should be repeated
    pub fn get_n_occurrences(&self) -> u32 {
        self.n_occurrences
    }

    pub fn get_actual_faults(&self, layers_info: Vec<(usize, bool)>, total_time: usize) -> ActualFault<D> {
        let mut rng = thread_rng();
        let component = (*self.components.choose(&mut rng).unwrap()).clone();
        let time_tbf = match self.fault_type {
            FaultType::TransientBitFlip => Some(rng.gen_range(0..total_time)),
            _ => None,
        };
        let offset = rng.gen_range(0..64);
        let (layer_id, neuron_id) = match component.clone() {
            Component::Inside(_) => {
                let layer_id = rng.gen_range(0..layers_info.len());
                let neuron_id = rng.gen_range(0..layers_info[layer_id]);
                (layer_id as u32, (neuron_id as u32, None))
            }
            Component::Outside(c) => {
                let layer_id = rng.gen_range(1..layers_info.len());
                let neuron_id_1 = rng.gen_range(0..layers_info[layer_id]);
                let neuron_id_2 = match c {
                    OuterComponent::Weights => rng.gen_range(0..layers_info[layer_id - 1]) as i32,
                    OuterComponent::InnerWeights => {
                        let mut neuron_2 = rng.gen_range(0..layers_info[layer_id].0);
                        if neuron_id_1 == neuron_2 {
                            if neuron_2 == layers_info[layer_id] - 1 {
                                neuron_2 -= 1;
                            } else {
                                neuron_2 += 1;
                            }
                        }
                        neuron_2 as i32
                    }
                    OuterComponent::Connections => -1,
                };

                (
                    layer_id as u32,
                    (
                        neuron_id_1 as u32,
                        if neuron_id_2 == -1 {
                            None
                        } else {
                            Some(neuron_id_2 as u32)
                        },
                    ),
                )
            }
        };

        //###### BUS NUMBER GENERATOR FOR CONNECTIONS FAULTS######//
        let bus = match component.clone() {
            Component::Outside(c) => match c {
                OuterComponent::Connections => Some(rng.gen_range(0..self.n_bus)),
                _ => None,
            },
            _ => None,
        };
        //######################################################//

        // Uncomment the following line for nice output printing
        //println!("The component selected for fault injection in this repetition is {:?}", component);
        ActualFault {
            component,
            layer_id,
            neuron_id,
            fault_type: self.fault_type.clone(),
            time_tbf,
            bus,
            offset,
        }
    }
}

impl<D: SpecificComponent + Clone + Debug> std::fmt::Display for FaultConfiguration<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut component = String::new();
        component.push_str("[");
        for c in self.components.clone() {
            match c {
                Component::Inside(c) => component.push_str(&format!("Inside({:?})", c)),
                Component::Outside(c) => component.push_str(&format!("Outside({:?})", c)),
            }
            component.push_str(", ");
        }
        component.push_str("]");
        write!(
            f,
            "{:#<100}\n\n Starting simulation with FaultConfiguration: {{ \n\tcomponents: {},\n\t n_bus: {},\n\t fault_type: {:?},\n\t n_occurrences: {}\n\t }}\n\n{:#<100} \n\n\n ",
            "",
            component,
            self.n_bus,
            self.fault_type,
            self.n_occurrences,
            ""
        )


    }
}

pub fn stuck_at_zero(x: &mut f64, offset: u8) -> () {
    //And - Tutti a 1 e il bit a 0 es: 111111111011111

    let mut value_bits = x.to_bits();

    value_bits &= !(1_u64 << offset);

    *x = f64::from_bits(value_bits);
}

pub fn stuck_at_one(x: &mut f64, offset: u8) -> () {
    //or - tutti a 0 e il bit a 1 0000000000100000

    let mut value_bits = x.to_bits();

    value_bits |= 1_u64 << offset;

    //let result = f64::from_bits(value_bits);
    *x = f64::from_bits(value_bits);
}

pub fn bit_flip(x: &mut f64, offset: u8) -> () {
    //xor - tutti a 0 e il bit a 1 es: 00001000000

    let mut value_bits = x.to_bits();

    value_bits ^= 1_u64 << offset;

    *x = f64::from_bits(value_bits);
}

pub fn apply_fault<D: SpecificComponent>(
    mut result: f64,
    actual_fault: Option<&ActualFault<D>>,
    its_me: bool,
) -> f64 {
    let ret = match actual_fault {
        Some(a_f) if its_me => {
            match a_f.fault_type {
                FaultType::StuckAtZero => stuck_at_zero(&mut result, a_f.offset),
                FaultType::StuckAtOne => stuck_at_one(&mut result, a_f.offset),
                FaultType::TransientBitFlip => bit_flip(&mut result, a_f.offset),
            }
            result
        }
        _ => result,
    };
    ret
}

pub fn fault_iter<D: SpecificComponent>(
    weights: &mut Vec<f64>,
    a_f: &ActualFault<D>,
    f: &dyn Fn(&mut f64, u8) -> (),
) {
    for i in 0..weights.len() {
        if (i + 1) % (a_f.bus.unwrap() + 1) == 0 {
            f(&mut weights[i], a_f.offset)
        }
    }
}
