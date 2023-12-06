use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

// Which of the three basic types of fault is going to happen
#[derive(Clone)]
pub enum FaultType {
    StuckAtZero,
    StuckAtOne,
    TransientBitFlip,
}

// Components that can generate faults
#[derive(Clone)]
pub enum Component {
    Inside(InnerComponent),
    Outside(OuterComponent),
}

#[derive(Clone)]
pub enum InnerComponent {
    Adder,      //add
    Multiplier, //mul
    Divider,
    Comparator, //compare
    Threshold,  //v_th
    Membrane,   //v_mem
    Rest,       //v_rest
}

#[derive(Clone)]
pub enum OuterComponent {
    Weights,          //weights
    InnerWeights,     //state_weights
    Connections,      //weight = 0
    InnerConnections, //state_weight = 0
}

pub struct ActualFault {
    pub component: Component,
    pub layer_id: u32,
    pub neuron_id: (u32, Option<u32>),
    pub fault_type: FaultType,
    pub time_tbf: Option<usize>,
    pub offset: u8,
}

pub struct FaultConfiguration {
    components: Vec<Component>,
    fault_type: FaultType,
    n_occurrences: u32,
}

impl FaultConfiguration {
    pub fn new(components: Vec<Component>, fault_type: FaultType, n_occurrences: u32) -> Self {
        FaultConfiguration {
            components,
            fault_type,
            n_occurrences,
        }
    }

    pub fn get_n_occurrences(&self) -> u32 {
        self.n_occurrences
    }

    pub fn get_actual_faults(&self, layers_info: Vec<usize>, total_time: usize) -> ActualFault {
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
                    OuterComponent::Weights | OuterComponent::Connections => {
                        rng.gen_range(0..layers_info[layer_id - 1])
                    }
                    OuterComponent::InnerWeights | OuterComponent::InnerConnections => {
                        let mut neuron_2 = rng.gen_range(0..layers_info[layer_id]);
                        if neuron_id_1 == neuron_2 {
                            if neuron_2 == layers_info[layer_id] - 1 {
                                neuron_2 -= 1;
                            } else {
                                neuron_2 += 1;
                            }
                        }
                        neuron_2
                    }
                };
                (
                    layer_id as u32,
                    (neuron_id_1 as u32, Some(neuron_id_2 as u32)),
                )
            }
        };
        ActualFault {
            component,
            layer_id,
            neuron_id,
            fault_type: self.fault_type.clone(),
            time_tbf,
            offset,
        }
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

    let result = f64::from_bits(value_bits);
    *x = f64::from_bits(value_bits);
}

pub fn bit_flip(x: &mut f64, offset: u8) -> () {
    //xor - tutti a 0 e il bit a 1 es: 00001000000

    let mut value_bits = x.to_bits();

    value_bits ^= 1_u64 << offset;

    *x = f64::from_bits(value_bits);
}

pub fn apply_fault(mut result: f64, actual_fault: Option<&ActualFault>, its_me: bool) -> f64 {
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
