use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cpu,
    Gpu,
    Auto,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Stimulus {
    pub neuron: usize,
    pub current: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NeuralParams {
    /// Per-step membrane retention. This is a numerical model parameter, not an observed MaleCNS value.
    pub membrane_decay: f32,
    pub threshold: f32,
    pub reset: f32,
    pub refractory_steps: u32,
    /// Activity trace used by the first local three-factor plasticity rule.
    pub trace_decay: f32,
    pub eligibility_decay: f32,
    pub learning_rate: f32,
    /// Mutable fast-synapse magnitude clamp.
    pub weight_max: f32,
    /// Neuromodulator concentration decay in the current coarse model.
    pub modulator_decay: f32,
    /// Initial fast synaptic magnitude per released MaleCNS synapse count.
    pub synapse_scale: f32,
    /// Neuromodulatory influence per released MaleCNS synapse count.
    pub modulator_scale: f32,
}

impl Default for NeuralParams {
    fn default() -> Self {
        Self {
            membrane_decay: 0.95,
            threshold: 1.0,
            reset: 0.0,
            refractory_steps: 2,
            trace_decay: 0.95,
            eligibility_decay: 0.995,
            learning_rate: 0.0005,
            // Numerical safety rail, not a biological measurement.
            weight_max: 1_000.0,
            modulator_decay: 0.98,
            // Bootstrap scaling. These values are intentionally explicit and
            // will be calibrated against whole-CNS activity after the first run.
            synapse_scale: 0.02,
            modulator_scale: 0.005,
        }
    }
}

/// Compact neurotransmitter code written by the MaleCNS preprocessor.
///
/// These values encode released annotations. The mapping from transmitter to
/// fast excitatory/inhibitory sign is an explicit modelling assumption and is
/// intentionally kept out of the dataset snapshot itself.
pub mod nt {
    pub const UNKNOWN: u8 = 0;
    pub const ACETYLCHOLINE: u8 = 1;
    pub const GABA: u8 = 2;
    pub const GLUTAMATE: u8 = 3;
    pub const DOPAMINE: u8 = 4;
    pub const OCTOPAMINE: u8 = 5;
    pub const SEROTONIN: u8 = 6;
    pub const HISTAMINE: u8 = 7;
}

#[inline]
pub fn assumed_fast_sign(transmitter: u8) -> f32 {
    match transmitter {
        nt::ACETYLCHOLINE => 1.0,
        nt::GABA | nt::GLUTAMATE | nt::HISTAMINE => -1.0,
        _ => 0.0,
    }
}
