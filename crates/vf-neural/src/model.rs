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

/// Compact event code used by the bootstrap neural runtime.
///
/// MaleCNS contains both spiking and strongly graded neurons. In particular,
/// Drosophila R1-R6 photoreceptors and lamina monopolar cells communicate with
/// graded changes around an ongoing baseline. A one-sided 0/1 spike code cannot
/// carry an inhibitory light response through that circuit from a quiescent
/// numerical origin, so the runtime stores the sign of an activity *deviation*.
/// Positive motor-neuron events remain the only events exposed as muscle spikes.
pub const ACTIVITY_SILENT: u8 = 0;
pub const ACTIVITY_DEPOLARIZING: u8 = 1;
pub const ACTIVITY_HYPERPOLARIZING: u8 = 2;

#[inline]
pub fn activity_sign_u8(event: u8) -> f32 {
    match event {
        ACTIVITY_DEPOLARIZING => 1.0,
        ACTIVITY_HYPERPOLARIZING => -1.0,
        _ => 0.0,
    }
}

#[inline]
pub fn activity_sign_u32(event: u32) -> f32 {
    match event {
        1 => 1.0,
        2 => -1.0,
        _ => 0.0,
    }
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

fn synapse_scale_override(default: f32) -> f32 {
    let Ok(raw) = std::env::var("VF_NEURAL_SYNAPSE_SCALE") else {
        return default;
    };
    let value = raw
        .parse::<f32>()
        .unwrap_or_else(|_| panic!("VF_NEURAL_SYNAPSE_SCALE must be a finite positive f32"));
    assert!(
        value.is_finite() && value > 0.0,
        "VF_NEURAL_SYNAPSE_SCALE must be a finite positive f32"
    );
    value
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
            learning_rate: 0.00001,
            weight_max: 1_000.0,
            modulator_decay: 0.98,
            // 0.02 is retained only as the historical bootstrap fallback. The
            // Flyppy launcher now calibrates this task-independently from a
            // one-pulse -> zero-input whole-CNS stability probe and exports the
            // selected value through VF_NEURAL_SYNAPSE_SCALE before constructing
            // any runtime. This changes only the numerical connectome scale; it
            // does not use reward, behavior, or a target action.
            synapse_scale: synapse_scale_override(0.02),
            modulator_scale: 0.00005,
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
