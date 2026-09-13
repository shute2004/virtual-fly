use anyhow::{Result, bail};

/// Serializable-by-the-runner state required to continue one virtual nervous
/// system without resetting fast dynamics or plasticity traces.
///
/// Topology, neurotransmitter annotations, modulator-role configuration, and
/// numerical model parameters are not duplicated here. They are supplied by the
/// same snapshot/configuration when a checkpoint is restored.
#[derive(Debug, Clone)]
pub struct NeuralState {
    pub membrane: Vec<f32>,
    pub spikes: Vec<u32>,
    pub refractory: Vec<u32>,
    pub activity_trace: Vec<f32>,
    pub modulation: Vec<f32>,
    pub weights: Vec<f32>,
    pub eligibility: Vec<f32>,
}

impl NeuralState {
    pub fn validate(&self, neuron_count: usize, edge_count: usize) -> Result<()> {
        for (name, len) in [
            ("membrane", self.membrane.len()),
            ("spikes", self.spikes.len()),
            ("refractory", self.refractory.len()),
            ("activity_trace", self.activity_trace.len()),
            ("modulation", self.modulation.len()),
        ] {
            if len != neuron_count {
                bail!("state {name} length {len} != neuron count {neuron_count}");
            }
        }
        for (name, len) in [
            ("weights", self.weights.len()),
            ("eligibility", self.eligibility.len()),
        ] {
            if len != edge_count {
                bail!("state {name} length {len} != edge count {edge_count}");
            }
        }
        if let Some((index, value)) = self
            .spikes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value > 1)
        {
            bail!("state spike at neuron {index} is {value}; expected 0 or 1");
        }
        if self
            .membrane
            .iter()
            .chain(self.activity_trace.iter())
            .chain(self.modulation.iter())
            .chain(self.weights.iter())
            .chain(self.eligibility.iter())
            .any(|value| !value.is_finite())
        {
            bail!("state contains non-finite floating-point values");
        }
        Ok(())
    }
}
