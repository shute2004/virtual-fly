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
    /// Activity-event code per neuron: 0=silent, 1=positive/depolarizing
    /// deviation, 2=negative/hyperpolarizing deviation. The runtime still
    /// exposes only code 1 as a motor-neuron spike to the body boundary.
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
            .find(|(_, value)| *value > 2)
        {
            bail!("state activity event at neuron {index} is {value}; expected 0, 1, or 2");
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

    /// Clear transient neural/plasticity memory while preserving learned synaptic weights.
    ///
    /// This is intended for frozen-behavior evaluation where the learned weights should
    /// be compared from the same quiescent initial condition as an untrained nervous
    /// system. Full checkpoint resume must not call this method.
    pub fn reset_dynamics_preserving_weights(&mut self) {
        self.membrane.fill(0.0);
        self.spikes.fill(0);
        self.refractory.fill(0);
        self.activity_trace.fill(0.0);
        self.modulation.fill(0.0);
        self.eligibility.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::NeuralState;

    #[test]
    fn dynamics_reset_preserves_only_weights() {
        let mut state = NeuralState {
            membrane: vec![1.0, -2.0],
            spikes: vec![1, 2],
            refractory: vec![2, 3],
            activity_trace: vec![0.4, -0.5],
            modulation: vec![-0.6, 0.7],
            weights: vec![3.0, 4.0, 5.0],
            eligibility: vec![0.8, -0.9, 1.0],
        };
        let learned_weights = state.weights.clone();

        state.reset_dynamics_preserving_weights();

        assert_eq!(state.weights, learned_weights);
        assert_eq!(state.membrane, vec![0.0; 2]);
        assert_eq!(state.spikes, vec![0; 2]);
        assert_eq!(state.refractory, vec![0; 2]);
        assert_eq!(state.activity_trace, vec![0.0; 2]);
        assert_eq!(state.modulation, vec![0.0; 2]);
        assert_eq!(state.eligibility, vec![0.0; 3]);
    }
}
