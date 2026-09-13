use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::{
    model::{NeuralParams, Stimulus, assumed_fast_sign, nt},
    snapshot::ConnectomeSnapshot,
    state::NeuralState,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct StepSummary {
    pub spike_count: usize,
    /// Diagnostic only. This aggregate is never fed back into neural dynamics,
    /// plasticity, motor output, or an action decoder.
    pub mean_modulation: f32,
    pub max_abs_membrane: f32,
}

/// Deterministic reference runtime.
///
/// The implementation is data-parallel but keeps the update order explicit:
/// propagation -> neuron state -> local plasticity -> activity trace. There is
/// no gradient, optimizer, target weight, externally supplied reward value, or
/// externally supplied positive/negative valence sign.
pub struct CpuRuntime {
    snapshot: ConnectomeSnapshot,
    params: NeuralParams,
    membrane: Vec<f32>,
    spikes: Vec<u8>,
    refractory: Vec<u32>,
    activity_trace: Vec<f32>,
    /// Coarse local dopaminergic drive at each postsynaptic neuron. The current
    /// bootstrap model derives this only from released dopaminergic presynaptic
    /// neurons and released connectivity; the environment never writes it.
    modulation: Vec<f32>,
    /// Mutable fast-synapse magnitudes. Sign comes from the current explicit
    /// transmitter model rather than being baked into the source connectome.
    weights: Vec<f32>,
    eligibility: Vec<f32>,
}

impl CpuRuntime {
    pub fn new(snapshot: ConnectomeSnapshot, params: NeuralParams) -> Self {
        let neuron_count = snapshot.neuron_count();
        let weights = snapshot
            .synapse_counts
            .iter()
            .map(|&count| count as f32 * params.synapse_scale)
            .collect();
        let edge_count = snapshot.edge_count();

        Self {
            snapshot,
            params,
            membrane: vec![0.0; neuron_count],
            spikes: vec![0; neuron_count],
            refractory: vec![0; neuron_count],
            activity_trace: vec![0.0; neuron_count],
            modulation: vec![0.0; neuron_count],
            weights,
            eligibility: vec![0.0; edge_count],
        }
    }

    pub fn snapshot(&self) -> &ConnectomeSnapshot {
        &self.snapshot
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn spikes(&self) -> &[u8] {
        &self.spikes
    }

    pub fn membrane(&self) -> &[f32] {
        &self.membrane
    }

    pub fn modulation(&self) -> &[f32] {
        &self.modulation
    }

    pub fn state(&self) -> NeuralState {
        NeuralState {
            membrane: self.membrane.clone(),
            spikes: self.spikes.iter().map(|&value| value as u32).collect(),
            refractory: self.refractory.clone(),
            activity_trace: self.activity_trace.clone(),
            modulation: self.modulation.clone(),
            weights: self.weights.clone(),
            eligibility: self.eligibility.clone(),
        }
    }

    pub fn load_state(&mut self, state: &NeuralState) -> Result<()> {
        state.validate(self.snapshot.neuron_count(), self.snapshot.edge_count())?;
        self.membrane.clone_from(&state.membrane);
        self.spikes = state.spikes.iter().map(|&value| value as u8).collect();
        self.refractory.clone_from(&state.refractory);
        self.activity_trace.clone_from(&state.activity_trace);
        self.modulation.clone_from(&state.modulation);
        self.weights.clone_from(&state.weights);
        self.eligibility.clone_from(&state.eligibility);
        Ok(())
    }

    pub fn step(&mut self, stimuli: &[Stimulus], plasticity_enabled: bool) -> Result<StepSummary> {
        let n = self.snapshot.neuron_count();
        let mut external = vec![0.0f32; n];
        for stimulus in stimuli {
            if stimulus.neuron >= n {
                bail!("stimulus neuron {} is out of range", stimulus.neuron);
            }
            external[stimulus.neuron] += stimulus.current;
        }

        let spikes_prev = &self.spikes;
        let weights = &self.weights;
        let params = self.params;
        let snapshot = &self.snapshot;

        let propagated: Vec<(f32, f32)> = (0..n)
            .into_par_iter()
            .map(|post| {
                let mut fast_current = external[post];
                let mut dopaminergic_input = 0.0f32;
                let start = snapshot.row_offsets[post] as usize;
                let end = snapshot.row_offsets[post + 1] as usize;

                for edge in start..end {
                    let pre = snapshot.pre_indices[edge] as usize;
                    if spikes_prev[pre] == 0 {
                        continue;
                    }
                    if snapshot.neurotransmitters[pre] == nt::DOPAMINE {
                        // No reward/aversive sign is supplied here. PAM, PPL1,
                        // and other dopamine neurons differ through their actual
                        // released connectivity and their own neural activity.
                        dopaminergic_input += snapshot.synapse_counts[edge] as f32
                            * params.modulator_scale;
                    } else {
                        let sign = assumed_fast_sign(snapshot.neurotransmitters[pre]);
                        fast_current += sign * weights[edge];
                    }
                }
                (fast_current, dopaminergic_input)
            })
            .collect();

        let previous_membrane = &self.membrane;
        let previous_refractory = &self.refractory;
        let updated: Vec<(f32, u8, u32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                if previous_refractory[i] > 0 {
                    return (params.reset, 0, previous_refractory[i] - 1);
                }

                let v = previous_membrane[i] * params.membrane_decay + propagated[i].0;
                if v >= params.threshold {
                    (params.reset, 1, params.refractory_steps)
                } else {
                    (v, 0, 0)
                }
            })
            .collect();

        let next_membrane: Vec<f32> = updated.iter().map(|x| x.0).collect();
        let next_spikes: Vec<u8> = updated.iter().map(|x| x.1).collect();
        let next_refractory: Vec<u32> = updated.iter().map(|x| x.2).collect();
        let next_modulation: Vec<f32> = self
            .modulation
            .par_iter()
            .enumerate()
            .map(|(i, old)| old * params.modulator_decay + propagated[i].1)
            .collect();

        if plasticity_enabled {
            let traces = &self.activity_trace;
            let modulation = &next_modulation;
            let pre_indices = &snapshot.pre_indices;
            let edge_posts = &snapshot.edge_posts;
            let neurotransmitters = &snapshot.neurotransmitters;
            let spikes_before = &self.spikes;
            let spikes_after = &next_spikes;

            self.weights
                .par_iter_mut()
                .zip(self.eligibility.par_iter_mut())
                .enumerate()
                .for_each(|(edge, (weight, eligibility))| {
                    let pre = pre_indices[edge] as usize;
                    let post = edge_posts[edge] as usize;
                    if assumed_fast_sign(neurotransmitters[pre]) == 0.0 {
                        return;
                    }

                    let local = traces[pre] * spikes_after[post] as f32
                        - traces[post] * spikes_before[pre] as f32;
                    *eligibility = *eligibility * params.eligibility_decay + local;
                    *weight = (*weight
                        + params.learning_rate * modulation[post] * *eligibility)
                        .clamp(0.0, params.weight_max);
                });
        }

        let next_trace: Vec<f32> = self
            .activity_trace
            .par_iter()
            .enumerate()
            .map(|(i, trace)| trace * params.trace_decay + next_spikes[i] as f32)
            .collect();

        let spike_count = next_spikes.par_iter().map(|&s| s as usize).sum();
        let mean_modulation = if n == 0 {
            0.0
        } else {
            next_modulation.par_iter().copied().sum::<f32>() / n as f32
        };
        let max_abs_membrane = next_membrane
            .par_iter()
            .map(|v| v.abs())
            .reduce(|| 0.0, f32::max);

        self.membrane = next_membrane;
        self.spikes = next_spikes;
        self.refractory = next_refractory;
        self.activity_trace = next_trace;
        self.modulation = next_modulation;

        Ok(StepSummary {
            spike_count,
            mean_modulation,
            max_abs_membrane,
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_ne;

    use crate::{model::nt, snapshot::EdgeInput};

    use super::*;

    fn tiny_snapshot() -> ConnectomeSnapshot {
        // sensory(0) -> association(1) -> output(2)
        // dopamine source(3) -> association(1)
        ConnectomeSnapshot::from_edges(
            4,
            &[
                EdgeInput { pre: 0, post: 1, synapse_count: 20 },
                EdgeInput { pre: 1, post: 2, synapse_count: 20 },
                EdgeInput { pre: 3, post: 1, synapse_count: 20 },
            ],
            vec![nt::ACETYLCHOLINE, nt::ACETYLCHOLINE, nt::ACETYLCHOLINE, nt::DOPAMINE],
        )
        .unwrap()
    }

    #[test]
    fn stimulation_propagates() {
        // This test is about the propagation mechanism, not calibration of the
        // whole-CNS bootstrap scale. Make the synthetic 20-synapse edge strong
        // enough to cross the 1.0 threshold in one step.
        let mut params = NeuralParams::default();
        params.synapse_scale = 0.06;
        let mut runtime = CpuRuntime::new(tiny_snapshot(), params);
        runtime.step(&[Stimulus { neuron: 0, current: 2.0 }], false).unwrap();
        assert_eq!(runtime.spikes()[0], 1);
        runtime.step(&[], false).unwrap();
        assert_eq!(runtime.spikes()[1], 1);
    }

    #[test]
    fn dopaminergic_activity_can_change_local_fast_weight() {
        let mut params = NeuralParams::default();
        params.learning_rate = 0.05;
        params.modulator_scale = 0.1;
        let mut runtime = CpuRuntime::new(tiny_snapshot(), params);
        let before = runtime.weights()[0];

        for _ in 0..50 {
            runtime
                .step(
                    &[
                        Stimulus { neuron: 0, current: 2.0 },
                        Stimulus { neuron: 3, current: 2.0 },
                    ],
                    true,
                )
                .unwrap();
            runtime.step(&[], true).unwrap();
        }

        assert_abs_diff_ne!(runtime.weights()[0], before, epsilon = 1e-6);
    }

    #[test]
    fn state_round_trip_restores_dynamics_and_plasticity_memory() {
        let mut runtime = CpuRuntime::new(tiny_snapshot(), NeuralParams::default());
        runtime
            .step(
                &[
                    Stimulus { neuron: 0, current: 2.0 },
                    Stimulus { neuron: 3, current: 2.0 },
                ],
                true,
            )
            .unwrap();
        let saved = runtime.state();
        runtime.step(&[], true).unwrap();
        runtime.load_state(&saved).unwrap();
        assert_eq!(runtime.state().spikes, saved.spikes);
        assert_eq!(runtime.state().weights, saved.weights);
        assert_eq!(runtime.state().eligibility, saved.eligibility);
    }
}
