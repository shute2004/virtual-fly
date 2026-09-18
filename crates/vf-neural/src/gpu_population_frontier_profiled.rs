include!("gpu_population_frontier.rs");

#[derive(Debug, Clone)]
pub struct PlasticityProfileState {
    pub trace: Vec<f32>,
    pub modulation: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub activity: Vec<u32>,
}

impl GpuPopulationRuntime {
    /// Diagnostic-only signed activity readback for one population slot.
    ///
    /// The production hot path never calls this. Values are returned as dense
    /// neuron indices whose current internal activity code is either 1 or 2;
    /// both signs propagate through the outgoing frontier.
    pub fn active_neuron_indices(&self, slot: usize) -> Result<Vec<u32>> {
        self.validate_slot(slot)?;
        let activity = self.signed_activity_codes(slot)?;
        Ok(activity
            .iter()
            .enumerate()
            .filter_map(|(index, &event)| {
                if event & 0x7fff_ffff != 0 {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .collect())
    }

    /// Diagnostic-only dense 0/1/2 activity codes for one slot.
    pub fn signed_activity_codes(&self, slot: usize) -> Result<Vec<u32>> {
        self.validate_slot(slot)?;
        let current_spikes = if self.current_is_b {
            &self.spikes_b
        } else {
            &self.spikes_a
        };
        let all = self.read_buffer::<u32>(current_spikes, self.neuron_count * self.slot_count)?;
        let start = slot * self.neuron_count;
        let end = start + self.neuron_count;
        Ok(all[start..end]
            .iter()
            .map(|event| event & 0x7fff_ffff)
            .collect())
    }

    /// Diagnostic-only state needed to size a future live-plasticity frontier.
    ///
    /// This performs large GPU readbacks and must never be used in the training
    /// hot path. Eligibility is the already-updated value for the current step;
    /// modulation is the current postsynaptic value used by that same step's
    /// weight-delta calculation.
    pub fn plasticity_profile_state(&self, slot: usize) -> Result<PlasticityProfileState> {
        self.validate_slot(slot)?;
        let all_neurons = self.read_buffer::<NeuronStateGpu>(
            &self._neuron_state_buffer,
            self.neuron_count * self.slot_count,
        )?;
        let all_synapses = self.read_buffer::<SynapseStateGpu>(
            &self._plastic_synapse_buffer,
            self.plastic_edge_count * self.slot_count,
        )?;
        let activity = self.signed_activity_codes(slot)?;

        let neuron_start = slot * self.neuron_count;
        let neuron_end = neuron_start + self.neuron_count;
        let plastic_start = slot * self.plastic_edge_count;
        let plastic_end = plastic_start + self.plastic_edge_count;

        Ok(PlasticityProfileState {
            trace: all_neurons[neuron_start..neuron_end]
                .iter()
                .map(|state| state.trace)
                .collect(),
            modulation: all_neurons[neuron_start..neuron_end]
                .iter()
                .map(|state| state.modulation)
                .collect(),
            eligibility: all_synapses[plastic_start..plastic_end]
                .iter()
                .map(|state| state.eligibility)
                .collect(),
            activity,
        })
    }
}
