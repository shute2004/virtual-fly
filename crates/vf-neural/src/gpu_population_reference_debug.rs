include!("gpu_population_sparse.rs");

#[derive(Debug, Clone)]
pub struct PopulationDebugState {
    pub membrane: Vec<f32>,
    pub trace: Vec<f32>,
    pub modulation: Vec<f32>,
    pub refractory: Vec<u32>,
    pub spikes: Vec<u32>,
    pub plastic_weight: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub transaction_shift: Vec<f32>,
    pub transaction_lo: Vec<f32>,
    pub transaction_hi: Vec<f32>,
}

impl GpuPopulationRuntime {
    /// Test/debug-only full slot state. Not used by the training hot path.
    pub fn debug_slot_state(&self, slot: usize) -> Result<PopulationDebugState> {
        self.validate_slot(slot)?;
        let all_neurons = self.read_buffer::<NeuronStateGpu>(
            &self._neuron_state_buffer,
            self.neuron_count * self.slot_count,
        )?;
        let current_spikes = if self.current_is_b { &self.spikes_b } else { &self.spikes_a };
        let all_spikes = self.read_buffer::<u32>(current_spikes, self.neuron_count * self.slot_count)?;
        let all_synapses = self.read_buffer::<SynapseStateGpu>(
            &self._plastic_synapse_buffer,
            self.plastic_edge_count * self.slot_count,
        )?;
        let all_transactions = self.read_buffer::<TransactionStateGpu>(
            &self._transaction_buffer,
            self.plastic_edge_count * self.slot_count,
        )?;

        let ns = slot * self.neuron_count;
        let ne = ns + self.neuron_count;
        let ps = slot * self.plastic_edge_count;
        let pe = ps + self.plastic_edge_count;
        let neurons = &all_neurons[ns..ne];
        let synapses = &all_synapses[ps..pe];
        let transactions = &all_transactions[ps..pe];

        Ok(PopulationDebugState {
            membrane: neurons.iter().map(|v| v.membrane).collect(),
            trace: neurons.iter().map(|v| v.trace).collect(),
            modulation: neurons.iter().map(|v| v.modulation).collect(),
            refractory: neurons.iter().map(|v| v.refractory).collect(),
            spikes: all_spikes[ns..ne].to_vec(),
            plastic_weight: synapses.iter().map(|v| v.weight).collect(),
            eligibility: synapses.iter().map(|v| v.eligibility).collect(),
            transaction_shift: transactions.iter().map(|v| v.shift).collect(),
            transaction_lo: transactions.iter().map(|v| v.lo).collect(),
            transaction_hi: transactions.iter().map(|v| v.hi).collect(),
        })
    }
}
