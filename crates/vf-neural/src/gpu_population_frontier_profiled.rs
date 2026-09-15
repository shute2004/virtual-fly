include!("gpu_population_frontier.rs");

impl GpuPopulationRuntime {
    /// Diagnostic-only signed activity readback for one population slot.
    ///
    /// The production hot path never calls this.  Values are returned as dense
    /// neuron indices whose current internal activity code is either 1 or 2;
    /// both signs propagate through the outgoing frontier.
    pub fn active_neuron_indices(&self, slot: usize) -> Result<Vec<u32>> {
        self.validate_slot(slot)?;
        let current_spikes = if self.current_is_b {
            &self.spikes_b
        } else {
            &self.spikes_a
        };
        let all = self.read_buffer::<u32>(
            current_spikes,
            self.neuron_count * self.slot_count,
        )?;
        let start = slot * self.neuron_count;
        let end = start + self.neuron_count;
        Ok(all[start..end]
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
}
