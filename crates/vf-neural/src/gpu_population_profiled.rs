// Temporary profiling extension around the sparse population runtime.
//
// The production dynamics remain in gpu_population_sparse.rs.  This wrapper
// adds a read-only transaction-state diagnostic so we can measure the real
// episode dirty fraction before choosing a compact transaction representation.
include!("gpu_population_sparse.rs");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransactionStats {
    pub plastic_edges: usize,
    pub dirty_edges: usize,
    pub nonzero_shift_edges: usize,
    pub lower_bound_changed_edges: usize,
    pub upper_bound_changed_edges: usize,
}

impl GpuPopulationRuntime {
    /// Read back one slot's current dense-P transaction state for profiling.
    ///
    /// This is intentionally not used in the training hot path.  The readback
    /// is large (~12P bytes per slot) and exists only to measure D/P before the
    /// dense transaction buffer is removed.
    pub fn transaction_stats(&self, slot: usize, weight_max: f32) -> Result<TransactionStats> {
        self.validate_slot(slot)?;
        let all = self.read_buffer::<TransactionStateGpu>(
            &self._transaction_buffer,
            self.plastic_edge_count * self.slot_count,
        )?;
        let start = slot * self.plastic_edge_count;
        let end = start + self.plastic_edge_count;
        let mut dirty_edges = 0usize;
        let mut nonzero_shift_edges = 0usize;
        let mut lower_bound_changed_edges = 0usize;
        let mut upper_bound_changed_edges = 0usize;
        for txn in &all[start..end] {
            let shift_changed = txn.shift != 0.0;
            let lower_changed = txn.lo != 0.0;
            let upper_changed = txn.hi != weight_max;
            if shift_changed || lower_changed || upper_changed {
                dirty_edges += 1;
            }
            nonzero_shift_edges += usize::from(shift_changed);
            lower_bound_changed_edges += usize::from(lower_changed);
            upper_bound_changed_edges += usize::from(upper_changed);
        }
        Ok(TransactionStats {
            plastic_edges: self.plastic_edge_count,
            dirty_edges,
            nonzero_shift_edges,
            lower_bound_changed_edges,
            upper_bound_changed_edges,
        })
    }
}
