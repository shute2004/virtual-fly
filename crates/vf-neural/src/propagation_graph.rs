use anyhow::{Context, Result};

use crate::{
    model::{assumed_fast_sign, nt},
    snapshot::ConnectomeSnapshot,
};

/// Shared outgoing adjacency used only to discover which postsynaptic neurons
/// can receive a causal signal from the currently active presynaptic frontier.
///
/// The dense/reference current accumulation remains incoming-CSR ordered. This
/// graph therefore does not replace the canonical topology; it is an auxiliary
/// index that lets the runtime skip incoming scans for posts that cannot have
/// received a synaptic event in the current step.
///
/// Only edges that can affect the current bootstrap dynamics are included:
/// released dopaminergic edges and edges whose presynaptic transmitter has a
/// non-zero fast sign. Other transmitter classes are causally silent in the
/// current model and do not need to create candidate posts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutgoingPropagationGraph {
    /// `row_offsets[pre]..row_offsets[pre + 1]` addresses causal outgoing posts.
    pub row_offsets: Vec<u32>,
    /// Postsynaptic neuron indices for the causal outgoing rows.
    pub post_indices: Vec<u32>,
}

impl OutgoingPropagationGraph {
    pub fn compile(snapshot: &ConnectomeSnapshot) -> Result<Self> {
        let n = snapshot.neuron_count();

        let mut row_offsets = vec![0u32; n + 1];
        for &pre_u32 in &snapshot.pre_indices {
            let pre = pre_u32 as usize;
            if !is_causal_presynaptic(snapshot, pre) {
                continue;
            }
            let next = pre + 1;
            row_offsets[next] = row_offsets[next]
                .checked_add(1)
                .context("outgoing degree exceeds u32")?;
        }
        for index in 1..row_offsets.len() {
            row_offsets[index] = row_offsets[index]
                .checked_add(row_offsets[index - 1])
                .context("outgoing CSR offset exceeds u32")?;
        }

        let causal_edge_count = row_offsets[n] as usize;
        let mut cursor = row_offsets[..n].to_vec();
        let mut post_indices = vec![0u32; causal_edge_count];
        for edge in 0..snapshot.edge_count() {
            let pre = snapshot.pre_indices[edge] as usize;
            if !is_causal_presynaptic(snapshot, pre) {
                continue;
            }
            let position = cursor[pre] as usize;
            post_indices[position] = snapshot.edge_posts[edge];
            cursor[pre] = cursor[pre]
                .checked_add(1)
                .context("outgoing CSR cursor exceeds u32")?;
        }

        debug_assert_eq!(row_offsets.len(), n + 1);
        debug_assert_eq!(row_offsets[n] as usize, causal_edge_count);
        debug_assert_eq!(post_indices.len(), causal_edge_count);
        Ok(Self {
            row_offsets,
            post_indices,
        })
    }

    pub fn edge_count(&self) -> usize {
        self.post_indices.len()
    }
}

fn is_causal_presynaptic(snapshot: &ConnectomeSnapshot, pre: usize) -> bool {
    let code = snapshot.neurotransmitters[pre];
    code == nt::DOPAMINE || assumed_fast_sign(code) != 0.0
}

#[cfg(test)]
mod tests {
    use crate::{model::nt, snapshot::EdgeInput};

    use super::*;

    #[test]
    fn compiles_outgoing_csr_from_incoming_snapshot() {
        let snapshot = ConnectomeSnapshot::from_edges(
            4,
            &[
                EdgeInput { pre: 0, post: 2, synapse_count: 1 },
                EdgeInput { pre: 0, post: 1, synapse_count: 1 },
                EdgeInput { pre: 2, post: 3, synapse_count: 1 },
                EdgeInput { pre: 1, post: 3, synapse_count: 1 },
            ],
            vec![nt::ACETYLCHOLINE; 4],
        )
        .unwrap();

        let graph = OutgoingPropagationGraph::compile(&snapshot).unwrap();
        assert_eq!(graph.row_offsets, vec![0, 2, 3, 4, 4]);
        assert_eq!(&graph.post_indices[0..2], &[1, 2]);
        assert_eq!(&graph.post_indices[2..3], &[3]);
        assert_eq!(&graph.post_indices[3..4], &[3]);
    }

    #[test]
    fn excludes_transmitter_classes_that_are_silent_in_current_model() {
        let snapshot = ConnectomeSnapshot::from_edges(
            4,
            &[
                EdgeInput { pre: 0, post: 3, synapse_count: 1 },
                EdgeInput { pre: 1, post: 3, synapse_count: 1 },
                EdgeInput { pre: 2, post: 3, synapse_count: 1 },
            ],
            vec![nt::ACETYLCHOLINE, nt::DOPAMINE, 5, nt::ACETYLCHOLINE],
        )
        .unwrap();
        let graph = OutgoingPropagationGraph::compile(&snapshot).unwrap();
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.row_offsets, vec![0, 1, 2, 2, 2]);
        assert_eq!(graph.post_indices, vec![3, 3]);
    }
}
