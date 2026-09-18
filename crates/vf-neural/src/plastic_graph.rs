use anyhow::{Context, Result};

use crate::{model::assumed_fast_sign, snapshot::ConnectomeSnapshot};

/// Structurally mutable subset of the released connectome under the current
/// bootstrap three-factor plasticity rule.
///
/// An edge belongs to this graph iff:
///
/// 1. its presynaptic transmitter has a non-zero fast sign, and
/// 2. its postsynaptic neuron has at least one released dopaminergic input.
///
/// The graph preserves the source incoming-CSR order within each post. That
/// property is important because later sparse propagation can discover a small
/// set of candidate posts from outgoing adjacency and still accumulate their
/// actual currents in the same incoming order as the dense reference runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlasticFastGraph {
    /// Incoming CSR offsets over plastic edges, one row per postsynaptic neuron.
    pub row_offsets: Vec<u32>,
    /// Source/full-connectome edge index for every compact plastic edge.
    pub edge_indices: Vec<u32>,
    /// Postsynaptic neuron for every compact plastic edge.
    pub post_indices: Vec<u32>,
    /// All posts that can receive released dopamine, including rare rows that
    /// might contain no mutable fast incoming edge.
    pub dopamine_capable_posts: Vec<u32>,
}

impl PlasticFastGraph {
    pub fn compile(snapshot: &ConnectomeSnapshot) -> Result<Self> {
        let n = snapshot.neuron_count();
        let m = snapshot.edge_count();

        let mut dopamine_capable_post = vec![false; n];
        let mut dopamine_capable_posts = Vec::new();
        for post in 0..n {
            let start = snapshot.row_offsets[post] as usize;
            let end = snapshot.row_offsets[post + 1] as usize;
            for edge in start..end {
                let pre = snapshot.pre_indices[edge] as usize;
                if snapshot.is_dopamine_modulator(pre) {
                    dopamine_capable_post[post] = true;
                    dopamine_capable_posts.push(
                        u32::try_from(post)
                            .with_context(|| format!("post neuron index {post} exceeds u32"))?,
                    );
                    break;
                }
            }
        }

        let mut row_offsets = Vec::with_capacity(n + 1);
        let mut edge_indices = Vec::new();
        let mut post_indices = Vec::new();
        row_offsets.push(0);

        for post in 0..n {
            if dopamine_capable_post[post] {
                let start = snapshot.row_offsets[post] as usize;
                let end = snapshot.row_offsets[post + 1] as usize;
                for edge in start..end {
                    let pre = snapshot.pre_indices[edge] as usize;
                    if assumed_fast_sign(snapshot.neurotransmitters[pre]) == 0.0 {
                        continue;
                    }
                    edge_indices.push(
                        u32::try_from(edge)
                            .with_context(|| format!("source edge index {edge} exceeds u32"))?,
                    );
                    post_indices.push(
                        u32::try_from(post)
                            .with_context(|| format!("post neuron index {post} exceeds u32"))?,
                    );
                }
            }
            row_offsets.push(
                u32::try_from(edge_indices.len())
                    .context("PlasticFastGraph edge count exceeds u32")?,
            );
        }

        debug_assert!(edge_indices.len() <= m);
        debug_assert_eq!(edge_indices.len(), post_indices.len());
        debug_assert_eq!(row_offsets.len(), n + 1);

        Ok(Self {
            row_offsets,
            edge_indices,
            post_indices,
            dopamine_capable_posts,
        })
    }

    pub fn edge_count(&self) -> usize {
        self.edge_indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edge_indices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::{model::nt, snapshot::EdgeInput};

    use super::*;

    #[test]
    fn compiles_only_fast_edges_into_dopamine_capable_posts() {
        let snapshot = ConnectomeSnapshot::from_edges(
            5,
            &[
                EdgeInput {
                    pre: 0,
                    post: 1,
                    synapse_count: 3,
                },
                EdgeInput {
                    pre: 2,
                    post: 1,
                    synapse_count: 2,
                },
                EdgeInput {
                    pre: 4,
                    post: 1,
                    synapse_count: 4,
                },
                EdgeInput {
                    pre: 0,
                    post: 3,
                    synapse_count: 5,
                },
            ],
            vec![
                nt::ACETYLCHOLINE,
                nt::ACETYLCHOLINE,
                nt::DOPAMINE,
                nt::ACETYLCHOLINE,
                nt::GABA,
            ],
        )
        .unwrap();

        let graph = PlasticFastGraph::compile(&snapshot).unwrap();

        // Source incoming order for post=1 is pre 0, 2, 4. The dopamine edge
        // makes the post structurally plastic but is not itself a mutable fast
        // edge, so source edges 0 and 2 are retained. post=3 has a fast edge but
        // no dopamine input and is excluded.
        assert_eq!(graph.edge_indices, vec![0, 2]);
        assert_eq!(graph.post_indices, vec![1, 1]);
        assert_eq!(graph.row_offsets, vec![0, 0, 2, 2, 2, 2]);
        assert_eq!(graph.dopamine_capable_posts, vec![1]);
    }

    #[test]
    fn preserves_original_incoming_edge_order() {
        let snapshot = ConnectomeSnapshot::from_edges(
            4,
            &[
                EdgeInput {
                    pre: 2,
                    post: 3,
                    synapse_count: 1,
                },
                EdgeInput {
                    pre: 0,
                    post: 3,
                    synapse_count: 1,
                },
                EdgeInput {
                    pre: 1,
                    post: 3,
                    synapse_count: 1,
                },
            ],
            vec![nt::GABA, nt::DOPAMINE, nt::ACETYLCHOLINE, nt::ACETYLCHOLINE],
        )
        .unwrap();

        let graph = PlasticFastGraph::compile(&snapshot).unwrap();
        // ConnectomeSnapshot sorts incoming edges by (post, pre): pre0, pre1,
        // pre2. pre1 is dopamine/non-fast, leaving source edge order 0 then 2.
        assert_eq!(graph.edge_indices, vec![0, 2]);
        assert_eq!(graph.post_indices, vec![3, 3]);
        assert_eq!(graph.row_offsets[3], 0);
        assert_eq!(graph.row_offsets[4], 2);
        assert_eq!(graph.dopamine_capable_posts, vec![3]);
    }

    #[test]
    fn retains_dopamine_capable_post_even_without_fast_inputs() {
        let snapshot = ConnectomeSnapshot::from_edges(
            2,
            &[EdgeInput {
                pre: 0,
                post: 1,
                synapse_count: 1,
            }],
            vec![nt::DOPAMINE, nt::ACETYLCHOLINE],
        )
        .unwrap();

        let graph = PlasticFastGraph::compile(&snapshot).unwrap();
        assert!(graph.edge_indices.is_empty());
        assert_eq!(graph.dopamine_capable_posts, vec![1]);
    }
}
