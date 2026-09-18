use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub format_version: u32,
    pub dataset: String,
    pub neuron_count: usize,
    pub edge_count: usize,
    pub body_ids_file: String,
    pub row_offsets_file: String,
    pub pre_indices_file: String,
    pub synapse_counts_file: String,
    pub neurotransmitters_file: String,
    #[serde(default)]
    pub modulator_roles_file: Option<String>,
    #[serde(default)]
    pub modulator_role_definition: Option<String>,
    #[serde(default)]
    pub runtime_semantics: Option<String>,
    #[serde(default)]
    pub source_sha256: serde_json::Value,
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeInput {
    pub pre: u32,
    pub post: u32,
    pub synapse_count: u32,
}

#[derive(Debug, Clone)]
pub struct ConnectomeSnapshot {
    pub manifest: SnapshotManifest,
    pub body_ids: Vec<u64>,
    /// Incoming CSR. `row_offsets[post]..row_offsets[post + 1]` addresses all
    /// released connections arriving at `post`.
    pub row_offsets: Vec<u32>,
    pub pre_indices: Vec<u32>,
    pub edge_posts: Vec<u32>,
    pub synapse_counts: Vec<u32>,
    /// Per-presynaptic-neuron consensus transmitter code.
    pub neurotransmitters: Vec<u8>,
    /// Per-neuron neuromodulator role. 0 means no modeled neuromodulatory
    /// release role; 1 means an annotation-supported dopaminergic neuron.
    pub modulator_roles: Vec<u8>,
}

impl ConnectomeSnapshot {
    pub fn load_dir(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let manifest_path = path.join("manifest.json");
        let manifest: SnapshotManifest = serde_json::from_slice(
            &fs::read(&manifest_path)
                .with_context(|| format!("failed to read {}", manifest_path.display()))?,
        )
        .context("invalid snapshot manifest")?;

        if manifest.format_version != 1 {
            bail!(
                "unsupported snapshot format {}; expected 1",
                manifest.format_version
            );
        }

        let is_malecns = manifest.dataset.starts_with("male-cns:");
        let allow_legacy_malecns = std::env::var("VF_ALLOW_LEGACY_MALECNS_SNAPSHOT")
            .map(|value| value == "1")
            .unwrap_or(false);
        if is_malecns && !allow_legacy_malecns {
            const EXPECTED_SEMANTICS: &str = "male-cns-v1-class-dan-v1";
            const EXPECTED_ROLE: &str = "consensus_nt=dopamine AND released annotation class=DAN";
            if manifest.runtime_semantics.as_deref() != Some(EXPECTED_SEMANTICS) {
                bail!(
                    "MaleCNS snapshot runtime semantics are missing or incompatible; expected {EXPECTED_SEMANTICS}. Rebuild the snapshot with current prepare_malecns.py. Set VF_ALLOW_LEGACY_MALECNS_SNAPSHOT=1 only for explicit historical replay."
                );
            }
            if manifest.modulator_roles_file.is_none() {
                bail!("MaleCNS production snapshot is missing modulator_roles_file");
            }
            if manifest.modulator_role_definition.as_deref() != Some(EXPECTED_ROLE) {
                bail!("MaleCNS production snapshot has incompatible modulator role definition");
            }
        }

        let body_ids = read_u64_le(path.join(&manifest.body_ids_file))?;
        let row_offsets = read_u32_le(path.join(&manifest.row_offsets_file))?;
        let pre_indices = read_u32_le(path.join(&manifest.pre_indices_file))?;
        let synapse_counts = read_u32_le(path.join(&manifest.synapse_counts_file))?;
        let neurotransmitters = fs::read(path.join(&manifest.neurotransmitters_file))?;
        let modulator_roles = if let Some(file) = &manifest.modulator_roles_file {
            fs::read(path.join(file))?
        } else {
            // Backward compatibility for synthetic/legacy snapshots created before
            // neuromodulator identity was separated from transmitter prediction.
            neurotransmitters
                .iter()
                .map(|&value| u8::from(value == crate::model::nt::DOPAMINE))
                .collect()
        };

        if body_ids.len() != manifest.neuron_count {
            bail!("body_ids length does not match manifest");
        }
        if !body_ids.windows(2).all(|pair| pair[0] < pair[1]) {
            bail!("body_ids must be strictly increasing");
        }
        if neurotransmitters.len() != manifest.neuron_count {
            bail!("neurotransmitter length does not match manifest");
        }
        if modulator_roles.len() != manifest.neuron_count {
            bail!("modulator role length does not match manifest");
        }
        if row_offsets.len() != manifest.neuron_count + 1 {
            bail!("row_offsets length does not match neuron count");
        }
        if pre_indices.len() != manifest.edge_count || synapse_counts.len() != manifest.edge_count {
            bail!("edge arrays do not match manifest edge count");
        }
        if row_offsets.last().copied().unwrap_or_default() as usize != manifest.edge_count {
            bail!("final CSR row offset does not match edge count");
        }

        let edge_posts = build_edge_posts(&row_offsets, manifest.edge_count)?;

        Ok(Self {
            manifest,
            body_ids,
            row_offsets,
            pre_indices,
            edge_posts,
            synapse_counts,
            neurotransmitters,
            modulator_roles,
        })
    }

    pub fn from_edges(
        neuron_count: usize,
        edges: &[EdgeInput],
        neurotransmitters: Vec<u8>,
    ) -> Result<Self> {
        if neurotransmitters.len() != neuron_count {
            bail!("neurotransmitter vector must have one entry per neuron");
        }
        let modulator_roles = neurotransmitters
            .iter()
            .map(|&value| u8::from(value == crate::model::nt::DOPAMINE))
            .collect::<Vec<_>>();
        if neuron_count > u32::MAX as usize || edges.len() > u32::MAX as usize {
            bail!("bootstrap snapshot exceeds current u32 CSR limits");
        }

        let mut sorted = edges.to_vec();
        sorted.sort_unstable_by_key(|edge| (edge.post, edge.pre));

        let mut row_offsets = vec![0u32; neuron_count + 1];
        for edge in &sorted {
            if edge.pre as usize >= neuron_count || edge.post as usize >= neuron_count {
                bail!("edge endpoint outside neuron range");
            }
            row_offsets[edge.post as usize + 1] += 1;
        }
        for i in 1..row_offsets.len() {
            row_offsets[i] += row_offsets[i - 1];
        }

        let pre_indices = sorted.iter().map(|edge| edge.pre).collect::<Vec<_>>();
        let edge_posts = sorted.iter().map(|edge| edge.post).collect::<Vec<_>>();
        let synapse_counts = sorted
            .iter()
            .map(|edge| edge.synapse_count)
            .collect::<Vec<_>>();

        Ok(Self {
            manifest: SnapshotManifest {
                format_version: 1,
                dataset: "synthetic:bootstrap".to_owned(),
                neuron_count,
                edge_count: sorted.len(),
                body_ids_file: String::new(),
                row_offsets_file: String::new(),
                pre_indices_file: String::new(),
                synapse_counts_file: String::new(),
                neurotransmitters_file: String::new(),
                modulator_roles_file: None,
                modulator_role_definition: None,
                runtime_semantics: None,
                source_sha256: serde_json::Value::Null,
            },
            body_ids: (0..neuron_count as u64).collect(),
            row_offsets,
            pre_indices,
            edge_posts,
            synapse_counts,
            neurotransmitters,
            modulator_roles,
        })
    }

    #[inline]
    pub fn is_dopamine_modulator(&self, neuron: usize) -> bool {
        self.modulator_roles
            .get(neuron)
            .copied()
            .unwrap_or_default()
            == 1
    }

    #[inline]
    pub fn packed_neuron_metadata(&self, neuron: usize) -> u32 {
        let nt = self.neurotransmitters[neuron] as u32;
        let modulator = u32::from(self.is_dopamine_modulator(neuron));
        nt | (modulator << 8)
    }

    pub fn neuron_count(&self) -> usize {
        self.manifest.neuron_count
    }

    pub fn edge_count(&self) -> usize {
        self.manifest.edge_count
    }

    /// Resolve a released MaleCNS body ID to the dense runtime neuron index.
    ///
    /// Runtime arrays use compact indices for efficient CPU/GPU access, while
    /// experiment configuration should refer to stable source body IDs.
    pub fn index_of_body_id(&self, body_id: u64) -> Option<usize> {
        self.body_ids.binary_search(&body_id).ok()
    }

    pub fn body_id(&self, index: usize) -> Option<u64> {
        self.body_ids.get(index).copied()
    }
}

fn build_edge_posts(row_offsets: &[u32], edge_count: usize) -> Result<Vec<u32>> {
    let mut posts = vec![0u32; edge_count];
    for post in 0..row_offsets.len().saturating_sub(1) {
        let start = row_offsets[post] as usize;
        let end = row_offsets[post + 1] as usize;
        if start > end || end > edge_count {
            bail!("invalid CSR offsets at row {post}");
        }
        posts[start..end].fill(post as u32);
    }
    Ok(posts)
}

fn read_u32_le(path: impl AsRef<Path>) -> Result<Vec<u32>> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} has invalid u32 byte length", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk length")))
        .collect())
}

fn read_u64_le(path: impl AsRef<Path>) -> Result<Vec<u64>> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() % 8 != 0 {
        bail!("{} has invalid u64 byte length", path.display());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("chunk length")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_incoming_csr() {
        let snapshot = ConnectomeSnapshot::from_edges(
            3,
            &[
                EdgeInput {
                    pre: 0,
                    post: 2,
                    synapse_count: 2,
                },
                EdgeInput {
                    pre: 1,
                    post: 2,
                    synapse_count: 3,
                },
                EdgeInput {
                    pre: 0,
                    post: 1,
                    synapse_count: 1,
                },
            ],
            vec![1, 1, 1],
        )
        .unwrap();

        assert_eq!(snapshot.row_offsets, vec![0, 0, 1, 3]);
        assert_eq!(snapshot.pre_indices, vec![0, 0, 1]);
        assert_eq!(snapshot.edge_posts, vec![1, 2, 2]);
        assert_eq!(snapshot.index_of_body_id(1), Some(1));
        assert_eq!(snapshot.body_id(2), Some(2));
        assert_eq!(snapshot.index_of_body_id(99), None);
    }
}
