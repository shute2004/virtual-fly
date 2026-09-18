use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use vf_neural::NeuralState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub schema_version: u32,
    pub dataset: String,
    pub neuron_count: usize,
    pub edge_count: usize,
    pub step: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_weight_version: Option<u64>,
    pub membrane_file: String,
    pub spikes_file: String,
    pub refractory_file: String,
    pub activity_trace_file: String,
    pub modulation_file: String,
    pub weights_file: String,
    pub eligibility_file: String,
}

// v2 changed modulation to released-dopamine-derived state only. v3 changes
// the meaning of `spikes`: 0=silent, 1=positive/depolarizing activity event,
// 2=negative/hyperpolarizing activity deviation. The latter is required for
// graded inhibitory circuits such as R1-R6 -> lamina and therefore old dynamic
// state must not be resumed under the new runtime semantics.
const SCHEMA_VERSION: u32 = 3;

pub fn save_checkpoint(
    directory: impl AsRef<Path>,
    dataset: &str,
    step: u64,
    state: &NeuralState,
) -> Result<CheckpointManifest> {
    save_checkpoint_impl(directory, dataset, step, state, None)
}

pub fn save_checkpoint_with_global_weight_version(
    directory: impl AsRef<Path>,
    dataset: &str,
    step: u64,
    state: &NeuralState,
    global_weight_version: u64,
) -> Result<CheckpointManifest> {
    save_checkpoint_impl(directory, dataset, step, state, Some(global_weight_version))
}

fn save_checkpoint_impl(
    directory: impl AsRef<Path>,
    dataset: &str,
    step: u64,
    state: &NeuralState,
    global_weight_version: Option<u64>,
) -> Result<CheckpointManifest> {
    let directory = directory.as_ref();
    state.validate(state.membrane.len(), state.weights.len())?;

    let temp = temp_directory(directory);
    if temp.exists() {
        fs::remove_dir_all(&temp).with_context(|| format!("failed to clear {}", temp.display()))?;
    }
    fs::create_dir_all(&temp).with_context(|| format!("failed to create {}", temp.display()))?;

    let manifest = CheckpointManifest {
        schema_version: SCHEMA_VERSION,
        dataset: dataset.to_owned(),
        neuron_count: state.membrane.len(),
        edge_count: state.weights.len(),
        step,
        global_weight_version,
        membrane_file: "membrane.f32le".to_owned(),
        spikes_file: "spikes.u32le".to_owned(),
        refractory_file: "refractory.u32le".to_owned(),
        activity_trace_file: "activity-trace.f32le".to_owned(),
        modulation_file: "modulation.f32le".to_owned(),
        weights_file: "weights.f32le".to_owned(),
        eligibility_file: "eligibility.f32le".to_owned(),
    };

    write_f32(temp.join(&manifest.membrane_file), &state.membrane)?;
    write_u32(temp.join(&manifest.spikes_file), &state.spikes)?;
    write_u32(temp.join(&manifest.refractory_file), &state.refractory)?;
    write_f32(
        temp.join(&manifest.activity_trace_file),
        &state.activity_trace,
    )?;
    write_f32(temp.join(&manifest.modulation_file), &state.modulation)?;
    write_f32(temp.join(&manifest.weights_file), &state.weights)?;
    write_f32(temp.join(&manifest.eligibility_file), &state.eligibility)?;
    fs::write(
        temp.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    if directory.exists() {
        fs::remove_dir_all(directory)
            .with_context(|| format!("failed to replace {}", directory.display()))?;
    }
    fs::rename(&temp, directory).with_context(|| {
        format!(
            "failed to move completed checkpoint {} -> {}",
            temp.display(),
            directory.display()
        )
    })?;
    Ok(manifest)
}

pub fn load_checkpoint(
    directory: impl AsRef<Path>,
    expected_dataset: &str,
    expected_neurons: usize,
    expected_edges: usize,
) -> Result<(CheckpointManifest, NeuralState)> {
    let directory = directory.as_ref();
    let manifest_path = directory.join("manifest.json");
    let manifest: CheckpointManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )
    .context("invalid neural checkpoint manifest")?;

    if manifest.schema_version != SCHEMA_VERSION {
        bail!(
            "unsupported checkpoint schema {}; expected {} (v1 used externally signed modulator roles; v2 used one-sided 0/1 activity events; both are intentionally incompatible)",
            manifest.schema_version,
            SCHEMA_VERSION
        );
    }
    if manifest.dataset != expected_dataset {
        bail!(
            "checkpoint dataset {} does not match snapshot {}",
            manifest.dataset,
            expected_dataset
        );
    }
    if manifest.neuron_count != expected_neurons || manifest.edge_count != expected_edges {
        bail!(
            "checkpoint shape neurons={} edges={} does not match snapshot neurons={} edges={}",
            manifest.neuron_count,
            manifest.edge_count,
            expected_neurons,
            expected_edges
        );
    }

    let state = NeuralState {
        membrane: read_f32(directory.join(&manifest.membrane_file))?,
        spikes: read_u32(directory.join(&manifest.spikes_file))?,
        refractory: read_u32(directory.join(&manifest.refractory_file))?,
        activity_trace: read_f32(directory.join(&manifest.activity_trace_file))?,
        modulation: read_f32(directory.join(&manifest.modulation_file))?,
        weights: read_f32(directory.join(&manifest.weights_file))?,
        eligibility: read_f32(directory.join(&manifest.eligibility_file))?,
    };
    state.validate(expected_neurons, expected_edges)?;
    Ok((manifest, state))
}

fn temp_directory(directory: &Path) -> PathBuf {
    let file_name = directory
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("checkpoint");
    directory.with_file_name(format!(".{file_name}.tmp"))
}

fn write_f32(path: PathBuf, values: &[f32]) -> Result<()> {
    write_le_words(path, values, |value| value.to_le_bytes())
}

fn write_u32(path: PathBuf, values: &[u32]) -> Result<()> {
    write_le_words(path, values, |value| value.to_le_bytes())
}

fn write_le_words<T: bytemuck::Pod + Copy>(
    path: PathBuf,
    values: &[T],
    to_le_bytes: impl Fn(T) -> [u8; 4],
) -> Result<()> {
    let file =
        fs::File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);

    #[cfg(target_endian = "little")]
    {
        writer.write_all(bytemuck::cast_slice(values))?;
    }

    #[cfg(not(target_endian = "little"))]
    {
        const CHUNK_WORDS: usize = 1 << 20;
        let mut buffer = Vec::with_capacity(CHUNK_WORDS * 4);
        for chunk in values.chunks(CHUNK_WORDS) {
            buffer.clear();
            for &value in chunk {
                buffer.extend_from_slice(&to_le_bytes(value));
            }
            writer.write_all(&buffer)?;
        }
    }

    // Keep the argument used on little-endian builds as part of the portable
    // signature without paying per-element conversion cost on the hot path.
    let _ = &to_le_bytes;
    writer.flush()?;
    Ok(())
}

fn read_f32(path: PathBuf) -> Result<Vec<f32>> {
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} does not contain whole f32 values", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size")))
        .collect())
}

fn read_u32(path: PathBuf) -> Result<Vec<u32>> {
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} does not contain whole u32 values", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk size")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_state_round_trip() {
        let root = std::env::temp_dir().join(format!(
            "virtual-fly-checkpoint-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let state = NeuralState {
            membrane: vec![0.1, -0.2, 0.3],
            spikes: vec![1, 2, 0],
            refractory: vec![2, 0, 1],
            activity_trace: vec![0.4, -0.5, 0.6],
            modulation: vec![0.0, 0.7, 0.8],
            weights: vec![1.1, 2.2],
            eligibility: vec![0.9, -0.4],
        };

        let saved = save_checkpoint(&root, "synthetic:test", 42, &state).unwrap();
        assert_eq!(saved.step, 42);
        assert_eq!(saved.schema_version, 3);
        assert_eq!(saved.global_weight_version, None);
        let (loaded_manifest, loaded) = load_checkpoint(&root, "synthetic:test", 3, 2).unwrap();
        assert_eq!(loaded_manifest.step, 42);
        assert_eq!(loaded_manifest.global_weight_version, None);
        assert_eq!(loaded.membrane, state.membrane);
        assert_eq!(loaded.spikes, state.spikes);
        assert_eq!(loaded.refractory, state.refractory);
        assert_eq!(loaded.activity_trace, state.activity_trace);
        assert_eq!(loaded.modulation, state.modulation);
        assert_eq!(loaded.weights, state.weights);
        assert_eq!(loaded.eligibility, state.eligibility);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn population_checkpoint_round_trips_global_weight_version() {
        let root = std::env::temp_dir().join(format!(
            "virtual-fly-population-checkpoint-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let state = NeuralState {
            membrane: vec![0.0],
            spikes: vec![0],
            refractory: vec![0],
            activity_trace: vec![0.0],
            modulation: vec![0.0],
            weights: vec![0.25],
            eligibility: vec![0.0],
        };

        let saved =
            save_checkpoint_with_global_weight_version(&root, "synthetic:test", 7, &state, 184)
                .unwrap();
        assert_eq!(saved.global_weight_version, Some(184));
        let (loaded_manifest, loaded) = load_checkpoint(&root, "synthetic:test", 1, 1).unwrap();
        assert_eq!(loaded_manifest.global_weight_version, Some(184));
        assert_eq!(loaded.weights, state.weights);
        fs::remove_dir_all(&root).unwrap();
    }
}
