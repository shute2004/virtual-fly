use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use vf_neural::{ConnectomeSnapshot, NeuralParams, Stimulus, gpu_population::GpuPopulationRuntime};
use vf_runner::checkpoint::{load_checkpoint, save_checkpoint};

#[derive(Debug, Parser)]
#[command(
    name = "population-neural-bridge",
    about = "Shared-weight asynchronous population CNS bridge for Flyppy"
)]
struct Args {
    #[arg(long, default_value = "artifacts/malecns-v1.0")]
    snapshot: PathBuf,
    #[arg(long)]
    groups: PathBuf,
    #[arg(long, default_value_t = 2)]
    slots: usize,
}

#[derive(Debug, Deserialize)]
struct GroupConfigFile {
    schema_version: u32,
    groups: HashMap<String, GroupConfig>,
}

#[derive(Debug, Deserialize)]
struct GroupConfig {
    body_ids: Vec<u64>,
}

#[derive(Debug)]
struct Group {
    indices: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct SlotStepRequest {
    slot: usize,
    #[serde(default)]
    stimulate: HashMap<String, f32>,
    #[serde(default)]
    stimulate_body: Vec<(u64, f32)>,
    #[serde(default)]
    read_body: Vec<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Request {
    Ping,
    LoadCheckpoint {
        path: PathBuf,
        #[serde(default)]
        global_weight_version: u64,
    },
    SaveCheckpoint { path: PathBuf },
    StepBatch {
        slots: Vec<SlotStepRequest>,
        #[serde(default = "default_true")]
        plasticity: bool,
    },
    StepSlot {
        slot: usize,
        #[serde(default)]
        stimulate: HashMap<String, f32>,
        #[serde(default)]
        stimulate_body: Vec<(u64, f32)>,
        #[serde(default = "default_true")]
        plasticity: bool,
        #[serde(default = "default_one")]
        steps: usize,
    },
    TransactionStats { slot: usize },
    CommitSlot {
        slot: usize,
        source_weight_version: u64,
    },
    RestartSlot { slot: usize },
    Quit,
}

fn default_true() -> bool { true }
fn default_one() -> usize { 1 }

#[derive(Debug, Serialize)]
struct ReadyResponse<'a> {
    ok: bool,
    event: &'a str,
    backend: String,
    slots: usize,
    neurons: usize,
    edges: usize,
    plastic_edges: usize,
    groups: Vec<&'a str>,
    global_weight_version: u64,
}

#[derive(Debug, Serialize)]
struct BodyReadout {
    body_id: u64,
    spike: bool,
}

#[derive(Debug, Serialize)]
struct SlotReadout {
    slot: usize,
    read_body: Vec<BodyReadout>,
}

#[derive(Debug, Serialize)]
struct BatchStepResponse {
    ok: bool,
    step: u64,
    slots: Vec<SlotReadout>,
}

#[derive(Debug, Serialize)]
struct StepResponse {
    ok: bool,
    step: u64,
}

#[derive(Debug, Serialize)]
struct TransactionStatsResponse {
    ok: bool,
    event: &'static str,
    slot: usize,
    plastic_edges: usize,
    dirty_edges: usize,
    dirty_fraction: f64,
    nonzero_shift_edges: usize,
    lower_bound_changed_edges: usize,
    upper_bound_changed_edges: usize,
}

#[derive(Debug, Serialize)]
struct CommitResponse {
    ok: bool,
    event: &'static str,
    slot: usize,
    source_weight_version: u64,
    commit_from_version: u64,
    commit_weight_version: u64,
    staleness: u64,
}

#[derive(Debug, Serialize)]
struct CheckpointResponse {
    ok: bool,
    event: &'static str,
    path: String,
    step: u64,
    global_weight_version: u64,
}

#[derive(Debug, Serialize)]
struct SimpleResponse<'a> {
    ok: bool,
    event: &'a str,
    global_weight_version: u64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    ok: bool,
    error: String,
}

fn resolve_groups(snapshot: &ConnectomeSnapshot, config: GroupConfigFile) -> Result<HashMap<String, Group>> {
    if !matches!(config.schema_version, 1 | 2) {
        bail!("unsupported group config schema {}", config.schema_version);
    }
    if config.groups.is_empty() {
        bail!("group config contains no groups");
    }
    let mut groups = HashMap::with_capacity(config.groups.len());
    for (name, group) in config.groups {
        let mut indices = Vec::with_capacity(group.body_ids.len());
        for body_id in group.body_ids {
            let index = snapshot
                .index_of_body_id(body_id)
                .with_context(|| format!("group {name:?} contains unknown body ID {body_id}"))?;
            indices.push(index);
        }
        if indices.is_empty() {
            bail!("group {name:?} contains no body IDs");
        }
        groups.insert(name, Group { indices });
    }
    Ok(groups)
}

fn build_stimuli(
    snapshot: &ConnectomeSnapshot,
    groups: &HashMap<String, Group>,
    named: &HashMap<String, f32>,
    body: &[(u64, f32)],
) -> Result<Vec<Stimulus>> {
    let mut stimuli = Vec::new();
    for (name, &current) in named {
        let group = groups
            .get(name)
            .with_context(|| format!("unknown stimulation group {name:?}"))?;
        stimuli.extend(group.indices.iter().copied().map(|neuron| Stimulus { neuron, current }));
    }
    for &(body_id, current) in body {
        let neuron = snapshot
            .index_of_body_id(body_id)
            .with_context(|| format!("unknown stimulation body ID {body_id}"))?;
        stimuli.push(Stimulus { neuron, current });
    }
    Ok(stimuli)
}

fn write_json<T: Serialize>(stdout: &mut impl Write, value: &T) -> Result<()> {
    serde_json::to_writer(&mut *stdout, value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn write_error(stdout: &mut impl Write, error: impl std::fmt::Display) -> Result<()> {
    write_json(
        stdout,
        &ErrorResponse {
            ok: false,
            error: error.to_string(),
        },
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.slots == 0 {
        bail!("--slots must be >= 1");
    }
    let snapshot = ConnectomeSnapshot::load_dir(&args.snapshot)
        .with_context(|| format!("load MaleCNS snapshot from {}", args.snapshot.display()))?;
    let config: GroupConfigFile = serde_json::from_slice(
        &fs::read(&args.groups)
            .with_context(|| format!("read group config from {}", args.groups.display()))?,
    )
    .with_context(|| format!("parse group config from {}", args.groups.display()))?;
    let groups = resolve_groups(&snapshot, config)?;
    let params = NeuralParams::default();
    let weight_max = params.weight_max;
    let mut runtime = GpuPopulationRuntime::new(&snapshot, params, args.slots)?;
    let mut global_weight_version = 0u64;
    let mut neural_step = 0u64;

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut group_names: Vec<&str> = groups.keys().map(String::as_str).collect();
    group_names.sort_unstable();
    write_json(
        &mut stdout,
        &ReadyResponse {
            ok: true,
            event: "ready",
            backend: format!("gpu-population:{}", runtime.adapter_name()),
            slots: runtime.slot_count(),
            neurons: runtime.neuron_count(),
            edges: runtime.edge_count(),
            plastic_edges: runtime.plastic_edge_count(),
            groups: group_names,
            global_weight_version,
        },
    )?;

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let request = match serde_json::from_str::<Request>(&line) {
            Ok(request) => request,
            Err(error) => {
                write_error(&mut stdout, format!("invalid request: {error}"))?;
                continue;
            }
        };

        let result = match request {
            Request::Ping => write_json(
                &mut stdout,
                &SimpleResponse { ok: true, event: "pong", global_weight_version },
            ),
            Request::LoadCheckpoint { path, global_weight_version: requested_version } => {
                let result = (|| -> Result<()> {
                    let (manifest, state) = load_checkpoint(
                        &path,
                        &snapshot.manifest.dataset,
                        snapshot.neuron_count(),
                        snapshot.edge_count(),
                    )?;
                    runtime.load_global_weights(&state.weights)?;
                    neural_step = manifest.step;
                    global_weight_version = requested_version;
                    write_json(
                        &mut stdout,
                        &CheckpointResponse {
                            ok: true,
                            event: "checkpoint_loaded",
                            path: path.display().to_string(),
                            step: neural_step,
                            global_weight_version,
                        },
                    )?;
                    Ok(())
                })();
                if let Err(error) = result { write_error(&mut stdout, error)?; }
                Ok(())
            }
            Request::SaveCheckpoint { path } => {
                let result = (|| -> Result<()> {
                    let state = runtime.global_state()?;
                    let manifest = save_checkpoint(&path, &snapshot.manifest.dataset, neural_step, &state)?;
                    write_json(
                        &mut stdout,
                        &CheckpointResponse {
                            ok: true,
                            event: "checkpoint_saved",
                            path: path.display().to_string(),
                            step: manifest.step,
                            global_weight_version,
                        },
                    )?;
                    Ok(())
                })();
                if let Err(error) = result { write_error(&mut stdout, error)?; }
                Ok(())
            }
            Request::StepBatch { slots, plasticity } => {
                let result = (|| -> Result<()> {
                    let mut seen = HashSet::new();
                    let mut stimuli_by_slot = vec![Vec::<Stimulus>::new(); runtime.slot_count()];
                    let mut active = vec![false; runtime.slot_count()];
                    let mut read_flat = Vec::<usize>::new();
                    let mut read_meta = Vec::<(usize, u64)>::new();
                    for slot_request in &slots {
                        if slot_request.slot >= runtime.slot_count() {
                            bail!("population slot {} is out of range", slot_request.slot);
                        }
                        if !seen.insert(slot_request.slot) {
                            bail!("duplicate population slot {} in one batch", slot_request.slot);
                        }
                        active[slot_request.slot] = true;
                        stimuli_by_slot[slot_request.slot] = build_stimuli(
                            &snapshot,
                            &groups,
                            &slot_request.stimulate,
                            &slot_request.stimulate_body,
                        )?;
                        for &body_id in &slot_request.read_body {
                            let neuron = snapshot
                                .index_of_body_id(body_id)
                                .with_context(|| format!("unknown read body ID {body_id}"))?;
                            read_flat.push(slot_request.slot * runtime.neuron_count() + neuron);
                            read_meta.push((slot_request.slot, body_id));
                        }
                    }
                    let events = runtime.step_batch_with_read(
                        &stimuli_by_slot,
                        &active,
                        &read_flat,
                        plasticity,
                    )?;
                    neural_step = neural_step
                        .checked_add(active.iter().filter(|&&v| v).count() as u64)
                        .context("population neural step counter overflow")?;
                    let mut by_slot = slots
                        .iter()
                        .map(|request| (request.slot, Vec::<BodyReadout>::new()))
                        .collect::<HashMap<_, _>>();
                    for ((slot, body_id), event) in read_meta.into_iter().zip(events) {
                        by_slot.entry(slot).or_default().push(BodyReadout {
                            body_id,
                            spike: event == 1,
                        });
                    }
                    let slot_readouts = slots
                        .iter()
                        .map(|request| SlotReadout {
                            slot: request.slot,
                            read_body: by_slot.remove(&request.slot).unwrap_or_default(),
                        })
                        .collect();
                    write_json(
                        &mut stdout,
                        &BatchStepResponse { ok: true, step: neural_step, slots: slot_readouts },
                    )?;
                    Ok(())
                })();
                if let Err(error) = result { write_error(&mut stdout, error)?; }
                Ok(())
            }
            Request::StepSlot { slot, stimulate, stimulate_body, plasticity, steps } => {
                let result = (|| -> Result<()> {
                    if steps == 0 { bail!("steps must be >= 1"); }
                    if slot >= runtime.slot_count() { bail!("population slot {slot} is out of range"); }
                    let mut stimuli_by_slot = vec![Vec::<Stimulus>::new(); runtime.slot_count()];
                    let mut active = vec![false; runtime.slot_count()];
                    active[slot] = true;
                    stimuli_by_slot[slot] = build_stimuli(&snapshot, &groups, &stimulate, &stimulate_body)?;
                    runtime.step_batch_no_read(&stimuli_by_slot, &active, plasticity, steps)?;
                    neural_step = neural_step
                        .checked_add(steps as u64)
                        .context("population neural step counter overflow")?;
                    write_json(&mut stdout, &StepResponse { ok: true, step: neural_step })?;
                    Ok(())
                })();
                if let Err(error) = result { write_error(&mut stdout, error)?; }
                Ok(())
            }
            Request::TransactionStats { slot } => {
                let result = (|| -> Result<()> {
                    let stats = runtime.transaction_stats(slot, weight_max)?;
                    write_json(
                        &mut stdout,
                        &TransactionStatsResponse {
                            ok: true,
                            event: "transaction_stats",
                            slot,
                            plastic_edges: stats.plastic_edges,
                            dirty_edges: stats.dirty_edges,
                            dirty_fraction: if stats.plastic_edges == 0 {
                                0.0
                            } else {
                                stats.dirty_edges as f64 / stats.plastic_edges as f64
                            },
                            nonzero_shift_edges: stats.nonzero_shift_edges,
                            lower_bound_changed_edges: stats.lower_bound_changed_edges,
                            upper_bound_changed_edges: stats.upper_bound_changed_edges,
                        },
                    )?;
                    Ok(())
                })();
                if let Err(error) = result { write_error(&mut stdout, error)?; }
                Ok(())
            }
            Request::CommitSlot { slot, source_weight_version } => {
                let result = (|| -> Result<()> {
                    if source_weight_version > global_weight_version {
                        bail!(
                            "slot source weight version {} is newer than global version {}",
                            source_weight_version,
                            global_weight_version
                        );
                    }
                    let commit_from_version = global_weight_version;
                    runtime.commit_and_restart_slot(slot)?;
                    global_weight_version = global_weight_version
                        .checked_add(1)
                        .context("global weight version overflow")?;
                    write_json(
                        &mut stdout,
                        &CommitResponse {
                            ok: true,
                            event: "transaction_committed",
                            slot,
                            source_weight_version,
                            commit_from_version,
                            commit_weight_version: global_weight_version,
                            staleness: commit_from_version - source_weight_version,
                        },
                    )?;
                    Ok(())
                })();
                if let Err(error) = result { write_error(&mut stdout, error)?; }
                Ok(())
            }
            Request::RestartSlot { slot } => {
                let result = runtime.restart_slot(slot);
                match result {
                    Ok(()) => write_json(
                        &mut stdout,
                        &SimpleResponse { ok: true, event: "slot_restarted", global_weight_version },
                    ),
                    Err(error) => write_error(&mut stdout, error),
                }
            }
            Request::Quit => {
                write_json(
                    &mut stdout,
                    &SimpleResponse { ok: true, event: "bye", global_weight_version },
                )?;
                break;
            }
        };
        result?;
    }
    Ok(())
}
