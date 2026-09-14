use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use vf_neural::{ConnectomeSnapshot, CpuRuntime, NeuralParams, NeuralState, Stimulus};
use vf_runner::checkpoint::{load_checkpoint, save_checkpoint};

#[derive(Debug, Parser)]
#[command(
    name = "neural-bridge",
    about = "Persistent virtual-fly CNS process for body/environment closed loops"
)]
struct Args {
    #[arg(long, default_value = "artifacts/malecns-v1.0")]
    snapshot: PathBuf,
    #[arg(long)]
    groups: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Gpu)]
    backend: Backend,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
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
#[serde(tag = "type", rename_all = "snake_case")]
enum Request {
    Ping,
    Step {
        #[serde(default)]
        stimulate: HashMap<String, f32>,
        #[serde(default)]
        stimulate_body: Vec<(u64, f32)>,
        #[serde(default = "default_true")]
        plasticity: bool,
        #[serde(default = "default_one")]
        steps: usize,
        #[serde(default)]
        read: Vec<String>,
        #[serde(default)]
        read_body: Vec<u64>,
    },
    SaveWeights {
        path: PathBuf,
    },
    SaveCheckpoint {
        path: PathBuf,
    },
    LoadCheckpoint {
        path: PathBuf,
    },
    Quit,
}

fn default_true() -> bool {
    true
}

fn default_one() -> usize {
    1
}

#[derive(Debug, Serialize)]
struct ReadyResponse<'a> {
    ok: bool,
    event: &'a str,
    backend: String,
    neurons: usize,
    edges: usize,
    groups: Vec<&'a str>,
}

#[derive(Debug, Serialize)]
struct GroupReadout {
    spikes: usize,
    neurons: usize,
    /// Diagnostic population statistic retained only for legacy probes. The
    /// target motor boundary uses individual body-ID readout instead.
    spike_fraction: f32,
}

#[derive(Debug, Serialize)]
struct BodyReadout {
    body_id: u64,
    spike: bool,
}

#[derive(Debug, Serialize)]
struct StepResponse {
    ok: bool,
    step: u64,
    read: HashMap<String, GroupReadout>,
    read_body: Vec<BodyReadout>,
}

#[derive(Debug, Serialize)]
struct WeightCheckpointResponse {
    ok: bool,
    event: &'static str,
    path: String,
    weights: usize,
}

#[derive(Debug, Serialize)]
struct StateCheckpointResponse {
    ok: bool,
    event: &'static str,
    path: String,
    step: u64,
    neurons: usize,
    edges: usize,
}

#[derive(Debug, Serialize)]
struct SimpleResponse<'a> {
    ok: bool,
    event: &'a str,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    ok: bool,
    error: String,
}

enum Runtime {
    Cpu(CpuRuntime),
    Gpu(vf_neural::gpu::GpuRuntime),
}

impl Runtime {
    fn step(&mut self, stimuli: &[Stimulus], plasticity: bool) -> Result<()> {
        match self {
            Runtime::Cpu(runtime) => {
                runtime.step(stimuli, plasticity)?;
                Ok(())
            }
            Runtime::Gpu(runtime) => runtime.step(stimuli, plasticity),
        }
    }

    fn spikes(&self) -> Result<Vec<u32>> {
        match self {
            Runtime::Cpu(runtime) => Ok(runtime
                .spikes()
                .iter()
                .map(|&value| value as u32)
                .collect()),
            Runtime::Gpu(runtime) => runtime.read_spikes(),
        }
    }

    fn weights(&self) -> Result<Vec<f32>> {
        match self {
            Runtime::Cpu(runtime) => Ok(runtime.weights().to_vec()),
            Runtime::Gpu(runtime) => Ok(runtime.state()?.weights),
        }
    }

    fn state(&self) -> Result<NeuralState> {
        match self {
            Runtime::Cpu(runtime) => Ok(runtime.state()),
            Runtime::Gpu(runtime) => runtime.state(),
        }
    }

    fn load_state(&mut self, state: &NeuralState) -> Result<()> {
        match self {
            Runtime::Cpu(runtime) => runtime.load_state(state),
            Runtime::Gpu(runtime) => runtime.load_state(state),
        }
    }

    fn backend_name(&self) -> String {
        match self {
            Runtime::Cpu(_) => "cpu-rayon".to_owned(),
            Runtime::Gpu(runtime) => format!("gpu:{}", runtime.adapter_name()),
        }
    }
}

fn resolve_groups(
    snapshot: &ConnectomeSnapshot,
    config: GroupConfigFile,
) -> Result<HashMap<String, Group>> {
    // Schema v2 only changes experiment/provenance metadata and keeps the
    // executable `groups.{name}.body_ids` shape identical to v1. Accept both so
    // older generated files remain usable while the current Flyppy setup can
    // carry explicit reinforcement provenance without breaking the bridge.
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
    let snapshot = ConnectomeSnapshot::load(&args.snapshot)
        .with_context(|| format!("load MaleCNS snapshot from {}", args.snapshot.display()))?;
    let config: GroupConfigFile = serde_json::from_slice(
        &fs::read(&args.groups)
            .with_context(|| format!("read group config from {}", args.groups.display()))?,
    )
    .with_context(|| format!("parse group config from {}", args.groups.display()))?;
    let groups = resolve_groups(&snapshot, config)?;
    let params = NeuralParams::default();
    let mut runtime = match args.backend {
        Backend::Cpu => Runtime::Cpu(CpuRuntime::new(snapshot.clone(), params.clone())?),
        Backend::Gpu => Runtime::Gpu(vf_neural::gpu::GpuRuntime::new(
            snapshot.clone(),
            params.clone(),
        )?),
    };

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut group_names: Vec<&str> = groups.keys().map(String::as_str).collect();
    group_names.sort_unstable();
    write_json(
        &mut stdout,
        &ReadyResponse {
            ok: true,
            event: "ready",
            backend: runtime.backend_name(),
            neurons: snapshot.neuron_count(),
            edges: snapshot.edge_count(),
            groups: group_names,
        },
    )?;

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
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
                &SimpleResponse {
                    ok: true,
                    event: "pong",
                },
            ),
            Request::Step {
                stimulate,
                stimulate_body,
                plasticity,
                steps,
                read,
                read_body,
            } => {
                if steps == 0 {
                    write_error(&mut stdout, "steps must be >= 1")
                } else {
                    let mut stimuli = Vec::new();
                    let mut request_error: Option<anyhow::Error> = None;
                    for (name, current) in stimulate {
                        match groups.get(&name) {
                            Some(group) => {
                                stimuli.extend(group.indices.iter().copied().map(|index| Stimulus {
                                    index,
                                    current,
                                }));
                            }
                            None => {
                                request_error = Some(anyhow::anyhow!(
                                    "unknown stimulation group {name:?}"
                                ));
                                break;
                            }
                        }
                    }
                    if request_error.is_none() {
                        for (body_id, current) in stimulate_body {
                            match snapshot.index_of_body_id(body_id) {
                                Some(index) => stimuli.push(Stimulus { index, current }),
                                None => {
                                    request_error = Some(anyhow::anyhow!(
                                        "unknown stimulation body ID {body_id}"
                                    ));
                                    break;
                                }
                            }
                        }
                    }

                    if let Some(error) = request_error {
                        write_error(&mut stdout, error)
                    } else {
                        let mut step_error: Option<anyhow::Error> = None;
                        for _ in 0..steps {
                            if let Err(error) = runtime.step(&stimuli, plasticity) {
                                step_error = Some(error);
                                break;
                            }
                        }
                        if let Some(error) = step_error {
                            write_error(&mut stdout, error)
                        } else {
                            match runtime.spikes() {
                                Ok(spikes) => {
                                    let mut readout = HashMap::new();
                                    let mut read_error: Option<anyhow::Error> = None;
                                    for name in read {
                                        match groups.get(&name) {
                                            Some(group) => {
                                                let count = group
                                                    .indices
                                                    .iter()
                                                    .filter(|&&index| spikes[index] != 0)
                                                    .count();
                                                readout.insert(
                                                    name,
                                                    GroupReadout {
                                                        spikes: count,
                                                        neurons: group.indices.len(),
                                                        spike_fraction: count as f32
                                                            / group.indices.len() as f32,
                                                    },
                                                );
                                            }
                                            None => {
                                                read_error = Some(anyhow::anyhow!(
                                                    "unknown read group {name:?}"
                                                ));
                                                break;
                                            }
                                        }
                                    }
                                    let mut body_readout = Vec::with_capacity(read_body.len());
                                    if read_error.is_none() {
                                        for body_id in read_body {
                                            match snapshot.index_of_body_id(body_id) {
                                                Some(index) => body_readout.push(BodyReadout {
                                                    body_id,
                                                    spike: spikes[index] != 0,
                                                }),
                                                None => {
                                                    read_error = Some(anyhow::anyhow!(
                                                        "unknown read body ID {body_id}"
                                                    ));
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    if let Some(error) = read_error {
                                        write_error(&mut stdout, error)
                                    } else {
                                        write_json(
                                            &mut stdout,
                                            &StepResponse {
                                                ok: true,
                                                step: runtime.state()?.step,
                                                read: readout,
                                                read_body: body_readout,
                                            },
                                        )
                                    }
                                }
                                Err(error) => write_error(&mut stdout, error),
                            }
                        }
                    }
                }
            }
            Request::SaveWeights { path } => {
                let result = (|| -> Result<()> {
                    let weights = runtime.weights()?;
                    let bytes: Vec<u8> = weights.iter().flat_map(|value| value.to_le_bytes()).collect();
                    if let Some(parent) = path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::write(&path, bytes)?;
                    write_json(
                        &mut stdout,
                        &WeightCheckpointResponse {
                            ok: true,
                            event: "weights_saved",
                            path: path.display().to_string(),
                            weights: weights.len(),
                        },
                    )?;
                    Ok(())
                })();
                if let Err(error) = result {
                    write_error(&mut stdout, error)?;
                }
                Ok(())
            }
            Request::SaveCheckpoint { path } => {
                let result = (|| -> Result<()> {
                    let state = runtime.state()?;
                    save_checkpoint(&path, &snapshot, &state)?;
                    write_json(
                        &mut stdout,
                        &StateCheckpointResponse {
                            ok: true,
                            event: "checkpoint_saved",
                            path: path.display().to_string(),
                            step: state.step,
                            neurons: snapshot.neuron_count(),
                            edges: snapshot.edge_count(),
                        },
                    )?;
                    Ok(())
                })();
                if let Err(error) = result {
                    write_error(&mut stdout, error)?;
                }
                Ok(())
            }
            Request::LoadCheckpoint { path } => {
                let result = (|| -> Result<()> {
                    let state = load_checkpoint(&path, &snapshot)?;
                    runtime.load_state(&state)?;
                    write_json(
                        &mut stdout,
                        &StateCheckpointResponse {
                            ok: true,
                            event: "checkpoint_loaded",
                            path: path.display().to_string(),
                            step: state.step,
                            neurons: snapshot.neuron_count(),
                            edges: snapshot.edge_count(),
                        },
                    )?;
                    Ok(())
                })();
                if let Err(error) = result {
                    write_error(&mut stdout, error)?;
                }
                Ok(())
            }
            Request::Quit => {
                write_json(
                    &mut stdout,
                    &SimpleResponse {
                        ok: true,
                        event: "bye",
                    },
                )?;
                break;
            }
        };
        if let Err(error) = result {
            write_error(&mut stdout, error)?;
        }
    }
    Ok(())
}
