use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use vf_neural::{ConnectomeSnapshot, CpuRuntime, NeuralParams, Stimulus};

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
    #[serde(default)]
    modulator_role: i8,
}

#[derive(Debug)]
struct Group {
    indices: Vec<usize>,
    modulator_role: i8,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Request {
    Ping,
    Step {
        #[serde(default)]
        stimulate: HashMap<String, f32>,
        #[serde(default = "default_true")]
        plasticity: bool,
        #[serde(default = "default_one")]
        steps: usize,
        #[serde(default)]
        read: Vec<String>,
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
    spike_fraction: f32,
}

#[derive(Debug, Serialize)]
struct StepResponse {
    ok: bool,
    step: u64,
    read: HashMap<String, GroupReadout>,
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
    fn set_modulator_role(&mut self, neuron: usize, role: i8) -> Result<()> {
        match self {
            Runtime::Cpu(runtime) => runtime.set_modulator_role(neuron, role),
            Runtime::Gpu(runtime) => runtime.set_modulator_role(neuron, role),
        }
    }

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
            Runtime::Cpu(runtime) => Ok(runtime.spikes().iter().map(|&value| value as u32).collect()),
            Runtime::Gpu(runtime) => runtime.read_spikes(),
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
    if config.schema_version != 1 {
        bail!("unsupported group config schema {}", config.schema_version);
    }
    if config.groups.is_empty() {
        bail!("group config contains no groups");
    }

    let mut groups = HashMap::with_capacity(config.groups.len());
    for (name, group) in config.groups {
        if !(-1..=1).contains(&group.modulator_role) {
            bail!("group {name}: modulator_role must be -1, 0, or 1");
        }
        let mut indices = Vec::with_capacity(group.body_ids.len());
        for body_id in group.body_ids {
            let index = snapshot
                .index_of_body_id(body_id)
                .with_context(|| format!("group {name}: body ID {body_id} not present in snapshot"))?;
            indices.push(index);
        }
        indices.sort_unstable();
        indices.dedup();
        if indices.is_empty() {
            bail!("group {name} resolved to zero neurons");
        }
        groups.insert(
            name,
            Group {
                indices,
                modulator_role: group.modulator_role,
            },
        );
    }
    Ok(groups)
}

fn stimuli_from_groups(
    groups: &HashMap<String, Group>,
    requested: &HashMap<String, f32>,
) -> Result<Vec<Stimulus>> {
    let capacity = requested
        .keys()
        .filter_map(|name| groups.get(name))
        .map(|group| group.indices.len())
        .sum();
    let mut stimuli = Vec::with_capacity(capacity);
    for (name, current) in requested {
        if !current.is_finite() {
            bail!("stimulus current for group {name} is non-finite");
        }
        let group = groups
            .get(name)
            .with_context(|| format!("unknown stimulus group: {name}"))?;
        stimuli.extend(group.indices.iter().copied().map(|neuron| Stimulus {
            neuron,
            current: *current,
        }));
    }
    Ok(stimuli)
}

fn read_groups(
    groups: &HashMap<String, Group>,
    names: &[String],
    spikes: &[u32],
) -> Result<HashMap<String, GroupReadout>> {
    let mut result = HashMap::with_capacity(names.len());
    for name in names {
        let group = groups
            .get(name)
            .with_context(|| format!("unknown read group: {name}"))?;
        let spike_count = group
            .indices
            .iter()
            .filter(|&&index| spikes[index] != 0)
            .count();
        result.insert(
            name.clone(),
            GroupReadout {
                spikes: spike_count,
                neurons: group.indices.len(),
                spike_fraction: spike_count as f32 / group.indices.len() as f32,
            },
        );
    }
    Ok(result)
}

fn write_json<T: Serialize>(stdout: &mut impl Write, value: &T) -> Result<()> {
    serde_json::to_writer(&mut *stdout, value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let snapshot = ConnectomeSnapshot::load_dir(&args.snapshot)
        .with_context(|| format!("failed to load snapshot {}", args.snapshot.display()))?;
    let config: GroupConfigFile = serde_json::from_slice(
        &fs::read(&args.groups)
            .with_context(|| format!("failed to read groups {}", args.groups.display()))?,
    )
    .context("invalid group config JSON")?;
    let groups = resolve_groups(&snapshot, config)?;

    let mut runtime = match args.backend {
        Backend::Cpu => Runtime::Cpu(CpuRuntime::new(snapshot.clone(), NeuralParams::default())),
        Backend::Gpu => Runtime::Gpu(vf_neural::gpu::GpuRuntime::new(
            &snapshot,
            NeuralParams::default(),
        )?),
    };
    for group in groups.values() {
        if group.modulator_role != 0 {
            for &neuron in &group.indices {
                runtime.set_modulator_role(neuron, group.modulator_role)?;
            }
        }
    }

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut group_names = groups.keys().map(String::as_str).collect::<Vec<_>>();
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

    let mut step_counter = 0u64;
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<Request>(&line) {
            Ok(request) => request,
            Err(error) => {
                write_json(
                    &mut stdout,
                    &ErrorResponse {
                        ok: false,
                        error: format!("invalid request JSON: {error}"),
                    },
                )?;
                continue;
            }
        };

        let result: Result<bool> = (|| match request {
            Request::Ping => {
                write_json(
                    &mut stdout,
                    &SimpleResponse {
                        ok: true,
                        event: "pong",
                    },
                )?;
                Ok(true)
            }
            Request::Quit => {
                write_json(
                    &mut stdout,
                    &SimpleResponse {
                        ok: true,
                        event: "bye",
                    },
                )?;
                Ok(false)
            }
            Request::Step {
                stimulate,
                plasticity,
                steps,
                read,
            } => {
                if steps == 0 {
                    bail!("step request requires steps >= 1");
                }
                let stimuli = stimuli_from_groups(&groups, &stimulate)?;
                for _ in 0..steps {
                    runtime.step(&stimuli, plasticity)?;
                    step_counter += 1;
                }
                let readout = if read.is_empty() {
                    HashMap::new()
                } else {
                    let spikes = runtime.spikes()?;
                    read_groups(&groups, &read, &spikes)?
                };
                write_json(
                    &mut stdout,
                    &StepResponse {
                        ok: true,
                        step: step_counter,
                        read: readout,
                    },
                )?;
                Ok(true)
            }
        })();

        match result {
            Ok(true) => {}
            Ok(false) => break,
            Err(error) => write_json(
                &mut stdout,
                &ErrorResponse {
                    ok: false,
                    error: format!("{error:#}"),
                },
            )?,
        }
    }
    Ok(())
}
