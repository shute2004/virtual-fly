use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use vf_neural::{ConnectomeSnapshot, CpuRuntime, NeuralParams, Stimulus};

#[derive(Debug, Parser)]
#[command(
    name = "malecns-conditioning",
    about = "Run associative conditioning on the released MaleCNS graph"
)]
struct Args {
    #[arg(long, default_value = "artifacts/malecns-v1.0")]
    snapshot: PathBuf,
    #[arg(
        long,
        default_value = "artifacts/malecns-v1.0/conditioning-v0.json"
    )]
    config: PathBuf,
    #[arg(long, default_value = "artifacts/experiments/conditioning-v0")]
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Gpu)]
    backend: Backend,
    #[arg(long, default_value_t = 24)]
    cycles: usize,
    #[arg(long, default_value_t = 2)]
    cue_steps: usize,
    #[arg(long, default_value_t = 2)]
    pairing_steps: usize,
    #[arg(long, default_value_t = 4)]
    intertrial_steps: usize,
    #[arg(long, default_value_t = 2.0)]
    cue_current: f32,
    #[arg(long, default_value_t = 2.0)]
    dan_current: f32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}

#[derive(Debug, Deserialize)]
struct ExperimentConfig {
    schema_version: u32,
    dataset: String,
    experiment: String,
    groups: Groups,
}

#[derive(Debug, Deserialize)]
struct Groups {
    reward_cue: Vec<GroupNeuron>,
    aversive_cue: Vec<GroupNeuron>,
    reward_dan: Vec<GroupNeuron>,
    aversive_dan: Vec<GroupNeuron>,
    readout: Vec<GroupNeuron>,
}

#[derive(Debug, Deserialize)]
struct GroupNeuron {
    body_id: u64,
}

#[derive(Debug, Serialize)]
struct ResultFile {
    dataset: String,
    experiment: String,
    backend: String,
    cycles: usize,
    elapsed_seconds: f64,
    groups: GroupSizes,
    synapses: SynapseStats,
    learned_weights_file: String,
    interpretation: String,
}

#[derive(Debug, Serialize)]
struct GroupSizes {
    reward_cue: usize,
    aversive_cue: usize,
    reward_dan: usize,
    aversive_dan: usize,
    readout: usize,
}

#[derive(Debug, Serialize)]
struct SynapseStats {
    total: usize,
    changed: usize,
    mean_abs_delta: f64,
    max_abs_delta: f32,
    reward_cue_outgoing_mean_delta: f64,
    aversive_cue_outgoing_mean_delta: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.cycles == 0 {
        bail!("--cycles must be greater than zero");
    }

    let snapshot = ConnectomeSnapshot::load_dir(&args.snapshot)
        .with_context(|| format!("failed to load {}", args.snapshot.display()))?;
    let config: ExperimentConfig = serde_json::from_slice(
        &fs::read(&args.config)
            .with_context(|| format!("failed to read {}", args.config.display()))?,
    )
    .context("invalid conditioning config")?;

    if config.schema_version != 1 {
        bail!("unsupported conditioning config schema {}", config.schema_version);
    }
    if config.dataset != snapshot.manifest.dataset {
        bail!(
            "config dataset {} does not match snapshot {}",
            config.dataset,
            snapshot.manifest.dataset
        );
    }

    let reward_cue = resolve_group(&snapshot, &config.groups.reward_cue, "reward_cue")?;
    let aversive_cue = resolve_group(&snapshot, &config.groups.aversive_cue, "aversive_cue")?;
    let reward_dan = resolve_group(&snapshot, &config.groups.reward_dan, "reward_dan")?;
    let aversive_dan = resolve_group(&snapshot, &config.groups.aversive_dan, "aversive_dan")?;
    let readout = resolve_group(&snapshot, &config.groups.readout, "readout")?;

    println!(
        "dataset={} neurons={} edges={}",
        snapshot.manifest.dataset,
        snapshot.neuron_count(),
        snapshot.edge_count()
    );
    println!(
        "groups reward_cue={} aversive_cue={} reward_dan={} aversive_dan={} readout={}",
        reward_cue.len(),
        aversive_cue.len(),
        reward_dan.len(),
        aversive_dan.len(),
        readout.len()
    );

    let started = Instant::now();
    let (before, after, backend_name) = match args.backend {
        Backend::Cpu => run_cpu(
            &snapshot,
            &args,
            &reward_cue,
            &aversive_cue,
            &reward_dan,
            &aversive_dan,
        )?,
        Backend::Gpu => run_gpu(
            &snapshot,
            &args,
            &reward_cue,
            &aversive_cue,
            &reward_dan,
            &aversive_dan,
        )?,
    };
    let elapsed = started.elapsed().as_secs_f64();

    let stats = synapse_stats(&snapshot, &before, &after, &reward_cue, &aversive_cue)?;
    args.output
        .parent()
        .map(fs::create_dir_all)
        .transpose()
        .context("failed to create experiment parent directory")?;
    fs::create_dir_all(&args.output).context("failed to create experiment output directory")?;

    let weights_path = args.output.join("learned_weights.f32le");
    write_f32_le(&weights_path, &after)?;

    let result = ResultFile {
        dataset: snapshot.manifest.dataset.clone(),
        experiment: config.experiment,
        backend: backend_name,
        cycles: args.cycles,
        elapsed_seconds: elapsed,
        groups: GroupSizes {
            reward_cue: reward_cue.len(),
            aversive_cue: aversive_cue.len(),
            reward_dan: reward_dan.len(),
            aversive_dan: aversive_dan.len(),
            readout: readout.len(),
        },
        synapses: stats,
        learned_weights_file: weights_path.display().to_string(),
        interpretation: concat!(
            "Experience-dependent synaptic state change in the released MaleCNS graph under ",
            "cue/DAN current pairing. Reward and aversive DANs are not assigned external +1/-1 ",
            "signs; their effects differ through the released dopaminergic circuits they occupy."
        )
        .to_owned(),
    };
    let result_path = args.output.join("result.json");
    fs::write(&result_path, serde_json::to_vec_pretty(&result)?)?;

    println!("backend={}", result.backend);
    println!("cycles={} elapsed={:.3}s", result.cycles, result.elapsed_seconds);
    println!(
        "changed_synapses={} / {} mean_abs_delta={:.9} max_abs_delta={:.6}",
        result.synapses.changed,
        result.synapses.total,
        result.synapses.mean_abs_delta,
        result.synapses.max_abs_delta
    );
    println!(
        "reward_cue_outgoing_mean_delta={:+.9}",
        result.synapses.reward_cue_outgoing_mean_delta
    );
    println!(
        "aversive_cue_outgoing_mean_delta={:+.9}",
        result.synapses.aversive_cue_outgoing_mean_delta
    );
    println!("learned_weights={}", weights_path.display());
    println!("result={}", result_path.display());

    if result.synapses.changed == 0 {
        bail!(
            "conditioning produced no synaptic changes; neural/plasticity calibration must be adjusted"
        );
    }
    println!("malecns_conditioning=PASS");
    Ok(())
}

fn resolve_group(
    snapshot: &ConnectomeSnapshot,
    group: &[GroupNeuron],
    name: &str,
) -> Result<Vec<usize>> {
    let mut indices = Vec::with_capacity(group.len());
    for neuron in group {
        let index = snapshot.index_of_body_id(neuron.body_id).with_context(|| {
            format!("{name}: body id {} is not present in snapshot", neuron.body_id)
        })?;
        indices.push(index);
    }
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() {
        bail!("{name} resolved to an empty group");
    }
    Ok(indices)
}

fn group_stimuli(group: &[usize], current: f32) -> Vec<Stimulus> {
    group
        .iter()
        .copied()
        .map(|neuron| Stimulus { neuron, current })
        .collect()
}

fn paired_stimuli(
    cue: &[usize],
    dan: &[usize],
    cue_current: f32,
    dan_current: f32,
) -> Vec<Stimulus> {
    let mut stimuli = group_stimuli(cue, cue_current);
    stimuli.extend(group_stimuli(dan, dan_current));
    stimuli
}

fn run_cpu(
    snapshot: &ConnectomeSnapshot,
    args: &Args,
    reward_cue: &[usize],
    aversive_cue: &[usize],
    reward_dan: &[usize],
    aversive_dan: &[usize],
) -> Result<(Vec<f32>, Vec<f32>, String)> {
    let mut runtime = CpuRuntime::new(snapshot.clone(), NeuralParams::default());
    let before = runtime.weights().to_vec();

    let reward_cue_stimuli = group_stimuli(reward_cue, args.cue_current);
    let aversive_cue_stimuli = group_stimuli(aversive_cue, args.cue_current);
    let reward_pair = paired_stimuli(reward_cue, reward_dan, args.cue_current, args.dan_current);
    let aversive_pair = paired_stimuli(
        aversive_cue,
        aversive_dan,
        args.cue_current,
        args.dan_current,
    );

    for cycle in 0..args.cycles {
        if cycle % 2 == 0 {
            train_trial_cpu(&mut runtime, &reward_cue_stimuli, &reward_pair, args)?;
            train_trial_cpu(&mut runtime, &aversive_cue_stimuli, &aversive_pair, args)?;
        } else {
            train_trial_cpu(&mut runtime, &aversive_cue_stimuli, &aversive_pair, args)?;
            train_trial_cpu(&mut runtime, &reward_cue_stimuli, &reward_pair, args)?;
        }
    }
    Ok((before, runtime.weights().to_vec(), "cpu-rayon".to_owned()))
}

fn train_trial_cpu(
    runtime: &mut CpuRuntime,
    cue: &[Stimulus],
    paired: &[Stimulus],
    args: &Args,
) -> Result<()> {
    for _ in 0..args.cue_steps {
        runtime.step(cue, true)?;
    }
    for _ in 0..args.pairing_steps {
        runtime.step(paired, true)?;
    }
    for _ in 0..args.intertrial_steps {
        runtime.step(&[], true)?;
    }
    Ok(())
}

fn run_gpu(
    snapshot: &ConnectomeSnapshot,
    args: &Args,
    reward_cue: &[usize],
    aversive_cue: &[usize],
    reward_dan: &[usize],
    aversive_dan: &[usize],
) -> Result<(Vec<f32>, Vec<f32>, String)> {
    use vf_neural::gpu::GpuRuntime;

    let mut runtime = GpuRuntime::new(snapshot, NeuralParams::default())?;
    let before = runtime.readback()?.weights;

    let reward_cue_stimuli = group_stimuli(reward_cue, args.cue_current);
    let aversive_cue_stimuli = group_stimuli(aversive_cue, args.cue_current);
    let reward_pair = paired_stimuli(reward_cue, reward_dan, args.cue_current, args.dan_current);
    let aversive_pair = paired_stimuli(
        aversive_cue,
        aversive_dan,
        args.cue_current,
        args.dan_current,
    );

    for cycle in 0..args.cycles {
        if cycle % 2 == 0 {
            train_trial_gpu(&mut runtime, &reward_cue_stimuli, &reward_pair, args)?;
            train_trial_gpu(&mut runtime, &aversive_cue_stimuli, &aversive_pair, args)?;
        } else {
            train_trial_gpu(&mut runtime, &aversive_cue_stimuli, &aversive_pair, args)?;
            train_trial_gpu(&mut runtime, &reward_cue_stimuli, &reward_pair, args)?;
        }
        if (cycle + 1) % 4 == 0 || cycle + 1 == args.cycles {
            println!("conditioning_cycle={}/{}", cycle + 1, args.cycles);
        }
    }

    let adapter = runtime.adapter_name().to_owned();
    let after = runtime.readback()?.weights;
    Ok((before, after, format!("gpu:{adapter}")))
}

fn train_trial_gpu(
    runtime: &mut vf_neural::gpu::GpuRuntime,
    cue: &[Stimulus],
    paired: &[Stimulus],
    args: &Args,
) -> Result<()> {
    for _ in 0..args.cue_steps {
        runtime.step(cue, true)?;
    }
    for _ in 0..args.pairing_steps {
        runtime.step(paired, true)?;
    }
    for _ in 0..args.intertrial_steps {
        runtime.step(&[], true)?;
    }
    Ok(())
}

fn synapse_stats(
    snapshot: &ConnectomeSnapshot,
    before: &[f32],
    after: &[f32],
    reward_cue: &[usize],
    aversive_cue: &[usize],
) -> Result<SynapseStats> {
    if before.len() != after.len() || before.len() != snapshot.edge_count() {
        bail!("weight arrays do not match snapshot edge count");
    }
    let reward: HashSet<u32> = reward_cue.iter().map(|&index| index as u32).collect();
    let aversive: HashSet<u32> = aversive_cue.iter().map(|&index| index as u32).collect();

    let mut changed = 0usize;
    let mut abs_sum = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut reward_sum = 0.0f64;
    let mut reward_count = 0usize;
    let mut aversive_sum = 0.0f64;
    let mut aversive_count = 0usize;

    for edge in 0..before.len() {
        let delta = after[edge] - before[edge];
        let abs = delta.abs();
        if abs > 1e-7 {
            changed += 1;
        }
        abs_sum += abs as f64;
        max_abs = max_abs.max(abs);
        let pre = snapshot.pre_indices[edge];
        if reward.contains(&pre) {
            reward_sum += delta as f64;
            reward_count += 1;
        }
        if aversive.contains(&pre) {
            aversive_sum += delta as f64;
            aversive_count += 1;
        }
    }

    Ok(SynapseStats {
        total: before.len(),
        changed,
        mean_abs_delta: if before.is_empty() {
            0.0
        } else {
            abs_sum / before.len() as f64
        },
        max_abs_delta: max_abs,
        reward_cue_outgoing_mean_delta: mean_or_zero(reward_sum, reward_count),
        aversive_cue_outgoing_mean_delta: mean_or_zero(aversive_sum, aversive_count),
    })
}

fn mean_or_zero(sum: f64, count: usize) -> f64 {
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn write_f32_le(path: &Path, values: &[f32]) -> Result<()> {
    let mut file = fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for &value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}
