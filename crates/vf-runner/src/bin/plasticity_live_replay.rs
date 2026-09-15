use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use vf_neural::{
    ConnectomeSnapshot, NeuralParams, PlasticFastGraph, Stimulus,
    gpu_population_frontier_profiled::GpuPopulationRuntime,
};
use vf_runner::checkpoint::load_checkpoint;

#[derive(Debug, Parser)]
#[command(name = "plasticity-live-replay")]
struct Args {
    #[arg(long, default_value = "artifacts/malecns-v1.0")]
    snapshot: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    sequence: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ReplayRecord {
    Step {
        stimuli: Vec<(usize, f32)>,
        #[serde(default = "default_true")]
        plasticity: bool,
        #[serde(default = "default_one")]
        steps: usize,
        #[serde(default)]
        sample: bool,
        #[serde(default)]
        normal_step: Option<u64>,
    },
    Commit,
    Restart,
}

fn default_true() -> bool { true }
fn default_one() -> usize { 1 }

#[derive(Debug, Serialize)]
struct SampleStats {
    sample: usize,
    neural_step: u64,
    normal_step: Option<u64>,
    eligibility_before_nonzero: usize,
    local_nonzero: usize,
    eager_live_union: usize,
    eligibility_after_nonzero: usize,
    modulated_posts: usize,
    plastic_edges_under_modulation: usize,
    delta_nonzero: usize,
    lazy_immediate_union: usize,
}

#[derive(Debug, Serialize)]
struct ReplaySummary {
    neuron_count: usize,
    edge_count: usize,
    plastic_edge_count: usize,
    sampled_steps: usize,
    replayed_neural_steps: u64,
    samples: Vec<SampleStats>,
}

fn activity_sign(event: u32) -> f32 {
    match event & 0x7fff_ffff {
        1 => 1.0,
        2 => -1.0,
        _ => 0.0,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let snapshot = ConnectomeSnapshot::load_dir(&args.snapshot)
        .with_context(|| format!("load snapshot from {}", args.snapshot.display()))?;
    let plastic = PlasticFastGraph::compile(&snapshot)?;
    let params = NeuralParams::default();
    let (_manifest, state) = load_checkpoint(
        &args.checkpoint,
        &snapshot.manifest.dataset,
        snapshot.neuron_count(),
        snapshot.edge_count(),
    )?;
    let mut runtime = GpuPopulationRuntime::new(&snapshot, params, 1)?;
    runtime.load_global_weights(&state.weights)?;

    let text = fs::read_to_string(&args.sequence)
        .with_context(|| format!("read replay sequence {}", args.sequence.display()))?;
    let mut records = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let raw = raw.trim();
        if raw.is_empty() { continue; }
        records.push(
            serde_json::from_str::<ReplayRecord>(raw)
                .with_context(|| format!("parse replay record line {}", line_no + 1))?,
        );
    }
    if records.is_empty() { bail!("replay sequence is empty"); }

    let mut neural_step = 0u64;
    let mut samples = Vec::new();
    for record in records {
        match record {
            ReplayRecord::Step { stimuli, plasticity, steps, sample, normal_step } => {
                if steps == 0 { bail!("replay step count must be >= 1"); }
                let stimuli = stimuli
                    .into_iter()
                    .map(|(neuron, current)| Stimulus { neuron, current })
                    .collect::<Vec<_>>();
                if sample && steps != 1 {
                    bail!("sampled replay record must contain exactly one neural step");
                }
                if sample {
                    let before = runtime.plasticity_profile_state(0)?;
                    runtime.step_batch_no_read(&[stimuli.clone()], &[true], plasticity, 1)?;
                    neural_step += 1;
                    let after = runtime.plasticity_profile_state(0)?;

                    let mut eligibility_before_nonzero = 0usize;
                    let mut local_nonzero = 0usize;
                    let mut eager_live_union = 0usize;
                    let mut eligibility_after_nonzero = 0usize;
                    let mut plastic_edges_under_modulation = 0usize;
                    let mut delta_nonzero = 0usize;
                    let mut lazy_immediate_union = 0usize;

                    for plastic_edge in 0..plastic.edge_count() {
                        let source_edge = plastic.edge_indices[plastic_edge] as usize;
                        let pre = snapshot.pre_indices[source_edge] as usize;
                        let post = plastic.post_indices[plastic_edge] as usize;
                        let eligibility_before = before.eligibility[plastic_edge];
                        let eligibility_after = after.eligibility[plastic_edge];
                        let local = before.trace[pre] * activity_sign(after.activity[post])
                            - before.trace[post] * activity_sign(before.activity[pre]);
                        let modulation = after.modulation[post];
                        let delta = params.learning_rate * modulation * eligibility_after;

                        let eligibility_before_live = eligibility_before != 0.0;
                        let local_live = local != 0.0;
                        let eligibility_after_live = eligibility_after != 0.0;
                        let modulation_live = modulation != 0.0;
                        let delta_live = delta != 0.0;

                        eligibility_before_nonzero += usize::from(eligibility_before_live);
                        local_nonzero += usize::from(local_live);
                        eager_live_union += usize::from(eligibility_before_live || local_live);
                        eligibility_after_nonzero += usize::from(eligibility_after_live);
                        plastic_edges_under_modulation += usize::from(modulation_live);
                        delta_nonzero += usize::from(delta_live);
                        // If geometric eligibility decay is represented lazily,
                        // an edge needs immediate work only when a new local
                        // contribution arrives or its current modulation makes
                        // the decayed eligibility causally affect weight now.
                        lazy_immediate_union += usize::from(local_live || delta_live);
                    }

                    let modulated_posts = plastic
                        .dopamine_capable_posts
                        .iter()
                        .filter(|&&post| after.modulation[post as usize] != 0.0)
                        .count();

                    samples.push(SampleStats {
                        sample: samples.len(),
                        neural_step,
                        normal_step,
                        eligibility_before_nonzero,
                        local_nonzero,
                        eager_live_union,
                        eligibility_after_nonzero,
                        modulated_posts,
                        plastic_edges_under_modulation,
                        delta_nonzero,
                        lazy_immediate_union,
                    });
                } else {
                    runtime.step_batch_no_read(&[stimuli], &[true], plasticity, steps)?;
                    neural_step = neural_step
                        .checked_add(steps as u64)
                        .context("replayed neural step counter overflow")?;
                }
            }
            ReplayRecord::Commit => {
                runtime.commit_and_restart_slot(0)?;
            }
            ReplayRecord::Restart => {
                runtime.restart_slot(0)?;
            }
        }
    }

    let summary = ReplaySummary {
        neuron_count: snapshot.neuron_count(),
        edge_count: snapshot.edge_count(),
        plastic_edge_count: plastic.edge_count(),
        sampled_steps: samples.len(),
        replayed_neural_steps: neural_step,
        samples,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
