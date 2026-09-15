use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use vf_neural::{
    ConnectomeSnapshot, NeuralParams, PlasticFastGraph, Stimulus,
    gpu_population::GpuPopulationRuntime as FrontierRuntime,
    gpu_population_reference::GpuPopulationRuntime as ReferenceRuntime,
    model::nt,
};

#[derive(Debug, Parser)]
#[command(
    name = "population-frontier-malecns-parity",
    about = "Compare dense reference and sparse population runtime on real MaleCNS"
)]
struct Args {
    #[arg(long, default_value = "artifacts/malecns-v1.0")]
    snapshot: PathBuf,
}

#[derive(Debug, Serialize)]
struct ResultJson {
    neuron_count: usize,
    edge_count: usize,
    plastic_edge_count: usize,
    chosen_pre: usize,
    chosen_post: usize,
    chosen_dopamine_pre: usize,
    steps_compared: usize,
    all_neuron_events_compared_each_step: bool,
    final_full_neuron_state_compared: bool,
    full_plastic_state_compared: bool,
    committed_global_weights_compared: bool,
    bitwise_equal: bool,
}

fn params() -> NeuralParams {
    NeuralParams {
        membrane_decay: 0.95,
        threshold: 1.0,
        reset: 0.0,
        refractory_steps: 2,
        trace_decay: 0.95,
        eligibility_decay: 0.995,
        learning_rate: 0.00001,
        weight_max: 1_000.0,
        modulator_decay: 0.98,
        // Keep the real graph subcritical for a deterministic parity probe.
        synapse_scale: 0.00025,
        modulator_scale: 0.00005,
    }
}

fn assert_f32_bits_eq(label: &str, left: &[f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        bail!("{label} length mismatch: {} != {}", left.len(), right.len());
    }
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        if a.to_bits() != b.to_bits() {
            bail!("{label}[{index}] mismatch: {a:?} != {b:?}");
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let snapshot = ConnectomeSnapshot::load_dir(&args.snapshot)
        .with_context(|| format!("load MaleCNS snapshot from {}", args.snapshot.display()))?;
    let plastic = PlasticFastGraph::compile(&snapshot).context("compile PlasticFastGraph")?;
    let &source_edge = plastic
        .edge_indices
        .first()
        .context("MaleCNS PlasticFastGraph is empty")?;
    let source_edge = source_edge as usize;
    let pre = snapshot.pre_indices[source_edge] as usize;
    let post = snapshot.edge_posts[source_edge] as usize;

    let begin = snapshot.row_offsets[post] as usize;
    let end = snapshot.row_offsets[post + 1] as usize;
    let dopamine_pre = (begin..end)
        .map(|edge| snapshot.pre_indices[edge] as usize)
        .find(|&candidate| snapshot.neurotransmitters[candidate] == nt::DOPAMINE)
        .with_context(|| format!("plastic post {post} unexpectedly lacks dopamine input"))?;

    let mut reference = ReferenceRuntime::new(&snapshot, params(), 1)?;
    let mut frontier = FrontierRuntime::new(&snapshot, params(), 1)?;
    let active = [true];
    let read_all = (0..snapshot.neuron_count()).collect::<Vec<_>>();
    let sequence = [
        vec![Stimulus { neuron: pre, current: 2.0 }],
        vec![
            Stimulus { neuron: dopamine_pre, current: 2.0 },
            Stimulus { neuron: post, current: 2.0 },
        ],
        vec![],
        vec![],
        vec![Stimulus { neuron: pre, current: -2.0 }],
        vec![Stimulus { neuron: dopamine_pre, current: 2.0 }],
        vec![Stimulus { neuron: post, current: -2.0 }],
        vec![],
        vec![],
        vec![],
    ];

    // Compare every released neuron's signed 0/1/2 event on every step. This
    // keeps the per-step contract strong without adding a production-only full
    // internal-state readback API to the live runtime.
    for (step, stimuli) in sequence.iter().enumerate() {
        let by_slot = [stimuli.clone()];
        let reference_events = reference
            .step_batch_with_read(&by_slot, &active, &read_all, true)?;
        let frontier_events = frontier
            .step_batch_with_read(&by_slot, &active, &read_all, true)?;
        if reference_events != frontier_events {
            bail!("all-neuron event mismatch at step {step}");
        }
    }

    // Read the large state arrays once after the fixed trajectory. This checks
    // all causally relevant neuron/plastic state bitwise while avoiding repeated
    // hundreds-of-MiB diagnostic transfers.
    let a = reference.debug_slot_state(0)?;
    let b = frontier.debug_slot_state(0)?;
    assert_f32_bits_eq("membrane", &a.membrane, &b.membrane)?;
    assert_f32_bits_eq("trace", &a.trace, &b.trace)?;
    assert_f32_bits_eq("modulation", &a.modulation, &b.modulation)?;
    if a.refractory != b.refractory { bail!("final refractory mismatch"); }
    if a.spikes != b.spikes { bail!("final spike mismatch"); }
    assert_f32_bits_eq("plastic_weight", &a.plastic_weight, &b.plastic_weight)?;
    assert_f32_bits_eq("eligibility", &a.eligibility, &b.eligibility)?;
    assert_f32_bits_eq("transaction_shift", &a.transaction_shift, &b.transaction_shift)?;
    assert_f32_bits_eq("transaction_lo", &a.transaction_lo, &b.transaction_lo)?;
    assert_f32_bits_eq("transaction_hi", &a.transaction_hi, &b.transaction_hi)?;

    reference.commit_and_restart_slot(0)?;
    frontier.commit_and_restart_slot(0)?;
    let reference_weights = reference.global_weights()?;
    let frontier_weights = frontier.global_weights()?;
    assert_f32_bits_eq("committed_global_weights", &reference_weights, &frontier_weights)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&ResultJson {
            neuron_count: snapshot.neuron_count(),
            edge_count: snapshot.edge_count(),
            plastic_edge_count: plastic.edge_count(),
            chosen_pre: pre,
            chosen_post: post,
            chosen_dopamine_pre: dopamine_pre,
            steps_compared: sequence.len(),
            all_neuron_events_compared_each_step: true,
            final_full_neuron_state_compared: true,
            full_plastic_state_compared: true,
            committed_global_weights_compared: true,
            bitwise_equal: true,
        })?
    );
    Ok(())
}
