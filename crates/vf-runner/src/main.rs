use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use vf_neural::{ConnectomeSnapshot, CpuRuntime, EdgeInput, NeuralParams, Stimulus, nt};

#[derive(Debug, Parser)]
#[command(name = "vf-runner", about = "virtual-fly bootstrap experiment runner")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate local dopamine-gated plasticity on a deliberately tiny circuit.
    /// This is a kernel smoke test, not a claim about fly behaviour or valence.
    KernelSmoke {
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[arg(long, default_value_t = 250)]
        cycles: usize,
        /// Independent CPU individuals. Ignored by the current single-device GPU smoke test.
        #[arg(long, default_value_t = 8)]
        flies: usize,
    },
    /// Inspect a preprocessed MaleCNS snapshot.
    SnapshotInfo {
        #[arg(long)]
        snapshot: PathBuf,
    },
    /// Run the released MaleCNS graph for timing/scale validation.
    /// Plasticity is disabled in this benchmark.
    SnapshotBenchmark {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
        backend: BackendArg,
        #[arg(long, default_value_t = 100)]
        steps: usize,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum BackendArg {
    Auto,
    Cpu,
    Gpu,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::KernelSmoke { backend, cycles, flies } => kernel_smoke(backend, cycles, flies),
        Command::SnapshotInfo { snapshot } => snapshot_info(snapshot),
        Command::SnapshotBenchmark { snapshot, backend, steps } => {
            snapshot_benchmark(snapshot, backend, steps)
        }
    }
}

fn kernel_smoke(backend: BackendArg, cycles: usize, flies: usize) -> Result<()> {
    if cycles == 0 {
        bail!("cycles must be greater than zero");
    }
    match backend {
        BackendArg::Cpu => cpu_kernel_smoke(cycles, flies),
        BackendArg::Gpu => gpu_kernel_smoke(cycles),
        BackendArg::Auto => match gpu_kernel_smoke(cycles) {
            Ok(()) => Ok(()),
            Err(error) => {
                eprintln!("GPU backend unavailable ({error:#}); falling back to CPU.");
                cpu_kernel_smoke(cycles, flies)
            }
        },
    }
}

fn smoke_snapshot() -> Result<ConnectomeSnapshot> {
    // Two independent synthetic pathways. Each has its own dopamine source
    // contacting the corresponding association neuron. There is deliberately
    // no externally assigned positive/negative role: this test only verifies
    // that local dopaminergic activity can gate local synaptic change.
    ConnectomeSnapshot::from_edges(
        8,
        &[
            EdgeInput { pre: 0, post: 2, synapse_count: 60 },
            EdgeInput { pre: 1, post: 3, synapse_count: 60 },
            EdgeInput { pre: 2, post: 4, synapse_count: 60 },
            EdgeInput { pre: 3, post: 5, synapse_count: 60 },
            EdgeInput { pre: 6, post: 2, synapse_count: 60 },
            EdgeInput { pre: 7, post: 3, synapse_count: 60 },
        ],
        vec![
            nt::ACETYLCHOLINE,
            nt::ACETYLCHOLINE,
            nt::ACETYLCHOLINE,
            nt::ACETYLCHOLINE,
            nt::ACETYLCHOLINE,
            nt::ACETYLCHOLINE,
            nt::DOPAMINE,
            nt::DOPAMINE,
        ],
    )
}

fn smoke_params() -> NeuralParams {
    NeuralParams {
        refractory_steps: 0,
        learning_rate: 0.01,
        modulator_scale: 0.02,
        ..NeuralParams::default()
    }
}

fn pathway_edge(snapshot: &ConnectomeSnapshot, pre: u32, post: u32) -> Result<usize> {
    snapshot
        .pre_indices
        .iter()
        .zip(snapshot.edge_posts.iter())
        .position(|(&candidate_pre, &candidate_post)| candidate_pre == pre && candidate_post == post)
        .with_context(|| format!("missing smoke edge {pre}->{post}"))
}

fn run_cpu_individual(cycles: usize) -> Result<(f32, f32, f32, f32)> {
    let snapshot = smoke_snapshot()?;
    let pathway_a_edge = pathway_edge(&snapshot, 0, 2)?;
    let pathway_b_edge = pathway_edge(&snapshot, 1, 3)?;
    let mut runtime = CpuRuntime::new(snapshot, smoke_params());
    let pathway_a_before = runtime.weights()[pathway_a_edge];
    let pathway_b_before = runtime.weights()[pathway_b_edge];

    for _ in 0..cycles {
        runtime.step(
            &[
                Stimulus { neuron: 0, current: 2.0 },
                Stimulus { neuron: 6, current: 2.0 },
            ],
            true,
        )?;
        runtime.step(&[], true)?;
        runtime.step(
            &[
                Stimulus { neuron: 1, current: 2.0 },
                Stimulus { neuron: 7, current: 2.0 },
            ],
            true,
        )?;
        runtime.step(&[], true)?;
    }

    Ok((
        pathway_a_before,
        runtime.weights()[pathway_a_edge],
        pathway_b_before,
        runtime.weights()[pathway_b_edge],
    ))
}

fn cpu_kernel_smoke(cycles: usize, flies: usize) -> Result<()> {
    if flies == 0 {
        bail!("flies must be greater than zero");
    }
    let started = Instant::now();
    let results: Vec<Result<(f32, f32, f32, f32)>> = (0..flies)
        .into_par_iter()
        .map(|_| run_cpu_individual(cycles))
        .collect();
    let results: Vec<_> = results.into_iter().collect::<Result<_>>()?;
    let first = results[0];
    let pathway_a_delta = results.iter().map(|x| x.1 - x.0).sum::<f32>() / flies as f32;
    let pathway_b_delta = results.iter().map(|x| x.3 - x.2).sum::<f32>() / flies as f32;

    println!("backend=cpu-rayon flies={flies} cycles={cycles}");
    println!(
        "pathway_a: {:.6} -> {:.6} (mean delta {pathway_a_delta:+.6})",
        first.0, first.1
    );
    println!(
        "pathway_b: {:.6} -> {:.6} (mean delta {pathway_b_delta:+.6})",
        first.2, first.3
    );
    println!("elapsed={:.3}s", started.elapsed().as_secs_f64());
    verify_smoke_change(pathway_a_delta, pathway_b_delta)
}

fn gpu_kernel_smoke(cycles: usize) -> Result<()> {
    use vf_neural::gpu::GpuRuntime;

    let snapshot = smoke_snapshot()?;
    let pathway_a_edge = pathway_edge(&snapshot, 0, 2)?;
    let pathway_b_edge = pathway_edge(&snapshot, 1, 3)?;
    let mut runtime = GpuRuntime::new(&snapshot, smoke_params())?;
    let before = runtime.readback()?;
    let started = Instant::now();

    for _ in 0..cycles {
        runtime.step(
            &[
                Stimulus { neuron: 0, current: 2.0 },
                Stimulus { neuron: 6, current: 2.0 },
            ],
            true,
        )?;
        runtime.step(&[], true)?;
        runtime.step(
            &[
                Stimulus { neuron: 1, current: 2.0 },
                Stimulus { neuron: 7, current: 2.0 },
            ],
            true,
        )?;
        runtime.step(&[], true)?;
    }
    let after = runtime.readback()?;
    let pathway_a_delta = after.weights[pathway_a_edge] - before.weights[pathway_a_edge];
    let pathway_b_delta = after.weights[pathway_b_edge] - before.weights[pathway_b_edge];

    println!("backend=gpu adapter={} cycles={cycles}", runtime.adapter_name());
    println!(
        "pathway_a: {:.6} -> {:.6} (delta {pathway_a_delta:+.6})",
        before.weights[pathway_a_edge], after.weights[pathway_a_edge]
    );
    println!(
        "pathway_b: {:.6} -> {:.6} (delta {pathway_b_delta:+.6})",
        before.weights[pathway_b_edge], after.weights[pathway_b_edge]
    );
    println!("elapsed={:.3}s", started.elapsed().as_secs_f64());
    verify_smoke_change(pathway_a_delta, pathway_b_delta)
}

fn verify_smoke_change(pathway_a_delta: f32, pathway_b_delta: f32) -> Result<()> {
    const EPSILON: f32 = 1e-7;
    if pathway_a_delta.abs() <= EPSILON {
        bail!("dopamine-gated pathway A produced no local synaptic change");
    }
    if pathway_b_delta.abs() <= EPSILON {
        bail!("dopamine-gated pathway B produced no local synaptic change");
    }
    println!("plasticity_smoke=PASS");
    Ok(())
}

fn snapshot_info(snapshot_path: PathBuf) -> Result<()> {
    let snapshot = ConnectomeSnapshot::load_dir(&snapshot_path)?;
    println!("dataset={}", snapshot.manifest.dataset);
    println!("neurons={}", snapshot.neuron_count());
    println!("edges={}", snapshot.edge_count());
    println!("snapshot={}", snapshot_path.display());
    Ok(())
}

fn snapshot_benchmark(snapshot_path: PathBuf, backend: BackendArg, steps: usize) -> Result<()> {
    if steps == 0 {
        bail!("steps must be greater than zero");
    }
    let snapshot = ConnectomeSnapshot::load_dir(&snapshot_path)?;
    println!(
        "dataset={} neurons={} edges={}",
        snapshot.manifest.dataset,
        snapshot.neuron_count(),
        snapshot.edge_count()
    );

    match backend {
        BackendArg::Cpu => benchmark_cpu(snapshot, steps),
        BackendArg::Gpu => benchmark_gpu(snapshot, steps),
        BackendArg::Auto => {
            let gpu_snapshot = snapshot.clone();
            match benchmark_gpu(gpu_snapshot, steps) {
                Ok(()) => Ok(()),
                Err(error) => {
                    eprintln!("GPU benchmark unavailable ({error:#}); falling back to CPU.");
                    benchmark_cpu(snapshot, steps)
                }
            }
        }
    }
}

fn benchmark_cpu(snapshot: ConnectomeSnapshot, steps: usize) -> Result<()> {
    let mut runtime = CpuRuntime::new(snapshot, NeuralParams::default());
    let started = Instant::now();
    for _ in 0..steps {
        runtime.step(&[], false)?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "backend=cpu-rayon steps={steps} elapsed={elapsed:.3}s steps_per_second={:.3}",
        steps as f64 / elapsed
    );
    Ok(())
}

fn benchmark_gpu(snapshot: ConnectomeSnapshot, steps: usize) -> Result<()> {
    use vf_neural::gpu::GpuRuntime;

    let mut runtime = GpuRuntime::new(&snapshot, NeuralParams::default())?;
    let started = Instant::now();
    for _ in 0..steps {
        runtime.step(&[], false)?;
    }
    // Force completion so the reported wall-clock time includes queued GPU work.
    let _ = runtime.readback()?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "backend=gpu adapter={} steps={steps} elapsed={elapsed:.3}s steps_per_second={:.3}",
        runtime.adapter_name(),
        steps as f64 / elapsed
    );
    Ok(())
}
