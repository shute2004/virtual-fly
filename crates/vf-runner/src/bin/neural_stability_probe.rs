use std::{fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use vf_neural::{
    ConnectomeSnapshot, CpuRuntime, NeuralParams, Stimulus,
    model::{ACTIVITY_DEPOLARIZING, ACTIVITY_HYPERPOLARIZING},
};

#[derive(Debug, Parser)]
#[command(
    name = "neural-stability-probe",
    about = "Measure released MaleCNS pulse propagation and zero-input recurrent decay"
)]
struct Args {
    #[arg(long, default_value = "artifacts/malecns-v1.0")]
    snapshot: PathBuf,
    #[arg(long)]
    stimuli_json: PathBuf,
    #[arg(long, value_enum, default_value_t = Backend::Gpu)]
    backend: Backend,
    #[arg(long, default_value_t = 24)]
    zero_input_steps: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "0.00025,0.0005,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.01,0.012,0.016,0.02"
    )]
    synapse_scales: Vec<f32>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}

#[derive(Debug, Deserialize)]
struct BodyStimulus {
    body_id: u64,
    current: f32,
}

#[derive(Debug, Serialize)]
struct EventCount {
    step: usize,
    external_input: bool,
    positive: usize,
    hyperpolarizing: usize,
    total: usize,
    downstream_positive: usize,
    downstream_hyperpolarizing: usize,
    downstream_total: usize,
}

#[derive(Debug, Serialize)]
struct ScaleResult {
    synapse_scale: f32,
    events: Vec<EventCount>,
    initial_direct_events: usize,
    propagated_after_input: bool,
    peak_downstream_events: usize,
    final_total_events: usize,
    peak_total_events: usize,
    zero_tail: bool,
}

#[derive(Debug, Serialize)]
struct ProbeResult {
    neuron_count: usize,
    edge_count: usize,
    stimulated_neurons: usize,
    zero_input_steps: usize,
    results: Vec<ScaleResult>,
}

enum Runtime {
    Cpu(CpuRuntime),
    Gpu(vf_neural::gpu::GpuRuntime),
}

impl Runtime {
    fn step(&mut self, stimuli: &[Stimulus]) -> Result<()> {
        match self {
            Runtime::Cpu(runtime) => {
                runtime.step(stimuli, false)?;
                Ok(())
            }
            Runtime::Gpu(runtime) => runtime.step(stimuli, false),
        }
    }

    fn spikes(&self) -> Result<Vec<u32>> {
        match self {
            Runtime::Cpu(runtime) => {
                Ok(runtime.spikes().iter().map(|&event| event as u32).collect())
            }
            Runtime::Gpu(runtime) => runtime.read_spikes(),
        }
    }
}

fn event_count(
    step: usize,
    external_input: bool,
    spikes: &[u32],
    stimulated: &[bool],
) -> EventCount {
    let mut positive = 0usize;
    let mut hyperpolarizing = 0usize;
    let mut downstream_positive = 0usize;
    let mut downstream_hyperpolarizing = 0usize;

    for (index, &event) in spikes.iter().enumerate() {
        if event == ACTIVITY_DEPOLARIZING as u32 {
            positive += 1;
            if !stimulated[index] {
                downstream_positive += 1;
            }
        } else if event == ACTIVITY_HYPERPOLARIZING as u32 {
            hyperpolarizing += 1;
            if !stimulated[index] {
                downstream_hyperpolarizing += 1;
            }
        }
    }

    EventCount {
        step,
        external_input,
        positive,
        hyperpolarizing,
        total: positive + hyperpolarizing,
        downstream_positive,
        downstream_hyperpolarizing,
        downstream_total: downstream_positive + downstream_hyperpolarizing,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.zero_input_steps < 4 {
        bail!("zero-input-steps must be >= 4");
    }
    if args.synapse_scales.is_empty() {
        bail!("at least one synapse scale is required");
    }
    for &scale in &args.synapse_scales {
        if !scale.is_finite() || scale <= 0.0 {
            bail!("synapse scales must be finite and positive");
        }
    }

    let snapshot = ConnectomeSnapshot::load_dir(&args.snapshot)
        .with_context(|| format!("load MaleCNS snapshot from {}", args.snapshot.display()))?;
    let body_stimuli: Vec<BodyStimulus> = serde_json::from_slice(
        &fs::read(&args.stimuli_json)
            .with_context(|| format!("read {}", args.stimuli_json.display()))?,
    )
    .with_context(|| format!("parse {}", args.stimuli_json.display()))?;
    if body_stimuli.is_empty() {
        bail!("stimuli JSON contains no neurons");
    }

    let mut stimuli = Vec::with_capacity(body_stimuli.len());
    let mut stimulated_mask = vec![false; snapshot.neuron_count()];
    for row in body_stimuli {
        if !row.current.is_finite() || row.current == 0.0 {
            bail!("stimulus for body {} is non-finite or zero", row.body_id);
        }
        let neuron = snapshot
            .index_of_body_id(row.body_id)
            .with_context(|| format!("unknown body ID {}", row.body_id))?;
        if stimulated_mask[neuron] {
            bail!("body ID {} is stimulated more than once", row.body_id);
        }
        stimulated_mask[neuron] = true;
        stimuli.push(Stimulus {
            neuron,
            current: row.current,
        });
    }

    let mut results = Vec::with_capacity(args.synapse_scales.len());
    for &scale in &args.synapse_scales {
        let mut params = NeuralParams::default();
        params.synapse_scale = scale;
        let mut runtime = match args.backend {
            Backend::Cpu => Runtime::Cpu(CpuRuntime::new(snapshot.clone(), params)),
            Backend::Gpu => Runtime::Gpu(vf_neural::gpu::GpuRuntime::new(&snapshot, params)?),
        };

        let mut events = Vec::with_capacity(args.zero_input_steps + 1);
        runtime.step(&stimuli)?;
        events.push(event_count(0, true, &runtime.spikes()?, &stimulated_mask));
        for step in 1..=args.zero_input_steps {
            runtime.step(&[])?;
            events.push(event_count(
                step,
                false,
                &runtime.spikes()?,
                &stimulated_mask,
            ));
        }

        // Propagation means activity in a neuron that was NOT directly stimulated.
        // This avoids mistaking recurrent activity of an R1-R6 cell itself for
        // traversal of a released MaleCNS edge. Six zero-input steps are enough to
        // cover the same propagation horizon used by the retinal smoke test.
        let propagation_window = args.zero_input_steps.min(6);
        let propagated_after_input = events
            .iter()
            .skip(1)
            .take(propagation_window)
            .any(|sample| sample.downstream_total > 0);
        let initial_direct_events = events.first().map_or(0, |sample| sample.total);
        let peak_downstream_events = events
            .iter()
            .skip(1)
            .map(|sample| sample.downstream_total)
            .max()
            .unwrap_or(0);
        let final_total_events = events.last().map_or(0, |sample| sample.total);
        let peak_total_events = events.iter().map(|sample| sample.total).max().unwrap_or(0);
        let zero_tail = events.iter().rev().take(4).all(|sample| sample.total == 0);
        results.push(ScaleResult {
            synapse_scale: scale,
            events,
            initial_direct_events,
            propagated_after_input,
            peak_downstream_events,
            final_total_events,
            peak_total_events,
            zero_tail,
        });
    }

    let output = ProbeResult {
        neuron_count: snapshot.neuron_count(),
        edge_count: snapshot.edge_count(),
        stimulated_neurons: stimuli.len(),
        zero_input_steps: args.zero_input_steps,
        results,
    };
    serde_json::to_writer(std::io::stdout().lock(), &output)?;
    println!();
    Ok(())
}
