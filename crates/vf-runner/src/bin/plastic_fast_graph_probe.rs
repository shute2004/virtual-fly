use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use vf_neural::{ConnectomeSnapshot, PlasticFastGraph};

#[derive(Debug, Parser)]
#[command(
    name = "plastic-fast-graph-probe",
    about = "Compile the current MaleCNS PlasticFastGraph in Rust and report structural counts"
)]
struct Args {
    #[arg(long, default_value = "artifacts/malecns-v1.0")]
    snapshot: PathBuf,
}

#[derive(Debug, Serialize)]
struct ProbeResult {
    neuron_count: usize,
    edge_count: usize,
    dopamine_capable_post_count: usize,
    plastic_edge_count: usize,
    plastic_edge_fraction: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let snapshot = ConnectomeSnapshot::load_dir(&args.snapshot)
        .with_context(|| format!("load MaleCNS snapshot from {}", args.snapshot.display()))?;
    let graph = PlasticFastGraph::compile(&snapshot).context("compile PlasticFastGraph")?;

    let plastic_edge_count = graph.edge_count();
    let result = ProbeResult {
        neuron_count: snapshot.neuron_count(),
        edge_count: snapshot.edge_count(),
        dopamine_capable_post_count: graph.dopamine_capable_posts.len(),
        plastic_edge_count,
        plastic_edge_fraction: if snapshot.edge_count() == 0 {
            0.0
        } else {
            plastic_edge_count as f64 / snapshot.edge_count() as f64
        },
    };

    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
