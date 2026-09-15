mod cpu;
pub mod model;
mod snapshot;
mod state;
mod transaction;

#[cfg(feature = "gpu")]
#[path = "gpu30.rs"]
pub mod gpu;

// The population runtime intentionally owns several GPU buffers only to keep
// resources referenced by bind groups alive for the lifetime of the runtime.
// Rust therefore sees those lifetime-anchor fields as unread even though they
// are required by the WGPU resource graph.
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub mod gpu_population;

pub use cpu::{CpuRuntime, StepSummary};
pub use model::{
    ACTIVITY_DEPOLARIZING, ACTIVITY_HYPERPOLARIZING, ACTIVITY_SILENT, BackendKind,
    NeuralParams, Stimulus, nt,
};
pub use snapshot::{ConnectomeSnapshot, EdgeInput, SnapshotManifest};
pub use state::NeuralState;
pub use transaction::{PlasticityTransaction, PlasticityTransform};
