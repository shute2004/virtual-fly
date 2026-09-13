mod cpu;
mod model;
mod snapshot;

#[cfg(feature = "gpu")]
#[path = "gpu30.rs"]
pub mod gpu;

pub use cpu::{CpuRuntime, StepSummary};
pub use model::{BackendKind, NeuralParams, Stimulus, nt};
pub use snapshot::{ConnectomeSnapshot, EdgeInput, SnapshotManifest};
