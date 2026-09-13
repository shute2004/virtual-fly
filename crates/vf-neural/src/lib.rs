mod cpu;
mod model;
mod snapshot;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::{CpuRuntime, StepSummary};
pub use model::{BackendKind, NeuralParams, Stimulus};
pub use snapshot::{ConnectomeSnapshot, EdgeInput, SnapshotManifest};
