mod cpu;
pub mod model;
mod plastic_graph;
mod snapshot;
mod state;
mod transaction;

#[cfg(feature = "gpu")]
#[path = "gpu30.rs"]
pub mod gpu;

// Population training now keeps slot-local mutable synapse state only for
// PlasticFastGraph edges.  The profiled wrapper adds read-only transaction
// diagnostics without changing the numerical hot path.
#[cfg(feature = "gpu")]
#[path = "gpu_population_profiled.rs"]
pub mod gpu_population;

pub use cpu::{CpuRuntime, StepSummary};
pub use model::{
    ACTIVITY_DEPOLARIZING, ACTIVITY_HYPERPOLARIZING, ACTIVITY_SILENT, BackendKind,
    NeuralParams, Stimulus, nt,
};
pub use plastic_graph::PlasticFastGraph;
pub use snapshot::{ConnectomeSnapshot, EdgeInput, SnapshotManifest};
pub use state::NeuralState;
pub use transaction::{PlasticityTransaction, PlasticityTransform};
