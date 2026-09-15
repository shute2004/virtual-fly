mod cpu;
pub mod model;
mod plastic_graph;
mod snapshot;
mod state;
mod transaction;

#[cfg(feature = "gpu")]
#[path = "gpu30.rs"]
pub mod gpu;

// Population training now keeps slot-local mutable synapse and transaction
// state only for PlasticFastGraph edges. The public module name remains stable
// so the bridge/trainer protocol does not change while the runtime evolves.
#[cfg(feature = "gpu")]
#[path = "gpu_population_sparse.rs"]
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
