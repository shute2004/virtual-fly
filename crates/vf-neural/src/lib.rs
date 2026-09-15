mod cpu;
pub mod model;
mod plastic_graph;
mod propagation_graph;
mod snapshot;
mod state;
mod transaction;

#[cfg(feature = "gpu")]
#[path = "gpu30.rs"]
pub mod gpu;

// Keep the P-backed dense-incoming runtime available as a numerical reference
// while the production-facing population module experiments with an outgoing
// propagation frontier. This lets parity probes run both implementations from
// the same build.
#[cfg(feature = "gpu")]
#[path = "gpu_population_sparse.rs"]
pub mod gpu_population_reference;

#[cfg(feature = "gpu")]
#[path = "gpu_population_frontier.rs"]
pub mod gpu_population;

pub use cpu::{CpuRuntime, StepSummary};
pub use model::{
    ACTIVITY_DEPOLARIZING, ACTIVITY_HYPERPOLARIZING, ACTIVITY_SILENT, BackendKind,
    NeuralParams, Stimulus, nt,
};
pub use plastic_graph::PlasticFastGraph;
pub use propagation_graph::OutgoingPropagationGraph;
pub use snapshot::{ConnectomeSnapshot, EdgeInput, SnapshotManifest};
pub use state::NeuralState;
pub use transaction::{PlasticityTransaction, PlasticityTransform};
