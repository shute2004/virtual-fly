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
// while the production-facing population module uses the outgoing propagation
// frontier. Debug wrappers expose full slot state only for golden tests;
// training does not call those readbacks.
#[cfg(feature = "gpu")]
#[path = "gpu_population_reference_debug.rs"]
pub mod gpu_population_reference;

#[cfg(feature = "gpu")]
#[path = "gpu_population_frontier_debug.rs"]
pub mod gpu_population;

// Diagnostic-only signed activity readback used by the real-Flyppy frontier
// profiler. It includes the same numerical runtime and adds no production-path
// behavior unless explicitly selected by the profiling bridge.
#[cfg(feature = "gpu")]
#[path = "gpu_population_frontier_profiled.rs"]
pub mod gpu_population_frontier_profiled;

#[cfg(all(test, feature = "gpu"))]
mod frontier_parity;

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
