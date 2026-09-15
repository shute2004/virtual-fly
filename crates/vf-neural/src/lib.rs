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

// Keep the P-backed dense-incoming runtime available as the numerical golden
// reference. The production-facing population runtime now combines the
// outgoing propagation frontier with an exact eager live plasticity bitmap.
#[cfg(feature = "gpu")]
#[path = "gpu_population_reference_debug.rs"]
pub mod gpu_population_reference;

// gpu_population_live wraps the frontier runtime and deliberately retains a few
// dense-P entry points/pipelines for parity/debug reuse. They are intentionally
// unreachable from the production hot path, so suppress dead-code warnings for
// this module only rather than deleting the numerical reference machinery.
#[cfg(feature = "gpu")]
#[allow(dead_code)]
#[path = "gpu_population_live.rs"]
pub mod gpu_population;

// Diagnostic-only signed activity/readback runtime. It deliberately keeps the
// dense-P plasticity path so profilers can inspect full internal state without
// adding readback machinery to the production hot path.
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
