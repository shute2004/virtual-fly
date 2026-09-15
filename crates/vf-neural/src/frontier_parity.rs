use crate::{
    EdgeInput, NeuralParams, Stimulus,
    gpu_population::GpuPopulationRuntime as FrontierRuntime,
    gpu_population_reference::GpuPopulationRuntime as ReferenceRuntime,
    model::nt,
    snapshot::ConnectomeSnapshot,
};

fn params() -> NeuralParams {
    NeuralParams {
        membrane_decay: 0.95,
        threshold: 1.0,
        reset: 0.0,
        refractory_steps: 2,
        trace_decay: 0.95,
        eligibility_decay: 0.995,
        learning_rate: 0.00001,
        weight_max: 1_000.0,
        modulator_decay: 0.98,
        synapse_scale: 0.02,
        modulator_scale: 0.00005,
    }
}

fn assert_f32_bits_eq(label: &str, left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "{label} length mismatch");
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}[{index}] mismatch: {a:?} vs {b:?}",
        );
    }
}

#[test]
fn outgoing_frontier_is_bitwise_equal_to_dense_incoming_reference() {
    let snapshot = ConnectomeSnapshot::from_edges(
        7,
        &[
            EdgeInput { pre: 0, post: 5, synapse_count: 50 },
            EdgeInput { pre: 1, post: 5, synapse_count: 25 },
            EdgeInput { pre: 2, post: 5, synapse_count: 10 },
            EdgeInput { pre: 6, post: 5, synapse_count: 20 },
            EdgeInput { pre: 0, post: 6, synapse_count: 10 },
            EdgeInput { pre: 2, post: 6, synapse_count: 5 },
            EdgeInput { pre: 3, post: 6, synapse_count: 40 },
            EdgeInput { pre: 4, post: 6, synapse_count: 100 },
            EdgeInput { pre: 5, post: 6, synapse_count: 30 },
        ],
        vec![
            nt::ACETYLCHOLINE,
            nt::GABA,
            nt::DOPAMINE,
            nt::HISTAMINE,
            nt::OCTOPAMINE,
            nt::ACETYLCHOLINE,
            nt::ACETYLCHOLINE,
        ],
    )
    .unwrap();

    let mut reference = ReferenceRuntime::new(&snapshot, params(), 1).unwrap();
    let mut frontier = FrontierRuntime::new(&snapshot, params(), 1).unwrap();
    let read_indices = (0..snapshot.neuron_count()).collect::<Vec<_>>();
    let active = [true];
    let sequence = [
        vec![
            Stimulus { neuron: 0, current: 1.2 },
            Stimulus { neuron: 1, current: -1.2 },
            Stimulus { neuron: 2, current: 1.2 },
        ],
        vec![],
        vec![Stimulus { neuron: 3, current: 1.2 }],
        vec![],
        vec![Stimulus { neuron: 0, current: -1.2 }],
        vec![],
        vec![Stimulus { neuron: 5, current: 1.2 }],
        vec![],
        vec![Stimulus { neuron: 2, current: 1.2 }],
        vec![],
        vec![],
        vec![],
    ];

    for (step, stimuli) in sequence.iter().enumerate() {
        let by_slot = [stimuli.clone()];
        let reference_events = reference
            .step_batch_with_read(&by_slot, &active, &read_indices, true)
            .unwrap();
        let frontier_events = frontier
            .step_batch_with_read(&by_slot, &active, &read_indices, true)
            .unwrap();
        assert_eq!(reference_events, frontier_events, "spike mismatch at step {step}");

        let a = reference.debug_slot_state(0).unwrap();
        let b = frontier.debug_slot_state(0).unwrap();
        assert_f32_bits_eq("membrane", &a.membrane, &b.membrane);
        assert_f32_bits_eq("trace", &a.trace, &b.trace);
        assert_f32_bits_eq("modulation", &a.modulation, &b.modulation);
        assert_eq!(a.refractory, b.refractory, "refractory mismatch at step {step}");
        assert_eq!(a.spikes, b.spikes, "debug spike mismatch at step {step}");
        assert_f32_bits_eq("plastic_weight", &a.plastic_weight, &b.plastic_weight);
        assert_f32_bits_eq("eligibility", &a.eligibility, &b.eligibility);
        assert_f32_bits_eq("transaction_shift", &a.transaction_shift, &b.transaction_shift);
        assert_f32_bits_eq("transaction_lo", &a.transaction_lo, &b.transaction_lo);
        assert_f32_bits_eq("transaction_hi", &a.transaction_hi, &b.transaction_hi);
    }

    reference.commit_and_restart_slot(0).unwrap();
    frontier.commit_and_restart_slot(0).unwrap();
    let reference_weights = reference.global_weights().unwrap();
    let frontier_weights = frontier.global_weights().unwrap();
    assert_f32_bits_eq("committed_global_weight", &reference_weights, &frontier_weights);
}
