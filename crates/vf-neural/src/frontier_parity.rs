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

fn snapshot() -> ConnectomeSnapshot {
    ConnectomeSnapshot::from_edges(
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
    .unwrap()
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

fn assert_slot_eq(
    label: &str,
    reference: &ReferenceRuntime,
    frontier: &FrontierRuntime,
    slot: usize,
) {
    let a = reference.debug_slot_state(slot).unwrap();
    let b = frontier.debug_slot_state(slot).unwrap();
    assert_f32_bits_eq(&format!("{label}.membrane"), &a.membrane, &b.membrane);
    assert_f32_bits_eq(&format!("{label}.trace"), &a.trace, &b.trace);
    assert_f32_bits_eq(&format!("{label}.modulation"), &a.modulation, &b.modulation);
    assert_eq!(a.refractory, b.refractory, "{label}.refractory mismatch");
    assert_eq!(a.spikes, b.spikes, "{label}.spikes mismatch");
    assert_f32_bits_eq(&format!("{label}.plastic_weight"), &a.plastic_weight, &b.plastic_weight);
    assert_f32_bits_eq(&format!("{label}.eligibility"), &a.eligibility, &b.eligibility);
    assert_f32_bits_eq(
        &format!("{label}.transaction_shift"),
        &a.transaction_shift,
        &b.transaction_shift,
    );
    assert_f32_bits_eq(
        &format!("{label}.transaction_lo"),
        &a.transaction_lo,
        &b.transaction_lo,
    );
    assert_f32_bits_eq(
        &format!("{label}.transaction_hi"),
        &a.transaction_hi,
        &b.transaction_hi,
    );
}

#[test]
fn outgoing_frontier_is_bitwise_equal_to_dense_incoming_reference() {
    let snapshot = snapshot();
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
        assert_slot_eq(&format!("step{step}"), &reference, &frontier, 0);
    }

    reference.commit_and_restart_slot(0).unwrap();
    frontier.commit_and_restart_slot(0).unwrap();
    let reference_weights = reference.global_weights().unwrap();
    let frontier_weights = frontier.global_weights().unwrap();
    assert_f32_bits_eq("committed_global_weight", &reference_weights, &frontier_weights);
}

#[test]
fn live_plasticity_bitmap_preserves_idle_slot_across_async_teaching_steps() {
    let snapshot = snapshot();
    let mut reference = ReferenceRuntime::new(&snapshot, params(), 2).unwrap();
    let mut frontier = FrontierRuntime::new(&snapshot, params(), 2).unwrap();
    let read_indices = (0..2 * snapshot.neuron_count()).collect::<Vec<_>>();

    // Establish non-zero trace/eligibility in both slots.
    let both_active = [true, true];
    let first = [
        vec![
            Stimulus { neuron: 0, current: 1.2 },
            Stimulus { neuron: 2, current: 1.2 },
        ],
        vec![
            Stimulus { neuron: 0, current: 1.2 },
            Stimulus { neuron: 2, current: 1.2 },
        ],
    ];
    let a = reference
        .step_batch_with_read(&first, &both_active, &read_indices, true)
        .unwrap();
    let b = frontier
        .step_batch_with_read(&first, &both_active, &read_indices, true)
        .unwrap();
    assert_eq!(a, b);

    let second = [vec![Stimulus { neuron: 5, current: 1.2 }], vec![Stimulus { neuron: 5, current: 1.2 }]];
    reference
        .step_batch_with_read(&second, &both_active, &read_indices, true)
        .unwrap();
    frontier
        .step_batch_with_read(&second, &both_active, &read_indices, true)
        .unwrap();
    assert_slot_eq("before_async.slot0", &reference, &frontier, 0);
    assert_slot_eq("before_async.slot1", &reference, &frontier, 1);

    // Slot 0 receives extra plasticity-enabled teaching steps while slot 1 is
    // idle. A process-global bitmap phase must not discard slot 1's live set.
    let only_zero = [true, false];
    let teaching = [vec![Stimulus { neuron: 2, current: 1.2 }], vec![]];
    for step in 0..3 {
        reference
            .step_batch_with_read(&teaching, &only_zero, &read_indices, true)
            .unwrap();
        frontier
            .step_batch_with_read(&teaching, &only_zero, &read_indices, true)
            .unwrap();
        assert_slot_eq(&format!("async{step}.slot0"), &reference, &frontier, 0);
        assert_slot_eq(&format!("async{step}.slot1"), &reference, &frontier, 1);
    }

    // Resume both slots and verify the previously idle slot evolves exactly as
    // the dense reference from its preserved eligibility state.
    let resume = [vec![], vec![Stimulus { neuron: 2, current: 1.2 }]];
    let a = reference
        .step_batch_with_read(&resume, &both_active, &read_indices, true)
        .unwrap();
    let b = frontier
        .step_batch_with_read(&resume, &both_active, &read_indices, true)
        .unwrap();
    assert_eq!(a, b);
    assert_slot_eq("resume.slot0", &reference, &frontier, 0);
    assert_slot_eq("resume.slot1", &reference, &frontier, 1);
}
