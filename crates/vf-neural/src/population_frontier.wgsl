struct NeuronState {
    membrane: f32,
    trace: f32,
    modulation: f32,
    refractory: u32,
}

struct SynapseState {
    weight: f32,
    eligibility: f32,
}

struct TransactionState {
    shift: f32,
    lo: f32,
    hi: f32,
}

struct Params {
    neuron_count: u32,
    edge_count: u32,
    plastic_edge_count: u32,
    slot_count: u32,
    pre_start: u32,
    count_start: u32,
    meta_start: u32,
    plastic_row_start: u32,
    plastic_edge_start: u32,
    plastic_post_start: u32,
    outgoing_row_start: u32,
    outgoing_post_start: u32,
    refractory_steps: u32,
    _pad_u0: u32,
    _pad_u1: u32,
    _pad_u2: u32,
    membrane_decay: f32,
    threshold: f32,
    reset: f32,
    trace_decay: f32,
    eligibility_decay: f32,
    learning_rate: f32,
    weight_max: f32,
    modulator_decay: f32,
    modulator_scale: f32,
    _pad_f0: f32,
    _pad_f1: f32,
    _pad_f2: f32,
}

struct Control {
    slot: u32,
    active_mask: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> topology: array<u32>;
@group(0) @binding(1) var<storage, read_write> neurons: array<NeuronState>;
@group(0) @binding(2) var<storage, read> spikes_prev: array<u32>;
// The high bit is temporary candidate-post scratch between the outgoing
// frontier dispatch and neuron_step. neuron_step always overwrites the word
// with the canonical 0/1/2 event, so no extra per-slot candidate buffer exists.
@group(0) @binding(3) var<storage, read_write> spikes_next: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> external_current: array<f32>;
@group(0) @binding(5) var<storage, read_write> plastic_synapses: array<SynapseState>;
@group(0) @binding(6) var<storage, read_write> transactions: array<TransactionState>;
@group(0) @binding(7) var<storage, read_write> global_weights: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;
@group(0) @binding(9) var<uniform> control: Control;

const WORKGROUP_SIZE: u32 = 256u;
const NT_DOPAMINE: u32 = 4u;
const ACTIVITY_SILENT: u32 = 0u;
const ACTIVITY_DEPOLARIZING: u32 = 1u;
const ACTIVITY_HYPERPOLARIZING: u32 = 2u;
const CANDIDATE_POST_BIT: u32 = 0x80000000u;

fn linear_invocation_index(gid: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return gid.x + gid.y * num_workgroups.x * WORKGROUP_SIZE;
}

fn nt_code(neuron: u32) -> u32 {
    return topology[params.meta_start + neuron] & 0xffu;
}

fn is_dopamine(neuron: u32) -> bool {
    return (topology[params.meta_start + neuron] & 0x100u) != 0u;
}

fn slot_is_active(slot: u32) -> bool {
    return (control.active_mask & (1u << slot)) != 0u;
}

fn activity_sign(event: u32) -> f32 {
    switch event & 0x7fffffffu {
        case ACTIVITY_DEPOLARIZING: { return 1.0; }
        case ACTIVITY_HYPERPOLARIZING: { return -1.0; }
        default: { return 0.0; }
    }
}

fn fast_sign(neuron: u32) -> f32 {
    switch nt_code(neuron) {
        case 1u: { return 1.0; }
        case 2u, 3u, 7u: { return -1.0; }
        default: { return 0.0; }
    }
}

/// Phase-D frontier discovery. We still dispatch O(N) lightweight invocations,
/// but only active presynaptic neurons walk their outgoing adjacency. The
/// resulting candidate marker only decides whether the canonical incoming loop
/// must run; it does not perform floating-point current accumulation.
@compute @workgroup_size(256)
fn mark_candidate_posts(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flat = linear_invocation_index(gid, num_workgroups);
    let total = params.slot_count * params.neuron_count;
    if flat >= total { return; }
    let slot = flat / params.neuron_count;
    if !slot_is_active(slot) { return; }
    if activity_sign(spikes_prev[flat]) == 0.0 { return; }

    let pre = flat % params.neuron_count;
    let neuron_base = slot * params.neuron_count;
    let begin = topology[params.outgoing_row_start + pre];
    let end = topology[params.outgoing_row_start + pre + 1u];
    var index = begin;
    loop {
        if index >= end { break; }
        let post = topology[params.outgoing_post_start + index];
        atomicOr(&spikes_next[neuron_base + post], CANDIDATE_POST_BIT);
        index += 1u;
    }
}

@compute @workgroup_size(256)
fn neuron_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flat = linear_invocation_index(gid, num_workgroups);
    let total = params.slot_count * params.neuron_count;
    if flat >= total { return; }
    let slot = flat / params.neuron_count;
    if !slot_is_active(slot) { return; }
    let post = flat % params.neuron_count;
    let neuron_base = slot * params.neuron_count;
    let plastic_base = slot * params.plastic_edge_count;

    let old = neurons[flat];
    var fast_current = external_current[flat];
    var dopaminergic_input = 0.0;
    let candidate = (atomicLoad(&spikes_next[flat]) & CANDIDATE_POST_BIT) != 0u;

    // Only posts reached by at least one previous-step event execute the
    // expensive incoming scan. The loop itself remains byte-for-byte ordered by
    // the original incoming CSR, preserving floating-point accumulation order.
    if candidate {
        let begin = topology[post];
        let end = topology[post + 1u];
        var plastic_cursor = topology[params.plastic_row_start + post];
        let plastic_end = topology[params.plastic_row_start + post + 1u];

        var edge = begin;
        loop {
            if edge >= end { break; }
            let pre = topology[params.pre_start + edge];
            let event_sign = activity_sign(spikes_prev[neuron_base + pre]);
            if is_dopamine(pre) {
                if event_sign > 0.0 {
                    let count = f32(topology[params.count_start + edge]);
                    dopaminergic_input += count * params.modulator_scale;
                }
            } else {
                let sign = fast_sign(pre);
                if sign != 0.0 {
                    var weight = global_weights[edge];
                    if plastic_cursor < plastic_end {
                        weight = plastic_synapses[plastic_base + plastic_cursor].weight;
                        plastic_cursor += 1u;
                    }
                    if event_sign != 0.0 {
                        fast_current += sign * weight * event_sign;
                    }
                }
            }
            edge += 1u;
        }
    }

    var next = old;
    next.modulation = old.modulation * params.modulator_decay + dopaminergic_input;

    var event = ACTIVITY_SILENT;
    if old.refractory > 0u {
        next.membrane = params.reset;
        next.refractory = old.refractory - 1u;
    } else {
        let membrane = old.membrane * params.membrane_decay + fast_current;
        if membrane >= params.threshold {
            next.membrane = params.reset;
            next.refractory = params.refractory_steps;
            event = ACTIVITY_DEPOLARIZING;
        } else if membrane <= -params.threshold {
            next.membrane = params.reset;
            next.refractory = params.refractory_steps;
            event = ACTIVITY_HYPERPOLARIZING;
        } else {
            next.membrane = membrane;
            next.refractory = 0u;
        }
    }
    neurons[flat] = next;
    atomicStore(&spikes_next[flat], event);
}

@compute @workgroup_size(256)
fn plasticity_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flat = linear_invocation_index(gid, num_workgroups);
    let total = params.slot_count * params.plastic_edge_count;
    if flat >= total { return; }
    let slot = flat / params.plastic_edge_count;
    if !slot_is_active(slot) { return; }
    let plastic_edge = flat % params.plastic_edge_count;
    let plastic_base = slot * params.plastic_edge_count;
    let neuron_base = slot * params.neuron_count;

    let source_edge = topology[params.plastic_edge_start + plastic_edge];
    let post = topology[params.plastic_post_start + plastic_edge];
    let pre = topology[params.pre_start + source_edge];

    var syn = plastic_synapses[plastic_base + plastic_edge];
    let local = neurons[neuron_base + pre].trace * activity_sign(atomicLoad(&spikes_next[neuron_base + post]))
        - neurons[neuron_base + post].trace * activity_sign(spikes_prev[neuron_base + pre]);
    syn.eligibility = syn.eligibility * params.eligibility_decay + local;
    let delta = params.learning_rate * neurons[neuron_base + post].modulation * syn.eligibility;

    if delta != 0.0 {
        syn.weight = clamp(syn.weight + delta, 0.0, params.weight_max);
        var txn = transactions[plastic_base + plastic_edge];
        txn.shift = txn.shift + delta;
        txn.lo = clamp(txn.lo + delta, 0.0, params.weight_max);
        txn.hi = clamp(txn.hi + delta, 0.0, params.weight_max);
        transactions[plastic_base + plastic_edge] = txn;
    }
    plastic_synapses[plastic_base + plastic_edge] = syn;
}

@compute @workgroup_size(256)
fn trace_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flat = linear_invocation_index(gid, num_workgroups);
    let total = params.slot_count * params.neuron_count;
    if flat >= total { return; }
    let slot = flat / params.neuron_count;
    if !slot_is_active(slot) { return; }
    var state = neurons[flat];
    state.trace = state.trace * params.trace_decay + activity_sign(atomicLoad(&spikes_next[flat]));
    neurons[flat] = state;
}

@compute @workgroup_size(256)
fn commit_slot(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let plastic_edge = linear_invocation_index(gid, num_workgroups);
    if plastic_edge >= params.plastic_edge_count { return; }
    let slot_edge = control.slot * params.plastic_edge_count + plastic_edge;
    let source_edge = topology[params.plastic_edge_start + plastic_edge];
    let txn = transactions[slot_edge];
    global_weights[source_edge] = clamp(global_weights[source_edge] + txn.shift, txn.lo, txn.hi);
}

@compute @workgroup_size(256)
fn restart_slot_synapses(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let plastic_edge = linear_invocation_index(gid, num_workgroups);
    if plastic_edge >= params.plastic_edge_count { return; }
    let slot_edge = control.slot * params.plastic_edge_count + plastic_edge;
    let source_edge = topology[params.plastic_edge_start + plastic_edge];
    plastic_synapses[slot_edge].weight = global_weights[source_edge];
    plastic_synapses[slot_edge].eligibility = 0.0;
    transactions[slot_edge].shift = 0.0;
    transactions[slot_edge].lo = 0.0;
    transactions[slot_edge].hi = params.weight_max;
}

@compute @workgroup_size(256)
fn reset_slot_neurons(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let neuron = linear_invocation_index(gid, num_workgroups);
    if neuron >= params.neuron_count { return; }
    let flat = control.slot * params.neuron_count + neuron;
    neurons[flat].membrane = 0.0;
    neurons[flat].trace = 0.0;
    neurons[flat].modulation = 0.0;
    neurons[flat].refractory = 0u;
    atomicStore(&spikes_next[flat], ACTIVITY_SILENT);
}
