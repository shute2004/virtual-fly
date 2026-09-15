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
    slot_count: u32,
    pre_start: u32,
    post_start: u32,
    count_start: u32,
    refractory_steps: u32,
    _pad_u0: u32,
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> topology: array<u32>;
@group(0) @binding(1) var<storage, read> metadata: array<u32>;
@group(0) @binding(2) var<storage, read_write> neurons: array<NeuronState>;
@group(0) @binding(3) var<storage, read> spikes_prev: array<u32>;
@group(0) @binding(4) var<storage, read_write> spikes_next: array<u32>;
@group(0) @binding(5) var<storage, read> external_current: array<f32>;
@group(0) @binding(6) var<storage, read_write> synapses: array<SynapseState>;
@group(0) @binding(7) var<storage, read_write> transactions: array<TransactionState>;
@group(0) @binding(8) var<storage, read_write> global_weights: array<f32>;
@group(0) @binding(9) var<storage, read> slot_active: array<u32>;
@group(0) @binding(10) var<uniform> params: Params;
@group(0) @binding(11) var<uniform> control: Control;

const WORKGROUP_SIZE: u32 = 256u;
const NT_DOPAMINE: u32 = 4u;
const ACTIVITY_SILENT: u32 = 0u;
const ACTIVITY_DEPOLARIZING: u32 = 1u;
const ACTIVITY_HYPERPOLARIZING: u32 = 2u;

fn linear_invocation_index(gid: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return gid.x + gid.y * num_workgroups.x * WORKGROUP_SIZE;
}

fn nt_code(neuron: u32) -> u32 {
    return metadata[neuron] & 0xffu;
}

fn is_dopamine(neuron: u32) -> bool {
    return nt_code(neuron) == NT_DOPAMINE;
}

fn activity_sign(event: u32) -> f32 {
    switch event {
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

@compute @workgroup_size(256)
fn neuron_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flat = linear_invocation_index(gid, num_workgroups);
    let total = params.slot_count * params.neuron_count;
    if flat >= total { return; }
    let slot = flat / params.neuron_count;
    if slot_active[slot] == 0u { return; }
    let post = flat % params.neuron_count;
    let neuron_base = slot * params.neuron_count;
    let edge_base = slot * params.edge_count;

    var fast_current = external_current[flat];
    var dopaminergic_input = 0.0;
    let begin = topology[post];
    let end = topology[post + 1u];

    var edge = begin;
    loop {
        if edge >= end { break; }
        let pre = topology[params.pre_start + edge];
        let event_sign = activity_sign(spikes_prev[neuron_base + pre]);
        if event_sign != 0.0 {
            if is_dopamine(pre) {
                if event_sign > 0.0 {
                    let count = f32(topology[params.count_start + edge]);
                    dopaminergic_input += count * params.modulator_scale;
                }
            } else {
                fast_current += fast_sign(pre) * synapses[edge_base + edge].weight * event_sign;
            }
        }
        edge += 1u;
    }

    let old = neurons[flat];
    var next = old;
    next.modulation = old.modulation * params.modulator_decay + dopaminergic_input;

    if old.refractory > 0u {
        next.membrane = params.reset;
        next.refractory = old.refractory - 1u;
        spikes_next[flat] = ACTIVITY_SILENT;
    } else {
        let membrane = old.membrane * params.membrane_decay + fast_current;
        if membrane >= params.threshold {
            next.membrane = params.reset;
            next.refractory = params.refractory_steps;
            spikes_next[flat] = ACTIVITY_DEPOLARIZING;
        } else if membrane <= -params.threshold {
            next.membrane = params.reset;
            next.refractory = params.refractory_steps;
            spikes_next[flat] = ACTIVITY_HYPERPOLARIZING;
        } else {
            next.membrane = membrane;
            next.refractory = 0u;
            spikes_next[flat] = ACTIVITY_SILENT;
        }
    }
    neurons[flat] = next;
}

@compute @workgroup_size(256)
fn plasticity_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flat = linear_invocation_index(gid, num_workgroups);
    let total = params.slot_count * params.edge_count;
    if flat >= total { return; }
    let slot = flat / params.edge_count;
    if slot_active[slot] == 0u { return; }
    let edge = flat % params.edge_count;
    let edge_base = slot * params.edge_count;
    let neuron_base = slot * params.neuron_count;

    let pre = topology[params.pre_start + edge];
    let post = topology[params.post_start + edge];
    if fast_sign(pre) == 0.0 { return; }

    var syn = synapses[edge_base + edge];
    let local = neurons[neuron_base + pre].trace * activity_sign(spikes_next[neuron_base + post])
        - neurons[neuron_base + post].trace * activity_sign(spikes_prev[neuron_base + pre]);
    syn.eligibility = syn.eligibility * params.eligibility_decay + local;
    let delta = params.learning_rate * neurons[neuron_base + post].modulation * syn.eligibility;
    syn.weight = clamp(syn.weight + delta, 0.0, params.weight_max);
    synapses[edge_base + edge] = syn;

    var txn = transactions[edge_base + edge];
    txn.shift = txn.shift + delta;
    txn.lo = clamp(txn.lo + delta, 0.0, params.weight_max);
    txn.hi = clamp(txn.hi + delta, 0.0, params.weight_max);
    transactions[edge_base + edge] = txn;
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
    if slot_active[slot] == 0u { return; }
    var state = neurons[flat];
    state.trace = state.trace * params.trace_decay + activity_sign(spikes_next[flat]);
    neurons[flat] = state;
}

@compute @workgroup_size(256)
fn commit_slot(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let edge = linear_invocation_index(gid, num_workgroups);
    if edge >= params.edge_count { return; }
    let slot_edge = control.slot * params.edge_count + edge;
    let txn = transactions[slot_edge];
    global_weights[edge] = clamp(global_weights[edge] + txn.shift, txn.lo, txn.hi);
}

@compute @workgroup_size(256)
fn restart_slot_synapses(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let edge = linear_invocation_index(gid, num_workgroups);
    if edge >= params.edge_count { return; }
    let slot_edge = control.slot * params.edge_count + edge;
    synapses[slot_edge].weight = global_weights[edge];
    synapses[slot_edge].eligibility = 0.0;
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
    spikes_next[flat] = ACTIVITY_SILENT;
}
