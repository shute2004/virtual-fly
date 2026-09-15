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

struct LiveParams {
    neuron_count: u32,
    plastic_edge_count: u32,
    slot_count: u32,
    bitmap_words: u32,
    pre_start: u32,
    plastic_row_start: u32,
    plastic_edge_start: u32,
    plastic_post_start: u32,
    plastic_out_edge_start: u32,
    _pad_u0: u32,
    _pad_u1: u32,
    _pad_u2: u32,
    eligibility_decay: f32,
    learning_rate: f32,
    weight_max: f32,
    _pad_f0: f32,
}

struct Control {
    slot: u32,
    active_mask: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> topology: array<u32>;
@group(0) @binding(1) var<storage, read> neurons: array<NeuronState>;
@group(0) @binding(2) var<storage, read> spikes_prev: array<u32>;
@group(0) @binding(3) var<storage, read> spikes_next: array<u32>;
@group(0) @binding(4) var<storage, read_write> plastic_synapses: array<SynapseState>;
@group(0) @binding(5) var<storage, read_write> transactions: array<TransactionState>;
@group(0) @binding(6) var<storage, read> plastic_outgoing: array<u32>;
@group(0) @binding(7) var<storage, read_write> live_current: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> live_next: array<atomic<u32>>;
@group(0) @binding(9) var<uniform> params: LiveParams;
@group(0) @binding(10) var<uniform> control: Control;

const WORKGROUP_SIZE: u32 = 256u;

fn linear_invocation_index(gid: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return gid.x + gid.y * num_workgroups.x * WORKGROUP_SIZE;
}

fn slot_is_active(slot: u32) -> bool {
    return (control.active_mask & (1u << slot)) != 0u;
}

fn activity_sign(event: u32) -> f32 {
    switch event & 0x7fffffffu {
        case 1u: { return 1.0; }
        case 2u: { return -1.0; }
        default: { return 0.0; }
    }
}

fn set_live_bit(slot: u32, plastic_edge: u32) {
    let word = plastic_edge >> 5u;
    let bit = plastic_edge & 31u;
    let flat_word = slot * params.bitmap_words + word;
    atomicOr(&live_current[flat_word], 1u << bit);
}

/// Add every plastic edge whose local three-factor term can be non-zero this
/// step. Existing non-zero eligibility edges are already present in
/// `live_current` from the previous step.
///
/// Two exact causes exist:
///   pre.trace * current_post_event != 0
///   post.trace * previous_pre_event != 0
/// The first is discovered from the plastic incoming rows of active current
/// posts; the second from a shared plastic-outgoing CSR keyed by active previous
/// presynaptic neurons.
@compute @workgroup_size(256)
fn discover_live_plasticity(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flat = linear_invocation_index(gid, num_workgroups);
    let total = params.slot_count * params.neuron_count;
    if flat >= total { return; }

    let slot = flat / params.neuron_count;
    if !slot_is_active(slot) { return; }
    let neuron = flat % params.neuron_count;
    let neuron_base = slot * params.neuron_count;

    let current_event = activity_sign(spikes_next[flat]);
    if current_event != 0.0 {
        let begin = topology[params.plastic_row_start + neuron];
        let end = topology[params.plastic_row_start + neuron + 1u];
        var plastic_edge = begin;
        loop {
            if plastic_edge >= end { break; }
            let source_edge = topology[params.plastic_edge_start + plastic_edge];
            let pre = topology[params.pre_start + source_edge];
            if neurons[neuron_base + pre].trace != 0.0 {
                set_live_bit(slot, plastic_edge);
            }
            plastic_edge += 1u;
        }
    }

    let previous_event = activity_sign(spikes_prev[flat]);
    if previous_event != 0.0 {
        let begin = plastic_outgoing[neuron];
        let end = plastic_outgoing[neuron + 1u];
        var index = begin;
        loop {
            if index >= end { break; }
            let plastic_edge = plastic_outgoing[params.plastic_out_edge_start + index];
            let post = topology[params.plastic_post_start + plastic_edge];
            if neurons[neuron_base + post].trace != 0.0 {
                set_live_bit(slot, plastic_edge);
            }
            index += 1u;
        }
    }
}

/// Eager exact live-set update. One invocation owns one 32-edge bitmap word,
/// so only P/32 lightweight invocations are dispatched. The dense numerical
/// plasticity expression is executed only for set bits.
///
/// Both bitmap buffers receive the same updated word. Population slots may be
/// stepped asynchronously (for example one slot receives extra DAN teaching
/// steps while another is idle), so a process-global A/B phase must never be
/// allowed to make an inactive slot's eligibility frontier disappear. Mirrored
/// words make either A/B role safe on the next active step without changing any
/// synaptic numerical state.
@compute @workgroup_size(256)
fn plasticity_live_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let flat_word = linear_invocation_index(gid, num_workgroups);
    let total_words = params.slot_count * params.bitmap_words;
    if flat_word >= total_words { return; }

    let slot = flat_word / params.bitmap_words;
    if !slot_is_active(slot) { return; }
    let word_index = flat_word % params.bitmap_words;
    let word = atomicLoad(&live_current[flat_word]);
    var next_word = 0u;

    let plastic_base = slot * params.plastic_edge_count;
    let neuron_base = slot * params.neuron_count;
    let first_edge = word_index << 5u;

    var bit = 0u;
    loop {
        if bit >= 32u { break; }
        let mask = 1u << bit;
        if (word & mask) != 0u {
            let plastic_edge = first_edge + bit;
            if plastic_edge < params.plastic_edge_count {
                let source_edge = topology[params.plastic_edge_start + plastic_edge];
                let post = topology[params.plastic_post_start + plastic_edge];
                let pre = topology[params.pre_start + source_edge];

                var syn = plastic_synapses[plastic_base + plastic_edge];
                let local = neurons[neuron_base + pre].trace * activity_sign(spikes_next[neuron_base + post])
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

                if syn.eligibility != 0.0 {
                    next_word = next_word | mask;
                }
            }
        }
        bit += 1u;
    }

    atomicStore(&live_current[flat_word], next_word);
    atomicStore(&live_next[flat_word], next_word);
}
