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

struct Params {
    neuron_count: u32,
    edge_count: u32,
    pre_start: u32,
    post_start: u32,
    count_start: u32,
    refractory_steps: u32,
    _pad_u0: u32,
    _pad_u1: u32,
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

@group(0) @binding(0) var<storage, read> topology: array<u32>;
@group(0) @binding(1) var<storage, read> metadata: array<u32>;
@group(0) @binding(2) var<storage, read_write> neurons: array<NeuronState>;
@group(0) @binding(3) var<storage, read> spikes_prev: array<u32>;
@group(0) @binding(4) var<storage, read_write> spikes_next: array<u32>;
@group(0) @binding(5) var<storage, read> external_current: array<f32>;
@group(0) @binding(6) var<storage, read_write> synapses: array<SynapseState>;
@group(0) @binding(7) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
const NT_DOPAMINE: u32 = 4u;

fn linear_invocation_index(gid: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    // Workgroups are tiled over X then Y. The local workgroup is 256x1x1, so
    // one Y row spans num_workgroups.x * 256 scalar invocations.
    return gid.x + gid.y * num_workgroups.x * WORKGROUP_SIZE;
}

fn nt_code(neuron: u32) -> u32 {
    return metadata[neuron] & 0xffu;
}

fn is_dopamine(neuron: u32) -> bool {
    return nt_code(neuron) == NT_DOPAMINE;
}

// This transmitter->fast-sign mapping is an explicit bootstrap assumption.
// It is not encoded as a fact in the MaleCNS source snapshot.
fn fast_sign(neuron: u32) -> f32 {
    switch nt_code(neuron) {
        case 1u: {
            return 1.0;
        }
        case 2u, 3u, 7u: {
            return -1.0;
        }
        default: {
            return 0.0;
        }
    }
}

@compute @workgroup_size(256)
fn neuron_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let post = linear_invocation_index(gid, num_workgroups);
    if post >= params.neuron_count {
        return;
    }

    var fast_current = external_current[post];
    var dopaminergic_input = 0.0;
    let begin = topology[post];
    let end = topology[post + 1u];

    var edge = begin;
    loop {
        if edge >= end {
            break;
        }
        let pre = topology[params.pre_start + edge];
        if spikes_prev[pre] != 0u {
            if is_dopamine(pre) {
                // PAM/PPL1/etc. are not assigned an external +1/-1 valence.
                // They differ through their actual released connectivity.
                let count = f32(topology[params.count_start + edge]);
                dopaminergic_input += count * params.modulator_scale;
            } else {
                fast_current += fast_sign(pre) * synapses[edge].weight;
            }
        }
        edge += 1u;
    }

    let old = neurons[post];
    var next = old;
    next.modulation = old.modulation * params.modulator_decay + dopaminergic_input;

    if old.refractory > 0u {
        next.membrane = params.reset;
        next.refractory = old.refractory - 1u;
        spikes_next[post] = 0u;
    } else {
        let membrane = old.membrane * params.membrane_decay + fast_current;
        if membrane >= params.threshold {
            next.membrane = params.reset;
            next.refractory = params.refractory_steps;
            spikes_next[post] = 1u;
        } else {
            next.membrane = membrane;
            next.refractory = 0u;
            spikes_next[post] = 0u;
        }
    }

    neurons[post] = next;
}

@compute @workgroup_size(256)
fn plasticity_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let edge = linear_invocation_index(gid, num_workgroups);
    if edge >= params.edge_count {
        return;
    }

    let pre = topology[params.pre_start + edge];
    let post = topology[params.post_start + edge];
    if fast_sign(pre) == 0.0 {
        return;
    }

    var syn = synapses[edge];
    let local = neurons[pre].trace * f32(spikes_next[post])
        - neurons[post].trace * f32(spikes_prev[pre]);
    syn.eligibility = syn.eligibility * params.eligibility_decay + local;
    syn.weight = clamp(
        syn.weight + params.learning_rate * neurons[post].modulation * syn.eligibility,
        0.0,
        params.weight_max,
    );
    synapses[edge] = syn;
}

@compute @workgroup_size(256)
fn trace_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let neuron = linear_invocation_index(gid, num_workgroups);
    if neuron >= params.neuron_count {
        return;
    }
    var state = neurons[neuron];
    state.trace = state.trace * params.trace_decay + f32(spikes_next[neuron]);
    neurons[neuron] = state;
}
