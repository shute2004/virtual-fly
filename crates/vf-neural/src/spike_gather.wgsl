@group(0) @binding(0) var<storage, read> spikes: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;

fn linear_invocation_index(gid: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return gid.x + gid.y * num_workgroups.x * WORKGROUP_SIZE;
}

@compute @workgroup_size(256)
fn gather_spikes(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let output_index = linear_invocation_index(gid, num_workgroups);
    if output_index >= arrayLength(&indices) {
        return;
    }
    output[output_index] = spikes[indices[output_index]];
}
