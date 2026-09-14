use std::{borrow::Cow, sync::mpsc, time::Duration};

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{
    model::{NeuralParams, Stimulus},
    snapshot::ConnectomeSnapshot,
    state::NeuralState,
};

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NeuronStateGpu {
    membrane: f32,
    trace: f32,
    modulation: f32,
    refractory: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SynapseStateGpu {
    weight: f32,
    eligibility: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsGpu {
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

#[derive(Debug, Clone)]
pub struct GpuReadback {
    pub spikes: Vec<u32>,
    pub membrane: Vec<f32>,
    pub modulation: Vec<f32>,
    pub weights: Vec<f32>,
}

pub struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    neuron_count: usize,
    edge_count: usize,
    external_cpu: Vec<f32>,
    _meta_buffer: wgpu::Buffer,
    external_buffer: wgpu::Buffer,
    neuron_state_buffer: wgpu::Buffer,
    synapse_state_buffer: wgpu::Buffer,
    spikes_a: wgpu::Buffer,
    spikes_b: wgpu::Buffer,
    bind_group_ab: wgpu::BindGroup,
    bind_group_ba: wgpu::BindGroup,
    neuron_pipeline: wgpu::ComputePipeline,
    plasticity_pipeline: wgpu::ComputePipeline,
    trace_pipeline: wgpu::ComputePipeline,
    reset_eligibility_pipeline: wgpu::ComputePipeline,
    current_is_b: bool,
}

impl GpuRuntime {
    pub fn new(snapshot: &ConnectomeSnapshot, params: NeuralParams) -> Result<Self> {
        pollster::block_on(Self::new_async(snapshot, params))
    }

    async fn new_async(snapshot: &ConnectomeSnapshot, params: NeuralParams) -> Result<Self> {
        if snapshot.neuron_count() > u32::MAX as usize || snapshot.edge_count() > u32::MAX as usize {
            bail!("GPU bootstrap backend currently requires u32-sized neuron/edge indices");
        }

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .context("no compatible GPU adapter found")?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("virtual-fly neural compute device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create GPU device")?;

        let n = snapshot.neuron_count();
        let m = snapshot.edge_count();
        let row_len = snapshot.row_offsets.len() as u32;
        let pre_start = row_len;
        let post_start = pre_start + m as u32;
        let count_start = post_start + m as u32;

        let mut topology = Vec::with_capacity(snapshot.row_offsets.len() + 3 * m);
        topology.extend_from_slice(&snapshot.row_offsets);
        topology.extend_from_slice(&snapshot.pre_indices);
        topology.extend_from_slice(&snapshot.edge_posts);
        topology.extend_from_slice(&snapshot.synapse_counts);

        let meta_cpu = snapshot
            .neurotransmitters
            .iter()
            .map(|&nt| nt as u32)
            .collect::<Vec<_>>();
        let neuron_state = vec![NeuronStateGpu::zeroed(); n];
        let synapse_state = snapshot
            .synapse_counts
            .iter()
            .map(|&count| SynapseStateGpu {
                weight: count as f32 * params.synapse_scale,
                eligibility: 0.0,
            })
            .collect::<Vec<_>>();
        let zero_spikes = vec![0u32; n];
        let external_cpu = vec![0.0f32; n];

        let topology_buffer = create_storage_init(
            &device,
            "vf topology",
            bytemuck::cast_slice(&topology),
            false,
        );
        let meta_buffer = create_storage_init(
            &device,
            "vf neuron metadata",
            bytemuck::cast_slice(&meta_cpu),
            false,
        );
        let neuron_state_buffer = create_storage_init(
            &device,
            "vf neuron state",
            bytemuck::cast_slice(&neuron_state),
            true,
        );
        let synapse_state_buffer = create_storage_init(
            &device,
            "vf synapse state",
            bytemuck::cast_slice(&synapse_state),
            true,
        );
        let spikes_a = create_storage_init(
            &device,
            "vf spikes a",
            bytemuck::cast_slice(&zero_spikes),
            true,
        );
        let spikes_b = create_storage_init(
            &device,
            "vf spikes b",
            bytemuck::cast_slice(&zero_spikes),
            true,
        );
        let external_buffer = create_storage_init(
            &device,
            "vf external current",
            bytemuck::cast_slice(&external_cpu),
            true,
        );

        let gpu_params = ParamsGpu {
            neuron_count: n as u32,
            edge_count: m as u32,
            pre_start,
            post_start,
            count_start,
            refractory_steps: params.refractory_steps,
            _pad_u0: 0,
            _pad_u1: 0,
            membrane_decay: params.membrane_decay,
            threshold: params.threshold,
            reset: params.reset,
            trace_decay: params.trace_decay,
            eligibility_decay: params.eligibility_decay,
            learning_rate: params.learning_rate,
            weight_max: params.weight_max,
            modulator_decay: params.modulator_decay,
            modulator_scale: params.modulator_scale,
            _pad_f0: 0.0,
            _pad_f1: 0.0,
            _pad_f2: 0.0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vf neural params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf neural bind group layout"),
            entries: &[
                storage_layout(0, true),
                storage_layout(1, true),
                storage_layout(2, false),
                storage_layout(3, true),
                storage_layout(4, false),
                storage_layout(5, true),
                storage_layout(6, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf neural pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vf neural shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("neural.wgsl"))),
        });

        let create_pipeline = |entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let neuron_pipeline = create_pipeline("neuron_step");
        let plasticity_pipeline = create_pipeline("plasticity_step");
        let trace_pipeline = create_pipeline("trace_step");
        let reset_eligibility_pipeline = create_pipeline("reset_eligibility");

        let bind_group_ab = create_bind_group(
            &device,
            &bind_group_layout,
            &topology_buffer,
            &meta_buffer,
            &neuron_state_buffer,
            &spikes_a,
            &spikes_b,
            &external_buffer,
            &synapse_state_buffer,
            &params_buffer,
            "vf neural a->b",
        );
        let bind_group_ba = create_bind_group(
            &device,
            &bind_group_layout,
            &topology_buffer,
            &meta_buffer,
            &neuron_state_buffer,
            &spikes_b,
            &spikes_a,
            &external_buffer,
            &synapse_state_buffer,
            &params_buffer,
            "vf neural b->a",
        );

        Ok(Self {
            device,
            queue,
            adapter_name: format!("{} ({:?})", adapter_info.name, adapter_info.backend),
            neuron_count: n,
            edge_count: m,
            external_cpu,
            _meta_buffer: meta_buffer,
            external_buffer,
            neuron_state_buffer,
            synapse_state_buffer,
            spikes_a,
            spikes_b,
            bind_group_ab,
            bind_group_ba,
            neuron_pipeline,
            plasticity_pipeline,
            trace_pipeline,
            reset_eligibility_pipeline,
            current_is_b: false,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub fn step(&mut self, stimuli: &[Stimulus], plasticity_enabled: bool) -> Result<()> {
        self.external_cpu.fill(0.0);
        for stimulus in stimuli {
            if stimulus.neuron >= self.neuron_count {
                bail!("stimulus neuron {} is out of range", stimulus.neuron);
            }
            self.external_cpu[stimulus.neuron] += stimulus.current;
        }
        self.queue.write_buffer(
            &self.external_buffer,
            0,
            bytemuck::cast_slice(&self.external_cpu),
        );

        let max_groups = self.device.limits().max_compute_workgroups_per_dimension;
        let neuron_dispatch = dispatch_grid(self.neuron_count as u32, max_groups)?;
        let edge_dispatch = if plasticity_enabled && self.edge_count > 0 {
            Some(dispatch_grid(self.edge_count as u32, max_groups)?)
        } else {
            None
        };

        let bind_group = if self.current_is_b {
            &self.bind_group_ba
        } else {
            &self.bind_group_ab
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf neural step encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf neural step"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(&self.neuron_pipeline);
            pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);

            if let Some((x, y)) = edge_dispatch {
                pass.set_pipeline(&self.plasticity_pipeline);
                pass.dispatch_workgroups(x, y, 1);
            }

            pass.set_pipeline(&self.trace_pipeline);
            pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
        }
        self.queue.submit([encoder.finish()]);
        self.current_is_b = !self.current_is_b;
        Ok(())
    }

    /// Reset trial-local dynamics while keeping learned synaptic weights resident
    /// on the GPU. This avoids a full ~25.6M-edge GPU -> CPU -> GPU round trip at
    /// every Flyppy episode boundary.
    pub fn reset_dynamics(&mut self) -> Result<()> {
        let zero_neurons = vec![NeuronStateGpu::zeroed(); self.neuron_count];
        let zero_spikes = vec![0u32; self.neuron_count];
        self.external_cpu.fill(0.0);
        self.queue.write_buffer(
            &self.neuron_state_buffer,
            0,
            bytemuck::cast_slice(&zero_neurons),
        );
        self.queue.write_buffer(&self.spikes_a, 0, bytemuck::cast_slice(&zero_spikes));
        self.queue.write_buffer(&self.spikes_b, 0, bytemuck::cast_slice(&zero_spikes));
        self.queue.write_buffer(
            &self.external_buffer,
            0,
            bytemuck::cast_slice(&self.external_cpu),
        );

        if self.edge_count > 0 {
            let max_groups = self.device.limits().max_compute_workgroups_per_dimension;
            let (x, y) = dispatch_grid(self.edge_count as u32, max_groups)?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vf reset eligibility encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vf reset eligibility"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.bind_group_ab, &[]);
                pass.set_pipeline(&self.reset_eligibility_pipeline);
                pass.dispatch_workgroups(x, y, 1);
            }
            self.queue.submit([encoder.finish()]);
        }
        self.current_is_b = false;
        Ok(())
    }

    pub fn state(&self) -> Result<NeuralState> {
        let spikes = self.read_buffer::<u32>(self.current_spike_buffer(), self.neuron_count)?;
        let neurons = self.read_buffer::<NeuronStateGpu>(&self.neuron_state_buffer, self.neuron_count)?;
        let synapses = self.read_buffer::<SynapseStateGpu>(&self.synapse_state_buffer, self.edge_count)?;
        Ok(NeuralState {
            membrane: neurons.iter().map(|state| state.membrane).collect(),
            spikes,
            refractory: neurons.iter().map(|state| state.refractory).collect(),
            activity_trace: neurons.iter().map(|state| state.trace).collect(),
            modulation: neurons.iter().map(|state| state.modulation).collect(),
            weights: synapses.iter().map(|state| state.weight).collect(),
            eligibility: synapses.iter().map(|state| state.eligibility).collect(),
        })
    }

    pub fn load_state(&mut self, state: &NeuralState) -> Result<()> {
        state.validate(self.neuron_count, self.edge_count)?;
        let neurons = (0..self.neuron_count)
            .map(|index| NeuronStateGpu {
                membrane: state.membrane[index],
                trace: state.activity_trace[index],
                modulation: state.modulation[index],
                refractory: state.refractory[index],
            })
            .collect::<Vec<_>>();
        let synapses = (0..self.edge_count)
            .map(|index| SynapseStateGpu {
                weight: state.weights[index],
                eligibility: state.eligibility[index],
            })
            .collect::<Vec<_>>();

        self.queue.write_buffer(
            &self.neuron_state_buffer,
            0,
            bytemuck::cast_slice(&neurons),
        );
        self.queue.write_buffer(
            &self.synapse_state_buffer,
            0,
            bytemuck::cast_slice(&synapses),
        );
        self.queue.write_buffer(
            &self.spikes_a,
            0,
            bytemuck::cast_slice(&state.spikes),
        );
        self.queue.write_buffer(
            &self.spikes_b,
            0,
            bytemuck::cast_slice(&state.spikes),
        );
        self.external_cpu.fill(0.0);
        self.queue.write_buffer(
            &self.external_buffer,
            0,
            bytemuck::cast_slice(&self.external_cpu),
        );
        self.current_is_b = false;
        Ok(())
    }

    pub fn readback(&self) -> Result<GpuReadback> {
        let state = self.state()?;
        Ok(GpuReadback {
            spikes: state.spikes,
            membrane: state.membrane,
            modulation: state.modulation,
            weights: state.weights,
        })
    }

    pub fn read_spikes(&self) -> Result<Vec<u32>> {
        self.read_buffer::<u32>(self.current_spike_buffer(), self.neuron_count)
    }

    /// Read only explicitly requested neuron events. Closed-loop body control
    /// normally asks for ~60 wing motor neurons instead of synchronously copying
    /// all 166,700 activity codes from GPU to CPU every control step.
    pub fn read_spikes_at(&self, indices: &[usize]) -> Result<Vec<u32>> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        for &index in indices {
            if index >= self.neuron_count {
                bail!("spike readback index {index} is out of range");
            }
        }

        let item_size = std::mem::size_of::<u32>() as u64;
        let size = indices.len() as u64 * item_size;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf selected spike staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let source = self.current_spike_buffer();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf selected spike readback encoder"),
            });
        for (output_index, &neuron_index) in indices.iter().enumerate() {
            encoder.copy_buffer_to_buffer(
                source,
                neuron_index as u64 * item_size,
                &staging,
                output_index as u64 * item_size,
                item_size,
            );
        }
        self.queue.submit([encoder.finish()]);
        self.map_staging::<u32>(&staging)
    }

    fn current_spike_buffer(&self) -> &wgpu::Buffer {
        if self.current_is_b {
            &self.spikes_b
        } else {
            &self.spikes_a
        }
    }

    fn map_staging<T: Pod + Copy>(&self, staging: &wgpu::Buffer) -> Result<Vec<T>> {
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let mut mapped = false;
        for _ in 0..120 {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_millis(500)),
            });
            match rx.try_recv() {
                Ok(result) => {
                    result.context("GPU readback map failed")?;
                    mapped = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => continue,
                Err(mpsc::TryRecvError::Disconnected) => {
                    bail!("GPU readback callback channel disconnected")
                }
            }
        }
        if !mapped {
            bail!("GPU readback timed out");
        }

        let view = slice
            .get_mapped_range()
            .context("failed to access mapped GPU readback range")?;
        let values = bytemuck::cast_slice::<u8, T>(&view).to_vec();
        drop(view);
        staging.unmap();
        Ok(values)
    }

    fn read_buffer<T: Pod + Copy>(&self, source: &wgpu::Buffer, count: usize) -> Result<Vec<T>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let size = (count * std::mem::size_of::<T>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf readback staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf readback encoder"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        self.map_staging::<T>(&staging)
    }
}

fn dispatch_grid(item_count: u32, max_groups_per_dimension: u32) -> Result<(u32, u32)> {
    if item_count == 0 {
        return Ok((0, 0));
    }
    if max_groups_per_dimension == 0 {
        bail!("GPU reports zero max compute workgroups per dimension");
    }

    let total_groups = item_count.div_ceil(WORKGROUP_SIZE);
    let x = total_groups.min(max_groups_per_dimension);
    let y = total_groups.div_ceil(x);
    if y > max_groups_per_dimension {
        bail!(
            "GPU dispatch requires {total_groups} workgroups, exceeding a {max_groups_per_dimension}x{max_groups_per_dimension} grid"
        );
    }
    Ok((x, y))
}

fn storage_layout(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_storage_init(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[u8],
    cpu_writable: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    if cpu_writable {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
    })
}

#[allow(clippy::too_many_arguments)]
fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    topology: &wgpu::Buffer,
    meta: &wgpu::Buffer,
    neuron_state: &wgpu::Buffer,
    spikes_prev: &wgpu::Buffer,
    spikes_next: &wgpu::Buffer,
    external: &wgpu::Buffer,
    synapse_state: &wgpu::Buffer,
    params: &wgpu::Buffer,
    label: &'static str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: topology.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: neuron_state.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: spikes_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: spikes_next.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: external.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: synapse_state.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::dispatch_grid;

    #[test]
    fn dispatch_grid_stays_one_dimensional_when_it_fits() {
        assert_eq!(dispatch_grid(166_700, 65_535).unwrap(), (652, 1));
    }

    #[test]
    fn dispatch_grid_tiles_malecns_edges_over_y() {
        assert_eq!(dispatch_grid(25_582_938, 65_535).unwrap(), (65_535, 2));
    }

    #[test]
    fn dispatch_grid_rejects_impossible_grid() {
        let max_items = 4 * 4 * 256;
        assert!(dispatch_grid(max_items + 1, 4).is_err());
    }
}
