use std::{borrow::Cow, sync::mpsc, time::Duration};

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{
    ConnectomeSnapshot, NeuralParams, NeuralState, OutgoingPropagationGraph, PlasticFastGraph,
    Stimulus,
};

const WORKGROUP_SIZE: u32 = 256;
const MAX_ACTIVE_MASK_SLOTS: usize = 32;

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
struct TransactionStateGpu {
    shift: f32,
    lo: f32,
    hi: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsGpu {
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ControlGpu {
    slot: u32,
    active_mask: u32,
    plasticity_mask: u32,
    _pad1: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransactionStats {
    pub plastic_edges: usize,
    pub dirty_edges: usize,
    pub nonzero_shift_edges: usize,
    pub lower_bound_changed_edges: usize,
    pub upper_bound_changed_edges: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransactionEffectStats {
    pub changed_edges: usize,
    pub shift_changed_edges: usize,
    pub bound_changed_edges: usize,
    pub positive_shift_edges: usize,
    pub negative_shift_edges: usize,
    pub sum_shift_delta: f64,
    pub sum_abs_shift_delta: f64,
    pub max_abs_shift_delta: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransactionContrastStats {
    pub plastic_edges: usize,
    pub reward: TransactionEffectStats,
    pub aversive: TransactionEffectStats,
    pub shared_changed_edges: usize,
    pub reward_only_changed_edges: usize,
    pub aversive_only_changed_edges: usize,
    pub shared_shift_edges: usize,
    pub shared_same_sign_shift_edges: usize,
    pub shared_opposite_sign_shift_edges: usize,
    pub shift_dot: f64,
    pub reward_shift_norm_sq: f64,
    pub aversive_shift_norm_sq: f64,
}

pub struct GpuPopulationRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    neuron_count: usize,
    edge_count: usize,
    plastic_edge_count: usize,
    outgoing_edge_count: usize,
    slot_count: usize,
    external_cpu: Vec<f32>,
    _topology_buffer: wgpu::Buffer,
    external_buffer: wgpu::Buffer,
    _neuron_state_buffer: wgpu::Buffer,
    _plastic_synapse_buffer: wgpu::Buffer,
    _transaction_buffer: wgpu::Buffer,
    global_weight_buffer: wgpu::Buffer,
    spikes_a: wgpu::Buffer,
    spikes_b: wgpu::Buffer,
    bind_group_ab: wgpu::BindGroup,
    bind_group_ba: wgpu::BindGroup,
    candidate_pipeline: wgpu::ComputePipeline,
    neuron_pipeline: wgpu::ComputePipeline,
    plasticity_pipeline: wgpu::ComputePipeline,
    trace_pipeline: wgpu::ComputePipeline,
    commit_slot_pipeline: wgpu::ComputePipeline,
    restart_slot_synapses_pipeline: wgpu::ComputePipeline,
    reset_slot_neurons_pipeline: wgpu::ComputePipeline,
    spike_gather_layout: wgpu::BindGroupLayout,
    spike_gather_pipeline: wgpu::ComputePipeline,
    control_buffer: wgpu::Buffer,
    current_is_b: bool,
}

impl GpuPopulationRuntime {
    pub fn new(
        snapshot: &ConnectomeSnapshot,
        params: NeuralParams,
        slot_count: usize,
    ) -> Result<Self> {
        pollster::block_on(Self::new_async(snapshot, params, slot_count))
    }

    async fn new_async(
        snapshot: &ConnectomeSnapshot,
        params: NeuralParams,
        slot_count: usize,
    ) -> Result<Self> {
        if slot_count == 0 {
            bail!("population runtime requires at least one slot");
        }
        if slot_count > MAX_ACTIVE_MASK_SLOTS {
            bail!("population runtime currently supports at most {MAX_ACTIVE_MASK_SLOTS} slots");
        }
        if snapshot.neuron_count() > u32::MAX as usize || snapshot.edge_count() > u32::MAX as usize
        {
            bail!("population GPU runtime requires u32-sized neuron/edge dimensions");
        }

        let plastic = PlasticFastGraph::compile(snapshot)?;
        let outgoing = OutgoingPropagationGraph::compile(snapshot)?;
        let n = snapshot.neuron_count();
        let m = snapshot.edge_count();
        let p = plastic.edge_count();
        let q = outgoing.edge_count();
        let total_neurons = n
            .checked_mul(slot_count)
            .context("population neuron-state size overflow")?;
        let total_plastic = p
            .checked_mul(slot_count)
            .context("population plastic-state size overflow")?;
        if total_neurons > u32::MAX as usize || total_plastic > u32::MAX as usize {
            bail!("population GPU runtime flattened dimensions exceed u32");
        }

        let row_len = snapshot.row_offsets.len();
        let pre_start = row_len;
        let count_start = pre_start
            .checked_add(m)
            .context("population topology offset overflow")?;
        let meta_start = count_start
            .checked_add(m)
            .context("population topology offset overflow")?;
        let plastic_row_start = meta_start
            .checked_add(n)
            .context("population topology offset overflow")?;
        let plastic_edge_start = plastic_row_start
            .checked_add(plastic.row_offsets.len())
            .context("population topology offset overflow")?;
        let plastic_post_start = plastic_edge_start
            .checked_add(p)
            .context("population topology offset overflow")?;
        let outgoing_row_start = plastic_post_start
            .checked_add(p)
            .context("population topology offset overflow")?;
        let outgoing_post_start = outgoing_row_start
            .checked_add(outgoing.row_offsets.len())
            .context("population topology offset overflow")?;
        let topology_len = outgoing_post_start
            .checked_add(q)
            .context("population topology size overflow")?;
        if topology_len > u32::MAX as usize {
            bail!("population topology buffer indexing exceeds u32");
        }

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
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
                label: Some("virtual-fly frontier population neural compute device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create frontier population GPU device")?;

        let mut topology = Vec::with_capacity(topology_len);
        topology.extend_from_slice(&snapshot.row_offsets);
        topology.extend_from_slice(&snapshot.pre_indices);
        topology.extend_from_slice(&snapshot.synapse_counts);
        topology.extend(
            (0..snapshot.neuron_count()).map(|neuron| snapshot.packed_neuron_metadata(neuron)),
        );
        topology.extend_from_slice(&plastic.row_offsets);
        topology.extend_from_slice(&plastic.edge_indices);
        topology.extend_from_slice(&plastic.post_indices);
        topology.extend_from_slice(&outgoing.row_offsets);
        topology.extend_from_slice(&outgoing.post_indices);

        let global_weights = snapshot
            .synapse_counts
            .iter()
            .map(|&count| count as f32 * params.synapse_scale)
            .collect::<Vec<_>>();

        let topology_buffer = create_storage_init(
            &device,
            "vf frontier population topology",
            bytemuck::cast_slice(&topology),
            false,
        );
        let global_weight_buffer = create_storage_init(
            &device,
            "vf frontier population global weights",
            bytemuck::cast_slice(&global_weights),
            true,
        );
        let neuron_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf frontier population neuron state"),
            size: byte_size::<NeuronStateGpu>(total_neurons),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plastic_synapse_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf frontier population plastic synapse state"),
            size: byte_size::<SynapseStateGpu>(total_plastic),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transaction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf frontier population plastic transaction"),
            size: byte_size::<TransactionStateGpu>(total_plastic),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_spikes = vec![0u32; total_neurons];
        let spikes_a = create_storage_init(
            &device,
            "vf frontier population spikes a",
            bytemuck::cast_slice(&zero_spikes),
            true,
        );
        let spikes_b = create_storage_init(
            &device,
            "vf frontier population spikes b",
            bytemuck::cast_slice(&zero_spikes),
            true,
        );
        let external_cpu = vec![0.0f32; total_neurons];
        let external_buffer = create_storage_init(
            &device,
            "vf frontier population external current",
            bytemuck::cast_slice(&external_cpu),
            true,
        );

        let params_gpu = ParamsGpu {
            neuron_count: n as u32,
            edge_count: m as u32,
            plastic_edge_count: p as u32,
            slot_count: slot_count as u32,
            pre_start: pre_start as u32,
            count_start: count_start as u32,
            meta_start: meta_start as u32,
            plastic_row_start: plastic_row_start as u32,
            plastic_edge_start: plastic_edge_start as u32,
            plastic_post_start: plastic_post_start as u32,
            outgoing_row_start: outgoing_row_start as u32,
            outgoing_post_start: outgoing_post_start as u32,
            refractory_steps: params.refractory_steps,
            _pad_u0: 0,
            _pad_u1: 0,
            _pad_u2: 0,
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
            label: Some("vf frontier population params"),
            contents: bytemuck::bytes_of(&params_gpu),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let control_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vf frontier population control"),
            contents: bytemuck::bytes_of(&ControlGpu::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vf frontier population bind group layout"),
            entries: &[
                storage_layout(0, true),
                storage_layout(1, false),
                storage_layout(2, true),
                storage_layout(3, false),
                storage_layout(4, true),
                storage_layout(5, false),
                storage_layout(6, false),
                storage_layout(7, false),
                uniform_layout(8),
                uniform_layout(9),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vf frontier population pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vf frontier population neural shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "population_frontier.wgsl"
            ))),
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
        let candidate_pipeline = create_pipeline("mark_candidate_posts");
        let neuron_pipeline = create_pipeline("neuron_step");
        let plasticity_pipeline = create_pipeline("plasticity_step");
        let trace_pipeline = create_pipeline("trace_step");
        let commit_slot_pipeline = create_pipeline("commit_slot");
        let restart_slot_synapses_pipeline = create_pipeline("restart_slot_synapses");
        let reset_slot_neurons_pipeline = create_pipeline("reset_slot_neurons");

        let bind_group_ab = create_population_bind_group(
            &device,
            &layout,
            &topology_buffer,
            &neuron_state_buffer,
            &spikes_a,
            &spikes_b,
            &external_buffer,
            &plastic_synapse_buffer,
            &transaction_buffer,
            &global_weight_buffer,
            &params_buffer,
            &control_buffer,
            "vf frontier population a->b",
        );
        let bind_group_ba = create_population_bind_group(
            &device,
            &layout,
            &topology_buffer,
            &neuron_state_buffer,
            &spikes_b,
            &spikes_a,
            &external_buffer,
            &plastic_synapse_buffer,
            &transaction_buffer,
            &global_weight_buffer,
            &params_buffer,
            &control_buffer,
            "vf frontier population b->a",
        );

        let spike_gather_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vf frontier population spike gather layout"),
                entries: &[
                    storage_layout(0, true),
                    storage_layout(1, true),
                    storage_layout(2, false),
                ],
            });
        let spike_gather_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vf frontier population spike gather pipeline layout"),
                bind_group_layouts: &[Some(&spike_gather_layout)],
                immediate_size: 0,
            });
        let spike_gather_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vf frontier population spike gather shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("spike_gather.wgsl"))),
        });
        let spike_gather_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gather_frontier_population_spikes"),
                layout: Some(&spike_gather_pipeline_layout),
                module: &spike_gather_shader,
                entry_point: Some("gather_spikes"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut runtime = Self {
            device,
            queue,
            adapter_name: format!("{} ({:?})", adapter_info.name, adapter_info.backend),
            neuron_count: n,
            edge_count: m,
            plastic_edge_count: p,
            outgoing_edge_count: q,
            slot_count,
            external_cpu,
            _topology_buffer: topology_buffer,
            external_buffer,
            _neuron_state_buffer: neuron_state_buffer,
            _plastic_synapse_buffer: plastic_synapse_buffer,
            _transaction_buffer: transaction_buffer,
            global_weight_buffer,
            spikes_a,
            spikes_b,
            bind_group_ab,
            bind_group_ba,
            candidate_pipeline,
            neuron_pipeline,
            plasticity_pipeline,
            trace_pipeline,
            commit_slot_pipeline,
            restart_slot_synapses_pipeline,
            reset_slot_neurons_pipeline,
            spike_gather_layout,
            spike_gather_pipeline,
            control_buffer,
            current_is_b: false,
        };
        runtime.restart_all_slots()?;
        Ok(runtime)
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }
    pub fn neuron_count(&self) -> usize {
        self.neuron_count
    }
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }
    pub fn plastic_edge_count(&self) -> usize {
        self.plastic_edge_count
    }
    pub fn outgoing_edge_count(&self) -> usize {
        self.outgoing_edge_count
    }

    pub fn load_global_weights(&mut self, weights: &[f32]) -> Result<()> {
        if weights.len() != self.edge_count {
            bail!(
                "global weight count {} does not match edge count {}",
                weights.len(),
                self.edge_count
            );
        }
        self.queue
            .write_buffer(&self.global_weight_buffer, 0, bytemuck::cast_slice(weights));
        self.restart_all_slots()
    }

    pub fn global_weights(&self) -> Result<Vec<f32>> {
        self.read_buffer::<f32>(&self.global_weight_buffer, self.edge_count)
    }

    pub fn global_state(&self) -> Result<NeuralState> {
        let weights = self.global_weights()?;
        Ok(NeuralState {
            membrane: vec![0.0; self.neuron_count],
            spikes: vec![0; self.neuron_count],
            refractory: vec![0; self.neuron_count],
            activity_trace: vec![0.0; self.neuron_count],
            modulation: vec![0.0; self.neuron_count],
            weights,
            eligibility: vec![0.0; self.edge_count],
        })
    }

    pub fn transaction_stats(&self, slot: usize, weight_max: f32) -> Result<TransactionStats> {
        self.validate_slot(slot)?;
        let all = self.read_buffer::<TransactionStateGpu>(
            &self._transaction_buffer,
            self.plastic_edge_count * self.slot_count,
        )?;
        let start = slot * self.plastic_edge_count;
        let end = start + self.plastic_edge_count;
        let mut dirty_edges = 0usize;
        let mut nonzero_shift_edges = 0usize;
        let mut lower_bound_changed_edges = 0usize;
        let mut upper_bound_changed_edges = 0usize;
        for txn in &all[start..end] {
            let shift_changed = txn.shift != 0.0;
            let lower_changed = txn.lo != 0.0;
            let upper_changed = txn.hi != weight_max;
            if shift_changed || lower_changed || upper_changed {
                dirty_edges += 1;
            }
            nonzero_shift_edges += usize::from(shift_changed);
            lower_bound_changed_edges += usize::from(lower_changed);
            upper_bound_changed_edges += usize::from(upper_changed);
        }
        Ok(TransactionStats {
            plastic_edges: self.plastic_edge_count,
            dirty_edges,
            nonzero_shift_edges,
            lower_bound_changed_edges,
            upper_bound_changed_edges,
        })
    }

    pub fn transaction_contrast(
        &self,
        control_slot: usize,
        reward_slot: usize,
        aversive_slot: usize,
        epsilon: f32,
    ) -> Result<TransactionContrastStats> {
        self.validate_slot(control_slot)?;
        self.validate_slot(reward_slot)?;
        self.validate_slot(aversive_slot)?;
        if !(epsilon.is_finite() && epsilon > 0.0) {
            bail!("transaction contrast epsilon must be finite and positive");
        }
        if control_slot == reward_slot
            || control_slot == aversive_slot
            || reward_slot == aversive_slot
        {
            bail!("transaction contrast requires three distinct slots");
        }

        let all = self.read_buffer::<TransactionStateGpu>(
            &self._transaction_buffer,
            self.plastic_edge_count * self.slot_count,
        )?;
        let slice = |slot: usize| {
            let start = slot * self.plastic_edge_count;
            &all[start..start + self.plastic_edge_count]
        };
        let control = slice(control_slot);
        let reward = slice(reward_slot);
        let aversive = slice(aversive_slot);

        let mut reward_stats = TransactionEffectStats {
            changed_edges: 0,
            shift_changed_edges: 0,
            bound_changed_edges: 0,
            positive_shift_edges: 0,
            negative_shift_edges: 0,
            sum_shift_delta: 0.0,
            sum_abs_shift_delta: 0.0,
            max_abs_shift_delta: 0.0,
        };
        let mut aversive_stats = reward_stats;
        let mut shared_changed_edges = 0usize;
        let mut reward_only_changed_edges = 0usize;
        let mut aversive_only_changed_edges = 0usize;
        let mut shared_shift_edges = 0usize;
        let mut shared_same_sign_shift_edges = 0usize;
        let mut shared_opposite_sign_shift_edges = 0usize;
        let mut shift_dot = 0.0f64;
        let mut reward_shift_norm_sq = 0.0f64;
        let mut aversive_shift_norm_sq = 0.0f64;

        for edge in 0..self.plastic_edge_count {
            let c = control[edge];
            let r = reward[edge];
            let a = aversive[edge];
            let r_shift = r.shift - c.shift;
            let a_shift = a.shift - c.shift;
            let r_bound = (r.lo - c.lo).abs() > epsilon || (r.hi - c.hi).abs() > epsilon;
            let a_bound = (a.lo - c.lo).abs() > epsilon || (a.hi - c.hi).abs() > epsilon;
            let r_shift_changed = r_shift.abs() > epsilon;
            let a_shift_changed = a_shift.abs() > epsilon;
            let r_changed = r_shift_changed || r_bound;
            let a_changed = a_shift_changed || a_bound;

            reward_stats.changed_edges += usize::from(r_changed);
            reward_stats.shift_changed_edges += usize::from(r_shift_changed);
            reward_stats.bound_changed_edges += usize::from(r_bound);
            reward_stats.positive_shift_edges += usize::from(r_shift > epsilon);
            reward_stats.negative_shift_edges += usize::from(r_shift < -epsilon);
            reward_stats.sum_shift_delta += r_shift as f64;
            reward_stats.sum_abs_shift_delta += r_shift.abs() as f64;
            reward_stats.max_abs_shift_delta = reward_stats.max_abs_shift_delta.max(r_shift.abs());

            aversive_stats.changed_edges += usize::from(a_changed);
            aversive_stats.shift_changed_edges += usize::from(a_shift_changed);
            aversive_stats.bound_changed_edges += usize::from(a_bound);
            aversive_stats.positive_shift_edges += usize::from(a_shift > epsilon);
            aversive_stats.negative_shift_edges += usize::from(a_shift < -epsilon);
            aversive_stats.sum_shift_delta += a_shift as f64;
            aversive_stats.sum_abs_shift_delta += a_shift.abs() as f64;
            aversive_stats.max_abs_shift_delta =
                aversive_stats.max_abs_shift_delta.max(a_shift.abs());

            match (r_changed, a_changed) {
                (true, true) => shared_changed_edges += 1,
                (true, false) => reward_only_changed_edges += 1,
                (false, true) => aversive_only_changed_edges += 1,
                (false, false) => {}
            }
            if r_shift_changed && a_shift_changed {
                shared_shift_edges += 1;
                if r_shift.signum() == a_shift.signum() {
                    shared_same_sign_shift_edges += 1;
                } else {
                    shared_opposite_sign_shift_edges += 1;
                }
            }
            let r64 = r_shift as f64;
            let a64 = a_shift as f64;
            shift_dot += r64 * a64;
            reward_shift_norm_sq += r64 * r64;
            aversive_shift_norm_sq += a64 * a64;
        }

        Ok(TransactionContrastStats {
            plastic_edges: self.plastic_edge_count,
            reward: reward_stats,
            aversive: aversive_stats,
            shared_changed_edges,
            reward_only_changed_edges,
            aversive_only_changed_edges,
            shared_shift_edges,
            shared_same_sign_shift_edges,
            shared_opposite_sign_shift_edges,
            shift_dot,
            reward_shift_norm_sq,
            aversive_shift_norm_sq,
        })
    }

    pub fn step_batch_with_read(
        &mut self,
        stimuli_by_slot: &[Vec<Stimulus>],
        active_slots: &[bool],
        read_flat_indices: &[usize],
        plasticity: bool,
    ) -> Result<Vec<u32>> {
        let plasticity_slots = vec![plasticity; self.slot_count];
        self.step_batch_with_read_per_slot(
            stimuli_by_slot,
            active_slots,
            read_flat_indices,
            &plasticity_slots,
        )
    }

    pub fn step_batch_with_read_per_slot(
        &mut self,
        stimuli_by_slot: &[Vec<Stimulus>],
        active_slots: &[bool],
        read_flat_indices: &[usize],
        plasticity_slots: &[bool],
    ) -> Result<Vec<u32>> {
        if plasticity_slots.len() != self.slot_count {
            bail!(
                "population plasticity mask requires exactly {} slot flags",
                self.slot_count
            );
        }
        let active_mask = self.prepare_step(stimuli_by_slot, active_slots)?;
        let mut plasticity_mask = 0u32;
        for slot in 0..self.slot_count {
            if active_slots[slot] && plasticity_slots[slot] {
                plasticity_mask |= 1u32 << slot;
            }
        }
        for &index in read_flat_indices {
            if index >= self.slot_count * self.neuron_count {
                bail!("population read index {index} is out of range");
            }
        }
        self.write_control(0, active_mask, plasticity_mask);
        let max_groups = self.device.limits().max_compute_workgroups_per_dimension;
        let neuron_dispatch =
            dispatch_grid((self.slot_count * self.neuron_count) as u32, max_groups)?;
        let plasticity = plasticity_mask != 0;
        let plastic_dispatch = if plasticity && self.plastic_edge_count > 0 {
            Some(dispatch_grid(
                (self.slot_count * self.plastic_edge_count) as u32,
                max_groups,
            )?)
        } else {
            None
        };

        let current = self.current_is_b;
        let bind_group = if current {
            &self.bind_group_ba
        } else {
            &self.bind_group_ab
        };
        let next_spikes = if current {
            &self.spikes_a
        } else {
            &self.spikes_b
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf frontier population neural+gather encoder"),
            });
        self.encode_neural_step(
            &mut encoder,
            bind_group,
            neuron_dispatch,
            plastic_dispatch,
            plasticity,
        );

        if read_flat_indices.is_empty() {
            self.queue.submit([encoder.finish()]);
            self.current_is_b = !self.current_is_b;
            return Ok(Vec::new());
        }

        let indices = read_flat_indices
            .iter()
            .map(|&v| v as u32)
            .collect::<Vec<_>>();
        let read_bytes = exact_byte_size::<u32>(indices.len());
        let index_buffer = device_storage_init(
            &self.device,
            "vf frontier population gather indices",
            bytemuck::cast_slice(&indices),
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf frontier population gather output"),
            size: read_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf frontier population gather staging"),
            size: read_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gather_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf frontier population gather bind group"),
            layout: &self.spike_gather_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: next_spikes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let gather_dispatch = dispatch_grid(indices.len() as u32, max_groups)?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf frontier population motor gather"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &gather_bind_group, &[]);
            pass.set_pipeline(&self.spike_gather_pipeline);
            pass.dispatch_workgroups(gather_dispatch.0, gather_dispatch.1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, read_bytes);
        self.queue.submit([encoder.finish()]);
        self.current_is_b = !self.current_is_b;
        self.map_staging::<u32>(&staging)
    }

    pub fn step_batch_no_read(
        &mut self,
        stimuli_by_slot: &[Vec<Stimulus>],
        active_slots: &[bool],
        plasticity: bool,
        steps: usize,
    ) -> Result<()> {
        if steps == 0 {
            bail!("population neural steps must be >= 1");
        }
        let active_mask = self.prepare_step(stimuli_by_slot, active_slots)?;
        self.write_control(0, active_mask, if plasticity { active_mask } else { 0 });
        let max_groups = self.device.limits().max_compute_workgroups_per_dimension;
        let neuron_dispatch =
            dispatch_grid((self.slot_count * self.neuron_count) as u32, max_groups)?;
        let plastic_dispatch = if plasticity && self.plastic_edge_count > 0 {
            Some(dispatch_grid(
                (self.slot_count * self.plastic_edge_count) as u32,
                max_groups,
            )?)
        } else {
            None
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf frontier population repeated neural encoder"),
            });
        let mut current = self.current_is_b;
        for _ in 0..steps {
            let bind_group = if current {
                &self.bind_group_ba
            } else {
                &self.bind_group_ab
            };
            self.encode_neural_step(
                &mut encoder,
                bind_group,
                neuron_dispatch,
                plastic_dispatch,
                plasticity,
            );
            current = !current;
        }
        self.queue.submit([encoder.finish()]);
        self.current_is_b = current;
        Ok(())
    }

    pub fn commit_and_restart_slot(&mut self, slot: usize) -> Result<()> {
        self.validate_slot(slot)?;
        self.write_control(slot, 0, 0);
        let max_groups = self.device.limits().max_compute_workgroups_per_dimension;
        let plastic_dispatch = dispatch_grid(self.plastic_edge_count as u32, max_groups)?;
        let neuron_dispatch = dispatch_grid(self.neuron_count as u32, max_groups)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf frontier population commit+restart encoder"),
            });
        let bind_group = if self.current_is_b {
            &self.bind_group_ba
        } else {
            &self.bind_group_ab
        };
        if self.plastic_edge_count > 0 {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vf frontier population commit slot transaction"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_pipeline(&self.commit_slot_pipeline);
                pass.dispatch_workgroups(plastic_dispatch.0, plastic_dispatch.1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vf frontier population restart slot synapses"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_pipeline(&self.restart_slot_synapses_pipeline);
                pass.dispatch_workgroups(plastic_dispatch.0, plastic_dispatch.1, 1);
            }
        }
        self.encode_reset_slot_neurons(&mut encoder, neuron_dispatch);
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    pub fn restart_slot(&mut self, slot: usize) -> Result<()> {
        self.validate_slot(slot)?;
        self.write_control(slot, 0, 0);
        let max_groups = self.device.limits().max_compute_workgroups_per_dimension;
        let plastic_dispatch = dispatch_grid(self.plastic_edge_count as u32, max_groups)?;
        let neuron_dispatch = dispatch_grid(self.neuron_count as u32, max_groups)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf frontier population restart slot encoder"),
            });
        let bind_group = if self.current_is_b {
            &self.bind_group_ba
        } else {
            &self.bind_group_ab
        };
        if self.plastic_edge_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf frontier population restart slot synapses"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(&self.restart_slot_synapses_pipeline);
            pass.dispatch_workgroups(plastic_dispatch.0, plastic_dispatch.1, 1);
        }
        self.encode_reset_slot_neurons(&mut encoder, neuron_dispatch);
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    fn restart_all_slots(&mut self) -> Result<()> {
        self.current_is_b = false;
        for slot in 0..self.slot_count {
            self.restart_slot(slot)?;
        }
        Ok(())
    }

    fn encode_neural_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        neuron_dispatch: (u32, u32),
        plastic_dispatch: Option<(u32, u32)>,
        plasticity: bool,
    ) {
        // Separate passes make the candidate atomic writes unambiguously visible
        // to neuron_step before any incoming-scan decision is made.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf frontier candidate discovery"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(&self.candidate_pipeline);
            pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf frontier neural step"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(&self.neuron_pipeline);
            pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
            if plasticity {
                if let Some((x, y)) = plastic_dispatch {
                    pass.set_pipeline(&self.plasticity_pipeline);
                    pass.dispatch_workgroups(x, y, 1);
                }
            }
            pass.set_pipeline(&self.trace_pipeline);
            pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
        }
    }

    fn encode_reset_slot_neurons(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        neuron_dispatch: (u32, u32),
    ) {
        for bind_group in [&self.bind_group_ab, &self.bind_group_ba] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf frontier population reset slot neurons"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(&self.reset_slot_neurons_pipeline);
            pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
        }
    }

    fn prepare_step(
        &mut self,
        stimuli_by_slot: &[Vec<Stimulus>],
        active_slots: &[bool],
    ) -> Result<u32> {
        if stimuli_by_slot.len() != self.slot_count || active_slots.len() != self.slot_count {
            bail!(
                "population step requires exactly {} slot payloads",
                self.slot_count
            );
        }
        self.external_cpu.fill(0.0);
        let mut active_mask = 0u32;
        for slot in 0..self.slot_count {
            if !active_slots[slot] {
                continue;
            }
            active_mask |= 1u32 << slot;
            let base = slot * self.neuron_count;
            for stimulus in &stimuli_by_slot[slot] {
                if stimulus.neuron >= self.neuron_count {
                    bail!("stimulus neuron {} is out of range", stimulus.neuron);
                }
                self.external_cpu[base + stimulus.neuron] += stimulus.current;
            }
        }
        self.queue.write_buffer(
            &self.external_buffer,
            0,
            bytemuck::cast_slice(&self.external_cpu),
        );
        Ok(active_mask)
    }

    fn validate_slot(&self, slot: usize) -> Result<()> {
        if slot >= self.slot_count {
            bail!(
                "population slot {slot} is out of range 0..{}",
                self.slot_count
            );
        }
        Ok(())
    }

    fn write_control(&self, slot: usize, active_mask: u32, plasticity_mask: u32) {
        let control = ControlGpu {
            slot: slot as u32,
            active_mask,
            plasticity_mask,
            _pad1: 0,
        };
        self.queue
            .write_buffer(&self.control_buffer, 0, bytemuck::bytes_of(&control));
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
                    result.context("population GPU readback map failed")?;
                    mapped = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => continue,
                Err(mpsc::TryRecvError::Disconnected) => {
                    bail!("population GPU readback callback channel disconnected")
                }
            }
        }
        if !mapped {
            bail!("population GPU readback timed out");
        }
        let view = slice
            .get_mapped_range()
            .context("failed to access population mapped GPU readback range")?;
        let values = bytemuck::cast_slice::<u8, T>(&view).to_vec();
        drop(view);
        staging.unmap();
        Ok(values)
    }

    fn read_buffer<T: Pod + Copy>(&self, source: &wgpu::Buffer, count: usize) -> Result<Vec<T>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let size = exact_byte_size::<T>(count);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf frontier population readback staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf frontier population readback encoder"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        self.map_staging::<T>(&staging)
    }
}

fn byte_size<T>(count: usize) -> u64 {
    (count.max(1) * std::mem::size_of::<T>()) as u64
}

fn exact_byte_size<T>(count: usize) -> u64 {
    (count * std::mem::size_of::<T>()) as u64
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
            "population GPU dispatch requires {total_groups} workgroups, exceeding a {max_groups_per_dimension}x{max_groups_per_dimension} grid"
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

fn uniform_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
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

fn device_storage_init(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[u8],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

#[allow(clippy::too_many_arguments)]
fn create_population_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    topology: &wgpu::Buffer,
    neurons: &wgpu::Buffer,
    spikes_prev: &wgpu::Buffer,
    spikes_next: &wgpu::Buffer,
    external: &wgpu::Buffer,
    plastic_synapses: &wgpu::Buffer,
    transactions: &wgpu::Buffer,
    global_weights: &wgpu::Buffer,
    params: &wgpu::Buffer,
    control: &wgpu::Buffer,
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
                resource: neurons.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: spikes_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: spikes_next.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: external.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: plastic_synapses.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: transactions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: global_weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: control.as_entire_binding(),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::dispatch_grid;

    #[test]
    fn frontier_population_dispatch_tiles_plastic_edge_sets() {
        assert_eq!(dispatch_grid(2 * 10_871_322, 65_535).unwrap(), (65_535, 2));
    }
}
