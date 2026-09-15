mod implementation {
    include!("gpu_population_frontier.rs");

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct LiveParamsGpu {
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

    #[derive(Debug, Clone)]
    pub struct PopulationDebugState {
        pub membrane: Vec<f32>,
        pub trace: Vec<f32>,
        pub modulation: Vec<f32>,
        pub refractory: Vec<u32>,
        pub spikes: Vec<u32>,
        pub plastic_weight: Vec<f32>,
        pub eligibility: Vec<f32>,
        pub transaction_shift: Vec<f32>,
        pub transaction_lo: Vec<f32>,
        pub transaction_hi: Vec<f32>,
    }

    /// Production-facing population runtime with the same numerical dynamics as
    /// the outgoing-frontier reference, but plasticity is dispatched through an
    /// exact eager live bitmap rather than over all P compact plastic edges.
    ///
    /// Per slot, two P-bit maps are added (~2.6 MiB at the current MaleCNS P).
    /// One map represents non-zero eligibility entering the step. New edges
    /// whose local term can become non-zero are ORed into that map, then only set
    /// bits execute the canonical plasticity equation. The second map receives
    /// edges whose updated eligibility remains non-zero. The roles swap only on
    /// plasticity-enabled steps.
    pub struct LiveRuntime {
        inner: GpuPopulationRuntime,
        bitmap_words: usize,
        live_bitmap_a: wgpu::Buffer,
        live_bitmap_b: wgpu::Buffer,
        live_bind_ab_bits_ab: wgpu::BindGroup,
        live_bind_ab_bits_ba: wgpu::BindGroup,
        live_bind_ba_bits_ab: wgpu::BindGroup,
        live_bind_ba_bits_ba: wgpu::BindGroup,
        live_discovery_pipeline: wgpu::ComputePipeline,
        live_plasticity_pipeline: wgpu::ComputePipeline,
        live_bitmap_is_b: bool,
    }

    impl LiveRuntime {
        pub fn new(
            snapshot: &ConnectomeSnapshot,
            params: NeuralParams,
            slot_count: usize,
        ) -> Result<Self> {
            let inner = GpuPopulationRuntime::new(snapshot, params, slot_count)?;
            let plastic = PlasticFastGraph::compile(snapshot)?;
            let n = snapshot.neuron_count();
            let p = plastic.edge_count();
            let bitmap_words = p.div_ceil(32);

            let mut plastic_out_offsets = vec![0u32; n + 1];
            for &source_edge in &plastic.edge_indices {
                let pre = snapshot.pre_indices[source_edge as usize] as usize;
                plastic_out_offsets[pre + 1] = plastic_out_offsets[pre + 1]
                    .checked_add(1)
                    .context("plastic outgoing degree exceeds u32")?;
            }
            for index in 1..plastic_out_offsets.len() {
                plastic_out_offsets[index] = plastic_out_offsets[index]
                    .checked_add(plastic_out_offsets[index - 1])
                    .context("plastic outgoing offset exceeds u32")?;
            }
            let mut cursor = plastic_out_offsets[..n].to_vec();
            let mut plastic_out_edges = vec![0u32; p];
            for (plastic_edge, &source_edge) in plastic.edge_indices.iter().enumerate() {
                let pre = snapshot.pre_indices[source_edge as usize] as usize;
                let position = cursor[pre] as usize;
                plastic_out_edges[position] = plastic_edge as u32;
                cursor[pre] = cursor[pre]
                    .checked_add(1)
                    .context("plastic outgoing cursor exceeds u32")?;
            }
            let plastic_out_edge_start = plastic_out_offsets.len();
            let mut plastic_out_topology = Vec::with_capacity(plastic_out_edge_start + p);
            plastic_out_topology.extend_from_slice(&plastic_out_offsets);
            plastic_out_topology.extend_from_slice(&plastic_out_edges);
            let plastic_out_buffer = create_storage_init(
                &inner.device,
                "vf live plastic outgoing topology",
                bytemuck::cast_slice(&plastic_out_topology),
                false,
            );

            let total_bitmap_words = bitmap_words
                .checked_mul(slot_count)
                .context("live plasticity bitmap size overflow")?;
            let zero_bitmap = vec![0u32; total_bitmap_words.max(1)];
            let live_bitmap_a = create_storage_init(
                &inner.device,
                "vf live plastic bitmap a",
                bytemuck::cast_slice(&zero_bitmap),
                true,
            );
            let live_bitmap_b = create_storage_init(
                &inner.device,
                "vf live plastic bitmap b",
                bytemuck::cast_slice(&zero_bitmap),
                true,
            );

            let m = snapshot.edge_count();
            let pre_start = snapshot.row_offsets.len();
            let count_start = pre_start + m;
            let meta_start = count_start + m;
            let plastic_row_start = meta_start + n;
            let plastic_edge_start = plastic_row_start + plastic.row_offsets.len();
            let plastic_post_start = plastic_edge_start + p;

            let live_params = LiveParamsGpu {
                neuron_count: n as u32,
                plastic_edge_count: p as u32,
                slot_count: slot_count as u32,
                bitmap_words: bitmap_words as u32,
                pre_start: pre_start as u32,
                plastic_row_start: plastic_row_start as u32,
                plastic_edge_start: plastic_edge_start as u32,
                plastic_post_start: plastic_post_start as u32,
                plastic_out_edge_start: plastic_out_edge_start as u32,
                _pad_u0: 0,
                _pad_u1: 0,
                _pad_u2: 0,
                eligibility_decay: params.eligibility_decay,
                learning_rate: params.learning_rate,
                weight_max: params.weight_max,
                _pad_f0: 0.0,
            };
            let live_params_buffer = inner.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("vf live plasticity params"),
                    contents: bytemuck::bytes_of(&live_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );

            let live_layout = inner.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("vf live plasticity bind group layout"),
                    entries: &[
                        storage_layout(0, true),
                        storage_layout(1, true),
                        storage_layout(2, true),
                        storage_layout(3, true),
                        storage_layout(4, false),
                        storage_layout(5, false),
                        storage_layout(6, true),
                        storage_layout(7, false),
                        storage_layout(8, false),
                        uniform_layout(9),
                        uniform_layout(10),
                    ],
                },
            );
            let live_pipeline_layout = inner.device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("vf live plasticity pipeline layout"),
                    bind_group_layouts: &[Some(&live_layout)],
                    immediate_size: 0,
                },
            );
            let live_shader = inner.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vf live plasticity shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "population_plasticity_live.wgsl"
                ))),
            });
            let create_live_pipeline = |entry_point: &'static str| {
                inner.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&live_pipeline_layout),
                    module: &live_shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            };
            let live_discovery_pipeline = create_live_pipeline("discover_live_plasticity");
            let live_plasticity_pipeline = create_live_pipeline("plasticity_live_step");

            let make_live_bind = |
                spikes_prev: &wgpu::Buffer,
                spikes_next: &wgpu::Buffer,
                bits_current: &wgpu::Buffer,
                bits_next: &wgpu::Buffer,
                label: &'static str,
            | {
                inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &live_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: inner._topology_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: inner._neuron_state_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: spikes_prev.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: spikes_next.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: inner._plastic_synapse_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: inner._transaction_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: plastic_out_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: bits_current.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: bits_next.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 9, resource: live_params_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 10, resource: inner.control_buffer.as_entire_binding() },
                    ],
                })
            };
            let live_bind_ab_bits_ab = make_live_bind(
                &inner.spikes_a,
                &inner.spikes_b,
                &live_bitmap_a,
                &live_bitmap_b,
                "vf live plastic spikes a->b bits a->b",
            );
            let live_bind_ab_bits_ba = make_live_bind(
                &inner.spikes_a,
                &inner.spikes_b,
                &live_bitmap_b,
                &live_bitmap_a,
                "vf live plastic spikes a->b bits b->a",
            );
            let live_bind_ba_bits_ab = make_live_bind(
                &inner.spikes_b,
                &inner.spikes_a,
                &live_bitmap_a,
                &live_bitmap_b,
                "vf live plastic spikes b->a bits a->b",
            );
            let live_bind_ba_bits_ba = make_live_bind(
                &inner.spikes_b,
                &inner.spikes_a,
                &live_bitmap_b,
                &live_bitmap_a,
                "vf live plastic spikes b->a bits b->a",
            );

            Ok(Self {
                inner,
                bitmap_words,
                live_bitmap_a,
                live_bitmap_b,
                live_bind_ab_bits_ab,
                live_bind_ab_bits_ba,
                live_bind_ba_bits_ab,
                live_bind_ba_bits_ba,
                live_discovery_pipeline,
                live_plasticity_pipeline,
                live_bitmap_is_b: false,
            })
        }

        pub fn adapter_name(&self) -> &str { self.inner.adapter_name() }
        pub fn slot_count(&self) -> usize { self.inner.slot_count() }
        pub fn neuron_count(&self) -> usize { self.inner.neuron_count() }
        pub fn edge_count(&self) -> usize { self.inner.edge_count() }
        pub fn plastic_edge_count(&self) -> usize { self.inner.plastic_edge_count() }
        pub fn outgoing_edge_count(&self) -> usize { self.inner.outgoing_edge_count() }

        pub fn load_global_weights(&mut self, weights: &[f32]) -> Result<()> {
            self.inner.load_global_weights(weights)?;
            self.clear_all_live_bitmaps();
            self.live_bitmap_is_b = false;
            Ok(())
        }

        pub fn global_weights(&self) -> Result<Vec<f32>> { self.inner.global_weights() }
        pub fn global_state(&self) -> Result<NeuralState> { self.inner.global_state() }
        pub fn transaction_stats(&self, slot: usize, weight_max: f32) -> Result<TransactionStats> {
            self.inner.transaction_stats(slot, weight_max)
        }

        fn live_bind_group(&self, spike_is_b: bool, bitmap_is_b: bool) -> &wgpu::BindGroup {
            match (spike_is_b, bitmap_is_b) {
                (false, false) => &self.live_bind_ab_bits_ab,
                (false, true) => &self.live_bind_ab_bits_ba,
                (true, false) => &self.live_bind_ba_bits_ab,
                (true, true) => &self.live_bind_ba_bits_ba,
            }
        }

        fn encode_one_step(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            spike_is_b: bool,
            bitmap_is_b: bool,
            neuron_dispatch: (u32, u32),
            bitmap_dispatch: (u32, u32),
            plasticity: bool,
        ) {
            let neural_bind = if spike_is_b {
                &self.inner.bind_group_ba
            } else {
                &self.inner.bind_group_ab
            };

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vf live frontier candidate discovery"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, neural_bind, &[]);
                pass.set_pipeline(&self.inner.candidate_pipeline);
                pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vf live frontier neuron step"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, neural_bind, &[]);
                pass.set_pipeline(&self.inner.neuron_pipeline);
                pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
            }

            if plasticity && self.inner.plastic_edge_count > 0 {
                let live_bind = self.live_bind_group(spike_is_b, bitmap_is_b);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("vf live plasticity discovery"),
                        timestamp_writes: None,
                    });
                    pass.set_bind_group(0, live_bind, &[]);
                    pass.set_pipeline(&self.live_discovery_pipeline);
                    pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("vf live plasticity bitmap step"),
                        timestamp_writes: None,
                    });
                    pass.set_bind_group(0, live_bind, &[]);
                    pass.set_pipeline(&self.live_plasticity_pipeline);
                    pass.dispatch_workgroups(bitmap_dispatch.0, bitmap_dispatch.1, 1);
                }
            }

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vf live frontier trace step"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, neural_bind, &[]);
                pass.set_pipeline(&self.inner.trace_pipeline);
                pass.dispatch_workgroups(neuron_dispatch.0, neuron_dispatch.1, 1);
            }
        }

        pub fn step_batch_with_read(
            &mut self,
            stimuli_by_slot: &[Vec<Stimulus>],
            active_slots: &[bool],
            read_flat_indices: &[usize],
            plasticity: bool,
        ) -> Result<Vec<u32>> {
            let active_mask = self.inner.prepare_step(stimuli_by_slot, active_slots)?;
            for &index in read_flat_indices {
                if index >= self.slot_count() * self.neuron_count() {
                    bail!("population read index {index} is out of range");
                }
            }
            self.inner.write_control(0, active_mask);
            let max_groups = self.inner.device.limits().max_compute_workgroups_per_dimension;
            let neuron_dispatch = dispatch_grid(
                (self.slot_count() * self.neuron_count()) as u32,
                max_groups,
            )?;
            let bitmap_dispatch = dispatch_grid(
                (self.slot_count() * self.bitmap_words) as u32,
                max_groups,
            )?;

            let spike_is_b = self.inner.current_is_b;
            let bitmap_is_b = self.live_bitmap_is_b;
            let next_spikes = if spike_is_b {
                &self.inner.spikes_a
            } else {
                &self.inner.spikes_b
            };
            let mut encoder = self.inner.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("vf live population neural+gather encoder"),
                },
            );
            self.encode_one_step(
                &mut encoder,
                spike_is_b,
                bitmap_is_b,
                neuron_dispatch,
                bitmap_dispatch,
                plasticity,
            );

            if read_flat_indices.is_empty() {
                self.inner.queue.submit([encoder.finish()]);
                self.inner.current_is_b = !self.inner.current_is_b;
                if plasticity { self.live_bitmap_is_b = !self.live_bitmap_is_b; }
                return Ok(Vec::new());
            }

            let indices = read_flat_indices.iter().map(|&v| v as u32).collect::<Vec<_>>();
            let read_bytes = exact_byte_size::<u32>(indices.len());
            let index_buffer = device_storage_init(
                &self.inner.device,
                "vf live population gather indices",
                bytemuck::cast_slice(&indices),
            );
            let output_buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vf live population gather output"),
                size: read_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vf live population gather staging"),
                size: read_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let gather_bind_group = self.inner.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("vf live population gather bind group"),
                    layout: &self.inner.spike_gather_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: next_spikes.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: index_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                    ],
                },
            );
            let gather_dispatch = dispatch_grid(indices.len() as u32, max_groups)?;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vf live population motor gather"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &gather_bind_group, &[]);
                pass.set_pipeline(&self.inner.spike_gather_pipeline);
                pass.dispatch_workgroups(gather_dispatch.0, gather_dispatch.1, 1);
            }
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, read_bytes);
            self.inner.queue.submit([encoder.finish()]);
            self.inner.current_is_b = !self.inner.current_is_b;
            if plasticity { self.live_bitmap_is_b = !self.live_bitmap_is_b; }
            self.inner.map_staging::<u32>(&staging)
        }

        pub fn step_batch_no_read(
            &mut self,
            stimuli_by_slot: &[Vec<Stimulus>],
            active_slots: &[bool],
            plasticity: bool,
            steps: usize,
        ) -> Result<()> {
            if steps == 0 { bail!("population neural steps must be >= 1"); }
            let active_mask = self.inner.prepare_step(stimuli_by_slot, active_slots)?;
            self.inner.write_control(0, active_mask);
            let max_groups = self.inner.device.limits().max_compute_workgroups_per_dimension;
            let neuron_dispatch = dispatch_grid(
                (self.slot_count() * self.neuron_count()) as u32,
                max_groups,
            )?;
            let bitmap_dispatch = dispatch_grid(
                (self.slot_count() * self.bitmap_words) as u32,
                max_groups,
            )?;
            let mut encoder = self.inner.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("vf live repeated neural encoder"),
                },
            );
            let mut spike_is_b = self.inner.current_is_b;
            let mut bitmap_is_b = self.live_bitmap_is_b;
            for _ in 0..steps {
                self.encode_one_step(
                    &mut encoder,
                    spike_is_b,
                    bitmap_is_b,
                    neuron_dispatch,
                    bitmap_dispatch,
                    plasticity,
                );
                spike_is_b = !spike_is_b;
                if plasticity { bitmap_is_b = !bitmap_is_b; }
            }
            self.inner.queue.submit([encoder.finish()]);
            self.inner.current_is_b = spike_is_b;
            self.live_bitmap_is_b = bitmap_is_b;
            Ok(())
        }

        pub fn commit_and_restart_slot(&mut self, slot: usize) -> Result<()> {
            self.inner.commit_and_restart_slot(slot)?;
            self.clear_live_slot(slot)
        }

        pub fn restart_slot(&mut self, slot: usize) -> Result<()> {
            self.inner.restart_slot(slot)?;
            self.clear_live_slot(slot)
        }

        fn clear_all_live_bitmaps(&self) {
            let count = self.bitmap_words * self.slot_count();
            let zeros = vec![0u32; count.max(1)];
            self.inner.queue.write_buffer(
                &self.live_bitmap_a,
                0,
                bytemuck::cast_slice(&zeros),
            );
            self.inner.queue.write_buffer(
                &self.live_bitmap_b,
                0,
                bytemuck::cast_slice(&zeros),
            );
        }

        fn clear_live_slot(&self, slot: usize) -> Result<()> {
            self.inner.validate_slot(slot)?;
            if self.bitmap_words == 0 { return Ok(()); }
            let zeros = vec![0u32; self.bitmap_words];
            let offset = (slot * self.bitmap_words * std::mem::size_of::<u32>()) as u64;
            self.inner.queue.write_buffer(
                &self.live_bitmap_a,
                offset,
                bytemuck::cast_slice(&zeros),
            );
            self.inner.queue.write_buffer(
                &self.live_bitmap_b,
                offset,
                bytemuck::cast_slice(&zeros),
            );
            Ok(())
        }

        pub fn debug_slot_state(&self, slot: usize) -> Result<PopulationDebugState> {
            self.inner.validate_slot(slot)?;
            let all_neurons = self.inner.read_buffer::<NeuronStateGpu>(
                &self.inner._neuron_state_buffer,
                self.neuron_count() * self.slot_count(),
            )?;
            let current_spikes = if self.inner.current_is_b {
                &self.inner.spikes_b
            } else {
                &self.inner.spikes_a
            };
            let all_spikes = self.inner.read_buffer::<u32>(
                current_spikes,
                self.neuron_count() * self.slot_count(),
            )?;
            let all_synapses = self.inner.read_buffer::<SynapseStateGpu>(
                &self.inner._plastic_synapse_buffer,
                self.plastic_edge_count() * self.slot_count(),
            )?;
            let all_transactions = self.inner.read_buffer::<TransactionStateGpu>(
                &self.inner._transaction_buffer,
                self.plastic_edge_count() * self.slot_count(),
            )?;

            let ns = slot * self.neuron_count();
            let ne = ns + self.neuron_count();
            let ps = slot * self.plastic_edge_count();
            let pe = ps + self.plastic_edge_count();
            let neurons = &all_neurons[ns..ne];
            let synapses = &all_synapses[ps..pe];
            let transactions = &all_transactions[ps..pe];

            Ok(PopulationDebugState {
                membrane: neurons.iter().map(|v| v.membrane).collect(),
                trace: neurons.iter().map(|v| v.trace).collect(),
                modulation: neurons.iter().map(|v| v.modulation).collect(),
                refractory: neurons.iter().map(|v| v.refractory).collect(),
                spikes: all_spikes[ns..ne].to_vec(),
                plastic_weight: synapses.iter().map(|v| v.weight).collect(),
                eligibility: synapses.iter().map(|v| v.eligibility).collect(),
                transaction_shift: transactions.iter().map(|v| v.shift).collect(),
                transaction_lo: transactions.iter().map(|v| v.lo).collect(),
                transaction_hi: transactions.iter().map(|v| v.hi).collect(),
            })
        }
    }
}

pub use implementation::{
    LiveRuntime as GpuPopulationRuntime,
    PopulationDebugState,
    TransactionStats,
};
