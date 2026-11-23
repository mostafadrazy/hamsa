//! The 24 conformer blocks and the encoder projector on the GPU, through wgpu (DirectX 12 / Vulkan / Metal).
//!
//! Five-value weights are uploaded packed (2-bit sign planes + 1-bit level planes + per-row lo/hi) and unpacked
//! inside the matmul kernel, so the encoder takes ~230 MB of GPU memory. One inference records every dispatch into a
//! single command buffer: the subsampled frames go up, the 640-dim encoder output comes back.


use crate::encoder::{relative_positions, Encoder, Norm, D};
use crate::gpu_shaders as shaders;
use crate::linear::Linear;

const FF: usize = 4096;
const HEADS: u32 = 8;
const DH: usize = 128;
/// The attention kernel keeps one score row in shared memory: at most this many frames (~40 s of audio).
pub const MAX_FRAMES: usize = 512;
const FRAMES_PER_GROUP: u32 = 64;

struct GFive {
    signs: wgpu::Buffer,
    flags: wgpu::Buffer,
    lohi: wgpu::Buffer,
    n: u32,
    k: u32,
}

struct GNorm {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
}

struct GBlock {
    norm_ff1: GNorm,
    ff1: [GFive; 2],
    norm_att: GNorm,
    q: GFive,
    k: GFive,
    v: GFive,
    pos: GFive,
    o: GFive,
    bias_u: wgpu::Buffer,
    bias_v: wgpu::Buffer,
    norm_conv: GNorm,
    pw1: GFive,
    dw_w: wgpu::Buffer,
    dw_b: wgpu::Buffer,
    pw2: GFive,
    norm_ff2: GNorm,
    ff2: [GFive; 2],
    norm_out: GNorm,
}

struct Pipeline {
    pipe: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub name: String,
    five: Pipeline,
    dense: Pipeline,
    norm: Pipeline,
    swish: Pipeline,
    attention: Pipeline,
    glu: Pipeline,
    blocks: Vec<GBlock>,
    proj_w: wgpu::Buffer,
    proj_b: wgpu::Buffer,
    proj_n: u32,
    /// Activation buffers sized for MAX_FRAMES, made once and reused, so memory stays flat across runs.
    work: Work,
    /// Dispatch parameters of the run being recorded; written into `work.params` before submitting.
    slots: std::sync::Mutex<Vec<[u32; 8]>>,
    /// One run at a time (they share `work`).
    running: std::sync::Mutex<()>,
}

struct Work {
    x: wgpu::Buffer,
    pe: wgpu::Buffer,
    ln: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    attn: wgpu::Buffer,
    act: wgpu::Buffer,
    h: wgpu::Buffer,
    pw: wgpu::Buffer,
    pos: wgpu::Buffer,
    out: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Uniform parameters, one 256-byte slot per dispatch.
    params: wgpu::Buffer,
}

/// Dispatches per run: ~21 per block x 24 blocks + the projector, with headroom.
const PARAM_SLOTS: usize = 1024;
/// Uniform buffer offsets must be multiples of this (the WebGPU default limit).
const SLOT_BYTES: usize = 256;

/// Bytes of weights written before waiting for the GPU to take them.
const UPLOAD_BATCH: usize = 16 << 20;

const USAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC).union(wgpu::BufferUsages::COPY_DST);

/// Private memory of this process in MB (Windows), for PHONON_GPU_TRACE.
pub(crate) fn private_mb() -> f64 {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut _ as *mut PROCESS_MEMORY_COUNTERS, c.cb);
        return c.PrivateUsage as f64 / 1e6;
    }
    #[allow(unreachable_code)]
    0.0
}

fn trace(stage: &str) {
    if std::env::var_os("PHONON_GPU_TRACE").is_some() {
        eprintln!("  gpu setup: {stage}: {:.0} MB", private_mb());
    }
}

impl Gpu {
    /// Pick a hardware GPU and upload the encoder. Fails (and the CPU path is used) when there's none.
    /// `allow_integrated`: also use a GPU built into the processor (Auto skips those, as the CPU path can be faster).
    pub fn new(enc: &Encoder, allow_integrated: bool) -> Result<Self, String> {
        trace("start");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("no GPU found")?;
        let info = adapter.get_info();
        match info.device_type {
            wgpu::DeviceType::DiscreteGpu => {}
            wgpu::DeviceType::IntegratedGpu if allow_integrated => {}
            wgpu::DeviceType::IntegratedGpu => return Err(format!("{} is a built-in GPU", info.name)),
            _ if allow_integrated && info.device_type != wgpu::DeviceType::Cpu => {}
            _ => return Err(format!("{} isn't a hardware GPU", info.name)),
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("phonon"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                // Same speed as `Performance` here, with smaller memory blocks.
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .map_err(|e| format!("couldn't open {}: {e}", info.name))?;
        trace("device open");
        // GPU errors must never take the app down: log them, and catch them in error scopes below.
        device.on_uncaptured_error(Box::new(|e| eprintln!("phonon: GPU error: {e}")));
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let pipeline = |src: &str, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: entry,
                compilation_options: Default::default(),
                cache: None,
            });
            let layout = pipe.get_bind_group_layout(0);
            Pipeline { pipe, layout }
        };
        let five = pipeline(shaders::LINEAR, "five");
        let dense = pipeline(shaders::LINEAR, "dense");
        let norm = pipeline(shaders::NORM, "main");
        let swish = pipeline(shaders::SWISH, "main");
        let attention = pipeline(shaders::ATTENTION, "main");
        let glu = pipeline(shaders::GLU_DWCONV, "main");

        // Weights go up through the queue in batches, waiting for each batch to land, so the staging copies in
        // system memory are reused instead of piling up (mapped-at-creation buffers kept ~650 MB of them alive).
        let pending = std::cell::Cell::new(0usize);
        let total = std::cell::Cell::new(0usize);
        let upload = |bytes: &[u8]| -> wgpu::Buffer {
            let size = (bytes.len() as u64).max(4).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: USAGE, mapped_at_creation: false });
            if bytes.len() % 4 == 0 {
                queue.write_buffer(&buf, 0, bytes);
            } else {
                let mut padded = bytes.to_vec();
                padded.resize(size as usize, 0);
                queue.write_buffer(&buf, 0, &padded);
            }
            pending.set(pending.get() + bytes.len());
            total.set(total.get() + bytes.len());
            if pending.get() >= UPLOAD_BATCH {
                queue.submit(None);
                device.poll(wgpu::Maintain::Wait);
                pending.set(0);
            }
            buf
        };
        let f32s = |v: &[f32]| upload(bytemuck::cast_slice(v));
        let gfive = |l: &Linear| -> Result<GFive, String> {
            let Linear::Five(f) = l else {
                return Err("an encoder layer isn't five-value".into());
            };
            // Two 16-bit flag words per u32 (WGSL has no 16-bit integers).
            let flags: Vec<u32> =
                f.flags.chunks(2).map(|c| c[0] as u32 | (*c.get(1).unwrap_or(&0) as u32) << 16).collect();
            let lohi: Vec<f32> = f.lo.iter().chain(&f.hi).copied().collect();
            Ok(GFive {
                signs: upload(bytemuck::cast_slice(&f.signs)),
                flags: upload(bytemuck::cast_slice(&flags)),
                lohi: f32s(&lohi),
                n: f.rows as u32,
                k: f.cols as u32,
            })
        };
        let gnorm = |n: &Norm| GNorm { w: f32s(&n.w), b: f32s(&n.b) };

        trace("pipelines");
        let mut blocks = Vec::with_capacity(enc.blocks.len());
        for b in &enc.blocks {
            blocks.push(GBlock {
                norm_ff1: gnorm(&b.norm_ff1),
                ff1: [gfive(&b.ff1[0])?, gfive(&b.ff1[1])?],
                norm_att: gnorm(&b.norm_att),
                q: gfive(&b.q)?,
                k: gfive(&b.k)?,
                v: gfive(&b.v)?,
                pos: gfive(&b.pos)?,
                o: gfive(&b.o)?,
                bias_u: f32s(&b.bias_u),
                bias_v: f32s(&b.bias_v),
                norm_conv: gnorm(&b.norm_conv),
                pw1: gfive(&b.pw1)?,
                dw_w: f32s(&b.dw_w),
                dw_b: f32s(&b.dw_b),
                pw2: gfive(&b.pw2)?,
                norm_ff2: gnorm(&b.norm_ff2),
                ff2: [gfive(&b.ff2[0])?, gfive(&b.ff2[1])?],
                norm_out: gnorm(&b.norm_out),
            });
        }
        let (proj_w, proj_n) = match &enc.proj {
            Linear::Dense { w, out, .. } => (f32s(w), *out as u32),
            Linear::Five(f) => (f32s(&f.to_dense()), f.rows as u32),
        };
        let proj_b = f32s(&enc.proj_b);

        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(format!("not enough GPU memory: {e}"));
        }
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(format!("GPU setup failed: {e}"));
        }

        trace(&format!("weights uploaded ({:.0} MB of GPU buffers)", total.get() as f64 / 1e6));
        // Finish the weight uploads now so their staging copies in system memory are released.
        queue.submit(None);
        device.poll(wgpu::Maintain::Wait);

        trace("uploads finished");
        let work = {
            let buf = |n: usize| {
                device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: USAGE, mapped_at_creation: false })
            };
            let (t, l, out_n) = (MAX_FRAMES, 2 * MAX_FRAMES - 1, proj_n as usize);
            Work {
                x: buf(t * D),
                pe: buf(l * D),
                ln: buf(t * D),
                q: buf(t * D),
                k: buf(t * D),
                v: buf(t * D),
                attn: buf(t * D),
                act: buf(t * D),
                h: buf(t * FF),
                pw: buf(t * 2 * D),
                pos: buf(l * D),
                out: buf(t * out_n),
                staging: device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (t * out_n * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                params: device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (PARAM_SLOTS * SLOT_BYTES) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            }
        };
        trace("work buffers");
        let name = format!("{} ({:?})", info.name, info.backend);
        Ok(Self {
            device,
            queue,
            name,
            five,
            dense,
            norm,
            swish,
            attention,
            glu,
            blocks,
            proj_w,
            proj_b,
            proj_n,
            work,
            slots: std::sync::Mutex::new(Vec::with_capacity(PARAM_SLOTS)),
            running: std::sync::Mutex::new(()),
        })
    }



    fn dispatch(&self, pass: &mut wgpu::ComputePass, p: &Pipeline, params: [u32; 8], bufs: &[(u32, &wgpu::Buffer)], groups: (u32, u32)) {
        let slot = {
            let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
            slots.push(params);
            // Past the last slot the run fails cleanly in `run` (checked before submitting).
            (slots.len() - 1).min(PARAM_SLOTS - 1)
        };
        let uniform = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.work.params,
            offset: (slot * SLOT_BYTES) as u64,
            size: std::num::NonZeroU64::new(32),
        });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: uniform }];
        entries.extend(bufs.iter().map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() }));
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.layout, entries: &entries });
        pass.set_pipeline(&p.pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    /// y[t × n] (=|+=) x[t × k] · Wᵀ
    fn linear(&self, pass: &mut wgpu::ComputePass, x: &wgpu::Buffer, w: &GFive, y: &wgpu::Buffer, t: u32, residual: Option<f32>) {
        let (mode, alpha) = residual.map_or((0, 0.0), |a| (1, a));
        self.dispatch(
            pass,
            &self.five,
            [t, w.n, w.k, mode, f32::to_bits(alpha), 0, 0, 0],
            &[(1, x), (2, &w.signs), (3, &w.flags), (4, &w.lohi), (5, y)],
            (w.n / 16, t.div_ceil(FRAMES_PER_GROUP)),
        );
    }

    fn layer_norm(&self, pass: &mut wgpu::ComputePass, x: &wgpu::Buffer, n: &GNorm, y: &wgpu::Buffer, t: u32) {
        self.dispatch(pass, &self.norm, [t, D as u32, 0, 0, 0, 0, 0, 0], &[(1, x), (2, &n.w), (3, &n.b), (4, y)], (t, 1));
    }

    fn swish(&self, pass: &mut wgpu::ComputePass, h: &wgpu::Buffer, n: u32) {
        self.dispatch(pass, &self.swish, [n, 0, 0, 0, 0, 0, 0, 0], &[(1, h)], (n.div_ceil(256), 1));
    }

    /// Encoder blocks + projector on the subsampled frames x [t × 1024] → [t × 640].
    pub fn forward(&self, x0: &[f32], t: usize) -> Result<Vec<f32>, String> {
        if t == 0 || t > MAX_FRAMES {
            return Err(format!("{t} frames is outside the GPU path's range"));
        }
        let dev = &self.device;
        dev.push_error_scope(wgpu::ErrorFilter::Validation);
        dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let result = self.run(x0, t);
        let oom = pollster::block_on(dev.pop_error_scope());
        let invalid = pollster::block_on(dev.pop_error_scope());
        if let Some(e) = oom.or(invalid) {
            return Err(format!("GPU error: {e}"));
        }
        result
    }

    fn run(&self, x0: &[f32], t: usize) -> Result<Vec<f32>, String> {
        let started = std::time::Instant::now();
        let l = 2 * t - 1;
        let tu = t as u32;
        let dev = &self.device;
        let _one_at_a_time = self.running.lock().unwrap_or_else(|e| e.into_inner());
        self.slots.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let w = &self.work;
        // Only the first t (or l) rows of each buffer are used; the kernels never read past them.
        self.queue.write_buffer(&w.x, 0, bytemuck::cast_slice(x0));
        self.queue.write_buffer(&w.pe, 0, bytemuck::cast_slice(&relative_positions(t)));
        let (x, pe, ln, q, k, v, attn, act, h, pw, pos, out, staging) =
            (&w.x, &w.pe, &w.ln, &w.q, &w.k, &w.v, &w.attn, &w.act, &w.h, &w.pw, &w.pos, &w.out, &w.staging);
        let out_n = self.proj_n as usize;

        let scale = 1.0f32 / (DH as f32).sqrt();
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });
        for b in &self.blocks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                let p = &mut pass;
                // x += ½ · ff1(LN(x))
                self.layer_norm(p, &x, &b.norm_ff1, &ln, tu);
                self.linear(p, &ln, &b.ff1[0], &h, tu, None);
                self.swish(p, &h, tu * FF as u32);
                self.linear(p, &h, &b.ff1[1], &x, tu, Some(0.5));
                // x += attention(LN(x))
                self.layer_norm(p, &x, &b.norm_att, &ln, tu);
                self.linear(p, &ln, &b.q, &q, tu, None);
                self.linear(p, &ln, &b.k, &k, tu, None);
                self.linear(p, &ln, &b.v, &v, tu, None);
                self.linear(p, &pe, &b.pos, &pos, l as u32, None);
                self.dispatch(
                    p,
                    &self.attention,
                    [tu, 0, 0, 0, f32::to_bits(scale), 0, 0, 0],
                    &[(1, &q), (2, &k), (3, &v), (4, &pos), (5, &b.bias_u), (6, &b.bias_v), (7, &attn)],
                    (tu, HEADS),
                );
                self.linear(p, &attn, &b.o, &x, tu, Some(1.0));
                // x += conv(LN(x))
                self.layer_norm(p, &x, &b.norm_conv, &ln, tu);
                self.linear(p, &ln, &b.pw1, &pw, tu, None);
                self.dispatch(
                    p,
                    &self.glu,
                    [tu, D as u32, 0, 0, 0, 0, 0, 0],
                    &[(1, &pw), (2, &b.dw_w), (3, &b.dw_b), (4, &act)],
                    ((tu * D as u32).div_ceil(256), 1),
                );
                self.linear(p, &act, &b.pw2, &x, tu, Some(1.0));
                // x += ½ · ff2(LN(x))
                self.layer_norm(p, &x, &b.norm_ff2, &ln, tu);
                self.linear(p, &ln, &b.ff2[0], &h, tu, None);
                self.swish(p, &h, tu * FF as u32);
                self.linear(p, &h, &b.ff2[1], &x, tu, Some(0.5));
                // x = LN_out(x), via ln and a copy (a buffer can't be read and written in one dispatch)
                self.layer_norm(p, &x, &b.norm_out, &ln, tu);
            }
            encoder.copy_buffer_to_buffer(&ln, 0, &x, 0, (t * D * 4) as u64);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            self.dispatch(
                &mut pass,
                &self.dense,
                [tu, self.proj_n, D as u32, 0, 0, 0, 0, 0],
                &[(1, &x), (5, &out), (6, &self.proj_w), (7, &self.proj_b)],
                (self.proj_n / 16, tu.div_ceil(FRAMES_PER_GROUP)),
            );
        }
        let out_bytes = (t * out_n * 4) as u64;
        encoder.copy_buffer_to_buffer(out, 0, staging, 0, out_bytes);

        // All dispatch parameters in one write, each in its own 256-byte slot.
        {
            let slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
            if slots.len() > PARAM_SLOTS {
                return Err(format!("{} dispatches is more than the {PARAM_SLOTS} parameter slots", slots.len()));
            }
            let mut bytes = vec![0u8; slots.len() * SLOT_BYTES];
            for (i, s) in slots.iter().enumerate() {
                bytes[i * SLOT_BYTES..i * SLOT_BYTES + 32].copy_from_slice(bytemuck::cast_slice(s));
            }
            self.queue.write_buffer(&w.params, 0, &bytes);
        }
        let recorded = std::time::Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|_| "GPU readback was lost".to_string())?.map_err(|e| format!("GPU readback failed: {e}"))?;
        let result = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        if std::env::var_os("PHONON_GPU_TRACE").is_some() {
            eprintln!(
                "  gpu: record {:.1} ms, execute+readback {:.1} ms",
                (recorded - started).as_secs_f64() * 1e3,
                recorded.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(result)
    }
}
