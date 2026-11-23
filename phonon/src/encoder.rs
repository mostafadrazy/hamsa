//! FastConformer encoder (transformers' ParakeetEncoder for parakeet-tdt-0.6b-v3):
//! mel [T × 128] → 8× convolutional subsampling → 24 conformer blocks → encoder_projector → [T' × 640].
//! No biases on the block linears, no input scaling, relative-position attention with per-layer bias_u / bias_v,
//! BatchNorm after the depthwise conv (folded into its weights at load).

use rayon::prelude::*;

use crate::container::{take_dense, take_linear, take_opt_vec, take_vec, Records};
use crate::linear::{add_bias, gemm, sigmoid, softmax, swish, Linear, Ptr};
use crate::mel::N_MELS;

pub(crate) const D: usize = 1024;
const HEADS: usize = 8;
const DH: usize = D / HEADS;
const FF: usize = 4096;
const KERNEL: usize = 9;
const SUB: usize = 256;
pub const JOINT: usize = 640;
const BLOCKS: usize = 24;
const LN_EPS: f64 = 1e-5;
const BN_EPS: f64 = 1e-5;

fn conv_out(n: usize) -> usize {
    (n - 1) / 2 + 1 // kernel 3, stride 2, padding 1
}

pub(crate) struct Norm {
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

impl Norm {
    fn load(r: &mut Records, prefix: &str) -> Result<Self, String> {
        Ok(Self { w: take_vec(r, &format!("{prefix}.weight"), D)?, b: take_vec(r, &format!("{prefix}.bias"), D)? })
    }

    fn apply(&self, x: &[f32], out: &mut [f32]) {
        out.par_chunks_mut(D).zip(x.par_chunks(D)).for_each(|(o, row)| layer_norm(row, &self.w, &self.b, o));
    }

    fn apply_in_place(&self, x: &mut [f32]) {
        x.par_chunks_mut(D).for_each(|row| {
            let copy = row.to_vec();
            layer_norm(&copy, &self.w, &self.b, row);
        });
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = (1.0 / (var + LN_EPS).sqrt()) as f32;
    let mean = mean as f32;
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

fn add_scaled(x: &mut [f32], y: &[f32], alpha: f32) {
    x.par_chunks_mut(D).zip(y.par_chunks(D)).for_each(|(a, b)| {
        for (u, v) in a.iter_mut().zip(b) {
            *u += alpha * v;
        }
    });
}

// ── Subsampling ──────────────────────────────────────────────────────────────

pub(crate) struct Subsampling {
    pub(crate) conv0_w: Vec<f32>,
    pub(crate) conv0_b: Vec<f32>,
    pub(crate) dw1_w: Vec<f32>,
    pub(crate) dw1_b: Vec<f32>,
    pub(crate) pw1_w: Vec<f32>,
    pub(crate) pw1_b: Vec<f32>,
    pub(crate) dw2_w: Vec<f32>,
    pub(crate) dw2_b: Vec<f32>,
    pub(crate) pw2_w: Vec<f32>,
    pub(crate) pw2_b: Vec<f32>,
    pub(crate) linear: Linear,
    pub(crate) linear_b: Vec<f32>,
}

/// Conv2d(1 → C, 3×3, stride 2, pad 1) over one [h × w] plane, or depthwise (groups = C) over [C × h × w].
fn conv3x3s2(x: &[f32], h: usize, w: usize, wt: &[f32], b: &[f32], channels: usize, depthwise: bool) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = (conv_out(h), conv_out(w));
    let mut out = vec![0.0f32; channels * ho * wo];
    out.par_chunks_mut(ho * wo).enumerate().for_each(|(c, o)| {
        let k = &wt[c * 9..c * 9 + 9];
        let plane = if depthwise { &x[c * h * w..(c + 1) * h * w] } else { &x[..h * w] };
        for oh in 0..ho {
            for ow in 0..wo {
                let mut acc = b[c];
                for kh in 0..3 {
                    let ih = 2 * oh + kh;
                    if ih == 0 || ih > h {
                        continue;
                    }
                    let row = &plane[(ih - 1) * w..ih * w];
                    for kw in 0..3 {
                        let iw = 2 * ow + kw;
                        if iw >= 1 && iw <= w {
                            acc += row[iw - 1] * k[kh * 3 + kw];
                        }
                    }
                }
                o[oh * wo + ow] = acc;
            }
        }
    });
    (out, ho, wo)
}

/// Pointwise Conv2d(C → C) over [C × hw] + bias, then ReLU.
fn conv1x1_relu(x: &[f32], hw: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; SUB * hw];
    gemm(SUB, SUB, hw, w, SUB, 1, x, hw, 1, &mut out, hw, 1);
    out.par_chunks_mut(hw).enumerate().for_each(|(c, row)| {
        for v in row.iter_mut() {
            *v = (*v + b[c]).max(0.0);
        }
    });
    out
}

impl Subsampling {
    fn load(r: &mut Records) -> Result<Self, String> {
        let p = "encoder.subsampling";
        let kernel = |r: &mut Records, layer: usize| take_dense(r, &format!("{p}.layers.{layer}.weight"), SUB, 9);
        let bias = |r: &mut Records, layer: usize| take_vec(r, &format!("{p}.layers.{layer}.bias"), SUB);
        let flat = SUB * conv_out(conv_out(conv_out(N_MELS)));
        Ok(Self {
            conv0_w: kernel(r, 0)?,
            conv0_b: bias(r, 0)?,
            dw1_w: kernel(r, 2)?,
            dw1_b: bias(r, 2)?,
            pw1_w: take_dense(r, &format!("{p}.layers.3.weight"), SUB, SUB)?,
            pw1_b: bias(r, 3)?,
            dw2_w: kernel(r, 5)?,
            dw2_b: bias(r, 5)?,
            pw2_w: take_dense(r, &format!("{p}.layers.6.weight"), SUB, SUB)?,
            pw2_b: bias(r, 6)?,
            linear: take_linear(r, &format!("{p}.linear.weight"), D, flat)?,
            linear_b: take_vec(r, &format!("{p}.linear.bias"), D)?,
        })
    }

    /// mel [frames × 128] → [T' × 1024]
    pub(crate) fn forward(&self, mel: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let (mut s, h1, w1) = conv3x3s2(mel, frames, N_MELS, &self.conv0_w, &self.conv0_b, SUB, false);
        s.par_iter_mut().for_each(|v| *v = v.max(0.0));
        let (s, h2, w2) = conv3x3s2(&s, h1, w1, &self.dw1_w, &self.dw1_b, SUB, true);
        let s = conv1x1_relu(&s, h2 * w2, &self.pw1_w, &self.pw1_b);
        let (s, h3, w3) = conv3x3s2(&s, h2, w2, &self.dw2_w, &self.dw2_b, SUB, true);
        let s = conv1x1_relu(&s, h3 * w3, &self.pw2_w, &self.pw2_b);

        // [C, T', W] → [T', C·W] (channel-major within a frame)
        let feat = SUB * w3;
        let mut flat = vec![0.0f32; h3 * feat];
        flat.par_chunks_mut(feat).enumerate().for_each(|(t, row)| {
            for c in 0..SUB {
                row[c * w3..(c + 1) * w3].copy_from_slice(&s[(c * h3 + t) * w3..(c * h3 + t + 1) * w3]);
            }
        });
        let mut x = vec![0.0f32; h3 * D];
        self.linear.forward(&flat, h3, &mut x);
        add_bias(&mut x, &self.linear_b);
        (x, h3)
    }

    fn bytes(&self) -> usize {
        self.linear.bytes() + 4 * (self.pw1_w.len() + self.pw2_w.len())
    }
}

// ── Conformer block ──────────────────────────────────────────────────────────

pub(crate) struct Block {
    pub(crate) norm_ff1: Norm,
    pub(crate) ff1: [Linear; 2],
    pub(crate) norm_att: Norm,
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) pos: Linear,
    pub(crate) o: Linear,
    pub(crate) bias_u: Vec<f32>,
    pub(crate) bias_v: Vec<f32>,
    pub(crate) norm_conv: Norm,
    pub(crate) pw1: Linear,
    pub(crate) dw_w: Vec<f32>,
    pub(crate) dw_b: Vec<f32>,
    pub(crate) pw2: Linear,
    pub(crate) norm_ff2: Norm,
    pub(crate) ff2: [Linear; 2],
    pub(crate) norm_out: Norm,
}

impl Block {
    fn load(r: &mut Records, i: usize) -> Result<Self, String> {
        let p = format!("encoder.layers.{i}");
        let lin = |r: &mut Records, name: &str, out: usize, inp: usize| take_linear(r, &format!("{p}.{name}.weight"), out, inp);
        let norm = |r: &mut Records, name: &str| Norm::load(r, &format!("{p}.{name}"));

        // Depthwise conv with the following BatchNorm folded in.
        let w = take_dense(r, &format!("{p}.conv.depthwise_conv.weight"), D, KERNEL)?;
        let conv_bias = take_opt_vec(r, &format!("{p}.conv.depthwise_conv.bias"), D)?;
        let gamma = take_vec(r, &format!("{p}.conv.norm.weight"), D)?;
        let beta = take_vec(r, &format!("{p}.conv.norm.bias"), D)?;
        let mean = take_vec(r, &format!("{p}.conv.norm.running_mean"), D)?;
        let var = take_vec(r, &format!("{p}.conv.norm.running_var"), D)?;
        let mut dw_w = vec![0.0f32; D * KERNEL];
        let mut dw_b = vec![0.0f32; D];
        for d in 0..D {
            let s = gamma[d] as f64 / (var[d] as f64 + BN_EPS).sqrt();
            for k in 0..KERNEL {
                dw_w[d * KERNEL + k] = (w[d * KERNEL + k] as f64 * s) as f32;
            }
            let cb = conv_bias.as_ref().map_or(0.0, |b| b[d] as f64);
            dw_b[d] = (beta[d] as f64 + (cb - mean[d] as f64) * s) as f32;
        }

        Ok(Self {
            norm_ff1: norm(r, "norm_feed_forward1")?,
            ff1: [lin(r, "feed_forward1.linear1", FF, D)?, lin(r, "feed_forward1.linear2", D, FF)?],
            norm_att: norm(r, "norm_self_att")?,
            q: lin(r, "self_attn.q_proj", D, D)?,
            k: lin(r, "self_attn.k_proj", D, D)?,
            v: lin(r, "self_attn.v_proj", D, D)?,
            pos: lin(r, "self_attn.relative_k_proj", D, D)?,
            o: lin(r, "self_attn.o_proj", D, D)?,
            bias_u: take_vec(r, &format!("{p}.self_attn.bias_u"), D)?,
            bias_v: take_vec(r, &format!("{p}.self_attn.bias_v"), D)?,
            norm_conv: norm(r, "norm_conv")?,
            pw1: lin(r, "conv.pointwise_conv1", 2 * D, D)?,
            dw_w,
            dw_b,
            pw2: lin(r, "conv.pointwise_conv2", D, D)?,
            norm_ff2: norm(r, "norm_feed_forward2")?,
            ff2: [lin(r, "feed_forward2.linear1", FF, D)?, lin(r, "feed_forward2.linear2", D, FF)?],
            norm_out: norm(r, "norm_out")?,
        })
    }

    fn bytes(&self) -> usize {
        let lins = [&self.ff1[0], &self.ff1[1], &self.q, &self.k, &self.v, &self.pos, &self.o, &self.pw1, &self.pw2, &self.ff2[0], &self.ff2[1]];
        lins.iter().map(|l| l.bytes()).sum::<usize>() + 4 * (self.dw_w.len() + 12 * D)
    }

    fn forward(&self, x: &mut [f32], t: usize, pe: &[f32]) {
        use crate::profile::{add, ATT, CONV, FF};
        let s = std::time::Instant::now();
        feed_forward(&self.norm_ff1, &self.ff1, x, t);
        add(FF, s);
        let s = std::time::Instant::now();
        self.attention(x, t, pe);
        add(ATT, s);
        let s = std::time::Instant::now();
        self.conv(x, t);
        add(CONV, s);
        let s = std::time::Instant::now();
        feed_forward(&self.norm_ff2, &self.ff2, x, t);
        add(FF, s);
        self.norm_out.apply_in_place(x);
    }

    /// Multi-head self-attention with Transformer-XL relative positions.
    fn attention(&self, x: &mut [f32], t: usize, pe: &[f32]) {
        let l = 2 * t - 1;
        let mut ln = vec![0.0f32; t * D];
        self.norm_att.apply(x, &mut ln);
        let (mut q, mut k, mut v) = (vec![0.0f32; t * D], vec![0.0f32; t * D], vec![0.0f32; t * D]);
        self.q.forward(&ln, t, &mut q);
        self.k.forward(&ln, t, &mut k);
        self.v.forward(&ln, t, &mut v);
        let mut p = vec![0.0f32; l * D];
        self.pos.forward(pe, l, &mut p);

        let core_start = std::time::Instant::now();
        let mut attn = vec![0.0f32; t * D];
        let out = Ptr::new(attn.as_mut_ptr());
        let scale = 1.0 / (DH as f32).sqrt();
        (0..HEADS).into_par_iter().for_each(|h| {
            let mut qu = vec![0.0f32; t * DH];
            let mut qv = vec![0.0f32; t * DH];
            for i in 0..t {
                for d in 0..DH {
                    let val = q[i * D + h * DH + d];
                    qu[i * DH + d] = val + self.bias_u[h * DH + d];
                    qv[i * DH + d] = val + self.bias_v[h * DH + d];
                }
            }
            let mut ac = vec![0.0f32; t * t];
            let mut bd = vec![0.0f32; t * l];
            // SAFETY: q/k/v/p are [rows × D] and head h reads columns [h·DH, (h+1)·DH); output columns are disjoint per head.
            unsafe {
                // ac = Q_u · Kᵀ   [t × t]
                matrixmultiply::sgemm(t, DH, t, 1.0, qu.as_ptr(), DH as isize, 1, k.as_ptr().add(h * DH), 1, D as isize, 0.0, ac.as_mut_ptr(), t as isize, 1);
                // bd = Q_v · Pᵀ   [t × (2t−1)]
                matrixmultiply::sgemm(t, DH, l, 1.0, qv.as_ptr(), DH as isize, 1, p.as_ptr().add(h * DH), 1, D as isize, 0.0, bd.as_mut_ptr(), l as isize, 1);
            }
            // Relative shift: query i, key j needs position i − j, at index t − 1 − i + j of bd's row.
            for i in 0..t {
                let row = &mut ac[i * t..(i + 1) * t];
                let shifted = &bd[i * l + (t - 1 - i)..i * l + (t - 1 - i) + t];
                for (a, b) in row.iter_mut().zip(shifted) {
                    *a = (*a + b) * scale;
                }
                softmax(row);
            }
            unsafe {
                // head output = scores · V   [t × DH] into columns of attn
                matrixmultiply::sgemm(t, t, DH, 1.0, ac.as_ptr(), t as isize, 1, v.as_ptr().add(h * DH), D as isize, 1, 0.0, out.get().add(h * DH), D as isize, 1);
            }
        });

        crate::profile::add(crate::profile::ATT_CORE, core_start);
        let mut o = vec![0.0f32; t * D];
        self.o.forward(&attn, t, &mut o);
        add_scaled(x, &o, 1.0);
    }

    /// LN → pointwise_conv1 → GLU → depthwise conv (+ folded BatchNorm) → SiLU → pointwise_conv2 → residual.
    fn conv(&self, x: &mut [f32], t: usize) {
        let mut ln = vec![0.0f32; t * D];
        self.norm_conv.apply(x, &mut ln);
        let mut pw = vec![0.0f32; t * 2 * D];
        self.pw1.forward(&ln, t, &mut pw);

        let pad = KERNEL / 2;
        let mut dw = vec![0.0f32; D * t]; // channel-major
        dw.par_chunks_mut(t).enumerate().for_each(|(d, out)| {
            let gated: Vec<f32> = (0..t).map(|i| pw[i * 2 * D + d] * sigmoid(pw[i * 2 * D + D + d])).collect();
            let w = &self.dw_w[d * KERNEL..(d + 1) * KERNEL];
            for i in 0..t {
                let mut acc = self.dw_b[d];
                for (kk, wk) in w.iter().enumerate() {
                    let src = i + kk;
                    if src >= pad && src - pad < t {
                        acc += gated[src - pad] * wk;
                    }
                }
                out[i] = swish(acc);
            }
        });
        let mut act = vec![0.0f32; t * D];
        act.par_chunks_mut(D).enumerate().for_each(|(i, row)| {
            for d in 0..D {
                row[d] = dw[d * t + i];
            }
        });
        let mut o = vec![0.0f32; t * D];
        self.pw2.forward(&act, t, &mut o);
        add_scaled(x, &o, 1.0);
    }
}

/// x += ½ · linear2(swish(linear1(LN(x))))
fn feed_forward(norm: &Norm, lin: &[Linear; 2], x: &mut [f32], t: usize) {
    let mut ln = vec![0.0f32; t * D];
    norm.apply(x, &mut ln);
    let mut h = vec![0.0f32; t * FF];
    lin[0].forward(&ln, t, &mut h);
    h.par_iter_mut().for_each(|v| *v = swish(*v));
    let mut o = vec![0.0f32; t * D];
    lin[1].forward(&h, t, &mut o);
    add_scaled(x, &o, 0.5);
}

/// [2t−1 × D]: row j holds relative position t−1−j; even columns sin, odd cos (frequency product in f32).
pub(crate) fn relative_positions(t: usize) -> Vec<f32> {
    let l = 2 * t - 1;
    let inv: Vec<f32> = (0..D / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / D as f32)).collect();
    let mut pe = vec![0.0f32; l * D];
    pe.par_chunks_mut(D).enumerate().for_each(|(j, row)| {
        let pos = (t as f32 - 1.0) - j as f32;
        for i in 0..D / 2 {
            let f = inv[i] * pos;
            row[2 * i] = f.sin();
            row[2 * i + 1] = f.cos();
        }
    });
    pe
}

// ── Encoder ──────────────────────────────────────────────────────────────────

pub struct Encoder {
    pub(crate) sub: Subsampling,
    pub(crate) blocks: Vec<Block>,
    pub(crate) proj: Linear,
    pub(crate) proj_b: Vec<f32>,
}

impl Encoder {
    pub fn load(r: &mut Records) -> Result<Self, String> {
        let sub = Subsampling::load(r)?;
        let blocks = (0..BLOCKS).map(|i| Block::load(r, i)).collect::<Result<Vec<_>, _>>()?;
        let proj = take_linear(r, "encoder_projector.weight", JOINT, D)?;
        let proj_b = take_vec(r, "encoder_projector.bias", JOINT)?;
        Ok(Self { sub, blocks, proj, proj_b })
    }

    pub fn bytes(&self) -> usize {
        self.sub.bytes() + self.blocks.iter().map(Block::bytes).sum::<usize>() + self.proj.bytes()
    }

    /// Free the conformer blocks and projector (once the GPU holds its own copy). The subsampling stays.
    pub fn drop_layers(&mut self) {
        self.blocks = Vec::new();
        self.proj = Linear::Dense { w: Vec::new(), out: 0, inp: 0 };
        self.proj_b = Vec::new();
    }

    /// Whether the conformer blocks are in memory (see `drop_layers`).
    pub fn has_layers(&self) -> bool {
        !self.blocks.is_empty()
    }

    /// Convolutional subsampling: mel [frames × 128] → ([T' × 1024], T').
    pub fn subsample(&self, mel: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let s = std::time::Instant::now();
        let r = self.sub.forward(mel, frames);
        crate::profile::add(crate::profile::SUB, s);
        r
    }

    /// Conformer blocks + projector on subsampled frames, on the CPU → [T' × 640].
    pub fn forward_blocks(&self, mut x: Vec<f32>, t: usize) -> Vec<f32> {
        let pe = relative_positions(t);
        for block in &self.blocks {
            block.forward(&mut x, t, &pe);
        }
        let mut out = vec![0.0f32; t * self.proj.out_features()];
        self.proj.forward(&x, t, &mut out);
        add_bias(&mut out, &self.proj_b);
        out
    }
}
