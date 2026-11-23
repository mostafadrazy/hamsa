//! LSTM prediction network, joint network and greedy TDT decoding (transformers' ParakeetTDTGenerationMixin, plus
//! NeMo's guard of at most 10 tokens per frame).

use rayon::prelude::*;

use crate::container::{take_quant, take_vec, Records};
use crate::encoder::JOINT;
use crate::linear::{sigmoid, Quant};

const HIDDEN: usize = 640;
const EMBED: usize = 640;
pub const VOCAB: usize = 8193; // 8192 SentencePiece pieces + blank
pub const BLANK: usize = 8192;
const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];
const N_LOGITS: usize = VOCAB + DURATIONS.len();
const MAX_SYMBOLS: usize = 10;
const ROWS_PER_TASK: usize = 128;

/// The decoder's tables stay as the container's 8-bit codes with a scale per row: 4× less memory to stream
/// per step than fp32, and the products are the same values.
pub struct Decoder {
    embedding: Quant,
    w_ih: [Quant; 2],
    w_hh: [Quant; 2],
    bias: [Vec<f32>; 2], // b_ih + b_hh
    proj_w: Quant,
    proj_b: Vec<f32>,
    head_w: Quant,
    head_b: Vec<f32>,
}

struct State {
    h: [Vec<f32>; 2],
    c: [Vec<f32>; 2],
}

/// out[r] = W[r] · x + b[r] for every row, spread over threads.
fn gemv(w: &Quant, x: &[f32], b: &[f32], out: &mut [f32]) {
    out.par_chunks_mut(ROWS_PER_TASK).enumerate().for_each(|(ci, chunk)| {
        for (j, o) in chunk.iter_mut().enumerate() {
            let r = ci * ROWS_PER_TASK + j;
            *o = w.dot_row(r, x) + b[r];
        }
    });
}

impl Decoder {
    pub fn load(r: &mut Records) -> Result<Self, String> {
        let mut lstm = |layer: usize| -> Result<(Quant, Quant, Vec<f32>), String> {
            let inp = if layer == 0 { EMBED } else { HIDDEN };
            let w_ih = take_quant(r, &format!("decoder.lstm.weight_ih_l{layer}"), 4 * HIDDEN, inp)?;
            let w_hh = take_quant(r, &format!("decoder.lstm.weight_hh_l{layer}"), 4 * HIDDEN, HIDDEN)?;
            let b_ih = take_vec(r, &format!("decoder.lstm.bias_ih_l{layer}"), 4 * HIDDEN)?;
            let b_hh = take_vec(r, &format!("decoder.lstm.bias_hh_l{layer}"), 4 * HIDDEN)?;
            Ok((w_ih, w_hh, b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect()))
        };
        let (ih0, hh0, b0) = lstm(0)?;
        let (ih1, hh1, b1) = lstm(1)?;
        Ok(Self {
            embedding: take_quant(r, "decoder.embedding.weight", VOCAB, EMBED)?,
            w_ih: [ih0, ih1],
            w_hh: [hh0, hh1],
            bias: [b0, b1],
            proj_w: take_quant(r, "decoder.decoder_projector.weight", JOINT, HIDDEN)?,
            proj_b: take_vec(r, "decoder.decoder_projector.bias", JOINT)?,
            head_w: take_quant(r, "joint.head.weight", N_LOGITS, JOINT)?,
            head_b: take_vec(r, "joint.head.bias", N_LOGITS)?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.embedding.bytes()
            + self.w_ih.iter().chain(&self.w_hh).map(Quant::bytes).sum::<usize>()
            + self.proj_w.bytes()
            + self.head_w.bytes()
    }

    /// One PyTorch nn.LSTM step (gate rows i, f, g, o).
    fn lstm(&self, layer: usize, x: &[f32], st: &mut State) {
        let h_prev = st.h[layer].clone();
        let (w_ih, w_hh, bias) = (&self.w_ih[layer], &self.w_hh[layer], &self.bias[layer]);
        let mut gates = vec![0.0f32; 4 * HIDDEN];
        gates.par_chunks_mut(ROWS_PER_TASK).enumerate().for_each(|(ci, chunk)| {
            for (j, g) in chunk.iter_mut().enumerate() {
                let r = ci * ROWS_PER_TASK + j;
                *g = w_ih.dot_row(r, x) + w_hh.dot_row(r, &h_prev) + bias[r];
            }
        });
        let (h, c) = (&mut st.h[layer], &mut st.c[layer]);
        for j in 0..HIDDEN {
            let i = sigmoid(gates[j]);
            let f = sigmoid(gates[HIDDEN + j]);
            let g = gates[2 * HIDDEN + j].tanh();
            let o = sigmoid(gates[3 * HIDDEN + j]);
            c[j] = f * c[j] + i * g;
            h[j] = o * c[j].tanh();
        }
    }

    /// embedding → 2-layer LSTM → decoder_projector
    fn step(&self, token: usize, st: &mut State, out: &mut [f32]) {
        let x = self.embedding.row(token);
        self.lstm(0, &x, st);
        let h0 = st.h[0].clone();
        self.lstm(1, &h0, st);
        gemv(&self.proj_w, &st.h[1], &self.proj_b, out);
    }

    fn joint(&self, enc_t: &[f32], dec: &[f32], logits: &mut [f32]) {
        let hidden: Vec<f32> = enc_t.iter().zip(dec).map(|(a, b)| (a + b).max(0.0)).collect();
        gemv(&self.head_w, &hidden, &self.head_b, logits);
    }

    /// Greedy token-and-duration decoding over encoder frames [t × 640]. Returns (token id, frame) pairs.
    /// With `allowed`, only tokens marked true (and blank) can be emitted.
    /// With `boost`, dictionary terms get a bonus (see boost.rs).
    pub fn greedy(
        &self,
        enc: &[f32],
        t: usize,
        allowed: Option<&[bool]>,
        boost: Option<&crate::boost::Boost>,
    ) -> Vec<(u32, u32)> {
        let mut st = State { h: [vec![0.0; HIDDEN], vec![0.0; HIDDEN]], c: [vec![0.0; HIDDEN], vec![0.0; HIDDEN]] };
        let mut dec = vec![0.0f32; JOINT];
        let mut logits = vec![0.0f32; N_LOGITS];
        self.step(BLANK, &mut st, &mut dec); // the blank row of the embedding is all zeros

        let mut tokens = Vec::new();
        let (mut frame, mut symbols) = (0usize, 0usize);
        let mut active: Vec<usize> = Vec::new();
        while frame < t {
            self.joint(&enc[frame * JOINT..(frame + 1) * JOINT], &dec, &mut logits);
            if let Some(b) = boost {
                b.apply(&active, &mut logits[..VOCAB]);
            }
            let token = match allowed {
                None => argmax(&logits[..VOCAB]),
                Some(mask) => argmax_allowed(&logits[..VOCAB], mask),
            };
            let mut duration = DURATIONS[argmax(&logits[VOCAB..])];
            if token == BLANK {
                duration = duration.max(1);
            } else {
                tokens.push((token as u32, frame as u32));
                if let Some(b) = boost {
                    active = b.advance(&active, token as u32);
                }
                self.step(token, &mut st, &mut dec);
                symbols += 1;
                if duration == 0 && symbols >= MAX_SYMBOLS {
                    duration = 1;
                }
            }
            if duration > 0 {
                symbols = 0;
            }
            frame += duration;
        }
        tokens
    }
}

/// Argmax over the tokens the mask allows; blank is always allowed.
fn argmax_allowed(v: &[f32], mask: &[bool]) -> usize {
    let mut best = BLANK;
    for i in 0..v.len() {
        if (mask[i] || i == BLANK) && v[i] > v[best] {
            best = i;
        }
    }
    best
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}
