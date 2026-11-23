//! Reader for Fermion's `fermion-five-value-parakeet-v1` containers (`model.fermion`).
//!
//! Layout: u64 LE header length, a JSON header `{format, index: [{n, k, shape, b}]}`, then the records back to back.
//! Record kinds:
//! - `five_value` [O, I]: base-3 codes, 5 per byte (code 1 = zero, sign = code − 1); then one bit per non-zero weight
//!   (LSB first, row-major) choosing hi over lo; then lo and hi as fp16 [O]. Kept packed here.
//! - `int6` / `int8` [O, …]: quantised codes then one fp16 scale per output row. Expanded to fp32.
//! - `fp16`: expanded to fp32.
//! Tensor names are transformers' ParakeetForTDT names; five-value records get `.weight` appended.

use std::collections::HashMap;

use half::f16;
use rayon::prelude::*;
use serde::Deserialize;

use crate::linear::{FiveValue, Linear, Quant, PANEL};

pub const FORMAT: &str = "fermion-five-value-parakeet-v1";

pub enum Record {
    Five(FiveValue),
    Quant(Quant),
    Dense { shape: Vec<usize>, data: Vec<f32> },
}

pub type Records = HashMap<String, Record>;

#[derive(Deserialize)]
struct Header {
    format: String,
    index: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    n: String,
    k: String,
    shape: Vec<usize>,
    b: usize,
}

pub fn read(bytes: &[u8]) -> Result<Records, String> {
    if bytes.len() < 8 {
        return Err("model file is too short".into());
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8usize
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or("model header length is out of range")?;
    let header: Header =
        serde_json::from_slice(&bytes[8..header_end]).map_err(|e| format!("model header is not valid JSON: {e}"))?;
    if header.format != FORMAT {
        return Err(format!("unsupported model format `{}` (expected `{FORMAT}`)", header.format));
    }

    // Locate every record first, then decode them in parallel.
    let mut off = header_end;
    let mut located = Vec::with_capacity(header.index.len());
    for e in header.index {
        let end = off
            .checked_add(e.b)
            .filter(|&x| x <= bytes.len())
            .ok_or_else(|| format!("{}: record runs past the end of the file", e.n))?;
        located.push((e, &bytes[off..end]));
        off = end;
    }
    if off != bytes.len() {
        return Err("unexpected bytes after the last record".into());
    }
    located.into_par_iter().map(|(e, blob)| decode(e, blob)).collect()
}

fn decode(e: Entry, blob: &[u8]) -> Result<(String, Record), String> {
    let numel: usize = e.shape.iter().product();
    Ok(match e.k.as_str() {
        "five_value" => (
            format!("{}.weight", e.n),
            Record::Five(five_value(blob, &e.shape).map_err(|m| format!("{}: {m}", e.n))?),
        ),
        "fp16" => {
            let data = fp16(blob, numel).map_err(|m| format!("{}: {m}", e.n))?;
            (e.n, Record::Dense { data, shape: e.shape })
        }
        kind if kind.starts_with("int") => {
            let bits: u32 = kind[3..].parse().map_err(|_| format!("{}: bad kind {kind}", e.n))?;
            (e.n.clone(), Record::Quant(intn(blob, &e.shape, bits).map_err(|m| format!("{}: {m}", e.n))?))
        }
        other => return Err(format!("{}: unknown record kind `{other}`", e.n)),
    })
}

fn fp16(blob: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if blob.len() != 2 * n {
        return Err(format!("fp16 record has {} bytes, expected {}", blob.len(), 2 * n));
    }
    Ok(blob.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect())
}

fn intn(blob: &[u8], shape: &[usize], bits: u32) -> Result<Quant, String> {
    let rows = *shape.first().ok_or("empty shape")?;
    let total: usize = shape.iter().product();
    let cols = total / rows.max(1);
    if blob.len() < 2 * rows {
        return Err("record too short for its scales".into());
    }
    let (body, scale_bytes) = blob.split_at(blob.len() - 2 * rows);
    let scales = fp16(scale_bytes, rows)?;
    let q: Vec<i32> = match bits {
        8 => {
            if body.len() < total {
                return Err("int8 record too short".into());
            }
            body[..total].iter().map(|&b| b as i8 as i32).collect()
        }
        6 => {
            let mut q = Vec::with_capacity(total + 4);
            for g in body.chunks_exact(3) {
                let p = g[0] as u32 | (g[1] as u32) << 8 | (g[2] as u32) << 16;
                for s in [0, 6, 12, 18] {
                    q.push(((p >> s) & 0x3f) as i32 - 32);
                }
            }
            if q.len() < total {
                return Err("int6 record too short".into());
            }
            q.truncate(total);
            q
        }
        _ => return Err(format!("unsupported int{bits} record")),
    };
    // int6 codes are −32…31 and int8 −128…127: both fit i8 exactly.
    Ok(Quant { rows, cols, codes: q.iter().map(|&v| v as i8).collect(), scales })
}

fn five_value(blob: &[u8], shape: &[usize]) -> Result<FiveValue, String> {
    let [rows, cols] = shape else {
        return Err(format!("five-value record must be 2-D, got {shape:?}"));
    };
    let (rows, cols) = (*rows, *cols);
    if rows % PANEL != 0 {
        return Err(format!("five-value height {rows} is not a multiple of {PANEL}"));
    }
    let rb = cols.div_ceil(5);
    let codes_len = rows * rb;
    if blob.len() < codes_len {
        return Err("five-value record too short".into());
    }
    const POW3: [u32; 5] = [1, 3, 9, 27, 81];
    // Panel-major (see FiveValue): row r's bits live in word (r / 16) · cols + c at position r % 16.
    let mut signs = vec![0u32; rows / PANEL * cols];
    let mut flags = vec![0u16; rows / PANEL * cols];
    let bits = &blob[codes_len..];
    let mut nz = 0usize;
    for r in 0..rows {
        let code_row = &blob[r * rb..(r + 1) * rb];
        let (p, j) = (r / PANEL, r % PANEL);
        for c in 0..cols {
            let code = (code_row[c / 5] as u32 / POW3[c % 5]) % 3;
            let i = p * cols + c;
            signs[i] |= code << (2 * j);
            if code != 1 {
                let byte = *bits.get(nz / 8).ok_or("five-value flag bits are truncated")?;
                if (byte >> (nz % 8)) & 1 == 1 {
                    flags[i] |= 1 << j;
                }
                nz += 1;
            }
        }
    }
    let off = codes_len + nz.div_ceil(8);
    if blob.len() != off + 4 * rows {
        return Err(format!("five-value record is {} bytes, expected {}", blob.len(), off + 4 * rows));
    }
    let lo = fp16(&blob[off..off + 2 * rows], rows)?;
    let hi = fp16(&blob[off + 2 * rows..], rows)?;
    Ok(FiveValue { rows, cols, signs, flags, lo, hi })
}

fn missing(key: &str) -> String {
    format!("model is missing `{key}`")
}

/// An fp32 vector of exactly `len` values.
pub fn take_vec(r: &mut Records, key: &str, len: usize) -> Result<Vec<f32>, String> {
    match r.remove(key).ok_or_else(|| missing(key))? {
        Record::Dense { data, .. } if data.len() == len => Ok(data),
        Record::Dense { data, .. } => Err(format!("`{key}` has {} values, expected {len}", data.len())),
        Record::Quant(q) if q.codes.len() == len => Ok(q.to_dense()),
        Record::Quant(q) => Err(format!("`{key}` has {} values, expected {len}", q.codes.len())),
        Record::Five(_) => Err(format!("`{key}` is five-value, expected a plain tensor")),
    }
}

/// A [rows × cols] table kept as 8-bit codes with a scale per row.
pub fn take_quant(r: &mut Records, key: &str, rows: usize, cols: usize) -> Result<Quant, String> {
    match r.remove(key).ok_or_else(|| missing(key))? {
        Record::Quant(q) if q.rows == rows && q.cols == cols => Ok(q),
        Record::Quant(q) => Err(format!("`{key}` is {}×{}, expected {rows}×{cols}", q.rows, q.cols)),
        _ => Err(format!("`{key}` is not an int6/int8 table")),
    }
}

pub fn take_opt_vec(r: &mut Records, key: &str, len: usize) -> Result<Option<Vec<f32>>, String> {
    if r.contains_key(key) {
        take_vec(r, key, len).map(Some)
    } else {
        Ok(None)
    }
}

/// A [rows × cols] matrix as fp32 (five-value records are expanded).
pub fn take_dense(r: &mut Records, key: &str, rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    match r.remove(key).ok_or_else(|| missing(key))? {
        Record::Dense { data, .. } if data.len() == rows * cols => Ok(data),
        Record::Dense { shape, .. } => Err(format!("`{key}` has shape {shape:?}, expected {rows}×{cols}")),
        Record::Quant(q) if q.codes.len() == rows * cols => Ok(q.to_dense()),
        Record::Quant(q) => Err(format!("`{key}` is {}×{}, expected {rows}×{cols}", q.rows, q.cols)),
        Record::Five(f) if f.rows == rows && f.cols == cols => Ok(f.to_dense()),
        Record::Five(f) => Err(format!("`{key}` is {}×{}, expected {rows}×{cols}", f.rows, f.cols)),
    }
}

/// A [out × in] linear layer, kept packed when it is five-value.
pub fn take_linear(r: &mut Records, key: &str, out: usize, inp: usize) -> Result<Linear, String> {
    match r.remove(key).ok_or_else(|| missing(key))? {
        Record::Five(f) if f.rows == out && f.cols == inp => Ok(Linear::Five(f)),
        Record::Five(f) => Err(format!("`{key}` is {}×{}, expected {out}×{inp}", f.rows, f.cols)),
        Record::Dense { data, .. } if data.len() == out * inp => Ok(Linear::Dense { w: data, out, inp }),
        Record::Dense { shape, .. } => Err(format!("`{key}` has shape {shape:?}, expected {out}×{inp}")),
        Record::Quant(q) if q.codes.len() == out * inp => Ok(Linear::Dense { w: q.to_dense(), out, inp }),
        Record::Quant(q) => Err(format!("`{key}` is {}×{}, expected {out}×{inp}", q.rows, q.cols)),
    }
}
