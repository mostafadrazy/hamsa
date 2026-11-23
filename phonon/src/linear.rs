//! Matrix products and small numeric helpers.
//!
//! Five-value weights stay packed (2-bit sign + 1-bit lo/hi flag per weight, one lo/hi pair per output row). A matmul
//! expands 64 output rows at a time into a thread-local fp32 panel and multiplies it with `matrixmultiply`'s
//! SIMD sgemm, so the fp32 copy of a matrix never exists in full.

use rayon::prelude::*;

/// Output rows expanded per work item.
const BLOCK: usize = 64;

#[derive(Clone, Copy)]
pub(crate) struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}
impl Ptr {
    pub fn new(p: *mut f32) -> Self {
        Ptr(p)
    }
    pub fn get(self) -> *mut f32 {
        self.0
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CPtr(*const f32);
unsafe impl Send for CPtr {}
unsafe impl Sync for CPtr {}
impl CPtr {
    pub fn new(p: *const f32) -> Self {
        CPtr(p)
    }
    pub fn get(self) -> *const f32 {
        self.0
    }
}

/// c[m×n] = a[m×k] · b[k×n] (strides in elements), split across threads into column or row blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    rsb: usize,
    csb: usize,
    c: &mut [f32],
    rsc: usize,
    csc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let (ap, bp, cp) = (CPtr::new(a.as_ptr()), CPtr::new(b.as_ptr()), Ptr::new(c.as_mut_ptr()));
    if n >= m {
        (0..n.div_ceil(BLOCK)).into_par_iter().for_each(|bi| {
            let j0 = bi * BLOCK;
            let nb = BLOCK.min(n - j0);
            // SAFETY: each block writes a disjoint set of columns of c; the callers size a, b and c for the strides.
            unsafe {
                matrixmultiply::sgemm(
                    m, k, nb, 1.0,
                    ap.get(), rsa as isize, csa as isize,
                    bp.get().add(j0 * csb), rsb as isize, csb as isize,
                    0.0,
                    cp.get().add(j0 * csc), rsc as isize, csc as isize,
                );
            }
        });
    } else {
        (0..m.div_ceil(BLOCK)).into_par_iter().for_each(|bi| {
            let i0 = bi * BLOCK;
            let mb = BLOCK.min(m - i0);
            // SAFETY: each block writes a disjoint set of rows of c.
            unsafe {
                matrixmultiply::sgemm(
                    mb, k, n, 1.0,
                    ap.get().add(i0 * rsa), rsa as isize, csa as isize,
                    bp.get(), rsb as isize, csb as isize,
                    0.0,
                    cp.get().add(i0 * rsc), rsc as isize, csc as isize,
                );
            }
        });
    }
}

/// (sign byte → four signs in {−1, 0, 1}, flag nibble → four 0/1 masks)
fn luts() -> &'static ([[f32; 4]; 256], [[f32; 4]; 16]) {
    static LUTS: std::sync::OnceLock<([[f32; 4]; 256], [[f32; 4]; 16])> = std::sync::OnceLock::new();
    LUTS.get_or_init(|| {
        let mut signs = [[0.0f32; 4]; 256];
        for (b, row) in signs.iter_mut().enumerate() {
            for (j, s) in row.iter_mut().enumerate() {
                *s = match (b >> (2 * j)) & 3 {
                    0 => -1.0,
                    2 => 1.0,
                    _ => 0.0,
                };
            }
        }
        let mut flags = [[0.0f32; 4]; 16];
        for (n, row) in flags.iter_mut().enumerate() {
            for (j, m) in row.iter_mut().enumerate() {
                *m = ((n >> j) & 1) as f32;
            }
        }
        (signs, flags)
    })
}

/// A matrix kept as 8-bit codes with one scale per row (the container's int6 / int8 tables, exactly).
pub struct Quant {
    pub rows: usize,
    pub cols: usize,
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
}

impl Quant {
    pub fn bytes(&self) -> usize {
        self.codes.len() + 4 * self.scales.len()
    }

    pub fn to_dense(&self) -> Vec<f32> {
        self.codes.iter().enumerate().map(|(i, &q)| q as f32 * self.scales[i / self.cols]).collect()
    }

    /// Row r as f32 (for lookups like embeddings).
    pub fn row(&self, r: usize) -> Vec<f32> {
        let s = self.scales[r];
        self.codes[r * self.cols..(r + 1) * self.cols].iter().map(|&q| q as f32 * s).collect()
    }

    /// W[r] · x for one row.
    #[inline]
    pub fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        dot_i8(&self.codes[r * self.cols..(r + 1) * self.cols], x) * self.scales[r]
    }
}

/// Σ q[i] · x[i] with 8 accumulators so it vectorises.
#[inline]
pub(crate) fn dot_i8(q: &[i8], x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (cq, cx) = (q.chunks_exact(8), x.chunks_exact(8));
    let (rq, rx) = (cq.remainder(), cx.remainder());
    for (a, b) in cq.zip(cx) {
        for j in 0..8 {
            acc[j] += a[j] as f32 * b[j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (a, b) in rq.iter().zip(rx) {
        s += *a as f32 * b;
    }
    s
}

/// Rows per weight panel: five-value matrices are stored panel-major so a panel unpacks with contiguous writes.
pub const PANEL: usize = 16;

/// A weight matrix stored as five values per row: {0, ±lo, ±hi}.
///
/// Layout: for panel p (rows 16p … 16p+15) and column c, `signs[p · cols + c]` holds row 16p+j's code (sign + 1,
/// 0..=2) at bits 2j, and bit j of `flags[p · cols + c]` is set when that weight uses `hi`. 3 bits per weight, as
/// in the container; reordering at load lets a [k × 16] panel unpack 16 neighbouring values at a time.
pub struct FiveValue {
    /// Output features (a multiple of 16).
    pub rows: usize,
    /// Input features.
    pub cols: usize,
    pub signs: Vec<u32>,
    pub flags: Vec<u16>,
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
}

impl FiveValue {
    pub fn bytes(&self) -> usize {
        4 * self.signs.len() + 2 * self.flags.len() + 4 * (self.lo.len() + self.hi.len())
    }

    #[inline]
    fn weight(&self, r: usize, c: usize) -> f32 {
        let i = (r / PANEL) * self.cols + c;
        let j = r % PANEL;
        let mag = if (self.flags[i] >> j) & 1 == 1 { self.hi[r] } else { self.lo[r] };
        match (self.signs[i] >> (2 * j)) & 3 {
            0 => -mag,
            2 => mag,
            _ => 0.0,
        }
    }

    /// Rows [r0, r1) as fp32 into `buf` ([r1 − r0] × cols). The portable (non-AVX2) path.
    fn expand_rows(&self, r0: usize, r1: usize, buf: &mut [f32]) {
        let k = self.cols;
        for (ri, r) in (r0..r1).enumerate() {
            for (c, out) in buf[ri * k..(ri + 1) * k].iter_mut().enumerate() {
                *out = self.weight(r, c);
            }
        }
    }

    /// `panel[kk · 16 + j] = W[n0 + j][k0 + kk]` for kk < kc (n0 a multiple of 16).
    ///
    /// Four weights at a time through lookup tables: a sign byte gives four signs, a flag nibble four 0/1 masks, and
    /// w = sign · (lo · (1 − m) + hi · m), which is exactly ±lo or ±hi and vectorises.
    pub fn fill_panel(&self, n0: usize, k0: usize, kc: usize, panel: &mut [f32]) {
        let (signs_lut, flags_lut) = luts();
        let lo: &[f32; PANEL] = self.lo[n0..n0 + PANEL].try_into().unwrap();
        let hi: &[f32; PANEL] = self.hi[n0..n0 + PANEL].try_into().unwrap();
        let base = (n0 / PANEL) * self.cols + k0;
        let signs = &self.signs[base..base + kc];
        let flags = &self.flags[base..base + kc];
        for (kk, out) in panel[..kc * PANEL].chunks_exact_mut(PANEL).enumerate() {
            let (s, f) = (signs[kk], flags[kk]);
            for b in 0..4 {
                let sl = &signs_lut[((s >> (8 * b)) & 255) as usize];
                let ml = &flags_lut[((f >> (4 * b)) & 15) as usize];
                for j in 0..4 {
                    let r = 4 * b + j;
                    out[r] = sl[j] * (lo[r] * (1.0 - ml[j]) + hi[r] * ml[j]);
                }
            }
        }
    }

    pub fn to_dense(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.cols];
        self.expand_rows(0, self.rows, &mut out);
        out
    }

    /// y[t × rows] = x[t × cols] · Wᵀ
    fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let (n, k) = (self.rows, self.cols);
        debug_assert!(x.len() >= t * k && y.len() >= t * n);
        let (xp, yp) = (CPtr::new(x.as_ptr()), Ptr::new(y.as_mut_ptr()));
        (0..n.div_ceil(BLOCK)).into_par_iter().for_each_init(
            || vec![0.0f32; BLOCK * k],
            |buf, bi| {
                let r0 = bi * BLOCK;
                let nb = BLOCK.min(n - r0);
                self.expand_rows(r0, r0 + nb, buf);
                // SAFETY: block bi writes columns [r0, r0 + nb) of y only.
                unsafe {
                    matrixmultiply::sgemm(
                        t, k, nb, 1.0,
                        xp.get(), k as isize, 1,
                        buf.as_ptr(), 1, k as isize,
                        0.0,
                        yp.get().add(r0), n as isize, 1,
                    );
                }
            },
        );
    }
}

#[cfg(target_arch = "x86_64")]
impl crate::kernel::Panels for FiveValue {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn fill(&self, n0: usize, k0: usize, kc: usize, panel: &mut [f32]) {
        self.fill_panel(n0, k0, kc, panel);
    }
}

/// A plain fp32 [out × in] matrix as a panel source.
#[cfg(target_arch = "x86_64")]
struct DensePanels<'a> {
    w: &'a [f32],
    out: usize,
    inp: usize,
}

#[cfg(target_arch = "x86_64")]
impl crate::kernel::Panels for DensePanels<'_> {
    fn rows(&self) -> usize {
        self.out
    }
    fn cols(&self) -> usize {
        self.inp
    }
    fn fill(&self, n0: usize, k0: usize, kc: usize, panel: &mut [f32]) {
        use crate::kernel::NR;
        for nn in 0..NR {
            let row = &self.w[(n0 + nn) * self.inp + k0..(n0 + nn) * self.inp + k0 + kc];
            for (kk, &v) in row.iter().enumerate() {
                panel[kk * NR + nn] = v;
            }
        }
    }
}

/// A linear layer without bias: five-value or plain fp32 ([out × in], row per output).
pub enum Linear {
    Dense { w: Vec<f32>, out: usize, inp: usize },
    Five(FiveValue),
}

impl Linear {
    pub fn out_features(&self) -> usize {
        match self {
            Linear::Dense { out, .. } => *out,
            Linear::Five(f) => f.rows,
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Linear::Dense { w, .. } => 4 * w.len(),
            Linear::Five(f) => f.bytes(),
        }
    }

    /// y[t × out] = x[t × in] · Wᵀ
    pub fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let start = std::time::Instant::now();
        self.forward_inner(x, t, y);
        crate::profile::add(crate::profile::LINEAR, start);
    }

    fn forward_inner(&self, x: &[f32], t: usize, y: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if crate::kernel::available() && std::env::var_os("PHONON_GENERIC").is_none() {
            use crate::kernel::{matmul, NR};
            match self {
                Linear::Dense { w, out, inp } if out % NR == 0 && inp % 8 == 0 => {
                    return matmul(&DensePanels { w, out: *out, inp: *inp }, x, t, y);
                }
                Linear::Five(f) if f.rows % NR == 0 => return matmul(f, x, t, y),
                _ => {}
            }
        }
        match self {
            Linear::Dense { w, out, inp } => gemm(t, *inp, *out, x, *inp, 1, w, 1, *inp, y, *out, 1),
            Linear::Five(f) => f.forward(x, t, y),
        }
    }
}

pub(crate) fn add_bias(y: &mut [f32], bias: &[f32]) {
    y.par_chunks_mut(bias.len()).for_each(|row| {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    });
}

/// Sigmoid with the exponent clamped so it never overflows (as the reference engine does).
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (if x < -87.0 { 87.0 } else { -x }).exp())
}

#[inline]
pub(crate) fn swish(x: f32) -> f32 {
    x * sigmoid(x)
}

pub(crate) fn softmax(row: &mut [f32]) {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

