//! AVX2 + FMA matmul for the encoder's shape: few frames (rows of x), large weight matrices.
//!
//! y[t × N] = x[t × K] · Wᵀ, W given as a source of [16 × KC] weight panels (unpacked five-value or plain fp32).
//! Threads own ranges of output columns. Each walks K in blocks of KC: it fills a 16-column panel (KC × 16, k-major,
//! 16 KB, so it stays in L1) and runs a 6×16 register-blocked FMA kernel over the frames, accumulating into y.
//! x is read once per K block per thread, so it is served from L2 rather than refetched per panel.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use rayon::prelude::*;

use crate::linear::{CPtr, Ptr};

pub const NR: usize = 16; // output columns per panel
pub const KC: usize = 256; // K block
const MR: usize = 6; // frames per register tile

/// Fills `panel[kk * 16 + nn] = W[n0 + nn][k0 + kk]` for kk < kc.
pub trait Panels: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn fill(&self, n0: usize, k0: usize, kc: usize, panel: &mut [f32]);
}

pub fn available() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

/// y[t × N] = x[t × K] · Wᵀ. Requires N % 16 == 0, K % 8 == 0 and `available()`.
pub fn matmul(w: &dyn Panels, x: &[f32], t: usize, y: &mut [f32]) {
    let (n, k) = (w.rows(), w.cols());
    debug_assert!(n % NR == 0 && k % 8 == 0 && x.len() >= t * k && y.len() >= t * n);
    if t == 0 {
        return;
    }
    // Enough tasks to fill the cores, few enough that each reuses its x blocks across several panels.
    let task_cols = if n >= 2048 { 128 } else { 64 };
    let (xp, yp) = (CPtr::new(x.as_ptr()), Ptr::new(y.as_mut_ptr()));
    (0..n.div_ceil(task_cols)).into_par_iter().for_each_init(
        || vec![0.0f32; KC * NR],
        |panel, task| {
            let c0 = task * task_cols;
            let c1 = (c0 + task_cols).min(n);
            let mut k0 = 0;
            while k0 < k {
                let kc = KC.min(k - k0);
                let mut n0 = c0;
                while n0 < c1 {
                    w.fill(n0, k0, kc, panel);
                    // SAFETY: AVX2/FMA checked by the caller; this task writes columns [c0, c1) of y only.
                    unsafe { panel_times_frames(xp.get().add(k0), k, panel.as_ptr(), kc, yp.get().add(n0), n, t, k0 > 0) };
                    n0 += NR;
                }
                k0 += kc;
            }
        },
    );
}

#[target_feature(enable = "avx2,fma")]
unsafe fn panel_times_frames(x: *const f32, ldx: usize, panel: *const f32, kc: usize, y: *mut f32, ldy: usize, t: usize, accumulate: bool) {
    let mut r = 0;
    while r + MR <= t {
        tile::<MR>(x.add(r * ldx), ldx, panel, kc, y.add(r * ldy), ldy, accumulate);
        r += MR;
    }
    match t - r {
        1 => tile::<1>(x.add(r * ldx), ldx, panel, kc, y.add(r * ldy), ldy, accumulate),
        2 => tile::<2>(x.add(r * ldx), ldx, panel, kc, y.add(r * ldy), ldy, accumulate),
        3 => tile::<3>(x.add(r * ldx), ldx, panel, kc, y.add(r * ldy), ldy, accumulate),
        4 => tile::<4>(x.add(r * ldx), ldx, panel, kc, y.add(r * ldy), ldy, accumulate),
        5 => tile::<5>(x.add(r * ldx), ldx, panel, kc, y.add(r * ldy), ldy, accumulate),
        _ => {}
    }
}

/// R frames × 16 columns, accumulated in 2R ymm registers over kc steps.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile<const R: usize>(x: *const f32, ldx: usize, panel: *const f32, kc: usize, y: *mut f32, ldy: usize, accumulate: bool) {
    let mut acc = [[_mm256_setzero_ps(); 2]; R];
    if accumulate {
        for (i, a) in acc.iter_mut().enumerate() {
            a[0] = _mm256_loadu_ps(y.add(i * ldy));
            a[1] = _mm256_loadu_ps(y.add(i * ldy + 8));
        }
    }
    for kk in 0..kc {
        let w0 = _mm256_loadu_ps(panel.add(kk * NR));
        let w1 = _mm256_loadu_ps(panel.add(kk * NR + 8));
        for (i, a) in acc.iter_mut().enumerate() {
            let xv = _mm256_set1_ps(*x.add(i * ldx + kk));
            a[0] = _mm256_fmadd_ps(xv, w0, a[0]);
            a[1] = _mm256_fmadd_ps(xv, w1, a[1]);
        }
    }
    for (i, a) in acc.iter().enumerate() {
        _mm256_storeu_ps(y.add(i * ldy), a[0]);
        _mm256_storeu_ps(y.add(i * ldy + 8), a[1]);
    }
}
