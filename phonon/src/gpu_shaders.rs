//! WGSL compute kernels for the GPU encoder. All activations are row-major f32.

/// y[t × n] = x[t × k] · Wᵀ (five-value, unpacked in shared memory) or dense W [n × k] + bias.
/// Workgroup: 16 output columns (one weight panel) × 64 frames; each thread accumulates 4 frames.
/// mode 0: y = x·Wᵀ (+ bias); mode 1: y += alpha · x·Wᵀ (residual).
pub const LINEAR: &str = r#"
struct Params { t: u32, n: u32, k: u32, mode: u32, alpha: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<u32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read> lohi: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<storage, read> wd: array<f32>;
@group(0) @binding(7) var<storage, read> bias: array<f32>;

const KC: u32 = 32u;
const XS: u32 = 33u;
const TT: u32 = 64u;

var<workgroup> ws: array<f32, 512>;
var<workgroup> xs: array<f32, 2112>;

fn load_x(tid: u32, t0: u32, k0: u32) {
    for (var j = 0u; j < 8u; j++) {
        let e = tid + 256u * j;
        let r = e / KC;
        let kk = e % KC;
        let t = t0 + r;
        var v = 0.0;
        if (t < p.t) { v = x[t * p.k + k0 + kk]; }
        xs[r * XS + kk] = v;
    }
}

fn store(n: u32, tl: u32, t0: u32, acc_in: array<f32, 4>, b: f32) {
    var acc = acc_in;
    for (var i = 0u; i < 4u; i++) {
        let t = t0 + tl + 16u * i;
        if (t < p.t) {
            let idx = t * p.n + n;
            if (p.mode == 0u) { y[idx] = acc[i] + b; } else { y[idx] = y[idx] + p.alpha * acc[i]; }
        }
    }
}

@compute @workgroup_size(16, 16)
fn five(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let panel = wg.x;
    let t0 = wg.y * TT;
    let nn = lid.x;
    let tl = lid.y;
    let tid = tl * 16u + nn;
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var k0 = 0u; k0 < p.k; k0 += KC) {
        for (var j = 0u; j < 2u; j++) {
            let e = tid + 256u * j;
            let kk = e / 16u;
            let c = e % 16u;
            let idx = panel * p.k + k0 + kk;
            let code = (signs[idx] >> (2u * c)) & 3u;
            let f = (flags[idx >> 1u] >> (16u * (idx & 1u) + c)) & 1u;
            let row = panel * 16u + c;
            let mag = select(lohi[row], lohi[p.n + row], f == 1u);
            var w = 0.0;
            if (code == 0u) { w = -mag; } else if (code == 2u) { w = mag; }
            ws[kk * 16u + c] = w;
        }
        load_x(tid, t0, k0);
        workgroupBarrier();
        for (var kk = 0u; kk < KC; kk++) {
            let w = ws[kk * 16u + nn];
            for (var i = 0u; i < 4u; i++) {
                acc[i] += xs[(tl + 16u * i) * XS + kk] * w;
            }
        }
        workgroupBarrier();
    }
    store(panel * 16u + nn, tl, t0, acc, 0.0);
}

@compute @workgroup_size(16, 16)
fn dense(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let panel = wg.x;
    let t0 = wg.y * TT;
    let nn = lid.x;
    let tl = lid.y;
    let tid = tl * 16u + nn;
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var k0 = 0u; k0 < p.k; k0 += KC) {
        for (var j = 0u; j < 2u; j++) {
            let e = tid + 256u * j;
            let kk = e / 16u;
            let c = e % 16u;
            ws[kk * 16u + c] = wd[(panel * 16u + c) * p.k + k0 + kk];
        }
        load_x(tid, t0, k0);
        workgroupBarrier();
        for (var kk = 0u; kk < KC; kk++) {
            let w = ws[kk * 16u + nn];
            for (var i = 0u; i < 4u; i++) {
                acc[i] += xs[(tl + 16u * i) * XS + kk] * w;
            }
        }
        workgroupBarrier();
    }
    let n = panel * 16u + nn;
    store(n, tl, t0, acc, bias[n]);
}
"#;

/// LayerNorm (eps 1e-5) of each row: one workgroup per row.
pub const NORM: &str = r#"
struct Params { t: u32, d: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let base = wg.x * p.d;
    let i = lid.x;
    var s = 0.0;
    for (var c = i; c < p.d; c += 256u) { s += x[base + c]; }
    red[i] = s;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w >>= 1u) {
        if (i < w) { red[i] += red[i + w]; }
        workgroupBarrier();
    }
    let mean = red[0] / f32(p.d);
    workgroupBarrier();
    var v = 0.0;
    for (var c = i; c < p.d; c += 256u) { let dl = x[base + c] - mean; v += dl * dl; }
    red[i] = v;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w >>= 1u) {
        if (i < w) { red[i] += red[i + w]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(p.d) + 1e-5);
    for (var c = i; c < p.d; c += 256u) {
        y[base + c] = (x[base + c] - mean) * inv * gamma[c] + beta[c];
    }
}
"#;

/// x · sigmoid(x) in place (sigmoid exponent clamped as on the CPU).
pub const SWISH: &str = r#"
struct Params { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < p.n) {
        let v = h[i];
        h[i] = v / (1.0 + exp(min(-v, 87.0)));
    }
}
"#;

/// Relative-position attention for one (query frame, head) per workgroup: scores, softmax, weighted sum of V.
pub const ATTENTION: &str = r#"
struct Params { t: u32, p0: u32, p1: u32, p2: u32, scale: f32, p3: u32, p4: u32, p5: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> pos: array<f32>;
@group(0) @binding(5) var<storage, read> bu: array<f32>;
@group(0) @binding(6) var<storage, read> bv: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<f32>;

const D: u32 = 1024u;
const DH: u32 = 128u;

var<workgroup> qu: array<f32, 128>;
var<workgroup> qv: array<f32, 128>;
var<workgroup> sc: array<f32, 512>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wg.x;
    let off = wg.y * DH;
    let dd = lid.x;
    let tn = p.t;
    let qval = q[i * D + off + dd];
    qu[dd] = qval + bu[off + dd];
    qv[dd] = qval + bv[off + dd];
    workgroupBarrier();

    // score(i, j) = (q_u·k_j + q_v·p_{t−1−i+j}) / sqrt(DH)   (Transformer-XL relative shift)
    for (var j = dd; j < tn; j += 128u) {
        var a = 0.0;
        var b = 0.0;
        let kb = j * D + off;
        let pb = (tn - 1u - i + j) * D + off;
        for (var e = 0u; e < DH; e++) {
            a += qu[e] * k[kb + e];
            b += qv[e] * pos[pb + e];
        }
        sc[j] = (a + b) * p.scale;
    }
    workgroupBarrier();

    var m = -3.4e38;
    for (var j = dd; j < tn; j += 128u) { m = max(m, sc[j]); }
    red[dd] = m;
    workgroupBarrier();
    for (var w = 64u; w > 0u; w >>= 1u) {
        if (dd < w) { red[dd] = max(red[dd], red[dd + w]); }
        workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();

    var s = 0.0;
    for (var j = dd; j < tn; j += 128u) {
        let e = exp(sc[j] - mx);
        sc[j] = e;
        s += e;
    }
    red[dd] = s;
    workgroupBarrier();
    for (var w = 64u; w > 0u; w >>= 1u) {
        if (dd < w) { red[dd] += red[dd + w]; }
        workgroupBarrier();
    }
    let inv = 1.0 / red[0];

    var acc = 0.0;
    for (var j = 0u; j < tn; j++) { acc += sc[j] * v[j * D + off + dd]; }
    o[i * D + off + dd] = acc * inv;
}
"#;

/// Conformer conv module middle: GLU → depthwise conv (k = 9, BatchNorm folded) → swish, [t × 2D] → [t × D].
pub const GLU_DWCONV: &str = r#"
struct Params { t: u32, d: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pw: array<f32>;
@group(0) @binding(2) var<storage, read> dww: array<f32>;
@group(0) @binding(3) var<storage, read> dwb: array<f32>;
@group(0) @binding(4) var<storage, read_write> act: array<f32>;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(min(-x, 87.0))); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    if (idx >= p.t * p.d) { return; }
    let t = idx / p.d;
    let c = idx % p.d;
    var acc = dwb[c];
    for (var kk = 0u; kk < 9u; kk++) {
        let src = i32(t) + i32(kk) - 4;
        if (src >= 0 && src < i32(p.t)) {
            let base = u32(src) * 2u * p.d;
            acc += pw[base + c] * sigmoid(pw[base + p.d + c]) * dww[c * 9u + kk];
        }
    }
    act[idx] = acc * sigmoid(acc);
}
"#;
