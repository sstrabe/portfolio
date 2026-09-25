// ---------------------------------------------------------------------------
// Convolution of the scene with the point-spread function in frequency
// space, on a grid of the output downsampled by `pf.grid2.x` and zero
// padded (see `fft.rs`).
//
// The image is carried as CIE XYZ (each channel has its own PSF: the
// pattern scales with wavelength, which is where colour fringes come from),
// packed two complex numbers per texel: (X + iY, Z + i0). Radix-2 FFTs run
// in workgroup memory:
//   cs_down      scene → XYZ grid image (box average)
//   cs_rows_fwd  forward FFT of every image row (padding rows stay zero)
//   cs_cols      a pair of columns kx, −kx: forward FFT, × PSF spectrum,
//                inverse FFT. Separating X and Y from the packed transform
//                needs the value at −k, which is why columns come in pairs.
//                In kernel mode it stores the PSF's spectrum instead.
//   cs_rows_inv  inverse FFT of the image rows → the wide part of the PSF
//                convolution; also the luminance histogram for metering.
// The PSF is real and even, so its spectrum is real: three floats per
// texel. Its value at k = 0 is the kernel's total energy, which leaves
// η = 1 − K(0) for the sharp full-resolution core (`post.wgsl`).
// ---------------------------------------------------------------------------

struct FftJob {
    n: vec4<u32>,      // nx, ny, log₂ nx, log₂ ny
    valid: vec4<u32>,  // source width (row stride), source rows, mode (0 convolve, 1 kernel), rows kept
}

@group(0) @binding(4) var scene_tex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> down: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> spec_buf: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> kspec: array<vec4<f32>>;
@group(0) @binding(9) var wide_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(10) var<storage, read> twiddle: array<vec2<f32>>;
@group(0) @binding(11) var<uniform> job: FftJob;
@group(0) @binding(12) var<storage, read_write> histogram: array<atomic<u32>, 256>;

const FFT_MAX: u32 = 2048u;
const FFT_THREADS: u32 = 256u;

var<workgroup> row: array<vec4<f32>, 2048>;
var<workgroup> col_a: array<vec4<f32>, 1024>;
var<workgroup> col_b: array<vec4<f32>, 1024>;
var<workgroup> hist_local: array<atomic<u32>, 256>;

fn bitrev(i: u32, bits: u32) -> u32 {
    return reverseBits(i) >> (32u - bits);
}

// Two complex products (w · v.xy, w · v.zw).
fn cmul2(w: vec2<f32>, v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(w.x * v.x - w.y * v.y, w.x * v.y + w.y * v.x, w.x * v.z - w.y * v.w, w.x * v.w + w.y * v.z);
}

fn tw(j: u32, m: u32, inverse: bool) -> vec2<f32> {
    let w = twiddle[j * (FFT_MAX / m)];
    return select(w, vec2<f32>(w.x, -w.y), inverse);
}

@compute @workgroup_size(8, 8)
fn cs_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pf.grid.z;
    let h = pf.grid.w;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let s = pf.grid2.x;
    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var dy = 0u; dy < s; dy++) {
        for (var dx = 0u; dx < s; dx++) {
            let p = gid.xy * s + vec2<u32>(dx, dy);
            if (p.x < pf.sizes.x && p.y < pf.sizes.y) {
                sum += textureLoad(scene_tex, p, 0).rgb;
                n += 1.0;
            }
        }
    }
    down[gid.y * w + gid.x] = vec4<f32>(rgb_to_xyz(sum / max(n, 1.0)), 0.0);
}

@compute @workgroup_size(256)
fn cs_rows_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nx = job.n.x;
    let bits = job.n.z;
    let y = wg.x;
    for (var i = li; i < nx; i += FFT_THREADS) {
        var v = vec4<f32>(0.0);
        if (i < job.valid.x) {
            v = vec4<f32>(src[y * job.valid.x + i].xyz, 0.0);
        }
        row[bitrev(i, bits)] = v;
    }
    workgroupBarrier();
    for (var m = 2u; m <= nx; m <<= 1u) {
        let half = m >> 1u;
        for (var b = li; b < nx / 2u; b += FFT_THREADS) {
            let j = b % half;
            let i0 = (b / half) * m + j;
            let i1 = i0 + half;
            let t = cmul2(tw(j, m, false), row[i1]);
            let u = row[i0];
            row[i0] = u + t;
            row[i1] = u - t;
        }
        workgroupBarrier();
    }
    for (var i = li; i < nx; i += FFT_THREADS) {
        spec_buf[y * nx + i] = row[i];
    }
}

// One stage of butterflies over both columns.
fn col_stage(li: u32, ny: u32, m: u32, inverse: bool) {
    let half = m >> 1u;
    for (var b = li; b < ny / 2u; b += FFT_THREADS) {
        let j = b % half;
        let i0 = (b / half) * m + j;
        let i1 = i0 + half;
        let w = tw(j, m, inverse);
        let ta = cmul2(w, col_a[i1]);
        let ua = col_a[i0];
        col_a[i0] = ua + ta;
        col_a[i1] = ua - ta;
        let tb = cmul2(w, col_b[i1]);
        let ub = col_b[i0];
        col_b[i0] = ub + tb;
        col_b[i1] = ub - tb;
    }
}

@compute @workgroup_size(256)
fn cs_cols(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nx = job.n.x;
    let ny = job.n.y;
    let bits = job.n.w;
    let a = wg.x;
    let b = (nx - a) % nx;
    for (var i = li; i < ny; i += FFT_THREADS) {
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        if (i < job.valid.y) {
            va = spec_buf[i * nx + a];
            vb = spec_buf[i * nx + b];
        }
        let r = bitrev(i, bits);
        col_a[r] = va;
        col_b[r] = vb;
    }
    workgroupBarrier();
    for (var m = 2u; m <= ny; m <<= 1u) {
        col_stage(li, ny, m, false);
        workgroupBarrier();
    }

    if (job.valid.z == 1u) {
        // Kernel mode: the real, even spectrum of the XYZ kernels. With
        // P = FT(X + iY) and P' = P(−k): X̂ = (P + P'*)/2, Ŷ = (P − P'*)/2i;
        // their real parts are the symmetric parts below.
        for (var ky = li; ky < ny; ky += FFT_THREADS) {
            let kyp = (ny - ky) % ny;
            let p = col_a[ky];
            let q = col_b[kyp];
            let k = vec4<f32>(0.5 * (p.x + q.x), 0.5 * (p.y + q.y), 0.5 * (p.z + q.z), 0.0);
            kspec[ky * nx + a] = k;
            kspec[kyp * nx + b] = k;
        }
        return;
    }

    // Multiply by the PSF spectrum, at k and at −k (column b, row −ky).
    for (var ky = li; ky < ny; ky += FFT_THREADS) {
        let kyp = (ny - ky) % ny;
        let p = col_a[ky];
        let q = col_b[kyp];
        let k = kspec[ky * nx + a].xyz;
        let xh = vec2<f32>(0.5 * (p.x + q.x), 0.5 * (p.y - q.y));
        let yh = vec2<f32>(0.5 * (p.y + q.y), -0.5 * (p.x - q.x));
        // X̂ K_X + i Ŷ K_Y, and its mirror X̂* K_X + i Ŷ* K_Y.
        col_a[ky] = vec4<f32>(xh.x * k.x - yh.y * k.y, xh.y * k.x + yh.x * k.y, p.zw * k.z);
        col_b[kyp] = vec4<f32>(xh.x * k.x + yh.y * k.y, -xh.y * k.x + yh.x * k.y, q.zw * k.z);
    }
    workgroupBarrier();
    // Back to bit-reversed order for the inverse transform.
    for (var i = li; i < ny; i += FFT_THREADS) {
        let j = bitrev(i, bits);
        if (j > i) {
            let ta = col_a[i];
            col_a[i] = col_a[j];
            col_a[j] = ta;
            let tb = col_b[i];
            col_b[i] = col_b[j];
            col_b[j] = tb;
        }
    }
    workgroupBarrier();
    for (var m = 2u; m <= ny; m <<= 1u) {
        col_stage(li, ny, m, true);
        workgroupBarrier();
    }
    for (var i = li; i < job.valid.w; i += FFT_THREADS) {
        spec_buf[i * nx + a] = col_a[i];
        if (b != a) {
            spec_buf[i * nx + b] = col_b[i];
        }
    }
}

@compute @workgroup_size(256)
fn cs_rows_inv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let nx = job.n.x;
    let ny = job.n.y;
    let bits = job.n.z;
    let y = wg.x;
    atomicStore(&hist_local[li], 0u);
    for (var i = li; i < nx; i += FFT_THREADS) {
        row[bitrev(i, bits)] = spec_buf[y * nx + i];
    }
    workgroupBarrier();
    for (var m = 2u; m <= nx; m <<= 1u) {
        let half = m >> 1u;
        for (var b = li; b < nx / 2u; b += FFT_THREADS) {
            let j = b % half;
            let i0 = (b / half) * m + j;
            let i1 = i0 + half;
            let t = cmul2(tw(j, m, true), row[i1]);
            let u = row[i0];
            row[i0] = u + t;
            row[i1] = u - t;
        }
        workgroupBarrier();
    }
    let scale = 1.0 / (f32(nx) * f32(ny));
    let eta_y = 1.0 - kspec[0].y;
    let w = job.valid.x;
    // Centre weighting of the meter (integer weights 1–8).
    let aspect = f32(w) / f32(job.valid.y);
    let vy = f32(y) / f32(job.valid.y) - 0.5;
    for (var x = li; x < w; x += FFT_THREADS) {
        let v = row[x] * scale;
        let xyz = vec3<f32>(v.x, v.y, v.z);
        textureStore(wide_out, vec2<u32>(x, y), vec4<f32>(xyz, 0.0));
        // What the eye or the meter sees: the sharp core plus the wings.
        let lum = eta_y * down[y * w + x].y + xyz.y;
        let bin = clamp((log2(max(lum, 1e-30)) - HIST_MIN) / HIST_RANGE * f32(HIST_BINS), 0.0, f32(HIST_BINS - 1u));
        let vx = (f32(x) / f32(w) - 0.5) * aspect;
        let weight = 1u + u32(7.0 * exp(-(vx * vx + vy * vy) / (2.0 * 0.3 * 0.3)) + 0.5);
        atomicAdd(&hist_local[u32(bin)], weight);
    }
    workgroupBarrier();
    let c = atomicLoad(&hist_local[li]);
    if (c > 0u) {
        atomicAdd(&histogram[li], c);
    }
}
