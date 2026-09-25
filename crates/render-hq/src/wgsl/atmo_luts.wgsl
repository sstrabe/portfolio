// ---------------------------------------------------------------------------
// Atmosphere lookup tables, per near-field planet slot (dispatch z = slot),
// rebuilt when the set of planets changes (Hillaire 2020, spectral):
//
// - transmittance: density-weighted columns (km) of the gas, the aerosols
//   and ozone from (r, μ) to the top of the atmosphere. Transmittance is
//   exp(−Σ β_i(λ) C_i), so one rgba32float texel serves all 16 bins exactly
//   (Hillaire stores RGB transmittance; columns are cheaper and exact).
// - multiple scattering Ψ_ms(r, μ_s): second-order light with an isotropic
//   phase, summed over all orders as the geometric series 1/(1 − f_ms)
//   (Hillaire §5.5), per unit solar irradiance; 4 layers of 4 bins.
// - sky irradiance E(r, μ_s) on a horizontal surface: single scattering
//   plus Ψ_ms over the upper hemisphere, per unit solar irradiance.
// ---------------------------------------------------------------------------

@group(0) @binding(1) var<storage, read> lut_params: array<AtmosphereParams>;
@group(0) @binding(2) var trans_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(3) var trans_in: texture_2d_array<f32>;
@group(0) @binding(4) var ms_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var ms_in: texture_2d_array<f32>;
@group(0) @binding(6) var irr_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(9) var lut_sampler: sampler;

const TRANS_STEPS: u32 = 512u;

@compute @workgroup_size(8, 8)
fn cs_transmittance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.z;
    let g = atmo_geom_of(lut_params[slot]);
    if (!atmo_present(g)) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(ATMO_TRANS_W, ATMO_TRANS_H);
    let rm = atmo_trans_r_mu(g, uv);
    let r = rm.x;
    let mu = rm.y;
    let d = atmo_dist_to_sphere(r, mu, atmo_top(g));
    // Quadratically spaced samples: dense where the ray starts (the lowest
    // point of upward rays, where density falls fastest).
    var col = vec3<f32>(0.0);
    let n = f32(TRANS_STEPS);
    for (var i = 0u; i < TRANS_STEPS; i++) {
        let s0 = f32(i) / n;
        let s1 = f32(i + 1u) / n;
        let t = d * 0.5 * (s0 * s0 + s1 * s1);
        let dt = d * (s1 * s1 - s0 * s0);
        let rs = sqrt(max(r * r + t * t + 2.0 * r * mu * t, 0.0));
        col += atmo_density(g, rs - g.shape.x) * dt;
    }
    textureStore(trans_out, gid.xy, slot, vec4<f32>(col, 0.0));
}

fn lut_sun_transmittance(a: AtmosphereParams, slot: u32, r: f32, mu_s: f32) -> Spectrum {
    let g = atmo_geom_of(a);
    let uv = atmo_trans_uv(g, r, mu_s);
    let col = textureSampleLevel(trans_in, lut_sampler, uv, slot, 0.0).xyz;
    return spec_scale(atmo_transmit_columns(a, col), atmo_sun_above_horizon(g, r, mu_s, 0.004));
}

fn lut_ms(a: AtmosphereParams, slot: u32, r: f32, mu_s: f32) -> Spectrum {
    let uv = atmo_ms_uv(atmo_geom_of(a), r, mu_s, ATMO_MS_SIZE);
    let l = slot * 4u;
    return Spectrum(
        textureSampleLevel(ms_in, lut_sampler, uv, l, 0.0),
        textureSampleLevel(ms_in, lut_sampler, uv, l + 1u, 0.0),
        textureSampleLevel(ms_in, lut_sampler, uv, l + 2u, 0.0),
        textureSampleLevel(ms_in, lut_sampler, uv, l + 3u, 0.0),
    );
}

// (1 − exp(−σ Δ)) / σ per bin: the integral of exp(−σ s) over a step.
fn lut_step_weight(sigma: Spectrum, e: Spectrum) -> Spectrum {
    let s = spec_max(sigma, spec(1e-9));
    return Spectrum((1.0 - e.a) / s.a, (1.0 - e.b) / s.b, (1.0 - e.c) / s.c, (1.0 - e.d) / s.d);
}

// Direction k of 64 spread evenly over the sphere (spherical Fibonacci);
// `hemisphere` restricts it to z > 0.
fn lut_direction(k: u32, hemisphere: bool) -> vec3<f32> {
    let n = 64.0;
    var z = 1.0 - (2.0 * f32(k) + 1.0) / n;
    if (hemisphere) {
        z = 1.0 - (f32(k) + 0.5) / n;
    }
    let phi = f32(k) * 2.39996323;
    let s = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(s * cos(phi), s * sin(phi), z);
}

var<workgroup> wg_a: array<Spectrum, 64>;
var<workgroup> wg_b: array<Spectrum, 64>;

fn lut_reduce(li: u32) {
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        workgroupBarrier();
        if (li < stride) {
            wg_a[li] = spec_add(wg_a[li], wg_a[li + stride]);
            wg_b[li] = spec_add(wg_b[li], wg_b[li + stride]);
        }
    }
    workgroupBarrier();
}

const MS_STEPS: u32 = 24u;

// One workgroup per texel, one direction per invocation.
@compute @workgroup_size(64)
fn cs_multiscatter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let slot = wid.z;
    let a = lut_params[slot];
    let g = atmo_geom_of(a);
    let present = atmo_present(g);
    var l2 = spec(0.0);
    var fms = spec(0.0);
    if (present) {
        let uv = (vec2<f32>(wid.xy) + 0.5) / ATMO_MS_SIZE;
        let rm = atmo_ms_r_mu(g, uv, ATMO_MS_SIZE);
        let r = rm.x;
        let sun = vec3<f32>(sqrt(max(1.0 - rm.y * rm.y, 0.0)), 0.0, rm.y);
        let x = vec3<f32>(0.0, 0.0, r);
        let w = lut_direction(li, false);
        let mu = w.z;
        let ground = atmo_hits_ground(g, r, mu);
        var t_max = atmo_dist_to_sphere(r, mu, atmo_top(g));
        if (ground) {
            let rg = a.shape.x;
            t_max = -r * mu - sqrt(max((rg - r) * (rg + r) + r * r * mu * mu, 0.0));
        }
        let dt = t_max / f32(MS_STEPS);
        var tr = spec(1.0);
        let pu = 1.0 / (4.0 * PI);
        for (var i = 0u; i < MS_STEPS; i++) {
            let p = x + (f32(i) + 0.5) * dt * w;
            let rp = length(p);
            let d = atmo_density(g, rp - a.shape.x);
            let sig_t = atmo_extinction(a, d);
            let sig_s = spec_axpy(a.mie_scattering, d.y, spec_scale(a.rayleigh_scattering, d.x));
            let ts = lut_sun_transmittance(a, slot, rp, dot(p, sun) / rp);
            let e = spec_transmit(spec_scale(sig_t, dt));
            let k = spec_mul(tr, lut_step_weight(sig_t, e));
            l2 = spec_fma(k, spec_scale(spec_mul(sig_s, ts), pu), l2);
            fms = spec_fma(k, sig_s, fms);
            tr = spec_mul(tr, e);
        }
        if (ground) {
            let p = x + t_max * w;
            let rp = length(p);
            let mu_g = dot(p, sun) / rp;
            let ts = lut_sun_transmittance(a, slot, rp, mu_g);
            l2 = spec_fma(tr, spec_scale(ts, max(mu_g, 0.0) * a.extra.x / PI), l2);
        }
    }
    wg_a[li] = l2;
    wg_b[li] = fms;
    lut_reduce(li);
    if (li == 0u && present) {
        // Isotropic phase: the sphere integral of L p_u is the mean over
        // directions.
        let l = spec_scale(wg_a[0], 1.0 / 64.0);
        let f = spec_scale(wg_b[0], 1.0 / 64.0);
        let one = spec(1.0);
        let psi = Spectrum(
            l.a / max(one.a - f.a, vec4<f32>(1e-3)),
            l.b / max(one.b - f.b, vec4<f32>(1e-3)),
            l.c / max(one.c - f.c, vec4<f32>(1e-3)),
            l.d / max(one.d - f.d, vec4<f32>(1e-3)),
        );
        let base = slot * 4u;
        textureStore(ms_out, wid.xy, base, psi.a);
        textureStore(ms_out, wid.xy, base + 1u, psi.b);
        textureStore(ms_out, wid.xy, base + 2u, psi.c);
        textureStore(ms_out, wid.xy, base + 3u, psi.d);
    }
}

const IRR_STEPS: u32 = 24u;

@compute @workgroup_size(64)
fn cs_irradiance(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let slot = wid.z;
    let a = lut_params[slot];
    let g = atmo_geom_of(a);
    let present = atmo_present(g);
    var e_sky = spec(0.0);
    if (present) {
        let uv = (vec2<f32>(wid.xy) + 0.5) / ATMO_IRR_SIZE;
        let rm = atmo_ms_r_mu(g, uv, ATMO_IRR_SIZE);
        let r = rm.x;
        let sun = vec3<f32>(sqrt(max(1.0 - rm.y * rm.y, 0.0)), 0.0, rm.y);
        let x = vec3<f32>(0.0, 0.0, r);
        let w = lut_direction(li, true);
        let c = dot(w, sun);
        let pr = atmo_phase_rayleigh(c);
        let pm = atmo_phase_mie(g, c);
        let t_max = atmo_dist_to_sphere(r, w.z, atmo_top(g));
        // Quadratic spacing: horizontal rays cross the dense bottom first.
        var tr = spec(1.0);
        var l = spec(0.0);
        let n = f32(IRR_STEPS);
        for (var i = 0u; i < IRR_STEPS; i++) {
            let s0 = f32(i) / n;
            let s1 = f32(i + 1u) / n;
            let t = t_max * 0.5 * (s0 * s0 + s1 * s1);
            let dt = t_max * (s1 * s1 - s0 * s0);
            let p = x + t * w;
            let rp = length(p);
            let mu_s = dot(p, sun) / rp;
            let d = atmo_density(g, rp - a.shape.x);
            let sig_t = atmo_extinction(a, d);
            let ts = lut_sun_transmittance(a, slot, rp, mu_s);
            let ms = lut_ms(a, slot, rp, mu_s);
            let sr = spec_scale(a.rayleigh_scattering, d.x);
            let sm = spec_scale(a.mie_scattering, d.y);
            let src = spec_add(spec_mul(sr, spec_axpy(ts, pr, ms)), spec_mul(sm, spec_axpy(ts, pm, ms)));
            let e = spec_transmit(spec_scale(sig_t, dt));
            l = spec_fma(spec_mul(tr, lut_step_weight(sig_t, e)), src, l);
            tr = spec_mul(tr, e);
        }
        // Uniform hemisphere samples: E = 2π ⟨L cos θ⟩.
        e_sky = spec_scale(l, w.z * TAU / 64.0);
    }
    wg_a[li] = e_sky;
    wg_b[li] = spec(0.0);
    lut_reduce(li);
    if (li == 0u && present) {
        let base = slot * 4u;
        textureStore(irr_out, wid.xy, base, wg_a[0].a);
        textureStore(irr_out, wid.xy, base + 1u, wg_a[0].b);
        textureStore(irr_out, wid.xy, base + 2u, wg_a[0].c);
        textureStore(irr_out, wid.xy, base + 3u, wg_a[0].d);
    }
}
