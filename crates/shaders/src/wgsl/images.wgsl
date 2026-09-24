// ---------------------------------------------------------------------------
// Image finder: for every (body, order) pair, solve for the ray on the
// pilot's past light cone that meets the body's worldline. Port of
// `crates/kerr/src/lensing.rs::find_image` (Newton on the closest-approach
// miss vector, swept angle selects the order), warm-started from the
// previous frame so a couple of iterations per frame suffice.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read_write> images: array<ImageState>;

struct Approach {
    ok: bool,
    miss: vec3<f32>,
    t: f32,
    pos: vec3<f32>,
    bvel: vec3<f32>,
    p: vec4<f32>,
    path: f32,
}

fn approach(body: u32, n: vec3<f32>, w0: f32, w1: f32) -> Approach {
    var best: Approach;
    best.ok = false;
    var best_d = 1e30;
    var s = backward_ray(n);
    var psi = 0.0;
    var path = 0.0;
    var have_prev = false;
    var prev_d = vec3<f32>(0.0);
    var prev_s: BodySample;
    let rp = frame.kerr.z;
    let r_esc = max(frame.kerr.w, 2.5 * length(frame.obs.yzw));
    for (var i = 0u; i < frame.counts2.x; i++) {
        let pos = s.x.yzw;
        let r = ks_radius(pos);
        if (r - rp < frame.march.w) {
            break;
        }
        let k1 = phase_rhs(s);
        let h = ray_step(s, k1, r);
        let nx = rk4(s, k1, h);
        let npos = nx.x.yzw;
        let dpsi = atan2(length(cross(pos, npos)), dot(pos, npos));
        let seg = length(npos - pos);
        let psi1 = psi + dpsi;
        if (nx.x.x < frame.hist.w || psi > w1) {
            break;
        }
        var keep_prev = false;
        if (psi1 >= w0) {
            let sb = body_at(body, nx.x.x);
            if (sb.ok) {
                var da = prev_d;
                var sa = prev_s;
                var have_a = have_prev;
                if (!have_a) {
                    let s0 = body_at(body, s.x.x);
                    if (s0.ok) {
                        da = pos - s0.pos;
                        sa = s0;
                        have_a = true;
                    }
                }
                let db = npos - sb.pos;
                if (have_a) {
                    let delta = db - da;
                    let dd = dot(delta, delta);
                    var u = 0.0;
                    if (dd > 0.0) {
                        u = clamp(-dot(da, delta) / dd, 0.0, 1.0);
                    }
                    let miss = da + u * delta;
                    let dist = length(miss);
                    if (dist < best_d) {
                        best_d = dist;
                        best = Approach(
                            true,
                            miss,
                            mix(s.x.x, nx.x.x, u),
                            mix(pos, npos, u),
                            mix(sa.vel, sb.vel, u),
                            mix(s.p, nx.p, u),
                            path + u * seg,
                        );
                    }
                }
                prev_d = db;
                prev_s = sb;
                keep_prev = true;
                if (best.ok && length(db) > 3.0 * best_d + 20.0 && psi1 > w0 + 0.2) {
                    break;
                }
            }
        }
        have_prev = keep_prev;
        psi = psi1;
        path += seg;
        s = nx;
        if (r > r_esc && dot(pos, k1.x.yzw) > 0.0) {
            break;
        }
    }
    return best;
}

// Ship-frame direction of a coordinate direction seen along a past null ray.
fn local_dir(d: vec3<f32>) -> vec3<f32> {
    let v = vec4<f32>(-1.0, normalize(d));
    let o = frame.obs.yzw;
    return normalize(vec3<f32>(mdot(o, v, frame.e1), mdot(o, v, frame.e2), mdot(o, v, frame.e3)));
}

fn coord_to_local(v: vec3<f32>) -> vec3<f32> {
    let w = vec4<f32>(0.0, v);
    let o = frame.obs.yzw;
    return normalize(vec3<f32>(mdot(o, w, frame.e1), mdot(o, w, frame.e2), mdot(o, w, frame.e3)));
}

fn local_to_coord(n: vec3<f32>) -> vec3<f32> {
    return n.x * frame.e1.yzw + n.y * frame.e2.yzw + n.z * frame.e3.yzw;
}

fn any_orth(n: vec3<f32>) -> vec3<f32> {
    if (abs(n.x) < 0.8) {
        return normalize(cross(n, vec3<f32>(1.0, 0.0, 0.0)));
    }
    return normalize(cross(n, vec3<f32>(0.0, 1.0, 0.0)));
}

fn rotate(v: vec3<f32>, axis: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

fn initial_guess(body: u32, order: u32) -> vec3<f32> {
    let bp = history[(frame.counts.y * frame.counts.x + body) * 2u].xyz;
    let op = frame.obs.yzw;
    let direct = local_dir(bp - op);
    if (order == 0u) {
        return direct;
    }
    let to_hole = local_dir(-op);
    let r_obs = length(op);
    let r_src = length(bp);
    let shadow = asin(min(5.3 * frame.kerr.x / r_obs, 1.0));
    let einstein = sqrt(4.0 * frame.kerr.x * r_src / (r_obs * (r_obs + r_src)));
    let alpha = max(1.08 * shadow, einstein);
    var axis = cross(to_hole, direct);
    if (length(axis) < 1e-6) {
        axis = any_orth(to_hole);
    }
    return rotate(to_hole, normalize(axis), -alpha);
}

@compute @workgroup_size(64)
fn cs_images(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= frame.counts.x * 2u) {
        return;
    }
    let body = idx / 2u;
    let order = idx % 2u;
    let bm = bodies[body];
    var st = images[idx];
    if (st.warm.w != bm.b.z) {
        st.dir.w = 0.0;
    }
    // Long gone, or not yet anywhere on the past light cone.
    if (bm.a.y < frame.hist.w || bm.a.x > 0.0) {
        st.dir.w = 0.0;
        st.warm.w = bm.b.z;
        images[idx] = st;
        return;
    }
    let w0 = f32(order) * PI;
    let w1 = w0 + PI;
    var n: vec3<f32>;
    if (st.dir.w > 0.5) {
        n = coord_to_local(st.warm.xyz);
    } else {
        n = initial_guess(body, order);
    }
    var a0 = approach(body, n, w0, w1);
    var ok = false;
    var j1 = vec3<f32>(0.0);
    var j2 = vec3<f32>(0.0);
    let eps = 3.0e-4;
    for (var it = 0u; it < frame.counts2.z; it++) {
        if (!a0.ok) {
            break;
        }
        let t1 = any_orth(n);
        let t2 = cross(n, t1);
        let a1 = approach(body, normalize(n + eps * t1), w0, w1);
        let a2 = approach(body, normalize(n + eps * t2), w0, w1);
        if (!a1.ok || !a2.ok) {
            a0.ok = false;
            break;
        }
        j1 = (a1.miss - a0.miss) / eps;
        j2 = (a2.miss - a0.miss) / eps;
        let tol = 0.02 + 5.0e-4 * a0.path;
        if (length(a0.miss) < tol) {
            ok = true;
            break;
        }
        let a11 = dot(j1, j1);
        let a12 = dot(j1, j2);
        let a22 = dot(j2, j2);
        let det = a11 * a22 - a12 * a12;
        if (abs(det) < 1e-20) {
            a0.ok = false;
            break;
        }
        let b1 = -dot(j1, a0.miss);
        let b2 = -dot(j2, a0.miss);
        var d = vec2<f32>(a22 * b1 - a12 * b2, a11 * b2 - a12 * b1) / det;
        let turn = length(d);
        if (turn > 0.2) {
            d *= 0.2 / turn;
        }
        n = normalize(n + d.x * t1 + d.y * t2);
        a0 = approach(body, n, w0, w1);
    }
    if (!ok && a0.ok && length(j1) > 0.0 && length(a0.miss) < 0.02 + 5.0e-4 * a0.path) {
        ok = true;
    }
    st.warm.w = bm.b.z;
    if (!ok) {
        st.dir.w = 0.0;
        images[idx] = st;
        return;
    }
    let w = four_velocity(a0.pos, a0.bvel);
    let g = 1.0 / max(dot(a0.p, w), 1e-6);
    let area = max(length(cross(j1, j2)), 1e-6);
    let g2 = g * g;
    st.dir = vec4<f32>(n, 1.0);
    st.info = vec4<f32>(g, bm.a.w * g2 * g2 / area, a0.t, length(a0.miss));
    st.warm = vec4<f32>(local_to_coord(n), bm.b.z);
    images[idx] = st;
}
