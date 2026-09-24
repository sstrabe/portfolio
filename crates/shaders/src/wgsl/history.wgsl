// ---------------------------------------------------------------------------
// Worldline history mirrored from `crates/kerr/src/history.rs`.
//
// `history[(slot * bodies + body) * 2 + {0: position, 1: velocity}]`, with
// ring slots 0..capacity-1 on the shared grid t_newest − k·dt and one extra
// slot (index = capacity) holding the live state at t = 0.
// ---------------------------------------------------------------------------

@group(0) @binding(1) var<storage, read> history: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bodies: array<BodyMeta>;

struct BodySample {
    ok: bool,
    pos: vec3<f32>,
    vel: vec3<f32>,
}

fn hist_load(slot: u32, body: u32) -> BodySample {
    let i = (slot * frame.counts.x + body) * 2u;
    return BodySample(true, history[i].xyz, history[i + 1u].xyz);
}

fn hermite(a: BodySample, b: BodySample, h: f32, s: f32) -> BodySample {
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    let d00 = 6.0 * s2 - 6.0 * s;
    let d10 = 3.0 * s2 - 4.0 * s + 1.0;
    let d11 = 3.0 * s2 - 2.0 * s;
    let pos = h00 * a.pos + h10 * h * a.vel + h01 * b.pos + h11 * h * b.vel;
    let vel = (d00 * a.pos - d00 * b.pos) / h + d10 * a.vel + d11 * b.vel;
    return BodySample(true, pos, vel);
}

// Position and coordinate velocity of `body` at time `t` (relative).
fn body_at(body: u32, t: f32) -> BodySample {
    var none: BodySample;
    none.ok = false;
    let bm = bodies[body];
    if (t < bm.a.x || t > bm.a.y || t > 0.0) {
        return none;
    }
    let cap = frame.counts.y;
    let t_newest = frame.hist.x;
    if (t > t_newest) {
        let a = hist_load(frame.counts.z, body);
        let b = hist_load(cap, body);
        let h = -t_newest;
        if (h < 1e-4) {
            return b;
        }
        return hermite(a, b, h, (t - t_newest) / h);
    }
    let age = (t_newest - t) / frame.hist.y;
    let k = u32(floor(age));
    if (k + 1u >= frame.counts.w) {
        return none;
    }
    let slot_new = (frame.counts.z + cap - k) % cap;
    let slot_old = (frame.counts.z + cap - k - 1u) % cap;
    return hermite(hist_load(slot_old, body), hist_load(slot_new, body), frame.hist.y, 1.0 - (age - f32(k)));
}
