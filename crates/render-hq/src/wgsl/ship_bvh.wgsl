// ---------------------------------------------------------------------------
// The ship's mesh traced in software through its BVH (`ship/bvh.rs`), for
// GPUs without ray-tracing hardware; `ship_rq.wgsl` answers the same calls
// with it.
// ---------------------------------------------------------------------------

// Möller–Trumbore: distance and barycentrics, or t < 0.
fn ship_triangle(o: vec3<f32>, d: vec3<f32>, tri: u32) -> vec3<f32> {
    let a = ship_vertex(ship_triangles[3u * tri]);
    let b = ship_vertex(ship_triangles[3u * tri + 1u]);
    let c = ship_vertex(ship_triangles[3u * tri + 2u]);
    let e1 = b - a;
    let e2 = c - a;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-9) {
        return vec3<f32>(-1.0);
    }
    let inv = 1.0 / det;
    let s = o - a;
    let u = dot(s, p) * inv;
    if (u < 0.0 || u > 1.0) {
        return vec3<f32>(-1.0);
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if (v < 0.0 || u + v > 1.0) {
        return vec3<f32>(-1.0);
    }
    return vec3<f32>(dot(e2, q) * inv, u, v);
}

fn ship_box(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, t_max: f32) -> bool {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let t0 = max(max(min(a.x, b.x), min(a.y, b.y)), max(min(a.z, b.z), 0.0));
    let t1 = min(min(max(a.x, b.x), max(a.y, b.y)), min(max(a.z, b.z), t_max));
    return t0 <= t1;
}

// Nearest hit before t_max (or, with `any`, the first one found).
fn ship_bvh(o: vec3<f32>, d: vec3<f32>, t_max: f32, any: bool) -> ShipRay {
    var best = ShipRay(t_max, 0xffffffffu, vec2<f32>(0.0));
    let inv = 1.0 / select(d, vec3<f32>(1e-12), abs(d) < vec3<f32>(1e-12));
    var stack: array<u32, 32>;
    var sp = 0u;
    var i = 0u;
    for (var guard = 0u; guard < 2048u; guard++) {
        let n = ship_nodes[i];
        var descend = false;
        if (ship_box(n.lo, n.hi, o, inv, best.t)) {
            if ((n.b & 3u) == 3u) {
                let count = n.b >> 2u;
                for (var k = 0u; k < count; k++) {
                    let h = ship_triangle(o, d, n.a + k);
                    if (h.x > 1e-4 && h.x < best.t) {
                        best = ShipRay(h.x, n.a + k, h.yz);
                        if (any) {
                            return best;
                        }
                    }
                }
            } else {
                // Visit the child on the ray's near side first.
                var first = i + 1u;
                var second = n.a;
                if (d[n.b] < 0.0) {
                    first = n.a;
                    second = i + 1u;
                }
                if (sp < 32u) {
                    stack[sp] = second;
                    sp++;
                }
                i = first;
                descend = true;
            }
        }
        if (!descend) {
            if (sp == 0u) {
                break;
            }
            sp--;
            i = stack[sp];
        }
    }
    return best;
}
