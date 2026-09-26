// ---------------------------------------------------------------------------
// Terrain heights at given body-fixed directions, read back by the CPU
// (start positions, site finding, tests). The same functions as the trace
// (`terrain.wgsl`), so what the CPU learns is what gets drawn.
// ---------------------------------------------------------------------------

struct ProbeParams {
    kind: u32,
    seed: u32,
    count: u32,
    air: u32,      // 1 with an atmosphere
    radius: f32,   // km
    relief: f32,   // km
    sea: f32,      // fraction of the relief below the datum
    t_eq: f32,     // K
    lod: f32,      // footprint, km
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(1) var<uniform> probe: ProbeParams;
@group(0) @binding(2) var<storage, read> probe_dirs: array<vec4<f32>>;
// Per direction: (visible surface height, solid height, what fills basins,
// unused), the macro channels (see `terrain_macro`), then the hotspot
// islands' field (see `tn_hotspots`).
@group(0) @binding(3) var<storage, read_write> probe_out: array<vec4<f32>>;
// The regional erosion (`region.wgsl`), as the tiles have it.
@group(0) @binding(4) var<uniform> region: Region;
@group(0) @binding(5) var<storage, read> region_cells: array<vec2<f32>>;

@compute @workgroup_size(64)
fn cs_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= probe.count) {
        return;
    }
    let tp = terrain_params(probe.kind, probe.seed, probe.radius, probe.relief, probe.sea, probe.t_eq, probe.air != 0u);
    let q = normalize(probe_dirs[i].xyz);
    let m = terrain_macro(tp, q, probe.lod);
    let solid = terrain_solid(tp, q, m, probe.lod) + region_delta(q, probe.seed, probe.radius, probe.lod);
    probe_out[3u * i] = vec4<f32>(terrain_surface(tp, solid), solid, f32(tp.liquid), 0.0);
    probe_out[3u * i + 1u] = m;
    probe_out[3u * i + 2u] = vec4<f32>(tn_hotspots(tp, q), 0.0);
}
