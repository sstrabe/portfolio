// ---------------------------------------------------------------------------
// The pilot's own ship (hook called first by the trace in `trace.wgsl`).
//
// The ship moves with the camera, so it has no aberration or Doppler shift
// and plain perspective is exact for it. `n` is the pixel's ship-frame
// direction (forward, left, up).
//
// Placeholder: no ship.
// ---------------------------------------------------------------------------

struct ShipHit {
    hit: bool,
    L: Spectrum,
}

fn ship_trace(n: vec3<f32>) -> ShipHit {
    var h: ShipHit;
    h.hit = false;
    return h;
}
