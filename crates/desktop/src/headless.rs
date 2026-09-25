//! Offscreen rendering to a PNG: `--headless WIDTHxHEIGHT`.

use crate::Options;
use crate::input::Controls;
use render_hq::{Gpu, Session};

pub fn run(o: &Options) -> Result<(), String> {
    let (w, h) = o.headless.expect("headless size");
    let mut world = crate::world(o.stars, o.start)?;
    if let Some(k) = o.near_star {
        park_near_star(&mut world, k)?;
    }
    if o.autopilot {
        let p = world.toggle_orbit_autopilot().ok_or("no planet for the autopilot")?;
        println!("autopilot: into orbit around {}", crate::hud::planet_name(&world, p));
    }
    let (n, cap) = (world.cluster.len() as u32, world.cluster.history.capacity() as u32);
    let gpu = pollster::block_on(Gpu::new(crate::instance(), None, (w, h), n, cap))?;
    let mut s = Session::new(world, gpu);
    s.fov_deg = o.fov;
    s.gpu.post.settings = o.optics;
    s.gpu.set_render_scale(o.scale.unwrap_or(1.0));

    let mut controls = Controls::default();
    let dt = 1.0 / 60.0;
    let frames = (o.seconds / dt).round().max(1.0) as usize;
    for _ in 0..frames {
        let mut input = controls.sample(dt);
        if o.burn {
            input.thrust = [1.0, 0.0, 0.0];
            input.boost = true;
        }
        s.frame(dt, &input);
        for note in s.events().iter().filter_map(|e| crate::hud::note(&s.world, e)) {
            println!("{note}");
        }
        // Keep the GPU from queueing up an unbounded amount of work.
        s.gpu.wait();
    }
    s.gpu.request_capture();
    s.frame(dt, &controls.sample(dt));
    s.gpu.wait();
    let raw = s.gpu.take_capture().ok_or("capture failed")?;
    let (cw, ch) =
        (u32::from_le_bytes(raw[0..4].try_into().unwrap()), u32::from_le_bytes(raw[4..8].try_into().unwrap()));
    let file = std::fs::File::create(&o.out).map_err(|e| format!("{}: {e}", o.out))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), cw, ch);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().and_then(|mut wr| wr.write_image_data(&raw[8..])).map_err(|e| format!("png: {e}"))?;
    let t = s.world.telemetry();
    println!("wrote {} ({cw}x{ch}) at τ = {:.1} M, t = {:.1} M, r = {:.2} M", o.out, t.tau, t.t, t.r);
    let mut line = format!("×{:.0} warp · {}", t.warp, crate::hud::throttle(&t));
    if let Some(p) = &t.planet {
        line = format!("{} · {line}", crate::hud::planet(&s.world, p));
    }
    if let Some(phase) = t.autopilot {
        line = format!("AUTOPILOT: {phase} · {line}");
    }
    println!("{line}");
    for (pass, ms, n) in s.gpu.post.timings() {
        println!("  {pass:<32} {ms:7.3} ms  ({n} frames)");
    }
    let m = s.metering();
    println!(
        "exposure {:.3e} per W m⁻² sr⁻¹; metered {:.3e} cd/m², 95th percentile {:.3e} cd/m², fifth star {:.3e} cd/m²",
        s.exposure(),
        m.metered,
        m.p95,
        m.fifth_star
    );
    Ok(())
}

/// Put the ship `k` radii from the nearest star, co-moving with it and
/// looking at it (for checking how resolved stars render).
fn park_near_star(world: &mut kerr::world::World, k: f64) -> Result<(), String> {
    use kerr::vec3;
    let pos = world.pilot.position();
    let star = world
        .cluster
        .bodies
        .iter()
        .filter(|b| b.alive && b.params.radius > 0.0)
        .min_by(|a, b| vec3::norm(vec3::sub(a.position(), pos)).total_cmp(&vec3::norm(vec3::sub(b.position(), pos))))
        .ok_or("no stars")?;
    let sp = star.position();
    let vel = kerr::geodesic::coordinate_velocity(&world.kerr, &star.state);
    let offset = vec3::scale(vec3::normalize(vec3::sub(pos, sp)), k * star.params.radius);
    let at = vec3::add(sp, offset);
    let look = vec3::scale(offset, -1.0);
    let up = vec3::any_orthogonal(look);
    let mut pilot = kerr::pilot::Pilot::new(&world.kerr, at, vel, look, up).ok_or("bad start")?;
    pilot.x[0] = world.pilot.x[0];
    println!(
        "parked {k} radii from a {:.0} K star of {:.2} R☉ and {:.3} L☉",
        star.params.temperature,
        star.params.radius / kerr::units::RSUN,
        star.params.luminosity
    );
    world.pilot = pilot;
    Ok(())
}
