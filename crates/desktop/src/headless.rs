//! Offscreen rendering to a PNG: `--headless WIDTHxHEIGHT`.

use crate::Options;
use crate::input::Controls;
use render::{Gpu, Session};

pub fn run(o: &Options) -> Result<(), String> {
    let (w, h) = o.headless.expect("headless size");
    let world = crate::world(o.stars);
    let (n, cap) = (world.cluster.len() as u32, world.cluster.history.capacity() as u32);
    let gpu = pollster::block_on(Gpu::new(crate::instance(), None, (w, h), n, cap))?;
    let mut s = Session::new(world, gpu);
    s.fov_deg = o.fov;
    s.gpu.set_render_scale(o.scale.unwrap_or(1.0));

    let mut controls = Controls::default();
    let dt = 1.0 / 60.0;
    let frames = (o.seconds / dt).round().max(1.0) as usize;
    for _ in 0..frames {
        s.frame(dt, &controls.sample(dt));
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
    Ok(())
}
