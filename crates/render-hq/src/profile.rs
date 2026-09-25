//! GPU timings of the passes, from timestamp queries: enabled when the
//! environment variable `KERR_GPU_TIMING` is set and the GPU supports
//! them. Results are read back without stalling and averaged per pass.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

type Totals = Arc<Mutex<BTreeMap<&'static str, (f64, u32)>>>;

pub struct Profiler {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    busy: Arc<Mutex<bool>>,
    capacity: u32,
    labels: Vec<&'static str>,
    period_ns: f64,
    totals: Totals,
}

/// Whether timings were asked for.
pub fn wanted() -> bool {
    std::env::var_os("KERR_GPU_TIMING").is_some()
}

impl Profiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, capacity: u32) -> Option<Self> {
        if !wanted() || !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let bytes = capacity as u64 * 2 * 8;
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        Some(Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timings"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity * 2,
            }),
            resolve: buffer("timings", wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: buffer("timings read-back", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            busy: Arc::default(),
            capacity,
            labels: Vec::new(),
            period_ns: queue.get_timestamp_period() as f64,
            totals: Arc::default(),
        })
    }

    fn slot(&mut self, label: &'static str) -> Option<(u32, u32)> {
        if *self.busy.lock().unwrap_or_else(|e| e.into_inner()) || self.labels.len() as u32 >= self.capacity {
            return None;
        }
        let i = self.labels.len() as u32;
        self.labels.push(label);
        Some((2 * i, 2 * i + 1))
    }

    pub fn compute(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (a, b) = self.slot(label)?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(a),
            end_of_pass_write_index: Some(b),
        })
    }

    pub fn render(&mut self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (a, b) = self.slot(label)?;
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(a),
            end_of_pass_write_index: Some(b),
        })
    }

    /// Resolve this frame's queries; they are added to the totals once the
    /// GPU finishes.
    pub fn finish(&mut self, enc: &mut wgpu::CommandEncoder) {
        let labels = std::mem::take(&mut self.labels);
        if labels.is_empty() {
            return;
        }
        let n = labels.len() as u32 * 2;
        enc.resolve_query_set(&self.set, 0..n, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, n as u64 * 8);
        *self.busy.lock().unwrap_or_else(|e| e.into_inner()) = true;
        let (read, busy, totals, period) = (self.read.clone(), self.busy.clone(), self.totals.clone(), self.period_ns);
        enc.map_buffer_on_submit(&self.read, wgpu::MapMode::Read, .., move |result| {
            if result.is_ok()
                && let Ok(data) = read.slice(..).get_mapped_range()
            {
                let t: &[u64] = bytemuck::cast_slice(&data);
                let mut totals = totals.lock().unwrap_or_else(|e| e.into_inner());
                for (i, label) in labels.iter().enumerate() {
                    let ms = t[2 * i + 1].wrapping_sub(t[2 * i]) as f64 * period * 1e-6;
                    let e = totals.entry(label).or_insert((0.0, 0));
                    e.0 += ms;
                    e.1 += 1;
                }
            }
            read.unmap();
            *busy.lock().unwrap_or_else(|e| e.into_inner()) = false;
        });
    }

    /// Mean milliseconds per pass so far.
    pub fn report(&self) -> Vec<(&'static str, f64, u32)> {
        let totals = self.totals.lock().unwrap_or_else(|e| e.into_inner());
        totals.iter().map(|(k, (sum, n))| (*k, sum / *n as f64, *n)).collect()
    }
}
