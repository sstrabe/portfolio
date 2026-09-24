//! Worldline history on a shared, uniformly spaced coordinate-time grid.
//!
//! Every body is sampled at the same instants `t_newest − k·dt`, which makes
//! lookups O(1) and lets the renderer mirror the ring buffer on the GPU one
//! column per tick. Between samples positions are cubic-Hermite
//! interpolated using the stored coordinate velocities.

use crate::vec3::V3;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sample {
    pub pos: V3,
    /// Coordinate velocity `dx/dt`.
    pub vel: V3,
}

#[derive(Clone, Debug)]
pub struct HistoryRing {
    dt: f64,
    cap: usize,
    bodies: usize,
    data: Vec<Sample>,
    newest_slot: usize,
    count: usize,
    t_newest: f64,
}

impl HistoryRing {
    pub fn new(bodies: usize, cap: usize, dt: f64) -> Self {
        assert!(cap >= 2 && dt > 0.0);
        Self {
            dt,
            cap,
            bodies,
            data: vec![Sample::default(); bodies * cap],
            newest_slot: cap - 1,
            count: 0,
            t_newest: f64::NEG_INFINITY,
        }
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }
    pub fn capacity(&self) -> usize {
        self.cap
    }
    pub fn bodies(&self) -> usize {
        self.bodies
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn t_newest(&self) -> f64 {
        self.t_newest
    }
    pub fn t_oldest(&self) -> f64 {
        self.t_newest - (self.count.max(1) - 1) as f64 * self.dt
    }
    pub fn newest_slot(&self) -> usize {
        self.newest_slot
    }
    /// Next sampling instant.
    pub fn t_next(&self) -> f64 {
        self.t_newest + self.dt
    }

    /// Ring slot of the sample that is `age` ticks old (0 = newest).
    pub fn slot(&self, age: usize) -> usize {
        (self.newest_slot + self.cap - age % self.cap) % self.cap
    }

    /// Append a column at time `t` (the first push sets the grid origin).
    pub fn push(&mut self, t: f64, mut sample: impl FnMut(usize) -> Sample) {
        self.newest_slot = (self.newest_slot + 1) % self.cap;
        let base = self.newest_slot * self.bodies;
        for b in 0..self.bodies {
            self.data[base + b] = sample(b);
        }
        self.t_newest = t;
        self.count = (self.count + 1).min(self.cap);
    }

    pub fn get(&self, body: usize, age: usize) -> Sample {
        self.data[self.slot(age) * self.bodies + body]
    }

    pub fn set(&mut self, body: usize, age: usize, s: Sample) {
        let i = self.slot(age) * self.bodies + body;
        self.data[i] = s;
    }

    /// All bodies' samples in one ring slot (for GPU upload).
    pub fn column(&self, slot: usize) -> &[Sample] {
        &self.data[slot * self.bodies..(slot + 1) * self.bodies]
    }

    /// Worldline position/velocity of `body` at coordinate time `t`.
    ///
    /// `now` supplies the live state beyond the newest sample. Returns
    /// `None` for times older than the stored history.
    pub fn at(&self, body: usize, t: f64, now: Option<(f64, Sample)>) -> Option<Sample> {
        if self.count == 0 {
            return now.filter(|(tn, _)| (t - tn).abs() < 1e-9).map(|(_, s)| s);
        }
        if t > self.t_newest {
            let (tn, sn) = now?;
            let h = tn - self.t_newest;
            if t >= tn || h <= 1e-12 {
                return Some(extrapolate(sn, t - tn));
            }
            return Some(hermite(self.get(body, 0), sn, h, (t - self.t_newest) / h));
        }
        let age = (self.t_newest - t) / self.dt;
        let k = age.floor() as usize;
        if k + 1 >= self.count {
            if k + 1 == self.count && age <= k as f64 {
                return Some(self.get(body, k));
            }
            return None;
        }
        let s_new = self.get(body, k);
        let s_old = self.get(body, k + 1);
        Some(hermite(s_old, s_new, self.dt, 1.0 - (age - k as f64)))
    }
}

fn extrapolate(s: Sample, dt: f64) -> Sample {
    Sample { pos: crate::vec3::axpy(s.pos, dt, s.vel), vel: s.vel }
}

/// Cubic Hermite interpolation between `a` (at s = 0) and `b` (at s = 1)
/// over an interval of length `h`.
pub fn hermite(a: Sample, b: Sample, h: f64, s: f64) -> Sample {
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    let d00 = 6.0 * s2 - 6.0 * s;
    let d10 = 3.0 * s2 - 4.0 * s + 1.0;
    let d01 = -d00;
    let d11 = 3.0 * s2 - 2.0 * s;
    let mut out = Sample::default();
    for i in 0..3 {
        out.pos[i] = h00 * a.pos[i] + h10 * h * a.vel[i] + h01 * b.pos[i] + h11 * h * b.vel[i];
        out.vel[i] = (d00 * a.pos[i] + d01 * b.pos[i]) / h + d10 * a.vel[i] + d11 * b.vel[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle(t: f64) -> Sample {
        let w = 0.05;
        Sample {
            pos: [10.0 * (w * t).cos(), 10.0 * (w * t).sin(), 0.0],
            vel: [-10.0 * w * (w * t).sin(), 10.0 * w * (w * t).cos(), 0.0],
        }
    }

    #[test]
    fn interpolates_and_wraps() {
        let mut h = HistoryRing::new(1, 16, 2.0);
        for k in 0..40 {
            let t = k as f64 * 2.0;
            h.push(t, |_| circle(t));
        }
        assert_eq!(h.len(), 16);
        assert!((h.t_oldest() - 48.0).abs() < 1e-12);
        for t in [49.0, 55.3, 70.1, 77.9, 78.0] {
            let s = h.at(0, t, None).unwrap();
            let e = circle(t);
            for i in 0..3 {
                assert!((s.pos[i] - e.pos[i]).abs() < 2e-4, "t={t}");
                assert!((s.vel[i] - e.vel[i]).abs() < 2e-4, "t={t}");
            }
        }
        assert!(h.at(0, 47.0, None).is_none());
        assert!(h.at(0, 48.0, None).is_some());
        let now = (79.2, circle(79.2));
        let s = h.at(0, 78.7, Some(now)).unwrap();
        assert!((s.pos[0] - circle(78.7).pos[0]).abs() < 1e-5);
    }
}
