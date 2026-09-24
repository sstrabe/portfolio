//! Radix-2 FFTs: the CPU reference of the compute-shader transforms in
//! `wgsl/fft.wgsl`, the twiddle table they share, and the grid on which
//! the HDR image is convolved with the point-spread function.
//!
//! The GPU convolves in frequency space. Its grid covers the output image
//! downsampled by an integer factor `s` (each grid pixel is s × s output
//! pixels), zero padded to powers of two, at least twice the image in each
//! direction where the size limits allow. The padding keeps a source's
//! diffraction spikes from wrapping around onto the far side of the image:
//! the kernel is tapered to reach no further than the padding.

use std::f64::consts::PI;

/// Largest transform length the shaders handle (their workgroup arrays
/// are sized for it).
pub const MAX_N: u32 = 2048;
/// Largest column length: a column pass holds two columns of this many
/// points in workgroup memory.
pub const MAX_NY: u32 = 1024;

/// A complex number (re, im).
pub type C = [f64; 2];

fn mul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// `e^{−2πi j / MAX_N}` for j < MAX_N / 2: forward twiddles for every
/// length N ≤ MAX_N (stride MAX_N / m at butterfly span m). Computed in f64.
pub fn twiddles() -> Vec<[f32; 2]> {
    (0..MAX_N / 2)
        .map(|j| {
            let a = -2.0 * PI * j as f64 / MAX_N as f64;
            [a.cos() as f32, a.sin() as f32]
        })
        .collect()
}

/// In-place iterative radix-2 decimation-in-time FFT, the same algorithm
/// the shaders run in workgroup memory: bit-reversed load, then log₂N
/// butterfly stages. `inverse` conjugates the twiddles (no 1/N scaling).
pub fn fft(data: &mut [C], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two() && n as u32 <= MAX_N);
    let bits = n.trailing_zeros();
    if bits > 0 {
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if j > i {
                data.swap(i, j);
            }
        }
    }
    let mut m = 2;
    while m <= n {
        let half = m / 2;
        for j in 0..half {
            let a = -2.0 * PI * j as f64 / m as f64;
            let w = [a.cos(), if inverse { -a.sin() } else { a.sin() }];
            for g in (0..n).step_by(m) {
                let (i0, i1) = (g + j, g + j + half);
                let t = mul(w, data[i1]);
                data[i1] = [data[i0][0] - t[0], data[i0][1] - t[1]];
                data[i0] = [data[i0][0] + t[0], data[i0][1] + t[1]];
            }
        }
        m *= 2;
    }
}

/// Size of the convolution grid for an output image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Output pixels per grid pixel along each axis.
    pub step: u32,
    /// Image size on the grid (the downsampled output).
    pub width: u32,
    pub height: u32,
    /// Transform sizes (powers of two).
    pub nx: u32,
    pub ny: u32,
}

impl Grid {
    pub fn for_output(w: u32, h: u32) -> Self {
        let (w, h) = (w.max(1), h.max(1));
        // Half the transform length for the image, the rest for padding.
        let step = w.div_ceil(MAX_N / 2).max(h.div_ceil(MAX_NY * 9 / 16)).max(1);
        let (width, height) = (w.div_ceil(step), h.div_ceil(step));
        let nx = (2 * width).next_power_of_two().clamp(16, MAX_N);
        let ny = (2 * height).next_power_of_two().clamp(16, MAX_NY);
        Self { step, width, height, nx, ny }
    }

    /// How far (grid pixels) the kernel may reach along x and y before a
    /// source's light would wrap around onto the image.
    pub fn reach(&self) -> (u32, u32) {
        (self.nx - self.width, self.ny - self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dft(x: &[C], inverse: bool) -> Vec<C> {
        let n = x.len();
        let s = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                x.iter().enumerate().fold([0.0, 0.0], |acc, (j, v)| {
                    let a = s * 2.0 * PI * (j * k % n) as f64 / n as f64;
                    let t = mul([a.cos(), a.sin()], *v);
                    [acc[0] + t[0], acc[1] + t[1]]
                })
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<C> {
        let mut h = 0x2545_f491_u64;
        (0..n)
            .map(|_| {
                h ^= h << 13;
                h ^= h >> 7;
                h ^= h << 17;
                [(h % 1000) as f64 / 500.0 - 1.0, ((h >> 20) % 1000) as f64 / 500.0 - 1.0]
            })
            .collect()
    }

    #[test]
    fn fft_matches_a_direct_dft() {
        for n in [1, 2, 4, 8, 64, 256] {
            let x = signal(n);
            for inverse in [false, true] {
                let mut y = x.clone();
                fft(&mut y, inverse);
                let d = dft(&x, inverse);
                for (a, b) in y.iter().zip(&d) {
                    assert!((a[0] - b[0]).abs() < 1e-9 * n as f64 && (a[1] - b[1]).abs() < 1e-9 * n as f64);
                }
            }
        }
    }

    #[test]
    fn inverse_undoes_forward() {
        let x = signal(2048);
        let mut y = x.clone();
        fft(&mut y, false);
        fft(&mut y, true);
        for (a, b) in y.iter().zip(&x) {
            assert!((a[0] / 2048.0 - b[0]).abs() < 1e-12 && (a[1] / 2048.0 - b[1]).abs() < 1e-12);
        }
    }

    /// Two real signals packed as re + i·im transform together; the shaders
    /// separate them with X̂ = (P[k] + P*[−k])/2, Ŷ = (P[k] − P*[−k])/2i.
    #[test]
    fn two_real_signals_in_one_transform() {
        let n = 32;
        let s = signal(n);
        let mut p = s.clone();
        fft(&mut p, false);
        let re: Vec<C> = s.iter().map(|v| [v[0], 0.0]).collect();
        let im: Vec<C> = s.iter().map(|v| [v[1], 0.0]).collect();
        let (xr, xi) = (dft(&re, false), dft(&im, false));
        for k in 0..n {
            let q = p[(n - k) % n];
            let x = [(p[k][0] + q[0]) / 2.0, (p[k][1] - q[1]) / 2.0];
            let y = [(p[k][1] + q[1]) / 2.0, -(p[k][0] - q[0]) / 2.0];
            assert!((x[0] - xr[k][0]).abs() < 1e-9 && (x[1] - xr[k][1]).abs() < 1e-9);
            assert!((y[0] - xi[k][0]).abs() < 1e-9 && (y[1] - xi[k][1]).abs() < 1e-9);
        }
    }

    #[test]
    fn twiddle_table_matches_the_stage_twiddles() {
        let t = twiddles();
        for m in [2u32, 16, 2048] {
            for j in 0..m / 2 {
                let w = t[(j * (MAX_N / m)) as usize];
                let a = -2.0 * PI * j as f64 / m as f64;
                assert!((w[0] as f64 - a.cos()).abs() < 1e-7 && (w[1] as f64 - a.sin()).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn grids_fit_the_shaders_and_pad_generously() {
        for (w, h) in [(1920, 1080), (1280, 720), (2560, 1440), (3840, 2160), (800, 600), (64, 48), (1080, 1920)] {
            let g = Grid::for_output(w, h);
            assert!(g.nx <= MAX_N && g.ny <= MAX_NY, "{g:?}");
            assert!(g.width * g.step >= w && g.height * g.step >= h);
            let (rx, ry) = g.reach();
            assert!(rx >= g.width / 2 && ry >= g.height * 3 / 4, "{w}x{h}: {g:?}");
        }
        assert_eq!(Grid::for_output(1920, 1080), Grid { step: 2, width: 960, height: 540, nx: 2048, ny: 1024 });
    }
}
