// Telemetry readout. Units are geometrized: times and lengths in M, the
// hole's mass. For Sagittarius A* (4.3 million solar masses) one M of time
// is GM/c³ ≈ 21 s, which gives a feel for how much time passes.

/** Indices into `Engine.telemetry()`. */
export const T = {
  Tau: 0,
  Time: 1,
  R: 2,
  DtDtau: 3,
  Gamma: 4,
  Speed: 5,
  ClockRate: 6,
  Status: 7,
  Station: 8,
  StationDistance: 9,
  StationSpeed: 10,
  RPlus: 11,
  Spin: 12,
  Alive: 13,
} as const;

const SGR_A_STAR_SECONDS_PER_M = 21.2;

export function humanDuration(m: number): string {
  const s = m * SGR_A_STAR_SECONDS_PER_M;
  if (s < 120) return `${s.toFixed(0)} s`;
  if (s < 7200) return `${(s / 60).toFixed(1)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
  if (s < 3.15e7 * 2) return `${(s / 86400).toFixed(1)} d`;
  return `${(s / 3.156e7).toFixed(1)} yr`;
}

export class Hud {
  readonly el: HTMLElement;
  private fields: Record<string, HTMLElement> = {};
  private limiter: HTMLElement;

  constructor() {
    this.el = document.createElement("section");
    this.el.className = "hud";
    this.el.setAttribute("aria-label", "Flight telemetry");
    const rows: [string, string, string][] = [
      ["tau", "Your proper time τ", "Time on the ship's clock"],
      ["t", "Cluster time t", "Kerr–Schild coordinate time of the cluster"],
      ["rate", "dt / dτ", "How fast the universe runs relative to you"],
      ["v", "Speed", "Relative to the local normal observer"],
      ["r", "Radius r", "Distance from the hole (horizon at r₊)"],
    ];
    for (const [key, label, title] of rows) {
      const row = document.createElement("div");
      row.className = "hud-row";
      row.title = title;
      const l = document.createElement("span");
      l.className = "hud-label";
      l.textContent = label;
      const v = document.createElement("span");
      v.className = "hud-value";
      row.append(l, v);
      this.el.append(row);
      this.fields[key] = v;
    }
    this.limiter = document.createElement("p");
    this.limiter.className = "hud-limiter";
    this.limiter.hidden = true;
    this.el.append(this.limiter);
  }

  update(t: Float64Array) {
    const f = this.fields;
    f.tau.textContent = `${t[T.Tau].toFixed(1)} M · ${humanDuration(t[T.Tau])}`;
    f.t.textContent = `${t[T.Time].toFixed(1)} M · ${humanDuration(t[T.Time])}`;
    f.rate.textContent = `${t[T.DtDtau].toFixed(3)}×`;
    f.v.textContent = `${t[T.Speed].toFixed(4)} c · γ ${t[T.Gamma].toFixed(3)}`;
    f.r.textContent = `${t[T.R].toFixed(2)} M · r₊ ${t[T.RPlus].toFixed(3)}`;
    const rate = t[T.ClockRate];
    this.limiter.hidden = rate > 0.98;
    this.limiter.textContent = `Clock limiter: the cluster can only be simulated so fast, so your own clock runs at ${(rate * 100).toFixed(0)}% of wall time.`;
  }
}
