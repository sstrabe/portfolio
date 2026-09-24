// HTML that floats over the lensed scene: station labels pinned to the
// stations' apparent (lensed, aberrated, light-delayed) screen positions,
// the station list, the docking panel with full content, toasts and help.

import type { Portfolio, Station } from "../content/portfolio";
import { esc, renderAboutBody, renderProjectBody } from "../content/render";

/** Layout of `Engine.stations()` per station. */
export const S = {
  Visible: 0,
  NdcX: 1,
  NdcY: 2,
  OnScreen: 3,
  Radius: 4,
  G: 5,
  Flux: 6,
  Distance: 7,
  Delay: 8,
  Stride: 10,
} as const;

export interface OverlayCallbacks {
  flyTo(station: number): void;
  undock(): void;
  highlight(station: number): void;
}

export class Overlay {
  readonly labels: HTMLElement;
  readonly nav: HTMLElement;
  readonly dock: HTMLElement;
  readonly toasts: HTMLElement;
  private labelEls: HTMLButtonElement[] = [];
  private navEls: HTMLButtonElement[] = [];
  private navDist: HTMLElement[] = [];
  private docked = -1;
  /** Hide labels for stations whose in-scene card is already legible. */
  inSceneCards = false;

  constructor(
    private stations: Station[],
    private portfolio: Portfolio,
    private cb: OverlayCallbacks,
  ) {
    this.labels = div("labels");
    this.nav = document.createElement("nav");
    this.nav.className = "station-nav";
    this.nav.setAttribute("aria-label", "Stations");
    const heading = document.createElement("h2");
    heading.textContent = "Stations";
    this.nav.append(heading);
    const list = document.createElement("ol");
    this.nav.append(list);

    for (const s of stations) {
      const label = document.createElement("button");
      label.className = "station-label";
      label.style.setProperty("--accent", s.accent);
      label.innerHTML = `<span class="station-label-title">${esc(s.title)}</span><span class="station-label-meta"></span>`;
      label.addEventListener("click", () => cb.flyTo(s.index));
      label.addEventListener("pointerenter", () => cb.highlight(s.index));
      label.addEventListener("pointerleave", () => cb.highlight(-1));
      label.hidden = true;
      this.labels.append(label);
      this.labelEls.push(label);

      const li = document.createElement("li");
      const b = document.createElement("button");
      b.style.setProperty("--accent", s.accent);
      b.innerHTML = `<kbd>${s.index + 1}</kbd><span class="nav-title">${esc(s.title)}</span><span class="nav-dist"></span>`;
      b.addEventListener("click", () => cb.flyTo(s.index));
      b.addEventListener("pointerenter", () => cb.highlight(s.index));
      b.addEventListener("pointerleave", () => cb.highlight(-1));
      li.append(b);
      list.append(li);
      this.navEls.push(b);
      this.navDist.push(b.querySelector(".nav-dist")!);
    }

    this.dock = document.createElement("aside");
    this.dock.className = "dock-panel";
    this.dock.hidden = true;
    this.dock.setAttribute("aria-live", "polite");
    this.toasts = div("toasts");
    this.toasts.setAttribute("role", "status");
  }

  /** Position labels from the engine's station data. */
  update(data: Float32Array, width: number, height: number, target: number) {
    for (let i = 0; i < this.stations.length; i++) {
      const o = i * S.Stride;
      const el = this.labelEls[i];
      const dist = data[o + S.Distance];
      this.navDist[i].textContent = `${dist.toFixed(dist < 10 ? 2 : 0)} M`;
      this.navEls[i].classList.toggle("active", i === target || i === this.docked);
      const radiusPx = data[o + S.Radius] * height * 0.5;
      const show =
        data[o + S.Visible] > 0.5 &&
        data[o + S.OnScreen] > 0.5 &&
        i !== this.docked &&
        !(this.inSceneCards && radiusPx > 70);
      el.hidden = !show;
      if (!show) continue;
      const x = (data[o + S.NdcX] * 0.5 + 0.5) * width;
      const y = (0.5 - data[o + S.NdcY] * 0.5) * height + Math.min(radiusPx * 0.7, height * 0.3) + 10;
      el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translateX(-50%)`;
      el.classList.toggle("target", i === target);
      const g = data[o + S.G];
      const shift = g > 1.02 ? `blueshift ×${g.toFixed(2)}` : g < 0.98 ? `redshift ×${g.toFixed(2)}` : "";
      el.querySelector(".station-label-meta")!.textContent =
        `${dist.toFixed(dist < 10 ? 2 : 1)} M · seen ${data[o + S.Delay].toFixed(1)} M ago${shift ? ` · ${shift}` : ""}`;
    }
  }

  showDock(i: number) {
    this.docked = i;
    const s = this.stations[i];
    const body = s.project ? renderProjectBody(s.project) : renderAboutBody(this.portfolio);
    this.dock.style.setProperty("--accent", s.accent);
    this.dock.innerHTML = `
      <header>
        <p class="dock-kicker">Docked · ${i === 0 ? "home station" : `station ${i + 1}`}</p>
        <h2>${esc(s.title)}</h2>
      </header>
      <div class="dock-body">${body}</div>
      <button class="undock" type="button">Undock <kbd>Esc</kbd></button>`;
    this.dock.querySelector<HTMLButtonElement>(".undock")!.addEventListener("click", () => this.cb.undock());
    this.dock.hidden = false;
  }

  hideDock() {
    this.docked = -1;
    this.dock.hidden = true;
  }

  toast(text: string, ms = 5000) {
    const t = document.createElement("p");
    t.className = "toast";
    t.textContent = text;
    this.toasts.append(t);
    setTimeout(() => t.remove(), ms);
  }
}

export function helpDialog(): HTMLDialogElement {
  const d = document.createElement("dialog");
  d.className = "help";
  d.innerHTML = `
    <h2>Flying in curved spacetime</h2>
    <dl>
      <dt><kbd>W</kbd> <kbd>S</kbd></dt><dd>Thrust forward / back</dd>
      <dt><kbd>A</kbd> <kbd>D</kbd></dt><dd>Strafe left / right</dd>
      <dt><kbd>Space</kbd> <kbd>C</kbd></dt><dd>Thrust up / down</dd>
      <dt>Click, then mouse; <kbd>←</kbd><kbd>→</kbd><kbd>↑</kbd><kbd>↓</kbd></dt><dd>Turn (<kbd>Esc</kbd> releases the mouse, <kbd>I</kbd> inverts up/down)</dd>
      <dt><kbd>Q</kbd> <kbd>E</kbd></dt><dd>Roll</dd>
      <dt><kbd>Shift</kbd></dt><dd>Boost (6× thrust)</dd>
      <dt><kbd>X</kbd></dt><dd>Brake: match the nearest station's velocity</dd>
      <dt><kbd>1</kbd>–<kbd>8</kbd>, click a label</dt><dd>Autopilot to a station and dock</dd>
      <dt><kbd>Esc</kbd></dt><dd>Cancel autopilot / undock</dd>
      <dt><kbd>H</kbd></dt><dd>This help</dd>
    </dl>
    <p>Everything you see is light that reached you along curved paths: stations appear where they <em>were</em>, bent around the hole, bluer when approaching and brighter ahead of you. Your clock drives the simulation, so going fast or hovering deep in the well makes the cluster race ahead. Times are in units of M (≈ 21 s for Sagittarius A*).</p>
    <form method="dialog"><button>Fly</button></form>`;
  return d;
}

function div(className: string): HTMLDivElement {
  const d = document.createElement("div");
  d.className = className;
  return d;
}

