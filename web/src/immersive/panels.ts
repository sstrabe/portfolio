// The station cards that are drawn *into* the lensed scene.
//
// The engine ray traces each nearby station's card as a flat panel moving
// with the station, sampling this atlas (4 × 2 cells of 512 × 320 px). The
// atlas is painted one of two ways:
//
// * HTML-in-Canvas (WICG proposal, behind a flag in Chromium): the cards
//   are real HTML elements laid out inside a `<canvas layoutsubtree>` and
//   rasterized with `drawElementImage`, so the scene shows actual HTML.
// * Everywhere else: the same information is drawn with Canvas 2D, and
//   the HTML labels float over the stations instead.

import type { Station } from "../content/portfolio";

export const CELL_W = 512;
export const CELL_H = 320;
export const COLS = 4;
export const ROWS = 2;

type DrawElementImage = (el: Element, x: number, y: number, w?: number, h?: number) => unknown;

interface HtmlCanvasContext extends CanvasRenderingContext2D {
  drawElementImage?: DrawElementImage;
}

export function htmlInCanvasSupported(): boolean {
  const proto = CanvasRenderingContext2D.prototype as HtmlCanvasContext;
  return typeof proto.drawElementImage === "function";
}

export class PanelAtlas {
  readonly canvas: HTMLCanvasElement;
  readonly htmlMode: boolean;
  private ctx: HtmlCanvasContext;
  private cards: HTMLElement[] = [];

  constructor(
    private stations: Station[],
    private onPaint: (canvas: HTMLCanvasElement) => void,
  ) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = CELL_W * COLS;
    this.canvas.height = CELL_H * ROWS;
    this.canvas.className = "panel-atlas";
    this.canvas.setAttribute("aria-hidden", "true");
    this.ctx = this.canvas.getContext("2d")! as HtmlCanvasContext;
    this.htmlMode = htmlInCanvasSupported();
    if (this.htmlMode) {
      this.canvas.setAttribute("layoutsubtree", "");
      for (const s of stations) this.cards.push(this.canvas.appendChild(cardElement(s)));
      this.canvas.addEventListener("paint", () => this.paintHtml());
    }
    document.body.appendChild(this.canvas);
  }

  /** Paint (after fonts are ready) and hand the atlas to the engine. */
  async paint() {
    await document.fonts?.ready;
    if (this.htmlMode) {
      const c = this.canvas as HTMLCanvasElement & { requestPaint?: () => void };
      if (typeof c.requestPaint === "function") c.requestPaint();
      else this.paintHtml();
      return;
    }
    this.paint2d();
    this.onPaint(this.canvas);
  }

  private paintHtml() {
    try {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.cards.forEach((el, i) => {
        const [x, y] = cellOrigin(i);
        this.ctx.drawElementImage!(el, x, y, CELL_W, CELL_H);
      });
      this.onPaint(this.canvas);
    } catch (e) {
      console.warn("HTML-in-Canvas paint failed; falling back to Canvas 2D", e);
      this.paint2d();
      this.onPaint(this.canvas);
    }
  }

  private paint2d() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.stations.forEach((s, i) => drawCard(ctx, s, ...cellOrigin(i)));
  }
}

function cellOrigin(i: number): [number, number] {
  return [(i % COLS) * CELL_W, Math.floor(i / COLS) * CELL_H];
}

function cardElement(s: Station): HTMLElement {
  const el = document.createElement("div");
  el.className = "scene-card";
  el.style.setProperty("--accent", s.accent);
  el.style.width = `${CELL_W}px`;
  el.style.height = `${CELL_H}px`;
  const kicker = document.createElement("p");
  kicker.className = "scene-card-kicker";
  kicker.textContent = s.index === 0 ? "Home station" : `Station ${s.index + 1}`;
  const title = document.createElement("h3");
  title.textContent = s.title;
  const sub = document.createElement("p");
  sub.textContent = s.subtitle;
  const tags = document.createElement("p");
  tags.className = "scene-card-tags";
  tags.textContent = s.project?.tags.slice(0, 4).join(" · ") ?? "About · Contact";
  el.append(kicker, title, sub, tags);
  return el;
}

function drawCard(ctx: CanvasRenderingContext2D, s: Station, x: number, y: number) {
  const pad = 30;
  ctx.save();
  ctx.translate(x, y);
  const bg = ctx.createLinearGradient(0, 0, CELL_W, CELL_H);
  bg.addColorStop(0, "rgba(14, 18, 30, 0.96)");
  bg.addColorStop(1, "rgba(6, 8, 14, 0.96)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, CELL_W, CELL_H);
  ctx.fillStyle = s.accent;
  ctx.fillRect(0, 0, 8, CELL_H);

  ctx.fillStyle = s.accent;
  ctx.font = "600 20px system-ui, -apple-system, 'Segoe UI', sans-serif";
  ctx.fillText((s.index === 0 ? "HOME STATION" : `STATION ${s.index + 1}`).toUpperCase(), pad, pad + 16);

  ctx.fillStyle = "#f4f6fb";
  ctx.font = "700 44px system-ui, -apple-system, 'Segoe UI', sans-serif";
  const titleLines = wrap(ctx, s.title, CELL_W - 2 * pad).slice(0, 2);
  titleLines.forEach((l, i) => ctx.fillText(l, pad, pad + 74 + i * 50));

  ctx.fillStyle = "#b9c1d6";
  ctx.font = "400 24px system-ui, -apple-system, 'Segoe UI', sans-serif";
  const top = pad + 74 + titleLines.length * 50 + 6;
  wrap(ctx, s.subtitle, CELL_W - 2 * pad)
    .slice(0, 4)
    .forEach((l, i) => ctx.fillText(l, pad, top + i * 32));

  ctx.fillStyle = "#7d879e";
  ctx.font = "500 18px system-ui, -apple-system, 'Segoe UI', sans-serif";
  ctx.fillText(s.project?.tags.slice(0, 4).join(" · ") ?? "About · Contact", pad, CELL_H - pad + 4);
  ctx.restore();
}

function wrap(ctx: CanvasRenderingContext2D, text: string, width: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (const w of words) {
    const next = line ? `${line} ${w}` : w;
    if (ctx.measureText(next).width > width && line) {
      lines.push(line);
      line = w;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}
