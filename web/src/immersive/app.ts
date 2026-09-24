// The immersive nucleus: owns the WebGPU canvas and the Wasm engine, runs
// the frame loop, and keeps the HTML layer in sync with the simulation.

import init, { createEngine, createHeadlessEngine, type Engine } from "../wasm/pkg/engine.js";
import { portfolio, stations as makeStations } from "../content/portfolio";
import { esc } from "../content/render";
import { Controls } from "./input";
import { Hud, T } from "./hud";
import { Overlay, helpDialog } from "./overlay";
import { PanelAtlas } from "./panels";
import { flags } from "../flags";

const EVENT_HORIZON = 1;
const EVENT_DOCKED = 2;
const EVENT_UNDOCKED = 3;
const EVENT_CAPTURE = 4;

export async function start(host: HTMLElement, fail: (reason: unknown) => void) {
  const stations = makeStations(portfolio);

  const gui = !flags.noGui;
  host.innerHTML = `
    <canvas class="scene" aria-label="A star cluster around a spinning black hole"></canvas>`;
  if (gui) {
    host.insertAdjacentHTML(
      "beforeend",
      `
    <header class="topbar">
      <h1>${esc(portfolio.owner.name)}</h1>
      <p>${esc(portfolio.owner.headline)}</p>
      <nav class="topbar-links">
        <button type="button" class="photo-button" title="Save what you see as a PNG">Photo <kbd>P</kbd></button>
        <button type="button" class="help-button">Controls <kbd>H</kbd></button>
        <a href="?plain" class="plain-link">Plain version</a>
      </nav>
    </header>
    <div class="loader" role="status"><p>Integrating geodesics…</p></div>
    <div class="touchpad" aria-hidden="true">
      <button data-k="forward">▲</button><button data-k="back">▼</button>
      <button data-k="boost">Boost</button><button data-k="brake">Brake</button>
    </div>`,
    );
    host.querySelector(".plain-link")!.addEventListener("click", () => {
      try {
        localStorage.setItem("kerr-mode", "plain");
      } catch {
        // ignore
      }
    });
  }
  const canvas = host.querySelector<HTMLCanvasElement>("canvas.scene")!;

  if (!navigator.gpu) throw unsupported("WebGPU is not available");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw unsupported("no WebGPU adapter");

  const dpr = Math.min(devicePixelRatio || 1, 2);
  const sizeCanvas = () => {
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  };
  sizeCanvas();

  await init();
  const small = Math.min(screen.width, screen.height) < 700 || (navigator.hardwareConcurrency ?? 8) <= 4;
  const params = new URLSearchParams(location.search);
  // ?debug&headless renders offscreen and exposes frames to automation
  // (for headless browsers that cannot present WebGPU canvases).
  const headless = params.has("debug") && params.has("headless");
  const stars = small ? 160 : 384;
  const engine: Engine = headless
    ? await createHeadlessEngine(canvas.width, canvas.height, stations.length, stars, 1)
    : await createEngine(canvas, stations.length, stars, 1);

  const controls = new Controls(canvas);
  let highlight = -1;
  // Everything HTML on top of the scene; absent when the GUI is disabled.
  const ui = gui ? buildGui() : null;
  function buildGui() {
    const hud = new Hud();
    const overlay: Overlay = new Overlay(stations, portfolio, {
      flyTo: (i) => {
        controls.flyTo(i);
        overlay.toast(`Autopilot: ${stations[i].title}`, 2500);
      },
      undock: () => controls.requestUndock(),
      highlight: (i) => {
        highlight = i;
      },
    });
    const help = helpDialog();
    host.append(overlay.labels, overlay.nav, overlay.dock, overlay.toasts, hud.el, help);
    controls.onHelp = () => (help.open ? help.close() : help.showModal());
    controls.onAutopilotCancelled = () => overlay.toast("Autopilot off", 1500);
    host.querySelector(".help-button")!.addEventListener("click", () => help.showModal());
    host
      .querySelectorAll<HTMLElement>(".touchpad button")
      .forEach((b) => controls.bindTouchButton(b, b.dataset.k as "forward" | "back" | "boost" | "brake"));
    return { hud, overlay };
  }
  // Captures must happen in the same task as the frame that drew them.
  const afterFrame: Array<() => void> = [];
  const savePhoto = () =>
    afterFrame.push(() =>
      canvas.toBlob((blob) => {
        if (!blob) return;
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `kerr-nucleus-${Date.now()}.png`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      }, "image/png"),
    );
  host.querySelector(".photo-button")?.addEventListener("click", savePhoto);
  addEventListener("keydown", (e) => {
    if (gui && e.code === "KeyP" && !e.repeat && !(e.target instanceof HTMLInputElement)) savePhoto();
  });
  if (params.has("debug")) {
    const snapshot = async () => {
      const png = await capturePng(engine);
      // Headless: show the capture behind the HTML layer so a page
      // screenshot contains both.
      if (headless) canvas.style.background = `center / 100% 100% url(${png})`;
      return png;
    };
    Object.assign(window, { kerrDebug: { engine, controls, snapshot } });
  }

  const atlas = new PanelAtlas(stations, (c) => engine.uploadPanelAtlas(c));
  if (ui) ui.overlay.inSceneCards = atlas.htmlMode;
  await atlas.paint();

  new ResizeObserver(() => {
    sizeCanvas();
    engine.resize(canvas.width, canvas.height);
  }).observe(canvas);

  // Adaptive resolution for the per-pixel ray tracer.
  let scale = small ? 0.6 : 0.85;
  engine.setRenderScale(scale);
  let avg = 16;
  let last = performance.now();
  let started = false;

  const frame = (now: number) => {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    try {
      engine.setHighlight(highlight);
      engine.frame(dt, controls.sample(dt));
      if (ui) {
        const tel = engine.telemetry();
        ui.hud.update(tel);
        const target = tel[T.Status] === 1 ? tel[T.Station] : -1;
        ui.overlay.update(engine.stations(), canvas.clientWidth, canvas.clientHeight, target);
      }
      const ev = engine.events();
      for (let i = 0; i < ev.length; i += 2) handleEvent(ev[i], ev[i + 1]);
      afterFrame.splice(0).forEach((f) => f());
    } catch (e) {
      fail(e);
      return;
    }
    if (!started) {
      started = true;
      host.querySelector(".loader")?.remove();
      if (ui && !sessionStorage.getItem("kerr-help-seen")) {
        sessionStorage.setItem("kerr-help-seen", "1");
        ui.overlay.toast("Press H for controls. Click a station to fly there.", 7000);
      }
    }
    avg = avg * 0.95 + dt * 1000 * 0.05;
    if (avg > 22 && scale > 0.5) scale = Math.max(0.5, scale * 0.98);
    else if (avg < 14 && scale < 1) scale = Math.min(1, scale * 1.01);
    engine.setRenderScale(scale);
    requestAnimationFrame(frame);
  };

  let lastCaptureToast = -Infinity;
  function handleEvent(code: number, arg: number) {
    if (code === EVENT_DOCKED) controls.autopilot = -1;
    const overlay = ui?.overlay;
    if (!overlay) return;
    switch (code) {
      case EVENT_HORIZON:
        overlay.hideDock();
        overlay.toast("You crossed the event horizon. Nothing gets out — except you, respawned at home.", 7000);
        break;
      case EVENT_DOCKED:
        overlay.showDock(arg);
        break;
      case EVENT_UNDOCKED:
        overlay.hideDock();
        break;
      case EVENT_CAPTURE:
        if (performance.now() - lastCaptureToast > 60_000) {
          lastCaptureToast = performance.now();
          overlay.toast("A star just crossed the horizon. Its last light is still on its way to you.", 4000);
        }
        break;
    }
  }

  requestAnimationFrame((t) => {
    last = t;
    requestAnimationFrame(frame);
  });
}

/** An error meaning "this browser can't", as opposed to "something broke". */
function unsupported(message: string): Error {
  const e = new Error(message);
  e.name = "Unsupported";
  return e;
}

/** Headless engines: the next frame as a PNG data URL. */
function capturePng(engine: Engine): Promise<string> {
  engine.requestCapture();
  return new Promise((resolve) => {
    const poll = () => {
      const raw = engine.capture();
      if (!raw) {
        requestAnimationFrame(poll);
        return;
      }
      const view = new DataView(raw.buffer, raw.byteOffset);
      const w = view.getUint32(0, true);
      const h = view.getUint32(4, true);
      const pixels = new Uint8ClampedArray(w * h * 4);
      pixels.set(raw.subarray(8, 8 + w * h * 4));
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      c.getContext("2d")!.putImageData(new ImageData(pixels, w, h), 0, 0);
      resolve(c.toDataURL("image/png"));
    };
    requestAnimationFrame(poll);
  });
}
