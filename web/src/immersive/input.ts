// Keyboard, mouse and touch → the engine's per-frame input vector:
// [thrust forward, left, up, turn roll, pitch, yaw, boost, brake,
//  autopilot station or -1, undock].
//
// Ship axes: forward, left, up (right-handed). Positive pitch is nose
// down, positive yaw turns left, positive roll lifts the left wing.

const MOUSE_RAD_PER_PX = 0.0035;
const TURN_RATE = 1.4; // must match WorldConfig::turn_rate (rad / wall s)

export class Controls {
  readonly vector = new Float32Array(10);
  autopilot = -1;
  onAutopilotCancelled: () => void = () => {};
  onHelp: () => void = () => {};
  private keys = new Set<string>();
  private dragX = 0;
  private dragY = 0;
  private dragging: number | null = null;
  private lastX = 0;
  private lastY = 0;
  private undock = false;
  private touchThrust = 0;
  private touchBoost = false;
  private touchBrake = false;

  constructor(surface: HTMLElement) {
    addEventListener("keydown", (e) => this.key(e, true));
    addEventListener("keyup", (e) => this.key(e, false));
    addEventListener("blur", () => this.keys.clear());
    surface.addEventListener("pointerdown", (e) => {
      if (this.dragging !== null) return;
      this.dragging = e.pointerId;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      surface.setPointerCapture(e.pointerId);
    });
    surface.addEventListener("pointermove", (e) => {
      if (e.pointerId !== this.dragging) return;
      this.dragX += e.clientX - this.lastX;
      this.dragY += e.clientY - this.lastY;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });
    const end = (e: PointerEvent) => {
      if (e.pointerId === this.dragging) this.dragging = null;
    };
    surface.addEventListener("pointerup", end);
    surface.addEventListener("pointercancel", end);
  }

  /** Hold-to-press buttons of the touch pad. */
  bindTouchButton(el: HTMLElement, kind: "forward" | "back" | "boost" | "brake") {
    const set = (on: boolean) => {
      if (kind === "forward") this.touchThrust = on ? 1 : 0;
      if (kind === "back") this.touchThrust = on ? -1 : 0;
      if (kind === "boost") this.touchBoost = on;
      if (kind === "brake") this.touchBrake = on;
      if (on && (kind === "forward" || kind === "back")) this.cancelAutopilot();
    };
    el.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      el.setPointerCapture(e.pointerId);
      set(true);
    });
    el.addEventListener("pointerup", () => set(false));
    el.addEventListener("pointercancel", () => set(false));
  }

  requestUndock() {
    this.undock = true;
  }

  flyTo(station: number) {
    this.autopilot = station;
  }

  private cancelAutopilot() {
    if (this.autopilot >= 0) {
      this.autopilot = -1;
      this.onAutopilotCancelled();
    }
  }

  private key(e: KeyboardEvent, down: boolean) {
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    const k = e.code;
    if (down) {
      if (/^Digit[1-8]$/.test(k)) {
        this.flyTo(Number(k.slice(5)) - 1);
        return;
      }
      if (k === "Escape") {
        if (this.autopilot >= 0) this.cancelAutopilot();
        else this.undock = true;
        return;
      }
      if (k === "KeyH" || k === "Slash") {
        this.onHelp();
        return;
      }
      if (["KeyW", "KeyS", "KeyA", "KeyD", "KeyR", "KeyF", "Space", "KeyC"].includes(k)) this.cancelAutopilot();
      if (k === "Space" || k.startsWith("Arrow")) e.preventDefault();
      this.keys.add(k);
    } else {
      this.keys.delete(k);
    }
  }

  private axis(pos: string[], neg: string[]): number {
    const p = pos.some((k) => this.keys.has(k)) ? 1 : 0;
    const n = neg.some((k) => this.keys.has(k)) ? 1 : 0;
    return p - n;
  }

  /**
   * Fill and return the input vector for a frame of `dt` seconds. Looking
   * around never cancels the autopilot; manual thrust does.
   */
  sample(dt: number): Float32Array {
    const v = this.vector;
    v[0] = this.axis(["KeyW"], ["KeyS"]) || this.touchThrust;
    v[1] = this.axis(["KeyA"], ["KeyD"]);
    v[2] = this.axis(["KeyR", "Space"], ["KeyF", "KeyC"]);
    // Mouse drag: turn by an angle proportional to the distance dragged.
    const k = dt > 0 ? MOUSE_RAD_PER_PX / (dt * TURN_RATE) : 0;
    v[3] = this.axis(["KeyE"], ["KeyQ"]);
    v[4] = this.axis(["ArrowDown"], ["ArrowUp"]) + this.dragY * k;
    v[5] = this.axis(["ArrowLeft"], ["ArrowRight"]) - this.dragX * k;
    this.dragX = 0;
    this.dragY = 0;
    v[6] = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") || this.touchBoost ? 1 : 0;
    v[7] = this.keys.has("KeyX") || this.touchBrake ? 1 : 0;
    v[8] = this.autopilot;
    v[9] = this.undock ? 1 : 0;
    this.undock = false;
    return v;
  }
}
