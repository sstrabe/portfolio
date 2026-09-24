// Feature flags. Each has a default and can be overridden from the URL
// (`?flag=on` / `?flag=off`) for testing without a rebuild.

const params = new URLSearchParams(location.search);

function flag(name: string, fallback: boolean): boolean {
  const v = params.get(name);
  if (v === null) return fallback;
  return !["0", "off", "false", "no"].includes(v.toLowerCase());
}

export const flags = {
  /**
   * Hide every HTML control in the immersive mode (title bar, station list,
   * labels, HUD, dock panel, toasts, help, touch pad), leaving only the
   * rendered scene. Keyboard and mouse flight still work.
   * Re-enable with `?noGui=off`.
   */
  noGui: flag("noGui", true),
};
