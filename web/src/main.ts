// Entry point. The plain page is already in the HTML; this only boots the
// immersive nucleus when the inline head script chose it, and falls back to
// the plain page if anything goes wrong.

const root = document.documentElement;

document.getElementById("enter-immersive")?.addEventListener("click", () => {
  try {
    localStorage.setItem("kerr-mode", "immersive");
  } catch {
    // Storage may be unavailable; the ?immersive link still works.
  }
});

function fallback(reason: unknown) {
  console.error("Immersive mode failed:", reason);
  root.dataset.mode = "plain";
  const host = document.getElementById("immersive");
  if (host) {
    host.hidden = true;
    host.replaceChildren();
  }
  const plain = document.getElementById("plain");
  if (plain && !document.getElementById("fallback-note")) {
    const note = document.createElement("p");
    note.id = "fallback-note";
    note.className = "fallback-note";
    note.textContent = `The interactive version could not start here (${reason instanceof Error ? reason.message : String(reason)}), so this is the plain version.`;
    plain.prepend(note);
  }
}

if (root.dataset.mode === "immersive") {
  const host = document.getElementById("immersive")!;
  host.hidden = false;
  import("./immersive/app")
    .then((m) => m.start(host, fallback))
    .catch(fallback);
}
