// Pure HTML renderers shared by the build-time prerender (plain page) and
// the immersive mode (docking panels). No DOM access, so they run in Node.

import type { Link, Portfolio, Project } from "./portfolio.ts";

const ESC: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ESC[c]);
}

function link(l: Link): string {
  const external = /^https?:/.test(l.href);
  const rel = external ? ' rel="noopener" target="_blank"' : "";
  return `<a href="${esc(l.href)}"${rel}>${esc(l.label)}</a>`;
}

export function renderProjectBody(p: Project): string {
  const meta = [p.year, p.role].filter(Boolean).map((m) => esc(m!)).join(" · ");
  return [
    meta ? `<p class="meta">${meta}</p>` : "",
    `<p class="tagline">${esc(p.tagline)}</p>`,
    ...p.description.map((d) => `<p>${esc(d)}</p>`),
    p.tags.length ? `<ul class="tags">${p.tags.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : "",
    p.links.length ? `<p class="links">${p.links.map(link).join(" ")}</p>` : "",
  ].join("\n");
}

export function renderAboutBody(p: Portfolio): string {
  return [
    `<p class="tagline">${esc(p.owner.headline)}</p>`,
    ...p.owner.bio.map((b) => `<p>${esc(b)}</p>`),
    p.owner.links.length ? `<p class="links">${p.owner.links.map(link).join(" ")}</p>` : "",
  ].join("\n");
}

/** The complete Wasm-free page body. */
export function renderPlain(p: Portfolio): string {
  const projects = p.projects
    .map(
      (proj) => `
      <article class="project" id="${esc(proj.id)}" style="--accent: ${esc(proj.accent)}">
        <h3>${esc(proj.title)}</h3>
        ${renderProjectBody(proj)}
      </article>`,
    )
    .join("\n");
  return `
  <div id="plain" class="plain">
    <header class="plain-header">
      <h1>${esc(p.owner.name)}</h1>
      ${renderAboutBody(p)}
      <p class="enter">
        <a href="?immersive" id="enter-immersive" class="button">Fly through the galactic nucleus</a>
        <span class="enter-note">Interactive, needs WebGPU.</span>
      </p>
    </header>
    <main>
      <h2>Projects</h2>
      ${projects}
    </main>
    <footer class="plain-footer">
      <p>© ${new Date().getFullYear()} ${esc(p.owner.name)}</p>
    </footer>
  </div>`;
}
