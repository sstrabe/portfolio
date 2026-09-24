import { defineConfig, type Plugin } from "vite";
import { portfolio } from "./src/content/portfolio.ts";
import { esc, renderPlain } from "./src/content/render.ts";

// Bake the plain portfolio into index.html so crawlers, browsers without
// WebGPU and visitors in a hurry get real content with no JS or Wasm.
function prerender(): Plugin {
  return {
    name: "prerender-plain",
    transformIndexHtml(html) {
      const description = portfolio.owner.headline;
      return html
        .replaceAll("%OWNER%", esc(portfolio.owner.name))
        .replaceAll("%DESCRIPTION%", esc(description))
        .replace("<!--plain-->", renderPlain(portfolio));
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [prerender()],
  build: {
    target: "es2022",
    assetsInlineLimit: 0,
  },
});
