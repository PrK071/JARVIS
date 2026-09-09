import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname);
const output = resolve(root, "dist", "server");
const [html, css, javascript] = await Promise.all([
  readFile(resolve(root, "index.html"), "utf8"),
  readFile(resolve(root, "styles.css"), "utf8"),
  readFile(resolve(root, "app.js"), "utf8"),
]);

const worker = `const files = ${JSON.stringify({
  "/": [html, "text/html; charset=utf-8"],
  "/index.html": [html, "text/html; charset=utf-8"],
  "/styles.css": [css, "text/css; charset=utf-8"],
  "/app.js": [javascript, "text/javascript; charset=utf-8"],
})};

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const file = files[url.pathname];

    if (!file) {
      return new Response("Página não encontrada.", {
        status: 404,
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    return new Response(file[0], {
      headers: {
        "content-type": file[1],
        "cache-control": url.pathname === "/" || url.pathname === "/index.html"
          ? "no-cache"
          : "public, max-age=3600",
        "x-content-type-options": "nosniff",
        "referrer-policy": "strict-origin-when-cross-origin",
      },
    });
  },
};
`;

await rm(resolve(root, "dist"), { recursive: true, force: true });
await mkdir(output, { recursive: true });
await writeFile(resolve(output, "index.js"), worker, "utf8");
