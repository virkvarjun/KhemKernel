import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../../");

// base "./" so the built site works both when served at root by demo/server.py
// and when deployed under a subpath (e.g. GitHub Pages /KhemKernel/).
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    // allow ?raw imports of the real source files that live above demo/web/
    fs: { allow: [here, repoRoot] },
    // forward the live-inference calls to the python demo backend in dev
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
