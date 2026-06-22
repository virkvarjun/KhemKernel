# KhemKernel interactive guide

A single-page interactive guide to the KhemKernel project, from the chemistry
down to the hand-written CUDA kernels. Vite + React + TypeScript, Computer
Modern throughout, KaTeX for math, Prism for code. Every code snippet is pulled
verbatim from the real repository source (`?raw` imports), and the SMILES
tokenizer and BPE lab run the real algorithms in the browser.

## Develop

```bash
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/api/*` to `http://localhost:8000`, so for the live
inference panel run the Python backend alongside it (from the repo root):

```bash
.venv/bin/python demo/server.py
```

Without the backend the page still works fully; the "Try the real model" panel
detects this via `/api/health` and falls back to precomputed results for the
five guide molecules.

## Build

```bash
npm run build        # -> dist/  (static, no runtime backend, no network at view time)
npm run preview      # serve the build locally
npm run typecheck    # tsc --noEmit
```

`demo/server.py` serves `dist/` as its static root, so `npm run build` is all
that is needed before running the Python server.

## Deploy

The build in `dist/` is fully static and can be hosted anywhere (GitHub Pages,
Vercel, any static host). The base path is relative (`base: "./"` in
`vite.config.ts`), so it works both at a domain root and under a subpath.

The live-inference panel needs the Python backend and so only works when served
by `demo/server.py` locally; on a static host the guide degrades gracefully to
the precomputed examples. Everything else (all nine widgets, every diagram, the
tokenizers) is self-contained.

## Layout

```
src/
  toc.ts             table of contents (drives the sidebar + scroll spy)
  data/              molecules + real traces, vendored SMILES vocab, raw source imports
  lib/               smiles + bpe + iupac tokenizers, code extraction, hooks, api client
  components/        Layout, Toc, CodeBlock, Math, Figure, SmilesTokens, ui primitives
  widgets/           W1–W9 interactives + LiveDemo + AccuracyLadder
  sections/          Landing + Part I–VII + Appendix
  styles/            Computer Modern @font-face, design system, Prism theme
```
