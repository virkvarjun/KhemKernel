"""Zero-dependency demo server for the picochem SMILES↔IUPAC model.

Loads the trained checkpoint + vocab once at startup and serves two endpoints:

    POST /api/smiles2iupac  {"smiles": "..."}  -> {"name", "trace", "ok"}
        Runs the trained transformer (greedy decode) on the host/CPU.
    POST /api/iupac2smiles  {"name": "..."}    -> {"smiles", "ok"}
        Uses OPSIN (py2opsin) — the same tool the eval uses for scoring.

The frontend is the interactive guide built from demo/web/ (Vite + React),
served as static files from demo/web/dist/. Build it once with:
    cd demo/web && npm install && npm run build

Run the server (from repo root, with the project venv):
    .venv/bin/python demo/server.py
then open http://localhost:8000

No external web framework required — only the trained model's deps
(numpy, and py2opsin + a Java runtime for the IUPAC→SMILES direction).
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── paths / imports ──────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Serve the built interactive guide (Vite app). Run `npm run build` in demo/web
# to produce it. The old static/ page has been replaced by this guide.
STATIC = os.path.join(HERE, "web", "dist")
sys.path.insert(0, ROOT)

import numpy as np

from picochem.checkpointing import load_checkpoint
from picochem.data import load_vocab, encode_smiles, decode_iupac
from picochem.model import greedy_decode, beam_decode
from picochem.evaluate import parse_trace, name_to_smiles, _canonicalize


def _default_ckpt():
    bpe = os.path.join(ROOT, "runs/device_bpe_d512_v2/ckpt_latest.npz")
    return bpe if os.path.isfile(bpe) else os.path.join(ROOT, "runs/device_full/ckpt_latest.npz")


def _default_bpe():
    p = os.path.join(ROOT, "data/iupac_bpe_v2.json")
    return p if os.path.isfile(p) else None


CHECKPOINT = os.environ.get("PICOCHEM_CKPT", _default_ckpt())
SMILES_VOCAB = os.path.join(ROOT, "data/smiles_vocab.json")
IUPAC_VOCAB = os.path.join(ROOT, "data/iupac_vocab.json")
PORT = int(os.environ.get("PORT", "8000"))

# Make sure a Homebrew-installed JDK is reachable so py2opsin (IUPAC→SMILES) works
# even when the user didn't put java on PATH themselves.
for _jdk in ("/opt/homebrew/opt/openjdk/bin", "/usr/local/opt/openjdk/bin"):
    if os.path.isdir(_jdk) and _jdk not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _jdk + os.pathsep + os.environ.get("PATH", "")


# ── model (loaded once) ──────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT}")
PARAMS, _, STEP, CONFIG = load_checkpoint(CHECKPOINT)
SMILES_STOI, SMILES_ITOS = load_vocab(SMILES_VOCAB)
_BPE_PATH = os.environ.get("PICOCHEM_IUPAC_BPE") or _default_bpe()
if _BPE_PATH:
    from picochem.bpe import BPETokenizer
    _bpe = BPETokenizer.load(_BPE_PATH)
    IUPAC_STOI, IUPAC_ITOS = _bpe.vocab, _bpe.itos  # itos joins losslessly in decode_iupac
    print(f"IUPAC tokenizer: BPE ({len(IUPAC_STOI)} tokens) from {_BPE_PATH}")
else:
    IUPAC_STOI, IUPAC_ITOS = load_vocab(IUPAC_VOCAB)
START_ID = IUPAC_STOI["<start>"]
END_ID = IUPAC_STOI["<end>"]
PAD_ID = IUPAC_STOI["<pad>"]
print(f"Model ready (step {STEP}, src_vocab {len(SMILES_STOI)}, tgt_vocab {len(IUPAC_STOI)})")

# OPSIN is optional; import lazily and report availability via /api/health.
try:
    from py2opsin import py2opsin as _py2opsin
    OPSIN_OK = True
except Exception:
    _py2opsin = None
    OPSIN_OK = False


BEAM_WIDTH = int(os.environ.get("PICOCHEM_BEAM", "20"))


def smiles_to_iupac(smiles):
    """Run the trained model on a SMILES string.

    Returns a dict: {name, trace, verified, opsin_smiles, decode}.

    When OPSIN is available we beam-decode and rerank: among the candidates we
    prefer the one whose name round-trips through OPSIN back to the *input*
    molecule ("verified"). Without OPSIN we fall back to plain greedy decode.
    """
    smiles = (smiles or "").strip()
    if not smiles:
        raise ValueError("empty SMILES")
    src_ids = encode_smiles(smiles, SMILES_STOI)
    if len(src_ids) == 0:
        raise ValueError("SMILES did not tokenize to any known tokens")
    if len(src_ids) > CONFIG["max_src_len"]:
        raise ValueError(f"SMILES too long ({len(src_ids)} > {CONFIG['max_src_len']} tokens)")
    src = src_ids[np.newaxis, :]
    src_mask = np.ones((1, len(src_ids)), dtype=np.float64)

    if not OPSIN_OK:
        gen_ids = greedy_decode(
            src, src_mask, PARAMS, CONFIG,
            start_token=START_ID, end_token=END_ID, pad_token=PAD_ID,
            max_length=CONFIG["max_tgt_len"],
        )
        trace = decode_iupac(np.array(gen_ids), IUPAC_ITOS)
        return {"name": parse_trace(trace), "trace": trace,
                "verified": False, "opsin_smiles": None, "decode": "greedy"}

    target = _canonicalize(smiles)
    beams = beam_decode(
        src, src_mask, PARAMS, CONFIG,
        start_token=START_ID, end_token=END_ID, pad_token=PAD_ID,
        max_length=CONFIG["max_tgt_len"], beam_width=BEAM_WIDTH,
    )
    cands = []
    for toks, _ in beams:
        text = decode_iupac(np.array(toks), IUPAC_ITOS)
        nm = parse_trace(text)
        sm = _canonicalize(name_to_smiles(nm)) if nm else None
        cands.append({"name": nm, "trace": text, "opsin_smiles": sm})

    verified = next((c for c in cands if c["opsin_smiles"] and c["opsin_smiles"] == target), None)
    parseable = next((c for c in cands if c["opsin_smiles"]), None)
    chosen = verified or parseable or cands[0]
    return {"name": chosen["name"], "trace": chosen["trace"],
            "verified": verified is not None,
            "opsin_smiles": chosen["opsin_smiles"], "decode": f"beam{BEAM_WIDTH}+rerank"}


def iupac_to_smiles(name):
    """Convert an IUPAC name -> SMILES via OPSIN."""
    name = (name or "").strip()
    if not name:
        raise ValueError("empty name")
    if not OPSIN_OK:
        raise RuntimeError("OPSIN (py2opsin) unavailable — install py2opsin and a Java runtime")
    smi = _py2opsin(name)
    if not smi or not smi.strip():
        raise ValueError("OPSIN could not parse that name")
    return smi.strip()


# ── HTTP handler ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    _MIME = {
        ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
        ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml",
        ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf",
        ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon",
        ".map": "application/json",
    }

    def _send_bytes(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel):
        # normalize, prevent path traversal, fall back to index.html (SPA)
        rel = rel.split("?", 1)[0].lstrip("/")
        path = os.path.normpath(os.path.join(STATIC, rel))
        if not path.startswith(STATIC):
            self._json(403, {"error": "forbidden"}); return
        if not os.path.isfile(path):
            index = os.path.join(STATIC, "index.html")
            if os.path.isfile(index):
                with open(index, "rb") as f:
                    self._send_bytes(f.read(), "text/html")
            else:
                self._json(404, {"error": "guide not built — run `npm run build` in demo/web"})
            return
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as f:
            self._send_bytes(f.read(), self._MIME.get(ext, "application/octet-stream"))

    def do_GET(self):
        if self.path == "/api/health":
            self._json(200, {"ok": True, "step": int(STEP), "opsin": OPSIN_OK})
        elif self.path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif self.path.startswith("/api/"):
            self._json(404, {"error": "not found"})
        else:
            self._serve_static(self.path)

    def do_POST(self):
        try:
            data = self._read_json()
            if self.path == "/api/smiles2iupac":
                res = smiles_to_iupac(data.get("smiles"))
                self._json(200, {"ok": True, **res})
            elif self.path == "/api/iupac2smiles":
                smi = iupac_to_smiles(data.get("name"))
                self._json(200, {"ok": True, "smiles": smi})
            else:
                self._json(404, {"error": "not found"})
        except Exception as e:
            self._json(400, {"ok": False, "error": str(e)})

    def log_message(self, fmt, *args):  # quieter logging
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\nServing demo at http://localhost:{PORT}  (Ctrl-C to stop)")
    print(f"  OPSIN (IUPAC→SMILES): {'available' if OPSIN_OK else 'UNAVAILABLE'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        srv.shutdown()


if __name__ == "__main__":
    main()
