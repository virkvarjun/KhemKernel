import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "../lib/hooks";

interface BoxDef {
  id: string;
  label: string;
  sub?: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

const W = 200;
const H = 46;
const BOXES: BoxDef[] = [
  { id: "smiles", label: "SMILES", sub: "c1ccc(O)cc1", x: 40, y: 16, w: W, h: H },
  { id: "e0", label: "encoder embeddings", sub: "token + position", x: 40, y: 92, w: W, h: H },
  { id: "e1", label: "encoder block 1", x: 40, y: 158, w: W, h: 38 },
  { id: "e2", label: "encoder block 2", x: 40, y: 206, w: W, h: 38 },
  { id: "e3", label: "encoder block 3", x: 40, y: 254, w: W, h: 38 },
  { id: "mem", label: "encoder memory", sub: "S × 512", x: 40, y: 320, w: W, h: H },

  { id: "d0", label: "decoder embeddings", sub: "token + position", x: 520, y: 92, w: W, h: H },
  { id: "d1", label: "decoder block 1", sub: "self + cross + ffn", x: 520, y: 158, w: W, h: 38 },
  { id: "d2", label: "decoder block 2", x: 520, y: 206, w: W, h: 38 },
  { id: "d3", label: "decoder block 3", x: 520, y: 254, w: W, h: 38 },
  { id: "fln", label: "final LayerNorm", x: 520, y: 320, w: W, h: 38 },
  { id: "head", label: "output head (tied)", sub: "→ 4,000 logits", x: 520, y: 368, w: W, h: H },
  { id: "name", label: "phenol", x: 520, y: 444, w: W, h: H },
];
const BY_ID = Object.fromEntries(BOXES.map((b) => [b.id, b]));
const ORDER = ["smiles", "e0", "e1", "e2", "e3", "mem", "d0", "d1", "d2", "d3", "fln", "head", "name"];

function center(id: string) {
  const b = BY_ID[id];
  return { cx: b.x + b.w / 2, cy: b.y + b.h / 2, b };
}

export function ForwardFlow({ reverse = false }: { reverse?: boolean }) {
  const reduced = usePrefersReducedMotion();
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const order = reverse ? [...ORDER].reverse() : ORDER;
  const active = order[idx];

  useEffect(() => {
    if (reduced) setIdx(order.length - 1);
  }, [reduced]);

  useEffect(() => {
    if (!playing || reduced) return;
    if (idx >= order.length - 1) {
      setPlaying(false);
      return;
    }
    timer.current = window.setTimeout(() => setIdx((i) => i + 1), 620);
    return () => window.clearTimeout(timer.current);
  }, [playing, idx, reduced]);

  const seen = (id: string) => order.indexOf(id) <= idx;

  // straight column connectors
  const colEnc = ["smiles", "e0", "e1", "e2", "e3", "mem"];
  const colDec = ["d0", "d1", "d2", "d3", "fln", "head", "name"];
  const conn = (ids: string[]) =>
    ids.slice(0, -1).map((id, i) => {
      const a = center(id), b = center(ids[i + 1]);
      const on = seen(ids[i + 1]);
      return (
        <line key={id + ids[i + 1]} x1={a.cx} y1={a.b.y + a.b.h} x2={b.cx} y2={b.b.y} stroke={on ? "var(--accent)" : "var(--rule)"} strokeWidth={on ? 2 : 1.4} markerEnd="url(#ffar)" />
      );
    });

  return (
    <div className="card breakout">
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: "0.6rem" }}>
        <button className="btn" onClick={() => setPlaying((p) => !p)}>{playing ? "pause" : "play"}</button>
        <button className="btn ghost" onClick={() => { setPlaying(false); setIdx((i) => Math.min(order.length - 1, i + 1)); }}>step</button>
        <button className="btn ghost" onClick={() => { setPlaying(false); setIdx(0); }}>reset</button>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: "0.85rem", color: "var(--ink-faint)" }}>stage {idx + 1} / {order.length}</span>
      </div>

      <div style={{ overflowX: "auto" }}>
        <svg viewBox="0 0 760 510" width="100%" role="img" aria-label="full forward pass">
          <defs>
            <marker id="ffar" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--accent)" />
            </marker>
            <marker id="ffarc" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--chem)" />
            </marker>
          </defs>

          {conn(colEnc)}
          {conn(colDec)}

          {/* cross-attention arrows from memory into each decoder block */}
          {["d1", "d2", "d3"].map((d) => {
            const m = center("mem"), b = center(d);
            const on = seen(d);
            return (
              <path
                key={"x" + d}
                d={`M ${m.b.x + m.b.w} ${m.cy} C 400 ${m.cy}, 400 ${b.cy}, ${b.b.x} ${b.cy}`}
                fill="none"
                stroke={on ? "var(--chem)" : "var(--rule)"}
                strokeWidth={on ? 1.8 : 1.2}
                strokeDasharray="4 3"
                markerEnd="url(#ffarc)"
                opacity={on ? 0.9 : 0.5}
              />
            );
          })}

          {BOXES.map((b) => {
            const isActive = b.id === active;
            const done = seen(b.id);
            const result = b.id === "name" || b.id === "smiles";
            return (
              <g key={b.id}>
                <rect
                  x={b.x} y={b.y} width={b.w} height={b.h} rx={8}
                  fill={isActive ? "var(--accent-soft)" : done ? "var(--panel)" : "var(--panel-2)"}
                  stroke={isActive ? "var(--accent)" : done ? "var(--accent-soft)" : "var(--panel-line)"}
                  strokeWidth={isActive ? 2 : 1}
                />
                <text x={b.x + b.w / 2} y={b.y + (b.sub ? b.h / 2 - 3 : b.h / 2 + 4)} textAnchor="middle" fontSize="13" fill={result ? "var(--accent-ink)" : "var(--ink)"} fontFamily={result ? "var(--mono)" : "var(--serif)"} fontWeight={result ? 700 : 400}>
                  {b.label}
                </text>
                {b.sub && (
                  <text x={b.x + b.w / 2} y={b.y + b.h / 2 + 13} textAnchor="middle" fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">{b.sub}</text>
                )}
              </g>
            );
          })}

          <text x={140} y={388} textAnchor="middle" fontSize="11" fill="var(--ink-faint)" fontFamily="var(--mono)">encoder</text>
          <text x={620} y={500} textAnchor="middle" fontSize="11" fill="var(--chem)" fontFamily="var(--mono)">cross-attention reads the memory</text>
        </svg>
      </div>
    </div>
  );
}
