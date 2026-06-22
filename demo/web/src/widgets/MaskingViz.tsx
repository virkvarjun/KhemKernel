import { useState } from "react";

type Mode = "causal" | "padding" | "combined";

// a short decoder sequence from the phenol trace, with two trailing pads
const TOKENS = ["<start>", "<parent>", "benzene", "</parent>", "<name>", "<pad>", "<pad>"];
const IS_PAD = TOKENS.map((t) => t === "<pad>");
const N = TOKENS.length;

// illustrative raw attention scores for one query row (schematic)
function rawScore(qi: number, j: number): number {
  return 1.4 - 0.18 * Math.abs(qi - j) + 0.15 * Math.cos(j * 1.7);
}

function softmax(xs: number[]): number[] {
  const mx = Math.max(...xs);
  const ex = xs.map((x) => (isFinite(x) ? Math.exp(x - mx) : 0));
  const s = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map((e) => e / s);
}

export function MaskingViz() {
  const [mode, setMode] = useState<Mode>("combined");
  const [qi, setQi] = useState(4);

  const blocked = (i: number, j: number) => {
    const causal = j > i;
    const pad = IS_PAD[j];
    if (mode === "causal") return causal;
    if (mode === "padding") return pad;
    return causal || pad;
  };

  const cell = 30;
  const pad = 22;
  const size = N * cell;

  const raw = TOKENS.map((_, j) => rawScore(qi, j));
  const before = softmax(raw);
  const after = softmax(raw.map((v, j) => (blocked(qi, j) ? -Infinity : v)));

  return (
    <div className="card breakout">
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
        {(["causal", "padding", "combined"] as Mode[]).map((m) => (
          <button key={m} className="btn ghost" onClick={() => setMode(m)} style={m === mode ? { borderColor: "var(--accent)", color: "var(--ink)" } : undefined}>
            {m}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        <span className="schematic">schematic scores</span>
      </div>

      <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "auto 1fr", marginTop: "1rem", alignItems: "start" }} className="mask-grid">
        <div style={{ overflowX: "auto" }}>
          <svg viewBox={`0 0 ${size + pad} ${size + pad}`} width={size + pad} height={size + pad} role="img" aria-label="additive mask grid">
            {TOKENS.map((t, j) => (
              <text key={"c" + j} x={pad + j * cell + cell / 2} y={pad - 7} textAnchor="middle" fontFamily="var(--mono)" fontSize="8.5" fill="var(--ink-faint)">
                {t.replace(/[<>]/g, "")}
              </text>
            ))}
            {TOKENS.map((t, i) => (
              <text key={"r" + i} x={pad - 5} y={pad + i * cell + cell / 2 + 3} textAnchor="end" fontFamily="var(--mono)" fontSize="8.5" fill={i === qi ? "var(--accent-ink)" : "var(--ink-faint)"}>
                {t.replace(/[<>]/g, "")}
              </text>
            ))}
            {Array.from({ length: N }).map((_, i) =>
              Array.from({ length: N }).map((__, j) => {
                const b = blocked(i, j);
                return (
                  <rect
                    key={`${i}-${j}`}
                    x={pad + j * cell}
                    y={pad + i * cell}
                    width={cell - 2}
                    height={cell - 2}
                    rx={3}
                    fill={b ? "var(--ink)" : "var(--accent)"}
                    opacity={b ? 0.82 : 0.16}
                    stroke={i === qi ? "var(--accent)" : "none"}
                    strokeWidth={i === qi ? 1.5 : 0}
                    onMouseEnter={() => setQi(i)}
                    style={{ cursor: "pointer" }}
                  >
                    <title>{`${TOKENS[i]} → ${TOKENS[j]}: ${b ? "blocked (−∞)" : "allowed (0)"}`}</title>
                  </rect>
                );
              }),
            )}
          </svg>
          <div style={{ fontSize: "0.78rem", color: "var(--ink-faint)", marginTop: 4 }}>
            dark = added −∞ (blocked), light = added 0 (allowed). Hover a row.
          </div>
        </div>

        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.78rem", color: "var(--ink-faint)" }}>
            softmax for query <code className="tokbox">{TOKENS[qi]}</code>
          </div>
          {[
            { label: "before mask", w: before, dim: false },
            { label: "after mask", w: after, dim: true },
          ].map((row) => (
            <div key={row.label} style={{ marginTop: 10 }}>
              <div style={{ fontSize: "0.8rem", color: "var(--ink-soft)", marginBottom: 3 }}>{row.label}</div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 60 }}>
                {row.w.map((w, j) => (
                  <div key={j} style={{ display: "flex", flexDirection: "column", alignItems: "center", flex: 1 }} title={`${TOKENS[j]}: ${(w * 100).toFixed(0)}%`}>
                    <div style={{ width: "70%", height: `${w * 50}px`, background: row.dim && blocked(qi, j) ? "var(--rule)" : "var(--accent)", borderRadius: 2 }} />
                    <span style={{ fontSize: 8, color: "var(--ink-faint)" }}>{TOKENS[j].replace(/[<>]/g, "").slice(0, 4)}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
          <div style={{ fontSize: "0.85rem", color: "var(--ink-faint)", marginTop: 8 }}>
            After adding the mask and re-normalizing, the blocked positions get
            exactly zero weight: the query cannot see the future or the padding.
          </div>
        </div>
      </div>
      <style>{`@media (max-width: 640px){ .mask-grid{ grid-template-columns: 1fr !important; } }`}</style>
    </div>
  );
}
