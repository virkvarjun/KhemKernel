import { useMemo, useState } from "react";
import { MOLECULES } from "../data/molecules";
import { tokenizeSmiles, TOK_CLASS } from "../lib/smiles";
import { attentionMatrix, HEAD_NAMES } from "../lib/attentionSchematic";

export function AttentionHeatmap() {
  const [molKey, setMolKey] = useState("aspirin");
  const [head, setHead] = useState(2); // ring closures: a visibly structured head
  const [sel, setSel] = useState(0);

  const smiles = MOLECULES.find((m) => m.key === molKey)?.smiles ?? MOLECULES[3].smiles;
  const toks = useMemo(() => tokenizeSmiles(smiles), [smiles]);
  const mat = useMemo(() => attentionMatrix(smiles, head), [smiles, head]);
  const n = toks.length;
  const selRow = mat[Math.min(sel, n - 1)] ?? [];
  const selMax = Math.max(1e-6, ...selRow);

  const cell = 19;
  const pad = 26; // room for labels
  const size = n * cell;

  return (
    <div className="card breakout">
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
        {MOLECULES.map((m) => (
          <button key={m.key} className="btn ghost" onClick={() => { setMolKey(m.key); setSel(0); }} style={m.key === molKey ? { borderColor: "var(--accent)", color: "var(--ink)" } : undefined}>
            {m.label}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        <span className="schematic">schematic</span>
      </div>

      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "0.7rem 0" }}>
        {HEAD_NAMES.map((nm, i) => (
          <button key={i} className="btn ghost" onClick={() => setHead(i)} style={i === head ? { borderColor: "var(--accent)", color: "var(--ink)" } : undefined} title={nm}>
            head {i + 1}
          </button>
        ))}
      </div>
      <div style={{ fontSize: "0.85rem", color: "var(--ink-faint)", marginBottom: "0.5rem" }}>
        head {head + 1}: <em>{HEAD_NAMES[head]}</em>. Hover a row to read off what
        that query token attends to. This illustrates the mechanism; it is not
        measured model output.
      </div>

      <div style={{ overflowX: "auto" }}>
        <svg
          viewBox={`0 0 ${size + pad} ${size + pad}`}
          width={size + pad}
          height={size + pad}
          role="img"
          aria-label="attention heatmap"
          style={{ maxWidth: "100%" }}
        >
          {/* column (key) labels */}
          {toks.map((t, j) => (
            <text key={"c" + j} x={pad + j * cell + cell / 2} y={pad - 8} textAnchor="middle" fontFamily="var(--mono)" fontSize="10" className={TOK_CLASS[t.type]} fill="currentColor">
              {t.text}
            </text>
          ))}
          {/* row (query) labels */}
          {toks.map((t, i) => (
            <text key={"r" + i} x={pad - 6} y={pad + i * cell + cell / 2 + 3} textAnchor="end" fontFamily="var(--mono)" fontSize="10" className={TOK_CLASS[t.type]} fill="currentColor">
              {t.text}
            </text>
          ))}
          {/* cells */}
          {mat.map((row, i) => {
            const rmax = Math.max(1e-6, ...row);
            return row.map((w, j) => (
              <rect
                key={`${i}-${j}`}
                x={pad + j * cell}
                y={pad + i * cell}
                width={cell - 1.5}
                height={cell - 1.5}
                rx={2}
                fill="var(--chem)"
                opacity={0.05 + 0.95 * (w / rmax)}
                stroke={i === sel ? "var(--accent)" : "none"}
                strokeWidth={i === sel ? 1 : 0}
                onMouseEnter={() => setSel(i)}
                style={{ cursor: "pointer" }}
              >
                <title>{`${toks[i].text} → ${toks[j].text}: ${(w * 100).toFixed(1)}%`}</title>
              </rect>
            ));
          })}
        </svg>
      </div>

      {/* weight bar strip for the selected query */}
      <div style={{ marginTop: "0.6rem" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: "0.78rem", color: "var(--ink-faint)", marginBottom: 4 }}>
          query <code className="tokbox">{toks[Math.min(sel, n - 1)]?.text}</code> attends to:
        </div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 56, overflowX: "auto" }}>
          {selRow.map((w, j) => (
            <div key={j} style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: cell }} title={`${toks[j].text}: ${(w * 100).toFixed(1)}%`}>
              <div style={{ width: cell - 4, height: `${(w / selMax) * 44}px`, background: "var(--chem)", borderRadius: 2 }} />
              <span className={"tok " + TOK_CLASS[toks[j].type]} style={{ fontSize: 9 }}>{toks[j].text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
