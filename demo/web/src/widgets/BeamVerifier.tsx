import { useEffect, useMemo, useState } from "react";
import { MOLECULES, MOL_BY_KEY } from "../data/molecules";
import { AccuracyLadder } from "./AccuracyLadder";

interface Cand {
  name: string;
  score: number; // length-normalized log prob (higher = better)
  opsin: boolean; // OPSIN parses it
  matches: boolean; // round-trips to the input molecule
}

// Illustrative beam candidates per molecule. The correct name is deliberately
// NOT always rank 1, to show why the verifier rerank helps. Schematic.
const CANDS: Record<string, Cand[]> = {
  ethanol: [
    { name: "ethanol", score: -0.12, opsin: true, matches: true },
    { name: "methanol", score: -0.34, opsin: true, matches: false },
    { name: "ethanal", score: -0.71, opsin: true, matches: false },
  ],
  phenol: [
    { name: "phenol", score: -0.1, opsin: true, matches: true },
    { name: "benzene", score: -0.4, opsin: true, matches: false },
    { name: "cyclohexanol", score: -0.83, opsin: true, matches: false },
  ],
  alanine: [
    { name: "3-aminopropanoic acid", score: -0.19, opsin: true, matches: false },
    { name: "2-aminopropanoic acid", score: -0.23, opsin: true, matches: true },
    { name: "2-aminobutanoic acid", score: -0.55, opsin: true, matches: false },
  ],
  aspirin: [
    { name: "4-acetyloxybenzoic acid", score: -0.21, opsin: true, matches: false },
    { name: "2-acetyloxybenzoic acid", score: -0.24, opsin: true, matches: true },
    { name: "2-acetyloxybenzaldehyd", score: -0.6, opsin: false, matches: false },
  ],
  ibuprofen: [
    { name: "2-[3-(2-methylpropyl)phenyl]propanoic acid", score: -0.27, opsin: true, matches: false },
    { name: "2-[4-(2-methylpropyl)phenyl]propanoic acid", score: -0.29, opsin: true, matches: true },
    { name: "2-[4-(2-methylpropyl)phenyl]ethanoic acid", score: -0.58, opsin: true, matches: false },
  ],
};

export function BeamVerifier() {
  const [molKey, setMolKey] = useState("aspirin");
  const [reveal, setReveal] = useState(0); // how many candidates verified
  const mol = MOL_BY_KEY[molKey];
  const cands = useMemo(() => [...CANDS[molKey]].sort((a, b) => b.score - a.score), [molKey]);

  useEffect(() => setReveal(0), [molKey]);

  // chosen = first (by score) that round-trips, else first that parses, else top
  const chosenIdx = (() => {
    const v = cands.findIndex((c) => c.matches);
    if (v >= 0) return v;
    const p = cands.findIndex((c) => c.opsin);
    return p >= 0 ? p : 0;
  })();
  const greedyWrong = chosenIdx !== 0;
  const done = reveal >= cands.length;

  return (
    <div className="card breakout">
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
        {MOLECULES.map((m) => (
          <button key={m.key} className="btn ghost" onClick={() => setMolKey(m.key)} style={m.key === molKey ? { borderColor: "var(--accent)", color: "var(--ink)" } : undefined}>
            {m.label}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        <span className="schematic">illustrative candidates</span>
      </div>

      {/* beam expansion schematic */}
      <svg viewBox="0 0 560 110" width="100%" role="img" aria-label="beam expansion" style={{ marginTop: "0.6rem" }}>
        <circle cx={40} cy={55} r={9} fill="var(--accent-soft)" stroke="var(--accent)" />
        <text x={40} y={80} textAnchor="middle" fontSize="9" fontFamily="var(--mono)" fill="var(--ink-faint)">&lt;start&gt;</text>
        {[0, 1, 2].map((r) => (
          <g key={r}>
            <line x1={49} y1={55} x2={140} y2={25 + r * 30} stroke="var(--rule)" />
            <line x1={150} y1={25 + r * 30} x2={250} y2={25 + r * 30} stroke="var(--rule)" />
            <circle cx={145} cy={25 + r * 30} r={6} fill="var(--panel)" stroke="var(--panel-line)" />
          </g>
        ))}
        <text x={150} y={104} fontSize="9" fontFamily="var(--mono)" fill="var(--ink-faint)">keep top-k each step …</text>
        <text x={330} y={59} fontSize="11" fontFamily="var(--mono)" fill="var(--ink-soft)">→ final candidates, ranked by length-normalized log prob</text>
      </svg>

      {/* candidates + verifier */}
      <div style={{ display: "grid", gap: 6, marginTop: "0.4rem" }}>
        {cands.map((c, i) => {
          const shown = i < reveal;
          const isChosen = done && i === chosenIdx;
          return (
            <div key={i} className="card" style={{ padding: "0.5rem 0.7rem", borderColor: isChosen ? "var(--good)" : "var(--panel-line)", background: isChosen ? "var(--accent-soft)" : "var(--panel)" }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
                <span style={{ fontFamily: "var(--mono)", fontSize: "0.7rem", color: "var(--ink-faint)" }}>#{i + 1}</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: "0.88rem", flex: 1, minWidth: 180 }}>{c.name}</span>
                <span style={{ fontFamily: "var(--mono)", fontSize: "0.78rem", color: "var(--ink-faint)" }}>score {c.score.toFixed(2)}</span>
              </div>
              {shown && (
                <div style={{ marginTop: 4, fontSize: "0.82rem", color: "var(--ink-soft)" }}>
                  OPSIN: {c.opsin ? "parses" : <span style={{ color: "var(--warn)" }}>fails to parse</span>}
                  {c.opsin && (
                    <>
                      {" → "}
                      round-trips to input{" "}
                      {c.matches ? (
                        <span style={{ color: "var(--good)" }}>✓ same molecule</span>
                      ) : (
                        <span style={{ color: "var(--warn)" }}>✗ different molecule</span>
                      )}
                    </>
                  )}
                  {isChosen && <strong style={{ color: "var(--good)" }}> ← kept</strong>}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: "0.7rem", alignItems: "center", flexWrap: "wrap" }}>
        <button className="btn" onClick={() => setReveal((r) => Math.min(cands.length, r + 1))} disabled={done}>
          {reveal === 0 ? "run verifier" : done ? "verified" : "verify next"}
        </button>
        <button className="btn ghost" onClick={() => setReveal(0)}>reset</button>
        {done && (
          <span style={{ fontSize: "0.88rem", color: "var(--ink-soft)" }}>
            input: <code className="tokbox">{mol.smiles}</code>
          </span>
        )}
      </div>

      {done && greedyWrong && (
        <div style={{ marginTop: "0.6rem", fontSize: "0.9rem", color: "var(--ink-soft)" }}>
          Note: the top-scored candidate here is a positional isomer that parses
          but is the wrong molecule. Greedy would have taken it. The verifier
          keeps candidate #{chosenIdx + 1} instead, because it is the one that
          round-trips back to the input. That is the whole trick.
        </div>
      )}

      <div style={{ marginTop: "1rem" }}>
        <AccuracyLadder showValid={false} />
      </div>
    </div>
  );
}
