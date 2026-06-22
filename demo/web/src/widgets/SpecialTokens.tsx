import { useMemo, useState } from "react";
import { MOLECULES } from "../data/molecules";

// special pieces kept atomic before BPE: trace tags + the ";" separator,
// matched longest-first so "</parent>" wins over "<".
const SPECIALS = [
  "<parent>", "</parent>", "<groups>", "</groups>",
  "<atoms>", "</atoms>", "<rings>", "</rings>",
  "<name>", "</name>", ";",
].sort((a, b) => b.length - a.length);

const SPECIAL_RE = new RegExp(
  "(" + SPECIALS.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")",
);
const SPECIAL_SET = new Set(SPECIALS);

export function SpecialTokens() {
  const [key, setKey] = useState("phenol");
  const trace = MOLECULES.find((m) => m.key === key)?.trace ?? MOLECULES[1].trace;
  const pieces = useMemo(() => trace.split(SPECIAL_RE).filter((p) => p !== ""), [trace]);
  const lossless = pieces.join("") === trace;

  return (
    <div className="card breakout">
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
        {MOLECULES.map((m) => (
          <button key={m.key} className="btn ghost" onClick={() => setKey(m.key)} style={m.key === key ? { borderColor: "var(--accent)", color: "var(--ink)" } : undefined}>
            {m.label}
          </button>
        ))}
      </div>

      <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)", marginBottom: 6 }}>
        raw trace
      </div>
      <div style={{ fontFamily: "var(--mono)", fontSize: "0.82rem", color: "var(--ink-soft)", wordBreak: "break-word", marginBottom: 12 }}>
        {trace}
      </div>

      <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)", marginBottom: 6 }}>
        peeled: special tokens stay atomic, text between goes to BPE
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
        {pieces.map((p, i) =>
          SPECIAL_SET.has(p) ? (
            <code
              key={i}
              className="tokbox"
              title="special token (atomic)"
              style={{ color: "var(--tok-aromatic)", borderColor: "var(--tok-aromatic)" }}
            >
              {p}
            </code>
          ) : (
            <span
              key={i}
              className="tokbox"
              title="text chunk → byte pair encoded"
              style={{ background: "var(--panel-2)", color: "var(--ink)" }}
            >
              {p}
            </span>
          ),
        )}
      </div>

      <div style={{ marginTop: 14, display: "grid", gap: 8, gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" }}>
        {[
          ["<start>", "first decoder input; tells the model to begin"],
          ["<end>", "model emits it to stop; decoding halts here"],
          ["<pad>", "fills short sequences in a batch; masked in the loss"],
          ["<unk>", "the fallback; BPE is built so real names never hit it"],
        ].map(([t, job]) => (
          <div key={t} style={{ fontSize: "0.84rem" }}>
            <code className="tokbox" style={{ color: "var(--accent-ink)" }}>{t}</code>
            <div style={{ color: "var(--ink-faint)", marginTop: 2 }}>{job}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12, fontSize: "0.9rem", color: lossless ? "var(--good)" : "var(--warn)" }}>
        {lossless ? "✓" : "✗"} lossless: concatenating the pieces reproduces the
        exact trace, so a generated name survives intact to the verifier.
      </div>
    </div>
  );
}
