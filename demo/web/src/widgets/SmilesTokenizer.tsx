import { useMemo, useState } from "react";
import { tokenizeSmiles, TOK_CLASS, tokTypeLabel } from "../lib/smiles";
import { inVocab, SMILES_VOCAB_SIZE } from "../data/smilesVocab";
import { MOLECULES } from "../data/molecules";

const TYPES: { type: keyof typeof TOK_CLASS; label: string }[] = [
  { type: "atom", label: "atom" },
  { type: "aromatic", label: "aromatic" },
  { type: "bond", label: "bond" },
  { type: "branch", label: "branch ( )" },
  { type: "ring", label: "ring digit" },
  { type: "bracket", label: "bracket atom" },
];

export function SmilesTokenizer() {
  const [text, setText] = useState(MOLECULES[3].smiles); // aspirin
  const toks = useMemo(() => tokenizeSmiles(text), [text]);

  return (
    <div className="card breakout">
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
        {MOLECULES.map((m) => (
          <button key={m.key} className="btn ghost" onClick={() => setText(m.smiles)}>
            {m.label}
          </button>
        ))}
      </div>
      <input
        type="text"
        value={text}
        spellCheck={false}
        onChange={(e) => setText(e.target.value)}
        aria-label="SMILES input"
        style={{
          width: "100%",
          fontFamily: "var(--mono)",
          fontSize: "1rem",
          padding: "0.55rem 0.7rem",
          border: "1px solid var(--panel-line)",
          borderRadius: 8,
          background: "var(--paper)",
          color: "var(--ink)",
        }}
      />

      <div style={{ marginTop: "0.9rem", display: "flex", flexWrap: "wrap", gap: 4, minHeight: "2.4rem" }}>
        {toks.length === 0 && (
          <span style={{ color: "var(--ink-faint)" }}>type a SMILES string</span>
        )}
        {toks.map((t, i) => {
          const known = inVocab(t.text);
          return (
            <span
              key={i}
              className={"tokbox " + TOK_CLASS[t.type]}
              title={`${t.text} · ${tokTypeLabel(t.type)} · ${known ? "in vocab" : "out of vocab (→ <unk>)"}`}
              style={{
                borderStyle: known ? "solid" : "dashed",
                borderColor: known ? "var(--panel-line)" : "var(--warn)",
              }}
            >
              {t.text}
            </span>
          );
        })}
      </div>

      <div style={{ marginTop: "0.7rem", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8, fontSize: "0.85rem", color: "var(--ink-faint)" }}>
        <span>
          <strong style={{ color: "var(--ink)" }}>{toks.length}</strong> tokens
          {" · "}
          <strong style={{ color: "var(--ink)" }}>
            {toks.filter((t) => !inVocab(t.text)).length}
          </strong>{" "}
          out of vocab
        </span>
        <span>vocabulary: {SMILES_VOCAB_SIZE} tokens</span>
      </div>

      <div style={{ marginTop: "0.7rem", display: "flex", flexWrap: "wrap", gap: "0.5rem 1rem" }}>
        {TYPES.map((t) => (
          <span key={t.type} style={{ fontSize: "0.78rem", display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span
              className={TOK_CLASS[t.type]}
              style={{ width: 11, height: 11, borderRadius: 3, background: "currentColor", display: "inline-block" }}
            />
            <span style={{ color: "var(--ink-soft)" }}>{t.label}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
