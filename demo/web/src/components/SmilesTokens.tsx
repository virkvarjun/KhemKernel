import { tokenizeSmiles, TOK_CLASS, tokTypeLabel } from "../lib/smiles";

/** Render a SMILES string as colored, boxed tokens (type-consistent colors). */
export function SmilesTokens({
  smiles,
  boxed = true,
  highlight,
}: {
  smiles: string;
  boxed?: boolean;
  highlight?: number; // index to emphasize
}) {
  const toks = tokenizeSmiles(smiles);
  return (
    <span style={{ display: "inline-flex", flexWrap: "wrap", gap: boxed ? 2 : 0 }}>
      {toks.map((t, i) => (
        <span
          key={i}
          className={(boxed ? "tokbox " : "tok ") + TOK_CLASS[t.type]}
          title={`${t.text} · ${tokTypeLabel(t.type)}`}
          style={
            highlight === i
              ? { outline: "2px solid var(--accent)", outlineOffset: 1 }
              : undefined
          }
        >
          {t.text}
        </span>
      ))}
    </span>
  );
}
