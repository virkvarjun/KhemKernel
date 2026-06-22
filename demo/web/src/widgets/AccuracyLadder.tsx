// Exact-structure-match accuracy ladder. Real numbers from the project README.
const BARS = [
  { label: "greedy (one pass)", v: 79.5 },
  { label: "beam 5 + verifier", v: 89.6 },
  { label: "beam 20 + verifier", v: 95.8 },
];

export function AccuracyLadder({ showValid = true }: { showValid?: boolean }) {
  const max = 100;
  return (
    <div className="card breakout">
      <div
        style={{
          fontFamily: "var(--mono)",
          fontSize: "0.74rem",
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          color: "var(--ink-faint)",
          marginBottom: "0.7rem",
        }}
      >
        exact structure match on 2,000 held-out molecules
      </div>
      <div style={{ display: "grid", gap: "0.55rem" }}>
        {BARS.map((b) => (
          <div
            key={b.label}
            style={{ display: "grid", gridTemplateColumns: "11rem 1fr", gap: "0.6rem", alignItems: "center" }}
            className="ladder-row"
          >
            <div style={{ fontSize: "0.9rem", color: "var(--ink-soft)" }}>{b.label}</div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
              <div
                style={{
                  height: 22,
                  width: `${(b.v / max) * 100}%`,
                  background: "var(--accent)",
                  borderRadius: 5,
                  minWidth: 4,
                }}
              />
              <strong style={{ color: "var(--accent-ink)" }}>{b.v}%</strong>
            </div>
          </div>
        ))}
      </div>
      {showValid && (
        <div style={{ marginTop: "0.8rem", fontSize: "0.9rem", color: "var(--ink-faint)" }}>
          Valid IUPAC name rate: <strong style={{ color: "var(--ink)" }}>97.9%</strong>.
          The verifier rerank is a free inference-time trick: it works because we
          can check our own answers (see Part VI).
        </div>
      )}
      <style>{`@media (max-width: 560px){ .ladder-row{ grid-template-columns: 1fr !important; } }`}</style>
    </div>
  );
}
