import { useEffect, useState } from "react";
import { MOLECULES, MOL_BY_KEY } from "../data/molecules";
import { SmilesTokens } from "../components/SmilesTokens";
import { health, smiles2iupac, iupac2smiles, type Health } from "../lib/api";

export function LiveDemo() {
  const [hp, setHp] = useState<Health | null | "checking">("checking");
  const [smiles, setSmiles] = useState(MOL_BY_KEY.aspirin.smiles);
  const [name, setName] = useState("2-acetyloxybenzoic acid");

  const [s2i, setS2i] = useState<null | { name: string; trace: string; verified: boolean; decode?: string; error?: string }>(null);
  const [i2s, setI2s] = useState<null | { smiles?: string; error?: string }>(null);
  const [busyS, setBusyS] = useState(false);
  const [busyI, setBusyI] = useState(false);

  useEffect(() => {
    health().then(setHp);
  }, []);

  const live = hp && hp !== "checking";

  async function runS2I() {
    setBusyS(true);
    setS2i(null);
    try {
      const r = await smiles2iupac(smiles);
      setS2i(r.ok ? r : { name: "", trace: "", verified: false, error: r.error || "failed" });
    } catch (e) {
      setS2i({ name: "", trace: "", verified: false, error: String(e) });
    } finally {
      setBusyS(false);
    }
  }
  async function runI2S() {
    setBusyI(true);
    setI2s(null);
    try {
      const r = await iupac2smiles(name);
      setI2s(r.ok ? { smiles: r.smiles } : { error: r.error || "failed" });
    } catch (e) {
      setI2s({ error: String(e) });
    } finally {
      setBusyI(false);
    }
  }

  // precomputed fallback for the SMILES box when there is no backend
  const fallback = MOLECULES.find((m) => m.smiles === smiles.trim());

  return (
    <div className="card breakout">
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: "0.8rem", flexWrap: "wrap" }}>
        <strong>Try the real model</strong>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: "0.72rem",
            padding: "0.15rem 0.5rem",
            borderRadius: 999,
            background: live ? "var(--accent-soft)" : "var(--warn-soft)",
            color: live ? "var(--accent-ink)" : "var(--warn)",
            border: `1px solid ${live ? "var(--accent)" : "var(--warn)"}`,
          }}
        >
          {hp === "checking" ? "checking backend…" : live ? `live · step ${hp.step}${hp.opsin ? " · OPSIN on" : ""}` : "backend offline"}
        </span>
      </div>

      {!live && hp !== "checking" && (
        <div className="aside warn" style={{ marginTop: 0 }}>
          <span className="label">no inference backend</span>
          Live inference needs the Python server. From the repo root run{" "}
          <code>.venv/bin/python demo/server.py</code> (see the README), then
          reload. Below, the SMILES box shows the precomputed result for the five
          guide molecules.
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }} className="demo-grid">
        {/* SMILES -> name */}
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)", marginBottom: 6 }}>
            SMILES → IUPAC name (the model)
          </div>
          <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: 6 }}>
            {MOLECULES.map((m) => (
              <button key={m.key} className="btn ghost" onClick={() => setSmiles(m.smiles)}>{m.label}</button>
            ))}
          </div>
          <textarea value={smiles} spellCheck={false} onChange={(e) => setSmiles(e.target.value)} rows={2}
            style={{ width: "100%", fontFamily: "var(--mono)", fontSize: "0.92rem", padding: "0.5rem", borderRadius: 8, border: "1px solid var(--panel-line)", background: "var(--paper)", color: "var(--ink)", resize: "vertical" }} />
          <div style={{ marginTop: 6 }}>
            <SmilesTokens smiles={smiles} />
          </div>
          <button className="btn" style={{ marginTop: 8 }} disabled={!live || busyS} onClick={runS2I}>
            {busyS ? "running…" : live ? "name it" : "needs backend"}
          </button>

          {live && s2i && (
            <div style={{ marginTop: 10 }}>
              {s2i.error ? (
                <div style={{ color: "var(--warn)" }}>{s2i.error}</div>
              ) : (
                <>
                  <div><strong style={{ color: "var(--accent-ink)", fontSize: "1.05rem" }}>{s2i.name}</strong>
                    {s2i.verified && <span style={{ marginLeft: 8, color: "var(--good)", fontSize: "0.85rem" }}>✓ verified</span>}
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: "0.76rem", color: "var(--ink-faint)", wordBreak: "break-word", marginTop: 4 }}>{s2i.trace}</div>
                </>
              )}
            </div>
          )}
          {!live && (
            <div style={{ marginTop: 10 }}>
              {fallback ? (
                <>
                  <div><strong style={{ color: "var(--accent-ink)", fontSize: "1.05rem" }}>{fallback.iupac}</strong>
                    <span style={{ marginLeft: 8, color: "var(--ink-faint)", fontSize: "0.8rem" }}>(precomputed)</span>
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: "0.76rem", color: "var(--ink-faint)", wordBreak: "break-word", marginTop: 4 }}>{fallback.trace}</div>
                </>
              ) : (
                <div style={{ color: "var(--ink-faint)", fontSize: "0.88rem" }}>Pick one of the five molecules above to see a precomputed result.</div>
              )}
            </div>
          )}
        </div>

        {/* name -> SMILES */}
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)", marginBottom: 6 }}>
            IUPAC name → SMILES (OPSIN, reverse check)
          </div>
          <input type="text" value={name} spellCheck={false} onChange={(e) => setName(e.target.value)}
            style={{ width: "100%", fontFamily: "var(--mono)", fontSize: "0.92rem", padding: "0.5rem", borderRadius: 8, border: "1px solid var(--panel-line)", background: "var(--paper)", color: "var(--ink)" }} />
          <button className="btn" style={{ marginTop: 8 }} disabled={!live || busyI} onClick={runI2S}>
            {busyI ? "running…" : live ? "parse it" : "needs backend"}
          </button>
          {live && i2s && (
            <div style={{ marginTop: 10 }}>
              {i2s.error ? <div style={{ color: "var(--warn)" }}>{i2s.error}</div> : (
                <SmilesTokens smiles={i2s.smiles || ""} />
              )}
            </div>
          )}
          <div style={{ marginTop: 10, fontSize: "0.84rem", color: "var(--ink-faint)" }}>
            This is the same OPSIN round-trip the verifier uses (Part VI): it turns
            a name back into a molecule, so you can sanity-check either direction.
          </div>
        </div>
      </div>
      <style>{`@media (max-width: 680px){ .demo-grid{ grid-template-columns: 1fr !important; } }`}</style>
    </div>
  );
}
