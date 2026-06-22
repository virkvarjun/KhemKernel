import { useState } from "react";

const X = [2.1, -0.5, 1.3, 3.0, -1.2, 0.4, 2.6, -0.8];
const EPS = 1e-5;

function Bars({ vals, color, zero = 0 }: { vals: number[]; color: string; zero?: number }) {
  const max = Math.max(1, ...vals.map((v) => Math.abs(v)));
  const h = 90;
  return (
    <svg viewBox={`0 0 ${vals.length * 26} ${h}`} width="100%" height={h} preserveAspectRatio="xMidYMid meet" role="img">
      <line x1={0} y1={h / 2} x2={vals.length * 26} y2={h / 2} stroke="var(--rule)" strokeWidth={1} />
      {vals.map((v, i) => {
        const bh = (Math.abs(v) / max) * (h / 2 - 6);
        const y = v >= 0 ? h / 2 - bh : h / 2;
        return (
          <rect key={i} x={i * 26 + 4} y={y} width={18} height={bh} rx={2} fill={color}>
            <title>{v.toFixed(2)}</title>
          </rect>
        );
      })}
    </svg>
  );
}

export function LayerNormViz() {
  const [gamma, setGamma] = useState(1);
  const [beta, setBeta] = useState(0);

  const mean = X.reduce((a, b) => a + b, 0) / X.length;
  const variance = X.reduce((a, b) => a + (b - mean) ** 2, 0) / X.length;
  const invStd = 1 / Math.sqrt(variance + EPS);
  const xhat = X.map((v) => (v - mean) * invStd);
  const y = xhat.map((v) => gamma * v + beta);

  return (
    <div className="card breakout">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)" }}>input x</div>
          <Bars vals={X} color="var(--ink-faint)" />
          <div style={{ fontSize: "0.8rem", color: "var(--ink-soft)" }}>
            mean {mean.toFixed(2)}, var {variance.toFixed(2)}
          </div>
        </div>
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)" }}>normalized x̂</div>
          <Bars vals={xhat} color="var(--chem)" />
          <div style={{ fontSize: "0.8rem", color: "var(--ink-soft)" }}>mean 0, var 1</div>
        </div>
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)" }}>output γ·x̂ + β</div>
          <Bars vals={y} color="var(--accent)" />
          <div style={{ fontSize: "0.8rem", color: "var(--ink-soft)" }}>γ {gamma.toFixed(2)}, β {beta.toFixed(2)}</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem", marginTop: "0.8rem" }}>
        <label style={{ fontSize: "0.85rem" }}>
          scale γ: {gamma.toFixed(2)}
          <input type="range" min={0} max={2} step={0.05} value={gamma} onChange={(e) => setGamma(parseFloat(e.target.value))} style={{ width: "100%" }} />
        </label>
        <label style={{ fontSize: "0.85rem" }}>
          shift β: {beta.toFixed(2)}
          <input type="range" min={-1.5} max={1.5} step={0.05} value={beta} onChange={(e) => setBeta(parseFloat(e.target.value))} style={{ width: "100%" }} />
        </label>
      </div>
      <div style={{ fontSize: "0.85rem", color: "var(--ink-faint)", marginTop: "0.5rem" }}>
        Normalization fixes mean 0 and variance 1 across the features of one
        position. The learned scale γ and shift β then let the layer put back any
        spread or offset it actually wants, so normalization does not cost the
        model expressiveness.
      </div>
    </div>
  );
}
