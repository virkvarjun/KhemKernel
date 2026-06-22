import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "../lib/hooks";

const NK = 3; // number of k-tiles
type Phase = "load" | "sync1" | "compute" | "sync2";
interface Beat {
  k: number;
  phase: Phase;
  label: string;
}
const BEATS: Beat[] = [];
for (let k = 0; k < NK; k++) {
  BEATS.push({ k, phase: "load", label: `load A-tile ${k} and B-tile ${k}: global → shared memory` });
  BEATS.push({ k, phase: "sync1", label: "__syncthreads(): wait until the whole tile is loaded" });
  BEATS.push({ k, phase: "compute", label: `accumulate: acc += sA · sB over the 16-wide tile` });
  BEATS.push({ k, phase: "sync2", label: "__syncthreads(): wait before overwriting shared memory" });
}
BEATS.push({ k: NK, phase: "compute", label: "write the finished output tile back to C" });

export function TiledMatmul() {
  const reduced = usePrefersReducedMotion();
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const beat = BEATS[i];

  useEffect(() => {
    if (reduced) setI(BEATS.length - 1);
  }, [reduced]);
  useEffect(() => {
    if (!playing || reduced) return;
    if (i >= BEATS.length - 1) {
      setPlaying(false);
      return;
    }
    timer.current = window.setTimeout(() => setI((x) => x + 1), 760);
    return () => window.clearTimeout(timer.current);
  }, [playing, i, reduced]);

  const accFill = Math.min(1, (beat.k + (beat.phase === "compute" ? 1 : 0)) / NK);
  const loaded = beat.phase !== "load"; // shared mem filled after load completes within a beat
  const C = 16;

  return (
    <div className="card breakout">
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: "0.6rem" }}>
        <button className="btn" onClick={() => setPlaying((p) => !p)}>{playing ? "pause" : "play"}</button>
        <button className="btn ghost" onClick={() => { setPlaying(false); setI((x) => Math.min(BEATS.length - 1, x + 1)); }}>step</button>
        <button className="btn ghost" onClick={() => { setPlaying(false); setI(0); }}>reset</button>
        <span style={{ flex: 1 }} />
        <span className="schematic">schematic</span>
      </div>

      <svg viewBox="0 0 560 280" width="100%" role="img" aria-label="tiled matmul animation">
        {/* global memory band */}
        <rect x={10} y={14} width={540} height={56} rx={8} fill="var(--panel-2)" stroke="var(--panel-line)" />
        <text x={18} y={30} fontSize="10" fontFamily="var(--mono)" fill="var(--ink-faint)">global memory</text>
        {Array.from({ length: NK }).map((_, k) => (
          <g key={"A" + k}>
            <rect x={40 + k * 56} y={38} width={48} height={22} rx={4}
              fill={k === beat.k && beat.k < NK ? "var(--accent-soft)" : "var(--panel)"}
              stroke={k === beat.k && beat.k < NK ? "var(--accent)" : "var(--panel-line)"} />
            <text x={64 + k * 56} y={53} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--ink)">A{k}</text>
          </g>
        ))}
        {Array.from({ length: NK }).map((_, k) => (
          <g key={"B" + k}>
            <rect x={320 + k * 56} y={38} width={48} height={22} rx={4}
              fill={k === beat.k && beat.k < NK ? "var(--chem-soft)" : "var(--panel)"}
              stroke={k === beat.k && beat.k < NK ? "var(--chem)" : "var(--panel-line)"} />
            <text x={344 + k * 56} y={53} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--ink)">B{k}</text>
          </g>
        ))}

        {/* arrows to shared memory */}
        <line x1={64} y1={70} x2={150} y2={120} stroke="var(--accent)" strokeWidth={beat.phase === "load" ? 2.2 : 1} markerEnd="url(#tm)" opacity={beat.k < NK ? 1 : 0.3} />
        <line x1={344} y1={70} x2={250} y2={120} stroke="var(--chem)" strokeWidth={beat.phase === "load" ? 2.2 : 1} markerEnd="url(#tmc)" opacity={beat.k < NK ? 1 : 0.3} />

        {/* shared memory box */}
        <rect x={120} y={120} width={170} height={70} rx={8} fill="var(--panel)" stroke={beat.phase.startsWith("sync") ? "var(--accent)" : "var(--panel-line)"} strokeWidth={beat.phase.startsWith("sync") ? 2 : 1} />
        <text x={128} y={136} fontSize="10" fontFamily="var(--mono)" fill="var(--ink-faint)">shared memory (per block)</text>
        <g>
          <rect x={135} y={144} width={56} height={36} rx={4} fill={loaded && beat.k < NK ? "var(--accent-soft)" : "var(--panel-2)"} stroke="var(--accent)" />
          <text x={163} y={166} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--accent-ink)">sA = A{Math.min(beat.k, NK - 1)}</text>
          <rect x={205} y={144} width={56} height={36} rx={4} fill={loaded && beat.k < NK ? "var(--chem-soft)" : "var(--panel-2)"} stroke="var(--chem)" />
          <text x={233} y={166} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--chem)">sB = B{Math.min(beat.k, NK - 1)}</text>
        </g>

        {/* compute arrow */}
        <line x1={290} y1={155} x2={380} y2={170} stroke="var(--ink-faint)" strokeWidth={beat.phase === "compute" ? 2.2 : 1} markerEnd="url(#tm2)" />

        {/* output tile C 4x4 */}
        <text x={420} y={120} fontSize="10" fontFamily="var(--mono)" fill="var(--ink-faint)">output tile of C</text>
        {Array.from({ length: 4 }).map((_, r) =>
          Array.from({ length: 4 }).map((__, c) => (
            <rect key={`${r}-${c}`} x={390 + c * (C + 2)} y={128 + r * (C + 2)} width={C} height={C} rx={2}
              fill="var(--accent)" opacity={0.1 + 0.85 * accFill} />
          )),
        )}
        <text x={420} y={210} fontSize="11" fontFamily="var(--mono)" fill="var(--ink-soft)">acc after {Math.min(beat.k + (beat.phase === "compute" ? 1 : 0), NK)} / {NK} k-tiles</text>

        <defs>
          <marker id="tm" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--accent)" /></marker>
          <marker id="tmc" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--chem)" /></marker>
          <marker id="tm2" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--ink-faint)" /></marker>
        </defs>
      </svg>

      <div style={{ marginTop: "0.5rem", fontFamily: "var(--mono)", fontSize: "0.85rem", color: beat.phase.startsWith("sync") ? "var(--accent-ink)" : "var(--ink-soft)" }}>
        beat {i + 1}/{BEATS.length}: {beat.label}
      </div>
    </div>
  );
}
