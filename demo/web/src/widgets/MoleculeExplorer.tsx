import { useEffect, useMemo, useRef, useState } from "react";
import { MOLECULES, MOL_BY_KEY } from "../data/molecules";
import { tokenizeSmiles } from "../lib/smiles";
import { tokenizeIupac } from "../lib/iupac";
import { SmilesTokens } from "../components/SmilesTokens";
import { usePrefersReducedMotion } from "../lib/hooks";

type Mode = "full" | "encoder" | "decoder";

// deterministic pseudo value in [0,1] for the schematic encoder-memory heatmap
function cellVal(i: number, j: number): number {
  const x = Math.sin(i * 12.9898 + j * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

export function MoleculeExplorer({
  mode: modeProp = "full",
  lockMode = false,
  initial = "aspirin",
}: {
  mode?: Mode;
  lockMode?: boolean;
  initial?: string;
}) {
  const reduced = usePrefersReducedMotion();
  const [molKey, setMolKey] = useState(initial);
  const [mode, setMode] = useState<Mode>(modeProp);
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(true);
  const mol = MOL_BY_KEY[molKey] ?? MOLECULES[0];
  const traceTokens = useMemo(() => tokenizeIupac(mol.trace), [mol.trace]);
  const nSmiles = useMemo(() => tokenizeSmiles(mol.smiles).length, [mol.smiles]);

  // stage count per mode
  const maxStage =
    mode === "full" ? 6 : mode === "encoder" ? 2 : 1 + traceTokens.length;

  const timer = useRef<number | undefined>(undefined);

  // restart the animation whenever the molecule or mode changes
  useEffect(() => {
    if (reduced) {
      setStage(maxStage);
      setPlaying(false);
    } else {
      setStage(0);
      setPlaying(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [molKey, mode, reduced]);

  useEffect(() => {
    if (!playing) return;
    if (stage >= maxStage) {
      setPlaying(false);
      return;
    }
    const delay = mode === "decoder" ? 360 : 620;
    timer.current = window.setTimeout(() => setStage((s) => s + 1), delay);
    return () => window.clearTimeout(timer.current);
  }, [playing, stage, maxStage, mode]);

  const replay = () => {
    setStage(0);
    setPlaying(true);
  };

  return (
    <div className="card breakout" style={{ padding: "1rem 1.1rem" }}>
      {/* controls */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {MOLECULES.map((m) => (
            <button
              key={m.key}
              className={"btn ghost"}
              onClick={() => setMolKey(m.key)}
              style={
                m.key === molKey
                  ? { borderColor: "var(--accent)", color: "var(--ink)" }
                  : undefined
              }
            >
              {m.label}
            </button>
          ))}
        </div>
        <span style={{ flex: 1 }} />
        {!lockMode && (
          <div style={{ display: "flex", gap: 6 }}>
            {(["full", "encoder", "decoder"] as Mode[]).map((m) => (
              <button
                key={m}
                className="btn ghost"
                onClick={() => setMode(m)}
                style={
                  m === mode
                    ? { borderColor: "var(--accent)", color: "var(--ink)" }
                    : undefined
                }
              >
                {m}
              </button>
            ))}
          </div>
        )}
        <button className="btn" onClick={replay} disabled={playing}>
          {playing ? "running" : "replay"}
        </button>
      </div>

      <div style={{ marginTop: "1rem" }}>
        {mode === "full" && <FullView mol={mol} stage={stage} />}
        {mode === "encoder" && (
          <EncoderView mol={mol} stage={stage} nTokens={nSmiles} />
        )}
        {mode === "decoder" && (
          <DecoderView mol={mol} reveal={Math.max(0, stage - 1)} started={stage >= 1} traceTokens={traceTokens} />
        )}
      </div>
    </div>
  );
}

function Row({ k, children, on }: { k: string; children: React.ReactNode; on: boolean }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "7.5rem 1fr",
        gap: "0.6rem",
        alignItems: "baseline",
        padding: "0.35rem 0",
        opacity: on ? 1 : 0.18,
        transition: "opacity 0.3s ease",
      }}
    >
      <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--ink-faint)" }}>
        {k}
      </div>
      <div>{children}</div>
    </div>
  );
}

function FullView({ mol, stage }: { mol: (typeof MOLECULES)[number]; stage: number }) {
  return (
    <div>
      <Row k="smiles" on={stage >= 0}>
        <span style={{ fontFamily: "var(--mono)" }}>{mol.smiles}</span>
      </Row>
      <Row k="tokens" on={stage >= 1}>
        <SmilesTokens smiles={mol.smiles} />
      </Row>
      <div style={{ borderTop: "1px solid var(--rule)", margin: "0.5rem 0" }} />
      <Row k="parent" on={stage >= 2}>
        <code className="tokbox">{mol.parsed.parent}</code>
        <span style={{ color: "var(--ink-faint)", fontSize: "0.85rem", marginLeft: 8 }}>
          ring system or longest chain
        </span>
      </Row>
      <Row k="groups" on={stage >= 3}>
        {mol.parsed.groups.length ? (
          mol.parsed.groups.map((g) => (
            <code className="tokbox" key={g}>{g}</code>
          ))
        ) : (
          <code className="tokbox">none</code>
        )}
      </Row>
      <Row k="atoms" on={stage >= 4}>
        <code className="tokbox">{mol.parsed.atoms}</code>
        <span style={{ color: "var(--ink-faint)", fontSize: "0.85rem", marginLeft: 8 }}>
          heavy atoms
        </span>
      </Row>
      <Row k="rings" on={stage >= 5}>
        <code className="tokbox">{mol.parsed.rings}</code>
      </Row>
      <div style={{ borderTop: "1px solid var(--rule)", margin: "0.5rem 0" }} />
      <Row k="name" on={stage >= 6}>
        <strong style={{ color: "var(--accent-ink)", fontSize: "1.1rem" }}>
          {mol.iupac}
        </strong>
      </Row>
    </div>
  );
}

function EncoderView({
  mol,
  stage,
  nTokens,
}: {
  mol: (typeof MOLECULES)[number];
  stage: number;
  nTokens: number;
}) {
  const cols = 16;
  const cell = 15;
  const w = cols * cell;
  const h = nTokens * cell;
  return (
    <div>
      <Row k="smiles" on={stage >= 0}>
        <span style={{ fontFamily: "var(--mono)" }}>{mol.smiles}</span>
      </Row>
      <Row k="tokens" on={stage >= 1}>
        <SmilesTokens smiles={mol.smiles} />
      </Row>
      <div
        style={{
          opacity: stage >= 2 ? 1 : 0.18,
          transition: "opacity 0.3s ease",
          marginTop: "0.6rem",
        }}
      >
        <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", color: "var(--ink-faint)", marginBottom: 4 }}>
          encoder memory &nbsp;
          <span className="schematic">schematic</span>
          <span style={{ marginLeft: 8 }}>
            {nTokens} tokens &times; 512 dim (16 shown)
          </span>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} role="img" aria-label="encoder memory matrix">
          {Array.from({ length: nTokens }).map((_, i) =>
            Array.from({ length: cols }).map((__, j) => (
              <rect
                key={`${i}-${j}`}
                x={j * cell}
                y={i * cell}
                width={cell - 1.5}
                height={cell - 1.5}
                rx={2}
                fill="var(--chem)"
                opacity={0.12 + 0.8 * cellVal(i, j)}
              />
            )),
          )}
        </svg>
      </div>
    </div>
  );
}

function DecoderView({
  mol,
  reveal,
  started,
  traceTokens,
}: {
  mol: (typeof MOLECULES)[number];
  reveal: number;
  started: boolean;
  traceTokens: string[];
}) {
  const shown = traceTokens.slice(0, reveal);
  const done = reveal >= traceTokens.length;
  return (
    <div>
      <Row k="memory" on={started}>
        <span style={{ color: "var(--ink-faint)", fontSize: "0.9rem" }}>
          encoder memory for <code className="tokbox">{mol.smiles}</code> is
          ready; the decoder attends to it while writing.
        </span>
      </Row>
      <Row k="decoding" on={started}>
        <span style={{ fontFamily: "var(--mono)", fontSize: "0.95rem", lineHeight: 1.7 }}>
          {shown.map((t, i) => (
            <span key={i} style={{ color: "var(--ink)" }}>{t}</span>
          ))}
          {!done && (
            <span style={{ borderLeft: "2px solid var(--accent)", marginLeft: 1 }}>&nbsp;</span>
          )}
        </span>
      </Row>
      <Row k="name" on={done}>
        <strong style={{ color: "var(--accent-ink)", fontSize: "1.1rem" }}>
          {mol.iupac}
        </strong>
      </Row>
    </div>
  );
}
