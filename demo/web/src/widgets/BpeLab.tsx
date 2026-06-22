import { useEffect, useMemo, useRef, useState } from "react";
import { BPE_CORPUS } from "../data/molecules";
import { runTraining, encodeSteps, type Pair } from "../lib/bpe";
import { usePrefersReducedMotion } from "../lib/hooks";

type Mode = "train" | "encode";

const ENCODE_PRESETS = ["propanol", "butanoic acid", "hexanol", "benzoic acid"];

/** indices of tokens that participate in an adjacent match of `pair`. */
function matchSet(tokens: string[], pair: Pair | null): Set<number> {
  const s = new Set<number>();
  if (!pair) return s;
  for (let i = 0; i < tokens.length - 1; i++) {
    if (tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
      s.add(i);
      s.add(i + 1);
    }
  }
  return s;
}

function Tokens({ tokens, hi }: { tokens: string[]; hi?: Set<number> }) {
  return (
    <span style={{ display: "inline-flex", flexWrap: "wrap", gap: 3 }}>
      {tokens.map((t, i) => (
        <span
          key={i}
          className="tokbox"
          style={{
            background: hi?.has(i) ? "var(--accent-soft)" : "var(--panel)",
            borderColor: hi?.has(i) ? "var(--accent)" : "var(--panel-line)",
            color: t.length > 1 ? "var(--accent-ink)" : "var(--ink)",
            whiteSpace: "pre",
          }}
        >
          {t === " " ? "␣" : t}
        </span>
      ))}
    </span>
  );
}

export function BpeLab() {
  const reduced = usePrefersReducedMotion();
  const training = useMemo(() => runTraining(BPE_CORPUS, 80), []);
  const [mode, setMode] = useState<Mode>("train");

  // train state
  const [s, setS] = useState(0);
  // encode state
  const [word, setWord] = useState("propanol");
  const enc = useMemo(() => encodeSteps(word, training.ranks), [word, training.ranks]);
  const [es, setEs] = useState(0);

  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  const maxTrain = training.steps.length;
  const maxEnc = Math.max(0, enc.length - 1);

  // reset encode position when the word changes
  useEffect(() => {
    setEs(0);
    setPlaying(false);
  }, [word]);

  useEffect(() => {
    if (!playing || reduced) return;
    const atEnd = mode === "train" ? s >= maxTrain : es >= maxEnc;
    if (atEnd) {
      setPlaying(false);
      return;
    }
    timer.current = window.setTimeout(() => {
      if (mode === "train") setS((x) => Math.min(maxTrain, x + 1));
      else setEs((x) => Math.min(maxEnc, x + 1));
    }, 800);
    return () => window.clearTimeout(timer.current);
  }, [playing, s, es, mode, maxTrain, maxEnc, reduced]);

  const displayWords =
    s === 0
      ? training.steps[0]?.before ?? BPE_CORPUS.map((w) => ({ word: w, tokens: [...w] }))
      : training.steps[s - 1].after;
  const nextChosen = s < maxTrain ? training.steps[s].chosen : null;
  const rankedNow = s < maxTrain ? training.steps[s].ranked : [];
  const mergesSoFar = training.steps.slice(0, s).map((st) => st.newToken);

  return (
    <div className="card breakout">
      {/* mode + controls */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        <div style={{ display: "flex", gap: 6 }}>
          {(["train", "encode"] as Mode[]).map((m) => (
            <button
              key={m}
              className="btn ghost"
              onClick={() => {
                setMode(m);
                setPlaying(false);
              }}
              style={m === mode ? { borderColor: "var(--accent)", color: "var(--ink)" } : undefined}
            >
              {m} mode
            </button>
          ))}
        </div>
        <span style={{ flex: 1 }} />
        <button
          className="btn ghost"
          onClick={() => {
            setPlaying(false);
            if (mode === "train") setS(0);
            else setEs(0);
          }}
        >
          reset
        </button>
        <button
          className="btn ghost"
          onClick={() => {
            setPlaying(false);
            if (mode === "train") setS((x) => Math.min(maxTrain, x + 1));
            else setEs((x) => Math.min(maxEnc, x + 1));
          }}
        >
          step
        </button>
        <button className="btn" onClick={() => setPlaying((p) => !p)}>
          {playing ? "pause" : "play"}
        </button>
      </div>

      {mode === "train" ? (
        <div style={{ marginTop: "1rem", display: "grid", gap: "1rem", gridTemplateColumns: "minmax(0,1.5fr) minmax(0,1fr)" }} className="bpe-grid">
          <div>
            <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)", marginBottom: 6 }}>
              corpus &nbsp; (merge {s} / {maxTrain})
            </div>
            <div style={{ display: "grid", gap: 6 }}>
              {displayWords.map((w) => (
                <div key={w.word} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Tokens tokens={w.tokens} hi={matchSet(w.tokens, nextChosen)} />
                </div>
              ))}
            </div>
            {nextChosen && (
              <div style={{ marginTop: 10, fontSize: "0.9rem", color: "var(--ink-soft)" }}>
                next merge:{" "}
                <code className="tokbox" style={{ borderColor: "var(--accent)" }}>
                  {nextChosen[0] === " " ? "␣" : nextChosen[0]} + {nextChosen[1] === " " ? "␣" : nextChosen[1]}
                </code>{" "}
                &rarr;{" "}
                <code className="tokbox" style={{ color: "var(--accent-ink)" }}>
                  {(nextChosen[0] + nextChosen[1]).replace(/ /g, "␣")}
                </code>
              </div>
            )}
            {s >= maxTrain && (
              <div style={{ marginTop: 10, color: "var(--good)" }}>
                no adjacent pairs left to merge. Every word is now a single token.
              </div>
            )}
          </div>

          <div>
            <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)", marginBottom: 6 }}>
              pair counts this round
            </div>
            <div style={{ display: "grid", gap: 3 }}>
              {rankedNow.length === 0 && <span style={{ color: "var(--ink-faint)" }}>none</span>}
              {rankedNow.map((r, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontFamily: "var(--mono)",
                    fontSize: "0.82rem",
                    padding: "2px 6px",
                    borderRadius: 5,
                    background: i === 0 ? "var(--accent-soft)" : "transparent",
                    color: i === 0 ? "var(--accent-ink)" : "var(--ink-soft)",
                  }}
                >
                  <span>{(r.pair[0] + r.pair[1]).replace(/ /g, "␣")}</span>
                  <span>{r.count}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ gridColumn: "1 / -1" }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)", marginBottom: 6 }}>
              ordered merge list ({mergesSoFar.length})
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {mergesSoFar.length === 0 && <span style={{ color: "var(--ink-faint)" }}>none yet</span>}
              {mergesSoFar.map((m, i) => (
                <code key={i} className="tokbox" style={{ color: "var(--accent-ink)" }}>
                  {m.replace(/ /g, "␣")}
                </code>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div style={{ marginTop: "1rem" }}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
            {ENCODE_PRESETS.map((w) => (
              <button key={w} className="btn ghost" onClick={() => setWord(w)} style={word === w ? { borderColor: "var(--accent)", color: "var(--ink)" } : undefined}>
                {w}
                {!BPE_CORPUS.includes(w) ? " (unseen)" : ""}
              </button>
            ))}
            <input
              type="text"
              value={word}
              spellCheck={false}
              onChange={(e) => setWord(e.target.value)}
              aria-label="word to encode"
              style={{ fontFamily: "var(--mono)", padding: "0.3rem 0.5rem", border: "1px solid var(--panel-line)", borderRadius: 8, background: "var(--paper)", color: "var(--ink)", minWidth: 160 }}
            />
          </div>

          <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", color: "var(--ink-faint)", marginBottom: 6 }}>
            apply merges in rank order &nbsp; (step {es} / {maxEnc})
          </div>
          <div style={{ minHeight: "2.4rem" }}>
            <Tokens tokens={enc[es]?.tokens ?? [...word]} hi={matchSet(enc[es]?.tokens ?? [], enc[es]?.chosen ?? null)} />
          </div>
          {enc[es]?.chosen ? (
            <div style={{ marginTop: 10, fontSize: "0.9rem", color: "var(--ink-soft)" }}>
              merge rank #{enc[es].rank}:{" "}
              <code className="tokbox" style={{ borderColor: "var(--accent)" }}>
                {(enc[es].chosen![0] + enc[es].chosen![1]).replace(/ /g, "␣")}
              </code>
            </div>
          ) : (
            <div style={{ marginTop: 10, color: "var(--good)", fontSize: "0.92rem" }}>
              done. Multi-character tokens are learned morphemes; any single
              characters left are the fallback for fragments BPE never merged
              (this is why BPE never needs <code>&lt;unk&gt;</code>).
            </div>
          )}
        </div>
      )}

      <style>{`@media (max-width: 620px){ .bpe-grid{ grid-template-columns: 1fr !important; } }`}</style>
    </div>
  );
}
