import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Figure } from "../components/Figure";
import { Math as Tex } from "../components/Math";
import { RAW } from "../data/raw";
import { lines } from "../lib/code";
import { TOK_COUNTS, BPE_VOCAB, BPE_MERGES, BPE_PIECES } from "../data/measured";
import { tokenizeSmiles, TOK_CLASS } from "../lib/smiles";

/* ---- p8-1 figure: compression bars ---- */
function CompressionFig() {
  const maxc = Math.max(...TOK_COUNTS.map((t) => t.chars));
  return (
    <div style={{ display: "grid", gap: "0.6rem" }}>
      {TOK_COUNTS.map((t) => (
        <div key={t.name}>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.78rem", color: "var(--ink-soft)", marginBottom: 2, wordBreak: "break-word" }}>
            {t.name}
          </div>
          {([
            ["chars", t.chars, "var(--ink-faint)"],
            ["word", t.word, "var(--chem)"],
            ["BPE", t.bpe, "var(--accent)"],
          ] as const).map(([lab, v, c]) => (
            <div key={lab} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.78rem" }}>
              <span style={{ width: "2.6rem", color: "var(--ink-faint)", fontFamily: "var(--mono)" }}>{lab}</span>
              <div style={{ height: 11, width: `${(v / maxc) * 100}%`, minWidth: 3, background: c, borderRadius: 3 }} />
              <span style={{ fontFamily: "var(--mono)", color: "var(--ink-soft)" }}>{v}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function Pieces({ name }: { name: keyof typeof BPE_PIECES }) {
  return (
    <span style={{ display: "inline-flex", flexWrap: "wrap", gap: 3 }}>
      {BPE_PIECES[name].map((p, i) => (
        <code key={i} className="tokbox" style={{ color: "var(--accent-ink)" }}>{p}</code>
      ))}
    </span>
  );
}

/* ---- p8-3 figure: saturated vs scaled softmax ---- */
function softmax(xs: number[]) {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}
function ScaleFig() {
  const raw = [2.1, -0.4, 1.3, 0.2, 1.0];
  const dk = 64;
  const unscaled = softmax(raw.map((x) => x * Math.sqrt(dk))); // logits ~ *8 -> saturated
  const scaled = softmax(raw); // already /sqrt(dk) in effect
  const Row = ({ w, label }: { w: number[]; label: string }) => (
    <div>
      <div style={{ fontSize: "0.78rem", color: "var(--ink-soft)", marginBottom: 3 }}>{label}</div>
      <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 50 }}>
        {w.map((v, i) => (
          <div key={i} style={{ flex: 1, height: `${v * 46 + 1}px`, background: "var(--accent)", borderRadius: 2 }} title={v.toFixed(2)} />
        ))}
      </div>
    </div>
  );
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.2rem" }}>
      <Row w={unscaled} label="without √dₖ: nearly one-hot (saturated, tiny gradient)" />
      <Row w={scaled} label="with √dₖ: responsive distribution" />
    </div>
  );
}

/* ---- p8-5 cross-attention alignment (schematic, structurally faithful) ---- */
function AlignmentFig() {
  const toks = tokenizeSmiles("CC(=O)Oc1ccccc1C(=O)O");
  const rows = BPE_PIECES.aspirin; // ['2-','acetyloxy','benzoic acid']
  // hand-set, chemically sensible alignment weights per (fragment, token index)
  const weight = (r: number, j: number): number => {
    const t = toks[j];
    if (r === 0) return j <= 7 ? 0.25 : 0.05; // locant: diffuse, slightly toward the substituted side
    if (r === 1) return j <= 6 ? 0.9 - 0.05 * j : 0.05; // acetyloxy -> the CC(=O)O ester atoms
    // benzoic acid -> the aromatic ring + the carboxyl carbon/oxygens
    if (t.type === "aromatic" || j >= 15) return 0.85;
    return 0.08;
  };
  const cell = 17;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg viewBox={`0 0 ${toks.length * cell + 110} ${rows.length * cell + 30}`} width="100%" role="img" aria-label="cross-attention alignment">
        {toks.map((t, j) => (
          <text key={j} x={110 + j * cell + cell / 2} y={16} textAnchor="middle" fontFamily="var(--mono)" fontSize="9" className={TOK_CLASS[t.type]} fill="currentColor">{t.text}</text>
        ))}
        {rows.map((r, i) => (
          <text key={"r" + i} x={104} y={30 + i * cell + cell / 2} textAnchor="end" fontFamily="var(--mono)" fontSize="9.5" fill="var(--accent-ink)">{r}</text>
        ))}
        {rows.map((_, i) =>
          toks.map((__, j) => (
            <rect key={`${i}-${j}`} x={110 + j * cell} y={24 + i * cell} width={cell - 1.5} height={cell - 1.5} rx={2} fill="var(--chem)" opacity={0.06 + 0.94 * weight(i, j)} />
          )),
        )}
      </svg>
    </div>
  );
}

/* ---- p8-6 per-head fingerprint (schematic) ---- */
const HEADS = [
  { h: 1, role: "local / adjacency", entropy: 0.42 },
  { h: 2, role: "previous token", entropy: 0.31 },
  { h: 3, role: "ring closures", entropy: 0.55 },
  { h: 4, role: "branch matching", entropy: 0.6 },
  { h: 5, role: "heteroatom neighbor", entropy: 0.48 },
  { h: 6, role: "attend-to-start", entropy: 0.5 },
  { h: 7, role: "look back", entropy: 0.7 },
  { h: 8, role: "diffuse / averaging", entropy: 0.95 },
];
function HeadFig() {
  return (
    <div style={{ display: "grid", gap: 4 }}>
      {HEADS.map((h) => (
        <div key={h.h} style={{ display: "grid", gridTemplateColumns: "3rem 9rem 1fr 2.4rem", gap: 8, alignItems: "center", fontSize: "0.8rem" }}>
          <span style={{ fontFamily: "var(--mono)", color: "var(--ink-faint)" }}>head {h.h}</span>
          <span style={{ color: "var(--ink-soft)" }}>{h.role}</span>
          <div style={{ height: 9, width: `${h.entropy * 100}%`, background: "var(--chem)", borderRadius: 3 }} />
          <span style={{ fontFamily: "var(--mono)", color: "var(--ink-faint)", fontSize: "0.72rem" }}>H={h.entropy}</span>
        </div>
      ))}
    </div>
  );
}

export function PartVIII() {
  return (
    <>
      <PartRule part="Part VIII" title="Going Deeper" />

      <Section id="p8-1" title="BPE internals">
        <p>
          Part II showed why byte pair encoding beats a word vocabulary; here is
          what the learned tokenizer actually does. Training is greedy lossless
          compression: starting from characters, it repeatedly merges the most
          frequent adjacent pair, and the ordered merge list <em>is</em> the
          model. The shipped tokenizer has {BPE_VOCAB.toLocaleString()} tokens
          from {BPE_MERGES.toLocaleString()} merges on top of the base characters
          and the reserved trace tokens.
        </p>
        <p>
          Because frequent substrings collapse into single tokens, common IUPAC
          morphemes become atomic. The names of our five molecules tokenize like
          this (characters, then the old word-level split, then the learned BPE):
        </p>
        <Figure caption="BPE turns a 42-character name into 4 tokens. The compression is largest exactly where it matters, the long systematic names.">
          <CompressionFig />
        </Figure>
        <p>
          The pieces are real morphemes, not arbitrary fragments. Aspirin's name
          becomes <Pieces name="aspirin" />, and ibuprofen's becomes{" "}
          <Pieces name="ibuprofen" />. Note that <code>benzoic acid</code> and{" "}
          <code>propanoic acid</code> are each a single token: the merges have
          discovered the suffixes that recur across the corpus, which follows the
          roughly Zipfian frequency of name fragments. Coverage is total: the base
          vocabulary contains every character seen in training, so encoding falls
          back to characters in the worst case and never emits{" "}
          <code>&lt;unk&gt;</code> on a real name.
        </p>
      </Section>

      <Section id="p8-2" title="Tokenizer ablation">
        <p>
          The tokenizer choice is a three-way tradeoff between vocabulary size,
          sequence length, and coverage. It is worth seeing all three points
          rather than just the winner.
        </p>
        <div className="table-scroll">
          <table className="spec">
            <thead>
              <tr><th>tokenizer</th><th>vocab</th><th>rare-name coverage</th><th>seq length (5 names)</th><th>valid-name rate</th></tr>
            </thead>
            <tbody>
              <tr><td className="mono">word-level</td><td>large, long-tailed</td><td><span style={{ color: "var(--warn)" }}>rare fragments → &lt;unk&gt;</span></td><td className="mono">~4.8 tok</td><td className="mono">~86%</td></tr>
              <tr><td className="mono">character</td><td>~40</td><td>total (no &lt;unk&gt;)</td><td className="mono">~19.8 tok</td><td className="mono">high but slow</td></tr>
              <tr><td className="mono">BPE (shipped)</td><td className="mono">4,000</td><td>total (no &lt;unk&gt;)</td><td className="mono">~2.2 tok</td><td className="mono"><strong>97.9%</strong></td></tr>
            </tbody>
          </table>
        </div>
        <p>
          The word tokenizer is short but has an <code>&lt;unk&gt;</code> cliff:
          any fragment seen fewer than a handful of times collapses to the unknown
          token, so the model literally cannot spell a rare name, and that caps
          the valid-name rate near 86%. Character level removes the cliff but
          pays for it in sequence length, which makes long-range dependencies
          harder to learn and decoding slower (one step per character). BPE keeps
          coverage total while producing the shortest sequences of the three. The
          sequence-length column is measured on the five example names; the
          ordering holds across the corpus.
        </p>
        <Aside label="why length matters">
          Decoding cost and the difficulty of carrying information across a name
          both scale with sequence length, so the ~9x shortening from character
          to BPE is not just convenience, it changes what the model can learn with
          a 64-token decoder context.
        </Aside>
      </Section>

      <Section id="p8-3" title="Scaling the scores">
        <p>
          One detail in attention is load bearing: the scores are divided by{" "}
          <Tex tex="\sqrt{d_k}" /> before the softmax. Here{" "}
          <Tex tex="d_k = 64" /> is the per-head dimension. The reason is variance.
          A score is a dot product of a query and a key, and if their components
          are roughly independent with unit variance, the dot product has variance
          proportional to <Tex tex="d_k" />.
        </p>
        <Tex block tex="\mathrm{Var}\!\left(q \cdot k\right) = \sum_{i=1}^{d_k} \mathrm{Var}(q_i k_i) \approx d_k" />
        <p>
          Without the scaling, scores grow like <Tex tex="\sqrt{d_k}" /> in
          magnitude, the softmax saturates into a near one-hot distribution, and
          its gradient nearly vanishes, so the layer stops learning. Dividing by{" "}
          <Tex tex="\sqrt{d_k}" /> holds the score variance near one and keeps the
          softmax in its responsive regime. It is the same idea as a softmax
          temperature.
        </p>
        <Figure schematic caption="The same scores, softmaxed with and without the √dₖ scaling. Unscaled, one key dominates and the gradient through the others is tiny.">
          <ScaleFig />
        </Figure>
        <CodeBlock path="picochem/attention.py" lang="python" code={lines(RAW.attention, 19, 22)} />
      </Section>

      <Section id="p8-4" title="Attention as retrieval">
        <p>
          It helps to read attention as a differentiable dictionary lookup. The
          keys are addresses, the values are payloads, and a query retrieves a
          convex combination of the values weighted by how well it matches each
          key. A hard lookup would <Tex tex="\arg\max" /> over keys and return one
          value; the softmax makes it a soft, differentiable average so gradients
          can flow.
        </p>
        <Tex block tex="\mathrm{out} = \sum_j a_j\, v_j, \qquad a = \mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right), \qquad H(a) = -\sum_j a_j \log a_j" />
        <p>
          The attention entropy <Tex tex="H(a)" /> is a useful summary of a row:
          low entropy means a sharp, almost hard retrieval (the position is
          copying from one place), high entropy means a diffuse average over many
          positions. Tracking entropy per head is one cheap way to tell a
          content-routing head from a smoothing head, which is exactly the lens of
          the next section. In this framing the encoder is building an associative
          memory of the molecule that the decoder reads through cross-attention.
        </p>
      </Section>

      <Section id="p8-5" title="Cross-attention alignment">
        <p>
          Cross-attention induces a soft alignment between the name being written
          and the atoms of the molecule, the same object as the word alignment
          that falls out of neural machine translation. When the decoder emits a
          fragment, its cross-attention row says which SMILES tokens it is reading
          to produce it. That alignment is where the chemistry actually gets used.
        </p>
        <Figure schematic caption="Illustrative cross-attention for aspirin: the 'acetyloxy' fragment aligns to the ester atoms CC(=O)O, and 'benzoic acid' aligns to the aromatic ring and the carboxyl. A real export would replace this with measured weights.">
          <AlignmentFig />
        </Figure>
        <p>
          Unlike translation between languages, this alignment is not monotonic:
          locants and the parent name can reference atoms anywhere in the string,
          so the decoder has to attend non-locally. That is part of why
          constitutional errors (a substituent on the wrong ring carbon) are the
          dominant failure mode from Part VII: getting the name right requires
          getting the alignment to the exact atom right.
        </p>
      </Section>

      <Section id="p8-6" title="Head specialization">
        <p>
          With 8 heads per layer, different heads can take on different jobs, the
          way language models grow induction heads and syntactic heads. You can
          probe this two ways without retraining: measure each head's attention
          entropy (sharp heads route specific content, diffuse heads average), and
          correlate a head's pattern with a known structural relation, like the
          adjacency matrix that pairs each ring digit with its partner. A head
          whose weights line up with the ring-pair matrix is a ring-closure head.
        </p>
        <Figure schematic caption="A sketch of what per-head fingerprints look like: a few sharp, content-routing heads and a diffuse averaging head. The roles mirror the heads you can switch between in the attention widget (Part III).">
          <HeadFig />
        </Figure>
        <p>
          You can also ask which heads matter by ablating them one at a time
          (zeroing a head and measuring the drop in exact match), the standard
          head-importance test. The honest status: the demo's attention patterns
          are schematic, so these are the analyses the architecture makes
          possible, not measured head assignments yet.
        </p>
      </Section>
    </>
  );
}
