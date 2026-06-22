import { PartRule, Section } from "../components/ui";
import { SmilesTokens } from "../components/SmilesTokens";
import { Figure } from "../components/Figure";
import { AccuracyLadder } from "../widgets/AccuracyLadder";
import { MOL_BY_KEY } from "../data/molecules";

export function PartI() {
  const aspirin = MOL_BY_KEY.aspirin;
  const ibuprofen = MOL_BY_KEY.ibuprofen;
  return (
    <>
      <PartRule part="Part I" title="The Problem" />

      <Section id="p1-1" title="What the model does">
        <p>
          You give the model a molecule written as a SMILES string and it writes
          back the molecule's systematic IUPAC name. SMILES is the line notation
          chemists type to store a structure (<span className="tok tok-atom">C</span>,{" "}
          <span className="tok tok-aromatic">c</span>, parentheses for branches,
          digits for rings); the IUPAC name is the formal, rule-based name you
          would find in a paper or on a reagent bottle. So the task is naming
          molecules automatically.
        </p>
        <p>
          Before it commits to a name, the model writes a short reasoning trace:
          the parent scaffold, the functional groups, the heavy-atom count, and
          the ring count. The <code>&lt;name&gt;</code> field is the answer; the
          rest is intermediate reasoning the model is trained to lay out first.
          Here are two worked examples (the hero above lets you run all five).
        </p>

        {[aspirin, ibuprofen].map((m) => (
          <div key={m.key} className="card" style={{ margin: "0.8rem 0" }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: "0.74rem", textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--ink-faint)" }}>
              {m.label}
            </div>
            <div style={{ margin: "0.4rem 0" }}>
              <SmilesTokens smiles={m.smiles} />
            </div>
            <div style={{ fontFamily: "var(--mono)", fontSize: "0.8rem", color: "var(--ink-soft)", wordBreak: "break-word" }}>
              {m.trace}
            </div>
            <div style={{ marginTop: "0.4rem" }}>
              &rarr;{" "}
              <strong style={{ color: "var(--accent-ink)" }}>{m.iupac}</strong>
            </div>
          </div>
        ))}
      </Section>

      <Section id="p1-2" title="A translation problem">
        <p>
          A SMILES string and an IUPAC name are two different surface forms of
          the same underlying object: the molecular graph. Neither is the
          molecule; both are encodings of it. Going from one encoding to the
          other is exactly machine translation, the same shape of problem as
          translating a sentence between two languages. That is what motivates an
          encoder-decoder transformer: an encoder reads the source notation, a
          decoder writes the target notation while looking back at what the
          encoder read.
        </p>
        <Figure
          caption="Phenol, written two ways. Both the SMILES string and the IUPAC name are surface forms of one molecular graph; the model translates between them."
        >
          <svg viewBox="0 0 640 170" width="100%" role="img" aria-label="phenol as two surface forms of one graph">
            <text x="120" y="28" textAnchor="middle" fill="var(--ink-faint)" fontFamily="var(--mono)" fontSize="12">SMILES</text>
            <text x="120" y="58" textAnchor="middle" fill="var(--tok-aromatic)" fontFamily="var(--mono)" fontSize="20">c1ccc(O)cc1</text>
            <text x="520" y="28" textAnchor="middle" fill="var(--ink-faint)" fontFamily="var(--mono)" fontSize="12">IUPAC name</text>
            <text x="520" y="58" textAnchor="middle" fill="var(--accent-ink)" fontSize="22">phenol</text>
            {/* central graph: benzene hexagon + OH */}
            <g transform="translate(320,110)" stroke="var(--chem)" strokeWidth="2" fill="none">
              <polygon points="0,-34 29,-17 29,17 0,34 -29,17 -29,-17" />
              <line x1="29" y1="-17" x2="52" y2="-30" />
            </g>
            <text x="320" y="158" textAnchor="middle" fill="var(--ink-faint)" fontFamily="var(--mono)" fontSize="12">molecular graph</text>
            <text x="378" y="80" fill="var(--tok-atom)" fontFamily="var(--mono)" fontSize="13">OH</text>
            {/* arrows */}
            <g stroke="var(--ink-faint)" strokeWidth="1.5" markerEnd="url(#ar)">
              <line x1="185" y1="95" x2="270" y2="95" />
              <line x1="455" y1="95" x2="370" y2="95" />
            </g>
            <defs>
              <marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--ink-faint)" />
              </marker>
            </defs>
          </svg>
        </Figure>
      </Section>

      <Section id="p1-3" title="The headline result">
        <p>
          On 2,000 held-out molecules, a single greedy pass gets the exact
          structure right 79.5% of the time. If we let the model propose a beam
          of candidates and keep the one that round-trips back to the input,
          accuracy climbs to 89.6% at beam 5 and 95.8% at beam 20. The valid
          IUPAC name rate is 97.9%. Every number is exact structure match: the
          generated name is parsed back to a molecule and compared to the input
          (Part VII explains the metric).
        </p>
        <AccuracyLadder />
        <p style={{ marginTop: "1rem", color: "var(--ink-faint)", fontSize: "0.95rem" }}>
          The whole thing trains on one GPU in about half a day and runs
          inference on a laptop CPU. The rest of this guide is how it works, from
          the tokenizers up through the architecture, training, the CUDA kernels,
          and the verifier.
        </p>
      </Section>
    </>
  );
}
