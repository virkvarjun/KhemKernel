import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Figure } from "../components/Figure";
import { AccuracyLadder } from "../widgets/AccuracyLadder";
import { RAW } from "../data/raw";
import { lines } from "../lib/code";

function StereoChart() {
  return (
    <svg viewBox="0 0 520 80" width="100%" role="img" aria-label="stereo is not the bottleneck">
      <text x={4} y={18} fontSize="11" fill="var(--ink-soft)">skeleton correct</text>
      <rect x={170} y={6} width={330} height={18} rx={3} fill="var(--rule)" />
      <rect x={170} y={6} width={330 * (887 / 896)} height={18} rx={3} fill="var(--accent)" />
      <text x={170 + 330 * (887 / 896) + 6} y={20} fontSize="10" fontFamily="var(--mono)" fill="var(--ink)">887 / 896 also stereo-correct</text>
      <text x={4} y={58} fontSize="11" fill="var(--ink-faint)">So pure stereo mistakes are under one point of the gap. The real gap is constitutional.</text>
    </svg>
  );
}

function BeamCeiling() {
  // endpoints are real (k=1: 79.5, k=20: 95.8); intermediate points illustrative
  const pts = [
    { k: 1, v: 79.5 },
    { k: 5, v: 90.2 },
    { k: 10, v: 93.6 },
    { k: 20, v: 95.8 },
  ];
  const W = 460, Hh = 150;
  const x = (k: number) => 40 + (Math.log2(k) / Math.log2(20)) * W;
  const y = (v: number) => Hh - 24 - ((v - 75) / 25) * (Hh - 44);
  return (
    <svg viewBox={`0 0 ${W + 60} ${Hh + 6}`} width="100%" role="img" aria-label="accuracy vs beam width">
      <line x1={40} y1={Hh - 24} x2={W + 40} y2={Hh - 24} stroke="var(--rule)" />
      <polyline points={pts.map((p) => `${x(p.k)},${y(p.v)}`).join(" ")} fill="none" stroke="var(--accent)" strokeWidth="2" />
      {pts.map((p) => (
        <g key={p.k}>
          <circle cx={x(p.k)} cy={y(p.v)} r={4} fill="var(--accent)" />
          <text x={x(p.k)} y={y(p.v) - 8} textAnchor="middle" fontSize="10" fontFamily="var(--mono)" fill="var(--accent-ink)">{p.v}%</text>
          <text x={x(p.k)} y={Hh - 8} textAnchor="middle" fontSize="10" fontFamily="var(--mono)" fill="var(--ink-faint)">k={p.k}</text>
        </g>
      ))}
      <text x={W - 40} y={y(95.8) - 22} fontSize="10" fill="var(--ink-faint)">still rising at 20</text>
    </svg>
  );
}

export function PartVII() {
  return (
    <>
      <PartRule part="Part VII" title="Results and What Is Left" />

      <Section id="p7-1" title="Evaluation">
        <p>
          The metric throughout is exact structure match. String-matching the name
          is the wrong test: one molecule has many valid name spellings and many
          equivalent SMILES. So instead, the generated name is parsed to a
          molecule, both it and the input are canonicalized by RDKit (put into one
          unique form), and they are compared as molecules. The other number
          reported is the valid name rate, how often the output even parses, which
          is 97.9%.
        </p>
        <AccuracyLadder />
        <CodeBlock path="picochem/evaluate.py · canonicalize + metrics" lang="python" code={lines(RAW.evaluate, 66, 77) + "\n\n" + lines(RAW.evaluate, 223, 228)} />
      </Section>

      <Section id="p7-2" title="Failure analysis">
        <p>
          Three small diagnostic scripts pin down the remaining errors rather than
          guessing. Stereochemistry looked like the obvious culprit, since 18% of
          targets carry stereocenters, but it is not: when the model gets the
          skeleton right it gets the stereochemistry right too, 887 times out of
          896. Pure stereo mistakes are under one point of the gap.
        </p>
        <Figure caption="Stereochemistry is not the bottleneck: get the skeleton right and the stereo almost always follows.">
          <StereoChart />
        </Figure>
        <p>
          The real gap is constitutional: right atoms, right groups, wrong
          arrangement. A substituent on the wrong ring carbon, a wrong locant, a
          wrong ring fusion. About half the remaining errors share the exact
          molecular formula of the target, the signature of positional isomers.
          And the beam ceiling shows the model often already knows the answer: the
          correct name sits in its top 20 candidates for 95.8% of molecules, and
          the curve is still climbing at 20. That is why the verifier rerank works
          so well.
        </p>
        <Figure schematic caption="Exact match if we accept any of the top-k candidates. Endpoints (k=1 and k=20) are measured; the curve is still rising at 20, so the model knows more than a greedy pass reveals.">
          <BeamCeiling />
        </Figure>
      </Section>

      <Section id="p7-3" title="What is next">
        <p>
          The deployed accuracy is already at the beam ceiling, so the project
          stops here on quality. The clear next lever for speed is the embedding
          host bottleneck: moving the embedding gather and scatter onto the GPU
          would lift utilization off the 35 to 40% floor. Beyond that, Tensor
          Cores and a from-scratch FlashAttention are the obvious directions, but
          both are future work, not part of the shipped path.
        </p>
        <Aside label="where this sits">
          ChemKernel is not state of the art on the chemistry, and it is not meant
          to be. STOUT is established prior art and is larger. The point of the
          project is everything underneath: a transformer, its hand-derived
          gradients, the optimizer, and the GPU kernels, all written by hand and
          checked against each other, with a free inference-time verifier on top.
        </Aside>
      </Section>
    </>
  );
}
