import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Figure } from "../components/Figure";
import { Math as Tex } from "../components/Math";
import { RAW } from "../data/raw";
import { pyDef } from "../lib/code";
import { GRAD_CHECKS } from "../data/measured";

/* ---- p9-1 trace ablation (illustrative) ---- */
function AblationFig() {
  const bars = [
    { label: "bare name target", v: 72, hyp: true },
    { label: "trace target (shipped)", v: 79.5, hyp: false },
  ];
  return (
    <div style={{ display: "grid", gap: "0.5rem" }}>
      {bars.map((b) => (
        <div key={b.label} style={{ display: "grid", gridTemplateColumns: "11rem 1fr", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: "0.85rem", color: "var(--ink-soft)" }}>{b.label}</span>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ height: 20, width: `${b.v}%`, background: b.hyp ? "var(--rule)" : "var(--accent)", borderRadius: 4, border: b.hyp ? "1px dashed var(--ink-faint)" : "none" }} />
            <span style={{ fontFamily: "var(--mono)", color: "var(--ink-soft)", fontSize: "0.82rem" }}>{b.hyp ? "?" : b.v + "%"}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ---- p9-2 rank-of-truth (masses derived from the real beam-ceiling numbers) ---- */
function RankFig() {
  const buckets = [
    { label: "rank 1 (greedy)", v: 79.5 },
    { label: "ranks 2–5", v: 10.1 },
    { label: "ranks 6–20", v: 6.2 },
    { label: "not in top 20", v: 4.2, miss: true },
  ];
  return (
    <div style={{ display: "grid", gap: "0.45rem" }}>
      {buckets.map((b) => (
        <div key={b.label} style={{ display: "grid", gridTemplateColumns: "9rem 1fr 3rem", gap: 8, alignItems: "center", fontSize: "0.82rem" }}>
          <span style={{ color: "var(--ink-soft)" }}>{b.label}</span>
          <div style={{ height: 16, width: `${b.v}%`, minWidth: 4, background: b.miss ? "var(--warn)" : "var(--accent)", borderRadius: 3 }} />
          <span style={{ fontFamily: "var(--mono)", color: "var(--ink-faint)" }}>{b.v}%</span>
        </div>
      ))}
    </div>
  );
}

/* ---- p9-3 probe by layer (schematic) ---- */
function ProbeFig() {
  const series = [
    { name: "ring count", c: "var(--accent)", v: [0.55, 0.86, 0.94] },
    { name: "has carboxylic acid", c: "var(--chem)", v: [0.62, 0.8, 0.97] },
  ];
  const W = 360, H = 130, pad = 28;
  const x = (i: number) => pad + (i / 2) * (W - 2 * pad);
  const y = (v: number) => H - pad - v * (H - 2 * pad);
  return (
    <svg viewBox={`0 0 ${W} ${H + 18}`} width="100%" role="img" aria-label="probe accuracy by layer">
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--rule)" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--rule)" />
      {series.map((s) => (
        <g key={s.name}>
          <polyline points={s.v.map((v, i) => `${x(i)},${y(v)}`).join(" ")} fill="none" stroke={s.c} strokeWidth="2" />
          {s.v.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r="3" fill={s.c} />)}
          <text x={W - pad} y={y(s.v[2]) - 6} textAnchor="end" fontSize="9" fill={s.c}>{s.name}</text>
        </g>
      ))}
      {[0, 1, 2].map((i) => <text key={i} x={x(i)} y={H - pad + 13} textAnchor="middle" fontSize="9" fontFamily="var(--mono)" fill="var(--ink-faint)">enc {i + 1}</text>)}
      <text x={6} y={pad} fontSize="9" fontFamily="var(--mono)" fill="var(--ink-faint)">acc</text>
    </svg>
  );
}

/* ---- p9-5 gpu time split (util is measured, the split is illustrative) ---- */
function GpuTimeFig() {
  const segs = [
    { label: "device kernels (matmul, attn, norm)", ms: 16, c: "var(--accent)" },
    { label: "host embedding gather / scatter", ms: 16, c: "var(--chem)" },
    { label: "host ↔ device transfer", ms: 12, c: "var(--warn)" },
  ];
  const total = 44;
  return (
    <div>
      <div style={{ display: "flex", height: 30, borderRadius: 6, overflow: "hidden", border: "1px solid var(--panel-line)" }}>
        {segs.map((s) => (
          <div key={s.label} style={{ width: `${(s.ms / total) * 100}%`, background: s.c }} title={`${s.label}: ${s.ms} ms`} />
        ))}
      </div>
      <div style={{ display: "grid", gap: 2, marginTop: 8 }}>
        {segs.map((s) => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem" }}>
            <span style={{ width: 11, height: 11, background: s.c, borderRadius: 3, display: "inline-block" }} />
            <span style={{ color: "var(--ink-soft)" }}>{s.label}</span>
            <span style={{ fontFamily: "var(--mono)", color: "var(--ink-faint)" }}>{s.ms} ms</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function PartIX() {
  return (
    <>
      <PartRule part="Part IX" title="Analysis" />

      <Section id="p9-1" title="Does the trace help?">
        <p>
          The model writes a reasoning trace (parent, groups, atom count, ring
          count) before it commits to a name. That is a chain-of-thought
          scaffold, and it should earn its keep: predicting a handful of
          intermediate, checkable facts first constrains the final name and gives
          the network auxiliary supervision, the same reason a scratchpad helps a
          language model. The clean test is an ablation: train one model on the
          bare name and one on the full trace, hold the architecture, data, and
          schedule fixed, and compare exact match.
        </p>
        <Figure schematic caption="The hypothesis, not a run: the trace scaffold should lift exact match over a bare-name target. The bare-name bar is illustrative; this ablation has not been run end to end yet.">
          <AblationFig />
        </Figure>
        <p>
          There is a second, free benefit either way. The trace fields are
          ground-truth labels (RDKit computes them), so training on them is
          multitask learning, and at inference the fields are an inspection point:
          you can read what the model believes the parent and groups are before it
          names anything, which is what makes the probing and failure analysis
          possible.
        </p>
      </Section>

      <Section id="p9-2" title="The verifier as a reward">
        <p>
          The OPSIN round-trip is a label-free, programmatic verifier: it turns a
          generated name back into a molecule and checks structural identity. That
          is the same shape as the verifiable rewards behind a lot of recent
          work, a compiler or unit test for code, a proof checker for math. We use
          it at inference as a reranker, but it is also a reward signal you could
          train against.
        </p>
        <p>
          Start by asking where the correct answer sits in the beam. The
          beam-ceiling numbers from Part VII pin the masses: the model ranks the
          right name first 79.5% of the time, finds it in the top 5 for another 10
          points, and in the top 20 for 6 more. So for most of the gap the model{" "}
          <em>has</em> the answer and simply mis-ranks it.
        </p>
        <Figure caption="Where the correct name lands in the beam. Bucket masses are derived from the measured greedy, beam-5, and beam-20 numbers (79.5 / 89.6 / 95.8).">
          <RankFig />
        </Figure>
        <p>
          Two things follow. First, selective prediction: because we can check
          answers, we can abstain when nothing verifies, trading coverage for
          precision, which is the right move when a wrong systematic name is worse
          than no name. Second, and more interesting, the verifier is a reward with
          no labels, so you could fine-tune the model to rank the correct answer
          first and close the greedy-to-reranked gap directly. That is the natural
          next step, and it is squarely the verifiable-reward idea applied to
          chemistry.
        </p>
        <Aside label="honest framing" warn>
          Round-tripping names through OPSIN is not new (STOUT V2 uses it too), and
          the reward fine-tuning above is future work, not a result. The shipped
          contribution is narrow: using the round-trip as a beam reranker, which is
          what turns 79.5% into 95.8% for free.
        </Aside>
      </Section>

      <Section id="p9-3" title="Linear probes">
        <p>
          To ask what the encoder represents, fit a linear probe from frozen
          encoder activations to a known property, then see how accurately a plain
          linear readout recovers it. High linear decodability means the property
          is represented in a roughly linear, accessible way. The trace fields
          give the labels for free (ring count, heavy-atom count, which functional
          groups are present), so no extra annotation is needed.
        </p>
        <Figure schematic caption="The intended readout: probe accuracy by encoder layer for two properties, showing where a feature becomes linearly decodable. Curves are illustrative; the probe has not been fit yet.">
          <ProbeFig />
        </Figure>
        <p>
          The standard caveats apply. A probe measures decodability, not whether
          the model uses the feature, and a sufficiently expressive probe can find
          structure that is not actually used downstream, which is why the probe is
          kept linear. Tracking accuracy across the three encoder layers is the
          interesting part: it shows where coarse features (does it have a ring)
          emerge before finer ones (which functional group).
        </p>
      </Section>

      <Section id="p9-4" title="Gradient checking">
        <p>
          The whole from-scratch claim rests on the hand-derived backward passes
          being correct, so they are checked numerically. For each op, nudge an
          input by a small <Tex tex="\epsilon" />, measure how a scalar loss
          changes, and compare that central finite difference to the analytic
          gradient.
        </p>
        <Tex block tex="\frac{\partial L}{\partial x_i} \;\approx\; \frac{L(x_i + \epsilon) - L(x_i - \epsilon)}{2\epsilon}" />
        <p>
          Run on the float64 reference with <Tex tex="\epsilon = 10^{-6}" />, every
          op agrees with the numerical estimate to better than two parts in a
          billion. These are real numbers from the reference implementation:
        </p>
        <Figure caption="Max relative error between the hand-derived gradient and a central finite difference, per op. Measured on the float64 NumPy reference.">
          <div className="table-scroll">
            <table className="spec">
              <thead><tr><th>operation</th><th>max relative error</th></tr></thead>
              <tbody>
                {GRAD_CHECKS.map((g) => (
                  <tr key={g.op}><td>{g.op}</td><td className="mono">{g.err}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Figure>
        <p>
          The CUDA path is then validated a second way: every kernel is compared
          against this NumPy reference, and the two agree to about{" "}
          <Tex tex="10^{-3}" />, which is the rounding you expect from doing the
          same arithmetic in float32 instead of float64. Two independent
          implementations landing on the same answer is the real test that the
          math is right.
        </p>
        <CodeBlock path="picochem/ops.py · linear_backward" lang="python" code={pyDef(RAW.ops, "linear_backward")} />
      </Section>

      <Section id="p9-5" title="Where the GPU time goes">
        <p>
          The device-resident step is about 60x faster than NumPy (44 ms versus
          2,640 ms), but the GPU still sits at only 35 to 40% utilization. That
          number is the interesting one, and it has a specific cause. The
          transformer matmuls are compute-bound and finish quickly on the device,
          but the embedding tables stay on the host: every step gathers rows on the
          CPU, scatters gradients back, and moves the activations across the bus.
          That host work and transfer is memory and bandwidth bound, and it
          cannot overlap away, so it sets the pace.
        </p>
        <Figure schematic caption="Roughly where one 44 ms step goes. The 35–40% GPU utilization is measured; the split between device compute, host embedding work, and transfer is illustrative.">
          <GpuTimeFig />
        </Figure>
        <p>
          In roofline terms the matmuls are well under the compute roof while the
          embedding path is pinned against the bandwidth roof, so the step is
          bound by the slow part. This also explains a non-result: faster matmul
          kernels (Tensor Cores, cuBLAS) barely move the wall clock, because by
          Amdahl's law you only speed up the part that was already fast. The real
          lever is moving the embedding gather and scatter onto the device, which
          is the main thing left on the table.
        </p>
      </Section>
    </>
  );
}
