import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Figure } from "../components/Figure";
import { Math as Tex } from "../components/Math";
import { ForwardFlow } from "../widgets/ForwardFlow";
import { RAW } from "../data/raw";
import { lines, pyDef } from "../lib/code";

const TGT = ["<start>", "<parent>", "benzene", "</parent>", "<name>", "phenol", "</name>", "<end>"];

function TeacherForcing() {
  const tin = TGT.slice(0, -1);
  const tout = TGT.slice(1);
  const cell = 92;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg viewBox={`0 0 ${tin.length * cell + 80} 150`} width="100%" role="img" aria-label="teacher forcing shift">
        <text x={8} y={44} fontSize="11" fontFamily="var(--mono)" fill="var(--ink-faint)">tgt_in</text>
        <text x={8} y={114} fontSize="11" fontFamily="var(--mono)" fill="var(--ink-faint)">tgt_out</text>
        {tin.map((t, i) => (
          <g key={"in" + i}>
            <rect x={70 + i * cell} y={26} width={cell - 8} height={30} rx={5} fill="var(--panel)" stroke="var(--panel-line)" />
            <text x={70 + (cell - 8) / 2 + i * cell} y={46} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--ink)">{t.replace(/[<>]/g, "")}</text>
          </g>
        ))}
        {tout.map((t, i) => (
          <g key={"out" + i}>
            <rect x={70 + i * cell} y={96} width={cell - 8} height={30} rx={5} fill="var(--accent-soft)" stroke="var(--accent)" />
            <text x={70 + (cell - 8) / 2 + i * cell} y={116} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--accent-ink)">{t.replace(/[<>]/g, "")}</text>
            <line x1={70 + (cell - 8) / 2 + i * cell} y1={56} x2={70 + (cell - 8) / 2 + i * cell} y2={96} stroke="var(--ink-faint)" strokeDasharray="3 2" markerEnd="url(#tf)" />
          </g>
        ))}
        <defs><marker id="tf" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--ink-faint)" /></marker></defs>
      </svg>
    </div>
  );
}

function LrSchedule() {
  const warmup = 1500, total = 120000, peak = 3e-4;
  const pts: string[] = [];
  const W = 480, Hh = 120;
  for (let i = 0; i <= 200; i++) {
    const step = (total * i) / 200;
    let lr: number;
    if (step < warmup) lr = (peak * step) / warmup;
    else {
      const p = (step - warmup) / (total - warmup);
      lr = 0.5 * (1 + Math.cos(Math.PI * p)) * peak;
    }
    pts.push(`${(20 + (step / total) * W).toFixed(1)},${(Hh - 10 - (lr / peak) * (Hh - 30)).toFixed(1)}`);
  }
  const wx = 20 + (warmup / total) * W;
  return (
    <svg viewBox={`0 0 ${W + 40} ${Hh + 10}`} width="100%" role="img" aria-label="learning rate schedule">
      <line x1={20} y1={Hh - 10} x2={W + 20} y2={Hh - 10} stroke="var(--rule)" />
      <line x1={wx} y1={10} x2={wx} y2={Hh - 10} stroke="var(--chem)" strokeDasharray="3 3" />
      <text x={wx + 4} y={22} fontSize="10" fill="var(--chem)" fontFamily="var(--mono)">warmup ends (1.5k)</text>
      <polyline points={pts.join(" ")} fill="none" stroke="var(--accent)" strokeWidth="2" />
      <text x={20} y={Hh + 6} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">0</text>
      <text x={W + 8} y={Hh + 6} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">120k steps</text>
      <text x={24} y={20} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">peak lr 3e-4</text>
    </svg>
  );
}

function LossSpike() {
  const W = 480, Hh = 130;
  const run2: string[] = [];
  for (let i = 0; i <= 200; i++) {
    const step = (120000 * i) / 200;
    const loss = 0.3 + 3.2 * Math.exp(-step / 12000);
    run2.push(`${(20 + (step / 120000) * W).toFixed(1)},${(Hh - 12 - (1 - loss / 3.6) * (Hh - 30)).toFixed(1)}`);
  }
  // run1: descends then spikes to NaN near 19k
  const run1: string[] = [];
  for (let i = 0; i <= 33; i++) {
    const step = (19000 * i) / 33;
    const loss = 0.37 + 3.2 * Math.exp(-step / 9000);
    run1.push(`${(20 + (step / 120000) * W).toFixed(1)},${(Hh - 12 - (1 - loss / 3.6) * (Hh - 30)).toFixed(1)}`);
  }
  const sx = 20 + (19000 / 120000) * W;
  return (
    <svg viewBox={`0 0 ${W + 40} ${Hh + 10}`} width="100%" role="img" aria-label="loss curve with NaN spike">
      <line x1={20} y1={Hh - 12} x2={W + 20} y2={Hh - 12} stroke="var(--rule)" />
      <polyline points={run2.join(" ")} fill="none" stroke="var(--accent)" strokeWidth="2" />
      <polyline points={run1.join(" ")} fill="none" stroke="var(--warn)" strokeWidth="1.8" strokeDasharray="5 3" />
      <line x1={sx} y1={14} x2={sx} y2={Hh - 12} stroke="var(--warn)" strokeDasharray="2 2" />
      <circle cx={sx} cy={20} r={4} fill="var(--warn)" />
      <text x={sx - 4} y={34} textAnchor="end" fontSize="10" fill="var(--warn)" fontFamily="var(--mono)">NaN at ~19k (first run)</text>
      <text x={W - 80} y={Hh - 20} fontSize="10" fill="var(--accent-ink)" fontFamily="var(--mono)">second run: full 120k</text>
    </svg>
  );
}

export function PartIV() {
  return (
    <>
      <PartRule part="Part IV" title="Training the Model" />

      <Section id="p4-1" title="The target and teacher forcing">
        <p>
          The model is trained to predict the next token of the trace. During
          training we already know the whole correct trace, so we feed it the true
          previous tokens rather than its own guesses (teacher forcing). The
          target sequence is shifted by one: the decoder input is everything
          except the last token, and what it must predict is everything except the
          first. So at each slot the model sees the tokens up to here and predicts
          the next one.
        </p>
        <Figure caption="The trace is shifted by one. tgt_out is tgt_in moved left, so predicting position i means predicting the token that follows the input at i.">
          <TeacherForcing />
        </Figure>
        <CodeBlock path="picochem/data_loader.py · make_batch" lang="python" code={lines(RAW.dataLoader, 119, 128)} />
      </Section>

      <Section id="p4-2" title="The loss">
        <p>
          The loss is softmax cross entropy over the vocabulary at every predicted
          position: turn the logits into probabilities, then penalize the negative
          log probability of the correct token. Padding positions are marked with
          an ignore index and dropped from the average. The forward uses the
          log-sum-exp trick so exponentials never overflow.
        </p>
        <Tex block tex="L = -\frac{1}{N}\sum_{i} \log \mathrm{softmax}(z_i)_{t_i}, \qquad \frac{\partial L}{\partial z_i} = \frac{\mathrm{softmax}(z_i) - \mathrm{onehot}(t_i)}{N}" />
        <p>
          The gradient is unusually clean: softmax minus the one-hot target,
          divided by the number of valid tokens. That falls straight out of the
          math and is exactly what the code computes.
        </p>
        <CodeBlock path="picochem/ops.py · softmax_cross_entropy_forward" lang="python" code={pyDef(RAW.ops, "softmax_cross_entropy_forward")} />
      </Section>

      <Section id="p4-3" title="Backpropagation by hand">
        <p>
          There is no autograd here. Reverse-mode automatic differentiation just
          means applying the chain rule backward through the layers: start from
          the loss gradient, and each layer turns the gradient of its output into
          the gradient of its input and its parameters. Every one of these
          backward functions was derived by hand and checked against finite
          differences (nudge an input, measure the change in output) to about
          1e-7. The linear and softmax-cross-entropy gradients are the cleanest.
        </p>
        <CodeBlock path="picochem/ops.py · linear_backward + softmax_cross_entropy_backward" lang="python" code={pyDef(RAW.ops, "linear_backward") + "\n\n" + pyDef(RAW.ops, "softmax_cross_entropy_backward")} />
        <p>
          The backward pass is the forward diagram run in reverse: the gradient
          enters at the output head and flows back through the decoder, into the
          encoder memory, and out to the embeddings. Step it backward:
        </p>
        <ForwardFlow reverse />
      </Section>

      <Section id="p4-4" title="The optimizer and schedule">
        <p>
          Parameters are updated with Adam, which keeps a running mean and
          variance of each parameter's gradient and bias-corrects them, so every
          parameter gets its own effective step size. The learning rate is not
          constant: it ramps up linearly for the first 1,500 steps (warmup), then
          decays along a cosine to near zero over the full 120,000 steps.
        </p>
        <Figure caption="Linear warmup to a peak of 3e-4, then cosine decay over 120k steps.">
          <LrSchedule />
        </Figure>
        <CodeBlock path="picochem/optimizer.py · adam_step" lang="python" code={pyDef(RAW.optimizer, "adam_step")} />
        <CodeBlock path="picochem/scheduler.py · linear_warmup_cosine_decay" lang="python" code={pyDef(RAW.scheduler, "linear_warmup_cosine_decay")} />
        <CodeBlock path="picochem/kernels/cuda/adam.cu" lang="cuda" code={lines(RAW.adam, 8, 21)} />
      </Section>

      <Section id="p4-5" title="The training loop and stability">
        <p>
          The first scaled run trained beautifully down to a loss of 0.37, then a
          gradient spike near peak learning rate drove the loss to NaN around step
          19,000. Worse, the trainer was overwriting a single checkpoint file, so
          the NaN weights erased the good ones. The device trainer has no gradient
          clipping (a clip needs a reduction kernel that does not exist yet).
        </p>
        <Figure schematic caption="The first run (dashed) spiked to NaN near step 19k; the fixed run (solid) went the full 120k with zero NaN events.">
          <LossSpike />
        </Figure>
        <p>
          The cheaper fix that shipped: skip the optimizer step whenever the loss
          is not finite, only checkpoint finite states, keep numbered snapshots,
          and drop the peak learning rate a little. The next run went the full
          120,000 steps with zero NaN events and zero skipped steps.
        </p>
        <CodeBlock path="scripts/train_device.py · the stability guard" lang="python" code={lines(RAW.trainDevice, 195, 226)} />
        <Aside label="why this is enough">
          A single bad batch can no longer poison the weights, because its update
          is skipped and the next batch starts from the still-good weights. And a
          NaN can never reach disk, because only finite states are checkpointed.
        </Aside>
      </Section>
    </>
  );
}
