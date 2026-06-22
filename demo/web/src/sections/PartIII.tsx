import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Figure } from "../components/Figure";
import { Math as Tex } from "../components/Math";
import { MoleculeExplorer } from "../widgets/MoleculeExplorer";
import { AttentionHeatmap } from "../widgets/AttentionHeatmap";
import { MaskingViz } from "../widgets/MaskingViz";
import { LayerNormViz } from "../widgets/LayerNormViz";
import { ForwardFlow } from "../widgets/ForwardFlow";
import { RAW } from "../data/raw";
import { lines, pyDef } from "../lib/code";

function GeluCurve() {
  const pts: string[] = [];
  for (let i = 0; i <= 48; i++) {
    const x = -4 + (8 * i) / 48;
    const y = 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3)));
    const px = 10 + ((x + 4) / 8) * 180;
    const py = 70 - y * 18;
    pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  return (
    <svg viewBox="0 0 200 100" width="200" height="100" role="img" aria-label="GeLU curve">
      <line x1="10" y1="70" x2="190" y2="70" stroke="var(--rule)" />
      <line x1="100" y1="6" x2="100" y2="94" stroke="var(--rule)" />
      <polyline points={pts.join(" ")} fill="none" stroke="var(--accent)" strokeWidth="2" />
      <text x="186" y="84" fontSize="9" fill="var(--ink-faint)" fontFamily="var(--mono)">x</text>
    </svg>
  );
}

export function PartIII() {
  return (
    <>
      <PartRule part="Part III" title="The Transformer Architecture" />

      <Section id="p3-1" title="Embeddings">
        <p>
          The model cannot do arithmetic on token ids directly, so each id is
          turned into a vector. An embedding table is a big matrix with one
          learned row per vocabulary token; looking up a token is just indexing
          its row. Because a plain lookup throws away word order, a second table
          of learned positional vectors is added on top, one per slot in the
          sequence, so the model knows which token came first. The target table
          is reused at the very end as the output projection (more on that in 13).
        </p>
        <CodeBlock path="picochem/embeddings.py" lang="python" code={pyDef(RAW.embeddings, "token_embedding_forward") + "\n\n" + pyDef(RAW.embeddings, "positional_embedding_forward")} />
        <p>
          The forward lookup and the backward scatter both run on the host (CPU).
          The scatter has to use <code>atomicAdd</code> in CUDA because several
          positions in a batch can point at the same table row, and their
          gradients must accumulate rather than overwrite.
        </p>
        <CodeBlock path="picochem/kernels/cuda/embedding.cu" lang="cuda" code={lines(RAW.embeddingCu, 8, 17)} />
      </Section>

      <Section id="p3-2" title="Linear layer">
        <p>
          Almost everything else is built from one operation: a linear layer,{" "}
          <Tex tex="y = xW + b" />. Multiply the input by a weight matrix, add a
          bias. Its three gradients (with respect to the input, the weights, and
          the bias) are short to write and are the backbone of the whole backward
          pass.
        </p>
        <Tex block tex="\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\,W^\top, \quad \frac{\partial L}{\partial W} = x^\top \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial b} = \sum_{\text{rows}} \frac{\partial L}{\partial y}" />
        <Figure caption="Shapes of a linear layer. The forward is one matmul plus a bias; the backward is two more matmuls (one per operand) and a column sum.">
          <svg viewBox="0 0 560 120" width="100%" role="img" aria-label="linear layer shapes">
            {[
              { x: 20, label: "x", sub: "rows × in", c: "var(--ink-faint)" },
              { x: 170, label: "W", sub: "in × out", c: "var(--chem)" },
              { x: 330, label: "y", sub: "rows × out", c: "var(--accent)" },
            ].map((b) => (
              <g key={b.label}>
                <rect x={b.x} y={30} width={110} height={56} rx={8} fill="var(--panel-2)" stroke="var(--panel-line)" />
                <text x={b.x + 55} y={56} textAnchor="middle" fontFamily="var(--mono)" fontSize="18" fill={b.c}>{b.label}</text>
                <text x={b.x + 55} y={74} textAnchor="middle" fontFamily="var(--mono)" fontSize="10" fill="var(--ink-faint)">{b.sub}</text>
              </g>
            ))}
            <text x={148} y={62} fontSize="18" fill="var(--ink-soft)">×</text>
            <text x={305} y={62} fontSize="18" fill="var(--ink-soft)">=</text>
            <text x={455} y={50} fontSize="13" fill="var(--ink-soft)">+ bias b</text>
            <text x={455} y={70} fontSize="11" fill="var(--ink-faint)" fontFamily="var(--mono)">(out,)</text>
          </svg>
        </Figure>
        <CodeBlock path="picochem/ops.py" lang="python" code={pyDef(RAW.ops, "linear_forward") + "\n\n" + pyDef(RAW.ops, "linear_backward")} />
        <p>
          On the GPU the matmul is the tiled kernel below: each 16 by 16 output
          tile is computed by a block that streams matching tiles of the inputs
          into fast shared memory and accumulates. Part V animates this.
        </p>
        <CodeBlock path="picochem/kernels/cuda/matmul_tiled.cu" lang="cuda" code={lines(RAW.matmulTiled, 9, 35)} />
      </Section>

      <Section id="p3-3" title="Scaled dot-product attention">
        <p>
          Attention is how a position pulls in information from other positions.
          Every position emits a query, and every position offers a key and a
          value. A query is compared against all keys by dot product to get a
          score, the scores are softmaxed into weights that sum to one, and the
          output is the weighted average of the values. The scores are divided by{" "}
          <Tex tex="\sqrt{d_k}" /> so they do not blow up as the head dimension
          grows.
        </p>
        <Tex block tex="\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V" />
        <CodeBlock path="picochem/attention.py" lang="python" code={pyDef(RAW.attention, "scaled_dot_product_attention_forward")} />
        <p>
          Below is attention over a real SMILES string. Pick a head and hover a
          query row to see what it attends to. The patterns are illustrative of
          what a head can learn (ring-closure pairing, locality, branch matching),
          not measured weights.
        </p>
        <AttentionHeatmap />
      </Section>

      <Section id="p3-4" title="Multi-head attention">
        <p>
          One attention is a bottleneck: a position can only mix information one
          way. Multi-head attention runs several attentions in parallel, each on
          its own slice of the 512-dimensional vector (here 8 heads of 64 dims).
          Different heads can specialize: one might follow ring closures, another
          might track which functional group a carbon belongs to. The heads are
          computed by reshaping the projected Q, K, V into a head axis, running
          attention per head, then merging back and applying an output
          projection.
        </p>
        <CodeBlock path="picochem/attention.py · multihead_self_attention_forward" lang="python" code={lines(RAW.attention, 61, 86)} />
        <p>
          The reshape that creates and removes the head axis is the same
          transpose the CUDA path implements as split-heads and merge-heads
          kernels. Switch heads in the widget above to see different structure.
        </p>
      </Section>

      <Section id="p3-5" title="Masking">
        <p>
          Two things must be hidden from attention. The decoder must not look at
          future tokens (it would be cheating during training), and no position
          should attend to padding. Both are done the same way: build an additive
          mask that is 0 where attention is allowed and a large negative number
          where it is not, and add it to the scores before the softmax. The
          forbidden entries get pushed to negative infinity, so after softmax
          their weight is exactly zero.
        </p>
        <CodeBlock path="picochem/model.py" lang="python" code={pyDef(RAW.model, "make_padding_mask") + "\n\n" + pyDef(RAW.model, "make_causal_mask")} />
        <p>
          The causal mask is upper-triangular; the padding mask blanks out pad
          columns; the decoder uses their sum. Toggle the modes and hover a row.
        </p>
        <MaskingViz />
      </Section>

      <Section id="p3-6" title="Self vs cross attention">
        <p>
          Self-attention and cross-attention are the same machinery wired two
          ways. In self-attention the queries, keys, and values all come from one
          sequence: it lets a sequence mix information within itself. In
          cross-attention the queries come from the decoder while the keys and
          values come from the encoder output. That is the bridge: it is how the
          decoder, while writing the name, reads the encoded molecule.
        </p>
        <Figure caption="Self-attention draws Q, K, V from one sequence. Cross-attention takes Q from the decoder and K, V from the encoder memory.">
          <svg viewBox="0 0 560 170" width="100%" role="img" aria-label="self vs cross attention">
            {/* self */}
            <text x="140" y="20" textAnchor="middle" fontSize="13" fill="var(--ink-soft)">self-attention</text>
            <rect x="60" y="40" width="160" height="34" rx="7" fill="var(--panel-2)" stroke="var(--panel-line)" />
            <text x="140" y="62" textAnchor="middle" fontSize="12" fill="var(--ink)">one sequence</text>
            {["Q", "K", "V"].map((t, i) => (
              <g key={t}>
                <line x1={90 + i * 50} y1={74} x2={90 + i * 50} y2={120} stroke="var(--accent)" strokeWidth="1.5" markerEnd="url(#sa)" />
                <text x={90 + i * 50} y={138} textAnchor="middle" fontFamily="var(--mono)" fontSize="12" fill="var(--accent-ink)">{t}</text>
              </g>
            ))}
            {/* cross */}
            <text x="420" y="20" textAnchor="middle" fontSize="13" fill="var(--ink-soft)">cross-attention</text>
            <rect x="320" y="40" width="90" height="34" rx="7" fill="var(--panel-2)" stroke="var(--panel-line)" />
            <text x="365" y="62" textAnchor="middle" fontSize="11" fill="var(--ink)">decoder</text>
            <rect x="430" y="40" width="100" height="34" rx="7" fill="var(--chem-soft)" stroke="var(--chem)" />
            <text x="480" y="62" textAnchor="middle" fontSize="11" fill="var(--chem)">enc memory</text>
            <line x1="365" y1="74" x2="365" y2="120" stroke="var(--accent)" strokeWidth="1.5" markerEnd="url(#sa)" />
            <text x="365" y="138" textAnchor="middle" fontFamily="var(--mono)" fontSize="12" fill="var(--accent-ink)">Q</text>
            <line x1="460" y1="74" x2="460" y2="120" stroke="var(--chem)" strokeWidth="1.5" markerEnd="url(#sac)" />
            <line x1="500" y1="74" x2="500" y2="120" stroke="var(--chem)" strokeWidth="1.5" markerEnd="url(#sac)" />
            <text x="480" y="138" textAnchor="middle" fontFamily="var(--mono)" fontSize="12" fill="var(--chem)">K, V</text>
            <defs>
              <marker id="sa" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--accent)" /></marker>
              <marker id="sac" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--chem)" /></marker>
            </defs>
          </svg>
        </Figure>
        <CodeBlock path="picochem/attention.py · multihead_cross_attention_forward" lang="python" code={lines(RAW.attention, 116, 155)} />
      </Section>

      <Section id="p3-7" title="LayerNorm">
        <p>
          Deep stacks are easier to train if activations stay at a stable scale.
          LayerNorm takes one position's vector, subtracts its mean and divides by
          its standard deviation across the features, then applies a learned scale{" "}
          <Tex tex="\gamma" /> and shift <Tex tex="\beta" /> so the layer can
          undo the normalization if it wants to. Drag the sliders.
        </p>
        <LayerNormViz />
        <CodeBlock path="picochem/ops.py · layer_norm_forward" lang="python" code={pyDef(RAW.ops, "layer_norm_forward")} />
        <p>
          On the GPU this is a reduction kernel: one block per row, 256 threads
          cooperate to sum the row (a tree reduction in shared memory) for the
          mean and again for the variance, then normalize.
        </p>
        <CodeBlock path="picochem/kernels/cuda/layer_norm.cu" lang="cuda" code={lines(RAW.layerNorm, 9, 31)} />
      </Section>

      <Section id="p3-8" title="Residuals and pre-norm">
        <p>
          Every sub-layer is wrapped the same way:{" "}
          <Tex tex="x + \text{sublayer}(\text{LN}(x))" />. The residual (the{" "}
          <Tex tex="x +" />) gives the gradient a short path straight back through
          the network, which is what makes deep stacks trainable. Normalizing
          before the sub-layer rather than after (pre-norm) keeps that path clean,
          at the cost of needing one final LayerNorm at the very end.
        </p>
        <Figure caption="One pre-norm sub-layer: normalize, transform, then add the untouched input back. The residual arrow is the gradient's shortcut.">
          <svg viewBox="0 0 520 120" width="100%" role="img" aria-label="pre-norm residual block">
            <line x1="20" y1="60" x2="500" y2="60" stroke="var(--rule)" />
            {[
              { x: 70, t: "LN" },
              { x: 200, t: "sub-layer" },
            ].map((b) => (
              <g key={b.t}>
                <rect x={b.x} y={40} width={b.t === "LN" ? 70 : 130} height={40} rx={8} fill="var(--panel-2)" stroke="var(--panel-line)" />
                <text x={b.x + (b.t === "LN" ? 35 : 65)} y={64} textAnchor="middle" fontSize="13" fill="var(--ink)">{b.t}</text>
              </g>
            ))}
            <circle cx={400} cy={60} r={16} fill="var(--accent-soft)" stroke="var(--accent)" />
            <text x={400} y={66} textAnchor="middle" fontSize="18" fill="var(--accent-ink)">+</text>
            <path d="M 30 60 C 30 110, 400 110, 400 78" fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeDasharray="4 3" markerEnd="url(#rr)" />
            <text x="210" y="108" fontSize="11" fill="var(--accent-ink)" fontFamily="var(--mono)">residual: x carried around the sub-layer</text>
            <defs><marker id="rr" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--accent)" /></marker></defs>
          </svg>
        </Figure>
        <CodeBlock path="picochem/encoder.py · encoder_block_forward" lang="python" code={lines(RAW.encoder, 13, 39)} />
      </Section>

      <Section id="p3-9" title="Feed-forward network">
        <p>
          Attention moves information between positions; the feed forward network
          (FFN) transforms each position on its own. It is a two-layer MLP that
          expands the 512-dim vector to 2,048 (4×), applies a GeLU nonlinearity,
          and projects back to 512. GeLU is a smooth gate: it passes large
          positive values through and squashes negatives toward zero.
        </p>
        <Figure caption="The FFN widens to 4× the model dimension, applies GeLU, and narrows back. GeLU (right) is a smooth version of a gate.">
          <div style={{ display: "flex", gap: "1.5rem", alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
            <svg viewBox="0 0 300 110" width="300" height="110" role="img" aria-label="ffn shape">
              {[
                { x: 10, h: 40, l: "512" },
                { x: 120, h: 90, l: "2048" },
                { x: 230, h: 40, l: "512" },
              ].map((b, i) => (
                <g key={i}>
                  <rect x={b.x} y={55 - b.h / 2} width={50} height={b.h} rx={6} fill={i === 1 ? "var(--chem-soft)" : "var(--panel-2)"} stroke={i === 1 ? "var(--chem)" : "var(--panel-line)"} />
                  <text x={b.x + 25} y={59} textAnchor="middle" fontSize="11" fontFamily="var(--mono)" fill="var(--ink)">{b.l}</text>
                </g>
              ))}
              <text x={95} y={20} textAnchor="middle" fontSize="10" fill="var(--ink-faint)">GeLU</text>
              <text x={95} y={59} fontSize="14" fill="var(--ink-soft)">→</text>
              <text x={205} y={59} fontSize="14" fill="var(--ink-soft)">→</text>
            </svg>
            <GeluCurve />
          </div>
        </Figure>
        <CodeBlock path="picochem/ffn.py · ffn_forward  +  picochem/ops.py · gelu_forward" lang="python" code={pyDef(RAW.ffn, "ffn_forward") + "\n\n" + pyDef(RAW.ops, "gelu_forward")} />
        <CodeBlock path="picochem/kernels/cuda/gelu.cu" lang="cuda" code={lines(RAW.gelu, 6, 16)} />
      </Section>

      <Section id="p3-10" title="Encoder block">
        <p>
          An encoder block is just two pre-norm sub-layers stacked: self-attention
          so the SMILES tokens can mix with each other, then an FFN. The molecule
          goes in as embeddings and comes out as encoder memory, a vector per
          token that the decoder will read. Run the encoder half of the demo:
        </p>
        <MoleculeExplorer mode="encoder" lockMode initial="phenol" />
      </Section>

      <Section id="p3-11" title="Decoder block">
        <p>
          A decoder block has three sub-layers: causal self-attention over the
          name written so far, then cross-attention into the encoder memory (this
          is where it actually looks at the molecule), then an FFN. Run the
          decoder half: the name is produced one token at a time, each step
          attending back to the molecule.
        </p>
        <MoleculeExplorer mode="decoder" lockMode initial="aspirin" />
        <CodeBlock path="picochem/decoder.py · decoder_block_forward" lang="python" code={lines(RAW.decoder, 11, 51)} />
      </Section>

      <Section id="p3-12" title="Stacking the layers">
        <p>
          The full model is three encoder blocks and three decoder blocks, all on
          a 512-dimensional residual stream with 8 heads of 64. The decoder
          context is 64 tokens, plenty for a trace. Each decoder block reads the
          same final encoder memory.
        </p>
        <Figure caption="The whole stack: embeddings, three encoder blocks producing memory, three decoder blocks that each cross-attend to it, a final norm, and the output head.">
          <svg viewBox="0 0 560 150" width="100%" role="img" aria-label="full stack">
            {[
              { x: 10, l: "embed", c: "var(--panel-2)" },
              { x: 95, l: "enc ×3", c: "var(--panel)" },
              { x: 185, l: "memory", c: "var(--chem-soft)" },
              { x: 285, l: "dec ×3", c: "var(--panel)" },
              { x: 380, l: "final LN", c: "var(--panel-2)" },
              { x: 470, l: "head", c: "var(--accent-soft)" },
            ].map((b, i, a) => (
              <g key={b.l}>
                <rect x={b.x} y={50} width={75} height={48} rx={8} fill={b.c} stroke="var(--panel-line)" />
                <text x={b.x + 37} y={78} textAnchor="middle" fontSize="12" fill="var(--ink)">{b.l}</text>
                {i < a.length - 1 && <line x1={b.x + 75} y1={74} x2={a[i + 1].x} y2={74} stroke="var(--ink-faint)" strokeWidth="1.4" markerEnd="url(#st)" />}
              </g>
            ))}
            <path d="M 222 50 C 222 20, 322 20, 322 48" fill="none" stroke="var(--chem)" strokeDasharray="4 3" markerEnd="url(#stc)" />
            <text x="272" y="16" textAnchor="middle" fontSize="10" fill="var(--chem)">cross-attn</text>
            <text x="280" y="120" textAnchor="middle" fontSize="11" fill="var(--ink-faint)" fontFamily="var(--mono)">512 dim · 8 heads × 64 · 64-token decoder context</text>
            <defs>
              <marker id="st" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--ink-faint)" /></marker>
              <marker id="stc" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="var(--chem)" /></marker>
            </defs>
          </svg>
        </Figure>
      </Section>

      <Section id="p3-13" title="Output head and weight tying">
        <p>
          To turn the final decoder vectors into a guess, apply one last
          LayerNorm and project onto the vocabulary to get a logit per token. The
          projection is not a new matrix: it reuses the target embedding table
          (transposed). This weight tying saves parameters and ties the two jobs
          of that table together, so it receives gradient from both the input
          embedding lookup and the output projection.
        </p>
        <CodeBlock path="picochem/model.py · output projection (tied)" lang="python" code={lines(RAW.model, 121, 129)} />
        <Aside label="two gradient paths">
          In the backward pass the target token embedding accumulates a gradient
          from the output projection and another from the decoder input embedding,
          and they are summed. You can see both terms added in{" "}
          <code>model_backward</code>.
        </Aside>
      </Section>

      <Section id="p3-14" title="The full forward pass">
        <p>
          Putting it together: tokenize the SMILES, embed it, run three encoder
          blocks to get memory, embed the name written so far, run three decoder
          blocks that cross-attend to the memory, apply the final norm, and
          project to logits. Step a token through the whole pipeline.
        </p>
        <ForwardFlow />
      </Section>
    </>
  );
}
