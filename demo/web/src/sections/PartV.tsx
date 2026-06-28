import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Figure } from "../components/Figure";
import { TiledMatmul } from "../widgets/TiledMatmul";
import { RAW } from "../data/raw";
import { lines } from "../lib/code";

function GpuHierarchy() {
  return (
    <svg viewBox="0 0 560 210" width="100%" role="img" aria-label="GPU execution and memory hierarchy">
      <rect x={14} y={14} width={300} height={182} rx={10} fill="var(--panel-2)" stroke="var(--panel-line)" />
      <text x={24} y={32} fontSize="11" fontFamily="var(--mono)" fill="var(--ink-faint)">grid</text>
      {[0, 1].map((bx) =>
        [0, 1].map((by) => (
          <g key={`${bx}-${by}`}>
            <rect x={30 + bx * 145} y={42 + by * 78} width={130} height={68} rx={7} fill="var(--panel)" stroke="var(--accent-soft)" />
            <text x={36 + bx * 145} y={56 + by * 78} fontSize="9" fontFamily="var(--mono)" fill="var(--ink-faint)">block (shared mem)</text>
            {Array.from({ length: 12 }).map((_, t) => (
              <rect key={t} x={38 + bx * 145 + (t % 6) * 19} y={64 + by * 78 + Math.floor(t / 6) * 18} width={15} height={14} rx={2} fill="var(--accent)" opacity={0.5} />
            ))}
          </g>
        )),
      )}
      <text x={150} y={190} fontSize="9" fontFamily="var(--mono)" fill="var(--ink-faint)">threads (registers)</text>
      {[
        { y: 24, l: "registers", s: "per thread · fastest" },
        { y: 74, l: "shared memory", s: "per block · fast" },
        { y: 124, l: "global memory", s: "whole GPU · slow" },
        { y: 174, l: "host (CPU) RAM", s: "over the bus · slowest" },
      ].map((m) => (
        <g key={m.l}>
          <rect x={340} y={m.y} width={206} height={40} rx={7} fill="var(--panel)" stroke="var(--panel-line)" />
          <text x={350} y={m.y + 18} fontSize="12" fill="var(--ink)">{m.l}</text>
          <text x={350} y={m.y + 32} fontSize="9.5" fontFamily="var(--mono)" fill="var(--ink-faint)">{m.s}</text>
        </g>
      ))}
    </svg>
  );
}

function SpeedupBar() {
  return (
    <svg viewBox="0 0 520 90" width="100%" role="img" aria-label="numpy vs device step time">
      {[
        { l: "NumPy (float64)", ms: 2640, y: 14, c: "var(--ink-faint)" },
        { l: "device resident", ms: 44, y: 50, c: "var(--accent)" },
      ].map((b) => (
        <g key={b.l}>
          <text x={4} y={b.y + 14} fontSize="11" fill="var(--ink-soft)">{b.l}</text>
          <rect x={130} y={b.y} width={(b.ms / 2640) * 340} height={22} rx={4} fill={b.c} />
          <text x={130 + (b.ms / 2640) * 340 + 6} y={b.y + 16} fontSize="11" fontFamily="var(--mono)" fill="var(--ink)">{b.ms} ms</text>
        </g>
      ))}
      <text x={130} y={86} fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono)">one training step (fwd+bwd), B=16 S=T=32 D=256, 3+3 layers, V=8000 → ~60×</text>
    </svg>
  );
}

// the kernel catalog, for the overview table
const KERNELS: [string, string, string][] = [
  ["matmul (naive, tiled)", "matmul_naive.cu, matmul_tiled.cu", "every linear layer"],
  ["backward matmul (NT, TN)", "matmul_backward.cu", "linear layer gradients"],
  ["batched matmul", "batched_matmul.cu", "attention QKᵀ and weights·V"],
  ["softmax (+ backward)", "softmax.cu, softmax_backward.cu", "attention weights"],
  ["layer norm (+ backward)", "layer_norm.cu, layer_norm_backward.cu", "every sub-layer"],
  ["cross entropy", "cross_entropy.cu", "the training loss"],
  ["GeLU (+ backward)", "gelu.cu", "the FFN nonlinearity"],
  ["bias add / colsum / scale", "bias.cu", "linear bias + its gradient"],
  ["vector add", "vector_add.cu", "residual connections"],
  ["embedding scatter", "embedding.cu", "embedding gradients (atomics)"],
  ["split / merge heads", "transpose.cu", "the head axis"],
  ["Adam", "adam.cu", "the optimizer update"],
];

export function PartV() {
  return (
    <>
      <PartRule part="Part V" title="The GPU and CUDA Implementation" />

      <Section id="p5-1" title="What a GPU is">
        <p>
          A GPU runs thousands of tiny threads at once. Threads are grouped into
          blocks, and blocks into a grid; the threads in one block can share a
          small, fast scratchpad called shared memory and synchronize with each
          other. Memory comes in a hierarchy: registers (per thread, fastest),
          shared memory (per block), global memory (the whole GPU, large but
          slow), and across the bus, the host CPU's RAM (slowest of all).
        </p>
        <Figure caption="Threads in blocks in a grid, and the memory hierarchy they reach. The slow step is anything that crosses the bus to the host.">
          <GpuHierarchy />
        </Figure>
        <p>
          Every operation in the transformer is one of a dozen hand-written
          kernels. Each one is its own <code>.cu</code> file with a standalone
          self-test that checks it against a plain CPU reference, and the same
          kernels are exercised from Python against the NumPy model (Part IX). The
          rest of this part walks the catalog, from the matmul that dominates the
          FLOPs down to the one-line elementwise kernels.
        </p>
        <div className="table-scroll">
          <table className="spec">
            <thead><tr><th>kernel</th><th>file</th><th>used for</th></tr></thead>
            <tbody>
              {KERNELS.map((k) => (
                <tr key={k[0]}><td>{k[0]}</td><td className="mono" style={{ fontSize: "0.8em" }}>{k[1]}</td><td>{k[2]}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section id="p5-2" title="The tiled matmul">
        <p>
          Matrix multiply dominates the compute, so it is worth getting right. The
          naive kernel assigns one thread per output element and has it walk the
          full contraction dimension, reading both inputs straight from slow
          global memory. It is correct and simple, but every input value is
          re-read many times from global memory.
        </p>
        <CodeBlock path="picochem/kernels/cuda/matmul_naive.cu" lang="cuda" code={lines(RAW.matmulNaive, 6, 17)} />
        <p>
          The tiled kernel fixes the memory traffic. The output is cut into 16 by
          16 tiles, one per block. The block cooperatively loads a 16 by 16 tile
          of each input into shared memory, calls <code>__syncthreads()</code> so
          every thread sees the full tile, multiply-accumulates from the fast copy,
          synchronizes again before overwriting, and advances to the next tile
          along the contraction axis. Each value loaded from global memory is now
          reused 16 times, which raises the arithmetic intensity enough to make the
          kernel compute-bound instead of bandwidth-bound.
        </p>
        <CodeBlock path="picochem/kernels/cuda/matmul_tiled.cu" lang="cuda" code={lines(RAW.matmulTiled, 9, 35)} />
        <p>Step through the two barriers and the accumulate loop:</p>
        <TiledMatmul />
        <Aside label="honesty: exploratory kernels" warn>
          There is also a Tensor Core matmul (<code>tensor_core.cu</code>, WMMA +
          fp16) and a cuBLAS wrapper (<code>matmul_cublas.cu</code>). Both were
          benchmarking experiments and are not in the shipped training path, which
          uses these hand-written fp32 kernels. Part IX explains why faster matmul
          would not have moved the wall clock much anyway.
        </Aside>
      </Section>

      <Section id="p5-4" title="Backward matmuls">
        <p>
          A linear layer's backward pass needs two more matmuls, and both involve a
          transpose: the gradient to the input is{" "}
          <code>grad_y @ Wᵀ</code> and the gradient to the weights is{" "}
          <code>xᵀ @ grad_y</code>. Rather than physically transpose a matrix
          (an extra pass over memory), the transpose is baked into the indexing.
          There are two variants: an NT kernel that treats the second operand as
          transposed, and a TN kernel that treats the first operand as transposed.
          Otherwise they are the same tiled scheme as the forward matmul.
        </p>
        <CodeBlock path="picochem/kernels/cuda/matmul_backward.cu" lang="cuda" code={lines(RAW.matmulBackward, 8, 36)} />
        <p>
          The only change from the forward kernel is which index gets multiplied by
          the stride when loading the shared-memory tile. Reading a logical
          transpose for free, just by swapping <code>row * K + k</code> for{" "}
          <code>k * M + row</code>, is the whole trick.
        </p>
      </Section>

      <Section id="p5-5" title="Batched matmul">
        <p>
          Attention is many small matmuls, one per (batch, head): the scores are{" "}
          <code>Q @ Kᵀ</code> per head, and the output is{" "}
          <code>weights @ V</code> per head. Doing them as a batch keeps the GPU
          busy. The batched kernel uses the grid's third dimension,{" "}
          <code>blockIdx.z</code>, to pick the batch element, offsets the three
          pointers by that element's stride, and then runs the same tiled inner
          loop. It also carries <code>transA</code> and <code>transB</code> flags,
          so the one kernel covers <code>Q @ Kᵀ</code> (transpose the keys) and{" "}
          <code>weights @ V</code> (no transpose) without a separate
          implementation.
        </p>
        <CodeBlock path="picochem/kernels/cuda/batched_matmul.cu" lang="cuda" code={lines(RAW.batchedMatmul, 8, 41)} />
      </Section>

      <Section id="p5-6" title="Reductions">
        <p>
          Softmax, layer norm, and cross entropy all need a per-row reduction: a
          sum or a max over the feature axis. These share one shape. Launch one
          block per row with 256 threads; each thread strides over the row
          accumulating a partial result, the partials go into shared memory, and a
          tree reduction halves the active threads each step until thread 0 holds
          the total. Softmax does this twice, once for the row max (for numerical
          stability) and once for the exponential sum, the standard log-sum-exp:
        </p>
        <CodeBlock path="picochem/kernels/cuda/softmax.cu" lang="cuda" code={lines(RAW.softmax, 11, 51)} />
        <p>
          Layer norm is the same kernel with two reductions, mean then variance,
          before it normalizes (you saw it in Part III). Cross entropy reduces for
          the row max and the log-sum-exp, then thread 0 reads off the negative log
          probability of the target. The tree reduction is the reusable primitive
          underneath all three; writing it once by hand is most of the work.
        </p>
      </Section>

      <Section id="p5-7" title="Elementwise kernels">
        <p>
          The cheap kernels are the elementwise ones: one thread per element, a
          bounds check, and a single arithmetic expression. GeLU applies its tanh
          approximation, <code>vector_add</code> is the residual connection, scale
          multiplies by a constant, and Adam does the in-place parameter update.
          The only subtlety is the bias: the forward broadcasts a length-N vector
          across all rows, and its gradient is a column sum, which is one thread
          per column walking down the rows.
        </p>
        <CodeBlock path="picochem/kernels/cuda/bias.cu" lang="cuda" code={lines(RAW.bias, 9, 30)} />
        <p>
          The Adam kernel (shown in Part IV) is also elementwise: one thread per
          parameter does the momentum and variance update and the bias-corrected
          step in place, so the weights never leave the device. None of these
          kernels touch shared memory; they are pure bandwidth, and they are fast
          because the data is already resident on the GPU.
        </p>
      </Section>

      <Section id="p5-8" title="Embedding scatter">
        <p>
          The embedding gradient is the one kernel that needs atomics. The forward
          is a gather (look up a row per token); the backward scatters each token's
          gradient back to its row of the table. The catch is collisions: in a
          batch, many positions map to the same token, so several threads add to
          the same table row at once. A plain <code>+=</code> would race and lose
          updates, so the kernel uses <code>atomicAdd</code>, which serializes the
          conflicting adds so none are dropped.
        </p>
        <CodeBlock path="picochem/kernels/cuda/embedding.cu" lang="cuda" code={lines(RAW.embeddingCu, 8, 17)} />
        <p>
          In the shipped trainer this scatter actually runs on the host, because
          the embedding tables stay on the CPU (the reason the GPU sits at 35 to
          40% utilization, covered in the last section and in Part IX). The kernel
          exists, is tested, and is the device-side version of the same operation.
        </p>
      </Section>

      <Section id="p5-9" title="Head transpose">
        <p>
          Multi-head attention needs the data laid out two ways: the projections
          produce <code>(B, S, H, Dh)</code>, but the batched attention matmul
          wants the head axis next to the batch axis, <code>(B, H, S, Dh)</code>,
          so each (batch, head) is a contiguous matrix. The split-heads and
          merge-heads kernels do that reshuffle. They are pure index arithmetic:
          one thread per element decodes its multi-dimensional index, recomputes
          the source offset under the other layout, and copies one value.
        </p>
        <CodeBlock path="picochem/kernels/cuda/transpose.cu" lang="cuda" code={lines(RAW.transpose, 9, 32)} />
        <p>
          There is no math here, just a permutation, but doing it as its own kernel
          keeps the attention matmul reading contiguous memory, which matters more
          for speed than the copy costs.
        </p>
      </Section>

      <Section id="p5-10" title="Binding to Python">
        <p>
          The kernels reach Python through pybind11. The core type is{" "}
          <code>DeviceTensor</code>, a thin RAII wrapper around a GPU buffer: its
          constructor <code>cudaMalloc</code>s and uploads a NumPy array, its
          destructor <code>cudaFree</code>s, and it is non-copyable so ownership is
          unambiguous. A <code>.numpy()</code> method downloads back to the host
          when you actually want to look at a result.
        </p>
        <CodeBlock path="picochem/kernels/cuda/bindings.cpp" lang="cpp" code={lines(RAW.bindings, 38, 65)} />
        <p>
          The module exposes two families of functions. The copy-based ones (used
          for the standalone benchmarks) take NumPy arrays, upload, run, and
          download. The device-resident <code>dt_</code> family takes and returns{" "}
          <code>DeviceTensor</code> handles, so a whole training step chains
          kernels without a single host round trip. Returns use{" "}
          <code>take_ownership</code> so Python's garbage collector frees the GPU
          memory.
        </p>
        <CodeBlock path="picochem/kernels/cuda/bindings.cpp · the module" lang="cpp" code={lines(RAW.bindings, 539, 552)} />
      </Section>

      <Section id="p5-3" title="Device-resident training">
        <p>
          The win is keeping the whole transformer stack on the GPU across both
          the forward and backward pass, so a training step does not copy weights
          back and forth every iteration. Only the batch goes in and the loss and
          gradients come out. That is about 60x faster per step than the NumPy
          reference.
        </p>
        <Figure caption="2,640 ms in NumPy down to 44 ms device-resident, on the benchmark configuration.">
          <SpeedupBar />
        </Figure>
        <p>
          One thing deliberately stays on the host: the embedding tables, with
          their gather and scatter. That split is why the GPU sits at only 35 to
          40% utilization during training: the host-side embedding work and its
          transfers are the limiting factor, not the matmuls. For a model this
          size that is fine, and it means the choice of GPU barely matters. The
          resident model forward mirrors the NumPy one exactly, op for op:
        </p>
        <CodeBlock path="picochem/device_layers.py · model_forward" lang="python" code={lines(RAW.deviceLayers, 300, 328)} />
        <Aside label="building it">
          The build auto-detects the GPU architecture and, when the card is newer
          than the toolkit, falls back to emitting compute_90 PTX that the driver
          compiles at load time. That is what let the final model train on a
          Blackwell card (compute capability 12.0) under a CUDA 12.4 toolkit that
          does not know that architecture.
        </Aside>
      </Section>
    </>
  );
}
