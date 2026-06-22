import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Figure } from "../components/Figure";
import { TiledMatmul } from "../widgets/TiledMatmul";
import { RAW } from "../data/raw";
import { lines } from "../lib/code";

function GpuHierarchy() {
  return (
    <svg viewBox="0 0 560 210" width="100%" role="img" aria-label="GPU execution and memory hierarchy">
      {/* grid -> blocks -> threads */}
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

      {/* memory hierarchy column */}
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
        <Figure caption="Threads in blocks in a grid, and the memory hierarchy they reach. The slow step is anything that crosses the bus to the host, which is what the design works to avoid.">
          <GpuHierarchy />
        </Figure>
        <p>
          The single most important fact for performance is that copying data
          between host and device is expensive. A kernel that spends more time
          shuttling data than computing is wasted. That tax is what motivates both
          the tiled matmul (reuse data in shared memory) and the device-resident
          training loop (keep the model on the GPU).
        </p>
      </Section>

      <Section id="p5-2" title="The kernels">
        <p>
          The centerpiece is the tiled matmul. A naive matmul reloads operands
          from slow global memory for every multiply. The tiled version has each
          block cooperatively load a 16 by 16 tile of each input into shared
          memory, synchronize, multiply-accumulate from the fast copy, synchronize
          again, and move to the next tile. Watch the two barriers and the
          accumulate loop:
        </p>
        <TiledMatmul />
        <p>
          The same backward matmuls are two transposed variants of this kernel.
          Softmax, LayerNorm, and cross entropy are reduction kernels: one block
          per row, 256 threads, a tree reduction in shared memory to get the row
          max and sum. The embedding gradient is a scatter with{" "}
          <code>atomicAdd</code>. Here is the cross-entropy forward, a clean
          example of the reduction pattern:
        </p>
        <CodeBlock path="picochem/kernels/cuda/cross_entropy.cu" lang="cuda" code={lines(RAW.crossEntropy, 9, 38)} />
        <CodeBlock path="picochem/kernels/cuda/matmul_backward.cu · transposed backward matmul" lang="cuda" code={lines(RAW.matmulBackward, 7, 31)} />
        <Aside label="honesty: exploratory kernels" warn>
          The repository also contains a Tensor Core kernel (<code>tensor_core.cu</code>,
          using the WMMA API and fp16) and a cuBLAS wrapper (<code>matmul_cublas.cu</code>).
          These were exploratory benchmarking experiments and are not part of the
          shipped training path, which uses the hand-written fp32 tiled kernels
          above.
        </Aside>
      </Section>

      <Section id="p5-3" title="Device resident training">
        <p>
          The win is keeping the whole transformer stack on the GPU across both
          the forward and backward pass, so a training step does not copy weights
          back and forth every iteration. Only the batch goes in and the loss and
          gradients come out. That is about 60× faster per step than the NumPy
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
