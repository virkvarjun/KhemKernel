import { PartRule, Section } from "../components/ui";

const SPEC: [string, string][] = [
  ["model dimension", "512"],
  ["attention heads", "8 (64 dims each)"],
  ["encoder layers", "3"],
  ["decoder layers", "3"],
  ["feed forward width", "2,048 (4× model dim)"],
  ["decoder context", "64 tokens"],
  ["source vocab (SMILES)", "341 (regex tokenizer)"],
  ["target vocab (IUPAC)", "4,000 (byte pair encoding)"],
  ["positional embeddings", "learned"],
  ["normalization", "pre-norm + final LayerNorm"],
  ["output projection", "weight-tied to target embedding"],
  ["init scale", "0.02 (Gaussian)"],
  ["optimizer", "Adam (β1 0.9, β2 0.999, ε 1e-8)"],
  ["schedule", "linear warmup 1,500 then cosine"],
  ["peak learning rate", "3e-4"],
  ["training steps", "~120,000"],
  ["batch size", "32"],
  ["loss", "softmax cross entropy, ignore_index −1"],
  ["precision", "float64 (NumPy ref) / float32 (CUDA)"],
];

const FILEMAP: [string, string][] = [
  ["ops.py · linear_forward", "matmul_tiled.cu"],
  ["ops.py · linear_backward", "matmul_backward.cu (two transposed variants)"],
  ["ops.py · gelu_forward / backward", "gelu.cu"],
  ["attention.py · softmax (in attention)", "softmax.cu / softmax_backward.cu"],
  ["ops.py · layer_norm_forward / backward", "layer_norm.cu / layer_norm_backward.cu"],
  ["ops.py · softmax_cross_entropy", "cross_entropy.cu"],
  ["embeddings.py · token_embedding_backward", "embedding.cu (atomicAdd scatter)"],
  ["optimizer.py · adam_step", "adam.cu"],
  ["attention.py · scaled dot product (batched)", "batched_matmul.cu"],
  ["attention.py · split / merge heads", "transpose.cu"],
  ["ops.py · linear bias add", "bias.cu"],
  ["residual add (x + sublayer)", "vector_add.cu"],
  ["device_layers.py (resident stack)", "bindings.cpp (DeviceTensor, pybind11)"],
];

const GLOSSARY: [string, string][] = [
  ["SMILES", "A line notation that writes a molecule's graph as a string of atoms, bonds, and ring digits."],
  ["IUPAC name", "The formal, rule-based systematic name of a molecule."],
  ["SMARTS", "A pattern language for matching substructures in molecules, used to detect functional groups for the trace."],
  ["OPSIN", "An open-source parser that turns an IUPAC name back into a molecule. The verifier depends on it."],
  ["canonical SMILES", "A unique normalized SMILES for a molecule, so two encodings of the same structure compare equal."],
  ["token", "A discrete unit the model has an id and an embedding for."],
  ["BPE", "Byte pair encoding: a tokenizer that starts from characters and merges frequent adjacent pairs into subword tokens."],
  ["attention", "A mechanism where each position pulls a weighted mix of information from other positions."],
  ["head", "One of several parallel attentions, each on a slice of the vector, able to track a different relationship."],
  ["LayerNorm", "Normalizes a vector to zero mean and unit variance across its features, then rescales and shifts it."],
  ["residual", "Adding a sub-layer's input to its output, giving gradients a short path and stabilizing deep stacks."],
  ["logits", "The raw, un-normalized scores over the vocabulary, before softmax turns them into probabilities."],
  ["teacher forcing", "Training the decoder on the true previous tokens rather than its own predictions."],
  ["kernel", "A function that runs on the GPU across many threads at once."],
  ["shared memory", "A small, fast scratchpad shared by the threads of one GPU block."],
  ["atomicAdd", "A GPU operation that adds to a memory location safely when many threads target the same address."],
  ["beam search", "Decoding that keeps the best few partial sequences at each step instead of committing to one."],
  ["ablation", "Removing or changing one component (a target, a head, a layer) to measure how much it mattered."],
  ["linear probe", "A linear classifier fit on frozen activations to test how decodable a property is from them."],
  ["attention entropy", "How spread out an attention row is; low means a sharp, near-hard lookup, high means a diffuse average."],
  ["soft alignment", "The attention weights read as a differentiable correspondence, here between name fragments and atoms."],
  ["selective prediction", "Choosing to abstain when unsure, trading coverage for higher precision on the answers given."],
  ["verifiable reward", "A reward computed by a checker (parser, compiler, proof checker) rather than a human label."],
  ["compression ratio", "How much shorter a sequence gets after tokenization, e.g. characters per token."],
  ["roofline", "A model of whether a kernel is limited by compute or by memory bandwidth."],
  ["induction head", "An attention head that learns to copy or continue a pattern it has seen earlier in the sequence."],
];

function Two({ rows, left, right }: { rows: [string, string][]; left: string; right: string }) {
  return (
    <div className="table-scroll">
      <table className="spec">
        <thead>
          <tr><th>{left}</th><th>{right}</th></tr>
        </thead>
        <tbody>
          {rows.map(([a, b]) => (
            <tr key={a}>
              <td className="mono">{a}</td>
              <td>{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Appendix() {
  return (
    <>
      <PartRule part="Appendix" title="" />

      <Section id="appendix-a" title="A. Spec sheet">
        <p>Every hyperparameter of the final model in one place.</p>
        <Two rows={SPEC} left="parameter" right="value" />
      </Section>

      <Section id="appendix-b" title="B. File map">
        <p>
          Each NumPy reference piece and the CUDA kernel that implements it. The
          two share weights and a tokenizer; the kernels are validated against the
          NumPy version with finite-difference and parity tests.
        </p>
        <Two rows={FILEMAP} left="NumPy reference" right="CUDA kernel" />
      </Section>

      <Section id="appendix-c" title="C. Glossary">
        <Two rows={GLOSSARY} left="term" right="meaning" />
      </Section>
    </>
  );
}
