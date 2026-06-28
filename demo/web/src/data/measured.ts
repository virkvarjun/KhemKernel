// Real measurements used by the "Going Deeper" and "Analysis" sections, so the
// figures cite actual numbers rather than illustrative ones. Anything NOT in
// this file (probe-by-layer, rank-of-truth, ablation deltas) is labeled
// schematic in the UI because it has not been run at scale yet.

// Finite-difference gradient checks: max relative error between the hand-derived
// analytic gradient and a central-difference estimate (eps 1e-6), float64.
export const GRAD_CHECKS: { op: string; err: string }[] = [
  { op: "linear (xW + b)", err: "5.6e-11" },
  { op: "GeLU", err: "9.1e-11" },
  { op: "LayerNorm", err: "1.5e-10" },
  { op: "softmax + cross-entropy", err: "1.6e-9" },
  { op: "scaled dot-product attention", err: "2.7e-10" },
];

export const BPE_VOCAB = 4000;
export const BPE_MERGES = 3927;

// chars vs old word-level tokens vs learned BPE tokens, real tokenizer (v2).
export const TOK_COUNTS: { name: string; chars: number; word: number; bpe: number }[] = [
  { name: "ethanol", chars: 7, word: 1, bpe: 1 },
  { name: "phenol", chars: 6, word: 1, bpe: 1 },
  { name: "2-aminopropanoic acid", chars: 21, word: 4, bpe: 2 },
  { name: "2-acetyloxybenzoic acid", chars: 23, word: 4, bpe: 3 },
  { name: "2-[4-(2-methylpropyl)phenyl]propanoic acid", chars: 42, word: 14, bpe: 4 },
];

// real BPE pieces for two of the names
export const BPE_PIECES: Record<string, string[]> = {
  aspirin: ["2-", "acetyloxy", "benzoic acid"],
  ibuprofen: ["2-[4-", "(2-methylpropyl)", "phenyl]", "propanoic acid"],
};
