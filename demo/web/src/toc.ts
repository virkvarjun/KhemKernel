// Canonical table of contents. Drives the ToC sidebar, scroll spy, and the
// ordering of section components. Section ids must match the rendered <section>
// element ids exactly. Part titles avoid em dashes per the guide's style rule.

export interface TocItem {
  id: string;
  label: string;
}
export interface TocPart {
  part: string; // e.g. "Part I"
  title: string; // e.g. "The Problem"
  items: TocItem[];
}

export const TOC: TocPart[] = [
  {
    part: "Part I",
    title: "The Problem",
    items: [
      { id: "p1-1", label: "1. What the model does" },
      { id: "p1-2", label: "2. A translation problem" },
      { id: "p1-3", label: "3. The headline result" },
    ],
  },
  {
    part: "Part II",
    title: "Tokenization",
    items: [
      { id: "p2-1", label: "1. SMILES tokenizer" },
      { id: "p2-2", label: "2. Why IUPAC needs BPE" },
      { id: "p2-3", label: "3. Special tokens" },
    ],
  },
  {
    part: "Part III",
    title: "The Transformer Architecture",
    items: [
      { id: "p3-1", label: "1. Embeddings" },
      { id: "p3-2", label: "2. Linear layer" },
      { id: "p3-3", label: "3. Scaled dot-product attention" },
      { id: "p3-4", label: "4. Multi-head attention" },
      { id: "p3-5", label: "5. Masking" },
      { id: "p3-6", label: "6. Self vs cross attention" },
      { id: "p3-7", label: "7. LayerNorm" },
      { id: "p3-8", label: "8. Residuals and pre-norm" },
      { id: "p3-9", label: "9. Feed-forward network" },
      { id: "p3-10", label: "10. Encoder block" },
      { id: "p3-11", label: "11. Decoder block" },
      { id: "p3-12", label: "12. Stacking the layers" },
      { id: "p3-13", label: "13. Output head and weight tying" },
      { id: "p3-14", label: "14. The full forward pass" },
    ],
  },
  {
    part: "Part IV",
    title: "Training",
    items: [
      { id: "p4-1", label: "1. Teacher forcing" },
      { id: "p4-2", label: "2. The loss" },
      { id: "p4-3", label: "3. Backpropagation" },
      { id: "p4-4", label: "4. Optimizer and schedule" },
      { id: "p4-5", label: "5. Stability" },
    ],
  },
  {
    part: "Part V",
    title: "GPU and CUDA Kernels",
    items: [
      { id: "p5-1", label: "1. What a GPU is" },
      { id: "p5-2", label: "2. The tiled matmul" },
      { id: "p5-4", label: "3. Backward matmuls" },
      { id: "p5-5", label: "4. Batched matmul" },
      { id: "p5-6", label: "5. Reductions" },
      { id: "p5-7", label: "6. Elementwise kernels" },
      { id: "p5-8", label: "7. Embedding scatter" },
      { id: "p5-9", label: "8. Head transpose" },
      { id: "p5-10", label: "9. Binding to Python" },
      { id: "p5-3", label: "10. Device-resident training" },
    ],
  },
  {
    part: "Part VI",
    title: "Inference and the Verifier",
    items: [
      { id: "p6-1", label: "1. Decoding" },
      { id: "p6-2", label: "2. The free verifier" },
    ],
  },
  {
    part: "Part VII",
    title: "Results",
    items: [
      { id: "p7-1", label: "1. Evaluation" },
      { id: "p7-2", label: "2. Failure analysis" },
      { id: "p7-3", label: "3. What is next" },
    ],
  },
  {
    part: "Part VIII",
    title: "Going Deeper",
    items: [
      { id: "p8-1", label: "1. BPE internals" },
      { id: "p8-2", label: "2. Tokenizer ablation" },
      { id: "p8-3", label: "3. Scaling the scores" },
      { id: "p8-4", label: "4. Attention as retrieval" },
      { id: "p8-5", label: "5. Cross-attention alignment" },
      { id: "p8-6", label: "6. Head specialization" },
    ],
  },
  {
    part: "Part IX",
    title: "Analysis",
    items: [
      { id: "p9-1", label: "1. Does the trace help?" },
      { id: "p9-2", label: "2. The verifier as a reward" },
      { id: "p9-3", label: "3. Linear probes" },
      { id: "p9-4", label: "4. Gradient checking" },
      { id: "p9-5", label: "5. Where the GPU time goes" },
    ],
  },
  {
    part: "Appendix",
    title: "",
    items: [
      { id: "appendix-a", label: "A. Spec sheet" },
      { id: "appendix-b", label: "B. File map" },
      { id: "appendix-c", label: "C. Glossary" },
    ],
  },
];

export const ALL_SECTION_IDS: string[] = [
  "top",
  ...TOC.flatMap((p) => p.items.map((i) => i.id)),
];
