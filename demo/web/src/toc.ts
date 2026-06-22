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
      { id: "p1-2", label: "2. Why it is a translation problem" },
      { id: "p1-3", label: "3. The headline result up front" },
    ],
  },
  {
    part: "Part II",
    title: "Tokenization and Encoding the Chemistry",
    items: [
      { id: "p2-1", label: "1. SMILES and the regex tokenizer" },
      { id: "p2-2", label: "2. Why IUPAC needs BPE" },
      { id: "p2-3", label: "3. Special tokens and losslessness" },
    ],
  },
  {
    part: "Part III",
    title: "The Transformer Architecture",
    items: [
      { id: "p3-1", label: "1. From ids to vectors: embeddings" },
      { id: "p3-2", label: "2. The building block: the Linear layer" },
      { id: "p3-3", label: "3. Attention I: scaled dot product" },
      { id: "p3-4", label: "4. Attention II: multi head" },
      { id: "p3-5", label: "5. Attention III: masking" },
      { id: "p3-6", label: "6. Self attention vs cross attention" },
      { id: "p3-7", label: "7. Normalization: LayerNorm" },
      { id: "p3-8", label: "8. Residuals and the pre norm block" },
      { id: "p3-9", label: "9. The feed forward network" },
      { id: "p3-10", label: "10. The encoder block" },
      { id: "p3-11", label: "11. The decoder block" },
      { id: "p3-12", label: "12. Stacking into encoder and decoder" },
      { id: "p3-13", label: "13. The output head and weight tying" },
      { id: "p3-14", label: "14. The full forward pass" },
    ],
  },
  {
    part: "Part IV",
    title: "Training the Model",
    items: [
      { id: "p4-1", label: "1. The target and teacher forcing" },
      { id: "p4-2", label: "2. The loss" },
      { id: "p4-3", label: "3. Backpropagation by hand" },
      { id: "p4-4", label: "4. The optimizer and schedule" },
      { id: "p4-5", label: "5. The training loop and stability" },
    ],
  },
  {
    part: "Part V",
    title: "The GPU and CUDA Implementation",
    items: [
      { id: "p5-1", label: "1. What a GPU is" },
      { id: "p5-2", label: "2. The kernels" },
      { id: "p5-3", label: "3. Device resident training" },
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
    title: "Results and What Is Left",
    items: [
      { id: "p7-1", label: "1. Evaluation" },
      { id: "p7-2", label: "2. Failure analysis" },
      { id: "p7-3", label: "3. What is next" },
    ],
  },
  {
    part: "Appendix",
    title: "",
    items: [
      { id: "appendix-a", label: "A. Architecture spec sheet" },
      { id: "appendix-b", label: "B. The file map" },
      { id: "appendix-c", label: "C. Glossary of terms" },
    ],
  },
];

export const ALL_SECTION_IDS: string[] = [
  "top",
  ...TOC.flatMap((p) => p.items.map((i) => i.id)),
];
