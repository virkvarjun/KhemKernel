import vocab from "./smiles_vocab.json";

// The real 341-token SMILES vocabulary (built by scripts/build_vocab.py),
// vendored so the guide is self-contained and can report token vocab status.
export const SMILES_VOCAB: Record<string, number> = vocab as Record<string, number>;
export const SMILES_VOCAB_SET = new Set(Object.keys(SMILES_VOCAB));
export const SMILES_VOCAB_SIZE = Object.keys(SMILES_VOCAB).length;

export function inVocab(tok: string): boolean {
  return SMILES_VOCAB_SET.has(tok);
}
