// Real SMILES tokenizer: the Schwaller et al. 2017 pattern, ported verbatim
// from picochem/data.py. Used live by the SMILES tokenizer widget (W2) and for
// coloring SMILES tokens consistently across the guide.

export const SMILES_PATTERN =
  /(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])/g;

export type TokType =
  | "atom"
  | "aromatic"
  | "bond"
  | "branch"
  | "ring"
  | "bracket"
  | "other";

export interface SmilesToken {
  text: string;
  type: TokType;
}

const TYPE_LABEL: Record<TokType, string> = {
  atom: "atom",
  aromatic: "aromatic atom",
  bond: "bond",
  branch: "branch",
  ring: "ring digit",
  bracket: "bracket atom",
  other: "other",
};

export function tokTypeLabel(t: TokType): string {
  return TYPE_LABEL[t];
}

export function classify(tok: string): TokType {
  if (tok.startsWith("[")) return "bracket";
  if (/^(Br|Cl|N|O|S|P|F|I|B|C)$/.test(tok)) return "atom"; // C via Cl?, B via Br?
  if (/^[bcnops]$/.test(tok)) return "aromatic";
  if (tok === "(" || tok === ")") return "branch";
  if (/^[0-9]$/.test(tok) || /^%[0-9]{2}$/.test(tok)) return "ring";
  if (/^[=#\-+\\/:~]$/.test(tok)) return "bond";
  return "other";
}

/** Tokenize a SMILES string into typed tokens. Never throws on partial input. */
export function tokenizeSmiles(s: string): SmilesToken[] {
  const out: SmilesToken[] = [];
  if (!s) return out;
  const re = new RegExp(SMILES_PATTERN.source, "g");
  let m: RegExpExecArray | null;
  while ((m = re.exec(s)) !== null) {
    if (m.index === re.lastIndex) re.lastIndex++; // guard against zero-width
    out.push({ text: m[0], type: classify(m[0]) });
  }
  return out;
}

export const TOK_CLASS: Record<TokType, string> = {
  atom: "tok-atom",
  aromatic: "tok-aromatic",
  bond: "tok-bond",
  branch: "tok-branch",
  ring: "tok-ring",
  bracket: "tok-bracket",
  other: "tok-other",
};
