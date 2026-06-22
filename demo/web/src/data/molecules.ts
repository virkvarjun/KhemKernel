// The five molecules used everywhere in the guide. Traces are the real RDKit
// reasoning-trace targets produced by picochem/traces.py build_trace(), the
// exact format the model is trained to emit (and gets right for these).

export interface Molecule {
  key: string;
  label: string; // common name
  smiles: string;
  iupac: string;
  trace: string;
  parsed: {
    parent: string;
    groups: string[];
    atoms: number;
    rings: number;
  };
}

function parse(trace: string) {
  const g = (tag: string) =>
    trace.match(new RegExp(`<${tag}>(.*?)</${tag}>`))?.[1] ?? "";
  const groupsRaw = g("groups");
  return {
    parent: g("parent"),
    groups: groupsRaw === "none" ? [] : groupsRaw.split(";"),
    atoms: parseInt(g("atoms"), 10),
    rings: parseInt(g("rings"), 10),
  };
}

const T = {
  ethanol:
    "<parent>chain_C2</parent><groups>alcohol</groups><atoms>3</atoms><rings>0</rings><name>ethanol</name>",
  phenol:
    "<parent>benzene</parent><groups>phenol</groups><atoms>7</atoms><rings>1</rings><name>phenol</name>",
  alanine:
    "<parent>chain_C3</parent><groups>carboxylic_acid;amine</groups><atoms>6</atoms><rings>0</rings><name>2-aminopropanoic acid</name>",
  aspirin:
    "<parent>benzene</parent><groups>carboxylic_acid;ester;ether</groups><atoms>13</atoms><rings>1</rings><name>2-acetyloxybenzoic acid</name>",
  ibuprofen:
    "<parent>benzene</parent><groups>carboxylic_acid</groups><atoms>15</atoms><rings>1</rings><name>2-[4-(2-methylpropyl)phenyl]propanoic acid</name>",
};

export const MOLECULES: Molecule[] = [
  { key: "ethanol", label: "ethanol", smiles: "CCO", iupac: "ethanol", trace: T.ethanol, parsed: parse(T.ethanol) },
  { key: "phenol", label: "phenol", smiles: "c1ccc(O)cc1", iupac: "phenol", trace: T.phenol, parsed: parse(T.phenol) },
  { key: "alanine", label: "alanine", smiles: "CC(N)C(=O)O", iupac: "2-aminopropanoic acid", trace: T.alanine, parsed: parse(T.alanine) },
  { key: "aspirin", label: "aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O", iupac: "2-acetyloxybenzoic acid", trace: T.aspirin, parsed: parse(T.aspirin) },
  { key: "ibuprofen", label: "ibuprofen", smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1", iupac: "2-[4-(2-methylpropyl)phenyl]propanoic acid", trace: T.ibuprofen, parsed: parse(T.ibuprofen) },
];

export const MOL_BY_KEY: Record<string, Molecule> = Object.fromEntries(
  MOLECULES.map((m) => [m.key, m]),
);

// Small corpus for the BPE lab: related names sharing morphemes (an, ol, anol,
// oic acid) so merges are visible as they form.
export const BPE_CORPUS = [
  "ethanol",
  "methanol",
  "propanol",
  "butanol",
  "propanoic acid",
  "butanoic acid",
  "benzoic acid",
];
