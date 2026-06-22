// Schematic attention weights. NOT real model output: these are deterministic
// patterns that respect SMILES structure (ring-closure pairing, branch
// matching, locality) so each "head" illustrates a relationship a real head
// could learn. Always label uses of this as schematic.

import { tokenizeSmiles, type SmilesToken } from "./smiles";

export interface HeadSpec {
  name: string;
  score: (i: number, j: number, ctx: Ctx) => number;
}

interface Ctx {
  toks: SmilesToken[];
  ring: (number | null)[]; // ring-closure partner index
  branch: (number | null)[]; // matching paren index
}

function ringPartners(toks: SmilesToken[]): (number | null)[] {
  const out: (number | null)[] = toks.map(() => null);
  const open: Record<string, number> = {};
  toks.forEach((t, i) => {
    if (t.type === "ring") {
      const k = t.text;
      if (open[k] === undefined) open[k] = i;
      else {
        out[i] = open[k];
        out[open[k]] = i;
        delete open[k];
      }
    }
  });
  return out;
}

function branchPartners(toks: SmilesToken[]): (number | null)[] {
  const out: (number | null)[] = toks.map(() => null);
  const stack: number[] = [];
  toks.forEach((t, i) => {
    if (t.text === "(") stack.push(i);
    else if (t.text === ")") {
      const o = stack.pop();
      if (o !== undefined) {
        out[i] = o;
        out[o] = i;
      }
    }
  });
  return out;
}

const HEADS: HeadSpec[] = [
  {
    name: "local / adjacent",
    score: (i, j) => -Math.abs(i - j) * 0.9 + (i === j ? 0.6 : 0),
  },
  {
    name: "previous token",
    score: (i, j) => (j === i - 1 ? 2.0 : j === i ? 0.8 : -Math.abs(i - j) * 1.1),
  },
  {
    name: "ring closures",
    score: (i, j, c) => {
      if (c.ring[i] != null && j === c.ring[i]) return 2.6;
      if (c.toks[i].type === "aromatic" && c.toks[j].type === "aromatic")
        return 0.7 - Math.abs(i - j) * 0.12;
      return -Math.abs(i - j) * 0.8;
    },
  },
  {
    name: "branch matching",
    score: (i, j, c) => {
      if (c.branch[i] != null && j === c.branch[i]) return 2.3;
      return -Math.abs(i - j) * 0.7 + (i === j ? 0.4 : 0);
    },
  },
  {
    name: "heteroatom neighbor",
    score: (i, j, c) => {
      const isHet = /^(O|N|S)$/.test(c.toks[i].text);
      if (isHet && Math.abs(i - j) === 1 && c.toks[j].type !== "branch")
        return 2.0;
      return -Math.abs(i - j) * 0.9;
    },
  },
  {
    name: "attend to start",
    score: (i, j) => (j === 0 ? 2.0 : -Math.abs(i - j) * 0.5),
  },
  {
    name: "look back",
    score: (i, j) => (j <= i ? -(i - j) * 0.25 : -4),
  },
  {
    name: "diffuse",
    score: (i, j) => -Math.abs(i - j) * 0.18,
  },
];

export const HEAD_NAMES = HEADS.map((h) => h.name);

/** Row-softmaxed attention matrix for one head over a SMILES string. */
export function attentionMatrix(smiles: string, head: number): number[][] {
  const toks = tokenizeSmiles(smiles);
  const ctx: Ctx = {
    toks,
    ring: ringPartners(toks),
    branch: branchPartners(toks),
  };
  const spec = HEADS[head % HEADS.length];
  const n = toks.length;
  const rows: number[][] = [];
  for (let i = 0; i < n; i++) {
    const raw = new Array(n);
    let mx = -Infinity;
    for (let j = 0; j < n; j++) {
      raw[j] = spec.score(i, j, ctx);
      if (raw[j] > mx) mx = raw[j];
    }
    let sum = 0;
    for (let j = 0; j < n; j++) {
      raw[j] = Math.exp((raw[j] - mx) / 0.7);
      sum += raw[j];
    }
    for (let j = 0; j < n; j++) raw[j] /= sum;
    rows.push(raw);
  }
  return rows;
}
