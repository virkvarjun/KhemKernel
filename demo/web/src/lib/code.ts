// Helpers to pull *real* source out of the repo files (imported as ?raw strings)
// so every code snippet in the guide is the actual code, not a paraphrase.

/** Remove common leading indentation from a block. */
export function dedent(text: string): string {
  const lines = text.replace(/\t/g, "    ").split("\n");
  let min = Infinity;
  for (const ln of lines) {
    if (!ln.trim()) continue;
    const m = ln.match(/^ */);
    if (m) min = Math.min(min, m[0].length);
  }
  if (!isFinite(min) || min === 0) return text.replace(/\s+$/, "");
  return lines
    .map((l) => l.slice(min))
    .join("\n")
    .replace(/\s+$/, "");
}

/** 1-indexed inclusive line range. */
export function lines(raw: string, start: number, end: number): string {
  return dedent(raw.split("\n").slice(start - 1, end).join("\n"));
}

/** Extract a python def (with any decorators directly above it). */
export function pyDef(raw: string, name: string): string {
  const src = raw.split("\n");
  let i = src.findIndex((l) => new RegExp(`^(\\s*)def ${name}\\b`).test(l));
  if (i < 0) return `# def ${name} not found`;
  const base = src[i].match(/^ */)?.[0].length ?? 0;
  // grab decorators immediately above
  let s = i;
  while (s - 1 >= 0 && src[s - 1].trim().startsWith("@")) s--;
  // grab body until indentation returns to <= base on a non-blank line
  let e = i + 1;
  for (; e < src.length; e++) {
    const ln = src[e];
    if (!ln.trim()) continue;
    const ind = ln.match(/^ */)?.[0].length ?? 0;
    if (ind <= base) break;
  }
  return dedent(src.slice(s, e).join("\n"));
}

/** Substring between the first occurrence of `from` and the next `to` (inclusive of both lines). */
export function between(raw: string, from: string, to: string): string {
  const src = raw.split("\n");
  const a = src.findIndex((l) => l.includes(from));
  if (a < 0) return `// "${from}" not found`;
  let b = a + 1;
  for (; b < src.length; b++) if (src[b].includes(to)) break;
  return dedent(src.slice(a, Math.min(b + 1, src.length)).join("\n"));
}

/** A line containing the given marker, plus optional trailing lines. */
export function lineWith(raw: string, marker: string, after = 0): string {
  const src = raw.split("\n");
  const a = src.findIndex((l) => l.includes(marker));
  if (a < 0) return `// "${marker}" not found`;
  return dedent(src.slice(a, a + 1 + after).join("\n"));
}
