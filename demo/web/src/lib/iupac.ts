// IUPAC / trace tokenizer, ported from picochem/data.py IUPAC_PATTERN. Trace
// XML tags and the ";" separator are single tokens; words, numbers, and
// punctuation split the rest. Used by the decoder-generation and special-token
// widgets.

export const IUPAC_PATTERN =
  /<\/?(?:parent|groups|atoms|rings|name)>|\d+|[a-zA-Z_]+|[;()[\],\-'.]/g;

export const TRACE_TAGS = new Set([
  "<parent>", "</parent>", "<groups>", "</groups>",
  "<atoms>", "</atoms>", "<rings>", "</rings>",
  "<name>", "</name>",
]);

export type TraceTokKind = "tag" | "sep" | "text";

export function tokenizeIupac(s: string): string[] {
  return s.match(IUPAC_PATTERN) ?? [];
}

export function traceTokKind(tok: string): TraceTokKind {
  if (TRACE_TAGS.has(tok)) return "tag";
  if (tok === ";") return "sep";
  return "text";
}
