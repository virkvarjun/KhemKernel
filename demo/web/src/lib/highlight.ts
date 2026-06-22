import Prism from "prismjs";
import "prismjs/components/prism-python";
import "prismjs/components/prism-c";
import "prismjs/components/prism-cpp";
import "prismjs/components/prism-bash";
import "prismjs/components/prism-json";

export type Lang = "python" | "cpp" | "cuda" | "bash" | "json" | "text";

const ALIAS: Record<string, string> = { cuda: "cpp" };

/** Syntax-highlight to HTML at mount time. Falls back to escaped text. */
export function highlight(code: string, lang: Lang): string {
  const key = ALIAS[lang] ?? lang;
  const grammar = Prism.languages[key];
  if (!grammar) return escapeHtml(code);
  try {
    return Prism.highlight(code, grammar, key);
  } catch {
    return escapeHtml(code);
  }
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
