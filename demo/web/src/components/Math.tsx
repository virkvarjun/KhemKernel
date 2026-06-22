import { useMemo } from "react";
import katex from "katex";

/** Render TeX with KaTeX. Never ships raw TeX: on error it shows the source
 *  in a monospace span so nothing breaks the layout. */
export function Math({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, {
        displayMode: block,
        throwOnError: false,
        output: "htmlAndMathml",
      });
    } catch {
      return null;
    }
  }, [tex, block]);

  if (html == null) {
    return (
      <code style={{ color: "var(--warn)" }}>{tex}</code>
    );
  }
  const Tag = block ? "div" : "span";
  return (
    <Tag
      className={block ? "katex-display-wrap" : "katex-inline-wrap"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
