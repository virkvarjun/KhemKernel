import { useMemo, useState } from "react";
import { highlight, type Lang } from "../lib/highlight";

export function CodeBlock({
  code,
  lang = "python",
  path,
}: {
  code: string;
  lang?: Lang;
  path?: string;
}) {
  const html = useMemo(() => highlight(code, lang), [code, lang]);
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard blocked; ignore */
    }
  };

  return (
    <div className="codeblock breakout">
      <div className="cap">
        <span className="path">{path ?? lang}</span>
        <span className="spacer" />
        <button className="copy" onClick={copy} aria-label="Copy code">
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre tabIndex={0}>
        <code dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
    </div>
  );
}
