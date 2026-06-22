import type { ReactNode } from "react";

/** Section wrapper: an <section id> in the reading column with an h2 heading. */
export function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="section col">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

/** Part divider rule shown above the first section of each part. */
export function PartRule({ part, title }: { part: string; title: string }) {
  return (
    <div className="part-rule">
      <span className="pl">{part}</span>
      <span className="pt">{title}</span>
    </div>
  );
}

export function Aside({
  children,
  label,
  warn = false,
}: {
  children: ReactNode;
  label?: string;
  warn?: boolean;
}) {
  return (
    <div className={warn ? "aside warn" : "aside"}>
      {label && <span className="label">{label}</span>}
      {children}
    </div>
  );
}

export function Stat({ v, l }: { v: string; l: string }) {
  return (
    <div className="stat">
      <div className="v">{v}</div>
      <div className="l">{l}</div>
    </div>
  );
}

export function StatStrip({ children }: { children: ReactNode }) {
  return <div className="statstrip breakout">{children}</div>;
}

/** Inline schematic tag for use in captions/text. */
export function SchematicTag() {
  return <span className="schematic">schematic</span>;
}
