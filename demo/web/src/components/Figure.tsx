import type { ReactNode } from "react";

/** A full-width figure with an optional caption and a "schematic" marker for
 *  anything that is illustrative rather than measured model output. */
export function Figure({
  children,
  caption,
  schematic = false,
}: {
  children: ReactNode;
  caption?: ReactNode;
  schematic?: boolean;
}) {
  return (
    <figure className="breakout" style={{ margin: "1.5rem auto" }}>
      <div className="card" style={{ padding: "1rem" }}>
        {children}
      </div>
      {(caption || schematic) && (
        <figcaption
          style={{
            marginTop: "0.5rem",
            fontSize: "0.88rem",
            color: "var(--ink-faint)",
            display: "flex",
            gap: "0.5rem",
            alignItems: "baseline",
            flexWrap: "wrap",
          }}
        >
          {schematic && <span className="schematic">schematic</span>}
          <span>{caption}</span>
        </figcaption>
      )}
    </figure>
  );
}
