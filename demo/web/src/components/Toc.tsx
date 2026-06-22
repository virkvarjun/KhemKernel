import { TOC } from "../toc";

export function Toc({
  active,
  open,
  onNavigate,
}: {
  active: string;
  open: boolean;
  onNavigate: (id: string) => void;
}) {
  return (
    <nav className={open ? "toc open" : "toc"} aria-label="Table of contents">
      <a
        className={"toc-link" + (active === "top" ? " active" : "")}
        href="#top"
        onClick={(e) => {
          e.preventDefault();
          onNavigate("top");
        }}
      >
        Overview
      </a>
      {TOC.map((p) => (
        <div key={p.part}>
          <div className="toc-part">
            {p.part}
            {p.title ? " · " + p.title : ""}
          </div>
          {p.items.map((it) => (
            <a
              key={it.id}
              href={"#" + it.id}
              className={"toc-link" + (active === it.id ? " active" : "")}
              aria-current={active === it.id ? "true" : undefined}
              onClick={(e) => {
                e.preventDefault();
                onNavigate(it.id);
              }}
            >
              {it.label}
            </a>
          ))}
        </div>
      ))}
    </nav>
  );
}
