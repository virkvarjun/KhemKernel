import { useCallback, useState, type ReactNode } from "react";
import { Toc } from "./Toc";
import { ALL_SECTION_IDS } from "../toc";
import { useReadingProgress, useScrollSpy, useTheme } from "../lib/hooks";

export function Layout({ children }: { children: ReactNode }) {
  const [theme, toggleTheme] = useTheme();
  const progress = useReadingProgress();
  const active = useScrollSpy(ALL_SECTION_IDS);
  const [tocOpen, setTocOpen] = useState(false);

  const navigate = useCallback((id: string) => {
    setTocOpen(false);
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      // move focus for keyboard users without scrolling again
      el.setAttribute("tabindex", "-1");
      (el as HTMLElement).focus({ preventScroll: true });
    } else if (id === "top") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, []);

  return (
    <>
      <div
        className="progress"
        style={{ width: `${(progress * 100).toFixed(2)}%` }}
        role="progressbar"
        aria-label="Reading progress"
        aria-valuenow={Math.round(progress * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      />
      <header className="topbar">
        <button
          className="iconbtn toc-toggle"
          aria-label="Toggle table of contents"
          aria-expanded={tocOpen}
          onClick={() => setTocOpen((o) => !o)}
        >
          {tocOpen ? "✕" : "☰"} contents
        </button>
        <a className="brand" href="#top" onClick={() => navigate("top")}>
          Chem<span className="k">Kernel</span>
        </a>
        <span className="spacer" />
        <a className="iconbtn" href="https://github.com/virkvarjun/ChemKernel">
          source
        </a>
        <button
          className="iconbtn"
          onClick={toggleTheme}
          aria-label="Toggle color theme"
        >
          {theme === "light" ? "☾ dark" : "☀ light"}
        </button>
      </header>

      <div className="shell">
        <div>
          {tocOpen && (
            <div className="toc-scrim" onClick={() => setTocOpen(false)} />
          )}
          <Toc active={active} open={tocOpen} onNavigate={navigate} />
        </div>
        <main className="content" id="main">
          {children}
        </main>
      </div>
    </>
  );
}
