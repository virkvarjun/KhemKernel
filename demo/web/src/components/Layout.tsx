import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Toc } from "./Toc";
import { ALL_SECTION_IDS } from "../toc";
import { useReadingProgress, useScrollSpy, useTheme } from "../lib/hooks";

export function Layout({ children }: { children: ReactNode }) {
  const [theme, toggleTheme] = useTheme();
  const progress = useReadingProgress();
  const active = useScrollSpy(ALL_SECTION_IDS);
  const [tocOpen, setTocOpen] = useState(false);

  // Reveal-on-scroll: the visual blocks (figures, widgets, code, tables,
  // headings) fade and slide up as they enter the viewport. Only blocks that
  // start below the fold get the hidden state, so there is no flash on load.
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (!("IntersectionObserver" in window)) return;
    const sel =
      ".content .breakout, .content .section > h2, .content figure, .content table";
    const els = Array.from(document.querySelectorAll<HTMLElement>(sel));
    if (els.length === 0) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add("reveal-in");
            io.unobserve(e.target);
          }
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.06 },
    );
    const fold = window.innerHeight * 0.92;
    for (const el of els) {
      if (el.getBoundingClientRect().top > fold) {
        el.classList.add("reveal");
        io.observe(el);
      }
    }
    return () => io.disconnect();
  }, []);

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
          Khem<span className="k">Kernel</span>
        </a>
        <span className="spacer" />
        <a className="iconbtn" href="https://github.com/virkvarjun/KhemKernel">
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
