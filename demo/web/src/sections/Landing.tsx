import { MoleculeExplorer } from "../widgets/MoleculeExplorer";
import { LiveDemo } from "../widgets/LiveDemo";
import { Stat, StatStrip } from "../components/ui";

export function Landing() {
  return (
    <section id="top">
      <div className="col">
        <h1>KhemKernel</h1>
        <p style={{ fontSize: "1.12rem", color: "var(--ink-soft)" }}>
          A from-scratch transformer that reads a molecule written as a SMILES
          string and writes its systematic IUPAC name, with a short reasoning
          trace. The model, its hand-derived gradients, the optimizer, and the
          GPU kernels underneath are all written by hand, twice: once in pure
          NumPy as the reference, once in CUDA C++ for speed.
        </p>
      </div>

      <StatStrip>
        <Stat v="79.5%" l="greedy match" />
        <Stat v="95.8%" l="verified match" />
        <Stat v="341 + 4,000" l="src + tgt vocab" />
        <Stat v="512" l="model dim" />
        <Stat v="8" l="heads" />
        <Stat v="3 + 3" l="enc + dec layers" />
        <Stat v="~60×" l="resident speedup" />
      </StatStrip>

      <div style={{ margin: "1.5rem auto 0", maxWidth: "var(--col-wide)" }}>
        <MoleculeExplorer mode="full" initial="aspirin" />
      </div>

      <div style={{ margin: "1.5rem auto 0", maxWidth: "var(--col-wide)" }}>
        <LiveDemo />
      </div>

      <div className="col" style={{ marginTop: "1.5rem" }}>
        <div className="aside">
          <span className="label">how to read this guide</span>
          It goes top to bottom, from the chemistry down to the CUDA kernels.
          Every section opens with the idea in one or two plain sentences, then
          builds down into the math and the real code. You can stop after the
          first paragraph of any section and still have learned something true.
          Widgets are interactive: select a molecule, step a tokenizer, switch an
          attention head. Anything marked <span className="schematic">schematic</span>{" "}
          illustrates the mechanism rather than showing measured model output.
          The five molecules above (ethanol, phenol, alanine, aspirin, ibuprofen)
          recur throughout so the examples stay coherent.
        </div>
      </div>
    </section>
  );
}
