import { Layout } from "./components/Layout";
import { Landing } from "./sections/Landing";
import { PartI } from "./sections/PartI";
import { PartII } from "./sections/PartII";
import { PartIII } from "./sections/PartIII";
import { PartIV } from "./sections/PartIV";
import { PartV } from "./sections/PartV";
import { PartVI } from "./sections/PartVI";
import { PartVII } from "./sections/PartVII";
import { Appendix } from "./sections/Appendix";

export function App() {
  return (
    <Layout>
      <Landing />
      <PartI />
      <PartII />
      <PartIII />
      <PartIV />
      <PartV />
      <PartVI />
      <PartVII />
      <Appendix />

      <footer className="foot">
        ChemKernel is a from-scratch SMILES to IUPAC translator. This guide is a
        teaching artifact built on top of the project's live demo. The model, its
        gradients, and the GPU kernels are all hand written; the code shown here
        is pulled verbatim from the repository.
      </footer>
    </Layout>
  );
}
