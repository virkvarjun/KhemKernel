import { Layout } from "./components/Layout";
import { Landing } from "./sections/Landing";
import { PartI } from "./sections/PartI";
import { PartII } from "./sections/PartII";
import { PartIII } from "./sections/PartIII";
import { PartIV } from "./sections/PartIV";
import { PartV } from "./sections/PartV";
import { PartVI } from "./sections/PartVI";
import { PartVII } from "./sections/PartVII";
import { PartVIII } from "./sections/PartVIII";
import { PartIX } from "./sections/PartIX";
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
      <PartVIII />
      <PartIX />
      <Appendix />

      <footer className="foot">
        <p>
          KhemKernel is a from-scratch SMILES to IUPAC translator. This guide is a
          teaching artifact built on top of the project's live demo. The model, its
          gradients, and the GPU kernels are all hand written; the code shown here
          is pulled verbatim from the repository.
        </p>
        <p style={{ marginTop: "0.6rem" }}>
          Want to go further? The full source, the NumPy reference, the CUDA
          kernels, and the training scripts are all on GitHub:{" "}
          <a
            href="https://github.com/virkvarjun/KhemKernel"
            target="_blank"
            rel="noopener noreferrer"
            style={{ fontWeight: 700 }}
          >
            github.com/virkvarjun/KhemKernel
          </a>
          .
        </p>
      </footer>
    </Layout>
  );
}
