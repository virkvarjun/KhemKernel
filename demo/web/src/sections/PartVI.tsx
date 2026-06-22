import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { Math as Tex } from "../components/Math";
import { BeamVerifier } from "../widgets/BeamVerifier";
import { RAW } from "../data/raw";
import { lines, pyDef } from "../lib/code";

export function PartVI() {
  return (
    <>
      <PartRule part="Part VI" title="Inference and the Verifier" />

      <Section id="p6-1" title="Decoding">
        <p>
          At inference the decoder writes the trace one token at a time. The
          simplest way is greedy: take the most likely next token every step.
          Temperature sampling instead draws from the distribution, trading
          accuracy for variety. The one that matters here is beam search: keep the
          best few partial sequences (beams) at each step, expand each by its top
          next tokens, and prune back to the best few.
        </p>
        <p>
          Beams are ranked by length-normalized log probability so the search does
          not just prefer short names. This is the GNMT convention, dividing the
          summed log prob by the length raised to a penalty around 0.6.
        </p>
        <Tex block tex="\text{score}(y) = \frac{\sum_t \log p(y_t)}{\,|y|^{\,\alpha}}, \qquad \alpha \approx 0.6" />
        <CodeBlock path="picochem/model.py · greedy_decode" lang="python" code={pyDef(RAW.model, "greedy_decode")} />
        <CodeBlock path="picochem/model.py · beam_decode (scoring + prune)" lang="python" code={lines(RAW.model, 302, 331)} />
      </Section>

      <Section id="p6-2" title="The free verifier">
        <p>
          Here is the trick that drives the headline number. The input is a known
          molecule, and OPSIN (an open-source IUPAC name parser) can turn any
          generated name back into a molecule. So the model can check its own
          answer with no labels at inference time: parse each candidate name back
          to a structure, canonicalize both sides with RDKit, and ask whether they
          are the same molecule.
        </p>
        <p>
          Instead of trusting the top beam, decode a beam of candidates,
          round-trip every one, and keep the one that reproduces the input. The
          model usually knows the right answer; it just does not always rank it
          first, and the verifier lets us pick it out. With a beam of 5 this lifts
          structure match from 79.5% to 89.6%; with a beam of 20 it reaches 95.8%.
          Zero extra training.
        </p>
        <BeamVerifier />
        <CodeBlock path="picochem/evaluate.py · the verifier rerank" lang="python" code={lines(RAW.evaluate, 200, 219)} />
        <Aside label="honesty: what is and is not new" warn>
          The chemistry task is not novel. STOUT is established prior art and is a
          larger model. Round-tripping names through OPSIN is also used by STOUT
          V2, so self-verification is not new either. The contribution here is
          narrow and specific: using that round-trip as a beam reranker at
          inference, which is what turns a 79.5% greedy model into a 95.8% one for
          free.
        </Aside>
      </Section>
    </>
  );
}
