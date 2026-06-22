import { PartRule, Section, Aside } from "../components/ui";
import { CodeBlock } from "../components/CodeBlock";
import { SmilesTokenizer } from "../widgets/SmilesTokenizer";
import { BpeLab } from "../widgets/BpeLab";
import { SpecialTokens } from "../widgets/SpecialTokens";
import { RAW } from "../data/raw";
import { lines, pyDef } from "../lib/code";

export function PartII() {
  return (
    <>
      <PartRule part="Part II" title="Tokenization and Encoding the Chemistry" />

      <Section id="p2-1" title="SMILES tokenizer">
        <p>
          Before the model sees anything, the text has to become a sequence of
          tokens (discrete pieces it has an id for). SMILES describes a molecule
          as a short string: capital letters are atoms (
          <span className="tok tok-atom">C</span>,{" "}
          <span className="tok tok-atom">O</span>,{" "}
          <span className="tok tok-atom">N</span>), lowercase letters are
          aromatic atoms (<span className="tok tok-aromatic">c</span> in a
          benzene ring), parentheses are branches, and matched digits mark where
          a ring opens and closes.
        </p>
        <p>
          A regular expression (regex) is a pattern that describes which
          substrings to pull out, tried left to right. The tokenizer is one regex
          taken from Schwaller et al. The order of the alternatives matters:{" "}
          <code>\[[^\]]+]</code> grabs a whole bracketed atom like{" "}
          <span className="tok tok-bracket">[C@@H]</span> before anything inside
          it can match, and <code>Br?</code> and <code>Cl?</code> keep the
          two-letter atoms <span className="tok tok-atom">Br</span> and{" "}
          <span className="tok tok-atom">Cl</span> intact (and, as a side effect,
          let a bare <span className="tok tok-atom">C</span> match through{" "}
          <code>Cl?</code>). The whole SMILES alphabet is 341 tokens.
        </p>
        <CodeBlock path="picochem/data.py" lang="python" code={lines(RAW.data, 6, 21)} />
        <p>Try it. Edit the string or pick a molecule; tokens are colored by type.</p>
        <SmilesTokenizer />
      </Section>

      <Section id="p2-2" title="Why IUPAC needs BPE">
        <p>
          IUPAC names are harder. The first version of the model split names on
          word boundaries, so a fragment like <code>acetyloxybenzoic</code>{" "}
          became one token, and anything seen fewer than five times collapsed to{" "}
          <code>&lt;unk&gt;</code>. The model literally could not spell rare
          names. That capped the valid name rate around 86% and showed up as
          broken output on anything off the beaten path.
        </p>
        <p>
          The fix is byte pair encoding (BPE), written from scratch for this. BPE
          starts from individual characters, so every name is representable, then
          repeatedly finds the most frequent adjacent pair of tokens and merges
          it into a new token. The ordered list of merges is the tokenizer. After
          enough merges, common fragments like <code>anol</code> or{" "}
          <code>oic acid</code> become single tokens while rare fragments stay as
          characters. The result has 4,000 tokens, never emits{" "}
          <code>&lt;unk&gt;</code> on real names, and reconstructs the input
          exactly. Valid name rate went from 86% to 98%.
        </p>
        <p>
          Watch it happen. In train mode, each step merges the top pair and adds
          it to the merge list. In encode mode, the learned merges collapse a
          word from characters into morphemes (try <code>hexanol</code>, which is
          not in the corpus, to see the character fallback).
        </p>
        <BpeLab />
        <p style={{ marginTop: "1rem" }}>
          This is the actual training procedure (the most frequent pair, with a
          deterministic tie-break) and the actual encoder (apply merges in the
          order they were learned).
        </p>
        <CodeBlock path="picochem/bpe.py · train" lang="python" code={pyDef(RAW.bpe, "train")} />
        <CodeBlock path="picochem/bpe.py · _bpe" lang="python" code={pyDef(RAW.bpe, "_bpe")} />
      </Section>

      <Section id="p2-3" title="Special tokens">
        <p>
          The trace has structure tags (<code>&lt;parent&gt;</code>,{" "}
          <code>&lt;name&gt;</code>, and so on) and a <code>;</code> separator
          between groups. Those must stay whole, so they are peeled off before
          BPE runs, matched longest first so <code>&lt;/parent&gt;</code> wins
          over a bare <code>&lt;</code>. On top of those sit four control tokens:{" "}
          <code>&lt;start&gt;</code>, <code>&lt;end&gt;</code>,{" "}
          <code>&lt;pad&gt;</code>, and <code>&lt;unk&gt;</code>.
        </p>
        <SpecialTokens />
        <Aside label="why losslessness matters">
          Spaces and punctuation are ordinary characters, so decoding is just
          concatenation and a generated name comes back byte for byte. That is
          what lets the verifier (Part VI) parse the model's own output and check
          it against the input molecule.
        </Aside>
        <CodeBlock path="picochem/bpe.py · tokenize / decode" lang="python" code={pyDef(RAW.bpe, "tokenize") + "\n\n" + pyDef(RAW.bpe, "decode")} />
      </Section>
    </>
  );
}
