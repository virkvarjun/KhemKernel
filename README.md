# KhemKernel

A chemistry translation model built from the ground up. It reads a molecule written as a SMILES string and writes back its IUPAC name, along with a short reasoning trace that names the parent scaffold, the functional groups, and the atom and ring counts it used to get there. The transformer, its gradients, the optimizer, and the GPU kernels underneath are all hand written. The Python package is called `picochem`; the repository and the demo carry the KhemKernel name.

The current model gets the exact molecule right **95.8% of the time** on held out data when it is allowed to propose several candidates and check them against a parser. On a single greedy pass it is at **79.5%**. The whole thing trains on one GPU in about half a day and runs inference on a laptop CPU.

**Interactive guide: [virkvarjun.github.io/KhemKernel](https://virkvarjun.github.io/KhemKernel/)** walks through the whole project from the chemistry down to the CUDA kernels, with live widgets and the real source. The older Astro writeup lives at [/KhemKernel/writeup/](https://virkvarjun.github.io/KhemKernel/writeup/). Live model inference in the guide runs locally (see below), since it needs a Python backend.

## What it does

Input a SMILES string, get back the systematic name plus the trace that produced it.

```
c1ccc(O)cc1                  ->  phenol
CC(N)C(=O)O                  ->  2-aminopropanoic acid
CC(=O)Oc1ccccc1C(=O)O        ->  2-acetyloxybenzoic acid      (aspirin)
CC(C)Cc1ccc(C(C)C(=O)O)cc1   ->  2-[4-(2-methylpropyl)phenyl]propanoic acid   (ibuprofen)
```

A full trace looks like this:

```
<parent>benzene</parent><groups>phenol</groups><atoms>7</atoms><rings>1</rings><name>phenol</name>
```

The `<name>` field is the answer. The other fields are intermediate reasoning the model is trained to produce before it commits to a name, which also gives a window into how it is thinking.

## Results

Every number below is exact structure match on 2,000 held out molecules: the generated name is parsed back to a molecule by OPSIN, canonicalized by RDKit, and compared to the input. Same evaluation seeds throughout, so the columns are directly comparable.

| Decoding | First model | Final model |
|---|---|---|
| Greedy (one pass) | 67.1% | 79.5% |
| Beam search plus verifier rerank | 81.0% | 89.6% |
| Wide beam (20) plus verifier rerank | n/a | 95.8% |
| Valid IUPAC name rate | 85.6% | 97.9% |

The jump from the first model to the final one came from two changes covered below: a byte pair tokenizer for names, and a wider model trained with a learning rate schedule. The jump from greedy to verifier rerank is a free inference time trick that works because we can check our own answers.

## Run the demo

The demo is a single page with two boxes. One takes a SMILES string and runs the model. The other takes an IUPAC name and runs OPSIN in reverse, so you can sanity check either direction.

```bash
PATH="/opt/homebrew/opt/openjdk/bin:$PATH" \
PICOCHEM_CKPT="$(pwd)/runs/device_bpe_d512_v2/ckpt_latest.npz" \
PICOCHEM_IUPAC_BPE="$(pwd)/data/iupac_bpe_v2.json" \
.venv/bin/python demo/server.py
# open http://localhost:8000
```

The server has no web framework. It is Python's standard library `http.server` plus the model's own dependencies. The SMILES box decodes a beam, checks each candidate by round tripping the name through OPSIN, and returns the one that reproduces your input molecule, with a note when it is verified. Set `PICOCHEM_BEAM=10` for a snappier response at slightly lower accuracy, or `=5` for the fastest setting.

See `demo/README.md` for the prerequisites and the regeneration steps on a fresh clone.

## The interactive guide

The full writeup is an interactive guide rather than a static page, live at [virkvarjun.github.io/KhemKernel](https://virkvarjun.github.io/KhemKernel/). It starts from the chemistry and works all the way down to the hand written CUDA kernels, with every code snippet pulled verbatim from this repository and the SMILES and byte pair tokenizers running live in the browser. It is organized as nine parts:

1. **The problem.** Why naming molecules is a translation task, and the headline numbers.
2. **Tokenization.** The SMILES regex, and the from scratch byte pair tokenizer for names, with a lab you can step through merge by merge.
3. **The architecture.** The transformer built one piece at a time (embeddings, attention, masking, LayerNorm, the encoder and decoder blocks) with interactive diagrams.
4. **Training.** The trace target, teacher forcing, the loss, hand derived backprop, the optimizer and schedule, and the NaN recovery story.
5. **The GPU and CUDA kernels.** A catalog of all twelve kernels, from the tiled matmul and its transposed backward passes down to the batched attention matmul, the reduction kernels, the atomic embedding scatter, the head transpose, and the pybind11 bindings.
6. **Inference and the verifier.** Greedy, beam search, and the free OPSIN round trip rerank.
7. **Results.** The exact match metric, valid name rate, and the failure analysis.
8. **Going deeper.** BPE internals and a tokenizer ablation, why scores are scaled by the square root of the head dimension, attention as soft retrieval, cross attention read as alignment, and head specialization.
9. **Analysis.** Whether the reasoning trace helps, the verifier read as a label free reward with the rank of truth distribution, linear probes, the real gradient check numbers, and where the GPU time actually goes.

The live "try the real model" panel needs the Python backend, so it works when you run `demo/server.py` locally and falls back to precomputed examples on the static site. The older Astro writeup is preserved at [/KhemKernel/writeup/](https://virkvarjun.github.io/KhemKernel/writeup/). Both build and deploy automatically from `main`.

## How it works

The model is a standard encoder decoder transformer. The encoder reads the SMILES tokens, the decoder writes the trace one token at a time and cross attends to the encoder. The final model is 512 wide, 8 heads, 3 encoder layers and 3 decoder layers, with a 64 token decoder context. Learned positional embeddings, tied decoder input and output embeddings.

The interesting part is everything underneath that, which is written by hand rather than pulled from a framework.

### The from scratch stack

There are two complete implementations of the same model that share weights and a tokenizer:

1. A pure NumPy version. Every forward pass has a matching backward pass derived by hand and checked against finite differences in the test suite, where every op agrees with the numerical gradient to better than two parts in a billion. This is the reference. It is slow but it is correct, and it is what the CUDA version is validated against.
2. A CUDA version. Hand written kernels for matmul and its two transposed backward passes, batched matmul for attention, softmax, layer norm, GeLU, cross entropy, the atomic embedding scatter, Adam, bias, the residual add, and the head split and merge transpose, bound to Python through pybind11 with a resident `DeviceTensor` type. The training loop keeps the transformer stack resident on the device and updates it in place, so a training step does not shuttle the weights back and forth across the bus on every iteration. The two implementations agree to about 1e-3, the rounding you expect from float32 against float64.

The host keeps the embedding tables and does the gather and scatter on the CPU, while the device holds the stack. That split is deliberate. It also turns out to be the bottleneck: during training the GPU sits around 35 to 40% utilization because the host side embedding work and the transfers are the limiting factor, not the matmuls. For a model this size that is fine, and it means the choice of GPU barely matters.

### The trace target

The training target is not the bare IUPAC name. It is the trace shown above, built for every molecule by `picochem/traces.py` using RDKit: the parent ring system or longest chain, the functional groups present, the heavy atom count, the ring count, and finally the name. Training the model to lay out that scaffold before naming gives two things. It makes the output easier to parse and check, and it gives a place to read off what the model believes about a molecule before it names it, which the interpretability experiments use.

### Tokenizing names is the hard part

SMILES tokenizes cleanly. The Schwaller regex handles multi character atoms like `[C@@H]`, `Cl`, and `Br`, and the whole alphabet is 341 tokens.

Names are harder. The first model split names on word boundaries, so a fragment like `acetyloxybenzoic` became a single token, and any fragment seen fewer than five times in the corpus collapsed to `<unk>`. The model literally could not spell rare names. That capped the valid name rate at about 86% and showed up as broken output on anything off the beaten path.

The fix was a byte pair tokenizer written for this purpose in `picochem/bpe.py`. It keeps the trace tags and the group separator as atomic tokens and learns merges over everything else starting from characters. The result has 4,000 tokens, never emits `<unk>` on real names, reconstructs the original string exactly, and produces shorter sequences than the word tokenizer did. Valid name rate went from 86% to 98%.

### Training

`scripts/train_device.py` is the GPU trainer. It runs the resident stack with a warmup then cosine learning rate schedule and writes a checkpoint every thousand steps, keeping numbered snapshots so a run is always recoverable.

It earned the recovery logic the hard way. The first scaled run trained beautifully down to a loss of 0.37, then a gradient spike near peak learning rate drove the loss to NaN at step 19,000 and, because the trainer was overwriting a single checkpoint file, the NaN weights erased the good ones. The device trainer has no gradient clipping, and a clip would need a reduction kernel that does not exist yet. The cheaper fix that shipped: skip the optimizer step whenever the loss is not finite, so one bad batch can no longer poison the weights, only checkpoint finite states, keep numbered snapshots, and drop the peak learning rate a little. The next run went the full 120,000 steps with zero NaN events and zero skipped steps.

### Evaluation, and a verifier you get for free

The eval metric is exact structure match, and the key tool is OPSIN, an open source IUPAC name parser. Because OPSIN turns a name back into a molecule, and the input to the model is a molecule, the model can check its own answers without any labels at inference time. Generate a name, parse it back, canonicalize both sides with RDKit, and ask whether they are the same molecule.

That single fact drives the headline result. Instead of trusting the model's top guess, the decoder produces a beam of candidates and keeps the one that round trips back to the input. With a beam of 5 that lifts structure match from 79.5% to 89.6%. With a beam of 20 it reaches 95.8%. The model usually knows the right answer; it just does not always rank it first, and the verifier lets us pick it out.

### What it still gets wrong

Three small diagnostic scripts under `experiments/` pin this down rather than guess at it.

`stereo_breakdown.py` was the surprise. Stereochemistry looked like the obvious culprit, since 18% of targets carry stereocenters. It is not. When the model gets a molecule's skeleton right it gets the stereochemistry right too, 887 times out of 896. Pure stereo mistakes account for under one point of the gap.

The real gap is constitutional. Right atoms and right groups, wrong arrangement: a substituent on the wrong ring carbon, a wrong locant, a wrong ring fusion. About half of the remaining errors share the exact molecular formula of the target, which is the signature of positional isomers.

`beam_ceiling.py` measured the headroom. A correct answer sits in the model's top 20 candidates for 95.8% of molecules, and the curve is still climbing at 20. So the model knows far more than a greedy pass reveals. That is why the verifier rerank works so well, and it is the reason the project stops here: the deployed accuracy is already at that ceiling.

## Reproduce it

On a fresh clone the data, vocab, and checkpoints are not in the repository (they are large and gitignored), but they regenerate deterministically.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python scripts/download_data.py     # streams 1M filtered PubChem pairs from HuggingFace
python scripts/build_vocab.py       # SMILES vocab (341 tokens)
python scripts/generate_traces.py   # RDKit reasoning traces -> data/traces.parquet
python scripts/build_bpe.py --vocab_size 4000 --train_lines 80000   # IUPAC byte pair tokenizer
```

Training needs a CUDA GPU and the compiled extension:

```bash
bash scripts/build_cuda.sh          # builds picochem_cuda.so
bash scripts/run_retrain.sh         # full pipeline: data, tokenizers, smoke test, train
```

`run_retrain.sh` takes the model size, step count, and learning rate as environment variables. The default reproduces the final model: 512 wide, 8 heads, 3 plus 3 layers, byte pair target, cosine schedule, 120,000 steps.

One note on hardware. The kernels build against the GPU's architecture and fall back to the newest virtual architecture the toolkit knows when the GPU is newer than the toolkit, emitting PTX that the driver compiles at load time. That is what let the final model train on a Blackwell card (compute capability 12.0) with a CUDA 12.4 toolkit that does not know that architecture. If you hit an architecture error, put CUDA on your `PATH` and the build will detect and handle it.

Evaluate a checkpoint, with the verifier rerank:

```bash
PATH="/opt/homebrew/opt/openjdk/bin:$PATH" python scripts/evaluate.py \
  --checkpoint runs/device_bpe_d512_v2/ckpt_latest.npz \
  --iupac_bpe data/iupac_bpe_v2.json \
  --n_samples 2000 --rerank --beam_width 20
```

OPSIN needs a Java runtime. Install `py2opsin` and a JDK, then make sure `java` is on the path. Without Java you still get the trace validity rate, just not the structure match.

## Repository layout

```
picochem/            the model: ops, attention, encoder, decoder, model, optimizer,
                     scheduler, checkpointing, data, bpe, device_layers, evaluate
picochem/kernels/    the CUDA extension and its source
scripts/             download_data, build_vocab, build_bpe, generate_traces,
                     train (CPU), train_device (GPU), evaluate, run_retrain, build_cuda
experiments/         failure_taxonomy, stereo_breakdown, beam_ceiling
demo/                local inference server (server.py) that serves the guide + the /api endpoints
demo/web/            the interactive guide (Vite + React + TypeScript, Computer Modern)
picochem-site/       the older Astro writeup, served at /KhemKernel/writeup/
tests/               gradient checks against finite differences
```

## Status

Done. The model translates SMILES to IUPAC at 95.8% exact match with verifier reranking, the demo serves it locally, and the failure analysis says the remaining gap is positional chemistry that the model mostly already solves and ranks just below first. The interactive guide is live at [virkvarjun.github.io/KhemKernel](https://virkvarjun.github.io/KhemKernel/) (built from `demo/web/`), with the Astro writeup at `/KhemKernel/writeup/` (from `picochem-site/`); both deploy automatically on every push.
