# KhemKernel animations (3Blue1Brown style)

Short (5 to 10 second) explainer clips for the project, rendered with
[Manim](https://www.manim.community/). Black background, Computer Modern
(CMU Serif / CMU Typewriter), the manim default palette, math set with the real
Computer Modern font through Pango (no LaTeX needed on this machine).

Rendered clips are in `videos/`:

| file | what it shows |
|---|---|
| `01_pipeline.mp4` | SMILES → tokens → transformer → IUPAC name |
| `02_attention.mp4` | the attention equation `softmax(QKᵀ/√dₖ)V` + a softmax row |
| `03_bpe.mp4` | byte pair merges collapsing characters into a morpheme token |
| `04_verifier.mp4` | name → OPSIN → molecule → match, and the 79.5 → 95.8 jump |
| `05_gradcheck.mp4` | the finite-difference gradient check, analytic ≈ numeric |
| `06_tiled_matmul.mp4` | tiles streaming from global into shared memory, accumulate |

## Re-render

```bash
# one-time: an isolated env (keeps the project's .venv untouched)
uv venv --python 3.12 ~/.venvs/manim
uv pip install --python ~/.venvs/manim/bin/python manim

# render one scene at 1080p60
cd media/manim
~/.venvs/manim/bin/manim -qh --format mp4 -o 01_pipeline scenes.py Pipeline
```

Scene classes in `scenes.py`: `Pipeline`, `BPE`, `Attention`, `Verifier`,
`GradCheck`, `TiledMatmul`. Needs `ffmpeg` on PATH and the Computer Modern fonts
installed (they ship with most TeX distributions, or the `computer-modern`
package). The math is composed with the Computer Modern font rather than LaTeX,
so no TeX install is required.
