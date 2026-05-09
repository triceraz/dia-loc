# DIA-LOC

**Localizing Dialect Representation in Open Norwegian-Capable LLMs.**

Where in an instruction-tuned open Norwegian-capable LLM does the
Bokmål-Nynorsk representational gap live, and does it ride on the same
internal machinery as the English-vs-Norwegian gap, or on separate
dimensions?

Paper draft: [`paper/paper.md`](paper/paper.md)
Figures: [`paper/figures/`](paper/figures/)

## TL;DR (v0.1)

We ran five mech-interp probes on **off-the-shelf Qwen 2.5 1.5B
Instruct** across three contrast sets:

| Set | Contents | Source | n |
|---|---|---|---|
| D1 | BM ↔ NN paraphrase | Norwegian Bokmål Wikipedia + Apertium nob→nno | 200 |
| D2 | NB ↔ EN translation | FLORES-200 nob_Latn / eng_Latn dev | 200 |
| D3 | BM ↔ BM paraphrase (control) | Norwegian Bokmål Wikipedia + Gemma 3 4B paraphrase | 100 |

**Headline finding** (see [`paper/figures/05_linear_probes.png`](paper/figures/05_linear_probes.png)):

| Contrast | 5-fold CV linear probe accuracy (across 28 layers) |
|---|---|
| D2 (NB↔EN) | **1.00** — perfectly linearly separable |
| D1 (BM↔NN) | **0.77 - 0.82** — small but reliable signal |
| D3 (BM↔BM control) | 0.68 - 0.70 — surface-variation noise floor |

The dialect signal is **invisible to direct similarity metrics** —
cosine similarity between paired BM/NN residuals is ~0.98 at every
layer, geometrically nearly identical. But a linear probe finds a
small, reliable, distributed-across-the-stack direction that
distinguishes them. ~10 percentage points above the noise floor at
every layer.

That reframes one paper question. "Where does the dialect signal
live?" presupposes a substantial signal to localize. At this model
size, off the shelf, the dialect direction is already there from
input through to output — the model encodes it as a stable linear
direction, not as a spike at any specific layer.

## Repo layout

```
dia-loc/
├── README.md                            (this file)
├── requirements.txt
├── paper/
│   ├── paper.md                         v0.1 draft
│   └── figures/                         03 / 04 / 05 / 06 PNGs
├── src/
│   ├── 01_build_contrast_sets.py        D1 + D2 + D3 builders
│   ├── 02_capture_activations.py        forward-hook capture (mean / last pool)
│   ├── 03_similarity.py                 layer-wise cosine + CKA
│   ├── 04_logit_lens.py                 top-1 / top-10 / JS at last token
│   ├── 05_linear_probes.py              5-fold CV LR per layer (the finding)
│   ├── 06_head_ablation.py              28×12 head ablation × probe
│   ├── 07_sae_train.py                  small SAE on a chosen layer
│   └── lib/{config,eval_set,hooks,__init__}.py
├── data/                                D1, D2, D3 JSONL pair files
└── runs/                                (gitignored) activations + probe outputs
```

## Compute

1× **RTX 3060 Ti**, ~30 minutes wall-clock for the full pipeline
including head ablation. Single-author, off-the-shelf model weights
(no fine-tuning). Python 3.12 + torch 2.6.0+cu124.

## Reproduce

```bash
git clone https://github.com/triceraz/dia-loc
cd dia-loc

py -3.12 -m venv .venv312
.venv312\Scripts\activate           # Windows
pip install -r requirements.txt

# D3 needs an LLM for paraphrase generation. Either set up the Tenki
# MLX tunnel and put the bearer in a local .env, or replace
# `llm_paraphrase_bm` in 01_build_contrast_sets.py with another OpenAI-
# compatible endpoint (Ollama, OpenAI, Anthropic). The .env approach:
#   LLM_BASE_URL=https://mlx.tenki.no/v1
#   LLM_API_KEY=<bearer>
#   LLM_MODEL=mlx-community/gemma-3-4b-it-4bit

python src/01_build_contrast_sets.py
python src/02_capture_activations.py
python src/03_similarity.py
python src/04_logit_lens.py
python src/05_linear_probes.py
python src/06_head_ablation.py --limit 30 --contrast d1
python src/07_sae_train.py --layer 14
```

## Status

- v0.1 ships §4.1-4.3 + §4.5 results + §4.4 head ablation (in
  flight). See [`paper/paper.md`](paper/paper.md) for the writeup.
- v0.2 (planned): pre-/post-BNCR comparison once a BNCR-trained
  Qwen 2.5 checkpoint is accessible.
- v0.3 (stretch): per-token activation capture, proper SAE training
  (~25k samples), Gemma 3 4B replication.

## Author

Andreas Grønbeck (Tenki Labs AS).
