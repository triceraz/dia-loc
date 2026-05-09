# DIA-LOC

**Localizing Dialect Representation in Open Norwegian-Capable LLMs.**

Where in an instruction-tuned open Norwegian-capable LLM does the Bokmål-Nynorsk
representational gap live, and does it ride on the same internal machinery as
the English-vs-Norwegian gap, or on separate dimensions?

## Status

Pre-experimental. Week 1 (data curation) in progress.

## Pre-registered hypotheses

- **H1 (sparsity).** The BM/NN representational divergence is concentrated in
  specific layers and a small subset of attention heads, not uniformly distributed.
- **H2 (entanglement).** The set of dialect-carrying heads is a strict subset of
  foreign-language-carrying heads. The model treats nynorsk as a kind of mild
  foreign language.
- **H2-alt (separation).** The two head sets are disjoint or only weakly overlap.
  The model treats BM/NN distinction as orthogonal to NB/EN.

## Method

Five probes applied to three sentence-pair contrast sets:

| Set | Contents                                          | Source                                     | Size |
|-----|---------------------------------------------------|--------------------------------------------|------|
| D1  | Paired BM ↔ NN (the dialectal contrast)           | Wikipedia seeds + Apertium nob→nno         | 200  |
| D2  | Paired NB ↔ EN (the foreign-language contrast)    | FLORES-200 nob_Latn / eng_Latn             | 200  |
| D3  | Paired BM ↔ BM near-paraphrases (control)         | Wikipedia seeds + LLM paraphrase via Hugin | 100  |

Probes:

1. Layer-wise cosine similarity between paired residual streams
2. Layer-wise CKA (kernel-robust alternative)
3. Logit-lens top-1 token agreement per layer
4. Linear probe for contrast identity (per layer, per checkpoint)
5. Attention-head ablation; identify top-K dialect heads and top-K
   foreign-language heads
6. *(Stretch)* Sparse autoencoder on the highest-divergence layer; look for
   features firing only on NN, only on EN, or both

The headline analysis is a comparison of (4)+(5) signatures across D1 vs D2:
head-set IoU and layer-signature correlation.

## Model

Primary: Qwen 2.5 1.5B Instruct.

Compute: 1× RTX 3060 Ti (8 GB), ~10 GPU-hours total wall-clock.

## Repo layout

```
dia-loc/
├── README.md
├── requirements.txt
├── paper/
│   ├── paper.md
│   └── figures/
├── src/
│   ├── 01_build_contrast_sets.py
│   ├── 02_capture_activations.py        (Week 2)
│   ├── 03_similarity.py                 (Week 2)
│   ├── 04_logit_lens.py                 (Week 2)
│   ├── 05_linear_probes.py              (Week 3)
│   ├── 06_attention_ablation.py         (Week 3)
│   ├── 07_entanglement_comparison.py    (Week 3)
│   ├── 08_sae_train.py                  (Week 4)
│   ├── 09_sae_features.py               (Week 4)
│   └── lib/
│       ├── hooks.py
│       ├── config.py
│       └── eval_set.py
├── data/                                (curated pairs as JSONL)
└── runs/                                (gitignored; activations, SAE weights)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Environment for D3 (LLM paraphrase via the Tenki MLX tunnel):

```bash
export LLM_BASE_URL=https://mlx.tenki.no/v1
export LLM_API_KEY=<your-bearer-from-tenki-env>
export LLM_MODEL=mlx-community/gemma-3-4b-it-4bit
```

## Reproducibility

Every artifact in this paper is produced on a single 3060 Ti from off-the-shelf
weights and public corpora. No fine-tuning is required to reproduce any result.
HF: `Triceraz/dia-loc-activations` (residual tensors + SAE weights, ~5 GB).
GitHub: `Triceraz/dia-loc` (code).

## Author

Andreas Grønbeck (Tenki Labs AS).
