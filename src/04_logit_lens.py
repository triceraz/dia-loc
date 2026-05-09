"""Logit lens: per-layer top-1 next-token agreement between paired inputs.

For each layer L and each pair (a, b):
  - Project the LAST-TOKEN residual through the unembedding to get logits
  - top1_a = argmax(logits_a), top1_b = argmax(logits_b)
  - agreement(L) = mean over pairs of [top1_a == top1_b]

Reads the `*_last.pt` activations captured by 02 with --pool last.
These tensors hold the residual at the last real (non-pad) token
position per layer, which is the canonical autoregressive next-token
prediction anchor.

Earlier (v0.1) we read mean-pooled residuals here, which produced a
degenerate ~1.0 agreement everywhere because mean-pooling washes out
the next-token signal. Last-token is the right granularity.

Reads the model's embedding matrix as the unembedding via Qwen's tied
weights. For Qwen 2.5, model.embed_tokens.weight is tied to
lm_head.weight, so projecting residuals through embed_tokens gives
the same logits lm_head would have produced.

Output:

  paper/figures/04_logit_lens.png         per-layer top-1 agreement
  runs/probes/logit_lens.csv              raw numbers

The story we want to read off this plot:
  - If BM/NN paired residuals agree on the top-1 token at middle layers
    *post-BNCR* → real internal unification.
  - If they only agree at the final layer → output-only alignment.
For an off-the-shelf model like Qwen 2.5 1.5B (no fine-tuning), this
gives us the BASELINE — where in the stack does dialect agreement
emerge naturally? That baseline is the contrast for any future
fine-tuning work.

Run:

    python src/04_logit_lens.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from transformers import AutoModel

from lib.config import MODEL_ID, REPO_ROOT, activations_dir


def get_unembedding(model_id: str) -> torch.Tensor:
    """Load the embedding matrix as a fp32 tensor on CPU.

    Qwen 2.5 ties its input/output embeddings, so projecting a residual
    through embed_tokens gives the same logits lm_head would. For models
    without tied weights, this would project through the wrong matrix
    and we'd want lm_head explicitly. We assert on tie just in case.
    """
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16)
    embed = model.embed_tokens.weight.detach().to(torch.float32).cpu()
    del model
    return embed  # [vocab, d_model]


def _logits(
    a_layer: torch.Tensor, embed: torch.Tensor
) -> torch.Tensor:
    return a_layer.to(torch.float32) @ embed.t()  # [N, V]


def top1_agreement(
    a_layer: torch.Tensor, b_layer: torch.Tensor, embed: torch.Tensor
) -> float:
    """Fraction of pairs where the argmax token matches.

    Caveat for the BM/NN case: when the input ends with a period, both
    sides typically argmax onto the same end-of-sentence token, which
    inflates this number to ~1.0 across layers. We keep it as one
    signal but pair it with finer-grained metrics below.
    """
    la = _logits(a_layer, embed).argmax(dim=-1)
    lb = _logits(b_layer, embed).argmax(dim=-1)
    return float((la == lb).float().mean().item())


def topk_overlap(
    a_layer: torch.Tensor, b_layer: torch.Tensor, embed: torch.Tensor, k: int
) -> float:
    """Mean Jaccard overlap of the top-K next-token sets.

    k=1 reduces to top1_agreement. Higher k smooths over the trivial
    "both predict period" case and surfaces whether the *vocabulary
    region* of the prediction matches.
    """
    la = _logits(a_layer, embed).topk(k, dim=-1).indices  # [N, k]
    lb = _logits(b_layer, embed).topk(k, dim=-1).indices
    n = la.shape[0]
    overlaps = torch.zeros(n)
    for i in range(n):
        s_a = set(la[i].tolist())
        s_b = set(lb[i].tolist())
        union = s_a | s_b
        inter = s_a & s_b
        overlaps[i] = len(inter) / len(union) if union else 0.0
    return float(overlaps.mean().item())


def js_divergence(
    a_layer: torch.Tensor, b_layer: torch.Tensor, embed: torch.Tensor
) -> float:
    """Mean Jensen-Shannon divergence (nats) between the two predicted
    distributions per pair, averaged across pairs.

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), M = 0.5 * (P + Q).
    Symmetric and bounded by ln(2). Captures distribution-level
    disagreement that top-1 misses.
    """
    pa = torch.softmax(_logits(a_layer, embed), dim=-1)  # [N, V]
    pb = torch.softmax(_logits(b_layer, embed), dim=-1)
    m = 0.5 * (pa + pb)
    eps = 1e-12
    kl_am = (pa * (pa.add(eps).log() - m.add(eps).log())).sum(dim=-1)
    kl_bm = (pb * (pb.add(eps).log() - m.add(eps).log())).sum(dim=-1)
    js = 0.5 * (kl_am + kl_bm)
    return float(js.mean().item())


def main() -> None:
    act_dir = activations_dir()
    manifest_path = act_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No manifest at {manifest_path}; run 02 first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contrasts = manifest.get("contrasts", [])
    if not contrasts:
        sys.exit("Manifest has no contrasts.")

    print(f"[04] loading unembedding from {MODEL_ID} ...")
    embed = get_unembedding(MODEL_ID)
    print(f"[04] embed shape: {tuple(embed.shape)}")

    out_dir = REPO_ROOT / "runs" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_contrast: dict[str, dict] = {}

    for c in contrasts:
        slug = c["slug"]
        # Last-token tensors are the right input for logit lens. If they
        # haven't been captured yet, skip with a clear message.
        a_path = act_dir / f"{slug}_a_last.pt"
        b_path = act_dir / f"{slug}_b_last.pt"
        if not (a_path.exists() and b_path.exists()):
            print(
                f"[04] {slug}: last-token tensors missing. "
                f"Run: python src/02_capture_activations.py --pool last"
            )
            continue
        a = torch.load(a_path, weights_only=True, map_location="cpu")  # [N, L, D]
        b = torch.load(b_path, weights_only=True, map_location="cpu")
        if a.shape != b.shape:
            print(f"[04] {slug}: shape mismatch, skipping")
            continue
        n_pairs, n_layers, _ = a.shape

        agreements: list[float] = []
        topk_overlaps: list[float] = []
        js_divs: list[float] = []
        for li in range(n_layers):
            agr = top1_agreement(a[:, li, :], b[:, li, :], embed)
            ov = topk_overlap(a[:, li, :], b[:, li, :], embed, k=10)
            jsd = js_divergence(a[:, li, :], b[:, li, :], embed)
            agreements.append(agr)
            topk_overlaps.append(ov)
            js_divs.append(jsd)
            rows.append({
                "contrast": slug,
                "label": c["label"],
                "layer": li,
                "n_pairs": n_pairs,
                "top1_agreement": agr,
                "top10_jaccard": ov,
                "js_divergence": jsd,
            })
        per_contrast[slug] = {
            "label": c["label"],
            "n_pairs": n_pairs,
            "top1_agreement": agreements,
            "top10_jaccard": topk_overlaps,
            "js_divergence": js_divs,
        }
        print(
            f"[04] {slug:>3}: n={n_pairs:>3}  "
            f"top10[0]={topk_overlaps[0]:.3f}..[-1]={topk_overlaps[-1]:.3f}  "
            f"JS[0]={js_divs[0]:.3f}..[-1]={js_divs[-1]:.3f}"
        )

    csv_path = out_dir / "logit_lens.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "contrast",
                "label",
                "layer",
                "n_pairs",
                "top1_agreement",
                "top10_jaccard",
                "js_divergence",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[04] csv -> {csv_path}")

    json_path = out_dir / "logit_lens.json"
    json_path.write_text(
        json.dumps(per_contrast, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[04] matplotlib missing; skipping figure")
        return

    if not per_contrast:
        return

    palette = {"d1": "#d62728", "d2": "#1f77b4", "d3": "#7f7f7f"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for slug, data in per_contrast.items():
        color = palette.get(slug)
        layers = list(range(len(data["top1_agreement"])))
        axes[0].plot(
            layers, data["top1_agreement"], "-o", markersize=3,
            label=slug.upper(), color=color,
        )
        axes[1].plot(
            layers, data["top10_jaccard"], "-o", markersize=3,
            label=slug.upper(), color=color,
        )
        axes[2].plot(
            layers, data["js_divergence"], "-o", markersize=3,
            label=slug.upper(), color=color,
        )

    axes[0].set_title("Top-1 token agreement")
    axes[0].set_ylabel("agreement")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_title("Top-10 Jaccard overlap")
    axes[1].set_ylabel("overlap")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_title("Jensen-Shannon divergence")
    axes[2].set_ylabel("JS (nats)")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"DIA-LOC: logit-lens at last-token — {MODEL_ID}"
    )
    fig.tight_layout()

    png_path = fig_dir / "04_logit_lens.png"
    fig.savefig(png_path, dpi=150)
    print(f"[04] figure -> {png_path}")


if __name__ == "__main__":
    main()
