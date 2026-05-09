"""Logit lens: per-layer top-1 token agreement between paired residuals.

For each layer L and each pair (a, b):
  - Compute logits_a = embed @ a[L]
  - Compute logits_b = embed @ b[L]
  - top1_a = argmax(logits_a), top1_b = argmax(logits_b)
  - agreement(L) = mean over pairs of [top1_a == top1_b]

Reads the same activations as 03_similarity.py, plus the model's
embedding matrix (used as the unembedding via Qwen's tied weights).
For Qwen 2.5, model.embed_tokens.weight is tied to lm_head.weight, so
we project residuals through embed_tokens to get the same logits
lm_head would have produced.

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


def top1_agreement(
    a_layer: torch.Tensor,  # [N, D]
    b_layer: torch.Tensor,  # [N, D]
    embed: torch.Tensor,    # [V, D]
) -> float:
    # Project to logits. We don't softmax; argmax is invariant to it.
    logits_a = a_layer.to(torch.float32) @ embed.t()  # [N, V]
    logits_b = b_layer.to(torch.float32) @ embed.t()
    top_a = logits_a.argmax(dim=-1)
    top_b = logits_b.argmax(dim=-1)
    return float((top_a == top_b).float().mean().item())


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
        a_path = act_dir / f"{slug}_a.pt"
        b_path = act_dir / f"{slug}_b.pt"
        if not (a_path.exists() and b_path.exists()):
            print(f"[04] {slug}: tensors missing, skipping")
            continue
        a = torch.load(a_path, weights_only=True, map_location="cpu")  # [N, L, D]
        b = torch.load(b_path, weights_only=True, map_location="cpu")
        if a.shape != b.shape:
            print(f"[04] {slug}: shape mismatch, skipping")
            continue
        n_pairs, n_layers, _ = a.shape

        agreements: list[float] = []
        for li in range(n_layers):
            agr = top1_agreement(a[:, li, :], b[:, li, :], embed)
            agreements.append(agr)
            rows.append({
                "contrast": slug,
                "label": c["label"],
                "layer": li,
                "n_pairs": n_pairs,
                "top1_agreement": agr,
            })
        per_contrast[slug] = {
            "label": c["label"],
            "n_pairs": n_pairs,
            "top1_agreement": agreements,
        }
        print(
            f"[04] {slug:>3}: n={n_pairs:>3}  "
            f"agr[0]={agreements[0]:.3f}..agr[-1]={agreements[-1]:.3f}"
        )

    csv_path = out_dir / "logit_lens.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["contrast", "label", "layer", "n_pairs", "top1_agreement"],
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
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for slug, data in per_contrast.items():
        layers = list(range(len(data["top1_agreement"])))
        ax.plot(
            layers,
            data["top1_agreement"],
            "-o",
            markersize=3,
            label=slug.upper(),
            color=palette.get(slug),
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("top-1 token agreement")
    ax.set_title(f"DIA-LOC: logit-lens agreement — {MODEL_ID}")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(1.0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png_path = fig_dir / "04_logit_lens.png"
    fig.savefig(png_path, dpi=150)
    print(f"[04] figure -> {png_path}")


if __name__ == "__main__":
    main()
