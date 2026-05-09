"""Layer-wise cosine + CKA similarity between paired residual streams.

Reads activations captured by `02_capture_activations.py` from
runs/activations/<checkpoint_slug>/<contrast>_a.pt and ..._b.pt.
Each tensor is shape [n_pairs, n_layers, d_model] in fp16.

For each contrast (D1 = BM↔NN, D2 = NB↔EN, D3 = BM↔BM control) and
each layer L:

  - cosine(L) = mean over pairs of cosine_similarity(a[i, L], b[i, L])
  - CKA(L)    = linear CKA between a[:, L, :] and b[:, L, :]

Both are sentence-level (mean-pooled in step 02). Cosine answers "do
paired residuals point in the same direction on average?". CKA answers
"do the per-layer representation spaces look the same up to invertible
transformation?". The two often agree but disagree at extremes — they're
complementary, not redundant.

Outputs:

  paper/figures/03_similarity.png   — per-layer cosine + CKA, all contrasts
  runs/probes/similarity.csv        — raw numbers for downstream stats
  runs/probes/similarity.json       — same as JSON for ad-hoc plotting

Run:

    python src/03_similarity.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Allow `from lib.foo import ...` regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from lib.config import REPO_ROOT, activations_dir


def cosine_per_layer(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Mean per-pair cosine similarity at each layer.

    Inputs: tensors of shape [N, L, D] (fp16 OK, we cast to fp32 for the
    arithmetic).
    Output: 1-D numpy array of length L.
    """
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    # Normalize along the d_model axis. Add eps to avoid 0/0 on degenerate
    # zero-vector residuals (shouldn't happen on real data, but be safe).
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    # Per-(pair, layer) dot product, then mean over pairs.
    cos = (a_n * b_n).sum(dim=-1)  # [N, L]
    return cos.mean(dim=0).numpy()


def linear_cka_per_layer(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Linear CKA per layer between two activation matrices.

    Linear CKA between centered matrices X, Y is
        ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F).
    See Kornblith et al. 2019, *Similarity of Neural Network
    Representations Revisited*.

    Inputs: tensors [N, L, D]. Output: 1-D numpy array of length L.
    """
    a = a.to(torch.float32).numpy()
    b = b.to(torch.float32).numpy()
    n_layers = a.shape[1]
    out = np.zeros(n_layers, dtype=np.float64)
    for li in range(n_layers):
        X = a[:, li, :]  # [N, D]
        Y = b[:, li, :]
        # Center along the sample axis.
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        # Frobenius-norm form of linear CKA.
        xy = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
        xx = float(np.linalg.norm(X.T @ X, ord="fro"))
        yy = float(np.linalg.norm(Y.T @ Y, ord="fro"))
        out[li] = xy / (xx * yy + 1e-12)
    return out


def main() -> None:
    act_dir = activations_dir()
    manifest_path = act_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(
            f"No manifest at {manifest_path}. "
            "Run src/02_capture_activations.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    contrasts = manifest.get("contrasts", [])
    if not contrasts:
        sys.exit(f"Manifest has no contrasts: {manifest_path}")

    out_dir = REPO_ROOT / "runs" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_contrast: dict[str, dict] = {}
    n_layers = manifest.get("n_layers")

    for c in contrasts:
        slug = c["slug"]
        a_path = act_dir / f"{slug}_a.pt"
        b_path = act_dir / f"{slug}_b.pt"
        if not (a_path.exists() and b_path.exists()):
            print(f"[03] {slug}: tensors missing, skipping")
            continue
        a = torch.load(a_path, weights_only=True, map_location="cpu")
        b = torch.load(b_path, weights_only=True, map_location="cpu")
        if a.shape != b.shape:
            print(f"[03] {slug}: shape mismatch a={a.shape} b={b.shape}, skipping")
            continue

        n_pairs = a.shape[0]
        if n_layers is None:
            n_layers = a.shape[1]

        cos = cosine_per_layer(a, b)
        cka = linear_cka_per_layer(a, b)
        per_contrast[slug] = {
            "label": c["label"],
            "lang_a": c["lang_a"],
            "lang_b": c["lang_b"],
            "n_pairs": n_pairs,
            "cosine": cos.tolist(),
            "cka": cka.tolist(),
        }
        for li in range(int(a.shape[1])):
            rows.append({
                "contrast": slug,
                "label": c["label"],
                "layer": li,
                "n_pairs": n_pairs,
                "cosine": float(cos[li]),
                "cka": float(cka[li]),
            })
        print(
            f"[03] {slug:>3}: n={n_pairs:>3}  "
            f"cos[0]={cos[0]:.3f}..cos[-1]={cos[-1]:.3f}  "
            f"cka[0]={cka[0]:.3f}..cka[-1]={cka[-1]:.3f}"
        )

    # CSV + JSON dumps for downstream stats / plotting from notebooks
    csv_path = out_dir / "similarity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["contrast", "label", "layer", "n_pairs", "cosine", "cka"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[03] csv -> {csv_path}")

    json_path = out_dir / "similarity.json"
    json_path.write_text(
        json.dumps(per_contrast, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[03] json -> {json_path}")

    # Matplotlib plot. Imported here so the script still runs (CSV+JSON
    # only) on systems without a usable display backend.
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print("[03] matplotlib not installed; skipping figure")
        return

    if not per_contrast:
        return

    palette = {
        "d1": "#d62728",  # red — dialectal contrast
        "d2": "#1f77b4",  # blue — foreign-language contrast
        "d3": "#7f7f7f",  # gray — control
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for slug, data in per_contrast.items():
        color = palette.get(slug, None)
        layers = list(range(len(data["cosine"])))
        axes[0].plot(
            layers, data["cosine"], "-o", markersize=3, label=slug.upper(), color=color,
        )
        axes[1].plot(
            layers, data["cka"], "-o", markersize=3, label=slug.upper(), color=color,
        )

    axes[0].set_title("Cosine similarity (mean over pairs)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("cos(a, b)")
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.3)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Linear CKA")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("CKA")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"DIA-LOC: layer-wise paired similarity — {manifest.get('model_id')}"
    )
    fig.tight_layout()

    png_path = fig_dir / "03_similarity.png"
    fig.savefig(png_path, dpi=150)
    print(f"[03] figure -> {png_path}")


if __name__ == "__main__":
    main()
