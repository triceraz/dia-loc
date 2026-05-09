"""Linear probe per layer for contrast identity.

For each contrast (d1=BM/NN, d2=NB/EN, d3=BM/BM) and each transformer
layer L:
  - Stack the paired residuals: X = [a; b] of shape [2*N, D],
    labels y = [0]*N + [1]*N (which side of the pair?)
  - Train a linear logistic regression on X -> y, with a held-out test
    split per layer
  - Record probe accuracy (and cross-validated) at each layer

Reads the mean-pooled tensors {slug}_a.pt / {slug}_b.pt from
runs/activations/<checkpoint_slug>/.

Why this matters for the paper:
  - If cosine + CKA say "D1 looks identical at every layer" but a
    linear probe trained on D1 reaches >>50% accuracy, the dialect
    signal is THERE — it's just a small direction the geometric
    metrics smear over.
  - Probe accuracy as a function of layer is the H2 falsifier: if
    BM/NN probe accuracy stays high through deep layers post-BNCR,
    BNCR didn't unify representations; it just aligned outputs.
  - For an off-the-shelf model (no BNCR), probe accuracy gives us
    the BASELINE: where in the stack does dialect identity LIVE in
    Qwen out of the box?

Output:

  paper/figures/05_linear_probes.png   per-layer probe accuracy
  runs/probes/linear_probes.csv        accuracy per (contrast, layer, fold)

Run:

    python src/05_linear_probes.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from lib.config import MODEL_ID, REPO_ROOT, activations_dir


def probe_layer(
    a: np.ndarray,  # [N, D]
    b: np.ndarray,  # [N, D]
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """5-fold cross-validated probe accuracy at one layer.

    Returns (mean_accuracy, std_accuracy) across folds.
    """
    X = np.concatenate([a, b], axis=0)  # [2N, D]
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))]).astype(np.int64)

    accs: list[float] = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(X, y):
        clf = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            n_jobs=1,
            random_state=seed,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        preds = clf.predict(X[te_idx])
        accs.append(float((preds == y[te_idx]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    act_dir = activations_dir()
    manifest_path = act_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No manifest at {manifest_path}; run 02 first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contrasts = manifest.get("contrasts", [])
    if not contrasts:
        sys.exit("Manifest has no contrasts.")

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
            print(f"[05] {slug}: tensors missing, skipping")
            continue
        a = torch.load(a_path, weights_only=True, map_location="cpu")
        b = torch.load(b_path, weights_only=True, map_location="cpu")
        if a.shape != b.shape:
            print(f"[05] {slug}: shape mismatch, skipping")
            continue
        n_pairs, n_layers, _ = a.shape

        means: list[float] = []
        stds: list[float] = []
        for li in range(n_layers):
            mean, std = probe_layer(
                a[:, li, :].to(torch.float32).numpy(),
                b[:, li, :].to(torch.float32).numpy(),
            )
            means.append(mean)
            stds.append(std)
            rows.append({
                "contrast": slug,
                "label": c["label"],
                "layer": li,
                "n_pairs": n_pairs,
                "probe_acc_mean": mean,
                "probe_acc_std": std,
            })
        per_contrast[slug] = {
            "label": c["label"],
            "n_pairs": n_pairs,
            "probe_acc_mean": means,
            "probe_acc_std": stds,
        }
        print(
            f"[05] {slug:>3}: n={n_pairs:>3}  "
            f"acc[0]={means[0]:.3f}±{stds[0]:.3f}.."
            f"[mid]={means[n_layers//2]:.3f}±{stds[n_layers//2]:.3f}.."
            f"[-1]={means[-1]:.3f}±{stds[-1]:.3f}"
        )

    csv_path = out_dir / "linear_probes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "contrast", "label", "layer", "n_pairs",
                "probe_acc_mean", "probe_acc_std",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[05] csv -> {csv_path}")

    json_path = out_dir / "linear_probes.json"
    json_path.write_text(
        json.dumps(per_contrast, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[05] matplotlib missing; skipping figure")
        return

    if not per_contrast:
        return

    palette = {"d1": "#d62728", "d2": "#1f77b4", "d3": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for slug, data in per_contrast.items():
        color = palette.get(slug)
        layers = np.arange(len(data["probe_acc_mean"]))
        means = np.array(data["probe_acc_mean"])
        stds = np.array(data["probe_acc_std"])
        ax.plot(layers, means, "-o", markersize=3, label=slug.upper(), color=color)
        ax.fill_between(layers, means - stds, means + stds, alpha=0.15, color=color)
    ax.axhline(0.5, color="black", linewidth=0.5, alpha=0.4, linestyle=":")
    ax.text(0.0, 0.51, "chance", fontsize=8, color="black", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear-probe accuracy (5-fold CV)")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(
        f"DIA-LOC: per-layer linear probe — can the model distinguish a vs b?"
        f"\n{MODEL_ID}"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png_path = fig_dir / "05_linear_probes.png"
    fig.savefig(png_path, dpi=150)
    print(f"[05] figure -> {png_path}")


if __name__ == "__main__":
    main()
