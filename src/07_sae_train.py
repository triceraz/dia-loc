"""Sparse autoencoder on residual stream activations at one chosen layer.

Trains a small SAE on the mean-pooled residuals across D1 + D2 + D3,
then identifies features that fire differentially on BM vs NN, NB vs
EN, and BM-paraphrase vs original.

Architecture (Anthropic-style topk SAE, very small):
  encoder: D -> 8*D linear, ReLU
  decoder: 8*D -> D linear (no bias)
  loss = ||x - reconstruct||^2 + lambda * ||features||_1

For Qwen 2.5 1.5B, D = 1536 -> SAE width = 12288. With ~500 mean-pooled
samples this is way undercomplete by SAE standards (real SAEs train on
millions of tokens), but it's a useful spot-check for whether the
mean-pooled sentence representation has linearly-decomposable
dialect/language features.

The proper SAE for this paper would train on per-token activations
(~25k samples). We're capturing only mean-pooled here for v1; per-
token SAE is a v2 task.

Usage:

    python src/07_sae_train.py [--layer L] [--width-mult M] [--epochs E]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lib.config import REPO_ROOT, activations_dir


class SimpleSAE(nn.Module):
    """Tied-decoder linear SAE with ReLU activation.

    Standard form: x' = W_d @ ReLU(W_e @ x + b_e) + b_d.
    No tied weights; both encoder and decoder are full d-by-(M*d).
    """

    def __init__(self, d_model: int, width_mult: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = d_model * width_mult
        self.encoder = nn.Linear(d_model, self.n_features, bias=True)
        self.decoder = nn.Linear(self.n_features, d_model, bias=True)
        # Initialize decoder columns with unit norm (standard SAE recipe)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=14, help="Which transformer layer to train SAE on (default: middle).")
    parser.add_argument("--width-mult", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    act_dir = activations_dir()
    manifest = json.loads((act_dir / "manifest.json").read_text(encoding="utf-8"))
    contrasts = manifest["contrasts"]

    # Concatenate all activations at the chosen layer, with side labels.
    pieces: list[torch.Tensor] = []
    side_labels: list[str] = []  # "{slug}_a" or "{slug}_b"
    for c in contrasts:
        slug = c["slug"]
        for side in ("a", "b"):
            t = torch.load(
                act_dir / f"{slug}_{side}.pt",
                weights_only=True,
                map_location="cpu",
            )  # [N, L, D]
            layer_slice = t[:, args.layer, :].to(torch.float32)  # [N, D]
            pieces.append(layer_slice)
            side_labels.extend([f"{slug}_{side}"] * len(layer_slice))

    X = torch.cat(pieces, dim=0)  # [total_N, D]
    print(f"[07] training data: {X.shape}, layer={args.layer}")
    X_dev = X.to(device)

    sae = SimpleSAE(d_model=X.shape[1], width_mult=args.width_mult).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    pbar = tqdm(range(args.epochs), desc="train")
    for epoch in pbar:
        x_hat, z = sae(X_dev)
        recon_loss = ((x_hat - X_dev) ** 2).mean()
        sparsity_loss = z.abs().mean()
        loss = recon_loss + args.l1 * sparsity_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            with torch.no_grad():
                active = (z > 1e-6).float().mean().item()
            pbar.set_postfix(
                recon=f"{recon_loss.item():.3f}",
                l1=f"{sparsity_loss.item():.3f}",
                active=f"{active:.3f}",
            )

    # Compute per-side feature activation means.
    sae.eval()
    with torch.no_grad():
        _, z_all = sae(X_dev)
        z_all = z_all.cpu()  # [total_N, n_features]

    # Group rows by side label
    sides = sorted(set(side_labels))
    side_means = {}
    for s in sides:
        mask = torch.tensor([sl == s for sl in side_labels])
        side_means[s] = z_all[mask].mean(dim=0).numpy()  # [n_features]

    # For each contrast, find the top-K features that differ most between
    # the two sides.
    results: dict[str, list[dict]] = {}
    for c in contrasts:
        slug = c["slug"]
        a = side_means[f"{slug}_a"]
        b = side_means[f"{slug}_b"]
        diff = a - b  # [n_features]
        top_pos = np.argsort(diff)[-10:][::-1]
        top_neg = np.argsort(diff)[:10]
        results[slug] = {
            "label": c["label"],
            "lang_a": c["lang_a"],
            "lang_b": c["lang_b"],
            "top_features_for_a": [
                {"feature": int(i), "mean_a": float(a[i]), "mean_b": float(b[i]), "diff": float(diff[i])}
                for i in top_pos
            ],
            "top_features_for_b": [
                {"feature": int(i), "mean_a": float(a[i]), "mean_b": float(b[i]), "diff": float(diff[i])}
                for i in top_neg
            ],
        }
        print(f"[07] {slug}: top {a.shape[0]:>5} features differ by up to {abs(diff).max():.3f}")

    out_dir = REPO_ROOT / "runs" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sae_layer{args.layer}.json"
    out_path.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "width_mult": args.width_mult,
                "n_features": sae.n_features,
                "epochs": args.epochs,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[07] -> {out_path}")

    # Plot: feature-activation overlap of top-10 BM/NN features vs top-10
    # NB/EN features. The headline question for the entanglement
    # hypothesis: do dialect-firing features overlap with foreign-language
    # firing features?
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig_dir = REPO_ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Compute IoU of "top-K differential features" sets across contrasts.
    K = 50  # widen for a meaningful overlap signal
    top_sets: dict[str, set[int]] = {}
    for c in contrasts:
        slug = c["slug"]
        a = side_means[f"{slug}_a"]
        b = side_means[f"{slug}_b"]
        diff = np.abs(a - b)
        top_sets[slug] = set(np.argsort(diff)[-K:].tolist())

    iou_pairs = [("d1", "d2"), ("d1", "d3"), ("d2", "d3")]
    iou_text_lines = ["Top-K differential-feature IoU between contrasts:"]
    for s1, s2 in iou_pairs:
        if s1 in top_sets and s2 in top_sets:
            inter = top_sets[s1] & top_sets[s2]
            union = top_sets[s1] | top_sets[s2]
            iou = len(inter) / len(union) if union else 0.0
            iou_text_lines.append(
                f"  {s1.upper()} vs {s2.upper()}:  K={K}  IoU={iou:.3f}  "
                f"shared={len(inter)}"
            )

    print("\n".join(iou_text_lines))


if __name__ == "__main__":
    main()
