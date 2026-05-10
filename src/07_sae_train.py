"""Sparse autoencoder on per-token residual stream activations.

Trains a small ReLU SAE on the flat per-token activations captured by
02 with --per-token-layers, then identifies features that fire
differentially between paired sides per contrast.

v0.2 upgrade over the v0.1 mean-pooled SAE:
  - Trains on per-token activations (~36k samples) instead of mean-
    pooled (~1k). 36x more training data.
  - Mini-batch SGD with shuffling + warmup + AdamW. The v0.1 full-
    batch full-dataset Adam was overkill for 1k samples but
    underkill at 36k.
  - Per-feature side-mean computed by aggregating tokens per input
    via the meta JSON, then averaging across inputs (so a 200-token
    sentence doesn't dominate a 20-token sentence).
  - Reports cross-contrast IoU on top differential features as the
    H2-entanglement signal, plus a per-input differential-feature
    plot per contrast.

Architecture:
  encoder: D -> M*D linear, ReLU
  decoder: M*D -> D linear (decoder columns kept unit-norm during
           training, standard SAE recipe)

Loss = mean((x_hat - x)^2) + lambda * mean(features.abs())

Usage:

    python src/07_sae_train.py --layer 20 [--width-mult 8] [--epochs 30]

(Layer 20 must have been captured by 02 with --per-token-layers 20.)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from lib.config import REPO_ROOT, activations_dir


class SimpleSAE(nn.Module):
    def __init__(self, d_model: int, width_mult: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_features = d_model * width_mult
        self.encoder = nn.Linear(d_model, self.n_features, bias=True)
        self.decoder = nn.Linear(self.n_features, d_model, bias=False)
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        # Decoder columns kept unit-norm (canonical SAE constraint).
        w = self.decoder.weight  # [D, M*D]
        norm = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
        w.div_(norm)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


def load_per_token(act_dir: Path, slug: str, side: str, layer: int):
    """Returns (tensor [n_tokens, d_model] fp16, meta list)."""
    t_path = act_dir / f"{slug}_{side}_l{layer:02d}_pertoken.pt"
    m_path = act_dir / f"{slug}_{side}_pertoken_meta.json"
    if not (t_path.exists() and m_path.exists()):
        raise FileNotFoundError(
            f"Per-token tensors missing at {t_path}. "
            f"Run: python src/02_capture_activations.py --per-token-layers {layer}"
        )
    t = torch.load(t_path, weights_only=True, map_location="cpu")
    meta = json.loads(m_path.read_text(encoding="utf-8"))
    return t, meta


def per_input_feature_means(
    z: torch.Tensor,            # [n_tokens, n_features]
    meta: list[dict],           # length n_tokens
) -> np.ndarray:
    """For each input (sentence), mean feature activation over its
    tokens. Returns [n_inputs, n_features].

    This avoids long sentences dominating short ones in the cross-
    side mean.
    """
    by_input: dict[int, list[int]] = defaultdict(list)
    for row_idx, m in enumerate(meta):
        by_input[m["input_idx"]].append(row_idx)
    n_inputs = len(by_input)
    n_features = z.shape[1]
    out = np.zeros((n_inputs, n_features), dtype=np.float32)
    z_np = z.numpy()
    for input_idx, rows in by_input.items():
        out[input_idx] = z_np[rows].mean(axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--width-mult", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1", type=float, default=5e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top-K differential-feature set size for cross-contrast IoU.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    act_dir = activations_dir()
    manifest = json.loads((act_dir / "manifest.json").read_text(encoding="utf-8"))
    contrasts = manifest["contrasts"]

    # ---------- gather all training samples (flat across contrasts) ----------
    all_acts: list[torch.Tensor] = []
    per_side_data: dict[str, tuple[torch.Tensor, list[dict]]] = {}
    for c in contrasts:
        slug = c["slug"]
        for side in ("a", "b"):
            t, meta = load_per_token(act_dir, slug, side, args.layer)
            per_side_data[f"{slug}_{side}"] = (t, meta)
            all_acts.append(t)

    X = torch.cat(all_acts, dim=0).to(torch.float32)  # [total_tokens, D]
    print(f"[07] training: {X.shape[0]} tokens, dim={X.shape[1]}, layer={args.layer}")

    # ---------- train SAE ----------
    sae = SimpleSAE(d_model=X.shape[1], width_mult=args.width_mult).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr)

    n = X.shape[0]
    n_batches = (n + args.batch_size - 1) // args.batch_size
    pbar = tqdm(range(args.epochs), desc="train")
    for epoch in pbar:
        perm = torch.randperm(n)
        running = {"recon": 0.0, "l1": 0.0, "active": 0.0}
        for b in range(n_batches):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            batch = X[idx].to(device)
            x_hat, z = sae(batch)
            recon = ((x_hat - batch) ** 2).mean()
            l1 = z.abs().mean()
            loss = recon + args.l1 * l1

            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae._normalize_decoder()

            running["recon"] += float(recon.item())
            running["l1"] += float(l1.item())
            running["active"] += float((z > 1e-6).float().mean().item())
        pbar.set_postfix(
            recon=f"{running['recon']/n_batches:.3f}",
            l1=f"{running['l1']/n_batches:.3f}",
            active=f"{running['active']/n_batches:.3f}",
        )

    # ---------- compute per-input feature means + side-mean per contrast ----------
    sae.eval()
    by_side_input_means: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for side_key, (t, meta) in per_side_data.items():
            z_chunks: list[torch.Tensor] = []
            for chunk in t.split(8192):
                _, z = sae(chunk.to(device).to(torch.float32))
                z_chunks.append(z.cpu())
            z = torch.cat(z_chunks, dim=0)
            input_means = per_input_feature_means(z, meta)  # [n_inputs, n_feat]
            by_side_input_means[side_key] = input_means.mean(axis=0)  # [n_feat]

    # ---------- top-K differential features per contrast ----------
    top_sets: dict[str, set[int]] = {}
    detail: dict[str, dict] = {}
    for c in contrasts:
        slug = c["slug"]
        a = by_side_input_means[f"{slug}_a"]
        b = by_side_input_means[f"{slug}_b"]
        diff = a - b
        abs_diff = np.abs(diff)
        # Top-K by absolute difference
        top_idx = np.argsort(abs_diff)[-args.top_k:][::-1]
        top_sets[slug] = set(int(i) for i in top_idx)
        # Top 5 in each direction for the human-readable summary
        detail[slug] = {
            "label": c["label"],
            "lang_a": c["lang_a"],
            "lang_b": c["lang_b"],
            "abs_diff_max": float(abs_diff.max()),
            "abs_diff_mean": float(abs_diff.mean()),
            "top_features_for_a": [
                {"feature": int(i), "diff": float(diff[i])}
                for i in np.argsort(diff)[-10:][::-1]
            ],
            "top_features_for_b": [
                {"feature": int(i), "diff": float(diff[i])}
                for i in np.argsort(diff)[:10]
            ],
        }
        print(
            f"[07] {slug}: |a-b|.max={abs_diff.max():.3f}  "
            f"|a-b|.mean={abs_diff.mean():.4f}  top-{args.top_k} cardinality={len(top_sets[slug])}"
        )

    # ---------- cross-contrast IoU (the entanglement test) ----------
    iou_lines = ["[07] Top-K differential-feature IoU between contrasts:"]
    iou_records = []
    for s1, s2 in (("d1", "d2"), ("d1", "d3"), ("d2", "d3")):
        if s1 in top_sets and s2 in top_sets:
            inter = top_sets[s1] & top_sets[s2]
            union = top_sets[s1] | top_sets[s2]
            iou = len(inter) / len(union) if union else 0.0
            iou_lines.append(
                f"        {s1.upper()} vs {s2.upper()}: K={args.top_k} "
                f"IoU={iou:.3f} shared={len(inter)}"
            )
            iou_records.append({"s1": s1, "s2": s2, "k": args.top_k, "iou": iou, "shared": len(inter)})
    print("\n".join(iou_lines))

    # ---------- save ----------
    out_dir = REPO_ROOT / "runs" / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sae_pertoken_layer{args.layer}.json"
    out_path.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "width_mult": args.width_mult,
                "n_features": sae.n_features,
                "n_tokens_total": int(X.shape[0]),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "l1": args.l1,
                "lr": args.lr,
                "top_k": args.top_k,
                "iou": iou_records,
                "results": detail,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[07] -> {out_path}")

    # ---------- plot: bar chart of cross-contrast IoUs ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not iou_records:
        return

    fig_dir = REPO_ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.0))
    pairs = [f"{r['s1'].upper()} vs {r['s2'].upper()}" for r in iou_records]
    vals = [r["iou"] for r in iou_records]
    bars = ax.bar(pairs, vals, color=["#a02020", "#a06060", "#205080"])
    ax.set_ylabel(f"top-{args.top_k} differential-feature IoU")
    ax.set_ylim(0, max(vals + [0.001]) * 1.2)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=9,
        )
    ax.set_title(
        f"DIA-LOC: SAE differential-feature IoU across contrasts\n"
        f"layer {args.layer}, n_features={sae.n_features}, "
        f"trained on {X.shape[0]} per-token activations"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png_path = fig_dir / f"07_sae_iou_layer{args.layer}.png"
    fig.savefig(png_path, dpi=150)
    print(f"[07] figure -> {png_path}")


if __name__ == "__main__":
    main()
