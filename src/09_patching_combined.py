"""Combine the per-contrast activation-patching results into one plot.

Reads runs/probes/activation_patching_{d1,d2,d3}.json and produces a
side-by-side comparison of last-token-residual transfer curves with
50% and 90% transfer markers per contrast.

This is the paper's headline figure for §4.6: language identity
consolidates early, lexical content middle, dialect late.

Usage:

    python src/09_patching_combined.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.config import REPO_ROOT


def first_layer_at(arr: list[float], threshold: float) -> int | None:
    for li, v in enumerate(arr):
        if v >= threshold:
            return li
    return None


def main() -> None:
    runs = REPO_ROOT / "runs" / "probes"
    results: dict[str, dict] = {}
    for slug in ("d1", "d2", "d3"):
        path = runs / f"activation_patching_{slug}.json"
        if not path.exists():
            print(f"[09] missing {path}; skipping {slug}")
            continue
        results[slug] = json.loads(path.read_text(encoding="utf-8"))

    if not results:
        sys.exit("[09] no patching results to combine")

    palette = {"d1": "#d62728", "d2": "#1f77b4", "d3": "#7f7f7f"}
    label = {
        "d1": "D1: BM↔NN (dialect)",
        "d2": "D2: NB↔EN (foreign-language)",
        "d3": "D3: BM↔BM (paraphrase control)",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    summary_lines = ["[09] consolidation thresholds:"]
    for slug, data in results.items():
        means = np.array(data["transfer_per_layer_mean"])
        stds = np.array(data["transfer_per_layer_std"])
        layers = np.arange(len(means))
        color = palette.get(slug)
        ax.plot(layers, means, "-o", markersize=3.5, label=label.get(slug, slug), color=color)
        ax.fill_between(layers, means - stds, means + stds, alpha=0.12, color=color)

        l50 = first_layer_at(list(means), 0.5)
        l90 = first_layer_at(list(means), 0.9)
        baseline_kl = data.get("kl_baseline_mean", float("nan"))
        summary_lines.append(
            f"  {slug.upper()}: 50%@L{l50}  90%@L{l90}  baseline KL={baseline_kl:.3f} nats"
        )
        # Vertical drops at 50%/90% markers, in the matching color
        for thresh, ls in ((0.5, "--"), (0.9, ":")):
            li = first_layer_at(list(means), thresh)
            if li is not None:
                ax.axvline(li, color=color, alpha=0.35, linewidth=1.0, linestyle=ls)

    ax.axhline(0.5, color="black", linewidth=0.4, alpha=0.5, linestyle="--")
    ax.axhline(0.9, color="black", linewidth=0.4, alpha=0.5, linestyle=":")
    ax.text(0.2, 0.51, "50%", fontsize=8, alpha=0.6)
    ax.text(0.2, 0.91, "90%", fontsize=8, alpha=0.6)
    ax.set_xlabel("Layer at which the LAST-TOKEN residual was patched A → B")
    ax.set_ylabel("transfer fraction  $1 - \\mathrm{KL}(p_{a\\,patched} \\| p_b) / \\mathrm{KL}(p_a \\| p_b)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "DIA-LOC §4.6: where each contrast 'commits' in the residual stream\n"
        "Foreign-language is decided early (~L10), paraphrase content middle (~L18),"
        " dialect late (~L21)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_dir = REPO_ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "08_activation_patching_combined.png"
    fig.savefig(png, dpi=150)
    print(f"[09] figure -> {png}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
