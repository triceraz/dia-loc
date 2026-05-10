"""Cross-size replication of the activation-patching consolidation finding.

Compares 1.5B (28 layers) and 3B (36 layers) Qwen 2.5 Instruct on the
last-token activation-patching curve per contrast. Plots two views:

  Left panel:  absolute layer index on the x-axis. Tells us if
               consolidation happens at the same absolute layer.
  Right panel: layer fraction (L / n_layers) on the x-axis. Tells us
               if it happens at the same proportional position in
               the stack.

Reads:
  runs/probes/activation_patching_{d1,d2,d3}.json   (1.5B, current)
  runs/probes_3b/activation_patching_{d1,d2,d3}.json (3B, stashed)

Writes:
  paper/figures/10_cross_size_consolidation.png
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


MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B (28L)", REPO_ROOT / "runs" / "probes"),
    ("Qwen/Qwen2.5-3B-Instruct",   "3B (36L)",   REPO_ROOT / "runs" / "probes_3b"),
]

COLOR = {"d1": "#d62728", "d2": "#1f77b4", "d3": "#7f7f7f"}
STYLE = {"1.5B (28L)": "-", "3B (36L)": "--"}
LABEL = {
    "d1": "D1: BM↔NN (dialect)",
    "d2": "D2: NB↔EN (foreign-language)",
    "d3": "D3: BM↔BM (paraphrase)",
}


def first_layer_at(arr: list[float], threshold: float) -> int | None:
    for li, v in enumerate(arr):
        if v >= threshold:
            return li
    return None


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    summary = []
    for slug in ("d2", "d3", "d1"):  # plot order: foreign → paraphrase → dialect
        for model_id, model_label, probes_dir in MODELS:
            path = probes_dir / f"activation_patching_{slug}.json"
            if not path.exists():
                print(f"[10] missing {path}; skipping")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            means = np.array(data["transfer_per_layer_mean"])
            n_layers = len(means)
            layers_abs = np.arange(n_layers)
            layers_frac = layers_abs / max(n_layers - 1, 1)

            color = COLOR[slug]
            style = STYLE[model_label]
            line_label = f"{LABEL[slug]} — {model_label}"

            axes[0].plot(layers_abs, means, style, color=color, label=line_label, linewidth=1.5)
            axes[1].plot(layers_frac, means, style, color=color, label=line_label, linewidth=1.5)

            l50 = first_layer_at(list(means), 0.5)
            l90 = first_layer_at(list(means), 0.9)
            summary.append({
                "model": model_label,
                "contrast": slug,
                "n_layers": n_layers,
                "l50": l50,
                "l50_frac": (l50 / (n_layers - 1)) if l50 is not None else None,
                "l90": l90,
                "l90_frac": (l90 / (n_layers - 1)) if l90 is not None else None,
                "baseline_kl": data.get("kl_baseline_mean"),
            })

    for ax in axes:
        ax.axhline(0.5, color="black", linewidth=0.4, alpha=0.5, linestyle="--")
        ax.axhline(0.9, color="black", linewidth=0.4, alpha=0.5, linestyle=":")
        ax.set_ylabel("transfer fraction")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Absolute layer index L")
    axes[1].set_xlabel("Layer fraction L / (n_layers - 1)")
    axes[0].set_title("Absolute layer")
    axes[1].set_title("Proportional position in stack")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "DIA-LOC §4.7: cross-size replication\n"
        "Qwen 2.5 1.5B (solid, 28 layers) vs 3B (dashed, 36 layers)"
    )
    fig.tight_layout()

    out_dir = REPO_ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "10_cross_size_consolidation.png"
    fig.savefig(png, dpi=150)
    print(f"[10] figure -> {png}")

    print()
    print("Consolidation thresholds:")
    print(f"  {'model':<10} {'contrast':<3}  L50  (frac)   L90  (frac)   baselineKL")
    for s in summary:
        l50 = s['l50'] if s['l50'] is not None else "-"
        l90 = s['l90'] if s['l90'] is not None else "-"
        l50f = f"{s['l50_frac']:.2f}" if s['l50_frac'] is not None else "-"
        l90f = f"{s['l90_frac']:.2f}" if s['l90_frac'] is not None else "-"
        print(f"  {s['model']:<10} {s['contrast'].upper():<3}  {l50!s:>3}  ({l50f})   {l90!s:>3}  ({l90f})   {s['baseline_kl']:.3f}")


if __name__ == "__main__":
    main()
