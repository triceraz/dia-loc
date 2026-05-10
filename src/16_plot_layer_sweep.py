"""Plot the v0.5 layer-window sweep result.

Reads runs/dpo/evaluation.json (produced by 15_dpo_evaluate.py with all
five adapters trained). Plots, for each variant, the chosen-vs-rejected
log-prob margin gain over the base. The x-axis shows the layer
window's centre on the 0-27 stack so the geographical story is
visible: where in the stack does 6-layer LoRA capacity have the most
effect?

Output:
    paper/figures/16_layer_sweep.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.config import REPO_ROOT


# Layer ranges per variant. Keep in sync with src/14_dpo_train.py.
WINDOWS = {
    "early": (0, 5),
    "earlymid": (7, 12),
    "mid": (14, 19),
    "targeted": (21, 26),
}


def main() -> None:
    eval_path = REPO_ROOT / "runs" / "dpo" / "evaluation.json"
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    by_label = {d["label"]: d for d in data}

    if "base" not in by_label:
        sys.exit("evaluation.json has no 'base' entry; cannot compute deltas")
    base_margin = by_label["base"]["mean_margin"]

    # Sliding-window points
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for variant, (lo, hi) in WINDOWS.items():
        if variant not in by_label:
            print(f"[16] missing variant {variant}, skipping")
            continue
        center = (lo + hi) / 2
        delta = by_label[variant]["mean_margin"] - base_margin
        xs.append(center)
        ys.append(delta)
        labels.append(variant)

    full_delta = (
        by_label["full"]["mean_margin"] - base_margin
        if "full" in by_label
        else None
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Sliding-window curve
    ax.plot(xs, ys, "-o", color="#1A4DFF", markersize=8, linewidth=2,
            label="6-layer LoRA window (~933K params)")
    for x, y, lbl in zip(xs, ys, labels):
        ax.text(x, y + 0.25, lbl, ha="center", fontsize=9, color="#0F0F12")
        ax.text(x, y - 0.7, f"L{WINDOWS[lbl][0]}-{WINDOWS[lbl][1]}",
                ha="center", fontsize=8, color="#6E6E76", family="monospace")

    if full_delta is not None:
        ax.axhline(full_delta, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"full-stack LoRA (~4.4M params)  delta={full_delta:+.2f}")

    ax.axhline(0, color="#0F0F12", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Layer-window centre (0 = input, 27 = final block)", color="#6E6E76")
    ax.set_ylabel("delta log-prob margin (chosen − rejected) vs base",
                  color="#6E6E76")
    ax.set_xlim(-1, 28)
    ax.set_xticks([0, 7, 14, 21, 27])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.suptitle(
        "DIA-LOC v0.5 · DPO LoRA layer-window sweep on Qwen 2.5 1.5B\n"
        "Where in the residual stack does dialect-direction LoRA capacity matter most?",
        fontsize=11, color="#0F0F12",
    )
    fig.tight_layout()

    out = REPO_ROOT / "paper" / "figures" / "16_layer_sweep.png"
    fig.savefig(out, dpi=150, facecolor="#FAFAFA")
    print(f"[16] -> {out}")
    print()
    print("delta margin per variant:")
    for v, x, y in sorted(zip(labels, xs, ys), key=lambda r: r[1]):
        print(f"  {v:<10} (window centre L{x:>4.1f}): delta = {y:+.3f}")
    if full_delta is not None:
        print(f"  {'full':<10} (all 28 layers):  delta = {full_delta:+.3f}")


if __name__ == "__main__":
    main()
