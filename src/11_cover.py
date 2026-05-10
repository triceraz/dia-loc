"""Generate a cover image for the DIA-LOC forskning page.

Editorial-style restatement of the consolidation hierarchy:
  - foreign-language committed at ~30% through the stack
  - paraphrase content at ~67%
  - dialect at ~80%

Uses Tenki design tokens. 16:9 aspect ratio, ~1800x1013 at 150 dpi.

Run:
    python src/11_cover.py
Output:
    paper/figures/cover.png
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
from matplotlib.patches import FancyArrowPatch

from lib.config import REPO_ROOT


# Tenki design tokens
COL_BG = "#FAFAFA"
COL_INK = "#0F0F12"
COL_MUTED = "#6E6E76"
COL_SUBTLE = "#E2E2E2"
COL_ACCENT = "#1A4DFF"

# Contrast palette (matches the per-probe figures)
COL_D1 = "#d62728"  # dialect
COL_D2 = "#1f77b4"  # foreign-language
COL_D3 = "#7f7f7f"  # paraphrase


def main() -> None:
    runs = REPO_ROOT / "runs" / "probes"

    # Pull the 1.5B activation-patching curves from the JSON outputs.
    series = {}
    for slug in ("d1", "d2", "d3"):
        path = runs / f"activation_patching_{slug}.json"
        if not path.exists():
            sys.exit(
                f"missing {path}; run src/08_activation_patching.py for all "
                "three contrasts first"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        series[slug] = np.array(data["transfer_per_layer_mean"])

    # Use the 1.5B 28-layer x-axis (proportional positions). The story
    # we want to tell is the relative timing, not the absolute layer
    # numbers, so we normalize to layer-fraction.
    n_layers = len(series["d1"])
    x = np.arange(n_layers) / (n_layers - 1)

    fig = plt.figure(figsize=(13.5, 7.6), dpi=150)
    fig.patch.set_facecolor(COL_BG)

    # Two-row layout: top eyebrow + title (text-only), bottom plot.
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 2.6],
        hspace=0.05,
        left=0.07,
        right=0.93,
        top=0.95,
        bottom=0.10,
    )

    # ---------- top: title text ----------
    ax_title = fig.add_subplot(gs[0])
    ax_title.set_facecolor(COL_BG)
    ax_title.axis("off")

    ax_title.text(
        0.0, 0.95,
        "DIA-LOC",
        fontsize=10, color=COL_MUTED, fontweight=500,
        family="DM Mono", transform=ax_title.transAxes,
        verticalalignment="top",
    )
    ax_title.text(
        0.0, 0.78,
        "Hvor i hjernen bestemmer en KI nynorsk?",
        fontsize=32, color=COL_INK, fontweight=500,
        family="DM Sans", transform=ax_title.transAxes,
        verticalalignment="top",
    )
    ax_title.text(
        0.0, 0.18,
        "Aktiverings-patching av Qwen 2.5 1.5B avslører et hierarki: "
        "språk-identitet bestemmes tidlig i nettverket, leksikalsk innhold midtveis, "
        "dialekt mot slutten.",
        fontsize=14, color=COL_MUTED, fontweight=400,
        family="DM Sans", transform=ax_title.transAxes,
        verticalalignment="top",
    )

    # ---------- bottom: the curves ----------
    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(COL_BG)

    # The three curves
    ax.plot(x, series["d2"], "-", color=COL_D2, linewidth=2.4,
            label="Foreign language (norsk vs engelsk)")
    ax.plot(x, series["d3"], "-", color=COL_D3, linewidth=2.4,
            label="Lexical content (parafrase)")
    ax.plot(x, series["d1"], "-", color=COL_D1, linewidth=2.4,
            label="Dialect (bokmål vs nynorsk)")

    # Find the 50% threshold for each
    def first_at(arr, t):
        for i, v in enumerate(arr):
            if v >= t:
                return i
        return None

    thresholds = {
        "d2": first_at(series["d2"], 0.5),
        "d3": first_at(series["d3"], 0.5),
        "d1": first_at(series["d1"], 0.5),
    }

    # Vertical guide lines at the 50% layers, faded
    for slug, li in thresholds.items():
        if li is None:
            continue
        col = {"d1": COL_D1, "d2": COL_D2, "d3": COL_D3}[slug]
        frac = li / (n_layers - 1)
        ax.axvline(frac, color=col, alpha=0.20, linewidth=1.2, linestyle="-")

    # Big inline annotations: layer band callouts. Placed ABOVE the
    # 1.0 reference (in the headroom band 1.05-1.18) so they don't
    # clash with the curves themselves which rise to 1.0 on the right.
    callouts = [
        # x_frac, color, line1 (small), line2 (big)
        (thresholds["d2"] / (n_layers - 1), COL_D2, "Lag ~10",  "språk"),
        (thresholds["d3"] / (n_layers - 1), COL_D3, "Lag ~18",  "innhold"),
        (thresholds["d1"] / (n_layers - 1), COL_D1, "Lag ~21",  "dialekt"),
    ]
    for xf, col, l1, l2 in callouts:
        ax.text(
            xf, 1.16, l2,
            ha="center", va="top",
            fontsize=18, color=col, fontweight=600,
            family="DM Sans",
        )
        ax.text(
            xf, 1.06, l1,
            ha="center", va="top",
            fontsize=11, color=col, fontweight=500,
            family="DM Mono",
        )

    # 50% reference line (faint)
    ax.axhline(0.5, color=COL_INK, linewidth=0.5, alpha=0.20, linestyle="--")
    ax.text(
        1.005, 0.5, "50%",
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=9, color=COL_MUTED, family="DM Mono",
    )

    # Axis cosmetics
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.06, 1.20)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(
        ["lag 0", "", "", "", f"lag {n_layers - 1}"],
        fontsize=11, color=COL_MUTED, family="DM Mono",
    )
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COL_SUBTLE)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="x", which="both", length=0, pad=8)

    # X-axis label
    ax.set_xlabel(
        "→ stadig dypere i Qwen 2.5 1.5B (28 lag)",
        fontsize=11, color=COL_MUTED, family="DM Sans",
        labelpad=12,
    )

    # Y-axis label (subtle, on the left, vertical)
    ax.text(
        -0.02, 0.5,
        "Hvor 'forpliktet' er prediksjonen?",
        transform=ax.transAxes,
        rotation=90, ha="right", va="center",
        fontsize=11, color=COL_MUTED, family="DM Sans",
    )

    # Legend below the plot, no box, in muted text
    leg = ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, -0.02),
        frameon=False,
        fontsize=10,
        labelcolor=COL_INK,
    )
    for text in leg.get_texts():
        text.set_family("DM Sans")

    # Footer: tenki signature, very subtle
    fig.text(
        0.93, 0.025,
        "tenki forskning · github.com/triceraz/dia-loc",
        ha="right", va="bottom",
        fontsize=9, color=COL_MUTED, family="DM Mono",
    )

    out = REPO_ROOT / "paper" / "figures" / "cover.png"
    fig.savefig(out, dpi=150, facecolor=COL_BG)
    print(f"[11] cover -> {out}")
    print(f"     size: 13.5 x 7.6 inches @ 150dpi  (16:9)")


if __name__ == "__main__":
    main()
