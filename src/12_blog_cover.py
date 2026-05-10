"""Blog-companion cover image.

Different aesthetic from the forskning cover (which shows the curves):
this one is a stylized 28-floor "building" with three highlighted
decision floors. Editorial poster style, easier to grok at thumbnail
size, less chart-like.

Run:
    python src/12_blog_cover.py
Output:
    paper/figures/blog_cover.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from lib.config import REPO_ROOT


# Tenki design tokens
COL_BG = "#FAFAFA"
COL_INK = "#0F0F12"
COL_MUTED = "#6E6E76"
COL_SUBTLE = "#E2E2E2"
COL_FLOOR = "#E8E8E8"

COL_D2 = "#1f77b4"  # foreign-language
COL_D3 = "#7f7f7f"  # paraphrase
COL_D1 = "#d62728"  # dialect

N_LAYERS = 28
HIGHLIGHT_LAYERS = {
    10: ("Lag 10", "Språk bestemmes", COL_D2, "(norsk eller engelsk?)"),
    18: ("Lag 18", "Innhold bestemmes", COL_D3, "(hvilke ord brukes?)"),
    21: ("Lag 21", "Dialekt bestemmes", COL_D1, "(bokmål eller nynorsk?)"),
}


def main() -> None:
    fig = plt.figure(figsize=(13.5, 7.6), dpi=150)
    fig.patch.set_facecolor(COL_BG)

    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.6],
        wspace=0.06,
        left=0.06,
        right=0.96,
        top=0.92,
        bottom=0.08,
    )

    # ---------- left: stylized 28-layer building ----------
    ax_left = fig.add_subplot(gs[0])
    ax_left.set_facecolor(COL_BG)
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(-0.5, N_LAYERS + 0.5)

    # Draw all layers as faint horizontal bars, with highlights overlaid
    bar_height = 0.85
    bar_width = 0.55
    bar_x = 0.05
    for li in range(N_LAYERS):
        is_highlight = li in HIGHLIGHT_LAYERS
        if is_highlight:
            _, _, color, _ = HIGHLIGHT_LAYERS[li]
            face = color
            edge = color
            alpha = 1.0
        else:
            face = COL_FLOOR
            edge = COL_FLOOR
            alpha = 1.0
        ax_left.add_patch(
            Rectangle(
                (bar_x, li),
                bar_width,
                bar_height,
                facecolor=face,
                edgecolor=edge,
                alpha=alpha,
                linewidth=0,
            )
        )

    # Floor numbers for the highlight layers, on the left of the bars
    for li, (label, _, color, _) in HIGHLIGHT_LAYERS.items():
        ax_left.text(
            bar_x - 0.03, li + bar_height / 2,
            str(li),
            ha="right", va="center",
            fontsize=14, color=color, fontweight=600,
            family="DM Mono",
        )
    # First and last layer numbers (subtle)
    for li in (0, N_LAYERS - 1):
        ax_left.text(
            bar_x - 0.03, li + bar_height / 2,
            str(li),
            ha="right", va="center",
            fontsize=10, color=COL_MUTED,
            family="DM Mono",
        )

    # Building cap labels: bottom = INPUT, top = OUTPUT
    ax_left.annotate(
        "",
        xy=(bar_x + bar_width / 2, -0.1),
        xytext=(bar_x + bar_width / 2, -0.4),
        arrowprops=dict(arrowstyle="->", color=COL_MUTED, linewidth=0.8),
    )
    ax_left.text(
        bar_x + bar_width / 2, -0.55,
        "INPUT",
        ha="center", va="top",
        fontsize=10, color=COL_MUTED, family="DM Mono",
        fontweight=500,
    )
    ax_left.annotate(
        "",
        xy=(bar_x + bar_width / 2, N_LAYERS + 0.4),
        xytext=(bar_x + bar_width / 2, N_LAYERS + 0.05),
        arrowprops=dict(arrowstyle="->", color=COL_MUTED, linewidth=0.8),
    )
    ax_left.text(
        bar_x + bar_width / 2, N_LAYERS + 0.55,
        "PREDIKSJON",
        ha="center", va="bottom",
        fontsize=10, color=COL_MUTED, family="DM Mono",
        fontweight=500,
    )

    ax_left.axis("off")

    # ---------- right: title + numbered findings ----------
    ax_right = fig.add_subplot(gs[1])
    ax_right.set_facecolor(COL_BG)
    ax_right.axis("off")
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)

    # Eyebrow
    ax_right.text(
        0.0, 0.95,
        "TENKI FORSKNING · BLOGG",
        fontsize=10, color=COL_MUTED, fontweight=500,
        family="DM Mono",
    )
    # Title
    ax_right.text(
        0.0, 0.85,
        "Hvor i KI-en\nbestemmes nynorsk?",
        fontsize=34, color=COL_INK, fontweight=600,
        family="DM Sans", linespacing=1.05,
        verticalalignment="top",
    )
    # Subtitle
    ax_right.text(
        0.0, 0.55,
        "Vi åpnet en åpen språkmodell og målte hvilken etasje som\n"
        "tar dialekt-valget. Det viste seg å være helt mot toppen.",
        fontsize=14, color=COL_MUTED, fontweight=400,
        family="DM Sans", linespacing=1.4,
        verticalalignment="top",
    )

    # Three findings stacked
    findings_y = [0.36, 0.24, 0.12]
    findings = [
        (HIGHLIGHT_LAYERS[10], findings_y[0]),
        (HIGHLIGHT_LAYERS[18], findings_y[1]),
        (HIGHLIGHT_LAYERS[21], findings_y[2]),
    ]
    for (label, what, color, qualifier), y in findings:
        # Floor number (large)
        ax_right.text(
            0.0, y,
            label.replace("Lag ", "Lag ").upper().replace("LAG ", "Lag "),
            fontsize=22, color=color, fontweight=700,
            family="DM Sans",
            verticalalignment="center",
        )
        # What it decides + qualifier
        ax_right.text(
            0.18, y + 0.012,
            what,
            fontsize=15, color=COL_INK, fontweight=500,
            family="DM Sans",
            verticalalignment="center",
        )
        ax_right.text(
            0.18, y - 0.025,
            qualifier,
            fontsize=11, color=COL_MUTED, fontweight=400,
            family="DM Sans",
            verticalalignment="top",
        )

    # Footer
    fig.text(
        0.96, 0.025,
        "tenki.no/forskning  ·  github.com/triceraz/dia-loc",
        ha="right", va="bottom",
        fontsize=9, color=COL_MUTED, family="DM Mono",
    )

    out = REPO_ROOT / "paper" / "figures" / "blog_cover.png"
    fig.savefig(out, dpi=150, facecolor=COL_BG)
    print(f"[12] blog cover -> {out}")
    print(f"     size: 13.5 x 7.6 inches @ 150dpi  (16:9)")


if __name__ == "__main__":
    main()
