from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent

sen_path = Path.home() / "AppData/Local/Microsoft/Windows/Fonts/Sen-VariableFont_wght.ttf"
if sen_path.exists():
    fm.fontManager.addfont(str(sen_path))
    font_family = "Sen"
else:
    font_family = "DejaVu Sans"

navy = "#005F73"
amber = "#EE9B00"
ink = "#262B30"
mid = "#68717A"
grid = "#D9DEE1"

mpl.rcParams.update(
    {
        "font.family": font_family,
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.edgecolor": mid,
        "axes.linewidth": 0.7,
        "xtick.color": ink,
        "ytick.color": ink,
        "text.color": ink,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def style_axis(ax, title, panel):
    ax.set_title(title, loc="left", pad=7)
    ax.text(
        -0.12,
        1.13,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
    )
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-3.80, 3.05)
    ax.set_xticks([0, 1], ["U-source", "P-source"])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.grid(axis="y", color=grid, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    ax.axhline(0, color=mid, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.7, pad=3)


def point_intervals(ax, values, lows, highs):
    xs = np.array([0.0, 1.0])
    colors = [navy, amber]
    markers = ["o", "D"]
    for x, value, low, high, color, marker in zip(xs, values, lows, highs, colors, markers):
        ax.errorbar(
            x,
            value,
            yerr=np.array([[value - low], [high - value]]),
            fmt=marker,
            markersize=7.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.6,
            capsize=4.0,
            capthick=1.6,
            zorder=3,
        )
        label_y = high + 0.12 if value >= 0 else low - 0.12
        ax.text(
            x,
            label_y,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9.0,
            fontweight="bold",
            color=color,
        )


fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), sharey=True, facecolor="white")
fig.subplots_adjust(left=0.080, right=0.99, bottom=0.23, top=0.82, wspace=0.24)

style_axis(axes[0], "Broad shift with the visible trace fixed", "A")
style_axis(axes[1], "Valid-fallacious contrast shift", "B")
axes[0].set_ylabel("change (logits)")
axes[1].tick_params(axis="y", labelleft=False)
axes[1].spines["left"].set_visible(False)

point_intervals(
    axes[0],
    values=np.array([2.078, 0.700]),
    lows=np.array([1.759, 0.487]),
    highs=np.array([2.408, 0.907]),
)
point_intervals(
    axes[1],
    values=np.array([-2.422, -0.797]),
    lows=np.array([-3.411, -1.219]),
    highs=np.array([-1.448, -0.391]),
)

# The cross-source interaction is secondary and belongs only to panel A.
# Place it in the otherwise empty negative region to keep it visually
# subordinate to the two co-primary source-conditioned estimates.
axes[0].text(
    0.5,
    -2.85,
    "secondary P-U: -1.378  [-1.720, -1.084]",
    ha="center",
    va="center",
    fontsize=9.0,
    color=ink,
)

fig.savefig(HERE / "fixed-trace-preview.png", dpi=320, facecolor="white")
fig.savefig(HERE / "fixed-trace-vector.pdf", facecolor="white")
fig.savefig(HERE / "fixed-trace-editable.svg", facecolor="white")
plt.close(fig)
