from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent

# Match the production Figure 1 typography when the local Sen installation is
# available. SVG output remains text-editable in Inkscape.
sen_path = Path.home() / "AppData/Local/Microsoft/Windows/Fonts/Sen-VariableFont_wght.ttf"
sen_bold_path = HERE / "Sen-Bold-700.ttf"
if sen_path.exists():
    fm.fontManager.addfont(str(sen_path))
    FONT_FAMILY = "Sen"
    if not sen_bold_path.exists():
        try:
            from fontTools.ttLib import TTFont
            from fontTools.varLib.instancer import instantiateVariableFont

            bold_font = TTFont(str(sen_path))
            instantiateVariableFont(
                bold_font,
                {"wght": 700},
                inplace=True,
                optimize=True,
                updateFontNames=True,
            )
            bold_font.save(str(sen_bold_path))
        except (ImportError, KeyError, OSError):
            pass
else:
    FONT_FAMILY = "DejaVu Sans"

if sen_bold_path.exists():
    fm.fontManager.addfont(str(sen_bold_path))
    NUMERIC_FONT = fm.FontProperties(fname=str(sen_bold_path), size=9.0)
else:
    NUMERIC_FONT = fm.FontProperties(family=FONT_FAMILY, weight="bold", size=9.0)


# Colors sampled from the stacked-bar result figures in
# When Truth Is Overridden (AAAI-26).
MINT = "#94D2BD"
AMBER = "#EE9B00"
SAND = "#E9D8A6"
NAVY = "#005F73"
INK = "#262B30"
MID = "#68717A"
OUTLINE = "#7B8084"
GRID = "#D9DEE1"
PANEL_BG = "#FAFAF8"
CARD_BG = "#F5F4EF"
POOL = "#AFCBD0"


mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": 9.0,
        "axes.titlesize": 9.8,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.edgecolor": OUTLINE,
        "axes.linewidth": 0.65,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "hatch.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def panel_label(ax, label):
    ax.text(
        -0.135,
        1.20,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
    )


def style_bar_axis(ax, ylabel, ylim, yticks):
    ax.set_facecolor("white")
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.grid(axis="y", color=GRID, lw=0.55, ls=(0, (2, 2)), zorder=0)
    ax.axhline(0, color=OUTLINE, lw=0.75, zorder=1)
    ax.tick_params(axis="both", length=2.5, width=0.6, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig = plt.figure(figsize=(7.05, 3.30), facecolor="white")
gs = fig.add_gridspec(
    2,
    2,
    left=0.082,
    right=0.985,
    bottom=0.14,
    top=0.88,
    wspace=0.31,
    hspace=0.78,
)


# A — Arm-level broad displacement.
ax = fig.add_subplot(gs[0, 0])
panel_label(ax, "A")
ax.set_title(r"Broad shift ($\Delta$ logits)", loc="left", pad=6)

labels = ["N\nNeutral", "R\nRecheck", "V\nValid", "F\nFallacy", "Pooled"]
values = np.array([2.215, 2.092, 2.950, 5.264, 3.130])
colors = [SAND, SAND, MINT, AMBER, POOL]
x = np.arange(len(labels))
bars = ax.bar(
    x,
    values,
    width=0.64,
    color=colors,
    edgecolor=OUTLINE,
    linewidth=0.7,
    zorder=2,
)
bars[-1].set_edgecolor(NAVY)
bars[-1].set_linewidth(0.9)

# All five estimates encode the p25-minus-uncapped contrast. A dedicated
# transparent layer makes the texture white while retaining neutral outlines.
for bar in bars:
    ax.add_patch(
        Rectangle(
            (bar.get_x(), 0),
            bar.get_width(),
            bar.get_height(),
            facecolor="none",
            edgecolor="white",
            linewidth=0,
            hatch="//",
            zorder=3,
        )
    )

pooled_low, pooled_high = 1.834, 4.486
ax.errorbar(
    x[-1],
    values[-1],
    yerr=np.array([[values[-1] - pooled_low], [pooled_high - values[-1]]]),
    fmt="none",
    ecolor=NAVY,
    elinewidth=1.15,
    capsize=2.6,
    capthick=1.15,
    zorder=4,
)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value / 2,
        f"{value:.2f}",
        ha="center",
        va="center",
        fontproperties=NUMERIC_FONT,
        color=INK if bar is not bars[-1] else NAVY,
        zorder=5,
    )

style_bar_axis(ax, "", (0, 6.25), [0, 2, 4, 6])
ax.set_xticks(x, labels)


# B — Decomposition of the continuous displacement.
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, "C")
ax.set_title(r"Margin decomposition ($\Delta$ logits)", loc="left", pad=6)

labels_b = ["Background", "Target", "Target-\nspecific"]
values_b = np.array([3.158, 3.074, -0.084])
low_b = np.array([2.125, 1.180, -1.110])
high_b = np.array([4.250, 5.038, 0.953])
colors_b = [MINT, AMBER, SAND]
xb = np.arange(len(labels_b))
bars_b = ax.bar(
    xb,
    values_b,
    width=0.62,
    color=colors_b,
    edgecolor=OUTLINE,
    linewidth=0.7,
    zorder=2,
)
ax.errorbar(
    xb,
    values_b,
    yerr=np.vstack([values_b - low_b, high_b - values_b]),
    fmt="none",
    ecolor=INK,
    elinewidth=1.0,
    capsize=2.5,
    capthick=1.0,
    zorder=4,
)

for i, (bar, value) in enumerate(zip(bars_b, values_b)):
    if i < 2:
        y, va = value / 2, "center"
    else:
        y, va = -0.31, "center"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        f"{value:+.2f}",
        ha="center",
        va=va,
        fontproperties=NUMERIC_FONT,
        zorder=5,
    )

style_bar_axis(ax, "", (-1.55, 5.55), [-1, 0, 1, 2, 3, 4, 5])
ax.set_xticks(xb, labels_b)


# C — Reference-style paired 100% stacked bars.
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, "B")
ax.set_title("Valid vs. fallacious updating (%)", loc="left", pad=6)

xc = np.array([0.00, 0.72, 1.85, 2.57])
rates = np.array([70.83, 91.67, 29.17, 45.83])
follow_colors = [MINT, MINT, AMBER, AMBER]
width = 0.52

for i, (xi, rate, color) in enumerate(zip(xc, rates, follow_colors)):
    capped = i in (1, 3)
    # Base fills have no internal gray edges; a white separator and thin outer
    # outline reproduce the reference paper's stacked-bar treatment.
    ax.bar(
        xi,
        rate,
        width=width,
        color=color,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    ax.bar(
        xi,
        100 - rate,
        bottom=rate,
        width=width,
        color=SAND,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    if capped:
        # Matplotlib ties hatch color to patch edge color. Draw a dedicated
        # transparent white-hatch layer, then restore the neutral outer edge.
        ax.bar(
            xi,
            rate,
            width=width,
            color="none",
            edgecolor="white",
            linewidth=0,
            hatch="//",
            zorder=3,
        )
        ax.bar(
            xi,
            100 - rate,
            bottom=rate,
            width=width,
            color="none",
            edgecolor="white",
            linewidth=0,
            hatch="//",
            zorder=3,
        )
    ax.add_patch(
        Rectangle(
            (xi - width / 2, 0),
            width,
            100,
            facecolor="none",
            edgecolor=OUTLINE,
            linewidth=0.55,
            zorder=4,
        )
    )
    ax.plot(
        [xi - width / 2, xi + width / 2],
        [rate, rate],
        color="white",
        lw=0.65,
        zorder=4,
    )
    ax.text(
        xi,
        rate / 2,
        f"{rate:.1f}",
        ha="center",
        va="center",
        fontproperties=NUMERIC_FONT,
        zorder=5,
    )

def change_bracket(axis, x0, x1, y, text):
    axis.plot([x0, x0, x1, x1], [y - 2.2, y, y, y - 2.2], color=INK, lw=0.75, clip_on=False)
    axis.text((x0 + x1) / 2, y + 1.3, text, ha="center", va="bottom", fontsize=9.0, fontweight="bold")


change_bracket(ax, xc[0], xc[1], 104.0, "+20.8 pp")
change_bracket(ax, xc[2], xc[3], 104.0, "+16.7 pp")

style_bar_axis(ax, "", (0, 113), [0, 25, 50, 75, 100])
ax.set_xlim(-0.48, 3.05)
ax.set_xticks(xc, ["V\nUncapped", "V\np25", "F\nUncapped", "F\np25"])
ax.axvline(1.285, color=GRID, lw=0.6, zorder=0)


# D — Cross-model slots. The cards contain no bar height or result direction.
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, "D")
ax.set_title("Matched cross-model transport", loc="left", pad=6)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

models = [(0.04, "Qwen3-32B"), (0.53, "Llama-3.3-70B")]
for left, model in models:
    card = FancyBboxPatch(
        (left, 0.16),
        0.43,
        0.70,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=CARD_BG,
        edgecolor=GRID,
        linewidth=0.8,
    )
    ax.add_patch(card)
    ax.text(left + 0.215, 0.76, model, ha="center", va="center", fontsize=9.0, fontweight="bold")
    ax.plot([left + 0.035, left + 0.395], [0.68, 0.68], color=GRID, lw=0.7)
    for y, metric in [(0.54, r"$\Delta m_{\mathrm{general}}$"), (0.33, r"$\Delta(V{-}F)$")]:
        ax.text(left + 0.035, y, metric, ha="left", va="center", fontsize=9.0, color=MID)
        pill = FancyBboxPatch(
            (left + 0.275, y - 0.055),
            0.115,
            0.11,
            boxstyle="round,pad=0.006,rounding_size=0.02",
            facecolor="white",
            edgecolor=OUTLINE,
            linewidth=0.65,
            linestyle=(0, (2, 2)),
        )
        ax.add_patch(pill)
        ax.text(left + 0.3325, y, "TBD", ha="center", va="center", fontsize=9.0, color=MID, fontweight="bold")

fig.savefig(HERE / "result-bars-paper-preview.png", dpi=320, facecolor="white")
fig.savefig(HERE / "result-bars-paper-vector.pdf", facecolor="white")
fig.savefig(HERE / "result-bars-paper-editable.svg", facecolor="white")
plt.close(fig)
