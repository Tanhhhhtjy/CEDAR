"""Create the supplementary summary figure from frozen, audited estimates.

The values below are copied from the machine-readable analysis artifacts named
in the technical supplement.  This script intentionally performs no model
inference and does not depend on the private data archive.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "supplement-results.pdf"
OUT_PNG = ROOT / "supplement-results.png"

NAVY = "#1F2E40"
BLUE = "#2F6FAE"
TEAL = "#0E8C7A"
ORANGE = "#DC542A"
GREY = "#6B7785"
LIGHT = "#E8EDF2"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def panel_label(ax, label):
    ax.text(
        -0.11,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.15), dpi=300)
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.95, wspace=0.31, hspace=0.42)

# A. Source-style signed steering calibration.
ax = axs[0, 0]
alpha = np.array([-200, -100, -50, 0, 50, 100, 200])
delta = np.array([-1.675694, -0.999861, -0.378333, 0, -0.403958, -1.174792, -2.431181])
lo = np.array([-1.850612, -1.128332, -0.450029, 0, -0.480944, -1.303832, -2.624792])
hi = np.array([-1.497066, -0.871659, -0.304348, 0, -0.324437, -1.043414, -2.240682])
ax.fill_between(alpha, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
ax.plot(alpha, delta, "-o", color=BLUE, linewidth=1.8, markersize=4.2)
ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
ax.set_title("Signed steering is non-monotone in direction")
ax.set_xlabel(r"addition coefficient $\alpha$")
ax.set_ylabel(r"$\Delta$ sycophancy margin (logits)")
ax.set_xticks(alpha)
clean(ax)
panel_label(ax, "A")

# B. General displacement in the development and new-salt banks.
ax = axs[0, 1]
arms = ["N", "R", "V", "F"]
x = np.arange(4)
dev = np.array([2.215, 2.092, 2.950, 5.264])
dev_lo = np.array([1.257, 1.464, 0.300, 1.889])
dev_hi = np.array([3.156, 2.839, 5.535, 9.005])
salt = np.array([2.244, 1.736, -0.658, 2.848])
salt_lo = np.array([1.550, 1.265, -2.439, -1.284])
salt_hi = np.array([2.926, 2.247, 0.929, 7.212])
offset = 0.13
ax.errorbar(
    x - offset,
    dev,
    yerr=[dev - dev_lo, dev_hi - dev],
    fmt="o",
    color=TEAL,
    capsize=2.5,
    linewidth=1.2,
    label="development",
)
ax.errorbar(
    x + offset,
    salt,
    yerr=[salt - salt_lo, salt_hi - salt],
    fmt="D",
    color=ORANGE,
    capsize=2.5,
    linewidth=1.2,
    markersize=4,
    label="new salt",
)
ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
ax.set_xticks(x, arms)
ax.set_title("The pooled shift recurs; arm allocation varies")
ax.set_ylabel(r"$\Delta m_{\mathrm{general}}$ (logits)")
ax.legend(frameon=False, ncol=2, loc="upper left")
clean(ax)
panel_label(ax, "B")

# C. Social target exposure and social framing.
ax = axs[1, 0]
labels = ["AO\n(target only)", "S\n(social)", "S − AO"]
vals = np.array([1.267917, 0.481667, -0.786250])
cis = np.array([[1.205625, 1.332083], [0.411250, 0.552917], [-0.860000, -0.711042]])
colors = [GREY, ORANGE, BLUE]
bars = ax.bar(np.arange(3), vals, width=0.62, color=colors, alpha=0.84, zorder=2)
ax.errorbar(
    np.arange(3),
    vals,
    yerr=[vals - cis[:, 0], cis[:, 1] - vals],
    fmt="none",
    ecolor=NAVY,
    capsize=3,
    linewidth=1.0,
    zorder=3,
)
ax.axhline(0, color=GREY, linewidth=0.8)
ax.set_xticks(np.arange(3), labels)
ax.set_title("Social framing attenuates a positive target shift")
ax.set_ylabel(r"$\Delta$ target margin (logits)")
for rect, value, ci in zip(bars, vals, cis):
    if value >= 0:
        y, va = ci[1] + 0.075, "bottom"
    else:
        y, va = ci[0] - 0.075, "top"
    ax.text(rect.get_x() + rect.get_width() / 2, y, f"{value:+.3f}",
            ha="center", va=va, fontsize=7.2, color=NAVY)
ax.set_ylim(-1.18, 1.58)
clean(ax)
panel_label(ax, "C")

# D. Fixed-trace replay: broad displacement and V-F selectivity.
ax = axs[1, 1]
groups = ["uncapped-source\ntrace", "p25-source\ntrace"]
broad = np.array([2.078, 0.700])
broad_ci = np.array([[1.759, 2.408], [0.487, 0.907]])
vf = np.array([-2.422, -0.797])
vf_ci = np.array([[-3.411, -1.448], [-1.219, -0.391]])
x = np.arange(2)
ax.errorbar(
    x - offset,
    broad,
    yerr=[broad - broad_ci[:, 0], broad_ci[:, 1] - broad],
    fmt="o",
    color=BLUE,
    capsize=3,
    linewidth=1.3,
    label=r"$\Delta m_{\mathrm{general}}$",
)
ax.errorbar(
    x + offset,
    vf,
    yerr=[vf - vf_ci[:, 0], vf_ci[:, 1] - vf],
    fmt="D",
    color=ORANGE,
    capsize=3,
    linewidth=1.3,
    markersize=4,
    label=r"$\Delta(V-F)$",
)
ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
ax.set_xticks(x, groups)
ax.set_title("The shift survives token-identical replay")
ax.set_ylabel("capped − uncapped (logits)")
ax.legend(frameon=False, loc="upper right")
clean(ax)
panel_label(ax, "D")

fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {OUT_PDF}")
print(f"wrote {OUT_PNG}")
