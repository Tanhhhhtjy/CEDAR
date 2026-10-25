# -*- coding: utf-8 -*-
"""Supplementary figure: family-level effects and LOFO stability (S4)."""
import json, io, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(r"<local-path>")

def signp(pos, n):
    k = max(pos, n - pos)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n)

E2A = json.load(io.open("<project-root>/artifacts/m2/e2_phase_validity_choice/E2A_ANALYSIS.json", encoding="utf-8"))
DEV = json.load(io.open("<project-root>/artifacts/m2/prover_generator_evidence_validation_v1_1/EVIDENCE_VALIDATION_ANALYSIS.json", encoding="utf-8"))
NS = json.load(io.open("<project-root>/artifacts/m2/e2_new_salt_confirmation/E2_NEW_SALT_ANALYSIS.json", encoding="utf-8"))
G = E2A["contrasts_vs_uncapped"]["geometry"]

rows = []
def add(label, b):
    v = np.array(list(b["family_scores"].values()), float)
    n = len(v); outs = [(v.sum() - v[i]) / (n - 1) for i in range(n)]
    pos = int((v > 0).sum())
    rows.append(dict(label=label, est=b["estimate"], ci=b["family_cluster_bootstrap_95ci"],
                     lofo=(min(outs), max(outs)), pos=pos, neg=int((v < 0).sum()),
                     p=signp(pos, n), scores=v))

add("Development pooled", G["full_p25"]["m_general"])
for a in ["N", "R", "V", "F"]:
    add("Development " + a, G["full_p25"]["m_general_by_arm"][a])
add("New salt pooled", NS["primary"])
add("Development $V-F$", DEV["margin_discrimination_change_complete_cases"])

for r in rows:
    print("%-24s %7.3f [%6.3f,%6.3f] LOFO [%6.3f,%6.3f] +%2d/-%2d p=%.4f"
          % (r["label"], r["est"], r["ci"][0], r["ci"][1], r["lofo"][0], r["lofo"][1],
             r["pos"], r["neg"], r["p"]))

NAVY, BLUE, TEAL, ORANGE, GREY, LIGHT = "#1F2E40", "#2F6FAE", "#0E8C7A", "#DC542A", "#6B7785", "#E8EDF2"
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "axes.edgecolor": NAVY, "xtick.color": NAVY,
    "ytick.color": NAVY, "text.color": NAVY, "axes.labelcolor": NAVY,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

def style(ax, title):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=LIGHT, lw=0.7, axis="x"); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=NAVY, pad=6)

sel = rows[::-1]
SHORT = {"Development pooled": "pooled", "Development N": "arm $N$", "Development R": "arm $R$",
         "Development V": "arm $V$", "Development F": "arm $F$", "New salt pooled": "new salt",
         "Development $V-F$": "$V-F$"}
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), dpi=300, sharey=True,
                        gridspec_kw={"width_ratios": [1.30, 1.0]})
plt.subplots_adjust(left=0.215, right=0.985, top=0.84, bottom=0.20, wspace=0.22)

ax = axs[0]
style(ax, "A. Family effects behind each estimate")
for i, r in enumerate(sel):
    ax.scatter(r["scores"], np.full(len(r["scores"]), i), s=8, color=GREY,
               alpha=0.85, zorder=2, linewidths=0)
    ax.plot(r["ci"], [i, i], color=BLUE, lw=1.6, solid_capstyle="round", zorder=3)
    ax.plot([r["est"]], [i], "o", color=BLUE, ms=4.2, zorder=4)
ax.axvline(0, color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(range(len(sel))); ax.set_yticklabels([r["label"] for r in sel])
ax.set_ylim(-0.7, len(sel) - 0.3)
ax.set_xlabel("logits, p25 minus uncapped")

ax = axs[1]
style(ax, "B. Leave-one-family-out range")
for i, r in enumerate(sel):
    ax.plot(r["lofo"], [i, i], color=BLUE, lw=4.0, solid_capstyle="butt", zorder=3)
    ax.plot([r["est"]], [i], "|", color=NAVY, ms=8, mew=1.3, zorder=4)
ax.axvline(0, color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(range(len(sel)))
ax.set_yticklabels([SHORT[r["label"]] for r in sel], fontsize=7.5)
ax.tick_params(axis="y", labelleft=True, left=True)
ax.set_ylim(-0.7, len(sel) - 0.3)
ax.set_xlabel("logits")

fig.savefig("<project-root>/figures/supplement-family.pdf",
            bbox_inches="tight", pad_inches=0.02)
fig.savefig("<project-root>/figures/supplement-family.png",
            dpi=300, bbox_inches="tight", pad_inches=0.02)
print("S4 written")
