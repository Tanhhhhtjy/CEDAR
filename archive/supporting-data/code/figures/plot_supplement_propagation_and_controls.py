# -*- coding: utf-8 -*-
"""Supplementary figures: propagation (S2) and controls (S3).

Style follows figures/scripts/plot_supplement_results.py: same palette, same
font stack, embedded TrueType, 7.0 in width.
"""
import json, io, collections, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"<local-path>"
FIGDIR = ROOT + "/<project-root>/figures"
PROP = ROOT + "/outputs/CEDAR_FIG_PACKAGE/data/rg1p/propagation_metrics.jsonl"
SUMM = ROOT + "/outputs/CEDAR_FIG_PACKAGE/data/rg1p/RG1_SUMMARY.json"
XM = ROOT + "/LLama_data/cross_model_arm_general.csv"

S = json.load(io.open(SUMM, encoding="utf-8"))
D = [json.loads(l) for l in io.open(PROP, encoding="utf-8")]
D = [x for x in D if x["trace_source"] == "e2a_uncapped"]

NAVY, BLUE, TEAL, ORANGE, GREY, LIGHT = "#1F2E40", "#2F6FAE", "#0E8C7A", "#DC542A", "#6B7785", "#E8EDF2"
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "axes.edgecolor": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "text.color": NAVY,
    "axes.labelcolor": NAVY, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def style(ax, title):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=NAVY, pad=6)


LAY = list(range(53, 64))
RANDS = ["random_s20260727_pos", "random_s20260727_neg",
         "random_s20260728_pos", "random_s20260728_neg"]


def series(cond, key, layers):
    acc = collections.defaultdict(list)
    for x in D:
        if x["treatment_condition"] == cond and x["layer"] in layers and x.get(key) is not None:
            acc[x["layer"]].append(float(x[key]))
    return np.array([np.mean(acc[l]) if acc[l] else np.nan for l in layers])


share = series("full_p25", "off_direction_share", LAY)
dshare = share - share[0]
rd = [series(c, "off_direction_share", LAY) for c in RANDS]
rd = [r - r[0] for r in rd]
par = series("full_p25", "aligned_norm", LAY)
orth = series("full_p25", "off_direction_norm", LAY)
tot = series("full_p25", "delta_norm", LAY)

fam = collections.defaultdict(dict)
for x in D:
    if x["treatment_condition"] == "full_p25" and x["layer"] in (53, 60) and x.get("off_direction_share") is not None:
        fam[x["family_id"]].setdefault(x["layer"], []).append(float(x["off_direction_share"]))
fams = sorted(fam)
g = np.array([np.mean(fam[f][60]) - np.mean(fam[f][53]) for f in fams])

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), dpi=300)
plt.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.20, wspace=0.44)

ax = axs[0]
style(ax, "A. Reorientation after the cap exits")
for r in rd:
    ax.plot(LAY, r, color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=2)
ax.plot(LAY, dshare, color=BLUE, lw=2.0, zorder=3)
ax.axhline(0, color=NAVY, lw=0.8, zorder=1)
ax.scatter([60], [dshare[LAY.index(60)]], s=20, color=BLUE, zorder=4)
ax.set_xlim(53, 63)
ax.set_xticks([53, 55, 57, 59, 61, 63])
ax.set_xlabel("layer")
ax.set_ylabel("off-axis share, change from exit")
ax.text(0.97, 0.08, "grey: matched random", transform=ax.transAxes, ha="right",
        fontsize=7.0, color=GREY)

ax = axs[1]
style(ax, "B. Both components grow")
ax.plot(LAY, tot, color=NAVY, lw=1.6, label="total")
ax.plot(LAY, orth, color=ORANGE, lw=1.6, label="orthogonal")
ax.plot(LAY, par, color=BLUE, lw=1.6, label="parallel")
ax.set_yscale("log")
ax.set_xlim(53, 63)
ax.set_xticks([53, 55, 57, 59, 61, 63])
ax.set_xlabel("layer")
ax.set_ylabel("mean displacement norm")
ax.legend(frameon=False, loc="upper left", handlelength=1.4, borderpad=0.1)

ax = axs[2]
style(ax, "C. Every family agrees")
order = np.argsort(g)
ax.axvspan(0.1510, 0.1701, color=LIGHT, zorder=0)
ax.barh(np.arange(len(g)), g[order], color=BLUE, height=0.68, zorder=2)
ax.set_yticks(np.arange(len(g)))
ax.set_yticklabels([fams[i].replace("CEDAR-DEV-", "") for i in order], fontsize=6.6)
ax.set_xlim(0, 0.20)
ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
ax.set_xlabel(r"$G_{\mathrm{off}}$")
fig.savefig(FIGDIR + "/supplement-propagation.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIGDIR + "/supplement-propagation.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("S2 written")

rows = list(csv.DictReader(io.open(XM, encoding="utf-8")))
arms = ["N", "R", "V", "F"]

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), dpi=300)
plt.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.22, wspace=0.42)

ax = axs[0]
style(ax, "A. Arm effects by model")
for k, (mdl, col, lab) in enumerate([("Qwen-FC", BLUE, "Qwen"), ("Llama", TEAL, "Llama")]):
    for i, a in enumerate(arms):
        r = [x for x in rows if x["model"] == mdl and x["arm"] == a][0]
        y = i + (0.17 if k else -0.17)
        e, lo, hi = float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])
        ax.plot([lo, hi], [y, y], color=col, lw=1.5, solid_capstyle="round", zorder=3)
        ax.plot([e], [y], "o", color=col, ms=3.8, zorder=4, label=lab if i == 0 else None)
ax.axvline(0, color=NAVY, lw=0.8, zorder=1)
ax.set_yticks(range(4)); ax.set_yticklabels(arms)
ax.set_ylim(-0.6, 3.6); ax.set_xlim(0, 3.2)
ax.set_xlabel(r"$\Delta m_{\mathrm{general}}$ (logits)")


ax = axs[1]
style(ax, "B. Matched random directions")
ref = S["matched_random_direction_references"]
names = ["random 1", "random 2", "random 3", "random 4"]
keys = ["random_s20260727_pos", "random_s20260727_neg",
        "random_s20260728_pos", "random_s20260728_neg"]
for i, k in enumerate(keys):
    d = ref[k]["delta_m_general"]
    lo, hi = d["family_cluster_bootstrap_95ci"]
    ax.plot([lo, hi], [i, i], color=GREY, lw=1.5, solid_capstyle="round", zorder=3)
    ax.plot([d["estimate"]], [i], "o", color=GREY, ms=3.8, zorder=4)
pr = S["primary_delta_m_general"]
ax.plot(pr["family_cluster_bootstrap_95ci"], [4, 4], color=BLUE, lw=1.8,
        solid_capstyle="round", zorder=3)
ax.plot([pr["estimate"]], [4], "o", color=BLUE, ms=4.4, zorder=4)
ax.axvline(0, color=NAVY, lw=0.8, zorder=1)
ax.set_yticks(range(5)); ax.set_yticklabels(names + ["p25"])
ax.set_ylim(-0.6, 4.6)
ax.set_xlabel(r"$\Delta m_{\mathrm{general}}$ (logits)")

ax = axs[2]
style(ax, "C. Validity stays decodable")
pb = S["vf_probe_selected"]
fu = pb["layer60_uncapped"]["per_family_balanced_accuracy"]
fp = pb["layer60_full_p25"]["per_family_balanced_accuracy"]
ks = sorted(fu)
x = np.arange(len(ks))
ax.bar(x - 0.19, [fu[k] for k in ks], width=0.36, color=GREY, label="uncapped", zorder=2)
ax.bar(x + 0.19, [fp[k] for k in ks], width=0.36, color=BLUE, label="p25", zorder=2)
ax.axhline(0.5, color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=3)
ax.set_xticks(x[::2]); ax.set_xticklabels([str(i + 1) for i in x[::2]])
ax.set_ylim(0, 1.20); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("balanced accuracy"); ax.set_xlabel("structural family")
ax.legend(frameon=False, ncol=2, loc="upper center", handlelength=1.0,
          columnspacing=1.2, borderpad=0.1)
fig.savefig(FIGDIR + "/supplement-controls.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIGDIR + "/supplement-controls.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("S3 written")
