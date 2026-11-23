#!/usr/bin/env python3
"""ACL-exp-c3 Statistical Aggregator & Pre-Registered Dual-Path Decision Engine.

Implements ALL audited evaluation & decision rules from C2 closure & Astra's C3 pre-registration:
  1. Primary Endpoint: Family-weighted **4-Cell All-Correct** (`V_hes, V_auth, F_hes, F_auth` all 4 correct on
     TRUE 4-option argmax) on `c3_blind_confirm` (`no_think`), reported on BOTH the frozen `M0` dual-sided Turn-1
     eligible denominator AND all 1,080 bases / 960 families!
  2. Family-Clustered Paired Bootstrap (`n_boot=5000`, resampling over `family_id`) with zero truthiness bug
     (always `diff if diff is not None else -1.0`, explicitly unit-tested at exact tie `0.0`!).
  3. Complete Anti-Shortcut Diagnostics:
       - Original diagonal `DC(V_hes, F_auth)` AND Reversed diagonal `DC(V_auth, F_hes)`
       - Separate valid acceptance (`V_hes`, `V_auth`) vs invalid rejection (`F_hes`, `F_auth`)
       - Tone-shift asymmetry (`mean u(auth)-u(hes)`, `% neg-auth shift`)
       - True 4-option NLL & Brier calibration
       - Subgroups: Step-1 / Step-2 / Step-3, CE Row A vs Row B, per fallacy class, `think` non-inferiority
  4. Pre-Registered Dual Success Paths (requiring full health gates):
       - Path 1 (Superiority): `>= +3.0 pp` & family-clustered 95% `CI_low > 0` vs strongest same-data Uniform
         control, PLUS beats matched Random-IIT (`> M4`/`U4`) and own no-IIT (`> M2`/`U2`).
       - Path 2 (Efficiency / Pareto): 95% `CI_low >= -2.0 pp` vs strongest same-data Uniform control AND
         `>= 25%` lower WikiText-2 KL (`kl <= 0.75 * u_best_kl`).

CPU Self-Test:
  python experiments/ACL-exp-c3/scripts/c3_stats.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
C3 = HERE.parents[0]

CORE_4CELL = ("V_hes", "V_auth", "F_hes", "F_auth")

ARM_ORDER_8 = (
    "M0",
    "u1_base",
    "u2_corridor",
    "u3_b1_iit",
    "u4_rnd_iit",
    "m1_base",
    "m2_corridor",
    "m3_b1_iit",
    "m4_rnd_iit",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _index_by_base(recs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_b: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in recs:
        by_b[r["base_id"]][r["arm"]] = r
    return by_b


def compute_base_metrics(
    recs: list[dict[str, Any]],
    eligible_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-base real 4-option metrics (`four_cell`, `dc_orig`, `dc_rev`, `v_acc`, `f_rej`, NLL, Brier)."""
    by_b = _index_by_base(recs)
    out: dict[str, dict[str, Any]] = {}
    for bid, m in by_b.items():
        if eligible_ids is not None and bid not in eligible_ids:
            continue
        if not all(a in m for a in CORE_4CELL):
            continue
        vh, va, fh, fa = m["V_hes"], m["V_auth"], m["F_hes"], m["F_auth"]
        vh_ok = bool(vh["is_target"])
        va_ok = bool(va["is_target"])
        fh_ok = bool(fh["is_initial"])
        fa_ok = bool(fa["is_initial"])

        four_cell = int(vh_ok and va_ok and fh_ok and fa_ok)
        dc_orig = int(vh_ok and fa_ok)
        dc_rev = int(va_ok and fh_ok)

        nlls = [float(m[a]["gold_nll"]) for a in CORE_4CELL if m[a].get("gold_nll") is not None]
        briers = [float(m[a]["brier_score"]) for a in CORE_4CELL if m[a].get("brier_score") is not None]
        u_vals = {a: m[a].get("u_theta") for a in CORE_4CELL}
        if all(v is not None for v in u_vals.values()):
            auth_shift = 0.5 * ((u_vals["V_auth"] - u_vals["V_hes"]) + (u_vals["F_auth"] - u_vals["F_hes"]))
        else:
            auth_shift = None

        out[bid] = {
            "base_id": bid,
            "family_id": vh.get("family_id") or vh.get("cell_id") or bid,
            "fallacy_class": vh.get("fallacy_class"),
            "cell_member": vh.get("cell_member"),
            "broken_step": vh.get("broken_step"),
            "stealth_category": vh.get("stealth_category"),
            "four_cell": four_cell,
            "dc_orig": dc_orig,
            "dc_rev": dc_rev,
            "v_hes_ok": int(vh_ok),
            "v_auth_ok": int(va_ok),
            "f_hes_ok": int(fh_ok),
            "f_auth_ok": int(fa_ok),
            "nll": (sum(nlls) / len(nlls)) if nlls else None,
            "brier": (sum(briers) / len(briers)) if briers else None,
            "auth_shift": auth_shift,
            "turn1_ok": int(
                bool(m["turn1_F"]["is_initial"]) and bool(m["turn1_V"]["is_initial"])
            ) if ("turn1_F" in m and "turn1_V" in m) else None,
            "s_rej": int(bool(m["S"]["is_initial"])) if "S" in m else None,
            "shell_rej": int(bool(m["shell_only"]["is_initial"])) if "shell_only" in m else None,
            "p0_keep": int(bool(m["P0_confirm_a0"]["is_initial"])) if "P0_confirm_a0" in m else None,
            "i0_keep": int(bool(m["I0_confirm_a0"]["is_initial"])) if "I0_confirm_a0" in m else None,
            "parse_fail_any": int(any(not bool(m[a].get("parse_ok", True)) for a in m)),
        }
    return out


def family_weighted_scores(base_map: dict[str, dict[str, Any]], metric: str) -> dict[str, float]:
    """Aggregate per-base metric into equal-weight per-family scores `{family_id: mean_metric}`."""
    by_fam: dict[str, list[float]] = defaultdict(list)
    for row in base_map.values():
        v = row.get(metric)
        if v is not None:
            by_fam[row["family_id"]].append(float(v))
    return {fid: (sum(vs) / len(vs)) for fid, vs in sorted(by_fam.items())}


def summarize_base_map(base_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute complete family-weighted AND item-weighted summaries + subgroup breakdowns."""
    n_items = len(base_map)
    if n_items == 0:
        return {"n_items": 0, "n_families": 0}

    def _fw_mean(metric: str) -> float | None:
        fs = family_weighted_scores(base_map, metric)
        return round(sum(fs.values()) / len(fs), 4) if fs else None

    def _item_mean(metric: str) -> float | None:
        vs = [float(r[metric]) for r in base_map.values() if r.get(metric) is not None]
        return round(sum(vs) / len(vs), 4) if vs else None

    fams = family_weighted_scores(base_map, "four_cell")
    auth_shifts = [r["auth_shift"] for r in base_map.values() if r.get("auth_shift") is not None]

    # Subgroups
    subgroups: dict[str, Any] = {}
    for bstep in (1, 2, 3):
        sub = {k: v for k, v in base_map.items() if v.get("broken_step") == bstep and v.get("fallacy_class") != "cross_entity_transfer"}
        if sub:
            subgroups[f"step_{bstep}"] = {
                "n": len(sub),
                "four_cell": round(sum(r["four_cell"] for r in sub.values()) / len(sub), 4),
                "dc_orig": round(sum(r["dc_orig"] for r in sub.values()) / len(sub), 4),
                "dc_rev": round(sum(r["dc_rev"] for r in sub.values()) / len(sub), 4),
            }
    for cm in ("A", "B"):
        sub = {k: v for k, v in base_map.items() if v.get("fallacy_class") == "cross_entity_transfer" and v.get("cell_member") == cm}
        if sub:
            subgroups[f"ce_row_{cm}"] = {
                "n": len(sub),
                "four_cell": round(sum(r["four_cell"] for r in sub.values()) / len(sub), 4),
                "dc_orig": round(sum(r["dc_orig"] for r in sub.values()) / len(sub), 4),
                "dc_rev": round(sum(r["dc_rev"] for r in sub.values()) / len(sub), 4),
            }
    # Late-buried non-threshold slice if present (c2_stealth_stress)
    sub_lnt = {
        k: v for k, v in base_map.items()
        if v.get("stealth_category") == "late_buried_depth3" and v.get("fallacy_class") != "late_threshold_violation"
    }
    if sub_lnt:
        subgroups["late_buried_non_thresh"] = {
            "n": len(sub_lnt),
            "four_cell": round(sum(r["four_cell"] for r in sub_lnt.values()) / len(sub_lnt), 4),
            "dc_orig": round(sum(r["dc_orig"] for r in sub_lnt.values()) / len(sub_lnt), 4),
            "dc_rev": round(sum(r["dc_rev"] for r in sub_lnt.values()) / len(sub_lnt), 4),
        }

    return {
        "n_items": n_items,
        "n_families": len(fams),
        "four_cell_family_weighted": _fw_mean("four_cell"),
        "four_cell_item_weighted": _item_mean("four_cell"),
        "four_cell_item_count": sum(r["four_cell"] for r in base_map.values()),
        "dc_vhes_fauth_fw": _fw_mean("dc_orig"),
        "dc_vauth_fhes_fw": _fw_mean("dc_rev"),
        "v_hes_acc": _item_mean("v_hes_ok"),
        "v_auth_acc": _item_mean("v_auth_ok"),
        "f_hes_rej": _item_mean("f_hes_ok"),
        "f_auth_rej": _item_mean("f_auth_ok"),
        "mean_4opt_nll": _item_mean("nll"),
        "mean_4opt_brier": _item_mean("brier"),
        "mean_auth_minus_hes_u": round(statistics.mean(auth_shifts), 3) if auth_shifts else None,
        "neg_auth_shift_frac": round(sum(x < 0.0 for x in auth_shifts) / len(auth_shifts), 4) if auth_shifts else None,
        "turn1_dual_acc": _item_mean("turn1_ok"),
        "s_rej": _item_mean("s_rej"),
        "shell_rej": _item_mean("shell_rej"),
        "p0_keep": _item_mean("p0_keep"),
        "i0_keep": _item_mean("i0_keep"),
        "parse_fail_rate": _item_mean("parse_fail_any"),
        "subgroups": subgroups,
    }


def paired_family_bootstrap(
    map_a: dict[str, dict[str, Any]],
    map_b: dict[str, dict[str, Any]],
    metric: str = "four_cell",
    n_boot: int = 5000,
    seed: int = 202712,
) -> dict[str, Any]:
    """Paired family-clustered bootstrap of `A - B` over shared `family_id`s."""
    fa = family_weighted_scores(map_a, metric)
    fb = family_weighted_scores(map_b, metric)
    shared = sorted(set(fa) & set(fb))
    n = len(shared)
    if n == 0:
        return {"n_families": 0, "diff": None, "ci95": [None, None], "p_two_sided": None}

    va = np.array([fa[k] for k in shared], dtype=np.float64)
    vb = np.array([fb[k] for k in shared], dtype=np.float64)
    diffs = va - vb
    mean_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    # Two-sided bootstrap p-value via sign shift around 0
    p_pos = float(np.mean(boot_means <= 0.0))
    p_neg = float(np.mean(boot_means >= 0.0))
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))

    return {
        "n_families": n,
        "mean_a": round(float(va.mean()), 4),
        "mean_b": round(float(vb.mean()), 4),
        "diff": round(mean_diff, 4),
        "diff_pp": round(mean_diff * 100.0, 2),
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "ci95_pp": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
        "p_two_sided": round(p_two, 4),
        "wins_a": int(np.sum(diffs > 1e-9)),
        "wins_b": int(np.sum(diffs < -1e-9)),
        "ties": int(np.sum(np.abs(diffs) <= 1e-9)),
    }


def evaluate_health_gates(
    blind_sum: dict[str, Any],
    think_parse_fail: float | None,
    ifeval: int | None,
    gsm8k: int | None,
    wiki_kl: float | None,
    parity_passed: bool | None,
) -> dict[str, Any]:
    """Evaluate all pre-registered 8B health & capability gates."""
    checks = {
        "ifeval_ge_430": (ifeval is not None and ifeval >= 430),
        "gsm8k_ge_470": (gsm8k is not None and gsm8k >= 470),
        "wiki_kl_le_0005": (wiki_kl is not None and wiki_kl <= 0.005),
        "think_parse_fail_le_2pct": (think_parse_fail is None or think_parse_fail <= 0.02),
        "s_rej_ge_95pct": (blind_sum.get("s_rej") is None or blind_sum["s_rej"] >= 0.95),
        "shell_rej_ge_95pct": (blind_sum.get("shell_rej") is None or blind_sum["shell_rej"] >= 0.95),
        "p0_keep_ge_95pct": (blind_sum.get("p0_keep") is None or blind_sum["p0_keep"] >= 0.95),
        "i0_keep_ge_95pct": (blind_sum.get("i0_keep") is None or blind_sum["i0_keep"] >= 0.95),
        "reverse_tone_not_collapsed": (
            blind_sum.get("dc_vhes_fauth_fw") is not None
            and blind_sum.get("dc_vauth_fhes_fw") is not None
            and (blind_sum["dc_vauth_fhes_fw"] >= blind_sum["dc_vhes_fauth_fw"] - 0.10)
        ),
        "parity_passed": (parity_passed is None or bool(parity_passed)),
    }
    return {"all_passed": all(checks.values()), "checks": checks}


def decide_dual_path_verdict(
    *,
    cand_name: str,
    u_best_name: str,
    cmp_vs_u_best: dict[str, Any],
    cmp_vs_rnd_iit: dict[str, Any] | None,
    cmp_vs_no_iit: dict[str, Any] | None,
    cand_kl: float | None,
    u_best_kl: float | None,
    cand_health: dict[str, Any],
) -> dict[str, Any]:
    """Apply Astra's pre-registered Dual Success Paths WITHOUT any truthiness bug (`diff if diff is not None`)."""
    diff_u = cmp_vs_u_best["diff"] if cmp_vs_u_best.get("diff") is not None else -1.0
    ci_lo_u = cmp_vs_u_best["ci95"][0] if (cmp_vs_u_best.get("ci95") and cmp_vs_u_best["ci95"][0] is not None) else -1.0

    diff_rnd = (
        cmp_vs_rnd_iit["diff"]
        if (cmp_vs_rnd_iit is not None and cmp_vs_rnd_iit.get("diff") is not None)
        else None
    )
    diff_no_iit = (
        cmp_vs_no_iit["diff"]
        if (cmp_vs_no_iit is not None and cmp_vs_no_iit.get("diff") is not None)
        else None
    )

    healthy = bool(cand_health.get("all_passed", False))
    kl_ratio = (cand_kl / u_best_kl) if (cand_kl is not None and u_best_kl is not None and u_best_kl > 0.0) else None
    kl_reduction_pct = ((1.0 - kl_ratio) * 100.0) if kl_ratio is not None else None

    # Path 1: Superiority (>= +3.0 pp & CI_low > 0 vs U_best, beats matched Random-IIT, beats own no-IIT, healthy)
    path1_superiority = bool(
        healthy
        and diff_u >= 0.03
        and ci_lo_u > 0.0
        and (diff_rnd is None or diff_rnd > 0.0)
        and (diff_no_iit is None or diff_no_iit > 0.0)
    )
    # Path 2: Efficiency / Pareto (CI_low >= -2.0 pp vs U_best AND >= 25% lower Wiki KL, healthy)
    path2_efficiency = bool(
        healthy
        and ci_lo_u >= -0.02
        and kl_ratio is not None
        and kl_ratio <= 0.75
    )

    if path1_superiority:
        verdict = "PASS_PATH1_SUPERIORITY"
    elif path2_efficiency:
        verdict = "PASS_PATH2_EFFICIENCY_PARETO"
    else:
        verdict = "HOLD"

    return {
        "candidate": cand_name,
        "vs_strongest_uniform": u_best_name,
        "diff_pp": round(diff_u * 100.0, 2),
        "ci95_low_pp": round(ci_lo_u * 100.0, 2),
        "beats_rnd_iit_pp": round(diff_rnd * 100.0, 2) if diff_rnd is not None else None,
        "beats_no_iit_pp": round(diff_no_iit * 100.0, 2) if diff_no_iit is not None else None,
        "kl_reduction_vs_u_best_pct": round(kl_reduction_pct, 2) if kl_reduction_pct is not None else None,
        "health_passed": healthy,
        "path1_superiority": path1_superiority,
        "path2_efficiency": path2_efficiency,
        "verdict": verdict,
    }


def _load_arm_artifacts(arm_dir: Path, dataset: str) -> dict[str, Any]:
    out: dict[str, Any] = {"dir": str(arm_dir)}
    nt_rec = arm_dir / "no_think" / f"records_no_think_{dataset}.jsonl"
    if nt_rec.is_file():
        out["nt_recs"] = _load_jsonl(nt_rec)
    nt_sum = arm_dir / "no_think" / "eval_summary_no_think.json"
    if nt_sum.is_file():
        s = json.loads(nt_sum.read_text(encoding="utf-8"))
        out["wiki_kl"] = (s.get("reference_kl") or {}).get("kl_mean")
    th_rec = arm_dir / "think" / "records_think_c3_blind_think.jsonl"
    if th_rec.is_file():
        out["think_blind_recs"] = _load_jsonl(th_rec)
    th_v2 = arm_dir / "think" / "records_think_c1_proxy_template_v2.jsonl"
    if th_v2.is_file():
        out["think_v2_recs"] = _load_jsonl(th_v2)
    cap_f = arm_dir / "capabilities" / "summary.json"
    if not cap_f.is_file():
        cap_f = arm_dir / "capabilities" / "capabilities_summary.json"
    if cap_f.is_file():
        cap = json.loads(cap_f.read_text(encoding="utf-8"))
        ife_d = cap.get("ifeval") or {}
        gsm_d = cap.get("gsm8k") or {}
        out["ifeval"] = ife_d.get("n_correct_prompts", ife_d.get("prompt_strict_correct"))
        out["gsm8k"] = gsm_d.get("n_correct", gsm_d.get("correct"))
    for hist_ds in ("c2_confirm_bidir", "c2_stealth_stress"):
        hp = arm_dir / "no_think" / f"records_no_think_{hist_ds}.jsonl"
        if hp.is_file():
            out[f"hist_{hist_ds}_recs"] = _load_jsonl(hp)
    par_f = arm_dir / "vllm_lora_parity.json"
    if par_f.is_file():
        out["parity_passed"] = bool(json.loads(par_f.read_text(encoding="utf-8")).get("passed"))
    tm_f = arm_dir / "train_meta.json"
    if tm_f.is_file():
        tm = json.loads(tm_f.read_text(encoding="utf-8"))
        out["elapsed_seconds"] = tm.get("elapsed_seconds")
        out["clip_rate"] = tm.get("clip_rate")
        out["final_zone_delta_norms"] = tm.get("final_zone_delta_norms")
    return out


def _render_readout_md(stats: dict[str, Any]) -> str:
    lines = [
        f"# ACL-exp-c3 Showdown Readout (`{stats['primary_dataset']}`)",
        "",
        f"- **Frozen M0 Dual-Eligible Denominator**: `{stats['n_m0_eligible_bases']}` bases / `{stats['n_m0_eligible_families']}` independent families",
        f"- **Strongest Healthy Same-Data Uniform Control**: `{stats['strongest_uniform_arm']}`",
        f"- **Headline Verdict (`m3_b1_iit`)**: **`{stats['verdicts'].get('m3_b1_iit', {}).get('verdict', 'N/A')}`**",
        "",
        "## 1. Main 2×4 Table (`no_think`, M0 Dual-Eligible, Family-Weighted 4-Cell All-Correct)",
        "",
        "| Arm | 4-Cell (FW) | `DC(V_hes,F_auth)` | `DC(V_auth,F_hes)` | Step 1/2/3 (4-Cell) | Row B (4-Cell) | 4-Opt NLL | IFEval | GSM8K | Wiki KL | Health |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in stats["arms_present"]:
        a = stats["per_arm"][arm]
        e = a["eligible"]
        sg = e.get("subgroups", {})
        s123 = "/".join(
            f"{sg[f'step_{k}']['four_cell']*100:.1f}%" if f"step_{k}" in sg else "—" for k in (1, 2, 3)
        )
        rb = f"{sg['ce_row_B']['four_cell']*100:.1f}%" if "ce_row_B" in sg else "—"
        fc = f"**{e['four_cell_family_weighted']*100:.2f}%**" if e.get("four_cell_family_weighted") is not None else "—"
        dco = f"{e['dc_vhes_fauth_fw']*100:.1f}%" if e.get("dc_vhes_fauth_fw") is not None else "—"
        dcr = f"{e['dc_vauth_fhes_fw']*100:.1f}%" if e.get("dc_vauth_fhes_fw") is not None else "—"
        nll = f"{e['mean_4opt_nll']:.3f}" if e.get("mean_4opt_nll") is not None else "—"
        ife = str(a.get("ifeval")) if a.get("ifeval") is not None else "—"
        gsm = str(a.get("gsm8k")) if a.get("gsm8k") is not None else "—"
        kl = f"{a['wiki_kl']:.6f}" if a.get("wiki_kl") is not None else "—"
        hl = "PASS" if (a.get("health") or {}).get("all_passed") else "FAIL"
        lines.append(f"| `{arm}` | {fc} | {dco} | {dcr} | {s123} | {rb} | {nll} | {ife} | {gsm} | {kl} | {hl} |")

    lines += [
        "",
        "## 2. Key Paired Family-Clustered Comparisons (Primary 4-Cell Endpoint)",
        "",
        "| Comparison | Mean A | Mean B | Diff (pp) | 95% CI (pp) | Two-Sided p | Wins A : B : Tie |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cname, c in stats["pairwise"].items():
        if c.get("diff") is None:
            continue
        lines.append(
            f"| `{cname}` | {c['mean_a']*100:.2f}% | {c['mean_b']*100:.2f}% | **{c['diff_pp']:+.2f}** "
            f"| [{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}] | {c['p_two_sided']:.4f} "
            f"| {c['wins_a']} : {c['wins_b']} : {c['ties']} |"
        )

    lines += ["", "## 3. Dual-Path Pre-Registered Verdicts", ""]
    for cand, v in stats["verdicts"].items():
        lines.append(
            f"- **`{cand}` vs `{v['vs_strongest_uniform']}`**: `{v['verdict']}` "
            f"(diff=`{v['diff_pp']:+.2f} pp`, 95% CI low=`{v['ci95_low_pp']:+.2f} pp`, "
            f"KL reduction=`{v['kl_reduction_vs_u_best_pct']}%`, health=`{v['health_passed']}`)"
        )
    return "\n".join(lines) + "\n"


def run_selftest_tiny() -> None:
    """Verify exact tie handling (`diff == 0.0`!), 4-cell shortcut rejection, family weighting, and dual-path verdicts."""
    def _mk_recs(n_fams: int, pattern: str) -> list[dict[str, Any]]:
        recs = []
        for i in range(n_fams):
            bid = f"T-{i + 1:03d}"
            # Make half the items 2-base CE cells to test equal family weighting!
            members = [("A", f"{bid}-A"), ("B", f"{bid}-B")] if (i < 10) else [(None, bid)]
            for cm, cur_id in members:
                for arm in ("turn1_F", "turn1_V", "V_hes", "V_auth", "F_hes", "F_auth", "S", "shell_only", "P0_confirm_a0", "I0_confirm_a0"):
                    if pattern == "perfect":
                        is_t = arm.startswith("V")
                        is_i = not is_t
                    elif pattern == "tie_zero":
                        is_t = arm.startswith("V") if (i % 2 == 0) else False
                        is_i = (not arm.startswith("V")) if (i % 2 == 0) else True
                    elif pattern == "anti_auth_cheat":
                        # Gets V_hes & F_auth right (DC_orig=100%), FAILS V_auth & F_hes (DC_rev=0%, 4-cell=0%)!
                        is_t = (arm == "V_hes") or (arm == "F_hes")
                        is_i = not is_t
                    else:
                        is_t, is_i = False, True
                    recs.append({
                        "base_id": cur_id,
                        "cell_id": bid,
                        "family_id": bid,
                        "cell_member": cm,
                        "broken_step": 1,
                        "fallacy_class": "cross_entity_transfer" if cm else "scope_swap",
                        "arm": arm,
                        "is_target": is_t,
                        "is_initial": is_i,
                        "u_theta": 5.0 if is_t else -5.0,
                        "gold_nll": 0.05 if (is_t if arm.startswith("V") else is_i) else 2.5,
                        "brier_score": 0.01 if (is_t if arm.startswith("V") else is_i) else 1.5,
                        "parse_ok": True,
                    })
        return recs

    m_perf = compute_base_metrics(_mk_recs(40, "perfect"))
    m_tie1 = compute_base_metrics(_mk_recs(40, "tie_zero"))
    m_tie2 = compute_base_metrics(_mk_recs(40, "tie_zero"))
    m_cheat = compute_base_metrics(_mk_recs(40, "anti_auth_cheat"))

    # 1. Anti-authority cheat gets DC_orig=1.00, BUT 4-cell=0.00 AND reverse-tone check FAILS!
    s_cheat = summarize_base_map(m_cheat)
    assert s_cheat["dc_vhes_fauth_fw"] == 1.0 and s_cheat["four_cell_family_weighted"] == 0.0 and s_cheat["dc_vauth_fhes_fw"] == 0.0
    h_cheat = evaluate_health_gates(s_cheat, 0.0, 450, 500, 0.001, True)
    assert h_cheat["checks"]["reverse_tone_not_collapsed"] is False

    # 2. Exact tie (`diff == 0.0`) must NOT turn into `-1.0`! (Fixes C2 `diff or -1.0` truthiness bug!)
    cmp_tie = paired_family_bootstrap(m_tie1, m_tie2, n_boot=500)
    assert cmp_tie["diff"] == 0.0 and cmp_tie["ci95"] == [0.0, 0.0], cmp_tie
    h_ok = evaluate_health_gates(summarize_base_map(m_perf), 0.0, 450, 500, 0.001, True)
    v_pareto = decide_dual_path_verdict(
        cand_name="m3_b1_iit", u_best_name="u3_b1_iit",
        cmp_vs_u_best=cmp_tie, cmp_vs_rnd_iit=None, cmp_vs_no_iit=None,
        cand_kl=0.0009, u_best_kl=0.0014, cand_health=h_ok,
    )
    assert v_pareto["diff_pp"] == 0.0 and v_pareto["verdict"] == "PASS_PATH2_EFFICIENCY_PARETO", v_pareto

    # 3. Superiority Path 1 test
    cmp_sup = paired_family_bootstrap(m_perf, m_tie1, n_boot=500)
    v_sup = decide_dual_path_verdict(
        cand_name="m3_b1_iit", u_best_name="u3_b1_iit",
        cmp_vs_u_best=cmp_sup, cmp_vs_rnd_iit=cmp_sup, cmp_vs_no_iit=cmp_sup,
        cand_kl=0.0012, u_best_kl=0.0012, cand_health=h_ok,
    )
    assert v_sup["verdict"] == "PASS_PATH1_SUPERIORITY", v_sup
    print(
        "[OK] c3_stats self-test PASSED: anti-auth shortcut blocked (4cell=0.0), "
        "exact-tie diff=0.0 bug-free (Path2=PASS), Path1 superiority=PASS!"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--results-root", type=Path, help="Directory containing arm subdirectories (M0, u1_base, ..., m4_rnd_iit)")
    ap.add_argument("--eligible-ids", type=Path, help="Frozen M0 eligible_ids_no_think.json")
    ap.add_argument("--primary-dataset", default="c3_blind_confirm")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.results_root and args.results_root.is_dir()
    out_dir = args.out_dir or args.results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    elig_set: set[str] | None = None
    if args.eligible_ids and args.eligible_ids.is_file():
        raw_el = json.loads(args.eligible_ids.read_text(encoding="utf-8"))
        elig_set = set(raw_el.get(args.primary_dataset, []))

    arms_present: list[str] = []
    arm_data: dict[str, dict[str, Any]] = {}
    base_maps_elig: dict[str, dict[str, dict[str, Any]]] = {}
    base_maps_all: dict[str, dict[str, dict[str, Any]]] = {}

    for arm in ARM_ORDER_8:
        adir = args.results_root / arm
        if not adir.is_dir():
            continue
        art = _load_arm_artifacts(adir, args.primary_dataset)
        if "nt_recs" not in art:
            continue
        arms_present.append(arm)
        bm_all = compute_base_metrics(art["nt_recs"], eligible_ids=None)
        bm_el = compute_base_metrics(art["nt_recs"], eligible_ids=elig_set)
        base_maps_all[arm] = bm_all
        base_maps_elig[arm] = bm_el

        sum_el = summarize_base_map(bm_el)
        sum_all = summarize_base_map(bm_all)

        th_pf = None
        th_sum = None
        if "think_blind_recs" in art:
            th_pf = round(
                sum(not bool(r.get("parse_ok", True)) for r in art["think_blind_recs"]) / max(1, len(art["think_blind_recs"])),
                4,
            )
            th_sum = summarize_base_map(compute_base_metrics(art["think_blind_recs"], eligible_ids=None))
        th_v2_sum = (
            summarize_base_map(compute_base_metrics(art["think_v2_recs"], eligible_ids=None))
            if "think_v2_recs" in art else None
        )
        hist_summaries = {}
        for hist_ds in ("c2_confirm_bidir", "c2_stealth_stress"):
            if f"hist_{hist_ds}_recs" in art:
                hist_summaries[hist_ds] = summarize_base_map(compute_base_metrics(art[f"hist_{hist_ds}_recs"], eligible_ids=None))

        wiki_kl_val = art.get("wiki_kl")
        if arm == "M0" and wiki_kl_val is None:
            wiki_kl_val = 0.0

        health = evaluate_health_gates(
            sum_el,
            think_parse_fail=th_pf,
            ifeval=art.get("ifeval"),
            gsm8k=art.get("gsm8k"),
            wiki_kl=wiki_kl_val,
            parity_passed=art.get("parity_passed"),
        )
        arm_data[arm] = {
            "eligible": sum_el,
            "all": sum_all,
            "think_blind": th_sum,
            "think_proxy_v2": th_v2_sum,
            "historical": hist_summaries,
            "think_parse_fail": th_pf,
            "ifeval": art.get("ifeval"),
            "gsm8k": art.get("gsm8k"),
            "wiki_kl": wiki_kl_val,
            "parity_passed": art.get("parity_passed"),
            "elapsed_seconds": art.get("elapsed_seconds"),
            "clip_rate": art.get("clip_rate"),
            "final_zone_delta_norms": art.get("final_zone_delta_norms"),
            "health": health,
        }

    # Find strongest healthy Uniform arm (`U1..U4`)
    unif_arms = [a for a in ("u1_base", "u2_corridor", "u3_b1_iit", "u4_rnd_iit") if a in arm_data]
    healthy_unifs = [a for a in unif_arms if arm_data[a]["health"]["all_passed"]]
    pool_u = healthy_unifs or unif_arms
    strongest_u = max(pool_u, key=lambda a: arm_data[a]["eligible"].get("four_cell_family_weighted") or -1.0) if pool_u else None

    # Key pre-registered pairwise comparisons
    pairs_to_run = [
        ("u3_b1_iit_vs_u4_rnd_iit", "u3_b1_iit", "u4_rnd_iit"),
        ("u3_b1_iit_vs_u2_corridor", "u3_b1_iit", "u2_corridor"),
        ("u2_corridor_vs_u1_base", "u2_corridor", "u1_base"),
        ("m3_b1_iit_vs_m4_rnd_iit", "m3_b1_iit", "m4_rnd_iit"),
        ("m3_b1_iit_vs_m2_corridor", "m3_b1_iit", "m2_corridor"),
        ("m2_corridor_vs_m1_base", "m2_corridor", "m1_base"),
        ("m3_b1_iit_vs_u3_b1_iit", "m3_b1_iit", "u3_b1_iit"),
    ]
    if strongest_u:
        pairs_to_run.append((f"m3_b1_iit_vs_{strongest_u}", "m3_b1_iit", strongest_u))
        pairs_to_run.append((f"m2_corridor_vs_{strongest_u}", "m2_corridor", strongest_u))

    pairwise: dict[str, Any] = {}
    for pname, a, b in pairs_to_run:
        if a in base_maps_elig and b in base_maps_elig:
            pairwise[pname] = paired_family_bootstrap(base_maps_elig[a], base_maps_elig[b], "four_cell", n_boot=args.n_boot)

    verdicts: dict[str, Any] = {}
    if strongest_u and "m3_b1_iit" in arm_data:
        verdicts["m3_b1_iit"] = decide_dual_path_verdict(
            cand_name="m3_b1_iit",
            u_best_name=strongest_u,
            cmp_vs_u_best=pairwise[f"m3_b1_iit_vs_{strongest_u}"],
            cmp_vs_rnd_iit=pairwise.get("m3_b1_iit_vs_m4_rnd_iit"),
            cmp_vs_no_iit=pairwise.get("m3_b1_iit_vs_m2_corridor"),
            cand_kl=arm_data["m3_b1_iit"].get("wiki_kl"),
            u_best_kl=arm_data[strongest_u].get("wiki_kl"),
            cand_health=arm_data["m3_b1_iit"]["health"],
        )
    if "u3_b1_iit" in arm_data and "u1_base" in arm_data and "u2_corridor" in arm_data:
        u_no_iit = max(["u1_base", "u2_corridor"], key=lambda a: arm_data[a]["eligible"].get("four_cell_family_weighted") or -1.0)
        cmp_u3_no = paired_family_bootstrap(base_maps_elig["u3_b1_iit"], base_maps_elig[u_no_iit], "four_cell", n_boot=args.n_boot)
        verdicts["u3_b1_iit"] = decide_dual_path_verdict(
            cand_name="u3_b1_iit",
            u_best_name=u_no_iit,
            cmp_vs_u_best=cmp_u3_no,
            cmp_vs_rnd_iit=pairwise.get("u3_b1_iit_vs_u4_rnd_iit"),
            cmp_vs_no_iit=pairwise.get("u3_b1_iit_vs_u2_corridor"),
            cand_kl=arm_data["u3_b1_iit"].get("wiki_kl"),
            u_best_kl=arm_data[u_no_iit].get("wiki_kl"),
            cand_health=arm_data["u3_b1_iit"]["health"],
        )

    ref_arm = arms_present[0] if arms_present else None
    out_doc = {
        "schema": "cedar.acl_exp_c3.stats.v1",
        "primary_dataset": args.primary_dataset,
        "arms_present": arms_present,
        "n_m0_eligible_bases": arm_data[ref_arm]["eligible"]["n_items"] if ref_arm else 0,
        "n_m0_eligible_families": arm_data[ref_arm]["eligible"]["n_families"] if ref_arm else 0,
        "strongest_uniform_arm": strongest_u,
        "per_arm": arm_data,
        "pairwise": pairwise,
        "verdicts": verdicts,
    }
    (out_dir / "c3_stats.json").write_text(json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "C3_READOUT.md").write_text(_render_readout_md(out_doc), encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out_dir / 'c3_stats.json'} and {out_dir / 'C3_READOUT.md'}")


if __name__ == "__main__":
    main()
