#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 4 Multi-Seed (`seed=42` + `seed=1024`) Strict Aggregator & Dual-Path Verdict Engine.

Evaluates BOTH independent seeds separately AND computes exact unit-clustered pooled 2-seed comparisons
(averaging each unit's score across seed=42 & seed=1024 BEFORE bootstrapping over the 900 independent generation units,
so the two seeds are never treated as falsely independent items!) with 100% strict zero-`None`-pass health gates.

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_stage4_multiseed_stats.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))

import c3_stats as C3S  # noqa: E402
import c35_stats as S35  # noqa: E402


def _pooled_two_seed_bootstrap(
    bm_a_42: dict[str, dict[str, Any]],
    bm_b_42: dict[str, dict[str, Any]],
    bm_a_1024: dict[str, dict[str, Any]],
    bm_b_1024: dict[str, dict[str, Any]],
    metric: str,
    n_boot: int = 5000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Exact 900-unit paired bootstrap on the 2-seed average per independent generation unit."""
    fa_42 = C3S.family_weighted_scores(bm_a_42, metric)
    fb_42 = C3S.family_weighted_scores(bm_b_42, metric)
    fa_1k = C3S.family_weighted_scores(bm_a_1024, metric)
    fb_1k = C3S.family_weighted_scores(bm_b_1024, metric)
    common = sorted(set(fa_42) & set(fb_42) & set(fa_1k) & set(fb_1k))
    assert common
    va = np.array([0.5 * (fa_42[u] + fa_1k[u]) for u in common], dtype=np.float64)
    vb = np.array([0.5 * (fb_42[u] + fb_1k[u]) for u in common], dtype=np.float64)
    diffs = va - vb
    n = len(diffs)
    mean_diff = float(diffs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    p_pos = float(np.mean(boot <= 0.0))
    p_neg = float(np.mean(boot >= 0.0))
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    return {
        "n_units": n,
        "mean_a": round(float(va.mean()), 4),
        "mean_b": round(float(vb.mean()), 4),
        "diff": round(mean_diff, 4),
        "diff_pp": round(mean_diff * 100.0, 2),
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "ci95_pp": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
        "p_two_sided": round(p_two, 4),
    }


def _load_one_arm(arm_dir: Path, primary_dataset: str, base_meta: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
    if not arm_dir.is_dir():
        return None
    art = C3S._load_arm_artifacts(arm_dir, primary_dataset)
    if "nt_recs" not in art:
        return None
    bm = S35.compute_base_metrics_v2(art["nt_recs"], base_meta=base_meta)
    s = S35.summarize_v2(bm)
    th_pf = None
    th_4c = None
    th_p = arm_dir / "think" / "records_think_c3_blind_think_v2.jsonl"
    if th_p.is_file():
        tr = C3S._load_jsonl(th_p)
        th_pf = round(sum(not bool(r.get("parse_ok", True)) for r in tr) / max(1, len(tr)), 4)
        th_4c = S35.summarize_v2(S35.compute_base_metrics_v2(tr))["four_cell_family_weighted"]
    health = S35.evaluate_health_gates_strict(
        s,
        think_parse_fail=th_pf,
        ifeval=art.get("ifeval"),
        gsm8k=art.get("gsm8k"),
        wiki_kl=art.get("wiki_kl"),
        parity_passed=art.get("parity_passed"),
        require_capabilities=True,
    )
    info = {
        "dir": str(arm_dir),
        "all": s,
        "think_v2_4cell": th_4c,
        "think_parse_fail": th_pf,
        "ifeval": art.get("ifeval"),
        "gsm8k": art.get("gsm8k"),
        "wiki_kl": art.get("wiki_kl"),
        "parity_passed": art.get("parity_passed"),
        "health": health,
    }
    return info, bm


def run_selftest_tiny() -> None:
    # Verify pooled 2-seed unit bootstrap math & symmetry
    bm_a1 = {f"B{i}": {"family_id": f"U{i}", "four_cell": 1 if i < 60 else 0} for i in range(100)}
    bm_b1 = {f"B{i}": {"family_id": f"U{i}", "four_cell": 1 if i < 50 else 0} for i in range(100)}
    bm_a2 = {f"B{i}": {"family_id": f"U{i}", "four_cell": 1 if i < 58 else 0} for i in range(100)}
    bm_b2 = {f"B{i}": {"family_id": f"U{i}", "four_cell": 1 if i < 50 else 0} for i in range(100)}
    res = _pooled_two_seed_bootstrap(bm_a1, bm_b1, bm_a2, bm_b2, "four_cell", n_boot=1000)
    assert abs(res["diff_pp"] - 9.0) < 1e-6 and res["ci95_pp"][0] > 0.0, res
    print(f"[OK] c35_stage4_multiseed_stats self-test PASSED: pooled diff={res['diff_pp']:+.2f} pp CI={res['ci95_pp']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--stage3-root", type=Path, default=C35 / "results" / "stage3")
    ap.add_argument("--stage4-root", type=Path, default=C35 / "results" / "stage4")
    ap.add_argument("--dataset-file", type=Path, default=C35 / "data" / "c3_blind_confirm_v2_bases.jsonl")
    ap.add_argument("--primary-dataset", default="c3_blind_confirm_v2")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    base_meta = {
        r["base_id"]: r
        for r in (json.loads(l) for l in args.dataset_file.read_text(encoding="utf-8").splitlines() if l.strip())
    }

    # Map logical method -> {seed: path}
    arm_paths: dict[str, dict[str, Path]] = {
        "u1_base": {
            "42": args.stage3_root / "u1_base",
            "1024": args.stage4_root / "s4_s1024_u1_base",
        },
        "u2_corridor": {
            "42": args.stage3_root / "u2_corridor",
            "1024": args.stage4_root / "s4_s1024_u2_corridor",
        },
        "M3-v3-PureMid": {
            "42": args.stage3_root / "s3_puremid_corr",
            "1024": args.stage4_root / "s4_s1024_puremid",
        },
        "M3-v3-PureMid-Random-Mask": {
            "42": args.stage3_root / "s3_puremid_randmask",
            "1024": args.stage4_root / "s4_s1024_puremid_randmask",
        },
        "M3-v4-PureMid-MLP+": {
            "42": args.stage4_root / "s4_s42_v4_mlpplus",
            "1024": args.stage4_root / "s4_s1024_v4_mlpplus",
        },
        "M3-v4-Bridge2": {
            "42": args.stage4_root / "s4_s42_v4_bridge2",
            "1024": args.stage4_root / "s4_s1024_v4_bridge2",
        },
    }

    per_arm: dict[str, dict[str, Any]] = {}
    bms: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for mname, sp in arm_paths.items():
        per_arm[mname] = {}
        for sd, p in sp.items():
            loaded = _load_one_arm(p, args.primary_dataset, base_meta)
            if loaded is not None:
                info, bm = loaded
                per_arm[mname][sd] = info
                bms[(mname, sd)] = bm

    pair_specs = [
        ("Corridor_vs_Base", "u2_corridor", "u1_base"),
        ("PureMid_vs_u2", "M3-v3-PureMid", "u2_corridor"),
        ("PureMid_vs_RandMask", "M3-v3-PureMid", "M3-v3-PureMid-Random-Mask"),
        ("v4_MLPplus_vs_u2", "M3-v4-PureMid-MLP+", "u2_corridor"),
        ("v4_MLPplus_vs_RandMask", "M3-v4-PureMid-MLP+", "M3-v3-PureMid-Random-Mask"),
        ("v4_Bridge2_vs_u2", "M3-v4-Bridge2", "u2_corridor"),
    ]

    comparisons: dict[str, dict[str, Any]] = {}
    for pname, ma, mb in pair_specs:
        entry: dict[str, Any] = {}
        for sd in ("42", "1024"):
            if (ma, sd) in bms and (mb, sd) in bms:
                entry[f"seed_{sd}"] = {
                    "four_cell": C3S.paired_family_bootstrap(bms[(ma, sd)], bms[(mb, sd)], "four_cell", n_boot=args.n_boot),
                    "joint_t1_4cell": C3S.paired_family_bootstrap(bms[(ma, sd)], bms[(mb, sd)], "joint_t1_4cell", n_boot=args.n_boot),
                }
        if all((x, sd) in bms for x in (ma, mb) for sd in ("42", "1024")):
            entry["pooled_2seed"] = {
                "four_cell": _pooled_two_seed_bootstrap(bms[(ma, "42")], bms[(mb, "42")], bms[(ma, "1024")], bms[(mb, "1024")], "four_cell", n_boot=args.n_boot),
                "joint_t1_4cell": _pooled_two_seed_bootstrap(bms[(ma, "42")], bms[(mb, "42")], bms[(ma, "1024")], bms[(mb, "1024")], "joint_t1_4cell", n_boot=args.n_boot),
            }
        if entry:
            comparisons[pname] = entry

    # Strict Dual-Path verdicts per seed AND pooled 2-seed for all M3 candidates vs u2_corridor!
    verdicts: dict[str, dict[str, Any]] = {}
    for cand, pkey in (
        ("M3-v3-PureMid", "PureMid_vs_u2"),
        ("M3-v4-PureMid-MLP+", "v4_MLPplus_vs_u2"),
        ("M3-v4-Bridge2", "v4_Bridge2_vs_u2"),
    ):
        if pkey not in comparisons:
            continue
        verdicts[cand] = {}
        for sd in ("42", "1024"):
            sk = f"seed_{sd}"
            if sk in comparisons[pkey] and sd in per_arm.get(cand, {}) and sd in per_arm.get("u2_corridor", {}):
                verdicts[cand][sk] = S35.decide_dual_path_strict(
                    cand_name=f"{cand}@s{sd}",
                    u_best_name=f"u2_corridor@s{sd}",
                    cmp_vs_u_best=comparisons[pkey][sk]["four_cell"],
                    cmp_joint_vs_u_best=comparisons[pkey][sk]["joint_t1_4cell"],
                    cmp_vs_rnd_iit=None,
                    cmp_vs_no_iit=None,
                    require_iit_controls=False,
                    cand_kl=per_arm[cand][sd]["wiki_kl"],
                    u_best_kl=per_arm["u2_corridor"][sd]["wiki_kl"],
                    cand_health=per_arm[cand][sd]["health"],
                )
        if "pooled_2seed" in comparisons[pkey]:
            kl_c = 0.5 * (per_arm[cand]["42"]["wiki_kl"] + per_arm[cand]["1024"]["wiki_kl"])
            kl_u = 0.5 * (per_arm["u2_corridor"]["42"]["wiki_kl"] + per_arm["u2_corridor"]["1024"]["wiki_kl"])
            both_h = {"all_passed": bool(per_arm[cand]["42"]["health"]["all_passed"] and per_arm[cand]["1024"]["health"]["all_passed"])}
            verdicts[cand]["pooled_2seed"] = S35.decide_dual_path_strict(
                cand_name=f"{cand}@2seed",
                u_best_name="u2_corridor@2seed",
                cmp_vs_u_best=comparisons[pkey]["pooled_2seed"]["four_cell"],
                cmp_joint_vs_u_best=comparisons[pkey]["pooled_2seed"]["joint_t1_4cell"],
                cmp_vs_rnd_iit=None,
                cmp_vs_no_iit=None,
                require_iit_controls=False,
                cand_kl=kl_c,
                u_best_kl=kl_u,
                cand_health=both_h,
            )

    doc = {
        "schema": "cedar.acl_exp_c35.multiseed_stats.v1",
        "primary_dataset": args.primary_dataset,
        "per_arm": per_arm,
        "comparisons": comparisons,
        "verdicts": verdicts,
    }
    args.stage4_root.mkdir(parents=True, exist_ok=True)
    (args.stage4_root / "c35_multiseed_stats.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    lines = [
        "# ACL-exp-c3.5 Stage 4 Multi-Seed Readout (`seed=42` + `seed=1024`, `900` Independent Units)",
        "",
        "## 1. Per-Seed Full Table",
        "",
        "| Method | Seed | 4-Cell (UW) | Joint T1∧4C | Row B | Turn-1 | NLL | `think_v2` | IFEval / GSM8K | Wiki KL | Health |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mname, sdict in per_arm.items():
        for sd in ("42", "1024"):
            if sd not in sdict:
                continue
            a = sdict[sd]
            s = a["all"]
            rb = s.get("subgroups", {}).get("ce_row_B", {})
            rb_s = f"{rb.get('four_cell',0)*100:.1f}%" if rb else "—"
            th_s = f"{a['think_v2_4cell']*100:.2f}%" if a.get("think_v2_4cell") is not None else "—"
            hl = "PASS" if a["health"]["all_passed"] else "FAIL"
            lines.append(
                f"| `{mname}` | {sd} | **{s['four_cell_family_weighted']*100:.2f}%** "
                f"| **{s['joint_t1_4cell_unit_weighted']*100:.2f}%** | {rb_s} "
                f"| {s['turn1_dual_acc']*100:.1f}% | {s['mean_4opt_nll']:.3f} | {th_s} "
                f"| {a['ifeval']} / {a['gsm8k']} | `{a['wiki_kl']:.6f}` | {hl} |"
            )

    lines += [
        "",
        "## 2. Paired Unit-Clustered Comparisons (`seed=42`, `seed=1024`, AND `Pooled 2-Seed`)",
        "",
        "| Comparison | `seed=42` 4-Cell (pp) | `seed=1024` 4-Cell (pp) | **Pooled 2-Seed 4-Cell [95% CI]** | **Pooled 2-Seed Joint [95% CI]** |",
        "|---|---|---|---|---|",
    ]
    for pname, c in comparisons.items():
        s42 = f"{c['seed_42']['four_cell']['diff_pp']:+.2f} (p={c['seed_42']['four_cell']['p_two_sided']:.4f})" if "seed_42" in c else "—"
        s1k = f"{c['seed_1024']['four_cell']['diff_pp']:+.2f} (p={c['seed_1024']['four_cell']['p_two_sided']:.4f})" if "seed_1024" in c else "—"
        p4 = f"**{c['pooled_2seed']['four_cell']['diff_pp']:+.2f}** `{c['pooled_2seed']['four_cell']['ci95_pp']}` (p={c['pooled_2seed']['four_cell']['p_two_sided']:.4f})" if "pooled_2seed" in c else "—"
        pj = f"**{c['pooled_2seed']['joint_t1_4cell']['diff_pp']:+.2f}** `{c['pooled_2seed']['joint_t1_4cell']['ci95_pp']}` (p={c['pooled_2seed']['joint_t1_4cell']['p_two_sided']:.4f})" if "pooled_2seed" in c else "—"
        lines.append(f"| `{pname}` | {s42} | {s1k} | {p4} | {pj} |")

    lines += [
        "",
        "## 3. Strict Dual-Path Verdicts (Per-Seed & Pooled 2-Seed)",
        "",
        "| Candidate | Split | 4-Cell Diff (CI_low) | Joint CI_low | KL Red% | Health | **Verdict** |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for cand, vdict in verdicts.items():
        for sp, v in vdict.items():
            lines.append(
                f"| `{cand}` | `{sp}` | {v['diff_pp']:+.2f} ({v['ci95_low_pp']:+.2f}) | {v['joint_ci95_low_pp']:+.2f} "
                f"| **{v['kl_reduction_vs_u_best_pct']:+.1f}%** | {v['health_passed']} | **{v['verdict']}** |"
            )

    (args.stage4_root / "C35_MULTISEED_READOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {args.stage4_root / 'c35_multiseed_stats.json'} and C35_MULTISEED_READOUT.md")


if __name__ == "__main__":
    main()
