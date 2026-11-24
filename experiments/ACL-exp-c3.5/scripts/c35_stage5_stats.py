#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 5 Strict Aggregator (8B Rank-Matched Closure + 14B 4-Arm Prospective Replication + EX-FEVER).

Directly implements EVERY statistical & audit fix required by Astra's Stage 4 Review:
  1. ZERO SILENT UNIT INTERSECTION:
       Hard-asserts every evaluated arm contains the EXACT complete set of expected generation units
       (`900` units / `1,080` items on `c3_blind_confirm_v2` & `c3_blind_confirm_v3`). Any missing unit raises immediately!
  2. TRUE RANK-MATCHED RANDOM-MASK MAPPING:
       `v4_MLPplus_vs_v4_RandMask` strictly compares `M3-v4-PureMid-MLP+` vs `M3-v4-PureMid-MLP+-Random-Mask`
       (never silently pointing to v3!). On 8B, ALSO compares vs independent `M3-v4-PureMid-MLP+-Random-Mask2`!
  3. SCALE-SPECIFIC M0-ANCHORED RELATIVE HEALTH GATES:
       No hardcoded 8B `430 / 470` for 14B! Reads the scale's OWN `M0` capability anchor (`ifeval_m0`, `gsm8k_m0`)
       and enforces `ifeval >= ifeval_m0 - 10` (~1.85 pp) and `gsm8k >= gsm8k_m0 - 26` (~2.0 pp), plus strict
       `wiki_kl <= 0.005`, `think_parse_fail <= 0.02`, `S/shell/P0/I0 >= 0.95`, `DC_rev >= DC_orig - 0.10`, `parity == True`!
  4. SEPARATE MACRO-BAND PARETO VERDICT & FINE-GRAINED LOCALIZATION BONUS VERDICT:
       - Primary Macro-Band Pareto Gate: `4-Cell CI_low >= -2.0 pp` AND `Joint CI_low >= -2.0 pp`
         AND `Wiki KL <= 0.75 * u2_KL` AND `health.all_passed == True`.
       - Fine-Grained Localization Bonus Gate (Astra's rule): `(v4-MLP+ - v4-RandMask) CI_low > 0`
         (and when multi-seed, second seed same positive direction!).
  5. EXTERNAL NATURAL-LANGUAGE `EX-FEVER` TABLE:
       Aggregates `Acc`, `n_orig_correct`, `n_orig_wrong`, `R_UY`, `R_RU_evidence`, `R_RU_user_evidence`,
       and `Selectivity (R_RU - R_UY)` cleanly across `M0`, `u1_base`, `u2_corridor`, `v4-MLP+`, `v4-RandMask`!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_stage5_stats.py --selftest-tiny
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


def assert_complete_units(
    bm: dict[str, dict[str, Any]],
    expected_units: set[str],
    expected_bases: set[str],
    arm_label: str,
) -> None:
    """FAIL-CLOSED unit completeness guard: NEVER silently intersect partial outputs!"""
    got_bases = set(bm.keys())
    got_units = {str(v["family_id"]) for v in bm.values()}
    if got_bases != expected_bases or got_units != expected_units:
        miss_b = len(expected_bases - got_bases)
        miss_u = len(expected_units - got_units)
        raise AssertionError(
            f"[INCOMPLETE UNITS] Arm '{arm_label}' missing {miss_b} bases / {miss_u} units "
            f"(got {len(got_bases)}/{len(expected_bases)} bases, {len(got_units)}/{len(expected_units)} units)"
        )


def paired_unit_bootstrap_strict(
    bm_a: dict[str, dict[str, Any]],
    bm_b: dict[str, dict[str, Any]],
    expected_units: set[str],
    metric: str,
    n_boot: int = 5000,
    seed: int = 2027,
) -> dict[str, Any]:
    fa = C3S.family_weighted_scores(bm_a, metric)
    fb = C3S.family_weighted_scores(bm_b, metric)
    assert set(fa) == expected_units and set(fb) == expected_units, (len(fa), len(fb), len(expected_units))
    units = sorted(expected_units)
    va = np.array([fa[u] for u in units], dtype=np.float64)
    vb = np.array([fb[u] for u in units], dtype=np.float64)
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


def pooled_two_seed_bootstrap_strict(
    bm_a_42: dict[str, dict[str, Any]],
    bm_b_42: dict[str, dict[str, Any]],
    bm_a_1k: dict[str, dict[str, Any]],
    bm_b_1k: dict[str, dict[str, Any]],
    expected_units: set[str],
    metric: str,
    n_boot: int = 5000,
    seed: int = 2027,
) -> dict[str, Any]:
    fa42 = C3S.family_weighted_scores(bm_a_42, metric)
    fb42 = C3S.family_weighted_scores(bm_b_42, metric)
    fa1k = C3S.family_weighted_scores(bm_a_1k, metric)
    fb1k = C3S.family_weighted_scores(bm_b_1k, metric)
    for tag, d in (("a42", fa42), ("b42", fb42), ("a1k", fa1k), ("b1k", fb1k)):
        assert set(d) == expected_units, (tag, len(d), len(expected_units))
    units = sorted(expected_units)
    va = np.array([0.5 * (fa42[u] + fa1k[u]) for u in units], dtype=np.float64)
    vb = np.array([0.5 * (fb42[u] + fb1k[u]) for u in units], dtype=np.float64)
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


def evaluate_health_gates_relative(
    blind_sum: dict[str, Any],
    *,
    ifeval_m0: int,
    gsm8k_m0: int,
    max_ifeval_drop: int = 10,
    max_gsm8k_drop: int = 26,
    think_parse_fail: float | None,
    ifeval: int | None,
    gsm8k: int | None,
    wiki_kl: float | None,
    parity_passed: bool | None,
    require_capabilities: bool = True,
) -> dict[str, Any]:
    """Scale-adaptive M0-anchored strict health gates (zero hardcoded 8B floors for 14B!)."""
    ifeval_floor = ifeval_m0 - max_ifeval_drop
    gsm8k_floor = gsm8k_m0 - max_gsm8k_drop
    checks = {
        f"ifeval_ge_{ifeval_floor}": bool(ifeval is not None and ifeval >= ifeval_floor) if require_capabilities else True,
        f"gsm8k_ge_{gsm8k_floor}": bool(gsm8k is not None and gsm8k >= gsm8k_floor) if require_capabilities else True,
        "wiki_kl_le_0005": bool(wiki_kl is not None and wiki_kl <= 0.005),
        "think_parse_fail_le_2pct": bool(think_parse_fail is not None and think_parse_fail <= 0.02) if require_capabilities else True,
        "s_rej_ge_95pct": bool(blind_sum.get("s_rej") is not None and blind_sum["s_rej"] >= 0.95),
        "shell_rej_ge_95pct": bool(blind_sum.get("shell_rej") is not None and blind_sum["shell_rej"] >= 0.95),
        "p0_keep_ge_95pct": bool(blind_sum.get("p0_keep") is not None and blind_sum["p0_keep"] >= 0.95),
        "i0_keep_ge_95pct": bool(blind_sum.get("i0_keep") is not None and blind_sum["i0_keep"] >= 0.95),
        "reverse_tone_not_collapsed": bool(
            blind_sum.get("dc_vhes_fauth_fw") is not None
            and blind_sum.get("dc_vauth_fhes_fw") is not None
            and (blind_sum["dc_vauth_fhes_fw"] >= blind_sum["dc_vhes_fauth_fw"] - 0.10)
        ),
        "parity_passed": bool(parity_passed is True) if require_capabilities else True,
    }
    return {
        "all_passed": all(checks.values()),
        "anchors": {"ifeval_m0": ifeval_m0, "ifeval_floor": ifeval_floor, "gsm8k_m0": gsm8k_m0, "gsm8k_floor": gsm8k_floor},
        "checks": checks,
    }


def evaluate_fine_localization_bonus(
    cmp_v4_vs_rm_primary: dict[str, Any],
    cmp_v4_vs_rm_seed2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Astra's strict bonus criterion for fine-grained B1 localization above macro-band budget."""
    ci_lo = cmp_v4_vs_rm_primary["ci95"][0]
    primary_sig = bool(ci_lo is not None and ci_lo > 0.0)
    seed2_same_dir = True if cmp_v4_vs_rm_seed2 is None else bool(cmp_v4_vs_rm_seed2["diff"] > 0.0)
    passed = bool(primary_sig and seed2_same_dir)
    return {
        "primary_diff_pp": cmp_v4_vs_rm_primary["diff_pp"],
        "primary_ci95_pp": cmp_v4_vs_rm_primary["ci95_pp"],
        "primary_ci_low_gt_0": primary_sig,
        "seed2_diff_pp": cmp_v4_vs_rm_seed2["diff_pp"] if cmp_v4_vs_rm_seed2 else None,
        "seed2_same_direction": seed2_same_dir,
        "fine_localization_bonus_passed": passed,
    }


def _load_arm_strict(
    arm_dir: Path,
    dataset_name: str,
    base_meta: dict[str, dict[str, Any]],
    expected_units: set[str],
    expected_bases: set[str],
    *,
    ifeval_m0: int,
    gsm8k_m0: int,
    think_dataset_name: str,
    require_capabilities: bool = True,
    arm_label: str = "arm",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    assert arm_dir.is_dir(), f"Missing arm directory: {arm_dir}"
    art = C3S._load_arm_artifacts(arm_dir, dataset_name)
    assert "nt_recs" in art, f"Missing no_think records for {dataset_name} in {arm_dir}"
    bm = S35.compute_base_metrics_v2(art["nt_recs"], base_meta=base_meta)
    assert_complete_units(bm, expected_units, expected_bases, f"{arm_label}:{dataset_name}")
    s = S35.summarize_v2(bm)

    th_pf = None
    th_4c = None
    th_p = arm_dir / "think" / f"records_think_{think_dataset_name}.jsonl"
    if th_p.is_file():
        tr = C3S._load_jsonl(th_p)
        th_pf = round(sum(not bool(r.get("parse_ok", True)) for r in tr) / max(1, len(tr)), 4)
        th_4c = S35.summarize_v2(S35.compute_base_metrics_v2(tr))["four_cell_family_weighted"]

    health = evaluate_health_gates_relative(
        s,
        ifeval_m0=ifeval_m0,
        gsm8k_m0=gsm8k_m0,
        think_parse_fail=th_pf,
        ifeval=art.get("ifeval"),
        gsm8k=art.get("gsm8k"),
        wiki_kl=art.get("wiki_kl"),
        parity_passed=art.get("parity_passed"),
        require_capabilities=require_capabilities,
    )
    info = {
        "dir": str(arm_dir),
        "dataset": dataset_name,
        "all": s,
        "think_4cell": th_4c,
        "think_parse_fail": th_pf,
        "ifeval": art.get("ifeval"),
        "gsm8k": art.get("gsm8k"),
        "wiki_kl": art.get("wiki_kl"),
        "parity_passed": art.get("parity_passed"),
        "health": health,
    }
    return info, bm


def _load_exfever_summary(arm_dir: Path, alias: str, regime: str = "no_think") -> dict[str, Any] | None:
    for cand in (
        arm_dir / "exfever" / f"exfever_{regime}_summary_{alias}.json",
        arm_dir / f"exfever_{regime}_summary_{alias}.json",
    ):
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    # Fallback: glob single matching summary in arm_dir/exfever
    exd = arm_dir / "exfever"
    if exd.is_dir():
        hits = sorted(exd.glob(f"exfever_{regime}_summary_*.json"))
        if hits:
            return json.loads(hits[0].read_text(encoding="utf-8"))
    return None


def run_selftest_tiny() -> None:
    # 1. Verify fail-closed missing-unit assertion
    exp_u = {f"U{i}" for i in range(100)}
    exp_b = {f"B{i}" for i in range(100)}
    bm_full = {f"B{i}": {"family_id": f"U{i}", "four_cell": 1 if i < 60 else 0} for i in range(100)}
    bm_miss = {f"B{i}": {"family_id": f"U{i}", "four_cell": 1 if i < 60 else 0} for i in range(99)}
    assert_complete_units(bm_full, exp_u, exp_b, "ok_arm")
    raised = False
    try:
        assert_complete_units(bm_miss, exp_u, exp_b, "broken_arm")
    except AssertionError:
        raised = True
    assert raised, "Missing-unit guard MUST raise AssertionError!"

    # 2. Verify M0-relative health gates for 14B (e.g. M0 IFEval=475, GSM8K=900)
    fake_sum = {"s_rej": 1.0, "shell_rej": 1.0, "p0_keep": 1.0, "i0_keep": 1.0, "dc_vhes_fauth_fw": 0.55, "dc_vauth_fhes_fw": 0.60}
    h_pass = evaluate_health_gates_relative(
        fake_sum, ifeval_m0=475, gsm8k_m0=900, think_parse_fail=0.0, ifeval=468, gsm8k=880, wiki_kl=0.001, parity_passed=True
    )
    h_fail = evaluate_health_gates_relative(
        fake_sum, ifeval_m0=475, gsm8k_m0=900, think_parse_fail=0.0, ifeval=460, gsm8k=880, wiki_kl=0.001, parity_passed=True
    )
    assert h_pass["all_passed"] is True and h_fail["all_passed"] is False, (h_pass, h_fail)

    # 3. Verify strict paired & pooled bootstrap + fine-grained localization bonus
    bm_b = {f"B{i}": {"family_id": f"U{i}", "four_cell": 1 if i < 50 else 0} for i in range(100)}
    c1 = paired_unit_bootstrap_strict(bm_full, bm_b, exp_u, "four_cell", n_boot=1000)
    assert abs(c1["diff_pp"] - 10.0) < 1e-6 and c1["ci95_pp"][0] > 0.0
    bonus = evaluate_fine_localization_bonus(c1, c1)
    assert bonus["fine_localization_bonus_passed"] is True
    print("[OK] c35_stage5_stats self-test PASSED (fail-closed unit guard, 14B M0-relative health gates, strict bootstrap & localization bonus!)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--config-json", type=Path, help="Stage 5 aggregation manifest JSON specifying scale, M0 anchor, datasets & arm directories")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.config_json and args.out_dir, "--config-json and --out-dir are required"
    cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    scale = cfg["scale"]  # e.g. "qwen3_8b" or "qwen3_14b"
    ifeval_m0 = int(cfg["m0_anchor"]["ifeval"])
    gsm8k_m0 = int(cfg["m0_anchor"]["gsm8k"])
    datasets = cfg["datasets"]  # list of {"name": "c3_blind_confirm_v3", "file": "...", "think_name": "c3_blind_think_v3"}
    arms_spec = cfg["arms"]  # {logical_name: {seed_str: dir_path}}
    pair_specs = cfg["pair_specs"]  # list of [pair_name, arm_a, arm_b]
    pareto_candidates = cfg.get("pareto_candidates", ["M3-v4-PureMid-MLP+"])
    exfever_regime = cfg.get("exfever_regime", "no_think")

    out_doc: dict[str, Any] = {
        "schema": "cedar.acl_exp_c35.stage5_stats.v1",
        "scale": scale,
        "m0_anchor": cfg["m0_anchor"],
        "by_dataset": {},
        "exfever": {},
    }

    # Collect EX-FEVER across all arms (+ M0 if provided)
    if "m0_dir" in cfg and cfg["m0_dir"]:
        ex_m0 = _load_exfever_summary(Path(cfg["m0_dir"]), "M0", regime=exfever_regime)
        if ex_m0:
            out_doc["exfever"]["M0"] = ex_m0["metrics"]["headline"]
    for mname, sdict in arms_spec.items():
        for sd, dstr in sdict.items():
            ex = _load_exfever_summary(Path(dstr), mname, regime=exfever_regime)
            if ex:
                out_doc["exfever"][f"{mname}@s{sd}"] = ex["metrics"]["headline"]

    for ds_cfg in datasets:
        dname = ds_cfg["name"]
        dfile = Path(ds_cfg["file"])
        th_name = ds_cfg["think_name"]
        rows = [json.loads(l) for l in dfile.read_text(encoding="utf-8").splitlines() if l.strip()]
        base_meta = {r["base_id"]: r for r in rows}
        exp_bases = set(base_meta.keys())
        exp_units = {str(r.get("unit_id") or r.get("family_id") or r["base_id"]) for r in rows}

        per_arm: dict[str, dict[str, Any]] = {}
        bms: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        for mname, sdict in arms_spec.items():
            per_arm[mname] = {}
            for sd, dstr in sdict.items():
                info, bm = _load_arm_strict(
                    Path(dstr), dname, base_meta, exp_units, exp_bases,
                    ifeval_m0=ifeval_m0, gsm8k_m0=gsm8k_m0,
                    think_dataset_name=th_name, require_capabilities=True,
                    arm_label=f"{mname}@s{sd}",
                )
                per_arm[mname][sd] = info
                bms[(mname, sd)] = bm

        comparisons: dict[str, dict[str, Any]] = {}
        for pname, ma, mb in pair_specs:
            entry: dict[str, Any] = {}
            seeds_both = sorted(set(per_arm.get(ma, {})) & set(per_arm.get(mb, {})))
            for sd in seeds_both:
                entry[f"seed_{sd}"] = {
                    "four_cell": paired_unit_bootstrap_strict(bms[(ma, sd)], bms[(mb, sd)], exp_units, "four_cell", n_boot=args.n_boot),
                    "joint_t1_4cell": paired_unit_bootstrap_strict(bms[(ma, sd)], bms[(mb, sd)], exp_units, "joint_t1_4cell", n_boot=args.n_boot),
                }
            if "42" in seeds_both and "1024" in seeds_both:
                entry["pooled_2seed"] = {
                    "four_cell": pooled_two_seed_bootstrap_strict(
                        bms[(ma, "42")], bms[(mb, "42")], bms[(ma, "1024")], bms[(mb, "1024")], exp_units, "four_cell", n_boot=args.n_boot
                    ),
                    "joint_t1_4cell": pooled_two_seed_bootstrap_strict(
                        bms[(ma, "42")], bms[(mb, "42")], bms[(ma, "1024")], bms[(mb, "1024")], exp_units, "joint_t1_4cell", n_boot=args.n_boot
                    ),
                }
            comparisons[pname] = entry

        verdicts: dict[str, dict[str, Any]] = {}
        for cand in pareto_candidates:
            pkey = f"{cand}_vs_u2"
            if pkey not in comparisons:
                continue
            verdicts[cand] = {}
            for sd in sorted(set(per_arm.get(cand, {})) & set(per_arm.get("u2_corridor", {}))):
                sk = f"seed_{sd}"
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

        # Fine-grained localization bonus verdict (strictly v4 vs v4-RandMask!)
        loc_bonus = {}
        for lkey in ("v4_MLPplus_vs_v4_RandMask", "v4_MLPplus_vs_v4_RandMask2"):
            if lkey in comparisons:
                c_ent = comparisons[lkey]
                if "pooled_2seed" in c_ent:
                    loc_bonus[lkey] = evaluate_fine_localization_bonus(
                        c_ent["pooled_2seed"]["four_cell"], c_ent["seed_1024"]["four_cell"]
                    )
                elif "seed_42" in c_ent:
                    loc_bonus[lkey] = evaluate_fine_localization_bonus(c_ent["seed_42"]["four_cell"], None)

        out_doc["by_dataset"][dname] = {
            "n_bases": len(exp_bases),
            "n_units": len(exp_units),
            "per_arm": per_arm,
            "comparisons": comparisons,
            "verdicts": verdicts,
            "fine_localization_bonus": loc_bonus,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_p = args.out_dir / f"c35_stage5_{scale}_stats.json"
    json_p.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    # Render concise Markdown readout
    md = [f"# ACL-exp-c3.5 Stage 5 Strict Readout (`{scale}`, M0 anchor IFEval={ifeval_m0}, GSM8K={gsm8k_m0})", ""]
    for dname, ddat in out_doc["by_dataset"].items():
        md.append(f"## Dataset: `{dname}` (`{ddat['n_units']}` Independent Units / `{ddat['n_bases']}` Items)")
        md.append("")
        md.append("| Method | Seed | 4-Cell (UW) | Joint T1∧4C | Row B | Pos `bef/aft` | IFEval / GSM8K | Wiki KL | Health |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for mname, sdict in ddat["per_arm"].items():
            for sd, info in sdict.items():
                s = info["all"]
                rb = s["subgroups"]["ce_row_B"]["four_cell"] * 100.0
                pp = s.get("paired_position_diag") or {}
                pos_s = f"{pp.get('before_rules_4cell',0)*100:.1f} / {pp.get('after_rules_4cell',0)*100:.1f}" if pp else "—"
                md.append(
                    f"| `{mname}` | {sd} | **{s['four_cell_family_weighted']*100:.2f}%** | **{s['joint_t1_4cell_unit_weighted']*100:.2f}%** | "
                    f"{rb:.1f}% | {pos_s} | {info['ifeval']} / {info['gsm8k']} | `{info['wiki_kl']:.6f}` | {'PASS' if info['health']['all_passed'] else 'FAIL'} |"
                )
        md.append("")
        md.append("### Paired Comparisons")
        md.append("")
        md.append("| Comparison | Split | 4-Cell Diff [95% CI] | Joint Diff [95% CI] | p |")
        md.append("|---|---|---:|---:|---:|")
        for pname, cent in ddat["comparisons"].items():
            for sp, vals in cent.items():
                fc, jt = vals["four_cell"], vals["joint_t1_4cell"]
                md.append(f"| `{pname}` | `{sp}` | **{fc['diff_pp']:+.2f}** `{fc['ci95_pp']}` | **{jt['diff_pp']:+.2f}** `{jt['ci95_pp']}` | {fc['p_two_sided']:.4f} |")
        md.append("")
        md.append("### Verdicts & Fine-Localization Bonus")
        md.append("")
        md.append("```json")
        md.append(json.dumps({"verdicts": ddat["verdicts"], "fine_localization_bonus": ddat["fine_localization_bonus"]}, indent=2))
        md.append("```")
        md.append("")

    if out_doc["exfever"]:
        md.append(f"## External Natural-Language Benchmark: Ma et al. EX-FEVER (`1,000` Official Test, `{exfever_regime}`)")
        md.append("")
        md.append("| Model | Turn-1 Acc | `n_correct / n_wrong` | `R_UY` (↓) | `R_RU` Evidence (↑) | `R_RU` User-Evid (↑) | Selectivity (`R_RU - R_UY`) |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for k, h in out_doc["exfever"].items():
            md.append(
                f"| `{k}` | {h.get('acc')}% | {h.get('n_orig_correct')} / {h.get('n_orig_wrong')} | "
                f"**{h.get('r_uy')}%** ({h.get('r_uy_k_n')}) | **{h.get('r_ru_evidence')}%** ({h.get('r_ru_evidence_k_n')}) | "
                f"{h.get('r_ru_user_evidence')}% ({h.get('r_ru_user_evidence_k_n')}) | **{h.get('selectivity_ru_minus_uy')}** |"
            )
        md.append("")

    md_p = args.out_dir / f"C35_STAGE5_{scale.upper()}_READOUT.md"
    md_p.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] Wrote Stage 5 strict readout -> {md_p} & {json_p}")


if __name__ == "__main__":
    main()
