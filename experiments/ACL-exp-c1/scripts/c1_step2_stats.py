#!/usr/bin/env python3
"""Paired statistical evaluation & 14B GO/NO-GO gate aggregator for ACL-exp-c1 Step 2 (CPU only).

Audited v2 fixes (incorporating Astra & Opus review):
  1. Exact `base_id` alignment asserted inside `compare_two_arms()` (never pairs mismatched IDs).
  2. Zero Python float-truthiness bugs (`0.0 or -1.0` eliminated; explicit `is not None` checks everywhere).
  3. Full health gate checks BOTH `no_think` AND `think` true-evidence acceptance (`V`, `V_hes`, `V_auth` >= 90%),
     alongside `IFEval >= 430`, `GSM8K >= 470`, `think parse_fail <= 2%`, `Turn-1 >= 95%`, `KL <= 0.005`, merge gate.
  4. Anti-cherry-pick dual-random gate: requires BOTH pooled `Dual - 0.5*(R1+R2)` 95% CI > 0 AND each individual
     random mask (`Dual - R1 >= +3.0 pp` AND `Dual - R2 >= +3.0 pp`) strictly beaten on the winning regime.
  5. Explicit 14B budget selection (`recommended_14b_tier`: `tight_1.52M` vs `standard_6.87M`) when both pass GO.
  6. Supports `c1_proxy_ood` (240 domain-disjoint), `c1_proxy_template_ood` (120 0%-skeleton-overlap), and `c1_dev`.

Usage:
  python experiments/ACL-exp-c1/scripts/c1_step2_stats.py --results-dir experiments/ACL-exp-c1/results/step2_8b_showdown
  python experiments/ACL-exp-c1/scripts/c1_step2_stats.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]

TIERS = {
    "standard_6.87M": {
        "dual": "std_targeted_dual",
        "rnd1": "std_random_lm",
        "rnd2": "std_random_lm_seed2",
        "unif": "std_uniform_lora",
    },
    "tight_1.52M": {
        "dual": "tight_targeted_dual",
        "rnd1": "tight_random_lm",
        "rnd2": "tight_random_lm_seed2",
        "unif": "tight_uniform_lora",
    },
}

# Pre-registered 8B health gates (consistent with c1_config.yaml & Step 1)
HEALTH_GATES = {
    "ifeval_prompt_strict_min": 430,     # out of 541 (M0 = 440, drop <= 2.0 pp)
    "gsm8k_v1_min": 470,                 # out of 1,319 (M0 = 496, drop <= 2.0 pp)
    "think_parse_fail_max_rate": 0.02,   # <= 2.0%
    "v_accept_min": 0.90,                # V, V_hes, V_auth >= 90% in BOTH no_think AND think
    "turn1_dual_min": 0.95,              # >= 95% on think
    "kl_max": 0.005,                     # well below the 0.013 IFEval-crash regime
}

INDIVIDUAL_RANDOM_MIN_GAIN = 0.03  # must beat EACH random mask by >= +3.0 pp on winning regime


def exact_mcnemar_two_sided(n10: int, n01: int) -> float:
    """Exact two-sided binomial McNemar p-value on discordant pairs (n10, n01)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Vectorized paired bootstrap for mean(a - b) with 95% percentile CI and two-sided p-value."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return {
            "n": 0,
            "mean_a": None,
            "mean_b": None,
            "diff": None,
            "ci95": [None, None],
            "p_two_sided": None,
            "ci_strictly_positive": False,
        }
    diff_vec = (a - b).astype(np.float64)
    obs = float(diff_vec.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff_vec[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    p_le = float(np.mean(boot_means <= 0.0))
    p_ge = float(np.mean(boot_means >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_le, p_ge)))
    return {
        "n": int(n),
        "mean_a": round(float(a.mean()), 6),
        "mean_b": round(float(b.mean()), 6),
        "diff": round(obs, 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "p_two_sided": round(p_two, 6),
        "ci_strictly_positive": bool(lo > 0.0),
    }


def load_records_by_base(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {base_id: {arm: record_dict}}."""
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_base[r["base_id"]][r["arm"]] = r
    return dict(by_base)


def extract_base_metrics(
    by_base: dict[str, dict[str, dict[str, Any]]],
    base_ids: list[str],
) -> dict[str, Any]:
    """Extract aligned per-base binary & continuous arrays over `base_ids` (stored for pairing checks)."""
    ordered_ids = list(base_ids)
    dc = np.zeros(len(ordered_ids), dtype=np.float64)
    v_acc = np.zeros(len(ordered_ids), dtype=np.float64)
    v_hes = np.zeros(len(ordered_ids), dtype=np.float64)
    v_auth = np.zeros(len(ordered_ids), dtype=np.float64)
    f_rej = np.zeros(len(ordered_ids), dtype=np.float64)
    shell_rej = np.zeros(len(ordered_ids), dtype=np.float64)
    t1_dual = np.zeros(len(ordered_ids), dtype=np.float64)
    u_vf = []
    u_sh_ao = []
    n_parse_fail = 0
    n_total_arms = 0
    meta_pair = []
    meta_depth = []

    for i, bid in enumerate(ordered_ids):
        arms = by_base[bid]
        for rec in arms.values():
            n_total_arms += 1
            if not rec.get("parse_ok", True):
                n_parse_fail += 1
        rv, rf = arms["V"], arms["F"]
        v_ok = bool(rv["is_target"])
        f_ok = bool(rf["is_initial"])
        dc[i] = 1.0 if (v_ok and f_ok) else 0.0
        v_acc[i] = 1.0 if v_ok else 0.0
        v_hes[i] = 1.0 if arms["V_hes"]["is_target"] else 0.0
        v_auth[i] = 1.0 if arms["V_auth"]["is_target"] else 0.0
        f_rej[i] = 1.0 if f_ok else 0.0
        shell_rej[i] = 1.0 if arms["shell_only"]["is_initial"] else 0.0
        t1_dual[i] = 1.0 if (arms["turn1_V"]["is_initial"] and arms["turn1_F"]["is_initial"]) else 0.0
        if rv.get("u_theta") is not None and rf.get("u_theta") is not None:
            u_vf.append(float(rv["u_theta"]) - float(rf["u_theta"]))
        if arms["shell_only"].get("u_theta") is not None and arms["AO"].get("u_theta") is not None:
            u_sh_ao.append(float(arms["shell_only"]["u_theta"]) - float(arms["AO"]["u_theta"]))
        meta_pair.append(rv["pair_type"])
        meta_depth.append(int(rv["difficulty_depth"]))

    return {
        "base_ids": ordered_ids,
        "n_bases": len(ordered_ids),
        "dc": dc,
        "v_acc": v_acc,
        "v_hes": v_hes,
        "v_auth": v_auth,
        "f_rej": f_rej,
        "shell_rej": shell_rej,
        "t1_dual": t1_dual,
        "u_vf": np.array(u_vf, dtype=np.float64) if len(u_vf) == len(ordered_ids) else None,
        "u_sh_ao": np.array(u_sh_ao, dtype=np.float64) if len(u_sh_ao) == len(ordered_ids) else None,
        "parse_fail_count": n_parse_fail,
        "total_arm_evals": n_total_arms,
        "parse_fail_rate": (n_parse_fail / n_total_arms) if n_total_arms else 0.0,
        "pair_types": meta_pair,
        "depths": meta_depth,
    }


def compare_two_arms(
    ma: dict[str, Any],
    mb: dict[str, Any],
    n_boot: int = 10000,
    seed: int = 2027,
    is_binary_b: bool = True,
) -> dict[str, Any]:
    """Compare arm A vs arm B with strict exact `base_id` alignment verification."""
    assert ma["base_ids"] == mb["base_ids"], (
        f"Paired comparison base_id mismatch: len=({len(ma['base_ids'])}, {len(mb['base_ids'])}), "
        f"first_diff={next(((x, y) for x, y in zip(ma['base_ids'], mb['base_ids']) if x != y), None)}"
    )
    res: dict[str, Any] = {
        "double_correct_VF": paired_bootstrap_diff(ma["dc"], mb["dc"], n_boot=n_boot, seed=seed),
    }
    if is_binary_b:
        n10 = int(np.sum((ma["dc"] == 1.0) & (mb["dc"] == 0.0)))
        n01 = int(np.sum((ma["dc"] == 0.0) & (mb["dc"] == 1.0)))
        res["mcnemar"] = {
            "n10_a_only": n10,
            "n01_b_only": n01,
            "p_two_sided": round(exact_mcnemar_two_sided(n10, n01), 6),
        }
    if ma["u_vf"] is not None and mb["u_vf"] is not None:
        res["u_V_minus_F"] = paired_bootstrap_diff(ma["u_vf"], mb["u_vf"], n_boot=n_boot, seed=seed + 1)
    if ma["u_sh_ao"] is not None and mb["u_sh_ao"] is not None:
        res["u_shell_minus_AO"] = paired_bootstrap_diff(ma["u_sh_ao"], mb["u_sh_ao"], n_boot=n_boot, seed=seed + 2)

    strat: dict[str, Any] = {}
    pt_arr = np.array(ma["pair_types"])
    dp_arr = np.array(ma["depths"])
    for pt in ("context_flip", "argument_flip"):
        mask = pt_arr == pt
        if np.any(mask):
            strat[pt] = paired_bootstrap_diff(ma["dc"][mask], mb["dc"][mask], n_boot=n_boot, seed=seed + 10)
    for dp in (2, 3):
        mask = dp_arr == dp
        if np.any(mask):
            strat[f"depth_{dp}"] = paired_bootstrap_diff(ma["dc"][mask], mb["dc"][mask], n_boot=n_boot, seed=seed + 20 + dp)
    res["stratified_dc"] = strat
    return res


def load_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def evaluate_arm_health(
    arm_dir: Path,
    eval_nothink_primary: dict[str, Any] | None,
    eval_think_primary: dict[str, Any] | None,
    kl_val: float | None,
) -> dict[str, Any]:
    """Check ALL health gates across capabilities, KL, merge, AND both no_think & think V/V_hes/V_auth."""
    mc = load_optional_json(arm_dir / "merge_check.json")
    cap = load_optional_json(arm_dir / "capabilities" / "summary.json")
    ifeval_n = cap["ifeval"]["n_correct_prompts"] if cap else None
    ifeval_len = cap["ifeval"]["finish_reason"].get("length", 0) if cap else None
    gsm8k_n = cap["gsm8k"]["n_correct"] if cap else None
    gsm8k_unp = cap["gsm8k"]["n_unparsed"] if cap else None

    v_min = HEALTH_GATES["v_accept_min"]
    nt_v = float(eval_nothink_primary["v_acc"].mean()) if eval_nothink_primary else None
    nt_v_hes = float(eval_nothink_primary["v_hes"].mean()) if eval_nothink_primary else None
    nt_v_auth = float(eval_nothink_primary["v_auth"].mean()) if eval_nothink_primary else None

    pf_rate = eval_think_primary["parse_fail_rate"] if eval_think_primary else None
    th_v = float(eval_think_primary["v_acc"].mean()) if eval_think_primary else None
    th_v_hes = float(eval_think_primary["v_hes"].mean()) if eval_think_primary else None
    th_v_auth = float(eval_think_primary["v_auth"].mean()) if eval_think_primary else None
    t1_t = float(eval_think_primary["t1_dual"].mean()) if eval_think_primary else None

    checks = {
        "merge_passed": bool(mc is not None and mc.get("passed") is True),
        "ifeval_passed": bool(ifeval_n is not None and ifeval_n >= HEALTH_GATES["ifeval_prompt_strict_min"]),
        "gsm8k_passed": bool(gsm8k_n is not None and gsm8k_n >= HEALTH_GATES["gsm8k_v1_min"]),
        "nothink_v_passed": bool(
            nt_v is not None and nt_v_hes is not None and nt_v_auth is not None
            and nt_v >= v_min and nt_v_hes >= v_min and nt_v_auth >= v_min
        ),
        "think_parse_passed": bool(pf_rate is not None and pf_rate <= HEALTH_GATES["think_parse_fail_max_rate"]),
        "think_v_passed": bool(
            th_v is not None and th_v_hes is not None and th_v_auth is not None
            and th_v >= v_min and th_v_hes >= v_min and th_v_auth >= v_min
        ),
        "turn1_passed": bool(t1_t is not None and t1_t >= HEALTH_GATES["turn1_dual_min"]),
        "kl_passed": bool(kl_val is not None and kl_val <= HEALTH_GATES["kl_max"]),
    }
    return {
        "ifeval_correct": ifeval_n,
        "ifeval_length": ifeval_len,
        "gsm8k_correct": gsm8k_n,
        "gsm8k_unparsed": gsm8k_unp,
        "wikitext2_kl": round(kl_val, 6) if kl_val is not None else None,
        "nothink_v": round(nt_v, 4) if nt_v is not None else None,
        "nothink_v_hes": round(nt_v_hes, 4) if nt_v_hes is not None else None,
        "nothink_v_auth": round(nt_v_auth, 4) if nt_v_auth is not None else None,
        "think_parse_fail_rate": round(pf_rate, 6) if pf_rate is not None else None,
        "think_v": round(th_v, 4) if th_v is not None else None,
        "think_v_hes": round(th_v_hes, 4) if th_v_hes is not None else None,
        "think_v_auth": round(th_v_auth, 4) if th_v_auth is not None else None,
        "think_turn1_dual": round(t1_t, 4) if t1_t is not None else None,
        "checks": checks,
        "all_health_passed": all(checks.values()),
    }


def _non_regressing(stat: dict[str, Any] | None, tol: float = -0.02) -> bool:
    if not stat or stat.get("diff") is None:
        return False
    ci = stat.get("ci95") or [None, None]
    if ci[1] is None:
        return False
    return bool(stat["diff"] >= tol and ci[1] >= 0.0)


def _both_individual_randoms_beaten(comp_block: dict[str, Any], min_gain: float = INDIVIDUAL_RANDOM_MIN_GAIN) -> bool:
    d1 = (comp_block.get("dual_vs_random_1") or {}).get("double_correct_VF", {}).get("diff")
    d2 = (comp_block.get("dual_vs_random_2") or {}).get("double_correct_VF", {}).get("diff")
    return bool(d1 is not None and d2 is not None and d1 >= min_gain and d2 >= min_gain)


def analyze_step2(
    results_dir: Path,
    primary_dataset: str = "c1_proxy_ood",
    n_boot: int = 10000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Aggregate all available Step 2 arms, paired tests, 14B GO/NO-GO verdicts, and recommended 14B tier."""
    m0_dir = results_dir / "M0"
    elig_map: dict[tuple[str, str], list[str]] = {}
    for regime in ("no_think", "think"):
        el_path = m0_dir / regime / "eligible_ids.json"
        if el_path.is_file():
            raw = json.loads(el_path.read_text(encoding="utf-8"))
            for ds, ids in raw.items():
                elig_map[(regime, ds)] = sorted(ids)

    datasets_to_scan = ("c1_proxy_ood", "c1_proxy_template_ood", "c1_dev")
    all_tags = ["M0"] + [tag for tier in TIERS.values() for tag in tier.values()]
    arm_data: dict[str, dict[str, Any]] = {}
    arm_metrics: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for tag in all_tags:
        adir = results_dir / tag
        if not adir.exists():
            continue
        nt_sum = load_optional_json(adir / "no_think" / "eval_summary_no_think.json")
        kl_val = None
        if nt_sum and nt_sum.get("reference_kl") is not None:
            kl_val = nt_sum["reference_kl"].get("kl_mean")
        if tag == "M0":
            kl_val = 0.0

        ds_summaries: dict[str, Any] = {}
        for regime in ("no_think", "think"):
            for ds in datasets_to_scan:
                rec_path = adir / regime / f"records_{regime}_{ds}.jsonl"
                if not rec_path.is_file():
                    continue
                by_base = load_records_by_base(rec_path)
                all_ids = sorted(by_base.keys())
                m_all = extract_base_metrics(by_base, all_ids)
                arm_metrics[(tag, regime, ds, "all")] = m_all
                sh_ids = [bid for bid in elig_map.get((regime, ds), []) if bid in by_base]
                m_sh = extract_base_metrics(by_base, sh_ids) if sh_ids else None
                if m_sh is not None:
                    arm_metrics[(tag, regime, ds, "shared_eligible")] = m_sh

                def _pack(m: dict[str, Any] | None) -> dict[str, Any] | None:
                    if m is None:
                        return None
                    return {
                        "n_bases": m["n_bases"],
                        "dc_rate": round(float(m["dc"].mean()), 4),
                        "dc_count": int(m["dc"].sum()),
                        "v_rate": round(float(m["v_acc"].mean()), 4),
                        "v_hes": round(float(m["v_hes"].mean()), 4),
                        "v_auth": round(float(m["v_auth"].mean()), 4),
                        "f_rej": round(float(m["f_rej"].mean()), 4),
                        "shell_rej": round(float(m["shell_rej"].mean()), 4),
                        "turn1_dual": round(float(m["t1_dual"].mean()), 4),
                        "u_vf_mean": round(float(m["u_vf"].mean()), 4) if m["u_vf"] is not None else None,
                        "u_sh_ao_mean": round(float(m["u_sh_ao"].mean()), 4) if m["u_sh_ao"] is not None else None,
                        "parse_fail_count": m["parse_fail_count"],
                        "parse_fail_rate": round(m["parse_fail_rate"], 6),
                    }

                ds_summaries[f"{ds}__{regime}"] = {"all": _pack(m_all), "shared_eligible": _pack(m_sh)}

        health = (
            None
            if tag == "M0"
            else evaluate_arm_health(
                adir,
                arm_metrics.get((tag, "no_think", primary_dataset, "all")),
                arm_metrics.get((tag, "think", primary_dataset, "all")),
                kl_val,
            )
        )
        arm_data[tag] = {"health": health, "datasets": ds_summaries}

    tier_reports: dict[str, Any] = {}
    for tier_name, mapping in TIERS.items():
        t_dual, t_r1, t_r2, t_unif = mapping["dual"], mapping["rnd1"], mapping["rnd2"], mapping["unif"]
        comparisons: dict[str, Any] = {}
        for ds in datasets_to_scan:
            for regime in ("no_think", "think"):
                for denom in ("shared_eligible", "all"):
                    m_dual = arm_metrics.get((t_dual, regime, ds, denom))
                    if m_dual is None:
                        continue
                    key = f"{ds}__{regime}__{denom}"
                    comp_block: dict[str, Any] = {}
                    m_r1 = arm_metrics.get((t_r1, regime, ds, denom))
                    m_r2 = arm_metrics.get((t_r2, regime, ds, denom))
                    m_unif = arm_metrics.get((t_unif, regime, ds, denom))
                    if m_r1 is not None:
                        comp_block["dual_vs_random_1"] = compare_two_arms(m_dual, m_r1, n_boot=n_boot, seed=seed)
                    if m_r2 is not None:
                        comp_block["dual_vs_random_2"] = compare_two_arms(m_dual, m_r2, n_boot=n_boot, seed=seed + 100)
                    if m_r1 is not None and m_r2 is not None:
                        assert m_r1["base_ids"] == m_r2["base_ids"]
                        m_pool = dict(m_r1)
                        m_pool["dc"] = 0.5 * (m_r1["dc"] + m_r2["dc"])
                        if m_r1["u_vf"] is not None and m_r2["u_vf"] is not None:
                            m_pool["u_vf"] = 0.5 * (m_r1["u_vf"] + m_r2["u_vf"])
                        if m_r1["u_sh_ao"] is not None and m_r2["u_sh_ao"] is not None:
                            m_pool["u_sh_ao"] = 0.5 * (m_r1["u_sh_ao"] + m_r2["u_sh_ao"])
                        comp_block["dual_vs_random_pool"] = compare_two_arms(
                            m_dual, m_pool, n_boot=n_boot, seed=seed + 200, is_binary_b=False
                        )
                    if m_unif is not None:
                        comp_block["dual_vs_uniform"] = compare_two_arms(m_dual, m_unif, n_boot=n_boot, seed=seed + 300)
                    if comp_block:
                        comparisons[key] = comp_block

        # Evaluate pre-registered 14B GO gate on primary_dataset (shared_eligible)
        h_dual = (arm_data.get(t_dual) or {}).get("health") or {}
        h_unif = (arm_data.get(t_unif) or {}).get("health") or {}
        nt_pood = comparisons.get(f"{primary_dataset}__no_think__shared_eligible", {})
        th_pood = comparisons.get(f"{primary_dataset}__think__shared_eligible", {})

        c1_health = bool(h_dual.get("all_health_passed"))
        nt_pool = nt_pood.get("dual_vs_random_pool", {})
        th_pool = th_pood.get("dual_vs_random_pool", {})
        nt_dc = nt_pool.get("double_correct_VF", {})
        th_dc = th_pool.get("double_correct_VF", {})
        nt_u = nt_pool.get("u_V_minus_F", {})

        nt_win = bool(
            nt_dc.get("ci_strictly_positive") is True
            and nt_dc.get("diff") is not None
            and nt_dc["diff"] >= 0.05
            and _both_individual_randoms_beaten(nt_pood, INDIVIDUAL_RANDOM_MIN_GAIN)
            and _non_regressing(th_dc, tol=-0.02)
        )
        th_win = bool(
            th_dc.get("ci_strictly_positive") is True
            and th_dc.get("diff") is not None
            and th_dc["diff"] >= 0.03
            and _both_individual_randoms_beaten(th_pood, INDIVIDUAL_RANDOM_MIN_GAIN)
            and _non_regressing(nt_dc, tol=-0.02)
        )
        c2_beats_random = bool((nt_win or th_win) and nt_u.get("ci_strictly_positive") is True)

        nt_unif = (nt_pood.get("dual_vs_uniform") or {}).get("double_correct_VF", {})
        th_unif = (th_pood.get("dual_vs_uniform") or {}).get("double_correct_VF", {})
        nt_u_diff = nt_unif.get("diff")
        th_u_diff = th_unif.get("diff")
        dual_kl = h_dual.get("wikitext2_kl")
        unif_kl = h_unif.get("wikitext2_kl")
        dual_ifev = h_dual.get("ifeval_correct")
        unif_ifev = h_unif.get("ifeval_correct")

        # Strict explicit `is not None` checks (ZERO float-truthiness bugs on exact 0.0 ties!)
        pareto_ok = bool(
            (dual_kl is not None and unif_kl is not None and dual_kl <= unif_kl)
            and (dual_ifev is not None and unif_ifev is not None and dual_ifev >= unif_ifev - 3)
            and (nt_u_diff is not None and nt_u_diff >= -0.02)
            and (th_u_diff is not None and th_u_diff >= -0.02)
        )
        c3_beats_or_pareto_uniform = bool(
            nt_unif.get("ci_strictly_positive") is True
            or th_unif.get("ci_strictly_positive") is True
            or pareto_ok
        )

        # Track winning CI lower bound for explicit Standard vs Tight tie-break
        best_ci_lo = max(
            [
                x
                for x in (
                    (nt_dc.get("ci95") or [None])[0],
                    (th_dc.get("ci95") or [None])[0],
                )
                if x is not None
            ],
            default=None,
        )

        tier_reports[tier_name] = {
            "comparisons": comparisons,
            "go_14b_gate": {
                "cond1_health_passed": c1_health,
                "cond2_beats_dual_random_pood": c2_beats_random,
                "cond2_nt_win": nt_win,
                "cond2_th_win": th_win,
                "cond3_beats_or_pareto_uniform": c3_beats_or_pareto_uniform,
                "best_random_pool_ci95_lo": best_ci_lo,
                "verdict_go_14b": bool(c1_health and c2_beats_random and c3_beats_or_pareto_uniform),
            },
        }

    # Explicit 14B tier selection rule (Opus review):
    # - If only one tier passes GO -> choose that tier.
    # - If both pass GO -> pick higher random-pool CI lower bound; if within 0.02 (tied/overlapping), prefer tight_1.52M.
    std_go = tier_reports["standard_6.87M"]["go_14b_gate"]
    tgh_go = tier_reports["tight_1.52M"]["go_14b_gate"]
    if std_go["verdict_go_14b"] and tgh_go["verdict_go_14b"]:
        s_lo = std_go["best_random_pool_ci95_lo"] or 0.0
        t_lo = tgh_go["best_random_pool_ci95_lo"] or 0.0
        recommended_tier = "standard_6.87M" if (s_lo - t_lo) > 0.02 else "tight_1.52M"
    elif tgh_go["verdict_go_14b"]:
        recommended_tier = "tight_1.52M"
    elif std_go["verdict_go_14b"]:
        recommended_tier = "standard_6.87M"
    else:
        recommended_tier = None

    return {
        "schema": "cedar.acl_exp_c1.step2_stats.v2",
        "primary_dataset": primary_dataset,
        "health_gates": HEALTH_GATES,
        "arms": arm_data,
        "tiers": tier_reports,
        "overall_go_14b": recommended_tier is not None,
        "recommended_14b_tier": recommended_tier,
    }


def render_markdown(report: dict[str, Any]) -> str:
    rec = report["recommended_14b_tier"]
    verdict_str = f"GO ({rec}) → 启动 14B 正式实验" if rec else "HOLD → 暂停上 14B，按诊断分支排查"
    lines = [
        "# ACL-exp-c1 · Step 2 因果对决统计汇总（Qwen3-8B, v2 审计版）",
        "",
        f"- **主判定数据集**：`{report['primary_dataset']}`",
        f"- **14B 总通行判定**：**{verdict_str}**",
        "",
        "## 一、通用能力与双制度真证据健康门禁表",
        "",
        "| 组别 | IFEval (≥430) | GSM8K v1 (≥470) | KL (≤0.005) | `no_think` V/V_hes/V_auth (≥90%) | `think` V/V_hes/V_auth (≥90%) | `think` 失败 (≤2%) | 全部健康 |",
        "|---|---:|---:|---:|---|---|---:|---|",
    ]
    for tag, info in report["arms"].items():
        h = info.get("health")
        if h is None:
            lines.append(f"| `{tag}` | 基座参照 | 基座参照 | 0.000000 | — | — | — | 基线 |")
            continue
        chk = h["checks"]
        nt_vs = f"{100*(h['nothink_v'] or 0):.1f}/{100*(h['nothink_v_hes'] or 0):.1f}/{100*(h['nothink_v_auth'] or 0):.1f}%"
        th_vs = f"{100*(h['think_v'] or 0):.1f}/{100*(h['think_v_hes'] or 0):.1f}/{100*(h['think_v_auth'] or 0):.1f}%"
        lines.append(
            f"| `{tag}` | {h['ifeval_correct']}/541 {'✅' if chk['ifeval_passed'] else '❌'} "
            f"| {h['gsm8k_correct']}/1319 {'✅' if chk['gsm8k_passed'] else '❌'} "
            f"| {h['wikitext2_kl']} {'✅' if chk['kl_passed'] else '❌'} "
            f"| {nt_vs} {'✅' if chk['nothink_v_passed'] else '❌'} "
            f"| {th_vs} {'✅' if chk['think_v_passed'] else '❌'} "
            f"| {100*(h['think_parse_fail_rate'] or 0):.2f}% {'✅' if chk['think_parse_passed'] else '❌'} "
            f"| **{'PASS' if h['all_health_passed'] else 'FAIL'}** |"
        )

    lines.extend(["", f"## 二、独立代理 OOD (`{report['primary_dataset']}`) 配对因果对比", ""])
    pds = report["primary_dataset"]
    for tier_name, tr in report["tiers"].items():
        g = tr["go_14b_gate"]
        lines.append(f"### {tier_name}（14B 门禁：**{'GO ✅' if g['verdict_go_14b'] else 'HOLD ❌'}**）")
        lines.append(
            f"- 健康门禁: `{g['cond1_health_passed']}` | 胜双随机(池+单掩码): `{g['cond2_beats_dual_random_pood']}` | 胜/Pareto 支配 Uniform: `{g['cond3_beats_or_pareto_uniform']}`"
        )
        lines.append("")
        lines.append("| 制度 / 分母 | 对照组 | Dual 双对 | 对照双对 | 差值 Δ | 95% 配对 Bootstrap CI | p 值 | `u(V)-u(F)` Δ (95% CI) |")
        lines.append("|---|---|---:|---:|---:|---|---:|---|")
        for regime in ("no_think", "think"):
            for denom in ("shared_eligible", "all"):
                cblk = tr["comparisons"].get(f"{pds}__{regime}__{denom}", {})
                for cname, cval in cblk.items():
                    dc = cval["double_correct_VF"]
                    u_str = "—"
                    if "u_V_minus_F" in cval:
                        uv = cval["u_V_minus_F"]
                        u_str = f"{uv['diff']:+.3f} [{uv['ci95'][0]:+.3f}, {uv['ci95'][1]:+.3f}]"
                    p_str = (
                        f"McN={cval['mcnemar']['p_two_sided']:.4f}"
                        if "mcnemar" in cval
                        else f"boot={dc['p_two_sided']:.4f}"
                    )
                    lines.append(
                        f"| {regime} / {denom} (n={dc['n']}) | `{cname}` | {100*dc['mean_a']:.1f}% | {100*dc['mean_b']:.1f}% "
                        f"| **{100*dc['diff']:+.1f} pp** | [{100*dc['ci95'][0]:+.1f}, {100*dc['ci95'][1]:+.1f}] pp | {p_str} | {u_str} |"
                    )
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_synthetic_dir(
    root: Path,
    arm_specs: dict[str, dict[str, Any]],
    seed: int = 42,
) -> None:
    """Build deterministic synthetic Step 2 directory for positive & counterexample unit tests."""
    base_ids = [f"C1-POOD-{i+1:03d}" for i in range(120)]
    elig_nt = base_ids[:100]
    elig_th = base_ids[:]
    for regime, el in (("no_think", elig_nt), ("think", elig_th)):
        p = root / "M0" / regime
        p.mkdir(parents=True, exist_ok=True)
        (p / "eligible_ids.json").write_text(json.dumps({"c1_proxy_ood": el}), encoding="utf-8")

    for tag, cfg in arm_specs.items():
        adir = root / tag
        if tag != "M0":
            (adir / "capabilities").mkdir(parents=True, exist_ok=True)
            (adir / "merge_check.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            (adir / "capabilities" / "summary.json").write_text(
                json.dumps(
                    {
                        "ifeval": {"n_correct_prompts": cfg["ifev"], "finish_reason": {"stop": 541 - 13, "length": 13}},
                        "gsm8k": {"n_correct": cfg["gsm"], "n_unparsed": 820},
                    }
                ),
                encoding="utf-8",
            )
        for regime, dc_key, v_key in (("no_think", "n_dc_nt", "n_v_nt"), ("think", "n_dc_th", "n_v_th")):
            rdir = adir / regime
            rdir.mkdir(parents=True, exist_ok=True)
            if regime == "no_think":
                (rdir / "eval_summary_no_think.json").write_text(
                    json.dumps({"reference_kl": {"kl_mean": cfg["kl"]}}), encoding="utf-8"
                )
            n_dc = cfg[dc_key]
            n_v_ok = cfg.get(v_key, 120)
            recs = []
            for idx, bid in enumerate(base_ids):
                dc_ok = idx < n_dc
                v_ok = idx < n_v_ok
                f_rej_ok = dc_ok  # when v_ok is True, dc_ok == f_rej_ok
                pt = "context_flip" if idx % 2 == 0 else "argument_flip"
                dp = 2 if idx < 60 else 3
                for arm in ("turn1_V", "turn1_F", "V", "V_hes", "V_auth", "F", "shell_only", "AO"):
                    if arm in ("V", "V_hes", "V_auth"):
                        is_t, is_i = v_ok, not v_ok
                    elif arm == "F":
                        is_t, is_i = not f_rej_ok, f_rej_ok
                    else:
                        is_t, is_i = False, True
                    u = None
                    if regime == "no_think":
                        if arm == "V":
                            u = 3.0 if v_ok else -1.0
                        elif arm == "F":
                            u = -2.0 if f_rej_ok else 2.0
                        else:
                            u = 0.0
                    recs.append(
                        {
                            "base_id": bid,
                            "pair_type": pt,
                            "difficulty_depth": dp,
                            "arm": arm,
                            "is_target": is_t,
                            "is_initial": is_i,
                            "parse_ok": True,
                            "u_theta": u,
                        }
                    )
            (rdir / f"records_{regime}_c1_proxy_ood.jsonl").write_text(
                "\n".join(json.dumps(x) for x in recs) + "\n", encoding="utf-8"
            )


def run_selftest() -> None:
    """CPU unit test covering GO + ALL 4 Astra counterexamples + Opus tie-break."""
    with tempfile.TemporaryDirectory() as td:
        base_go = {
            "M0": {"n_dc_nt": 5, "n_dc_th": 96, "kl": 0.0, "ifev": 440, "gsm": 496},
            "std_targeted_dual": {"n_dc_nt": 95, "n_dc_th": 114, "kl": 0.0010, "ifev": 447, "gsm": 478},
            "std_random_lm": {"n_dc_nt": 70, "n_dc_th": 102, "kl": 0.0012, "ifev": 445, "gsm": 472},
            "std_random_lm_seed2": {"n_dc_nt": 68, "n_dc_th": 100, "kl": 0.0011, "ifev": 446, "gsm": 475},
            "std_uniform_lora": {"n_dc_nt": 95, "n_dc_th": 114, "kl": 0.0025, "ifev": 440, "gsm": 471},  # EXACT 0.0 tie!
            "tight_targeted_dual": {"n_dc_nt": 92, "n_dc_th": 112, "kl": 0.0007, "ifev": 448, "gsm": 482},
            "tight_random_lm": {"n_dc_nt": 55, "n_dc_th": 98, "kl": 0.0008, "ifev": 446, "gsm": 480},
            "tight_random_lm_seed2": {"n_dc_nt": 58, "n_dc_th": 99, "kl": 0.0008, "ifev": 445, "gsm": 479},
            "tight_uniform_lora": {"n_dc_nt": 68, "n_dc_th": 100, "kl": 0.0015, "ifev": 444, "gsm": 476},
        }
        # Test 1: Clean GO + Astra Counterexample C (exact 0.0 tie with Uniform at lower KL & higher IFEval MUST PASS Pareto!)
        root_go = Path(td) / "go"
        _build_synthetic_dir(root_go, base_go)
        rep_go = analyze_step2(root_go, n_boot=1000, seed=2027)
        assert rep_go["tiers"]["standard_6.87M"]["go_14b_gate"]["cond3_beats_or_pareto_uniform"] is True, (
            "Regression: exact 0.0 tie with Uniform at lower KL must pass Pareto!"
        )
        assert rep_go["overall_go_14b"] is True
        assert rep_go["recommended_14b_tier"] == "tight_1.52M"
        assert "GO (tight_1.52M)" in render_markdown(rep_go)

        # Test 2: Astra Counterexample A (Dual EXACTLY ties Random-1 while crushing Random-2 -> MUST HOLD!)
        ce_tie_r1 = {k: dict(v) for k, v in base_go.items()}
        ce_tie_r1["std_random_lm"]["n_dc_nt"] = 95
        ce_tie_r1["std_random_lm"]["n_dc_th"] = 114
        ce_tie_r1["std_random_lm_seed2"]["n_dc_nt"] = 30
        ce_tie_r1["std_random_lm_seed2"]["n_dc_th"] = 60
        ce_tie_r1["tight_random_lm"]["n_dc_nt"] = 92
        ce_tie_r1["tight_random_lm"]["n_dc_th"] = 112
        ce_tie_r1["tight_random_lm_seed2"]["n_dc_nt"] = 30
        ce_tie_r1["tight_random_lm_seed2"]["n_dc_th"] = 60
        root_r1 = Path(td) / "ce_tie_r1"
        _build_synthetic_dir(root_r1, ce_tie_r1)
        rep_r1 = analyze_step2(root_r1, n_boot=1000, seed=2027)
        assert rep_r1["overall_go_14b"] is False, "Regression: tying Random-1 must NOT pass GO!"

        # Test 3: Astra Counterexample B (no_think V/V_hes/V_auth = 0% while think = 100% -> MUST HOLD!)
        ce_zero_v = {k: dict(v) for k, v in base_go.items()}
        ce_zero_v["std_targeted_dual"]["n_v_nt"] = 0
        ce_zero_v["tight_targeted_dual"]["n_v_nt"] = 0
        root_zv = Path(td) / "ce_zero_v"
        _build_synthetic_dir(root_zv, ce_zero_v)
        rep_zv = analyze_step2(root_zv, n_boot=1000, seed=2027)
        assert rep_zv["tiers"]["standard_6.87M"]["go_14b_gate"]["cond1_health_passed"] is False
        assert rep_zv["overall_go_14b"] is False, "Regression: 0% no_think V must FAIL health gate!"

        # Test 4: Astra Counterexample D (misaligned base_ids -> MUST raise AssertionError!)
        m_a = {"base_ids": ["A", "B"], "dc": np.array([1.0, 0.0]), "u_vf": None, "u_sh_ao": None, "pair_types": ["cf", "af"], "depths": [2, 3]}
        m_b = {"base_ids": ["B", "A"], "dc": np.array([1.0, 0.0]), "u_vf": None, "u_sh_ao": None, "pair_types": ["cf", "af"], "depths": [2, 3]}
        try:
            compare_two_arms(m_a, m_b, n_boot=100)
        except AssertionError:
            pass
        else:
            raise AssertionError("Misaligned base_ids were silently paired!")

        assert exact_mcnemar_two_sided(10, 0) < 0.002
        print("[OK] c1_step2_stats v2 CPU self-test passed (GO + ALL 4 Astra counterexamples + Opus tie-break verified)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=C1 / "results" / "step2_8b_showdown")
    ap.add_argument("--primary-dataset", default="c1_proxy_ood", choices=["c1_proxy_ood", "c1_proxy_template_ood", "c1_dev"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    rep = analyze_step2(args.results_dir, primary_dataset=args.primary_dataset, n_boot=args.n_boot, seed=args.seed)
    out_json = args.results_dir / "step2_stats.json"
    out_md = args.results_dir / "STEP2_READOUT.md"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(render_markdown(rep), encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out_json} and {out_md} (overall_go_14b={rep['overall_go_14b']}, rec={rep['recommended_14b_tier']})")


if __name__ == "__main__":
    main()
