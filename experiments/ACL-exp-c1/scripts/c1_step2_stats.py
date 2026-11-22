#!/usr/bin/env python3
"""Paired statistical evaluation & 14B GO/CANDIDATE/HOLD/STOP gate aggregator for ACL-exp-c1 Step 2B (v4, CPU only).

Audited v4 fixes (incorporating ALL Astra pre-flight audit findings):
  1. Large-N exact McNemar in log-space (`math.lgamma`): handles n=1,200+ (600:600) with zero float underflow.
  2. Cluster-aware paired bootstrap: automatically clusters by `cell_id` on 2x2 crossed confirmation sets.
  3. Zero silent fallbacks:
     - Per-arm `vllm_lora_parity.json` required (with `passed=True`, `u_coverage_rate=1.0`, non-empty SHAs).
     - `m_pool["u_vf"]` & `m_pool["u_sh_ao"]` explicitly set to `None` if EITHER `Random-1` or `Random-2` lacks them.
     - Both `primary_dataset` AND `c1_dev` are ALWAYS required in `no_think` health splits; missing files FAIL!
  4. Strict 4-state verdict (`GO` / `CANDIDATE` / `HOLD` / `STOP`):
     - Single-seed can AT MOST reach `CANDIDATE` (`overall_go_14b=False`).
     - `GO` (`overall_go_14b=True`) strictly requires:
       (a) `n_shared_eligible >= min_eligible_bases` (default 800),
       (b) all multi-split dual-regime health & per-arm parity gates passed,
       (c) pooled `>= +5 pp` (CI > 0), BOTH individual randoms `>= +3 pp` (CI > 0), `u(V)-u(F)` CI > 0,
           true `think` non-inferiority (`ci95[0] >= -0.02`), and beating/Pareto-dominating Uniform, AND
       (d) verified distinct `seed=1024` directional replication (`results_dir != seed2_dir`, distinct `seed`
           metadata `42 != 1024`, distinct checkpoint SHA-256).
     - Explicit `HOLD` vs `STOP`: if CI upper bound still allows `>= +5 pp` (or point diff `> +2 pp`), returns `HOLD`;
       only when both tiers rule out `>= +5 pp` at `n >= min_eligible_bases` with point diff `<= +2 pp` returns `STOP`.

Usage:
  python experiments/ACL-exp-c1/scripts/c1_step2_stats.py \
      --results-dir experiments/ACL-exp-c1/results/step2b_8b_showdown \
      --primary-dataset c1_step2b_confirm_ce --think-dataset c1_dev
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

ALL_SCANNED_DATASETS = (
    "c1_step2b_confirm_ce",
    "c1_proxy_ood",
    "c1_proxy_template_v2",
    "c1_proxy_template_ood",
    "c1_dev",
)

HEALTH_GATES = {
    "ifeval_prompt_strict_min": 430,     # out of 541 (M0 = 440, drop <= 2.0 pp)
    "gsm8k_v1_min": 470,                 # out of 1,319 (M0 = 496, drop <= 2.0 pp)
    "think_parse_fail_max_rate": 0.02,   # <= 2.0% across all designated think splits
    "v_accept_min": 0.90,                # V, V_hes, V_auth >= 90% in BOTH no_think AND think across all health splits
    "turn1_dual_min": 0.95,              # >= 95% in BOTH no_think AND think across all health splits
    "kl_max": 0.005,                     # well below the 0.013 IFEval-crash regime
}

POOLED_RANDOM_MIN_GAIN = 0.05      # >= +5.0 pp pooled on primary endpoint
INDIVIDUAL_RANDOM_MIN_GAIN = 0.03  # >= +3.0 pp AND 95% CI > 0 against EACH random mask
NON_INFERIORITY_TOL = -0.02        # true non-inferiority: 95% CI LOWER bound >= -2.0 pp
DEFAULT_MIN_ELIGIBLE_BASES = 800   # 80% power floor for a 5 pp primary gap


def exact_mcnemar_two_sided(n10: int, n01: int) -> float:
    """Numerically stable exact two-sided binomial McNemar p-value (log-space, zero underflow at large N)."""
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    log_denom = n * math.log(2.0)
    lg_n1 = math.lgamma(n + 1)
    log_terms = [lg_n1 - math.lgamma(i + 1) - math.lgamma(n - i + 1) - log_denom for i in range(k + 1)]
    m = max(log_terms)
    tail = math.exp(m) * sum(math.exp(lt - m) for lt in log_terms)
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    clusters: list[str] | None = None,
    n_boot: int = 10000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Vectorized paired bootstrap (cluster-resampled by `cell_id` when clusters are present)."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return {
            "n": 0,
            "n_clusters": 0,
            "clustered": False,
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

    use_clusters = bool(clusters is not None and len(clusters) == n and all(c is not None for c in clusters))
    if use_clusters:
        by_c: dict[str, list[int]] = defaultdict(list)
        for idx, cid in enumerate(clusters):  # type: ignore[arg-type]
            by_c[cid].append(idx)
        c_keys = sorted(by_c.keys())
        c_sums = np.array([diff_vec[by_c[k]].sum() for k in c_keys], dtype=np.float64)
        c_cnts = np.array([len(by_c[k]) for k in c_keys], dtype=np.float64)
        n_c = len(c_keys)
        c_idx = rng.integers(0, n_c, size=(n_boot, n_c))
        boot_means = c_sums[c_idx].sum(axis=1) / c_cnts[c_idx].sum(axis=1)
    else:
        n_c = n
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = diff_vec[idx].mean(axis=1)

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    p_le = float(np.mean(boot_means <= 0.0))
    p_ge = float(np.mean(boot_means >= 0.0))
    p_two = float(min(1.0, 2.0 * min(p_le, p_ge)))
    return {
        "n": int(n),
        "n_clusters": int(n_c),
        "clustered": bool(use_clusters and n_c < n),
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
    """Extract aligned per-base binary, continuous, and cluster (`cell_id`) arrays over `base_ids`."""
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
    cell_ids: list[str | None] = []

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
        cell_ids.append(rv.get("cell_id"))

    clusters = cell_ids if all(c is not None for c in cell_ids) else None
    return {
        "base_ids": ordered_ids,
        "clusters": clusters,
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
    """Compare arm A vs arm B with strict exact `base_id` alignment and cluster-aware bootstrap."""
    assert ma["base_ids"] == mb["base_ids"], (
        f"Paired comparison base_id mismatch: len=({len(ma['base_ids'])}, {len(mb['base_ids'])}), "
        f"first_diff={next(((x, y) for x, y in zip(ma['base_ids'], mb['base_ids']) if x != y), None)}"
    )
    clusters = ma.get("clusters")
    res: dict[str, Any] = {
        "double_correct_VF": paired_bootstrap_diff(ma["dc"], mb["dc"], clusters=clusters, n_boot=n_boot, seed=seed),
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
        res["u_V_minus_F"] = paired_bootstrap_diff(ma["u_vf"], mb["u_vf"], clusters=clusters, n_boot=n_boot, seed=seed + 1)
    if ma["u_sh_ao"] is not None and mb["u_sh_ao"] is not None:
        res["u_shell_minus_AO"] = paired_bootstrap_diff(ma["u_sh_ao"], mb["u_sh_ao"], clusters=clusters, n_boot=n_boot, seed=seed + 2)

    strat: dict[str, Any] = {}
    pt_arr = np.array(ma["pair_types"])
    dp_arr = np.array(ma["depths"])
    for pt in ("crossed_2x2", "context_flip", "argument_flip"):
        mask = pt_arr == pt
        if np.any(mask):
            sub_cl = [clusters[i] for i in range(len(mask)) if mask[i]] if clusters is not None else None
            strat[pt] = paired_bootstrap_diff(ma["dc"][mask], mb["dc"][mask], clusters=sub_cl, n_boot=n_boot, seed=seed + 10)
    for dp in (2, 3):
        mask = dp_arr == dp
        if np.any(mask):
            sub_cl = [clusters[i] for i in range(len(mask)) if mask[i]] if clusters is not None else None
            strat[f"depth_{dp}"] = paired_bootstrap_diff(ma["dc"][mask], mb["dc"][mask], clusters=sub_cl, n_boot=n_boot, seed=seed + 20 + dp)
    res["stratified_dc"] = strat
    return res


def load_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _verify_arm_parity(arm_dir: Path) -> tuple[bool, str]:
    """Require per-arm SHA-bound `vllm_lora_parity.json` with 100% 4-option coverage (NO global/legacy fallback!)."""
    p = load_optional_json(arm_dir / "vllm_lora_parity.json")
    if p is None:
        return False, "missing_arm_vllm_lora_parity"
    lora_block = p.get("lora_hf_live_vs_vllm_native") or {}
    m0_block = p.get("m0_hf_vs_vllm") or {}
    ok = bool(
        p.get("passed") is True
        and lora_block.get("passed") is True
        and m0_block.get("passed") is True
        and lora_block.get("u_coverage_rate") == 1.0
        and m0_block.get("u_coverage_rate") == 1.0
        and bool(p.get("adapter_state_sha256"))
        and bool(p.get("peft_adapter_model_sha256"))
        and isinstance(p.get("vllm_max_lora_rank"), int)
    )
    return ok, (f"arm_parity_r{p.get('vllm_max_lora_rank')}" if ok else "arm_parity_failed_or_incomplete")


def evaluate_arm_health(
    arm_dir: Path,
    nothink_health_splits: dict[str, dict[str, Any]],
    think_health_splits: dict[str, dict[str, Any]],
    required_nothink_splits: tuple[str, ...],
    required_think_splits: tuple[str, ...],
    expected_base_counts: dict[str, int],
    kl_val: float | None,
) -> dict[str, Any]:
    """Check ALL v4 health gates across capabilities, KL, per-arm SHA-bound parity, AND ALL required splits."""
    fidelity_passed, fidelity_source = _verify_arm_parity(arm_dir)

    cap = load_optional_json(arm_dir / "capabilities" / "summary.json")
    ifeval_n = cap["ifeval"]["n_correct_prompts"] if cap else None
    ifeval_len = cap["ifeval"]["finish_reason"].get("length", 0) if cap else None
    gsm8k_n = cap["gsm8k"]["n_correct"] if cap else None
    gsm8k_unp = cap["gsm8k"]["n_unparsed"] if cap else None

    v_min = HEALTH_GATES["v_accept_min"]
    t1_min = HEALTH_GATES["turn1_dual_min"]
    pf_max = HEALTH_GATES["think_parse_fail_max_rate"]

    by_split: dict[str, Any] = {"no_think": {}, "think": {}}

    def _split_complete(ds: str, m: dict[str, Any]) -> bool:
        exp_n = expected_base_counts.get(ds)
        return True if exp_n is None else (m["n_bases"] == exp_n)

    nt_all_req_present = (
        len(required_nothink_splits) > 0
        and all(ds in nothink_health_splits and _split_complete(ds, nothink_health_splits[ds]) for ds in required_nothink_splits)
    )
    nt_v_ok = nt_all_req_present
    nt_t1_ok = nt_all_req_present
    nt_v_min = nt_v_hes_min = nt_v_auth_min = nt_t1_min = None
    for ds, m in sorted(nothink_health_splits.items()):
        v = float(m["v_acc"].mean())
        vh = float(m["v_hes"].mean())
        va = float(m["v_auth"].mean())
        t1 = float(m["t1_dual"].mean())
        comp = _split_complete(ds, m)
        by_split["no_think"][ds] = {
            "n_bases": m["n_bases"],
            "complete": comp,
            "v": round(v, 4),
            "v_hes": round(vh, 4),
            "v_auth": round(va, 4),
            "turn1_dual": round(t1, 4),
            "v_passed": bool(comp and v >= v_min and vh >= v_min and va >= v_min),
            "turn1_passed": bool(comp and t1 >= t1_min),
        }
        nt_v_ok = nt_v_ok and by_split["no_think"][ds]["v_passed"]
        nt_t1_ok = nt_t1_ok and by_split["no_think"][ds]["turn1_passed"]
        nt_v_min = v if nt_v_min is None else min(nt_v_min, v)
        nt_v_hes_min = vh if nt_v_hes_min is None else min(nt_v_hes_min, vh)
        nt_v_auth_min = va if nt_v_auth_min is None else min(nt_v_auth_min, va)
        nt_t1_min = t1 if nt_t1_min is None else min(nt_t1_min, t1)

    th_all_req_present = (
        len(required_think_splits) > 0
        and all(ds in think_health_splits and _split_complete(ds, think_health_splits[ds]) for ds in required_think_splits)
    )
    th_v_ok = th_all_req_present
    th_t1_ok = th_all_req_present
    th_pf_ok = th_all_req_present
    th_v_min = th_v_hes_min = th_v_auth_min = th_t1_min = th_pf_max = None
    for ds, m in sorted(think_health_splits.items()):
        v = float(m["v_acc"].mean())
        vh = float(m["v_hes"].mean())
        va = float(m["v_auth"].mean())
        t1 = float(m["t1_dual"].mean())
        pf = float(m["parse_fail_rate"])
        comp = _split_complete(ds, m)
        by_split["think"][ds] = {
            "n_bases": m["n_bases"],
            "complete": comp,
            "v": round(v, 4),
            "v_hes": round(vh, 4),
            "v_auth": round(va, 4),
            "turn1_dual": round(t1, 4),
            "parse_fail_rate": round(pf, 6),
            "v_passed": bool(comp and v >= v_min and vh >= v_min and va >= v_min),
            "turn1_passed": bool(comp and t1 >= t1_min),
            "parse_passed": bool(comp and pf <= pf_max),
        }
        th_v_ok = th_v_ok and by_split["think"][ds]["v_passed"]
        th_t1_ok = th_t1_ok and by_split["think"][ds]["turn1_passed"]
        th_pf_ok = th_pf_ok and by_split["think"][ds]["parse_passed"]
        th_v_min = v if th_v_min is None else min(th_v_min, v)
        th_v_hes_min = vh if th_v_hes_min is None else min(th_v_hes_min, vh)
        th_v_auth_min = va if th_v_auth_min is None else min(th_v_auth_min, va)
        th_t1_min = t1 if th_t1_min is None else min(th_t1_min, t1)
        th_pf_max = pf if th_pf_max is None else max(th_pf_max, pf)

    checks = {
        "fidelity_passed": fidelity_passed,
        "required_nothink_splits_complete": nt_all_req_present,
        "required_think_splits_complete": th_all_req_present,
        "ifeval_passed": bool(ifeval_n is not None and ifeval_n >= HEALTH_GATES["ifeval_prompt_strict_min"]),
        "gsm8k_passed": bool(gsm8k_n is not None and gsm8k_n >= HEALTH_GATES["gsm8k_v1_min"]),
        "nothink_v_passed": bool(nt_v_ok),
        "nothink_turn1_passed": bool(nt_t1_ok),
        "think_parse_passed": bool(th_pf_ok),
        "think_v_passed": bool(th_v_ok),
        "think_turn1_passed": bool(th_t1_ok),
        "kl_passed": bool(kl_val is not None and kl_val <= HEALTH_GATES["kl_max"]),
    }
    return {
        "fidelity_source": fidelity_source,
        "ifeval_correct": ifeval_n,
        "ifeval_length": ifeval_len,
        "gsm8k_correct": gsm8k_n,
        "gsm8k_unparsed": gsm8k_unp,
        "wikitext2_kl": round(kl_val, 6) if kl_val is not None else None,
        "nothink_v_min": round(nt_v_min, 4) if nt_v_min is not None else None,
        "nothink_v_hes_min": round(nt_v_hes_min, 4) if nt_v_hes_min is not None else None,
        "nothink_v_auth_min": round(nt_v_auth_min, 4) if nt_v_auth_min is not None else None,
        "nothink_turn1_min": round(nt_t1_min, 4) if nt_t1_min is not None else None,
        "think_parse_fail_max": round(th_pf_max, 6) if th_pf_max is not None else None,
        "think_v_min": round(th_v_min, 4) if th_v_min is not None else None,
        "think_v_hes_min": round(th_v_hes_min, 4) if th_v_hes_min is not None else None,
        "think_v_auth_min": round(th_v_auth_min, 4) if th_v_auth_min is not None else None,
        "think_turn1_min": round(th_t1_min, 4) if th_t1_min is not None else None,
        "by_split": by_split,
        "checks": checks,
        "all_health_passed": all(checks.values()),
    }


def _non_regressing(stat: dict[str, Any] | None, tol: float = NON_INFERIORITY_TOL) -> bool:
    """True non-inferiority: 95% paired bootstrap CI LOWER bound must be >= tol (default -0.02)."""
    if not stat or stat.get("diff") is None:
        return False
    ci = stat.get("ci95") or [None, None]
    if ci[0] is None:
        return False
    return bool(ci[0] >= tol)


def _both_individual_randoms_beaten(
    comp_block: dict[str, Any],
    min_gain: float = INDIVIDUAL_RANDOM_MIN_GAIN,
) -> bool:
    """Require BOTH Dual vs Random-1 AND Dual vs Random-2 to have 95% CI lower bound > 0 AND diff >= min_gain."""
    c1 = (comp_block.get("dual_vs_random_1") or {}).get("double_correct_VF") or {}
    c2 = (comp_block.get("dual_vs_random_2") or {}).get("double_correct_VF") or {}
    d1, d2 = c1.get("diff"), c2.get("diff")
    return bool(
        c1.get("ci_strictly_positive") is True
        and c2.get("ci_strictly_positive") is True
        and d1 is not None
        and d2 is not None
        and d1 >= min_gain
        and d2 >= min_gain
    )


def _read_seed_evidence(arm_dir: Path) -> tuple[int | None, str | None]:
    """Read `(seed, adapter_sha256)` from `train_meta.json` or `seed_meta.json` inside `arm_dir`."""
    tm = load_optional_json(arm_dir / "train_meta.json")
    sm = load_optional_json(arm_dir / "seed_meta.json")
    seed_val = None
    sha_val = None
    if tm:
        seed_val = (tm.get("hyperparams") or {}).get("seed")
        sha_val = tm.get("adapter_state_sha256")
    if sm:
        if seed_val is None:
            seed_val = sm.get("seed")
        if sha_val is None:
            sha_val = sm.get("adapter_state_sha256")
    if sha_val is None:
        par = load_optional_json(arm_dir / "vllm_lora_parity.json")
        if par:
            sha_val = par.get("adapter_state_sha256")
    return (int(seed_val) if seed_val is not None else None), sha_val


def _load_seed2_directional_check(
    results_dir: Path,
    seed2_dir: Path,
    tier_name: str,
    primary_dataset: str,
    elig_ids: list[str],
    min_eligible_bases: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Verify TRUE independent cross-seed (seed=1024) directional replication (rejects reused dir/seed/SHA!)."""
    if results_dir.resolve() == seed2_dir.resolve():
        return {"status": "rejected_same_directory", "passed": False}

    mapping = TIERS[tier_name]
    t_dual, t_r1, t_r2 = mapping["dual"], mapping["rnd1"], mapping["rnd2"]

    seed_audit: dict[str, Any] = {}
    for tag in (t_dual, t_r1, t_r2):
        s1, sha1 = _read_seed_evidence(results_dir / tag)
        s2, sha2 = _read_seed_evidence(seed2_dir / tag)
        seed_audit[tag] = {"seed1": s1, "seed2": s2, "sha1": sha1, "sha2": sha2}
        if s1 is None or s2 is None or s1 == s2:
            return {"status": "rejected_missing_or_identical_seed", "seed_audit": seed_audit, "passed": False}
        if not sha1 or not sha2 or sha1 == sha2:
            return {"status": "rejected_missing_or_identical_adapter_sha", "seed_audit": seed_audit, "passed": False}

    loaded: dict[str, dict[str, Any]] = {}
    for tag in (t_dual, t_r1, t_r2):
        rec_path = seed2_dir / tag / "no_think" / f"records_no_think_{primary_dataset}.jsonl"
        if not rec_path.is_file():
            return {"status": "missing_records", "missing": str(rec_path), "passed": False}
        by_base = load_records_by_base(rec_path)
        sh_ids = [bid for bid in elig_ids if bid in by_base]
        if len(sh_ids) < min_eligible_bases or len(sh_ids) != len(elig_ids):
            return {
                "status": "insufficient_or_incomplete_shared_eligible",
                "n_found": len(sh_ids),
                "n_expected": len(elig_ids),
                "min_required": min_eligible_bases,
                "passed": False,
            }
        loaded[tag] = extract_base_metrics(by_base, sh_ids)

    c_r1 = compare_two_arms(loaded[t_dual], loaded[t_r1], n_boot=n_boot, seed=seed + 500)
    c_r2 = compare_two_arms(loaded[t_dual], loaded[t_r2], n_boot=n_boot, seed=seed + 600)
    d1 = c_r1["double_correct_VF"]["diff"]
    d2 = c_r2["double_correct_VF"]["diff"]
    m_d = loaded[t_dual]
    v_min = HEALTH_GATES["v_accept_min"]
    t1_min = HEALTH_GATES["turn1_dual_min"]
    nt_health_ok = bool(
        float(m_d["v_acc"].mean()) >= v_min
        and float(m_d["v_hes"].mean()) >= v_min
        and float(m_d["v_auth"].mean()) >= v_min
        and float(m_d["t1_dual"].mean()) >= t1_min
    )
    passed = bool(d1 is not None and d2 is not None and d1 > 0.0 and d2 > 0.0 and nt_health_ok)
    return {
        "status": "verified" if passed else "failed_direction_or_health",
        "tier": tier_name,
        "dataset": primary_dataset,
        "seed_audit": seed_audit,
        "dual_vs_random_1_dc": c_r1["double_correct_VF"],
        "dual_vs_random_2_dc": c_r2["double_correct_VF"],
        "dual_nothink_health_ok": nt_health_ok,
        "passed": passed,
    }


def analyze_step2(
    results_dir: Path,
    primary_dataset: str = "c1_step2b_confirm_ce",
    think_dataset: str | None = None,
    min_eligible_bases: int = DEFAULT_MIN_ELIGIBLE_BASES,
    seed2_dir: Path | None = None,
    n_boot: int = 10000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Aggregate Step 2B arms, cluster-paired tests, multi-split health gates, and 4-state verdict."""
    if think_dataset is None:
        think_dataset = "c1_dev" if primary_dataset == "c1_step2b_confirm_ce" else primary_dataset

    m0_dir = results_dir / "M0"
    elig_map: dict[tuple[str, str], list[str]] = {}
    for regime in ("no_think", "think"):
        el_path = m0_dir / regime / "eligible_ids.json"
        if el_path.is_file():
            raw = json.loads(el_path.read_text(encoding="utf-8"))
            for ds, ids in raw.items():
                elig_map[(regime, ds)] = sorted(ids)

    # Determine expected full base counts from M0 records if present (catches silently truncated files!)
    expected_base_counts: dict[str, int] = {}
    for ds in ALL_SCANNED_DATASETS:
        for regime in ("no_think", "think"):
            mp = m0_dir / regime / f"records_{regime}_{ds}.jsonl"
            if mp.is_file():
                expected_base_counts[ds] = max(expected_base_counts.get(ds, 0), len(load_records_by_base(mp)))

    all_tags = ["M0"] + [tag for tier in TIERS.values() for tag in tier.values()]
    arm_data: dict[str, dict[str, Any]] = {}
    arm_metrics: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    req_nt_splits = tuple(dict.fromkeys((primary_dataset, "c1_dev")))
    req_th_splits = (think_dataset,)

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
            for ds in ALL_SCANNED_DATASETS:
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

        nt_splits_for_health = {
            ds: arm_metrics[(tag, "no_think", ds, "all")]
            for ds in req_nt_splits
            if (tag, "no_think", ds, "all") in arm_metrics
        }
        th_splits_for_health = {
            ds: arm_metrics[(tag, "think", ds, "all")]
            for ds in req_th_splits
            if (tag, "think", ds, "all") in arm_metrics
        }
        health = (
            None
            if tag == "M0"
            else evaluate_arm_health(
                arm_dir=adir,
                nothink_health_splits=nt_splits_for_health,
                think_health_splits=th_splits_for_health,
                required_nothink_splits=req_nt_splits,
                required_think_splits=req_th_splits,
                expected_base_counts=expected_base_counts,
                kl_val=kl_val,
            )
        )
        arm_data[tag] = {"health": health, "datasets": ds_summaries}

    n_primary_elig = len(elig_map.get(("no_think", primary_dataset), []))
    primary_power_ok = bool(n_primary_elig >= min_eligible_bases)

    tier_reports: dict[str, Any] = {}
    for tier_name, mapping in TIERS.items():
        t_dual, t_r1, t_r2, t_unif = mapping["dual"], mapping["rnd1"], mapping["rnd2"], mapping["unif"]
        comparisons: dict[str, Any] = {}
        for ds in ALL_SCANNED_DATASETS:
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
                        # NEVER silently reuse Random-1's continuous margins if Random-2 is missing them!
                        if m_r1["u_vf"] is not None and m_r2["u_vf"] is not None:
                            m_pool["u_vf"] = 0.5 * (m_r1["u_vf"] + m_r2["u_vf"])
                        else:
                            m_pool["u_vf"] = None
                        if m_r1["u_sh_ao"] is not None and m_r2["u_sh_ao"] is not None:
                            m_pool["u_sh_ao"] = 0.5 * (m_r1["u_sh_ao"] + m_r2["u_sh_ao"])
                        else:
                            m_pool["u_sh_ao"] = None
                        comp_block["dual_vs_random_pool"] = compare_two_arms(
                            m_dual, m_pool, n_boot=n_boot, seed=seed + 200, is_binary_b=False
                        )
                    if m_unif is not None:
                        comp_block["dual_vs_uniform"] = compare_two_arms(m_dual, m_unif, n_boot=n_boot, seed=seed + 300)
                    if comp_block:
                        comparisons[key] = comp_block

        h_dual = (arm_data.get(t_dual) or {}).get("health") or {}
        h_unif = (arm_data.get(t_unif) or {}).get("health") or {}
        nt_prim = comparisons.get(f"{primary_dataset}__no_think__shared_eligible", {})
        th_comp = comparisons.get(f"{think_dataset}__think__shared_eligible", {})

        c0_power = bool(
            primary_power_ok
            and (nt_prim.get("dual_vs_random_pool") or {}).get("double_correct_VF", {}).get("n") == n_primary_elig
        )
        c1_health = bool(h_dual.get("all_health_passed"))
        nt_pool = nt_prim.get("dual_vs_random_pool", {})
        th_pool = th_comp.get("dual_vs_random_pool", {})
        nt_dc = nt_pool.get("double_correct_VF", {})
        th_dc = th_pool.get("double_correct_VF", {})
        nt_u = nt_pool.get("u_V_minus_F", {})

        c2_beats_random = bool(
            nt_dc.get("ci_strictly_positive") is True
            and nt_dc.get("diff") is not None
            and nt_dc["diff"] >= POOLED_RANDOM_MIN_GAIN
            and _both_individual_randoms_beaten(nt_prim, INDIVIDUAL_RANDOM_MIN_GAIN)
            and nt_u.get("ci_strictly_positive") is True
            and _non_regressing(th_dc, tol=NON_INFERIORITY_TOL)
        )

        nt_unif = (nt_prim.get("dual_vs_uniform") or {}).get("double_correct_VF", {})
        th_unif = (th_comp.get("dual_vs_uniform") or {}).get("double_correct_VF", {})
        dual_kl = h_dual.get("wikitext2_kl")
        unif_kl = h_unif.get("wikitext2_kl")
        dual_ifev = h_dual.get("ifeval_correct")
        unif_ifev = h_unif.get("ifeval_correct")

        pareto_ok = bool(
            (dual_kl is not None and unif_kl is not None and dual_kl <= unif_kl)
            and (dual_ifev is not None and unif_ifev is not None and dual_ifev >= unif_ifev - 3)
            and _non_regressing(nt_unif, tol=NON_INFERIORITY_TOL)
            and _non_regressing(th_unif, tol=NON_INFERIORITY_TOL)
        )
        c3_beats_or_pareto_uniform = bool(
            nt_unif.get("ci_strictly_positive") is True
            or th_unif.get("ci_strictly_positive") is True
            or pareto_ok
        )

        primary_ci_lo = (nt_dc.get("ci95") or [None, None])[0]
        primary_ci_hi = (nt_dc.get("ci95") or [None, None])[1]
        tier_reports[tier_name] = {
            "comparisons": comparisons,
            "go_14b_gate": {
                "cond0_min_eligible_power": c0_power,
                "cond1_health_passed": c1_health,
                "cond2_beats_dual_random_primary": c2_beats_random,
                "cond2_think_non_inferior": _non_regressing(th_dc, tol=NON_INFERIORITY_TOL),
                "cond2_both_individual_randoms_ci_gt0": _both_individual_randoms_beaten(nt_prim, INDIVIDUAL_RANDOM_MIN_GAIN),
                "cond3_beats_or_pareto_uniform": c3_beats_or_pareto_uniform,
                "cond3_pareto_ok": pareto_ok,
                "primary_random_pool_diff": nt_dc.get("diff"),
                "primary_random_pool_ci95_lo": primary_ci_lo,
                "primary_random_pool_ci95_hi": primary_ci_hi,
                "verdict_single_seed_pass": bool(c0_power and c1_health and c2_beats_random and c3_beats_or_pareto_uniform),
            },
        }

    # Tier selection among single-seed passing tiers
    std_go = tier_reports["standard_6.87M"]["go_14b_gate"]
    tgh_go = tier_reports["tight_1.52M"]["go_14b_gate"]
    if std_go["verdict_single_seed_pass"] and tgh_go["verdict_single_seed_pass"]:
        s_lo = std_go["primary_random_pool_ci95_lo"] or 0.0
        t_lo = tgh_go["primary_random_pool_ci95_lo"] or 0.0
        candidate_tier = "standard_6.87M" if (s_lo - t_lo) > 0.02 else "tight_1.52M"
    elif tgh_go["verdict_single_seed_pass"]:
        candidate_tier = "tight_1.52M"
    elif std_go["verdict_single_seed_pass"]:
        candidate_tier = "standard_6.87M"
    else:
        candidate_tier = None

    # Strict 4-state verdict: GO | CANDIDATE | HOLD | STOP
    seed2_rep: dict[str, Any] | None = None
    if candidate_tier is not None:
        if seed2_dir is not None:
            seed2_rep = _load_seed2_directional_check(
                results_dir=results_dir,
                seed2_dir=seed2_dir,
                tier_name=candidate_tier,
                primary_dataset=primary_dataset,
                elig_ids=elig_map.get(("no_think", primary_dataset), []),
                min_eligible_bases=min_eligible_bases,
                n_boot=n_boot,
                seed=seed,
            )
            if seed2_rep["passed"]:
                verdict = "GO"
                overall_go = True
                recommended_tier = candidate_tier
            else:
                verdict = "HOLD"
                overall_go = False
                recommended_tier = None
        else:
            # Single seed NEVER unlocks overall_go_14b=True!
            verdict = "CANDIDATE"
            overall_go = False
            recommended_tier = None
    else:
        overall_go = False
        recommended_tier = None
        # Distinguish STOP vs HOLD: STOP only when BOTH tiers are powered AND rule out a >= +5 pp gain (CI_hi < 0.05 & diff <= 0.02)
        both_powered = bool(std_go["cond0_min_eligible_power"] and tgh_go["cond0_min_eligible_power"])
        both_ruled_out = bool(
            both_powered
            and std_go["primary_random_pool_diff"] is not None
            and tgh_go["primary_random_pool_diff"] is not None
            and std_go["primary_random_pool_ci95_hi"] is not None
            and tgh_go["primary_random_pool_ci95_hi"] is not None
            and std_go["primary_random_pool_diff"] <= 0.02
            and tgh_go["primary_random_pool_diff"] <= 0.02
            and std_go["primary_random_pool_ci95_hi"] < POOLED_RANDOM_MIN_GAIN
            and tgh_go["primary_random_pool_ci95_hi"] < POOLED_RANDOM_MIN_GAIN
        )
        verdict = "STOP" if both_ruled_out else "HOLD"

    return {
        "schema": "cedar.acl_exp_c1.step2_stats.v4",
        "primary_dataset": primary_dataset,
        "think_dataset": think_dataset,
        "min_eligible_bases": min_eligible_bases,
        "n_primary_shared_eligible": n_primary_elig,
        "health_gates": HEALTH_GATES,
        "non_inferiority_tol": NON_INFERIORITY_TOL,
        "arms": arm_data,
        "tiers": tier_reports,
        "candidate_single_seed_tier": candidate_tier,
        "seed2_replication": seed2_rep,
        "verdict": verdict,
        "overall_go_14b": overall_go,
        "recommended_14b_tier": recommended_tier,
    }


def render_markdown(report: dict[str, Any]) -> str:
    v = report["verdict"]
    rec = report["recommended_14b_tier"]
    cand = report["candidate_single_seed_tier"]
    if v == "GO":
        verdict_str = f"GO ({rec}, seed=1024 verified) → 启动 14B 正式实验"
    elif v == "CANDIDATE":
        verdict_str = f"CANDIDATE ({cand}) → 单种子达标，待补跑独立 seed=1024 方向复现后解锁 14B"
    elif v == "STOP":
        verdict_str = "STOP → 两档在大样本下均已排除 ≥+5 pp 随机掩码增益，按预注册方案转向 Pareto/部署保真主叙事"
    else:
        verdict_str = "HOLD → 暂缓上 14B，按预注册诊断分支排查"

    lines = [
        "# ACL-exp-c1 · Step 2B 因果对决统计汇总（Qwen3-8B, v4 审计版）",
        "",
        f"- **唯一预注册主终点**：`{report['primary_dataset']}` (`no_think`, M0 共享合格分母 `n={report['n_primary_shared_eligible']}`, 门槛 `≥ {report['min_eligible_bases']}`)",
        f"- **`think` 非劣性对照集**：`{report['think_dataset']}` (95% CI 下界 `≥ {100*report['non_inferiority_tol']:+.1f} pp`)",
        f"- **14B 最终四态判定**：**{verdict_str}** (`overall_go_14b={report['overall_go_14b']}`)",
        "",
        "## 一、通用能力与多切片双制度健康门禁表（取全部必需切片最严值）",
        "",
        "| 组别 | 逐组保真校验 | IFEval (≥430) | GSM8K v1 (≥470) | KL (≤0.005) | `no_think` V/V_hes/V_auth / T1 | `think` V/V_hes/V_auth / T1 / 失败 | 全部健康 |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for tag, info in report["arms"].items():
        h = info.get("health")
        if h is None:
            lines.append(f"| `{tag}` | 基线 | 基座参照 | 基座参照 | 0.000000 | — | — | 基线 |")
            continue
        chk = h["checks"]
        nt_vs = (
            f"{100*(h['nothink_v_min'] or 0):.1f}/{100*(h['nothink_v_hes_min'] or 0):.1f}/"
            f"{100*(h['nothink_v_auth_min'] or 0):.1f}% / T1={100*(h['nothink_turn1_min'] or 0):.1f}%"
        )
        th_vs = (
            f"{100*(h['think_v_min'] or 0):.1f}/{100*(h['think_v_hes_min'] or 0):.1f}/"
            f"{100*(h['think_v_auth_min'] or 0):.1f}% / T1={100*(h['think_turn1_min'] or 0):.1f}% / "
            f"{100*(h['think_parse_fail_max'] or 0):.2f}%"
        )
        lines.append(
            f"| `{tag}` | {h['fidelity_source']} {'✅' if chk['fidelity_passed'] else '❌'} "
            f"| {h['ifeval_correct']}/541 {'✅' if chk['ifeval_passed'] else '❌'} "
            f"| {h['gsm8k_correct']}/1319 {'✅' if chk['gsm8k_passed'] else '❌'} "
            f"| {h['wikitext2_kl']} {'✅' if chk['kl_passed'] else '❌'} "
            f"| {nt_vs} {'✅' if (chk['nothink_v_passed'] and chk['nothink_turn1_passed']) else '❌'} "
            f"| {th_vs} {'✅' if (chk['think_v_passed'] and chk['think_turn1_passed'] and chk['think_parse_passed']) else '❌'} "
            f"| **{'PASS' if h['all_health_passed'] else 'FAIL'}** |"
        )

    lines.extend(["", f"## 二、预注册主终点 (`{report['primary_dataset']}`) 与 `think` 非劣性 (`{report['think_dataset']}`) 配对因果对比", ""])
    pds = report["primary_dataset"]
    tds = report["think_dataset"]
    for tier_name, tr in report["tiers"].items():
        g = tr["go_14b_gate"]
        lines.append(f"### {tier_name}（单种子门禁：**{'PASS ✅' if g['verdict_single_seed_pass'] else 'HOLD ❌'}**）")
        lines.append(
            f"- 检出力(`≥{report['min_eligible_bases']}`): `{g['cond0_min_eligible_power']}` | 健康: `{g['cond1_health_passed']}` "
            f"| 主终点胜双随机: `{g['cond2_beats_dual_random_primary']}` | 胜/Pareto Uniform: `{g['cond3_beats_or_pareto_uniform']}`"
        )
        lines.append("")
        lines.append("| 切片 / 制度 / 分母 | 对照组 | Dual 双对 | 对照双对 | 差值 Δ | 95% 配对 Bootstrap CI | p 值 | `u(V)-u(F)` Δ (95% CI) |")
        lines.append("|---|---|---:|---:|---:|---|---:|---|")
        for ds, regime in ((pds, "no_think"), (tds, "think")):
            for denom in ("shared_eligible", "all"):
                cblk = tr["comparisons"].get(f"{ds}__{regime}__{denom}", {})
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
                    cl_tag = f" [cl={dc['n_clusters']}]" if dc.get("clustered") else ""
                    lines.append(
                        f"| `{ds}` / {regime} / {denom} (n={dc['n']}{cl_tag}) | `{cname}` | {100*dc['mean_a']:.1f}% | {100*dc['mean_b']:.1f}% "
                        f"| **{100*dc['diff']:+.1f} pp** | [{100*dc['ci95'][0]:+.1f}, {100*dc['ci95'][1]:+.1f}] pp | {p_str} | {u_str} |"
                    )
        lines.append("")
    s2 = report["seed2_replication"]
    if s2 is not None:
        lines.extend(
            [
                "## 三、跨种子方向复现 (`seed=1024`)",
                f"- 状态：`{s2.get('status')}` | 档位：`{s2.get('tier')}` | 通过：**{s2.get('passed')}**",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_synthetic_dir(
    root: Path,
    arm_specs: dict[str, dict[str, Any]],
    *,
    primary_ds: str = "c1_step2b_confirm_ce",
    think_ds: str = "c1_dev",
    n_primary: int = 200,
    n_primary_elig: int = 160,
    n_dev: int = 80,
    train_seed: int = 42,
    sha_salt: str = "s1",
) -> None:
    """Build deterministic synthetic Step 2B directory with 2x2 cell_ids, per-arm parity, and seed metadata."""
    assert n_primary % 2 == 0
    p_ids = [f"C1-S2BCE-{i+1:04d}" for i in range(n_primary)]
    d_ids = [f"C1-DEV-{i+1:03d}" for i in range(n_dev)]
    root.mkdir(parents=True, exist_ok=True)

    (root / "M0" / "no_think").mkdir(parents=True, exist_ok=True)
    (root / "M0" / "no_think" / "eligible_ids.json").write_text(
        json.dumps({primary_ds: p_ids[:n_primary_elig], think_ds: d_ids[: int(0.8 * n_dev)]}),
        encoding="utf-8",
    )
    (root / "M0" / "think").mkdir(parents=True, exist_ok=True)
    (root / "M0" / "think" / "eligible_ids.json").write_text(
        json.dumps({think_ds: d_ids[:]}),
        encoding="utf-8",
    )

    def _emit_records(
        ids: list[str],
        regime: str,
        dc_mask: list[bool],
        v_ok_n: int,
        v_hes_ok_n: int,
        t1_ok_n: int,
        *,
        with_cells: bool,
        drop_u: bool = False,
    ) -> list[dict[str, Any]]:
        recs = []
        for idx, bid in enumerate(ids):
            dc_ok = dc_mask[idx]
            v_ok = idx < v_ok_n
            vh_ok = idx < v_hes_ok_n
            f_rej_ok = dc_ok
            t1_ok = idx < t1_ok_n
            pt = "crossed_2x2" if with_cells else "context_flip"
            cid = f"CELL-{idx // 2 + 1:04d}" if with_cells else None
            dp = 2 if idx < len(ids) // 2 else 3
            for arm in ("turn1_V", "turn1_F", "V", "V_hes", "V_auth", "F", "shell_only", "AO"):
                if arm == "V":
                    is_t, is_i = v_ok, not v_ok
                elif arm == "V_hes":
                    is_t, is_i = vh_ok, not vh_ok
                elif arm == "V_auth":
                    is_t, is_i = v_ok, not v_ok
                elif arm == "F":
                    is_t, is_i = not f_rej_ok, f_rej_ok
                elif arm in ("turn1_V", "turn1_F"):
                    is_t, is_i = False, t1_ok
                else:
                    is_t, is_i = False, True
                u = None
                if regime == "no_think" and not drop_u:
                    if arm == "V":
                        u = 3.0 if v_ok else -1.0
                    elif arm == "F":
                        u = -2.0 if f_rej_ok else 2.0
                    else:
                        u = 0.0
                recs.append(
                    {
                        "base_id": bid,
                        "cell_id": cid,
                        "pair_type": pt,
                        "difficulty_depth": dp,
                        "arm": arm,
                        "is_target": is_t,
                        "is_initial": is_i,
                        "parse_ok": True,
                        "u_theta": u,
                    }
                )
        return recs

    for tag, cfg in arm_specs.items():
        adir = root / tag
        if tag != "M0":
            (adir / "capabilities").mkdir(parents=True, exist_ok=True)
            (adir / "capabilities" / "summary.json").write_text(
                json.dumps(
                    {
                        "ifeval": {"n_correct_prompts": cfg["ifev"], "finish_reason": {"stop": 541 - 13, "length": 13}},
                        "gsm8k": {"n_correct": cfg["gsm"], "n_unparsed": 20},
                    }
                ),
                encoding="utf-8",
            )
            if not cfg.get("omit_parity"):
                (adir / "vllm_lora_parity.json").write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "vllm_max_lora_rank": 32 if tag.startswith("tight") else 128,
                            "adapter_state_sha256": f"sha_{sha_salt}_{tag}",
                            "peft_adapter_model_sha256": f"peft_{sha_salt}_{tag}",
                            "m0_hf_vs_vllm": {"passed": True, "u_coverage_rate": 1.0},
                            "lora_hf_live_vs_vllm_native": {"passed": True, "u_coverage_rate": 1.0},
                        }
                    ),
                    encoding="utf-8",
                )
            (adir / "train_meta.json").write_text(
                json.dumps({"hyperparams": {"seed": train_seed}, "adapter_state_sha256": f"sha_{sha_salt}_{tag}"}),
                encoding="utf-8",
            )

        nt_dir = adir / "no_think"
        nt_dir.mkdir(parents=True, exist_ok=True)
        (nt_dir / "eval_summary_no_think.json").write_text(
            json.dumps({"reference_kl": {"kl_mean": cfg["kl"]}}), encoding="utf-8"
        )
        p_dc_mask = cfg.get("p_dc_mask") or [i < cfg["n_dc_nt"] for i in range(n_primary)]
        p_recs = _emit_records(
            p_ids,
            "no_think",
            p_dc_mask,
            cfg.get("n_v_nt", n_primary),
            cfg.get("n_vh_nt", n_primary),
            cfg.get("n_t1_nt", n_primary),
            with_cells=True,
            drop_u=bool(cfg.get("drop_u_nt")),
        )
        (nt_dir / f"records_no_think_{primary_ds}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in p_recs) + "\n", encoding="utf-8"
        )
        if not cfg.get("omit_dev_nt"):
            d_nt_mask = [i < int(0.85 * n_dev) for i in range(n_dev)]
            d_nt_recs = _emit_records(
                d_ids,
                "no_think",
                d_nt_mask,
                cfg.get("n_v_dev_nt", n_dev),
                cfg.get("n_vh_dev_nt", n_dev),
                cfg.get("n_t1_dev_nt", n_dev),
                with_cells=False,
            )
            (nt_dir / f"records_no_think_{think_ds}.jsonl").write_text(
                "\n".join(json.dumps(x) for x in d_nt_recs) + "\n", encoding="utf-8"
            )

        th_dir = adir / "think"
        th_dir.mkdir(parents=True, exist_ok=True)
        th_dc_mask = cfg.get("th_dc_mask") or [i < cfg["n_dc_th"] for i in range(n_dev)]
        th_recs = _emit_records(
            d_ids,
            "think",
            th_dc_mask,
            cfg.get("n_v_th", n_dev),
            cfg.get("n_vh_th", n_dev),
            cfg.get("n_t1_th", n_dev),
            with_cells=False,
        )
        (th_dir / f"records_think_{think_ds}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in th_recs) + "\n", encoding="utf-8"
        )


def run_selftest() -> None:
    """CPU unit test covering GO + ALL 12 Astra/Opus counterexamples (v2 + v3 + v4)."""
    # 0. Large-N exact McNemar numerical stability (Astra's 600:600 overflow test!)
    assert abs(exact_mcnemar_two_sided(600, 600) - 1.0) < 1e-6
    assert 0.05 < exact_mcnemar_two_sided(630, 570) < 0.10
    assert exact_mcnemar_two_sided(10, 0) < 0.002

    with tempfile.TemporaryDirectory() as td:
        base_go = {
            "M0": {"n_dc_nt": 10, "n_dc_th": 64, "kl": 0.0, "ifev": 440, "gsm": 496},
            "std_targeted_dual": {"n_dc_nt": 156, "n_dc_th": 76, "kl": 0.0010, "ifev": 447, "gsm": 478},
            "std_random_lm": {"n_dc_nt": 116, "n_dc_th": 72, "kl": 0.0012, "ifev": 445, "gsm": 472},
            "std_random_lm_seed2": {"n_dc_nt": 114, "n_dc_th": 70, "kl": 0.0011, "ifev": 446, "gsm": 475},
            "std_uniform_lora": {"n_dc_nt": 156, "n_dc_th": 76, "kl": 0.0025, "ifev": 440, "gsm": 471},
            "tight_targeted_dual": {"n_dc_nt": 152, "n_dc_th": 75, "kl": 0.0007, "ifev": 448, "gsm": 482},
            "tight_random_lm": {"n_dc_nt": 100, "n_dc_th": 70, "kl": 0.0008, "ifev": 446, "gsm": 480},
            "tight_random_lm_seed2": {"n_dc_nt": 102, "n_dc_th": 71, "kl": 0.0008, "ifev": 445, "gsm": 479},
            "tight_uniform_lora": {"n_dc_nt": 120, "n_dc_th": 71, "kl": 0.0015, "ifev": 444, "gsm": 476},
        }
        # Test 1: Single-seed -> MUST be CANDIDATE (overall_go_14b=False!), and cluster bootstrap active!
        root_go = Path(td) / "go"
        _build_synthetic_dir(root_go, base_go, n_primary=200, n_primary_elig=160, train_seed=42, sha_salt="s42")
        rep_cand = analyze_step2(root_go, min_eligible_bases=160, n_boot=1000, seed=2027)
        assert rep_cand["verdict"] == "CANDIDATE" and rep_cand["overall_go_14b"] is False
        assert rep_cand["candidate_single_seed_tier"] == "tight_1.52M"
        assert rep_cand["tiers"]["tight_1.52M"]["comparisons"]["c1_step2b_confirm_ce__no_think__shared_eligible"]["dual_vs_random_pool"]["double_correct_VF"]["clustered"] is True

        # Test 2 (Astra CE-1): 160 eligible with default min_eligible_bases=800 -> MUST NOT be CANDIDATE/GO!
        rep_low_n = analyze_step2(root_go, min_eligible_bases=800, n_boot=500, seed=2027)
        assert rep_low_n["verdict"] == "HOLD" and rep_low_n["overall_go_14b"] is False

        # Test 3 (Astra CE-2): Passing SAME directory as seed2_dir -> MUST BE REJECTED!
        rep_same_dir = analyze_step2(root_go, min_eligible_bases=160, seed2_dir=root_go, n_boot=500, seed=2027)
        assert rep_same_dir["verdict"] == "HOLD" and rep_same_dir["overall_go_14b"] is False
        assert rep_same_dir["seed2_replication"]["status"] == "rejected_same_directory"

        # Test 4 (Astra CE-2b): Different directory but SAME seed=42 -> MUST BE REJECTED!
        root_fake_s2 = Path(td) / "fake_s2"
        _build_synthetic_dir(root_fake_s2, base_go, n_primary=200, n_primary_elig=160, train_seed=42, sha_salt="s42b")
        rep_same_seed = analyze_step2(root_go, min_eligible_bases=160, seed2_dir=root_fake_s2, n_boot=500, seed=2027)
        assert rep_same_seed["overall_go_14b"] is False
        assert rep_same_seed["seed2_replication"]["status"] == "rejected_missing_or_identical_seed"

        # Test 5: Real independent seed=1024 + distinct SHA -> GO!
        root_s2 = Path(td) / "real_s2"
        _build_synthetic_dir(root_s2, base_go, n_primary=200, n_primary_elig=160, train_seed=1024, sha_salt="s1024")
        rep_real_go = analyze_step2(root_go, min_eligible_bases=160, seed2_dir=root_s2, n_boot=1000, seed=2027)
        assert rep_real_go["verdict"] == "GO" and rep_real_go["overall_go_14b"] is True
        assert rep_real_go["recommended_14b_tier"] == "tight_1.52M"
        assert "GO (tight_1.52M, seed=1024 verified)" in render_markdown(rep_real_go)

        # Test 6 (Astra CE-3): Deleting c1_dev/no_think -> MUST FAIL health!
        ce_no_dev = {k: dict(v) for k, v in base_go.items()}
        ce_no_dev["std_targeted_dual"]["omit_dev_nt"] = True
        ce_no_dev["tight_targeted_dual"]["omit_dev_nt"] = True
        root_nd = Path(td) / "no_dev"
        _build_synthetic_dir(root_nd, ce_no_dev, n_primary=200, n_primary_elig=160)
        rep_nd = analyze_step2(root_nd, min_eligible_bases=160, n_boot=500, seed=2027)
        assert rep_nd["arms"]["tight_targeted_dual"]["health"]["checks"]["required_nothink_splits_complete"] is False
        assert rep_nd["candidate_single_seed_tier"] is None

        # Test 7 (Astra CE-4): Missing Random-2 u_vf MUST NOT reuse Random-1 in m_pool!
        ce_drop_u = {k: dict(v) for k, v in base_go.items()}
        ce_drop_u["tight_random_lm_seed2"]["drop_u_nt"] = True
        root_du = Path(td) / "drop_u"
        _build_synthetic_dir(root_du, ce_drop_u, n_primary=200, n_primary_elig=160)
        rep_du = analyze_step2(root_du, min_eligible_bases=160, n_boot=500, seed=2027)
        pool_cmp = rep_du["tiers"]["tight_1.52M"]["comparisons"]["c1_step2b_confirm_ce__no_think__shared_eligible"]["dual_vs_random_pool"]
        assert "u_V_minus_F" not in pool_cmp, "Regression: missing Random-2 u_vf silently reused Random-1!"
        assert rep_du["tiers"]["tight_1.52M"]["go_14b_gate"]["verdict_single_seed_pass"] is False

        # Test 8 (Astra CE-5): Missing per-arm parity JSON -> MUST FAIL fidelity (even if global file exists)!
        ce_no_par = {k: dict(v) for k, v in base_go.items()}
        ce_no_par["tight_targeted_dual"]["omit_parity"] = True
        root_np = Path(td) / "no_par"
        _build_synthetic_dir(root_np, ce_no_par, n_primary=200, n_primary_elig=160)
        (root_np / "vllm_lora_parity.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
        rep_np = analyze_step2(root_np, min_eligible_bases=160, n_boot=500, seed=2027)
        assert rep_np["arms"]["tight_targeted_dual"]["health"]["checks"]["fidelity_passed"] is False

        # Test 9 (True non-inferiority CI lower bound < -0.02 -> FAIL!)
        ce_ni = {k: dict(v) for k, v in base_go.items()}
        dual_th = [False] + [True] * 59 + [True] * 10 + [False] * 10
        rnd_th = [True] * 60 + [False] * 10 + [True] * 10
        for prefix in ("std", "tight"):
            ce_ni[f"{prefix}_targeted_dual"]["th_dc_mask"] = list(dual_th)
            ce_ni[f"{prefix}_random_lm"]["th_dc_mask"] = list(rnd_th)
            ce_ni[f"{prefix}_random_lm_seed2"]["th_dc_mask"] = list(rnd_th)
        root_ni = Path(td) / "ce_ni"
        _build_synthetic_dir(root_ni, ce_ni, n_primary=200, n_primary_elig=160)
        rep_ni = analyze_step2(root_ni, min_eligible_bases=160, n_boot=1000, seed=2027)
        assert rep_ni["tiers"]["tight_1.52M"]["go_14b_gate"]["cond2_think_non_inferior"] is False

        # Test 10 (Explicit STOP vs HOLD): both tiers exact 0.0 diff with tight CI [0, 0] < 0.05 -> STOP!
        ce_stop = {k: dict(v) for k, v in base_go.items()}
        for prefix in ("std", "tight"):
            ce_stop[f"{prefix}_targeted_dual"]["n_dc_nt"] = 120
            ce_stop[f"{prefix}_random_lm"]["n_dc_nt"] = 120
            ce_stop[f"{prefix}_random_lm_seed2"]["n_dc_nt"] = 120
        root_stop = Path(td) / "ce_stop"
        _build_synthetic_dir(root_stop, ce_stop, n_primary=200, n_primary_elig=160)
        rep_stop = analyze_step2(root_stop, min_eligible_bases=160, n_boot=500, seed=2027)
        assert rep_stop["verdict"] == "STOP" and rep_stop["overall_go_14b"] is False

        # Test 11 (Dual ties Random-1 -> NOT CANDIDATE, HOLD!)
        ce_tie_r1 = {k: dict(v) for k, v in base_go.items()}
        ce_tie_r1["std_random_lm"]["n_dc_nt"] = 156
        ce_tie_r1["tight_random_lm"]["n_dc_nt"] = 152
        root_r1 = Path(td) / "ce_tie_r1"
        _build_synthetic_dir(root_r1, ce_tie_r1, n_primary=200, n_primary_elig=160)
        rep_r1 = analyze_step2(root_r1, min_eligible_bases=160, n_boot=500, seed=2027)
        assert rep_r1["verdict"] == "HOLD" and rep_r1["candidate_single_seed_tier"] is None

        # Test 12 (Misaligned base_ids -> AssertionError!)
        m_a = {"base_ids": ["A", "B"], "clusters": None, "dc": np.array([1.0, 0.0]), "u_vf": None, "u_sh_ao": None, "pair_types": ["cf", "cf"], "depths": [2, 3]}
        m_b = {"base_ids": ["B", "A"], "clusters": None, "dc": np.array([1.0, 0.0]), "u_vf": None, "u_sh_ao": None, "pair_types": ["cf", "cf"], "depths": [2, 3]}
        try:
            compare_two_arms(m_a, m_b, n_boot=100)
        except AssertionError:
            pass
        else:
            raise AssertionError("Misaligned base_ids were silently paired!")

        print("[OK] c1_step2_stats v4 CPU self-test passed (McNemar 600:600 + ALL 12 Astra/Opus counterexamples verified)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=C1 / "results" / "step2b_8b_showdown")
    ap.add_argument("--primary-dataset", default="c1_step2b_confirm_ce", choices=list(ALL_SCANNED_DATASETS))
    ap.add_argument("--think-dataset", default=None, choices=list(ALL_SCANNED_DATASETS))
    ap.add_argument("--min-eligible-bases", type=int, default=DEFAULT_MIN_ELIGIBLE_BASES)
    ap.add_argument("--seed2-dir", type=Path, default=None, help="independent seed=1024 replication results dir")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    rep = analyze_step2(
        args.results_dir,
        primary_dataset=args.primary_dataset,
        think_dataset=args.think_dataset,
        min_eligible_bases=args.min_eligible_bases,
        seed2_dir=args.seed2_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    out_json = args.results_dir / "step2b_stats.json"
    out_md = args.results_dir / "STEP2B_READOUT.md"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(render_markdown(rep), encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out_json} and {out_md} (verdict={rep['verdict']}, overall_go_14b={rep['overall_go_14b']}, rec={rep['recommended_14b_tier']})")


if __name__ == "__main__":
    main()
