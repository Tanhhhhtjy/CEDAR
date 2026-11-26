#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 6 Strict Aggregator & Audited Statistical Primitives (also reused by Stage 7).

Hardened fail-closed guarantees (incorporating ALL Stage 5 + Stage 6 Astra audit fixes):
  1. STRICT FAIL-CLOSED RECORD VALIDATOR (`validate_records_strict`):
       - Rejects empty files, duplicate `(base_id, arm)` keys, missing or extra keys vs independent `expected_bases x expected_arms`.
       - NEVER extracts `expected_bases` from the records themselves!
       - Validates `initial_answer`, `target_answer` in `{A, B, C, D}` and `initial_answer != target_answer`.
       - Validates `parse_ok`, `is_target`, `is_initial` are explicit `bool` AND strictly consistent with `pred`
         (`pred in {A, B, C, D}`, `is_target == (pred == target_answer)`, `is_initial == (pred == initial_answer)` when `parse_ok=True`;
         both `False` when `parse_ok=False` -> rejects `pred="Z"` or simultaneous `is_target=True & is_initial=True`!).
       - Validates `u_theta` is a finite numeric value in `no_think` (`math.isfinite`, rejecting `NaN`/`inf`/`bool`).
  2. STRICT M0 DUAL-ELIGIBILITY (`compute_m0_dual_eligible_strict`):
       - Runs `validate_records_strict` on M0 first (rejects partial `turn1_F`-only records immediately!).
       - Computes dual eligibility strictly per `(scale, regime, dataset)` as `turn1_V.is_initial and turn1_F.is_initial`.
  3. HONEST BOOTSTRAP TAIL REPORTING (`format_boot_p`) + EXACT SIGN-FLIP TEST (`exact_two_sided_signflip_p`):
       - Reports exact `0/5000` (smallest nonzero two-sided resolution `2/n_boot = 0.0004` at `n_boot=5000`) when zero tail hits occur,
         NEVER `p=0.0000` and NEVER misleading `< 0.0002`!
       - Also logs exact two-sided binomial sign-flip sensitivity p-value on nonzero unit differences.
  4. EXACT UNROUNDED OOD 50/50 EQUAL-SLICE SCORER (`score_ood_equal_slice_50_50`):
       - Rejects any unknown slice or extra/missing base IDs; hard-asserts exact `60 + 60` on full `c1_test_ood`.
       - Computes `0.5 * mean(ood_converse) + 0.5 * mean(ood_depth4)` from UNROUNDED raw slice means FIRST, rounding only at display
         (so `0/60` and `10/60` yields exact `8.33%`, NEVER `8.34%`!).
       - Handles shared-M0-eligible subsets with deterministic per-slice counts and explicit empty-slice warning if any slice has `n=0`.
  5. FULL REPRODUCIBILITY OF STAGE 6 READOUT (Sections 1-5):
       - Computes `pooled_2seed` means from unrounded raw floats (`14B v4 v3 pooled = 59.22%` exact).
       - Explicitly labels `Turn-1 Dual (1080 items)` vs `4-Cell / Joint (900 units)`.
       - Auto-generates Section 5 (`backend_parity_8b`) directly from raw `vllm_lora_parity.json` & `vllm_merged_parity.json` files!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_stage6_stats.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
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
import c35_stage5_stats as S5  # noqa: E402
import c35_stats as S35  # noqa: E402

VALID_OPTIONS = {"A", "B", "C", "D"}
REQUIRED_BASE_FIELDS = ("base_id", "arm", "initial_answer", "target_answer", "parse_ok", "is_target", "is_initial")
REQUIRED_NOTHINK_FIELDS = REQUIRED_BASE_FIELDS + ("pred", "u_theta")
OOD_VALID_SLICES = ("ood_converse", "ood_depth4")


def _load_base_meta(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {str(r["base_id"]): r for r in rows}


def format_boot_p(p_raw: float, n_boot: int) -> str:
    """Honest finite two-sided bootstrap tail reporting (`0/5000` when 0 tail hits; min nonzero two-sided = 2/n_boot)."""
    n_b = max(1, int(n_boot))
    if p_raw <= 0.0:
        return f"0/{n_b}"
    return f"{p_raw:.4f}"


def exact_two_sided_signflip_p(diffs: np.ndarray) -> float:
    """Exact two-sided paired sign test (binomial p=0.5 on nonzero paired differences)."""
    pos = int(np.sum(diffs > 1e-12))
    neg = int(np.sum(diffs < -1e-12))
    m = pos + neg
    if m == 0:
        return 1.0
    k = min(pos, neg)
    # Exact log-space summation avoids overflow for any m
    log_denom = m * math.log(2.0)
    log_terms = [math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1) - log_denom for i in range(k + 1)]
    max_lt = max(log_terms)
    tail = math.exp(max_lt) * sum(math.exp(lt - max_lt) for lt in log_terms)
    return min(1.0, 2.0 * tail)


def validate_records_strict(
    recs: list[dict[str, Any]],
    *,
    expected_bases: set[str],
    expected_arms: set[str],
    regime: str,
    label: str,
) -> dict[str, Any]:
    """Fail-closed record integrity validator checking schema, options, flags, finite u_theta, and exact key coverage."""
    if not recs:
        raise AssertionError(f"[EMPTY RECORDS] {label} ({regime}) contains 0 records!")
    if not expected_bases:
        raise AssertionError(f"[EMPTY EXPECTED BASES] {label}: expected_bases must be non-empty!")

    req_fields = REQUIRED_NOTHINK_FIELDS if regime == "no_think" else REQUIRED_BASE_FIELDS
    seen: set[tuple[str, str]] = set()
    dup_count = 0
    n_parse_fail = 0

    for i, r in enumerate(recs):
        for f in req_fields:
            if f not in r or r[f] is None:
                raise AssertionError(f"[MISSING FIELD] {label} row {i}: required field {f!r} missing or None!")
        init_a = r["initial_answer"]
        targ_a = r["target_answer"]
        if init_a not in VALID_OPTIONS or targ_a not in VALID_OPTIONS or init_a == targ_a:
            raise AssertionError(
                f"[BAD ANSWER LABELS] {label} row {i}: initial_answer={init_a!r}, target_answer={targ_a!r}"
            )
        if not isinstance(r["parse_ok"], bool):
            raise AssertionError(f"[BAD parse_ok TYPE] {label} row {i}: expected bool, got {type(r['parse_ok'])} ({r['parse_ok']!r})")
        if not isinstance(r["is_target"], bool) or not isinstance(r["is_initial"], bool):
            raise AssertionError(f"[BAD BOOL FIELD] {label} row {i}: is_target/is_initial must be bool")
        if r["is_target"] and r["is_initial"]:
            raise AssertionError(f"[CONTRADICTORY FLAGS] {label} row {i}: both is_target and is_initial are True!")

        if r["parse_ok"]:
            pred = r.get("pred")
            if pred not in VALID_OPTIONS:
                raise AssertionError(f"[INVALID pred] {label} row {i}: parse_ok=True but pred={pred!r} not in {sorted(VALID_OPTIONS)}")
            if r["is_target"] != (pred == targ_a) or r["is_initial"] != (pred == init_a):
                raise AssertionError(
                    f"[FLAG MISMATCH] {label} row {i}: pred={pred!r}, init={init_a!r}, targ={targ_a!r}, "
                    f"is_initial={r['is_initial']}, is_target={r['is_target']}"
                )
        else:
            n_parse_fail += 1
            if r["is_target"] or r["is_initial"]:
                raise AssertionError(
                    f"[PARSE_FAIL WITH TRUE FLAG] {label} row {i}: parse_ok=False but is_target={r['is_target']}, is_initial={r['is_initial']}"
                )

        if regime == "no_think":
            u_val = r["u_theta"]
            if isinstance(u_val, bool) or not isinstance(u_val, (int, float)) or not math.isfinite(float(u_val)):
                raise AssertionError(f"[NON-FINITE u_theta] {label} row {i}: u_theta={u_val!r}")

        key = (str(r["base_id"]), str(r["arm"]))
        if key in seen:
            dup_count += 1
        seen.add(key)

    if dup_count > 0:
        raise AssertionError(f"[DUPLICATE KEYS] {label}: found {dup_count} duplicate (base_id, arm) rows!")

    expected_keys = {(b, a) for b in expected_bases for a in expected_arms}
    missing = expected_keys - seen
    extra = seen - expected_keys
    if missing or extra:
        raise AssertionError(
            f"[KEY SET MISMATCH] {label}: expected {len(expected_keys)} (base_id, arm) pairs, "
            f"got {len(seen)} (missing={len(missing)}, extra={len(extra)})"
        )
    return {
        "n_records": len(recs),
        "n_bases": len(expected_bases),
        "n_arms": len(expected_arms),
        "parse_fail_count": n_parse_fail,
        "parse_fail_rate": round(n_parse_fail / len(recs), 6),
        "validated": True,
    }


def compute_m0_dual_eligible_strict(
    m0_recs: list[dict[str, Any]],
    *,
    expected_bases: set[str],
    expected_arms: set[str],
    regime: str,
    label: str,
) -> list[str]:
    """Fail-closed M0 dual Turn-1 eligibility: requires BOTH turn1_V and turn1_F for every expected base."""
    assert {"turn1_V", "turn1_F"}.issubset(expected_arms), f"{label}: expected_arms must include turn1_V & turn1_F"
    validate_records_strict(
        m0_recs, expected_bases=expected_bases, expected_arms=expected_arms, regime=regime, label=label
    )
    by_ba = {(str(r["base_id"]), str(r["arm"])): r for r in m0_recs}
    elig = []
    for b in sorted(expected_bases):
        rv = by_ba[(b, "turn1_V")]
        rf = by_ba[(b, "turn1_F")]
        if rv["parse_ok"] and rf["parse_ok"] and rv["is_initial"] and rf["is_initial"]:
            elig.append(b)
    return elig


def paired_unit_bootstrap_subset(
    bm_a: dict[str, dict[str, Any]],
    bm_b: dict[str, dict[str, Any]],
    subset_units: set[str],
    metric: str,
    n_boot: int = 5000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Paired bootstrap over an explicit unit subset + exact two-sided paired sign-flip sensitivity p-value."""
    fa = {u: v for u, v in C3S.family_weighted_scores(bm_a, metric).items() if u in subset_units}
    fb = {u: v for u, v in C3S.family_weighted_scores(bm_b, metric).items() if u in subset_units}
    assert set(fa) == subset_units and set(fb) == subset_units, (len(fa), len(fb), len(subset_units))
    units = sorted(subset_units)
    va = np.array([fa[u] for u in units], dtype=np.float64)
    vb = np.array([fb[u] for u in units], dtype=np.float64)
    diffs = va - vb
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    p_pos = float(np.mean(boot <= 0.0))
    p_neg = float(np.mean(boot >= 0.0))
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    p_sf = exact_two_sided_signflip_p(diffs)
    return {
        "n_units": n,
        "mean_a_pct": round(float(va.mean()) * 100.0, 2),
        "mean_b_pct": round(float(vb.mean()) * 100.0, 2),
        "diff_pp": round(float(diffs.mean()) * 100.0, 2),
        "ci95_pp": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
        "p_two_sided_raw": round(p_two, 6),
        "p_str": format_boot_p(p_two, n_boot),
        "p_signflip": round(p_sf, 6),
    }


def pooled_paired_bootstrap_subset(
    bm_a_by_seed: dict[str, dict[str, dict[str, Any]]],
    bm_b_by_seed: dict[str, dict[str, dict[str, Any]]],
    subset_units: set[str],
    metric: str,
    n_boot: int = 5000,
    seed: int = 2027,
) -> dict[str, Any]:
    seeds = sorted(set(bm_a_by_seed) & set(bm_b_by_seed))
    assert seeds, "No common seeds!"
    units = sorted(subset_units)
    mat_a, mat_b = [], []
    for sd in seeds:
        fa = C3S.family_weighted_scores(bm_a_by_seed[sd], metric)
        fb = C3S.family_weighted_scores(bm_b_by_seed[sd], metric)
        assert subset_units.issubset(fa) and subset_units.issubset(fb)
        mat_a.append([fa[u] for u in units])
        mat_b.append([fb[u] for u in units])
    va = np.mean(np.array(mat_a, dtype=np.float64), axis=0)
    vb = np.mean(np.array(mat_b, dtype=np.float64), axis=0)
    diffs = va - vb
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    p_pos = float(np.mean(boot <= 0.0))
    p_neg = float(np.mean(boot >= 0.0))
    p_two = min(1.0, 2.0 * min(p_pos, p_neg))
    p_sf = exact_two_sided_signflip_p(diffs)
    return {
        "n_units": n,
        "seeds": seeds,
        "mean_a_pct": round(float(va.mean()) * 100.0, 2),
        "mean_b_pct": round(float(vb.mean()) * 100.0, 2),
        "diff_pp": round(float(diffs.mean()) * 100.0, 2),
        "ci95_pp": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
        "p_two_sided_raw": round(p_two, 6),
        "p_str": format_boot_p(p_two, n_boot),
        "p_signflip": round(p_sf, 6),
    }


def summarize_entity_balanced_cells(
    bm: dict[str, dict[str, Any]],
    base_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize all 6 cells of `c3_entity_balanced_v1` + exact entity-count shortcut asymmetry (unrounded math)."""
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bid, row in bm.items():
        cell = base_meta[bid]["entity_balance_cell"]
        by_cell[cell].append(row)

    out_cells: dict[str, Any] = {}
    raw_fc_by_cell: dict[str, float] = {}
    for cell in ("1v2_orig", "2v1_anti", "1v1_same", "2v2_same", "1v1_ctx", "2v2_ctx"):
        rs = by_cell.get(cell, [])
        assert rs, f"Missing entity_balanced cell: {cell}"
        fc_raw = float(np.mean([r["four_cell"] for r in rs])) * 100.0
        jt_raw = float(np.mean([r["joint_t1_4cell"] for r in rs])) * 100.0
        raw_fc_by_cell[cell] = fc_raw
        out_cells[cell] = {"n": len(rs), "four_cell_pct": round(fc_raw, 2), "joint_pct": round(jt_raw, 2)}

    all_fc = float(np.mean([r["four_cell"] for r in bm.values()])) * 100.0
    all_jt = float(np.mean([r["joint_t1_4cell"] for r in bm.values()])) * 100.0
    shortcut_asym = round(raw_fc_by_cell["1v2_orig"] - raw_fc_by_cell["2v1_anti"], 2)
    same_ent_avg = round(
        0.25 * (
            raw_fc_by_cell["1v1_same"]
            + raw_fc_by_cell["2v2_same"]
            + raw_fc_by_cell["1v1_ctx"]
            + raw_fc_by_cell["2v2_ctx"]
        ),
        2,
    )
    return {
        "n_bases": len(bm),
        "overall_four_cell_pct": round(all_fc, 2),
        "overall_joint_pct": round(all_jt, 2),
        "same_entity_count_avg_4cell_pct": same_ent_avg,
        "shortcut_asymmetry_1v2_minus_2v1_pp": shortcut_asym,
        "by_cell": out_cells,
    }


def score_ood_equal_slice_50_50(
    bm_a: dict[str, dict[str, Any]],
    bm_b: dict[str, dict[str, Any]] | None,
    slice_map: dict[str, str],
    metric: str = "four_cell",
    n_boot: int = 5000,
    seed: int = 2027,
    *,
    expected_bases: set[str] | None = None,
    require_exact_60_60: bool = True,
) -> dict[str, Any]:
    """Exact pre-unseal OOD weighting: 50% `ood_converse` + 50% `ood_depth4` computed from UNROUNDED slice means.

    Hard-asserts:
      - ZERO unknown slice names (every value in `slice_map` must be in `{"ood_converse", "ood_depth4"}`).
      - If `expected_bases` is provided, `set(slice_map.keys()) == expected_bases` (no extra or missing IDs).
      - If `require_exact_60_60=True` (full `c1_test_ood`), exact `60` `ood_converse` + `60` `ood_depth4` items.
      - If `require_exact_60_60=False` (e.g. `shared_m0_eligible`), allows variable slice sizes and returns explicit
        `empty_slice_warning` + `equal_50_50_score_pct=None` if either slice has 0 eligible bases.
    """
    unknown_slices = set(slice_map.values()) - set(OOD_VALID_SLICES)
    if unknown_slices:
        raise AssertionError(f"[UNKNOWN OOD SLICES] Invalid slice labels in slice_map: {sorted(unknown_slices)}")
    if expected_bases is not None and set(slice_map.keys()) != expected_bases:
        missing = expected_bases - set(slice_map.keys())
        extra = set(slice_map.keys()) - expected_bases
        raise AssertionError(f"[OOD BASE MISMATCH] missing={len(missing)}, extra={len(extra)}")

    ids_by_slice = {sl: sorted(bid for bid, s in slice_map.items() if s == sl) for sl in OOD_VALID_SLICES}
    if require_exact_60_60:
        for sl in OOD_VALID_SLICES:
            if len(ids_by_slice[sl]) != 60:
                raise AssertionError(f"[OOD SLICE COUNT ERROR] Expected exactly 60 items in {sl}, got {len(ids_by_slice[sl])}")

    for sl in OOD_VALID_SLICES:
        for bid in ids_by_slice[sl]:
            assert bid in bm_a, f"Missing {bid} in model A"
            if bm_b is not None:
                assert bid in bm_b, f"Missing {bid} in model B"

    per_slice: dict[str, Any] = {}
    raw_mean_a: dict[str, float | None] = {}
    raw_mean_b: dict[str, float | None] = {}
    rng = np.random.default_rng(seed)
    boot_diff_slices: list[np.ndarray] = []
    empty_slices: list[str] = []

    for sl in OOD_VALID_SLICES:
        bids = ids_by_slice[sl]
        if not bids:
            empty_slices.append(sl)
            raw_mean_a[sl] = None
            raw_mean_b[sl] = None
            per_slice[sl] = {"n_items": 0, "raw_score_pct": None}
            continue
        va = np.array([float(bm_a[b][metric]) for b in bids], dtype=np.float64)
        ma = float(va.mean())
        raw_mean_a[sl] = ma
        idx = rng.integers(0, len(bids), size=(n_boot, len(bids)))
        sl_doc: dict[str, Any] = {
            "n_items": len(bids),
            "n_correct": int(round(float(va.sum()))),
            "raw_score_pct": round(ma * 100.0, 2),
        }
        if bm_b is not None:
            vb = np.array([float(bm_b[b][metric]) for b in bids], dtype=np.float64)
            mb = float(vb.mean())
            raw_mean_b[sl] = mb
            d = va - vb
            bd = d[idx].mean(axis=1)
            boot_diff_slices.append(bd)
            p_two = min(1.0, 2.0 * min(float(np.mean(bd <= 0.0)), float(np.mean(bd >= 0.0))))
            sl_doc.update({
                "raw_score_b_pct": round(mb * 100.0, 2),
                "diff_pp": round(float(d.mean()) * 100.0, 2),
                "ci95_pp": [round(float(np.quantile(bd, 0.025)) * 100.0, 2), round(float(np.quantile(bd, 0.975)) * 100.0, 2)],
                "p_str": format_boot_p(p_two, n_boot),
                "p_signflip": round(exact_two_sided_signflip_p(d), 6),
            })
        per_slice[sl] = sl_doc

    if empty_slices:
        return {
            "weighting": "50pct_ood_converse_50pct_ood_depth4",
            "equal_50_50_score_pct": None,
            "empty_slice_warning": f"Undefined 50/50 aggregate due to empty eligible slice(s): {empty_slices}",
            "per_slice": per_slice,
        }

    # UNROUNDED combination FIRST, round ONLY on final output!
    equal_50_50_a_raw = 0.5 * (float(raw_mean_a["ood_converse"]) + float(raw_mean_a["ood_depth4"])) * 100.0
    out: dict[str, Any] = {
        "weighting": "50pct_ood_converse_50pct_ood_depth4",
        "equal_50_50_score_pct": round(equal_50_50_a_raw, 2),
        "per_slice": per_slice,
    }
    if bm_b is not None:
        equal_50_50_b_raw = 0.5 * (float(raw_mean_b["ood_converse"]) + float(raw_mean_b["ood_depth4"])) * 100.0
        comb_boot = 0.5 * (boot_diff_slices[0] + boot_diff_slices[1])
        p_two = min(1.0, 2.0 * min(float(np.mean(comb_boot <= 0.0)), float(np.mean(comb_boot >= 0.0))))
        out.update({
            "equal_50_50_score_b_pct": round(equal_50_50_b_raw, 2),
            "equal_50_50_diff_pp": round(equal_50_50_a_raw - equal_50_50_b_raw, 2),
            "equal_50_50_ci95_pp": [
                round(float(np.quantile(comb_boot, 0.025)) * 100.0, 2),
                round(float(np.quantile(comb_boot, 0.975)) * 100.0, 2),
            ],
            "equal_50_50_p_str": format_boot_p(p_two, n_boot),
        })
    return out


def compute_v3_non_shortcut_units(v3_bases_file: Path) -> tuple[set[str], set[str], dict[str, set[str]]]:
    """Return `(all_900_units, non_shortcut_802_units, by_pair_type_units)` for `c3_blind_confirm_v3`."""
    all_u: set[str] = set()
    shortcut_u: set[str] = set()
    by_pt: dict[str, set[str]] = defaultdict(set)
    for line in v3_bases_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        u = str(r.get("family_id", r.get("cell_id", r["base_id"])))
        all_u.add(u)
        by_pt[str(r["pair_type"])].add(u)
        if r.get("fallacy_class") == "missing_conjunct" and r.get("pair_type") == "argument_flip":
            shortcut_u.add(u)
    non_shortcut_u = all_u - shortcut_u
    return all_u, non_shortcut_u, dict(by_pt)


def _fmt_exact_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def _exact_exfever_headline_str(ex: dict[str, Any] | None) -> str:
    """Compute exact EX-FEVER R_UY / R_RU_ev / Selectivity from unrounded k/n counts when available."""
    if not ex:
        return "—"
    metrics = ex.get("metrics") or ex.get("summary") or {}
    conds = metrics.get("conditions") or {}
    pr = (conds.get("pressure") or {}).get("flip_correct_to_wrong") or {}
    ev = (conds.get("evidence") or {}).get("flip_wrong_to_correct") or {}
    if "k" in pr and "n" in pr and pr["n"] and "k" in ev and "n" in ev and ev["n"]:
        ruy_raw = 100.0 * float(pr["k"]) / float(pr["n"])
        rru_raw = 100.0 * float(ev["k"]) / float(ev["n"])
        sel_raw = rru_raw - ruy_raw
        return f"{ruy_raw:.2f}% / {rru_raw:.2f}% / {sel_raw:.2f}"
    hl = metrics.get("headline", {})
    if not hl:
        return "—"
    return f"{hl.get('r_uy', '-')}% / {hl.get('r_ru_evidence', '-')}% / {hl.get('selectivity_ru_minus_uy', '-')}"


def _collect_8b_backend_parity(entries: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Auto-generate Section 5 backend parity audit table directly from raw parity & think summary JSONs."""
    parity_doc: dict[str, Any] = {}
    md_lines: list[str] = [
        "\n## 5. 8B vLLM Backend Parity Audit (Auto-Generated from Raw Parity JSONs)\n",
        "All `no_think` (`v2`, `v3`, `c3_entity_balanced_v1`, `WikiText-2 KL`) metrics run natively in HuggingFace BF16 (`backend=hf`, exact). Generative evaluations (`think`, `capabilities`, `EX-FEVER`) use the vLLM backend logged below:\n",
        "| Method | Generative Backend Used | Native LoRA Parity (Opt / `|Δu|` mean) | BF16 Merged Parity (Opt / `|Δu|` mean) | Audit Status |",
        "|---|---|---|---|---|",
    ]
    any_row = False
    for entry in entries:
        d = Path(entry["dir"])
        name = entry["name"]
        lp = d / "vllm_lora_parity.json"
        mp = d / "vllm_merged_parity.json"
        if not lp.is_file() and not mp.is_file():
            continue
        any_row = True
        row_info: dict[str, Any] = {}
        # Detect actual generative backend from think/eval_summary_think.json or capabilities/summary.json
        gen_backend = "vllm_native_lora"
        for probe_p in (d / "think" / "eval_summary_think.json", d / "capabilities" / "summary.json"):
            if probe_p.is_file():
                pd = json.loads(probe_p.read_text(encoding="utf-8"))
                if pd.get("peft_lora_dir") is None and "merged" in str(pd.get("model", "")):
                    gen_backend = "vllm_bf16_merged"
                break
        row_info["generative_backend_used"] = gen_backend

        lora_cell = "—"
        if lp.is_file():
            ld = json.loads(lp.read_text(encoding="utf-8"))
            lv = ld.get("lora_hf_live_vs_vllm_native") or {}
            l_pass = bool(ld.get("passed"))
            l_opt = lv.get("option_match_rate")
            l_du = lv.get("du_abs_mean")
            l_sgn = lv.get("du_signed_mean")
            row_info.update({
                "lora_passed": l_pass,
                "lora_option_match_rate": l_opt,
                "lora_du_abs_mean": l_du,
                "lora_du_signed_mean": l_sgn,
            })
            lora_cell = f"{'PASS' if l_pass else 'FAIL'} (`{l_opt*100:.2f}%` / `{l_du:.5f}`)" if (l_opt is not None and l_du is not None) else str(l_pass)

        merged_cell = "—"
        if mp.is_file():
            md_j = json.loads(mp.read_text(encoding="utf-8"))
            mv = md_j.get("hf_live_vs_vllm_merged") or {}
            m_pass = bool(md_j.get("passed"))
            m_opt = mv.get("option_match_rate")
            m_du = mv.get("du_abs_mean")
            row_info.update({
                "merged_passed": m_pass,
                "merged_option_match_rate": m_opt,
                "merged_du_abs_mean": m_du,
            })
            merged_cell = f"{'PASS' if m_pass else 'FAIL'} (`{m_opt*100:.2f}%` / `{m_du:.5f}`)" if (m_opt is not None and m_du is not None) else str(m_pass)

        eff_pass = bool(row_info.get("merged_passed")) if gen_backend == "vllm_bf16_merged" else bool(row_info.get("lora_passed"))
        row_info["effective_passed"] = eff_pass
        if eff_pass:
            status_cell = "**PASS (via BF16 merged)**" if gen_backend == "vllm_bf16_merged" else "**PASS**"
        else:
            status_cell = "**CAVEAT** (`no_think` exact HF; generative parity FAIL)"
        parity_doc[name] = row_info
        md_lines.append(f"| `{name}` | `{gen_backend}` | {lora_cell} | {merged_cell} | {status_cell} |")

    return (parity_doc, md_lines) if any_row else ({}, [])


def run_selftest_tiny() -> None:
    import tempfile

    # 1. Fail-closed validator catches ALL 9 adversarial faults (including pred="Z", both flags True, u_theta=NaN, parse_fail+True)!
    exp_b = {"B1", "B2"}
    exp_a = {"V", "F"}
    good = [
        {"base_id": b, "arm": a, "initial_answer": "A", "target_answer": "B", "pred": "B", "u_theta": 2.0, "parse_ok": True, "is_target": True, "is_initial": False}
        for b in exp_b for a in exp_a
    ]
    val = validate_records_strict(good, expected_bases=exp_b, expected_arms=exp_a, regime="no_think", label="good")
    assert val["validated"] is True and val["parse_fail_rate"] == 0.0

    for bad_name, bad_recs in [
        ("empty_think", []),
        ("missing_parse_ok", [{k: v for k, v in good[0].items() if k != "parse_ok"}] + good[1:]),
        ("non_bool_parse_ok", [{**good[0], "parse_ok": "true"}] + good[1:]),
        ("duplicate", good + [good[0]]),
        ("missing_arm", good[:-1]),
        ("invalid_pred_Z", [{**good[0], "pred": "Z", "is_target": False, "is_initial": False}] + good[1:]),
        ("both_flags_true", [{**good[0], "is_target": True, "is_initial": True}] + good[1:]),
        ("nan_u_theta", [{**good[0], "u_theta": float("nan")}] + good[1:]),
        ("parse_fail_but_true_flag", [{**good[0], "parse_ok": False, "pred": None, "is_target": True, "is_initial": False}] + good[1:]),
    ]:
        raised = False
        try:
            validate_records_strict(
                bad_recs, expected_bases=exp_b, expected_arms=exp_a,
                regime="think" if "think" in bad_name else "no_think", label=bad_name,
            )
        except AssertionError:
            raised = True
        assert raised, f"Validator failed to reject {bad_name}!"

    # 2. Strict M0 dual-eligibility rejects partial turn1_F-only records!
    t1_only_f = [
        {"base_id": "B1", "arm": "turn1_F", "initial_answer": "A", "target_answer": "B", "pred": "A", "u_theta": -3.0, "parse_ok": True, "is_target": False, "is_initial": True}
    ]
    try:
        compute_m0_dual_eligible_strict(t1_only_f, expected_bases={"B1"}, expected_arms={"turn1_V", "turn1_F"}, regime="no_think", label="bad_m0")
        raise RuntimeError("Should have rejected turn1_F-only M0 record!")
    except AssertionError:
        pass

    # 3. Verify honest finite bootstrap tail reporting (`0/5000`, never `< 0.0002` or `0.0000`) & exact sign-flip test
    assert format_boot_p(0.0, 5000) == "0/5000"
    assert format_boot_p(0.0012, 5000) == "0.0012"
    assert abs(exact_two_sided_signflip_p(np.array([1.0] * 10)) - (2.0 / 1024.0)) < 1e-12

    # 4. Verify exact 900 -> 802 non-shortcut unit count on real frozen `c3_blind_confirm_v3_bases.jsonl`
    v3_p = C35 / "data" / "c3_blind_confirm_v3_bases.jsonl"
    all_u, non_sc_u, by_pt = compute_v3_non_shortcut_units(v3_p)
    assert len(all_u) == 900 and len(non_sc_u) == 802, (len(all_u), len(non_sc_u))
    assert len(by_pt["argument_flip"]) == 390 and len(by_pt["context_flip"]) == 390 and len(by_pt["crossed_2x2"]) == 120

    # 5. Verify OOD 50/50 strictness:
    #    (a) Rejects unknown slice ("1 converse + 1 depth4 + 1 unknown")
    #    (b) Rejects non-60/60 when require_exact_60_60=True
    #    (c) Exact unrounded math: 0/60 converse + 10/60 depth4 MUST equal 8.33% (NEVER 8.34%!)
    bad_slice_map = {"C0": "ood_converse", "D0": "ood_depth4", "X0": "ood_unknown"}
    bad_bm = {k: {"family_id": k, "four_cell": 1} for k in bad_slice_map}
    try:
        score_ood_equal_slice_50_50(bad_bm, None, bad_slice_map, require_exact_60_60=False)
        raise RuntimeError("Should have rejected unknown OOD slice!")
    except AssertionError:
        pass
    try:
        score_ood_equal_slice_50_50(bad_bm, None, {"C0": "ood_converse", "D0": "ood_depth4"}, require_exact_60_60=True)
        raise RuntimeError("Should have rejected 1+1 when require_exact_60_60=True!")
    except AssertionError:
        pass

    slice_map = {f"C{i}": "ood_converse" for i in range(60)} | {f"D{i}": "ood_depth4" for i in range(60)}
    bm_round = {f"C{i}": {"family_id": f"CF{i % 8}", "four_cell": 0} for i in range(60)} | {
        f"D{i}": {"family_id": f"DF{i}", "four_cell": 1 if i < 10 else 0} for i in range(60)
    }
    ood_round = score_ood_equal_slice_50_50(bm_round, None, slice_map, expected_bases=set(slice_map.keys()), require_exact_60_60=True, n_boot=500)
    assert ood_round["equal_50_50_score_pct"] == 8.33, ood_round  # exact 0.5*(0/60 + 10/60)*100 = 8.3333... -> 8.33%!

    # Empty eligible slice handling when require_exact_60_60=False
    empty_res = score_ood_equal_slice_50_50(bm_round, None, {"D0": "ood_depth4"}, require_exact_60_60=False)
    assert empty_res["equal_50_50_score_pct"] is None and "empty_slice_warning" in empty_res

    # 6. Verify entity_balanced_v1 6-cell summary on real frozen metadata
    eb_p = C35 / "data" / "c3_entity_balanced_v1_bases.jsonl"
    eb_meta = {r["base_id"]: r for r in (json.loads(l) for l in eb_p.read_text(encoding="utf-8").splitlines() if l.strip())}
    bm_eb = {bid: {"family_id": bid, "four_cell": 1 if m["entity_balance_cell"] == "1v2_orig" else 0, "joint_t1_4cell": 0} for bid, m in eb_meta.items()}
    eb_sum = summarize_entity_balanced_cells(bm_eb, eb_meta)
    assert eb_sum["shortcut_asymmetry_1v2_minus_2v1_pp"] == 100.0 and eb_sum["same_entity_count_avg_4cell_pct"] == 0.0, eb_sum

    # 7. FULL REAL-SCHEMA END-TO-END SMOKE TEST of `run_stage6_aggregation()` including Section 5 auto-generation!
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        mini_v3_rows = []
        for i in range(12):
            bid = f"MINI-V3-{i + 1:03d}"
            mini_v3_rows.append({
                "base_id": bid, "family_id": bid, "unit_id": bid,
                "pair_type": "argument_flip" if i < 6 else "context_flip",
                "fallacy_class": "missing_conjunct" if i < 2 else "scope_swap",
                "broken_step": (i % 3) + 1, "initial_answer": "A", "target_answer": "B",
            })
        mini_v3_file = td_p / "mini_v3.jsonl"
        mini_v3_file.write_text("\n".join(json.dumps(r) for r in mini_v3_rows) + "\n", encoding="utf-8")
        mini_th_file = td_p / "mini_think_v3.jsonl"
        mini_th_file.write_text("\n".join(json.dumps(r) for r in mini_v3_rows[:4]) + "\n", encoding="utf-8")

        mini_eb_rows = [
            {"base_id": f"MINI-EB-{cell}", "family_id": f"MINI-EB-{cell}", "unit_id": f"MINI-EB-{cell}",
             "entity_balance_cell": cell, "initial_answer": "A", "target_answer": "B"}
            for cell in ("1v2_orig", "2v1_anti", "1v1_same", "2v2_same", "1v1_ctx", "2v2_ctx")
        ]
        mini_eb_file = td_p / "mini_eb.jsonl"
        mini_eb_file.write_text("\n".join(json.dumps(r) for r in mini_eb_rows) + "\n", encoding="utf-8")

        arms_15 = [
            "turn1_V", "turn1_F", "V", "F", "V_auth", "V_hes", "F_auth", "F_hes",
            "F_src_third_party", "S", "shell_only", "AO", "N", "P0_confirm_a0", "I0_confirm_a0",
        ]
        arms_th = ["turn1_V", "turn1_F", "V_auth", "V_hes", "F_auth", "F_hes"]

        def _write_fake_arm(arm_dir: Path, *, t1_ok_n: int, fc_ok_n: int, wiki_kl: float, alias: str) -> None:
            for sub in ("no_think", "think", "capabilities", "exfever"):
                (arm_dir / sub).mkdir(parents=True, exist_ok=True)
            nt_recs = []
            for idx_b, r in enumerate(mini_v3_rows):
                t1_pass = idx_b < t1_ok_n
                fc_pass = idx_b < fc_ok_n
                for a in arms_15:
                    if a in ("turn1_V", "turn1_F"):
                        is_i = t1_pass; is_t = not is_i
                    elif a in ("V", "V_auth", "V_hes"):
                        is_t = fc_pass; is_i = not is_t
                    elif a in ("F", "F_auth", "F_hes"):
                        is_i = fc_pass; is_t = not is_i
                    else:
                        is_i = True; is_t = False
                    pred = "B" if is_t else "A"
                    nt_recs.append({
                        "base_id": r["base_id"], "family_id": r["family_id"], "arm": a,
                        "initial_answer": "A", "target_answer": "B", "pred": pred,
                        "parse_ok": True, "is_target": is_t, "is_initial": is_i,
                        "u_theta": 5.0 if is_t else -5.0, "gold_nll": 0.1, "brier_score": 0.05,
                    })
            (arm_dir / "no_think" / "records_no_think_c3_blind_confirm_v3.jsonl").write_text(
                "\n".join(json.dumps(x) for x in nt_recs) + "\n", encoding="utf-8"
            )
            (arm_dir / "no_think" / "eval_summary_no_think.json").write_text(
                json.dumps({"reference_kl": {"kl_mean": wiki_kl}}, indent=2), encoding="utf-8"
            )
            th_recs = [
                {
                    "base_id": r["base_id"], "family_id": r["family_id"], "arm": a,
                    "initial_answer": "A", "target_answer": "B", "pred": "B" if a.startswith("V") else "A",
                    "parse_ok": True, "is_target": a.startswith("V"), "is_initial": not a.startswith("V"),
                }
                for r in mini_v3_rows[:4] for a in arms_th
            ]
            (arm_dir / "think" / "records_think_c3_blind_think_v3.jsonl").write_text(
                "\n".join(json.dumps(x) for x in th_recs) + "\n", encoding="utf-8"
            )
            (arm_dir / "capabilities" / "summary.json").write_text(
                json.dumps({"ifeval": {"n_correct_prompts": 460}, "gsm8k": {"n_correct": 635}}), encoding="utf-8"
            )
            (arm_dir / "vllm_lora_parity.json").write_text(
                json.dumps({"passed": True, "lora_hf_live_vs_vllm_native": {"option_match_rate": 1.0, "du_abs_mean": 0.15, "du_signed_mean": 0.01}}),
                encoding="utf-8",
            )
            (arm_dir / "exfever" / f"exfever_no_think_summary_{alias}.json").write_text(
                json.dumps({"metrics": {"conditions": {"pressure": {"flip_correct_to_wrong": {"k": 43, "n": 645}}, "evidence": {"flip_wrong_to_correct": {"k": 209, "n": 355}}}}}),
                encoding="utf-8",
            )

        _write_fake_arm(td_p / "u2_s42", t1_ok_n=12, fc_ok_n=6, wiki_kl=0.0012, alias="u2")
        _write_fake_arm(td_p / "u2_s1024", t1_ok_n=12, fc_ok_n=6, wiki_kl=0.0012, alias="u2")
        _write_fake_arm(td_p / "v4_s42", t1_ok_n=9, fc_ok_n=10, wiki_kl=0.0008, alias="v4")
        _write_fake_arm(td_p / "v4_s1024", t1_ok_n=9, fc_ok_n=10, wiki_kl=0.0008, alias="v4")

        eb_recs = [
            {
                "base_id": r["base_id"], "family_id": r["family_id"], "arm": a,
                "initial_answer": "A", "target_answer": "B", "pred": "B" if a.startswith("V") else "A",
                "parse_ok": True, "is_target": a.startswith("V"), "is_initial": not a.startswith("V"),
                "u_theta": 5.0 if a.startswith("V") else -5.0,
            }
            for r in mini_eb_rows for a in arms_th
        ]
        eb_rec_file = td_p / "eb_v4.jsonl"
        eb_rec_file.write_text("\n".join(json.dumps(x) for x in eb_recs) + "\n", encoding="utf-8")

        smoke_cfg = {
            "scale_14b": {
                "v3_file": str(mini_v3_file),
                "think_v3_file": str(mini_th_file),
                "ifeval_m0": 461,
                "gsm8k_m0": 638,
                "arms": {
                    "u2_corridor": {"42": str(td_p / "u2_s42"), "1024": str(td_p / "u2_s1024")},
                    "M3-v4-PureMid-MLP+": {"42": str(td_p / "v4_s42"), "1024": str(td_p / "v4_s1024")},
                },
                "pair_specs": [["v4_vs_u2", "M3-v4-PureMid-MLP+", "u2_corridor"]],
            },
            "fixed_step200": {"v3_file": str(mini_v3_file), "entries": [{"name": "v4_step200", "seed": 42, "dir": str(td_p / "v4_s42")}]},
            "entity_balanced": {"file": str(mini_eb_file), "entries": [{"name": "v4", "records": str(eb_rec_file)}]},
            "external_8b": {"v3_file": str(mini_v3_file), "entries": [{"name": "v4", "dir": str(td_p / "v4_s42")}]},
        }
        out_doc, out_md = run_stage6_aggregation(smoke_cfg, td_p / "out", n_boot=200)
        assert "52.21" in out_md  # exact unrounded 209/355 - 43/645 = 52.207... -> 52.21!
        assert "## 5. 8B vLLM Backend Parity Audit" in out_md
        assert "backend_parity_8b" in out_doc and "v4" in out_doc["backend_parity_8b"]

    print(
        "[OK] c35_stage6_stats self-test PASSED: 9/9 adversarial record faults rejected, strict M0 dual-elig, "
        "0/5000 tail format + exact sign-flip test, exact 8.33% unrounded OOD 50/50, and full Section 1-5 auto-reproducibility!"
    )


def run_stage6_aggregation(cfg: dict[str, Any], out_dir: Path, n_boot: int = 5000) -> tuple[dict[str, Any], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_doc: dict[str, Any] = {
        "schema": "cedar.acl_exp_c35.stage6_stats.v2",
        "n_boot": n_boot,
        "boot_p_note": f"Two-sided bootstrap p = 2*min(P(boot<=0), P(boot>=0)); 0 tail hits shown as 0/{n_boot} (min nonzero resolution {2.0/n_boot:.4f}); exact paired sign-flip p_signflip logged in JSON.",
        "sections": {},
    }
    md: list[str] = ["# ACL-exp-c3.5 Stage 6 Strict Closure Readout\n"]

    # Section 1: 14B Dual-Seed (42 + 1024) on v3 (900 + 802 sensitivity)
    if "scale_14b" in cfg:
        s14 = cfg["scale_14b"]
        v3_file = Path(s14["v3_file"])
        v3_meta = _load_base_meta(v3_file)
        th_v3_file = Path(s14.get("think_v3_file", C35 / "data" / "c3_blind_think_v3_bases.jsonl"))
        exp_th_b = set(_load_base_meta(th_v3_file).keys()) if th_v3_file.is_file() else None
        all_u, non_sc_u, _ = compute_v3_non_shortcut_units(v3_file)
        exp_b = set(v3_meta.keys())
        exp_arms = set(s14.get("expected_nothink_arms", [
            "turn1_V", "turn1_F", "V", "F", "V_auth", "V_hes", "F_auth", "F_hes",
            "F_src_third_party", "S", "shell_only", "AO", "N", "P0_confirm_a0", "I0_confirm_a0",
        ]))
        bms_14: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        rows_14: dict[str, Any] = {}
        md.append("## 1. Qwen3-14B Dual-Seed (`seed=42, 1024`) on `c3_blind_confirm_v3` (`900` Units + `802` Non-Shortcut)\n")
        md.append("| Model | Seed | 4-Cell (900 units) | Joint (900 units) | Turn-1 Dual (1080 items) | Wiki KL | IFEval / GSM8K | Health |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for mname, sdict in s14["arms"].items():
            seed_4c, seed_jt, seed_t1, seed_kl = [], [], [], []
            for sd, dstr in sorted(sdict.items(), key=lambda kv: int(kv[0])):
                d = Path(dstr)
                nt_p = d / "no_think" / "records_no_think_c3_blind_confirm_v3.jsonl"
                recs = C3S._load_jsonl(nt_p)
                validate_records_strict(recs, expected_bases=exp_b, expected_arms=exp_arms, regime="no_think", label=f"14b:{mname}@s{sd}")
                info, bm = S5._load_arm_strict(
                    d, "c3_blind_confirm_v3", v3_meta, all_u, exp_b,
                    ifeval_m0=int(s14["ifeval_m0"]), gsm8k_m0=int(s14["gsm8k_m0"]),
                    think_dataset_name="c3_blind_think_v3", arm_label=f"14b:{mname}@s{sd}",
                )
                th_p = d / "think" / "records_think_c3_blind_think_v3.jsonl"
                if th_p.is_file():
                    assert exp_th_b is not None, f"Independent think_v3_file missing for validating {th_p}"
                    th_recs = C3S._load_jsonl(th_p)
                    validate_records_strict(
                        th_recs, expected_bases=exp_th_b,
                        expected_arms={"turn1_V", "turn1_F", "V_auth", "V_hes", "F_auth", "F_hes"},
                        regime="think", label=f"14b_think:{mname}@s{sd}",
                    )
                bms_14[mname][sd] = bm
                s = info["all"]
                # Keep UNROUNDED unit-weighted raw floats for exact pooled_2seed means!
                fa_raw = C3S.family_weighted_scores(bm, "four_cell")
                fj_raw = C3S.family_weighted_scores(bm, "joint_t1_4cell")
                fc = float(np.mean([fa_raw[u] for u in all_u])) * 100.0
                jt = float(np.mean([fj_raw[u] for u in all_u])) * 100.0
                t1 = float(np.mean([float(v["turn1_ok"]) for v in bm.values()])) * 100.0
                kl = info["wiki_kl"]
                seed_4c.append(fc); seed_jt.append(jt); seed_t1.append(t1); seed_kl.append(kl)
                h_str = "PASS" if info["health"]["all_passed"] else "FAIL"
                md.append(f"| `{mname}` | {sd} | **{fc:.2f}%** | **{jt:.2f}%** | {t1:.2f}% | `{kl:.6f}` | {info['ifeval']} / {info['gsm8k']} | {h_str} |")
                rows_14[f"{mname}@s{sd}"] = {
                    "four_cell_pct": round(fc, 2),
                    "joint_pct": round(jt, 2),
                    "turn1_dual_pct": round(t1, 2),
                    "wiki_kl": kl,
                    "ifeval": info["ifeval"],
                    "gsm8k": info["gsm8k"],
                    "health_passed": bool(info["health"]["all_passed"]),
                }
            if len(seed_4c) >= 2:
                md.append(f"| `{mname}` | **pooled_2seed** | **{float(np.mean(seed_4c)):.2f}%** | **{float(np.mean(seed_jt)):.2f}%** | {float(np.mean(seed_t1)):.2f}% | `{float(np.mean(seed_kl)):.6f}` | — | — |")

        md.append("\n### 14B Paired Bootstrap (All `900` Units + Post-Hoc `802` Non-Shortcut Sensitivity)\n")
        md.append(f"*Note: Paired 95% CIs reflect item/unit sampling uncertainty given fixed training seeds (`n_boot={n_boot}`). Zero two-sided tail hits reported as `0/{n_boot}` (min nonzero two-sided resolution `{2.0/n_boot:.4f}`); exact sign-flip p-values logged in JSON.*\n")
        md.append("| Comparison | Split | Subset | 4-Cell Diff [95% CI] | Joint Diff [95% CI] | Boot `p` (4C) | Sign-Flip `p` (4C) |")
        md.append("|---|---|---|---:|---:|---:|---:|")
        cmp_14: dict[str, Any] = {}
        for pname, ma, mb in s14["pair_specs"]:
            for sd in sorted(set(bms_14[ma]) & set(bms_14[mb]), key=int):
                for sub_name, sub_u in (("all_900", all_u), ("non_shortcut_802", non_sc_u)):
                    r4 = paired_unit_bootstrap_subset(bms_14[ma][sd], bms_14[mb][sd], sub_u, "four_cell", n_boot=n_boot)
                    rj = paired_unit_bootstrap_subset(bms_14[ma][sd], bms_14[mb][sd], sub_u, "joint_t1_4cell", n_boot=n_boot)
                    cmp_14[f"{pname}|s{sd}|{sub_name}"] = {"four_cell": r4, "joint": rj}
                    md.append(f"| `{pname}` | `seed_{sd}` | `{sub_name}` | **{r4['diff_pp']:+.2f}** `[{r4['ci95_pp'][0]:+.2f}, {r4['ci95_pp'][1]:+.2f}]` | **{rj['diff_pp']:+.2f}** `[{rj['ci95_pp'][0]:+.2f}, {rj['ci95_pp'][1]:+.2f}]` | `{r4['p_str']}` | `{_fmt_exact_p(r4['p_signflip'])}` |")
            if len(set(bms_14[ma]) & set(bms_14[mb])) >= 2:
                for sub_name, sub_u in (("all_900", all_u), ("non_shortcut_802", non_sc_u)):
                    r4 = pooled_paired_bootstrap_subset(bms_14[ma], bms_14[mb], sub_u, "four_cell", n_boot=n_boot)
                    rj = pooled_paired_bootstrap_subset(bms_14[ma], bms_14[mb], sub_u, "joint_t1_4cell", n_boot=n_boot)
                    cmp_14[f"{pname}|pooled_2seed|{sub_name}"] = {"four_cell": r4, "joint": rj}
                    md.append(f"| `{pname}` | **`pooled_2seed`** | `{sub_name}` | **{r4['diff_pp']:+.2f}** `[{r4['ci95_pp'][0]:+.2f}, {r4['ci95_pp'][1]:+.2f}]` | **{rj['diff_pp']:+.2f}** `[{rj['ci95_pp'][0]:+.2f}, {rj['ci95_pp'][1]:+.2f}]` | `{r4['p_str']}` | `{_fmt_exact_p(r4['p_signflip'])}` |")
        out_doc["sections"]["scale_14b"] = {"arms": rows_14, "comparisons": cmp_14}

    # Section 2: Fixed Step-200 Matched-Step Confound Check (14B)
    if "fixed_step200" in cfg:
        md.append("\n## 2. Qwen3-14B Fixed `step=200` Matched-Step Confound Audit (`v3` & WikiText-2 KL)\n")
        md.append("| Model (at exact `step 200`) | Seed | 4-Cell (900 units) | Joint (900 units) | Wiki KL (`step 200`) |")
        md.append("|---|---:|---:|---:|---:|")
        s200_out: dict[str, Any] = {}
        v3_meta = _load_base_meta(Path(cfg["fixed_step200"]["v3_file"]))
        for entry in cfg["fixed_step200"]["entries"]:
            d = Path(entry["dir"])
            art = C3S._load_arm_artifacts(d, "c3_blind_confirm_v3")
            bm = S35.compute_base_metrics_v2(art["nt_recs"], base_meta=v3_meta)
            fa_r = C3S.family_weighted_scores(bm, "four_cell")
            fj_r = C3S.family_weighted_scores(bm, "joint_t1_4cell")
            fc = float(np.mean(list(fa_r.values()))) * 100.0
            jt = float(np.mean(list(fj_r.values()))) * 100.0
            kl = art.get("wiki_kl")
            md.append(f"| `{entry['name']}` | {entry['seed']} | **{fc:.2f}%** | **{jt:.2f}%** | `{kl:.6f}` |")
            s200_out[f"{entry['name']}@s{entry['seed']}"] = {"four_cell_pct": round(fc, 2), "joint_pct": round(jt, 2), "wiki_kl": kl}
        out_doc["sections"]["fixed_step200"] = s200_out

    # Section 3: Entity-Count-Balanced 6-Cell Diagnostic (`c3_entity_balanced_v1`, 480 units)
    if "entity_balanced" in cfg:
        eb_file = Path(cfg["entity_balanced"]["file"])
        eb_meta = {r["base_id"]: r for r in (json.loads(l) for l in eb_file.read_text(encoding="utf-8").splitlines() if l.strip())}
        md.append("\n## 3. Entity-Count-Balanced Diagnostic (`c3_entity_balanced_v1`, 480 Units, Exact 50/50 Entity Count)\n")
        md.append("| Model | Overall 4C | `1v2_orig` | `2v1_anti` | `Δ_shortcut` | `1v1_same` | `2v2_same` | `1v1_ctx` | `2v2_ctx` | Same-Ent Avg |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        eb_out: dict[str, Any] = {}
        for entry in cfg["entity_balanced"]["entries"]:
            rec_p = Path(entry["records"])
            recs = C3S._load_jsonl(rec_p)
            bm = S35.compute_base_metrics_v2(recs)
            res = summarize_entity_balanced_cells(bm, eb_meta)
            c = res["by_cell"]
            md.append(
                f"| `{entry['name']}` | **{res['overall_four_cell_pct']:.2f}%** | {c['1v2_orig']['four_cell_pct']:.1f}% | "
                f"{c['2v1_anti']['four_cell_pct']:.1f}% | **{res['shortcut_asymmetry_1v2_minus_2v1_pp']:+.1f}** | "
                f"{c['1v1_same']['four_cell_pct']:.1f}% | {c['2v2_same']['four_cell_pct']:.1f}% | "
                f"{c['1v1_ctx']['four_cell_pct']:.1f}% | {c['2v2_ctx']['four_cell_pct']:.1f}% | **{res['same_entity_count_avg_4cell_pct']:.2f}%** |"
            )
            eb_out[entry["name"]] = res
        out_doc["sections"]["entity_balanced"] = eb_out

    # Section 4: 8B External SOTA & Internal Loss-Ablation Table + Auto-Generated Section 5 Backend Parity
    if "external_8b" in cfg:
        e8 = cfg["external_8b"]
        v3_file = Path(e8["v3_file"])
        v3_meta = _load_base_meta(v3_file)
        all_u, non_sc_u, _ = compute_v3_non_shortcut_units(v3_file)
        md.append("\n## 4. Qwen3-8B External SOTA & Internal Loss Ablations (`v3`, KL, Capabilities, EX-FEVER)\n")
        md.append("| Method | `v3` 4C (900) | `v3` 4C (802) | `v3` Joint | `v3` NLL (4-tone) | `|u|` med (4-tone) | Wiki KL | IFEval / GSM8K | EX-FEVER `R_UY / R_RU_ev / Sel` |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ext_out: dict[str, Any] = {}
        tone_arms = {"V_auth", "V_hes", "F_auth", "F_hes"}
        for entry in e8["entries"]:
            d = Path(entry["dir"])
            art = C3S._load_arm_artifacts(d, "c3_blind_confirm_v3")
            bm = S35.compute_base_metrics_v2(art["nt_recs"], base_meta=v3_meta)
            fa_all = C3S.family_weighted_scores(bm, "four_cell")
            fj_all = C3S.family_weighted_scores(bm, "joint_t1_4cell")
            fc900 = float(np.mean([fa_all[u] for u in all_u])) * 100.0
            fc802 = float(np.mean([fa_all[u] for u in non_sc_u])) * 100.0
            jt900 = float(np.mean([fj_all[u] for u in all_u])) * 100.0
            s_v3 = S35.summarize_v2(bm)
            nll4 = s_v3.get("mean_4opt_nll")
            u4 = [abs(float(r["u_theta"])) for r in art["nt_recs"] if r.get("arm") in tone_arms and r.get("u_theta") is not None]
            u4_med = float(np.median(u4)) if u4 else None
            kl = art.get("wiki_kl")
            ife, gsm = art.get("ifeval"), art.get("gsm8k")
            ex = S5._load_exfever_summary(d, entry["name"], regime="no_think")
            ex_str = _exact_exfever_headline_str(ex)
            kl_s = f"`{kl:.6f}`" if kl is not None else "—"
            cap_s = f"{ife} / {gsm}" if (ife is not None and gsm is not None) else "—"
            nll_s = f"{nll4:.2f}" if nll4 is not None else "—"
            u4_s = f"{u4_med:.3f}" if u4_med is not None else "—"
            md.append(f"| `{entry['name']}` | **{fc900:.2f}%** | **{fc802:.2f}%** | **{jt900:.2f}%** | {nll_s} | {u4_s} | {kl_s} | {cap_s} | {ex_str} |")
            ext_out[entry["name"]] = {
                "fc900": round(fc900, 2), "fc802": round(fc802, 2), "jt900": round(jt900, 2),
                "nll_4tone": round(float(nll4), 4) if nll4 is not None else None,
                "abs_u_median_4tone": round(u4_med, 4) if u4_med is not None else None,
                "wiki_kl": kl, "ifeval": ife, "gsm8k": gsm, "exfever": ex_str,
            }
        out_doc["sections"]["external_8b"] = ext_out

        parity_doc, parity_md = _collect_8b_backend_parity(e8["entries"])
        if parity_doc:
            out_doc["backend_parity_8b"] = parity_doc
            md.extend(parity_md)

    out_json = out_dir / "c35_stage6_stats.json"
    out_md = out_dir / "C35_STAGE6_READOUT.md"
    md_text = "\n".join(md) + "\n"
    out_json.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(md_text, encoding="utf-8", newline="\n")
    print(f"[OK] Wrote Stage 6 strict readout -> {out_md} & {out_json}")
    return out_doc, md_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--config-json", type=Path, help="Stage 6 aggregation manifest JSON")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.config_json and args.out_dir, "--config-json and --out-dir are required"
    cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    run_stage6_aggregation(cfg, args.out_dir, n_boot=int(args.n_boot))


if __name__ == "__main__":
    main()
