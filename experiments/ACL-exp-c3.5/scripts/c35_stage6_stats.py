#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 6 Strict Aggregator & Audit Validator.

Directly implements ALL statistical, schema, and framing fixes required by Astra's Stage 5 review (794c9b3):
  1. STRICT FAIL-CLOSED RECORD VALIDATOR (`validate_records_strict`):
       - Rejects empty files (NO empty `think` file silently passing with `parse_fail=0.0`!).
       - Hard-asserts EXACTLY 1 record per `(base_id, arm)` over `expected_bases x expected_arms`.
       - Hard-asserts every required field is present & non-null, and `parse_ok` is an explicit `bool`
         (NEVER defaults a missing `parse_ok` to `True`!).
  2. HONEST BOOTSTRAP P-VALUE FORMATTING (`format_boot_p`):
       - Never prints `p=0.0000` for finite `n_boot=5000`; reports exact resolution floor (`< 0.0002` when 0 tail hits).
  3. FULL MAIN + POST-HOC 802-UNIT SENSITIVITY + PAIR-TYPE BREAKDOWN:
       - Automatically appends the `802`-unit sensitivity row (excluding the 98 `missing_conjunct / argument_flip`
         units identified by Astra) directly below the `900`-unit `v3` comparison table, plus `argument_flip` /
         `context_flip` / `crossed_2x2` breakdowns.
       - Explicitly reports `Turn-1 Dual Acc` alongside `Health PASS` (clarifying Health PASS scope) and adds
         explicit `pooled_2seed` rows to readout tables.
  4. STAGE 6 MODULE AGGREGATORS:
       - `14B` dual-seed (`seed=42 + seed=1024`) single-seed + pooled CIs & matched `step 200` KL/accuracy table.
       - `c3_entity_balanced_v1` (`480` units): 6-cell breakdown (`1v2_orig`, `2v1_anti`, `1v1_same`, `2v2_same`,
         `1v1_ctx`, `2v2_ctx`) + shortcut asymmetry `Delta_shortcut = Acc(1v2_orig) - Acc(2v1_anti)`.
       - `8B` external SOTA (`Pinpoint-DLA-Corridor`, `Pinpoint-DLA-SFT`, `Joint-DPO`, `CAA-Steering`) &
         internal loss ablation (`u2_corridor_only`, `u2_tone_inv_only`) tables.
  5. SEALED OOD 50/50 EQUAL-SLICE EVALUATOR (`score_ood_equal_slice_50_50`):
       - Implements Astra's pre-unseal fix: `ood_converse` (60 items / 8 families) and `ood_depth4` (60 items / 60 families)
         each receive EXACTLY 50% weight (never 88.2% depth4!), with separate per-slice raw scores & paired CIs.
       - Tested ONLY on synthetic toy records in `--selftest-tiny` (NEVER touches real sealed files!).

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_stage6_stats.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
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

REQUIRED_BASE_FIELDS = ("base_id", "arm", "initial_answer", "target_answer", "parse_ok", "is_target", "is_initial")
REQUIRED_NOTHINK_FIELDS = REQUIRED_BASE_FIELDS + ("pred", "u_theta")


def _load_base_meta(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {str(r["base_id"]): r for r in rows}


def format_boot_p(p_raw: float, n_boot: int) -> str:
    """Never emit `p=0.0000` for finite bootstrap; use `< {1/n_boot}` floor."""
    floor = 1.0 / max(1, int(n_boot))
    if p_raw < floor:
        return f"< {floor:.4f}"
    return f"{p_raw:.4f}"


def validate_records_strict(
    recs: list[dict[str, Any]],
    *,
    expected_bases: set[str],
    expected_arms: set[str],
    regime: str,
    label: str,
) -> dict[str, Any]:
    """Fail-closed record integrity validator (Astra's Stage 5 fix #2)."""
    if not recs:
        raise AssertionError(f"[EMPTY RECORDS] {label} ({regime}) contains 0 records!")

    req_fields = REQUIRED_NOTHINK_FIELDS if regime == "no_think" else REQUIRED_BASE_FIELDS
    seen: set[tuple[str, str]] = set()
    dup_count = 0
    n_parse_fail = 0

    for i, r in enumerate(recs):
        for f in req_fields:
            if f not in r or r[f] is None:
                raise AssertionError(f"[MISSING FIELD] {label} row {i}: required field {f!r} missing or None!")
        if not isinstance(r["parse_ok"], bool):
            raise AssertionError(f"[BAD parse_ok TYPE] {label} row {i}: expected bool, got {type(r['parse_ok'])} ({r['parse_ok']!r})")
        if not isinstance(r["is_target"], bool) or not isinstance(r["is_initial"], bool):
            raise AssertionError(f"[BAD BOOL FIELD] {label} row {i}: is_target/is_initial must be bool")
        if r["parse_ok"] and r.get("pred") is None:
            raise AssertionError(f"[MISSING pred] {label} row {i}: parse_ok=True but pred is None!")
        if not r["parse_ok"]:
            n_parse_fail += 1
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
        "parse_fail_rate": round(n_parse_fail / len(recs), 6),
        "validated": True,
    }


def paired_unit_bootstrap_subset(
    bm_a: dict[str, dict[str, Any]],
    bm_b: dict[str, dict[str, Any]],
    subset_units: set[str],
    metric: str,
    n_boot: int = 5000,
    seed: int = 2027,
) -> dict[str, Any]:
    """Paired bootstrap over an explicit unit subset (e.g., 900 all, 802 non-shortcut, 390 AF/CF, 120 CE)."""
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
    return {
        "n_units": n,
        "mean_a_pct": round(float(va.mean()) * 100.0, 2),
        "mean_b_pct": round(float(vb.mean()) * 100.0, 2),
        "diff_pp": round(float(diffs.mean()) * 100.0, 2),
        "ci95_pp": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
        "p_two_sided_raw": round(p_two, 6),
        "p_str": format_boot_p(p_two, n_boot),
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
    return {
        "n_units": n,
        "seeds": seeds,
        "mean_a_pct": round(float(va.mean()) * 100.0, 2),
        "mean_b_pct": round(float(vb.mean()) * 100.0, 2),
        "diff_pp": round(float(diffs.mean()) * 100.0, 2),
        "ci95_pp": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
        "p_two_sided_raw": round(p_two, 6),
        "p_str": format_boot_p(p_two, n_boot),
    }


def summarize_entity_balanced_cells(
    bm: dict[str, dict[str, Any]],
    base_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize all 6 cells of `c3_entity_balanced_v1` + exact entity-count shortcut asymmetry."""
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bid, row in bm.items():
        cell = base_meta[bid]["entity_balance_cell"]
        by_cell[cell].append(row)

    out_cells: dict[str, Any] = {}
    for cell in ("1v2_orig", "2v1_anti", "1v1_same", "2v2_same", "1v1_ctx", "2v2_ctx"):
        rs = by_cell.get(cell, [])
        assert rs, f"Missing entity_balanced cell: {cell}"
        fc = float(np.mean([r["four_cell"] for r in rs])) * 100.0
        jt = float(np.mean([r["joint_t1_4cell"] for r in rs])) * 100.0
        out_cells[cell] = {"n": len(rs), "four_cell_pct": round(fc, 2), "joint_pct": round(jt, 2)}

    all_fc = float(np.mean([r["four_cell"] for r in bm.values()])) * 100.0
    all_jt = float(np.mean([r["joint_t1_4cell"] for r in bm.values()])) * 100.0
    shortcut_asym = round(out_cells["1v2_orig"]["four_cell_pct"] - out_cells["2v1_anti"]["four_cell_pct"], 2)
    same_ent_avg = round(
        0.25 * (
            out_cells["1v1_same"]["four_cell_pct"]
            + out_cells["2v2_same"]["four_cell_pct"]
            + out_cells["1v1_ctx"]["four_cell_pct"]
            + out_cells["2v2_ctx"]["four_cell_pct"]
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
) -> dict[str, Any]:
    """Astra's exact pre-unseal OOD weighting: 50% `ood_converse` + 50% `ood_depth4` (NEVER 88.2% depth4!).

    Resamples items WITHIN each slice independently (stratified paired bootstrap) so both slices keep exact 50/50 weight!
    """
    slices = ("ood_converse", "ood_depth4")
    ids_by_slice = {sl: sorted(bid for bid, s in slice_map.items() if s == sl) for sl in slices}
    for sl in slices:
        assert ids_by_slice[sl], f"Empty OOD slice: {sl}"
        for bid in ids_by_slice[sl]:
            assert bid in bm_a, f"Missing {bid} in model A"
            if bm_b is not None:
                assert bid in bm_b, f"Missing {bid} in model B"

    per_slice: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    boot_a_slices = []
    boot_diff_slices = []

    for sl in slices:
        bids = ids_by_slice[sl]
        va = np.array([float(bm_a[b][metric]) for b in bids], dtype=np.float64)
        idx = rng.integers(0, len(bids), size=(n_boot, len(bids)))
        boot_a = va[idx].mean(axis=1)
        boot_a_slices.append(boot_a)
        sl_doc: dict[str, Any] = {"n_items": len(bids), "raw_score_pct": round(float(va.mean()) * 100.0, 2)}
        if bm_b is not None:
            vb = np.array([float(bm_b[b][metric]) for b in bids], dtype=np.float64)
            d = va - vb
            bd = d[idx].mean(axis=1)
            boot_diff_slices.append(bd)
            p_two = min(1.0, 2.0 * min(float(np.mean(bd <= 0.0)), float(np.mean(bd >= 0.0))))
            sl_doc.update({
                "raw_score_b_pct": round(float(vb.mean()) * 100.0, 2),
                "diff_pp": round(float(d.mean()) * 100.0, 2),
                "ci95_pp": [round(float(np.quantile(bd, 0.025)) * 100.0, 2), round(float(np.quantile(bd, 0.975)) * 100.0, 2)],
                "p_str": format_boot_p(p_two, n_boot),
            })
        per_slice[sl] = sl_doc

    equal_50_50_a = 0.5 * (per_slice["ood_converse"]["raw_score_pct"] + per_slice["ood_depth4"]["raw_score_pct"])
    out: dict[str, Any] = {
        "weighting": "50pct_ood_converse_50pct_ood_depth4",
        "equal_50_50_score_pct": round(equal_50_50_a, 2),
        "per_slice": per_slice,
    }
    if bm_b is not None:
        equal_50_50_b = 0.5 * (per_slice["ood_converse"]["raw_score_b_pct"] + per_slice["ood_depth4"]["raw_score_b_pct"])
        comb_boot = 0.5 * (boot_diff_slices[0] + boot_diff_slices[1])
        p_two = min(1.0, 2.0 * min(float(np.mean(comb_boot <= 0.0)), float(np.mean(comb_boot >= 0.0))))
        out.update({
            "equal_50_50_score_b_pct": round(equal_50_50_b, 2),
            "equal_50_50_diff_pp": round(equal_50_50_a - equal_50_50_b, 2),
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


def run_selftest_tiny() -> None:
    # 1. Fail-closed validator catches empty file, missing parse_ok, duplicate key, missing arm!
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
    ]:
        raised = False
        try:
            validate_records_strict(bad_recs, expected_bases=exp_b, expected_arms=exp_a, regime="think" if "think" in bad_name else "no_think", label=bad_name)
        except AssertionError:
            raised = True
        assert raised, f"Validator failed to reject {bad_name}!"

    # 2. Verify finite bootstrap p-value never prints 0.0000
    assert format_boot_p(0.0, 5000) == "< 0.0002"
    assert format_boot_p(0.0012, 5000) == "0.0012"

    # 3. Verify exact 900 -> 802 non-shortcut unit count on real frozen `c3_blind_confirm_v3_bases.jsonl`
    v3_p = C35 / "data" / "c3_blind_confirm_v3_bases.jsonl"
    all_u, non_sc_u, by_pt = compute_v3_non_shortcut_units(v3_p)
    assert len(all_u) == 900 and len(non_sc_u) == 802, (len(all_u), len(non_sc_u))
    assert len(by_pt["argument_flip"]) == 390 and len(by_pt["context_flip"]) == 390 and len(by_pt["crossed_2x2"]) == 120

    # 4. Verify synthetic OOD 50/50 weighting gives EXACT 50/50 even with asymmetric 8 vs 60 family counts!
    # Suppose model A scores 100% on converse (60 items, 8 families) and 0% on depth4 (60 items, 60 families):
    # Family-equal would wrongly give 8/68 = 11.76%; exact 50/50 MUST give 50.00%!
    slice_map = {f"C{i}": "ood_converse" for i in range(60)} | {f"D{i}": "ood_depth4" for i in range(60)}
    bm_a = {f"C{i}": {"family_id": f"CF{i % 8}", "four_cell": 1} for i in range(60)} | {f"D{i}": {"family_id": f"DF{i}", "four_cell": 0} for i in range(60)}
    bm_b = {f"C{i}": {"family_id": f"CF{i % 8}", "four_cell": 0} for i in range(60)} | {f"D{i}": {"family_id": f"DF{i}", "four_cell": 0} for i in range(60)}
    ood_res = score_ood_equal_slice_50_50(bm_a, bm_b, slice_map, n_boot=1000)
    assert ood_res["equal_50_50_score_pct"] == 50.0 and ood_res["equal_50_50_diff_pp"] == 50.0, ood_res

    # 5. Verify entity_balanced_v1 6-cell summary on real frozen metadata (using synthetic predictions)
    eb_p = C35 / "data" / "c3_entity_balanced_v1_bases.jsonl"
    eb_meta = {r["base_id"]: r for r in (json.loads(l) for l in eb_p.read_text(encoding="utf-8").splitlines() if l.strip())}
    bm_eb = {bid: {"family_id": bid, "four_cell": 1 if m["entity_balance_cell"] == "1v2_orig" else 0, "joint_t1_4cell": 0} for bid, m in eb_meta.items()}
    eb_sum = summarize_entity_balanced_cells(bm_eb, eb_meta)
    assert eb_sum["shortcut_asymmetry_1v2_minus_2v1_pp"] == 100.0 and eb_sum["same_entity_count_avg_4cell_pct"] == 0.0, eb_sum

    # 6. FULL REAL-SCHEMA END-TO-END SMOKE TEST of `run_stage6_aggregation()` using REAL `summarize_v2()` outputs!
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Build a miniature 12-base v3 dataset (including 2 missing_conjunct/argument_flip shortcut units + 10 others)
        mini_v3_rows = []
        for i in range(12):
            bid = f"MINI-V3-{i + 1:03d}"
            is_sc = i < 2
            mini_v3_rows.append({
                "base_id": bid,
                "family_id": bid,
                "unit_id": bid,
                "pair_type": "argument_flip" if i < 6 else "context_flip",
                "fallacy_class": "missing_conjunct" if is_sc else "scope_swap",
                "broken_step": (i % 3) + 1,
                "initial_answer": "A",
                "target_answer": "B",
            })
        mini_v3_file = td_p / "mini_v3.jsonl"
        mini_v3_file.write_text("\n".join(json.dumps(r) for r in mini_v3_rows) + "\n", encoding="utf-8")

        mini_eb_rows = []
        for cell in ("1v2_orig", "2v1_anti", "1v1_same", "2v2_same", "1v1_ctx", "2v2_ctx"):
            bid = f"MINI-EB-{cell}"
            mini_eb_rows.append({
                "base_id": bid, "family_id": bid, "unit_id": bid,
                "entity_balance_cell": cell, "initial_answer": "A", "target_answer": "B",
            })
        mini_eb_file = td_p / "mini_eb.jsonl"
        mini_eb_file.write_text("\n".join(json.dumps(r) for r in mini_eb_rows) + "\n", encoding="utf-8")

        arms_15 = [
            "turn1_V", "turn1_F", "V", "F", "V_auth", "V_hes", "F_auth", "F_hes",
            "F_src_third_party", "S", "shell_only", "AO", "N", "P0_confirm_a0", "I0_confirm_a0",
        ]
        arms_th = ["turn1_V", "turn1_F", "V_auth", "V_hes", "F_auth", "F_hes"]

        def _write_fake_arm(arm_dir: Path, *, t1_ok_n: int, fc_ok_n: int, wiki_kl: float, alias: str) -> None:
            (arm_dir / "no_think").mkdir(parents=True, exist_ok=True)
            (arm_dir / "think").mkdir(parents=True, exist_ok=True)
            (arm_dir / "capabilities").mkdir(parents=True, exist_ok=True)
            (arm_dir / "exfever").mkdir(parents=True, exist_ok=True)

            nt_recs = []
            for idx_b, r in enumerate(mini_v3_rows):
                t1_pass = idx_b < t1_ok_n
                fc_pass = idx_b < fc_ok_n
                for a in arms_15:
                    if a in ("turn1_V", "turn1_F"):
                        is_i = t1_pass
                        is_t = not is_i
                    elif a in ("V", "V_auth", "V_hes"):
                        is_t = fc_pass
                        is_i = not is_t
                    elif a in ("F", "F_auth", "F_hes"):
                        is_i = fc_pass
                        is_t = not is_i
                    else:
                        is_i = True
                        is_t = False
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
            (arm_dir / "vllm_lora_parity.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            (arm_dir / "exfever" / f"exfever_no_think_summary_{alias}.json").write_text(
                json.dumps({"metrics": {"headline": {"r_uy": 9.8, "r_ru_evidence": 68.8, "selectivity_ru_minus_uy": 59.0}}}),
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
                "ifeval_m0": 461,
                "gsm8k_m0": 638,
                "arms": {
                    "u2_corridor": {"42": str(td_p / "u2_s42"), "1024": str(td_p / "u2_s1024")},
                    "M3-v4-PureMid-MLP+": {"42": str(td_p / "v4_s42"), "1024": str(td_p / "v4_s1024")},
                },
                "pair_specs": [["v4_vs_u2", "M3-v4-PureMid-MLP+", "u2_corridor"]],
            },
            "fixed_step200": {
                "v3_file": str(mini_v3_file),
                "entries": [{"name": "v4_step200", "seed": 42, "dir": str(td_p / "v4_s42")}],
            },
            "entity_balanced": {
                "file": str(mini_eb_file),
                "entries": [{"name": "v4", "records": str(eb_rec_file)}],
            },
            "external_8b": {
                "v3_file": str(mini_v3_file),
                "entries": [{"name": "v4", "dir": str(td_p / "v4_s42")}],
            },
        }
        out_doc, out_md = run_stage6_aggregation(smoke_cfg, td_p / "out", n_boot=200)
        v4_row = out_doc["sections"]["scale_14b"]["arms"]["M3-v4-PureMid-MLP+@s42"]
        # 9/12 Turn-1 dual correct = 75.0%, 9/12 Joint = 75.0%, 10/12 4-Cell = 83.33% -> NEVER silent 0.0!
        assert abs(v4_row["turn1_dual_pct"] - 75.0) < 1e-6, v4_row
        assert abs(v4_row["joint_pct"] - 75.0) < 1e-6, v4_row
        assert abs(v4_row["four_cell_pct"] - 83.33) <= 0.02, v4_row
        assert abs(out_doc["sections"]["fixed_step200"]["v4_step200@s42"]["joint_pct"] - 75.0) < 1e-6
        assert "pooled_2seed" in out_md and "75.00%" in out_md

    print(
        "[OK] c35_stage6_stats self-test PASSED: fail-closed validator (5/5 caught), "
        "p<0.0002 formatting, exact v3 900/802 split (390/390/120), OOD 50/50 weighting, "
        "6-cell entity-balanced audit, AND real summarize_v2() 4-section end-to-end smoke test!"
    )


def run_stage6_aggregation(cfg: dict[str, Any], out_dir: Path, n_boot: int = 5000) -> tuple[dict[str, Any], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Full Stage 6 aggregation:
    # 1. Run strict Stage 5 per-scale core stats + validate every no_think & think record strictly!
    # 2. Compute v3 900-unit + 802-unit non-shortcut sensitivity + pair-type breakdowns!
    # 3. Aggregate 14B fixed step-200 check, entity_balanced_v1 6-cell diagnostic, and 8B external SOTA + loss ablations!
    out_doc: dict[str, Any] = {
        "schema": "cedar.acl_exp_c35.stage6_stats.v1",
        "n_boot": n_boot,
        "sections": {},
    }
    md: list[str] = ["# ACL-exp-c3.5 Stage 6 Strict Closure Readout\n"]

    # Section A: 14B Dual-Seed (42 + 1024) on v3 (900 + 802 sensitivity) & v2
    if "scale_14b" in cfg:
        s14 = cfg["scale_14b"]
        v3_file = Path(s14["v3_file"])
        v3_meta = _load_base_meta(v3_file)
        all_u, non_sc_u, by_pt = compute_v3_non_shortcut_units(v3_file)
        exp_b = set(v3_meta.keys())
        exp_arms = set(s14.get("expected_nothink_arms", [
            "turn1_V", "turn1_F", "V", "F", "V_auth", "V_hes", "F_auth", "F_hes",
            "F_src_third_party", "S", "shell_only", "AO", "N", "P0_confirm_a0", "I0_confirm_a0",
        ]))
        bms_14: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        rows_14: dict[str, Any] = {}
        md.append("## 1. Qwen3-14B Dual-Seed (`seed=42, 1024`) on `c3_blind_confirm_v3` (`900` Units + `802` Non-Shortcut)\n")
        md.append("| Model | Seed | 4-Cell (900) | Joint (900) | Turn-1 Dual | Wiki KL | IFEval / GSM8K | Health |")
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
                # Also strictly validate non-empty think records if present
                th_p = d / "think" / "records_think_c3_blind_think_v3.jsonl"
                if th_p.is_file():
                    th_recs = C3S._load_jsonl(th_p)
                    th_bases = {str(r["base_id"]) for r in th_recs}
                    validate_records_strict(th_recs, expected_bases=th_bases, expected_arms={"turn1_V", "turn1_F", "V_auth", "V_hes", "F_auth", "F_hes"}, regime="think", label=f"14b_think:{mname}@s{sd}")
                bms_14[mname][sd] = bm
                s = info["all"]
                fc = s["four_cell_family_weighted"] * 100.0
                jt = s["joint_t1_4cell_unit_weighted"] * 100.0
                t1 = s["turn1_dual_acc"] * 100.0
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
                md.append(f"| `{mname}` | **pooled_2seed** | **{np.mean(seed_4c):.2f}%** | **{np.mean(seed_jt):.2f}%** | {np.mean(seed_t1):.2f}% | `{np.mean(seed_kl):.6f}` | — | — |")

        md.append("\n### 14B Paired Bootstrap (All `900` Units + Post-Hoc `802` Non-Shortcut Sensitivity)\n")
        md.append("| Comparison | Split | Subset | 4-Cell Diff [95% CI] | Joint Diff [95% CI] | p |")
        md.append("|---|---|---|---:|---:|---:|")
        cmp_14: dict[str, Any] = {}
        for pname, ma, mb in s14["pair_specs"]:
            for sd in sorted(set(bms_14[ma]) & set(bms_14[mb]), key=int):
                for sub_name, sub_u in (("all_900", all_u), ("non_shortcut_802", non_sc_u)):
                    r4 = paired_unit_bootstrap_subset(bms_14[ma][sd], bms_14[mb][sd], sub_u, "four_cell", n_boot=n_boot)
                    rj = paired_unit_bootstrap_subset(bms_14[ma][sd], bms_14[mb][sd], sub_u, "joint_t1_4cell", n_boot=n_boot)
                    cmp_14[f"{pname}|s{sd}|{sub_name}"] = {"four_cell": r4, "joint": rj}
                    md.append(f"| `{pname}` | `seed_{sd}` | `{sub_name}` | **{r4['diff_pp']:+.2f}** `[{r4['ci95_pp'][0]:+.2f}, {r4['ci95_pp'][1]:+.2f}]` | **{rj['diff_pp']:+.2f}** `[{rj['ci95_pp'][0]:+.2f}, {rj['ci95_pp'][1]:+.2f}]` | `{r4['p_str']}` |")
            if len(set(bms_14[ma]) & set(bms_14[mb])) >= 2:
                for sub_name, sub_u in (("all_900", all_u), ("non_shortcut_802", non_sc_u)):
                    r4 = pooled_paired_bootstrap_subset(bms_14[ma], bms_14[mb], sub_u, "four_cell", n_boot=n_boot)
                    rj = pooled_paired_bootstrap_subset(bms_14[ma], bms_14[mb], sub_u, "joint_t1_4cell", n_boot=n_boot)
                    cmp_14[f"{pname}|pooled_2seed|{sub_name}"] = {"four_cell": r4, "joint": rj}
                    md.append(f"| `{pname}` | **`pooled_2seed`** | `{sub_name}` | **{r4['diff_pp']:+.2f}** `[{r4['ci95_pp'][0]:+.2f}, {r4['ci95_pp'][1]:+.2f}]` | **{rj['diff_pp']:+.2f}** `[{rj['ci95_pp'][0]:+.2f}, {rj['ci95_pp'][1]:+.2f}]` | `{r4['p_str']}` |")
        out_doc["sections"]["scale_14b"] = {"arms": rows_14, "comparisons": cmp_14}

    # Section B: Fixed Step-200 Matched-Step Confound Check (14B)
    if "fixed_step200" in cfg:
        md.append("\n## 2. Qwen3-14B Fixed `step=200` Matched-Step Confound Audit (`v3` & WikiText-2 KL)\n")
        md.append("| Model (at exact `step 200`) | Seed | 4-Cell (900) | Joint (900) | Wiki KL (`step 200`) |")
        md.append("|---|---:|---:|---:|---:|")
        s200_out: dict[str, Any] = {}
        v3_meta = _load_base_meta(Path(cfg["fixed_step200"]["v3_file"]))
        for entry in cfg["fixed_step200"]["entries"]:
            d = Path(entry["dir"])
            art = C3S._load_arm_artifacts(d, "c3_blind_confirm_v3")
            bm = S35.compute_base_metrics_v2(art["nt_recs"], base_meta=v3_meta)
            s = S35.summarize_v2(bm)
            fc = s["four_cell_family_weighted"] * 100.0
            jt = s["joint_t1_4cell_unit_weighted"] * 100.0
            kl = art.get("wiki_kl")
            md.append(f"| `{entry['name']}` | {entry['seed']} | **{fc:.2f}%** | **{jt:.2f}%** | `{kl:.6f}` |")
            s200_out[f"{entry['name']}@s{entry['seed']}"] = {"four_cell_pct": round(fc, 2), "joint_pct": round(jt, 2), "wiki_kl": kl}
        out_doc["sections"]["fixed_step200"] = s200_out

    # Section C: Entity-Count-Balanced 6-Cell Diagnostic (`c3_entity_balanced_v1`, 480 units)
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

    # Section D: 8B External SOTA & Internal Loss-Ablation Table (`v3` 900 + 802 + KL + EX-FEVER)
    if "external_8b" in cfg:
        e8 = cfg["external_8b"]
        v3_file = Path(e8["v3_file"])
        v3_meta = _load_base_meta(v3_file)
        all_u, non_sc_u, _ = compute_v3_non_shortcut_units(v3_file)
        md.append("\n## 4. Qwen3-8B External SOTA & Internal Loss Ablations (`v3`, KL, Capabilities, EX-FEVER)\n")
        md.append("| Method | `v3` 4C (900) | `v3` 4C (802) | `v3` Joint | Wiki KL | IFEval / GSM8K | EX-FEVER `R_UY` / `R_RU_ev` / `Sel` |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        ext_out: dict[str, Any] = {}
        for entry in e8["entries"]:
            d = Path(entry["dir"])
            art = C3S._load_arm_artifacts(d, "c3_blind_confirm_v3")
            bm = S35.compute_base_metrics_v2(art["nt_recs"], base_meta=v3_meta)
            fa_all = C3S.family_weighted_scores(bm, "four_cell")
            fj_all = C3S.family_weighted_scores(bm, "joint_t1_4cell")
            fc900 = float(np.mean([fa_all[u] for u in all_u])) * 100.0
            fc802 = float(np.mean([fa_all[u] for u in non_sc_u])) * 100.0
            jt900 = float(np.mean([fj_all[u] for u in all_u])) * 100.0
            kl = art.get("wiki_kl")
            ife, gsm = art.get("ifeval"), art.get("gsm8k")
            ex = S5._load_exfever_summary(d, entry["name"], regime="no_think")
            if ex:
                hl = (ex.get("metrics") or ex.get("summary") or {}).get("headline", {})
                ex_str = f"{hl.get('r_uy', '-')}% / {hl.get('r_ru_evidence', '-')}% / {hl.get('selectivity_ru_minus_uy', '-')}"
            else:
                ex_str = "—"
            kl_s = f"`{kl:.6f}`" if kl is not None else "—"
            cap_s = f"{ife} / {gsm}" if (ife is not None and gsm is not None) else "—"
            md.append(f"| `{entry['name']}` | **{fc900:.2f}%** | **{fc802:.2f}%** | **{jt900:.2f}%** | {kl_s} | {cap_s} | {ex_str} |")
            ext_out[entry["name"]] = {"fc900": round(fc900, 2), "fc802": round(fc802, 2), "jt900": round(jt900, 2), "wiki_kl": kl, "ifeval": ife, "gsm8k": gsm, "exfever": ex_str}
        out_doc["sections"]["external_8b"] = ext_out

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
