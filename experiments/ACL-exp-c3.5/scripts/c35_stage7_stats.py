#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 7 One-Shot Sealed-Test Aggregator & Pre-Unseal Open Dry-Run Validator.

Implements EVERY statistical, provenance, denominator, and fail-closed rule locked in
`experiments/ACL-exp-c3.5/PRE_REGISTRATION_SEALED_TEST.md`:
  1. STRICT PROVENANCE LOCK (`verify_arm_provenance`):
       - Verifies recorded `adapter_state_sha256`, `peft_adapter_model_sha256`, `merged_shards_sha256`,
         and `frozen_vector_sha256` against the signed Stage 7 config manifest (fail-closed on any mismatch).
  2. FAIL-CLOSED RECORD VALIDATION (`S6.validate_records_strict`):
       - Independent base-ID lists loaded from the dataset `.jsonl` (NEVER extracted from prediction files).
       - Rejects empty files, missing/extra keys, duplicate keys, invalid `pred` (`"Z"`), contradictory
         `is_target`/`is_initial`, non-bool `parse_ok`, and non-finite `u_theta` (`NaN`/`inf`).
  3. STRICT PER-(SCALE, REGIME, DATASET) M0 DUAL-ELIGIBILITY (`S6.compute_m0_dual_eligible_strict`):
       - Rejects partial `turn1_F`-only records; locks `M0` dual eligibility independently for every
         `(scale, regime, dataset)`.
  4. EXACT UNROUNDED OOD 50/50 EQUAL-SLICE SCORING (`S6.score_ood_equal_slice_50_50`):
       - Hard-asserts exact `60 ood_converse + 60 ood_depth4` (zero unknown/extra IDs) on full `c1_test_ood`.
       - Computes `0.5 * mean(ood_converse) + 0.5 * mean(ood_depth4)` from unrounded slice means FIRST
         (rounding only at final display: `0/60` & `10/60` -> `8.33%`, NEVER `8.34%`).
       - Also reports `ood_converse (n=60)`, `ood_depth4 (n=60)` + broken-step `1/2/3`, `argument_flip` vs
         `context_flip`, `shared_m0_eligible` (with explicit per-slice counts and empty-slice warning),
         `parse_fail`, `truncation`, `NLL (4-tone)`, and `|u| median (4-tone)`.
  5. PRE-REGISTERED CONFIRMATORY PRIMARY COMPARISONS + HOLM-BONFERRONI MULTIPLICITY:
       - Primary family (14B `pooled_2seed` on `c1_test_ood` 50/50 & `c1_test_id`):
           (1) `M3-v4-PureMid-MLP+` vs `u2_corridor`
           (2) `M3-v4-PureMid-MLP+` vs `M3-v4-PureMid-MLP+-Random-Mask`
       - Reports paired 95% bootstrap CIs (`n_boot=5000`), exact `0/5000` tail hits (min nonzero two-sided
         resolution `0.0004`), exact two-sided paired sign-flip p-values, and Holm-Bonferroni adjusted p-values.
  6. COMPLETE OPEN-DATA END-TO-END DRY-RUN (`--dry-run-open`):
       - Runs on REAL open `c1_dev_bases.jsonl` + synthetic `60+60` OOD with 7 adversarial fault injections!
       - NEVER opens or touches sealed `c1_test_id_bases.jsonl` or `c1_test_ood_bases.jsonl`.

Usage:
  python experiments/ACL-exp-c3.5/scripts/c35_stage7_stats.py --dry-run-open
  python experiments/ACL-exp-c3.5/scripts/c35_stage7_stats.py --config-json <manifest.json> --out-dir <out_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))

import c3_stats as C3S  # noqa: E402
import c35_eval as EV35  # noqa: E402
import c35_stage6_stats as S6  # noqa: E402
import c35_stats as S35  # noqa: E402

NOTHINK_ARMS_15 = {
    "turn1_V", "turn1_F", "V", "F", "V_auth", "V_hes", "F_auth", "F_hes",
    "F_src_third_party", "S", "shell_only", "AO", "N", "P0_confirm_a0", "I0_confirm_a0",
}
THINK_ARMS_6 = {"turn1_V", "turn1_F", "V_auth", "V_hes", "F_auth", "F_hes"}
TONE_ARMS_4 = {"V_auth", "V_hes", "F_auth", "F_hes"}


def _load_meta(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {str(r["base_id"]): r for r in rows}


def verify_arm_provenance(
    arm_dir: Path,
    *,
    regime: str,
    spec: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Fail-closed check that recorded eval provenance matches the signed manifest SHA-256 hashes."""
    sum_p = arm_dir / regime / f"eval_summary_{regime}.json"
    if not sum_p.is_file():
        raise FileNotFoundError(f"[MISSING SUMMARY] {label}: {sum_p}")
    doc = json.loads(sum_p.read_text(encoding="utf-8"))
    prov = doc.get("provenance")
    if not isinstance(prov, dict):
        raise AssertionError(f"[MISSING PROVENANCE] {label}: {sum_p} lacks 'provenance' block!")

    # Verify adapter_state_sha256 (HF no_think adapters)
    exp_adapter = spec.get("expected_adapter_sha256")
    if regime == "no_think" and exp_adapter:
        act = prov.get("adapter_state_sha256")
        if str(act or "").lower() != str(exp_adapter).lower():
            raise AssertionError(f"[ADAPTER SHA MISMATCH] {label}: actual={act} != expected={exp_adapter}")

    # Verify peft_adapter_model_sha256 (vLLM native LoRA think)
    exp_peft = spec.get("expected_peft_sha256")
    if regime == "think" and exp_peft:
        act = prov.get("peft_adapter_model_sha256")
        if str(act or "").lower() != str(exp_peft).lower():
            raise AssertionError(f"[PEFT SHA MISMATCH] {label}: actual={act} != expected={exp_peft}")

    # Verify BF16 merged shards (Pinpoint-DLA-Corridor think)
    exp_shards = spec.get("expected_merged_shards_sha256")
    if regime == "think" and exp_shards:
        act_shards = prov.get("merged_shards_sha256")
        if not isinstance(act_shards, dict) or {k: str(v).lower() for k, v in act_shards.items()} != {
            k: str(v).lower() for k, v in exp_shards.items()
        }:
            raise AssertionError(f"[MERGED SHARDS MISMATCH] {label}: actual={act_shards} != expected={exp_shards}")

    # Verify CAA frozen vector SHA-256
    exp_vec = spec.get("expected_vector_sha256")
    if regime == "no_think" and exp_vec:
        act = prov.get("frozen_vector_sha256")
        if str(act or "").lower() != str(exp_vec).lower():
            raise AssertionError(f"[CAA VECTOR SHA MISMATCH] {label}: actual={act} != expected={exp_vec}")

    return prov


def _count_truncations(recs: list[dict[str, Any]]) -> int:
    n = 0
    for r in recs:
        if r.get("truncated") is True or r.get("finish_reason") == "length":
            n += 1
    return n


def _tone4_calibration(recs: list[dict[str, Any]], subset_bases: set[str] | None = None) -> tuple[float | None, float | None]:
    nlls, abs_us = [], []
    for r in recs:
        if r.get("arm") not in TONE_ARMS_4:
            continue
        if subset_bases is not None and str(r["base_id"]) not in subset_bases:
            continue
        if r.get("gold_nll") is not None and math.isfinite(float(r["gold_nll"])):
            nlls.append(float(r["gold_nll"]))
        if r.get("u_theta") is not None and math.isfinite(float(r["u_theta"])):
            abs_us.append(abs(float(r["u_theta"])))
    return (round(float(np.mean(nlls)), 4) if nlls else None), (round(float(np.median(abs_us)), 4) if abs_us else None)


def _score_id_subset(
    bm: dict[str, dict[str, Any]],
    recs: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    base_ids: set[str],
) -> dict[str, Any]:
    if not base_ids:
        return {"n_items": 0, "four_cell_pct": None, "joint_pct": None, "turn1_dual_pct": None}
    sub_bm = {b: bm[b] for b in sorted(base_ids)}
    fc_raw = float(np.mean([float(v["four_cell"]) for v in sub_bm.values()])) * 100.0
    jt_raw = float(np.mean([float(v["joint_t1_4cell"]) for v in sub_bm.values()])) * 100.0
    t1_raw = float(np.mean([float(v["turn1_ok"]) for v in sub_bm.values()])) * 100.0
    by_pt = {}
    for pt in ("argument_flip", "context_flip"):
        pt_ids = [b for b in sub_bm if meta[b].get("pair_type") == pt]
        by_pt[pt] = {
            "n_items": len(pt_ids),
            "four_cell_pct": round(float(np.mean([float(sub_bm[b]["four_cell"]) for b in pt_ids])) * 100.0, 2) if pt_ids else None,
        }
    nll4, u4 = _tone4_calibration(recs, base_ids)
    sub_recs = [r for r in recs if str(r["base_id"]) in base_ids]
    return {
        "n_items": len(base_ids),
        "four_cell_pct": round(fc_raw, 2),
        "joint_pct": round(jt_raw, 2),
        "turn1_dual_pct": round(t1_raw, 2),
        "nll_4tone": nll4,
        "abs_u_median_4tone": u4,
        "parse_fail_count": sum(1 for r in sub_recs if not r["parse_ok"]),
        "truncation_count": _count_truncations(sub_recs),
        "by_pair_type": by_pt,
    }


def _average_base_maps_by_seed(bms_by_seed: dict[str, dict[str, dict[str, Any]]], base_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Average per-base metrics across training seeds (unrounded float) for exact pooled paired inference."""
    seeds = sorted(bms_by_seed.keys())
    assert seeds
    out: dict[str, dict[str, Any]] = {}
    for b in sorted(base_ids):
        out[b] = {
            "base_id": b,
            "family_id": b,  # item-level for c1_test_id & stratified c1_test_ood
            "four_cell": float(np.mean([float(bms_by_seed[s][b]["four_cell"]) for s in seeds])),
            "joint_t1_4cell": float(np.mean([float(bms_by_seed[s][b]["joint_t1_4cell"]) for s in seeds])),
            "turn1_ok": float(np.mean([float(bms_by_seed[s][b]["turn1_ok"]) for s in seeds])),
        }
    return out


def _paired_item_bootstrap(
    bm_a: dict[str, dict[str, Any]],
    bm_b: dict[str, dict[str, Any]],
    base_ids: set[str],
    metric: str,
    n_boot: int = 5000,
    seed: int = 2027,
) -> dict[str, Any]:
    ids = sorted(base_ids)
    if not ids:
        return {"n_items": 0, "diff_pp": None, "ci95_pp": None, "p_str": "N/A", "p_signflip": None}
    va = np.array([float(bm_a[b][metric]) for b in ids], dtype=np.float64)
    vb = np.array([float(bm_b[b][metric]) for b in ids], dtype=np.float64)
    d = va - vb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    bd = d[idx].mean(axis=1)
    p_two = min(1.0, 2.0 * min(float(np.mean(bd <= 0.0)), float(np.mean(bd >= 0.0))))
    return {
        "n_items": len(ids),
        "mean_a_pct": round(float(va.mean()) * 100.0, 2),
        "mean_b_pct": round(float(vb.mean()) * 100.0, 2),
        "diff_pp": round(float(d.mean()) * 100.0, 2),
        "ci95_pp": [round(float(np.quantile(bd, 0.025)) * 100.0, 2), round(float(np.quantile(bd, 0.975)) * 100.0, 2)],
        "p_two_sided_raw": round(p_two, 6),
        "p_str": S6.format_boot_p(p_two, n_boot),
        "p_signflip": round(S6.exact_two_sided_signflip_p(d), 6),
    }


def holm_bonferroni(p_map: dict[str, tuple[float, bool]]) -> dict[str, dict[str, Any]]:
    """Step-down Holm-Bonferroni adjusted p-values (preserving conservative upper-bound flag when 0/B tail hits)."""
    items = sorted(p_map.items(), key=lambda kv: kv[1][0])
    m = len(items)
    adj: dict[str, dict[str, Any]] = {}
    running_max = 0.0
    running_upper = False
    for rank, (k, (p_val, is_upper)) in enumerate(items):
        val = min(1.0, (m - rank) * float(p_val))
        if val > running_max + 1e-12:
            running_max = val
            running_upper = is_upper
        elif abs(val - running_max) <= 1e-12:
            running_upper = running_upper and is_upper
        p_fmt = f"<= {running_max:.4f}" if running_upper else f"{running_max:.4f}"
        adj[k] = {"p_adj": round(running_max, 6), "is_upper_bound": running_upper, "p_adj_str": p_fmt}
    return adj


def run_stage7_aggregation(
    cfg: dict[str, Any],
    out_dir: Path,
    *,
    n_boot: int = 5000,
    allow_reaggregate: bool = False,
) -> tuple[dict[str, Any], str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "c35_stage7_stats.json"
    out_md = out_dir / "C35_STAGE7_SEALED_READOUT.md"
    if not allow_reaggregate and (out_json.exists() or out_md.exists()):
        raise FileExistsError(
            f"[NO-OVERWRITE] Stage 7 output already exists in {out_dir}; pass --allow-reaggregate to regenerate summary only."
        )

    id_name = cfg.get("id_dataset_name", "c1_test_id")
    ood_name = cfg.get("ood_dataset_name", "c1_test_ood")
    id_meta = _load_meta(Path(cfg["id_bases_file"]))
    ood_meta = _load_meta(Path(cfg["ood_bases_file"]))
    id_bases = set(id_meta.keys())
    ood_bases = set(ood_meta.keys())
    ood_slice_map = {bid: str(m["split"]) for bid, m in ood_meta.items()}
    require_exact_60_60 = bool(cfg.get("require_exact_60_60", True))

    # Pre-validate OOD metadata structure independently before reading any model predictions
    unknown_sl = set(ood_slice_map.values()) - set(S6.OOD_VALID_SLICES)
    assert not unknown_sl, f"Invalid OOD slice labels in metadata: {sorted(unknown_sl)}"
    if require_exact_60_60:
        assert sum(1 for v in ood_slice_map.values() if v == "ood_converse") == 60
        assert sum(1 for v in ood_slice_map.values() if v == "ood_depth4") == 60

    out_doc: dict[str, Any] = {
        "schema": "cedar.acl_exp_c35.stage7_sealed_stats.v1",
        "n_boot": n_boot,
        "id_dataset": id_name,
        "ood_dataset": ood_name,
        "boot_p_note": f"Two-sided paired bootstrap (`n_boot={n_boot}`); 0 tail hits shown as `0/{n_boot}` (min nonzero resolution `{2.0/n_boot:.4f}`); exact sign-flip p-values & Holm-Bonferroni adjusted primary p-values logged.",
        "scales": {},
    }
    md: list[str] = [
        "# ACL-exp-c3.5 Stage 7 Sealed-Test Confirmatory Readout\n",
        f"- **ID Dataset**: `{id_name}` (`n={len(id_bases)}`) | **OOD Dataset**: `{ood_name}` (`n={len(ood_bases)}`, exact 50/50 unrounded weighting `ood_converse` + `ood_depth4`)",
        f"- **Bootstrap**: `n_boot={n_boot}` (item sampling uncertainty given fixed seeds; 0 tail hits -> `0/{n_boot}`, min nonzero two-sided resolution `{2.0/n_boot:.4f}`).\n",
    ]

    for scale_key, sc_cfg in cfg["scales"].items():
        scale_doc: dict[str, Any] = {"m0_eligible": {}, "arms": {}, "comparisons": {}}
        md.append(f"\n## Scale: `{scale_key}`\n")

        # 1. Load & validate M0 strictly per regime and dataset -> lock shared_m0_eligible sets!
        m0_spec = sc_cfg["m0"]
        m0_dir = Path(m0_spec["dir"])
        m0_elig: dict[str, dict[str, set[str]]] = {"no_think": {}, "think": {}}
        for regime in sc_cfg.get("m0_regimes", ["no_think", "think"]):
            exp_arms = NOTHINK_ARMS_15 if regime == "no_think" else THINK_ARMS_6
            verify_arm_provenance(m0_dir, regime=regime, spec=m0_spec, label=f"{scale_key}:M0:{regime}")
            for ds_n, exp_b in ((id_name, id_bases), (ood_name, ood_bases)):
                rec_p = m0_dir / regime / f"records_{regime}_{ds_n}.jsonl"
                recs = C3S._load_jsonl(rec_p)
                elig_list = S6.compute_m0_dual_eligible_strict(
                    recs, expected_bases=exp_b, expected_arms=exp_arms,
                    regime=regime, label=f"{scale_key}:M0:{regime}:{ds_n}",
                )
                m0_elig[regime][ds_n] = set(elig_list)
                scale_doc["m0_eligible"][f"{regime}:{ds_n}"] = {
                    "n_eligible": len(elig_list),
                    "n_total": len(exp_b),
                }
                if ds_n == ood_name:
                    scale_doc["m0_eligible"][f"{regime}:{ds_n}"]["n_converse"] = sum(
                        1 for b in elig_list if ood_slice_map[b] == "ood_converse"
                    )
                    scale_doc["m0_eligible"][f"{regime}:{ds_n}"]["n_depth4"] = sum(
                        1 for b in elig_list if ood_slice_map[b] == "ood_depth4"
                    )

        # 2. Load & strictly validate EVERY arm in this scale across its pre-registered regimes!
        loaded_bms: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        # loaded_bms[regime][ds_name][model_group][seed] -> base_map

        for regime in ("no_think", "think"):
            exp_arms = NOTHINK_ARMS_15 if regime == "no_think" else THINK_ARMS_6
            md.append(f"\n### `{scale_key}` — Regime: `{regime}`\n")
            id_m0_n = len(m0_elig[regime].get(id_name, set()))
            ood_m0_set = m0_elig[regime].get(ood_name, set())
            ood_m0_c = sum(1 for b in ood_m0_set if ood_slice_map[b] == "ood_converse")
            ood_m0_d = sum(1 for b in ood_m0_set if ood_slice_map[b] == "ood_depth4")
            md.append(
                f"*Shared M0 Dual-Eligible Denominators (`{scale_key}`, `{regime}`): "
                f"`{id_name}` = `{id_m0_n}/{len(id_bases)}`; "
                f"`{ood_name}` = `{len(ood_m0_set)}/{len(ood_bases)}` (`converse={ood_m0_c}`, `depth4={ood_m0_d}`).*\n"
            )
            md.append(
                "| Model | Seed | ID 4C (`all`) | ID Joint | ID 4C (`M0-elig`) | OOD 50/50 (`all`) | `converse` (60) | `depth4` (60) | OOD 50/50 (`M0-elig`) | Turn-1 (ID/OOD) | PF / Trunc |"
            )
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

            for arm_entry in sc_cfg["arms"]:
                m_group = arm_entry["model_group"]
                sd = str(arm_entry["seed"])
                regimes_allowed = arm_entry.get("regimes", ["no_think", "think"])
                if regime not in regimes_allowed:
                    continue
                arm_dir = Path(arm_entry["dir"])
                label = f"{scale_key}:{m_group}@s{sd}:{regime}"
                prov = verify_arm_provenance(arm_dir, regime=regime, spec=arm_entry, label=label)

                recs_id = C3S._load_jsonl(arm_dir / regime / f"records_{regime}_{id_name}.jsonl")
                recs_ood = C3S._load_jsonl(arm_dir / regime / f"records_{regime}_{ood_name}.jsonl")
                S6.validate_records_strict(recs_id, expected_bases=id_bases, expected_arms=exp_arms, regime=regime, label=f"{label}:{id_name}")
                S6.validate_records_strict(recs_ood, expected_bases=ood_bases, expected_arms=exp_arms, regime=regime, label=f"{label}:{ood_name}")

                bm_id = S35.compute_base_metrics_v2(recs_id, base_meta=id_meta)
                bm_ood = S35.compute_base_metrics_v2(recs_ood, base_meta=ood_meta)
                loaded_bms[regime][id_name][m_group][sd] = bm_id
                loaded_bms[regime][ood_name][m_group][sd] = bm_ood

                id_all = _score_id_subset(bm_id, recs_id, id_meta, id_bases)
                id_elig = _score_id_subset(bm_id, recs_id, id_meta, m0_elig[regime][id_name])
                ood_all_4c = S6.score_ood_equal_slice_50_50(
                    bm_ood, None, ood_slice_map, metric="four_cell", n_boot=n_boot,
                    expected_bases=ood_bases, require_exact_60_60=require_exact_60_60,
                )
                ood_all_jt = S6.score_ood_equal_slice_50_50(
                    bm_ood, None, ood_slice_map, metric="joint_t1_4cell", n_boot=n_boot,
                    expected_bases=ood_bases, require_exact_60_60=require_exact_60_60,
                )
                ood_elig_map = {b: ood_slice_map[b] for b in m0_elig[regime][ood_name]}
                ood_elig_4c = S6.score_ood_equal_slice_50_50(
                    bm_ood, None, ood_elig_map, metric="four_cell", n_boot=n_boot,
                    expected_bases=set(ood_elig_map.keys()), require_exact_60_60=False,
                )
                ood_t1 = round(float(np.mean([float(bm_ood[b]["turn1_ok"]) for b in ood_bases])) * 100.0, 2)
                ood_d4_steps = {}
                for st in (1, 2, 3):
                    s_ids = [b for b in ood_bases if ood_meta[b].get("split") == "ood_depth4" and ood_meta[b].get("broken_step_index") == st]
                    if s_ids:
                        ood_d4_steps[f"step_{st}"] = round(float(np.mean([float(bm_ood[b]["four_cell"]) for b in s_ids])) * 100.0, 2)
                ood_by_pt = {
                    pt: round(float(np.mean([float(bm_ood[b]["four_cell"]) for b in ood_bases if ood_meta[b].get("pair_type") == pt])) * 100.0, 2)
                    for pt in ("argument_flip", "context_flip")
                    if any(ood_meta[b].get("pair_type") == pt for b in ood_bases)
                }
                pf_tot = id_all["parse_fail_count"] + sum(1 for r in recs_ood if not r["parse_ok"])
                tr_tot = id_all["truncation_count"] + _count_truncations(recs_ood)

                scale_doc["arms"][f"{m_group}@s{sd}:{regime}"] = {
                    "provenance": prov,
                    "id_all": id_all,
                    "id_m0_elig": id_elig,
                    "ood_all_4cell": ood_all_4c,
                    "ood_all_joint": ood_all_jt,
                    "ood_m0_elig_4cell": ood_elig_4c,
                    "ood_turn1_dual_pct": ood_t1,
                    "ood_depth4_by_broken_step": ood_d4_steps,
                    "ood_by_pair_type": ood_by_pt,
                    "total_parse_fail": pf_tot,
                    "total_truncations": tr_tot,
                }
                id_e_s = f"{id_elig['four_cell_pct']:.2f}%" if id_elig["four_cell_pct"] is not None else "N/A"
                ood_e_s = f"{ood_elig_4c['equal_50_50_score_pct']:.2f}%" if ood_elig_4c["equal_50_50_score_pct"] is not None else "N/A"
                md.append(
                    f"| `{m_group}` | {sd} | **{id_all['four_cell_pct']:.2f}%** | {id_all['joint_pct']:.2f}% | {id_e_s} | "
                    f"**{ood_all_4c['equal_50_50_score_pct']:.2f}%** | {ood_all_4c['per_slice']['ood_converse']['raw_score_pct']:.2f}% | "
                    f"{ood_all_4c['per_slice']['ood_depth4']['raw_score_pct']:.2f}% | {ood_e_s} | "
                    f"{id_all['turn1_dual_pct']:.1f}% / {ood_t1:.1f}% | {pf_tot} / {tr_tot} |"
                )

            # 3. Pre-registered paired comparisons (primary + secondary) for this scale & regime!
            comp_specs = sc_cfg.get("comparisons", [])
            if comp_specs:
                md.append(f"\n#### `{scale_key}` (`{regime}`) Paired Comparisons (ID & OOD 50/50 Equal-Slice)\n")
                md.append("| Tier | Comparison | Split | ID 4C Diff [95% CI] | OOD 50/50 4C Diff [95% CI] | OOD 50/50 Joint Diff | `converse` / `depth4` | Boot `p` (OOD 4C) | Holm `p` |")
                md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")

                raw_primary_p_ood: dict[str, tuple[float, bool]] = {}
                comp_rows: list[tuple[str, str, str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
                for cspec in comp_specs:
                    cname, tier, ma, mb = cspec["name"], cspec.get("tier", "secondary"), cspec["model_a"], cspec["model_b"]
                    common_seeds = sorted(set(loaded_bms[regime][ood_name].get(ma, {})) & set(loaded_bms[regime][ood_name].get(mb, {})), key=lambda x: int(x) if x.isdigit() else 0)
                    if not common_seeds:
                        continue
                    for sd in common_seeds:
                        r_id = _paired_item_bootstrap(loaded_bms[regime][id_name][ma][sd], loaded_bms[regime][id_name][mb][sd], id_bases, "four_cell", n_boot=n_boot)
                        r_id_jt = _paired_item_bootstrap(loaded_bms[regime][id_name][ma][sd], loaded_bms[regime][id_name][mb][sd], id_bases, "joint_t1_4cell", n_boot=n_boot)
                        r_ood = S6.score_ood_equal_slice_50_50(
                            loaded_bms[regime][ood_name][ma][sd], loaded_bms[regime][ood_name][mb][sd],
                            ood_slice_map, metric="four_cell", n_boot=n_boot,
                            expected_bases=ood_bases, require_exact_60_60=require_exact_60_60,
                        )
                        r_ood_jt = S6.score_ood_equal_slice_50_50(
                            loaded_bms[regime][ood_name][ma][sd], loaded_bms[regime][ood_name][mb][sd],
                            ood_slice_map, metric="joint_t1_4cell", n_boot=n_boot,
                            expected_bases=ood_bases, require_exact_60_60=require_exact_60_60,
                        )
                        comp_rows.append((tier, cname, f"seed_{sd}", r_id, r_id_jt, r_ood, r_ood_jt))
                    if len(common_seeds) >= 2:
                        pool_id_a = _average_base_maps_by_seed(loaded_bms[regime][id_name][ma], id_bases)
                        pool_id_b = _average_base_maps_by_seed(loaded_bms[regime][id_name][mb], id_bases)
                        pool_ood_a = _average_base_maps_by_seed(loaded_bms[regime][ood_name][ma], ood_bases)
                        pool_ood_b = _average_base_maps_by_seed(loaded_bms[regime][ood_name][mb], ood_bases)
                        r_id_p = _paired_item_bootstrap(pool_id_a, pool_id_b, id_bases, "four_cell", n_boot=n_boot)
                        r_id_jt_p = _paired_item_bootstrap(pool_id_a, pool_id_b, id_bases, "joint_t1_4cell", n_boot=n_boot)
                        r_ood_p = S6.score_ood_equal_slice_50_50(
                            pool_ood_a, pool_ood_b, ood_slice_map, metric="four_cell", n_boot=n_boot,
                            expected_bases=ood_bases, require_exact_60_60=require_exact_60_60,
                        )
                        r_ood_jt_p = S6.score_ood_equal_slice_50_50(
                            pool_ood_a, pool_ood_b, ood_slice_map, metric="joint_t1_4cell", n_boot=n_boot,
                            expected_bases=ood_bases, require_exact_60_60=require_exact_60_60,
                        )
                        comp_rows.append((tier, cname, "pooled_2seed", r_id_p, r_id_jt_p, r_ood_p, r_ood_jt_p))
                        if tier == "primary":
                            p_str = r_ood_p["equal_50_50_p_str"]
                            if "/" in p_str:
                                # Conservative two-sided resolution upper bound (2 / n_boot) -> never reports 0.0000
                                raw_primary_p_ood[cname] = (2.0 / float(n_boot), True)
                            else:
                                raw_primary_p_ood[cname] = (float(p_str), False)

                holm_adj = holm_bonferroni(raw_primary_p_ood) if raw_primary_p_ood else {}
                for tier, cname, sp, r_id, r_id_jt, r_ood, r_ood_jt in comp_rows:
                    ckey = f"{regime}|{cname}|{sp}"
                    h_info = holm_adj.get(cname) if (tier == "primary" and sp == "pooled_2seed") else None
                    scale_doc["comparisons"][ckey] = {
                        "tier": tier,
                        "id_4cell": r_id,
                        "id_joint": r_id_jt,
                        "ood_50_50_4cell": r_ood,
                        "ood_50_50_joint": r_ood_jt,
                        "holm_p_ood_4cell": h_info,
                    }
                    h_s = f"`{h_info['p_adj_str']}`" if h_info is not None else "—"
                    md.append(
                        f"| {tier} | `{cname}` | `{sp}` | **{r_id['diff_pp']:+.2f}** `[{r_id['ci95_pp'][0]:+.2f}, {r_id['ci95_pp'][1]:+.2f}]` | "
                        f"**{r_ood['equal_50_50_diff_pp']:+.2f}** `[{r_ood['equal_50_50_ci95_pp'][0]:+.2f}, {r_ood['equal_50_50_ci95_pp'][1]:+.2f}]` | "
                        f"{r_ood_jt['equal_50_50_diff_pp']:+.2f} `[{r_ood_jt['equal_50_50_ci95_pp'][0]:+.2f}, {r_ood_jt['equal_50_50_ci95_pp'][1]:+.2f}]` | "
                        f"{r_ood['per_slice']['ood_converse']['diff_pp']:+.2f} / {r_ood['per_slice']['ood_depth4']['diff_pp']:+.2f} | "
                        f"`{r_ood['equal_50_50_p_str']}` | {h_s} |"
                    )

        out_doc["scales"][scale_key] = scale_doc

    md_text = "\n".join(md) + "\n"
    out_json.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_md.write_text(md_text, encoding="utf-8", newline="\n")
    print(f"[OK] Wrote Stage 7 sealed readout -> {out_md} & {out_json}")
    return out_doc, md_text


def run_dry_run_open() -> None:
    """Full end-to-end Stage 7 dry-run on REAL open `c1_dev_bases.jsonl` + synthetic 60+60 OOD + 7 adversarial fault injections!

    NEVER opens or reads sealed `c1_test_id_bases.jsonl` or `c1_test_ood_bases.jsonl`.
    """
    c1_dev_path = C1 / "data" / "c1_dev_bases.jsonl"
    assert c1_dev_path.is_file(), f"Missing open c1_dev: {c1_dev_path}"
    dev_rows = [json.loads(l) for l in c1_dev_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(dev_rows) == 80

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        # Build synthetic 60 ood_converse (8 families) + 60 ood_depth4 (60 families, broken_step 1/2/3)
        sim_ood_rows = []
        for i in range(60):
            sim_ood_rows.append({
                "base_id": f"SIM-OOD-CONV-{i + 1:03d}",
                "family_id": f"SIM-FAM-CONV-{(i % 8) + 1:02d}",
                "split": "ood_converse",
                "pair_type": "argument_flip" if i < 30 else "context_flip",
                "initial_answer": "A",
                "target_answer": "B",
            })
        for i in range(60):
            sim_ood_rows.append({
                "base_id": f"SIM-OOD-D4-{i + 1:03d}",
                "family_id": f"SIM-OOD-D4-{i + 1:03d}",
                "split": "ood_depth4",
                "pair_type": "argument_flip" if i < 30 else "context_flip",
                "broken_step_index": (i % 3) + 1,
                "initial_answer": "A",
                "target_answer": "B",
            })
        sim_ood_path = td_p / "sim_ood_60_60.jsonl"
        sim_ood_path.write_text("\n".join(json.dumps(r) for r in sim_ood_rows) + "\n", encoding="utf-8")

        def _make_recs(rows: list[dict[str, Any]], arms: set[str], regime: str, *, t1_ok_fn, fc_ok_fn, fail_first: bool = False) -> list[dict[str, Any]]:
            out = []
            for idx, r in enumerate(rows):
                a0, at = r["initial_answer"], r["target_answer"]
                t1_ok = bool(t1_ok_fn(idx, r))
                fc_ok = bool(fc_ok_fn(idx, r))
                for a in sorted(arms):
                    if fail_first and idx == 0 and a == "F_auth":
                        out.append({
                            "base_id": r["base_id"], "family_id": r.get("family_id", r["base_id"]), "arm": a,
                            "initial_answer": a0, "target_answer": at, "pred": None,
                            "parse_ok": False, "is_target": False, "is_initial": False,
                            "finish_reason": "length",
                            **({"u_theta": 0.0, "gold_nll": 2.0} if regime == "no_think" else {}),
                        })
                        continue
                    if a in ("turn1_V", "turn1_F"):
                        is_i = t1_ok; is_t = not is_i
                    elif a in ("V", "V_auth", "V_hes"):
                        is_t = fc_ok; is_i = not is_t
                    elif a in ("F", "F_auth", "F_hes"):
                        is_i = fc_ok; is_t = not is_i
                    else:
                        is_i = True; is_t = False
                    pred = at if is_t else a0
                    rec = {
                        "base_id": r["base_id"], "family_id": r.get("family_id", r["base_id"]), "arm": a,
                        "initial_answer": a0, "target_answer": at, "pred": pred,
                        "parse_ok": True, "is_target": is_t, "is_initial": is_i,
                        "finish_reason": "stop",
                    }
                    if regime == "no_think":
                        rec["u_theta"] = 6.0 if is_t else -6.0
                        rec["gold_nll"] = 0.25
                        rec["brier_score"] = 0.05
                    out.append(rec)
            return out

        def _write_arm(arm_dir: Path, regimes: list[str], prov_extra: dict[str, Any], *, t1_fn, id_fc_fn, ood_fc_fn, fail_first_th: bool = False) -> None:
            for reg in regimes:
                rdir = arm_dir / reg
                rdir.mkdir(parents=True, exist_ok=True)
                arms = NOTHINK_ARMS_15 if reg == "no_think" else THINK_ARMS_6
                r_id = _make_recs(dev_rows, arms, reg, t1_ok_fn=t1_fn, fc_ok_fn=id_fc_fn, fail_first=(fail_first_th and reg == "think"))
                r_ood = _make_recs(sim_ood_rows, arms, reg, t1_ok_fn=t1_fn, fc_ok_fn=ood_fc_fn, fail_first=False)
                (rdir / "records_" + reg + "_c1_dev.jsonl" if False else rdir / f"records_{reg}_c1_dev.jsonl").write_text(
                    "\n".join(json.dumps(x) for x in r_id) + "\n", encoding="utf-8"
                )
                (rdir / f"records_{reg}_sim_ood.jsonl").write_text(
                    "\n".join(json.dumps(x) for x in r_ood) + "\n", encoding="utf-8"
                )
                (rdir / f"eval_summary_{reg}.json").write_text(
                    json.dumps({"provenance": {"schema": "test", **prov_extra}}, indent=2) + "\n", encoding="utf-8"
                )

        # Write valid simulated arms: M0, u2 (s42, s1024), v4 (s42, s1024), rm (s42, s1024), DPO (no_think only), CAA (no_think only), Pinpoint-Corr (merged think)
        _write_arm(td_p / "m0", ["no_think", "think"], {}, t1_fn=lambda i, r: i % 2 == 0, id_fc_fn=lambda i, r: False, ood_fc_fn=lambda i, r: False)
        _write_arm(td_p / "u2_42", ["no_think", "think"], {"adapter_state_sha256": "a" * 64, "peft_adapter_model_sha256": "b" * 64},
                   t1_fn=lambda i, r: True, id_fc_fn=lambda i, r: i < 40, ood_fc_fn=lambda i, r: (r["split"] == "ood_depth4" and i < 70))  # 0/60 conv, 10/60 d4 -> 8.33%!
        _write_arm(td_p / "u2_1024", ["no_think", "think"], {"adapter_state_sha256": "c" * 64, "peft_adapter_model_sha256": "d" * 64},
                   t1_fn=lambda i, r: True, id_fc_fn=lambda i, r: i < 40, ood_fc_fn=lambda i, r: (r["split"] == "ood_depth4" and i < 70))
        _write_arm(td_p / "v4_42", ["no_think", "think"], {"adapter_state_sha256": "e" * 64, "peft_adapter_model_sha256": "f" * 64},
                   t1_fn=lambda i, r: True, id_fc_fn=lambda i, r: i < 60, ood_fc_fn=lambda i, r: (i < 45 or (r["split"] == "ood_depth4" and i < 100)), fail_first_th=True)
        _write_arm(td_p / "v4_1024", ["no_think", "think"], {"adapter_state_sha256": "1" * 64, "peft_adapter_model_sha256": "2" * 64},
                   t1_fn=lambda i, r: True, id_fc_fn=lambda i, r: i < 60, ood_fc_fn=lambda i, r: (i < 45 or (r["split"] == "ood_depth4" and i < 100)))
        _write_arm(td_p / "dpo_42", ["no_think"], {"adapter_state_sha256": "3" * 64},
                   t1_fn=lambda i, r: True, id_fc_fn=lambda i, r: i < 50, ood_fc_fn=lambda i, r: i < 30)
        _write_arm(td_p / "caa", ["no_think"], {"frozen_vector_sha256": "4" * 64},
                   t1_fn=lambda i, r: True, id_fc_fn=lambda i, r: False, ood_fc_fn=lambda i, r: False)
        _write_arm(td_p / "pinpoint_corr", ["no_think", "think"],
                   {"adapter_state_sha256": "5" * 64, "merged_shards_sha256": {"shard1.safetensors": "6" * 64}},
                   t1_fn=lambda i, r: True, id_fc_fn=lambda i, r: False, ood_fc_fn=lambda i, r: False)

        cfg = {
            "id_dataset_name": "c1_dev",
            "ood_dataset_name": "sim_ood",
            "id_bases_file": str(c1_dev_path),
            "ood_bases_file": str(sim_ood_path),
            "require_exact_60_60": True,
            "scales": {
                "sim_14b": {
                    "m0": {"dir": str(td_p / "m0")},
                    "arms": [
                        {"model_group": "M0", "seed": 42, "dir": str(td_p / "m0"), "regimes": ["no_think", "think"]},
                        {"model_group": "u2_corridor", "seed": 42, "dir": str(td_p / "u2_42"), "regimes": ["no_think", "think"], "expected_adapter_sha256": "a" * 64, "expected_peft_sha256": "b" * 64},
                        {"model_group": "u2_corridor", "seed": 1024, "dir": str(td_p / "u2_1024"), "regimes": ["no_think", "think"], "expected_adapter_sha256": "c" * 64, "expected_peft_sha256": "d" * 64},
                        {"model_group": "v4_mlpplus", "seed": 42, "dir": str(td_p / "v4_42"), "regimes": ["no_think", "think"], "expected_adapter_sha256": "e" * 64, "expected_peft_sha256": "f" * 64},
                        {"model_group": "v4_mlpplus", "seed": 1024, "dir": str(td_p / "v4_1024"), "regimes": ["no_think", "think"], "expected_adapter_sha256": "1" * 64, "expected_peft_sha256": "2" * 64},
                        {"model_group": "joint_dpo", "seed": 42, "dir": str(td_p / "dpo_42"), "regimes": ["no_think"], "expected_adapter_sha256": "3" * 64},
                        {"model_group": "caa_steering", "seed": 42, "dir": str(td_p / "caa"), "regimes": ["no_think"], "expected_vector_sha256": "4" * 64},
                        {"model_group": "pinpoint_dla_corridor", "seed": 42, "dir": str(td_p / "pinpoint_corr"), "regimes": ["no_think", "think"], "expected_adapter_sha256": "5" * 64, "expected_merged_shards_sha256": {"shard1.safetensors": "6" * 64}},
                    ],
                    "comparisons": [
                        {"name": "v4_vs_u2", "tier": "primary", "model_a": "v4_mlpplus", "model_b": "u2_corridor"},
                        {"name": "dpo_vs_v4", "tier": "secondary", "model_a": "joint_dpo", "model_b": "v4_mlpplus"},
                    ],
                }
            },
        }

        out_dir = td_p / "stage7_out"
        doc, md = run_stage7_aggregation(cfg, out_dir, n_boot=500)
        # Verify exact unrounded 8.33% on u2 (0/60 converse + 10/60 depth4)!
        u2_ood_pct = doc["scales"]["sim_14b"]["arms"]["u2_corridor@s42:no_think"]["ood_all_4cell"]["equal_50_50_score_pct"]
        assert u2_ood_pct == 8.33, u2_ood_pct
        # Verify parse_fail & truncation count recorded on v4_42 think!
        v4_th = doc["scales"]["sim_14b"]["arms"]["v4_mlpplus@s42:think"]
        assert v4_th["total_parse_fail"] == 1 and v4_th["total_truncations"] == 1, v4_th
        # Verify Holm-Bonferroni & Joint present for primary comparison (never 0.0000)!
        pri = doc["scales"]["sim_14b"]["comparisons"]["no_think|v4_vs_u2|pooled_2seed"]
        assert pri["holm_p_ood_4cell"] is not None and pri["holm_p_ood_4cell"]["p_adj_str"] != "0.0000", pri
        assert pri["ood_50_50_joint"]["equal_50_50_diff_pp"] is not None, pri

        # === ADVERSARIAL FAULT INJECTION SUITE (ALL 6 MUST FAIL CLOSED!) ===
        # Fault 1: Duplicate aggregation run without --allow-reaggregate MUST block overwrite
        try:
            run_stage7_aggregation(cfg, out_dir, n_boot=200, allow_reaggregate=False)
            raise RuntimeError("Fault 1 failed: allowed overwrite!")
        except FileExistsError:
            pass

        # Fault 2: Wrong adapter SHA-256 MUST fail closed
        cfg_bad_sha = json.loads(json.dumps(cfg))
        cfg_bad_sha["scales"]["sim_14b"]["arms"][1]["expected_adapter_sha256"] = "9" * 64
        try:
            run_stage7_aggregation(cfg_bad_sha, td_p / "out_f2", n_boot=100)
            raise RuntimeError("Fault 2 failed: missed bad SHA-256!")
        except AssertionError:
            pass

        # Fault 3: Drop ONE entire base from `think` records (self-referential validator would have missed this!)
        th_p = td_p / "v4_42" / "think" / "records_think_c1_dev.jsonl"
        orig_th = th_p.read_text(encoding="utf-8")
        dropped_id = dev_rows[0]["base_id"]
        th_p.write_text("\n".join(l for l in orig_th.splitlines() if dropped_id not in l) + "\n", encoding="utf-8")
        try:
            run_stage7_aggregation(cfg, td_p / "out_f3", n_boot=100)
            raise RuntimeError("Fault 3 failed: missed missing entire base in think!")
        except AssertionError:
            pass
        th_p.write_text(orig_th, encoding="utf-8")

        # Fault 4: Corrupt one `no_think` record with pred="Z" / NaN u_theta
        nt_p = td_p / "dpo_42" / "no_think" / "records_no_think_c1_dev.jsonl"
        orig_nt = nt_p.read_text(encoding="utf-8")
        nt_lines = orig_nt.splitlines()
        bad0 = json.loads(nt_lines[0])
        bad0["u_theta"] = float("nan")
        nt_p.write_text(json.dumps(bad0) + "\n" + "\n".join(nt_lines[1:]) + "\n", encoding="utf-8")
        try:
            run_stage7_aggregation(cfg, td_p / "out_f4", n_boot=100)
            raise RuntimeError("Fault 4 failed: missed NaN u_theta!")
        except AssertionError:
            pass
        nt_p.write_text(orig_nt, encoding="utf-8")

        # Fault 5: Corrupt M0 to drop `turn1_V` (leaving `turn1_F`-only)
        m0_p = td_p / "m0" / "no_think" / "records_no_think_c1_dev.jsonl"
        orig_m0 = m0_p.read_text(encoding="utf-8")
        m0_p.write_text("\n".join(l for l in orig_m0.splitlines() if '"turn1_V"' not in l) + "\n", encoding="utf-8")
        try:
            run_stage7_aggregation(cfg, td_p / "out_f5", n_boot=100)
            raise RuntimeError("Fault 5 failed: missed missing turn1_V in M0!")
        except AssertionError:
            pass
        m0_p.write_text(orig_m0, encoding="utf-8")

        # Fault 6: Wrong OOD slice label / 59+61 split in metadata MUST fail closed
        bad_ood_path = td_p / "sim_ood_bad.jsonl"
        bad_rows = [dict(r) for r in sim_ood_rows]
        bad_rows[0]["split"] = "ood_unknown"
        bad_ood_path.write_text("\n".join(json.dumps(r) for r in bad_rows) + "\n", encoding="utf-8")
        cfg_bad_ood = json.loads(json.dumps(cfg))
        cfg_bad_ood["ood_bases_file"] = str(bad_ood_path)
        try:
            run_stage7_aggregation(cfg_bad_ood, td_p / "out_f6", n_boot=100)
            raise RuntimeError("Fault 6 failed: missed unknown OOD slice!")
        except AssertionError:
            pass

    print(
        "[OK] c35_stage7_stats --dry-run-open PASSED: real open c1_dev (80 bases) + synthetic 60+60 OOD "
        "(exact 8.33% unrounded check, dual-seed Holm-Bonferroni, DPO/CAA no_think-only, Pinpoint merged shards, "
        "parse_fail/truncation tracking) AND 6/6 adversarial fault injections failed closed!"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-open", action="store_true", help="End-to-end open-data dry-run + fault injection (zero sealed access!)")
    ap.add_argument("--config-json", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--allow-reaggregate", action="store_true")
    args = ap.parse_args()

    if args.dry_run_open:
        run_dry_run_open()
        return

    assert args.config_json and args.out_dir, "--config-json and --out-dir are required"
    cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    run_stage7_aggregation(cfg, args.out_dir, n_boot=int(args.n_boot), allow_reaggregate=bool(args.allow_reaggregate))


if __name__ == "__main__":
    main()
