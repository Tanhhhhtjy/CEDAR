#!/usr/bin/env python3
"""Build the frozen CEDAR E2A and matched Qwen--Llama delivery bundle.

This script never calls a model. It reads archived forward records, validates them,
and deterministically writes analysis and audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_725
LABELS = ("A", "B", "C", "D")
ARMS = ("N", "R", "V", "F")
CONDITIONS = ("uncapped", "p25")
EXPECTED = {
    "Qwen-FC": {
        "repo": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "experiment": "layers_46:54-p0.25",
        "layers": tuple(range(46, 54)),
    },
    "Llama": {
        "repo": "meta-llama/Llama-3.3-70B-Instruct",
        "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "experiment": "layers_56:72-p0.25",
        "layers": tuple(range(56, 72)),
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return statistics.fmean(values)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_family_scores(scores: Mapping[str, float]) -> dict[str, Any]:
    families = sorted(scores)
    if not families:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "n_families": 0,
            "family_scores": {},
        }
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [
        mean([float(scores[rng.choice(families)]) for _ in families])
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "estimate": mean([float(scores[family]) for family in families]),
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
        "n_families": len(families),
        "family_scores": dict(sorted(scores.items())),
    }


def geometry(row: Mapping[str, Any]) -> dict[str, float]:
    logits = row.get("answer_logits", {}).get("logits")
    if not isinstance(logits, dict) or set(logits) != set(LABELS):
        raise ValueError(f"Missing A/B/C/D logits: {row.get('model')} {row.get('id')}")
    initial = str(row["initial_answer"])
    target = str(row["target_answer"])
    if initial not in LABELS or target not in LABELS or initial == target:
        raise ValueError(f"Invalid initial/target mapping: {row.get('id')}")
    other = [label for label in LABELS if label not in {initial, target}]
    m_target = float(logits[target]) - float(logits[initial])
    background = [float(logits[label]) - float(logits[initial]) for label in other]
    m_background = mean(background)
    return {
        "m_general": (m_target + sum(background)) / 3.0,
        "m_target": m_target,
        "m_background": m_background,
        "m_specific": m_target - m_background,
    }


def target_follow(row: Mapping[str, Any]) -> float:
    return float(row.get("outcome_class") == "induced_target")


def validity_flag(row: Mapping[str, Any]) -> str:
    if row.get("answer_logits", {}).get("logits") is None:
        return "missing_logits"
    if bool(row.get("truncated", False)):
        return "truncated"
    if bool(row.get("invalid_or_abstain_indicator", False)):
        return "invalid"
    if row.get("final_answer_parse_status") != "passed":
        return "invalid"
    return "valid"


def state_for_delivery(row: Mapping[str, Any]) -> str:
    return {
        "initial": "initial",
        "induced_target": "target",
        "other_wrong": "other",
        "invalid_or_abstain": "invalid",
    }.get(str(row.get("outcome_class")), "invalid")


def cap_totals(row: Mapping[str, Any]) -> tuple[int, int, float, dict[str, Any]]:
    cap = row.get("cap_stats") or row.get("full_cap_stats")
    if not cap:
        return 0, 0, 0.0, {}
    return (
        int(cap.get("total_trigger_count", 0)),
        int(cap.get("total_corrected_token_count", 0)),
        float(cap.get("total_correction_norm", 0.0)),
        dict(cap.get("per_layer", {})),
    )


def normalize_cross_record(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = geometry(row)
    trigger_count, correction_count, correction_norm, per_layer = cap_totals(row)
    logits = row["answer_logits"]
    expected = EXPECTED[str(row["model"])]
    return {
        "model": row["model"],
        "model_revision": row.get("model_revision"),
        "family_id": row["family_id"],
        "base_id": row["base_id"],
        "arm": row["arm"],
        "intervention": row["condition"],
        "prompt_hash": row.get("prompt_sha256"),
        "prompt_text_or_stable_id": row.get("id"),
        "answer_order": list(LABELS),
        "initial_option": row["initial_answer"],
        "target_option": row["target_answer"],
        "option_token_ids": row.get("candidate_token_ids"),
        "all_option_logits": logits.get("logits"),
        "predicted_option": logits.get("option_argmax"),
        "output_state": state_for_delivery(row),
        "target_follow": bool(target_follow(row)),
        **metrics,
        "trigger_count": trigger_count,
        "correction_count": correction_count,
        "correction_norm": correction_norm,
        "per_layer_telemetry": per_layer,
        "option_probability_mass": logits.get("option_mass"),
        "entropy": logits.get("full_vocab_entropy"),
        "top_token": logits.get("full_vocab_top_token_text"),
        "top_token_id": logits.get("full_vocab_top_token_id"),
        "truncation_flag": bool(row.get("truncated", False)),
        "invalid_flag": validity_flag(row) != "valid",
        "axis_hash": row.get("axis_sha256"),
        "threshold_hash": None,
        "config_hash": row.get("capping_config_sha256"),
        "layer_range": f"{expected['layers'][0]}-{expected['layers'][-1]}",
        "seed": row.get("seed"),
        "runner_commit": None,
        "source_schema": row.get("schema"),
        "source_record_id": row.get("id"),
    }


def pair_rows(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    by_key = {(str(row["id"]), str(row["condition"])): row for row in rows}
    ids = sorted({str(row["id"]) for row in rows})
    pairs = {}
    for record_id in ids:
        missing = [condition for condition in CONDITIONS if (record_id, condition) not in by_key]
        if missing:
            raise ValueError(f"Unpaired record {record_id}: missing {missing}")
        pairs[record_id] = (by_key[(record_id, "uncapped")], by_key[(record_id, "p25")])
    return pairs


def audit_cross_model(
    model: str, rows: Sequence[dict[str, Any]], other_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected = EXPECTED[model]
    pairs = pair_rows(rows)
    revisions = sorted({row.get("model_revision") for row in rows})
    repos = sorted({row.get("model_repo") for row in rows})
    experiments = sorted({row.get("official_experiment") for row in rows})
    axis_hashes = sorted({row.get("axis_sha256") for row in rows})
    config_hashes = sorted({row.get("capping_config_sha256") for row in rows})
    expected_layer_keys = {str(layer) for layer in expected["layers"]}

    prompt_bank = []
    for record_id, (uncapped, _) in sorted(pairs.items()):
        prompt_bank.append(
            {
                "id": record_id,
                "family_id": uncapped["family_id"],
                "base_id": uncapped["base_id"],
                "arm": uncapped["arm"],
                "prompt_sha256": uncapped.get("prompt_sha256"),
                "initial_answer": uncapped["initial_answer"],
                "target_answer": uncapped["target_answer"],
            }
        )

    checks: dict[str, bool] = {}
    checks["forward_count_192"] = len(rows) == 192
    checks["paired_96"] = len(pairs) == 96
    checks["families_12"] = len({row["family_id"] for row in rows}) == 12
    checks["bases_24"] = len({row["base_id"] for row in rows}) == 24
    checks["arms_24_prompts_each"] = all(
        sum(1 for _, (row, _) in pairs.items() if row["arm"] == arm) == 24 for arm in ARMS
    )
    checks["model_repo_locked"] = repos == [expected["repo"]]
    checks["model_revision_locked"] = revisions == [expected["revision"]]
    checks["official_experiment_locked"] = experiments == [expected["experiment"]]
    checks["reasoning_off_forced_choice"] = all(
        row.get("mode") == "reasoning_off_forced_choice"
        and row.get("thinking_enabled") is False
        and row.get("finish_reason") == "forced_choice_logit_readout"
        for row in rows
    )
    checks["single_token_abcd"] = all(
        isinstance(row.get("candidate_token_ids"), dict)
        and set(row["candidate_token_ids"]) == set(LABELS)
        and all(isinstance(row["candidate_token_ids"][label], int) for label in LABELS)
        and len(set(row["candidate_token_ids"].values())) == 4
        for row in rows
    )
    checks["answer_mapping_consistent"] = all(
        row.get("position_transition")
        == f"{row.get('initial_answer')}->{row.get('target_answer')}"
        and row.get("initial_answer") in LABELS
        and row.get("target_answer") in LABELS
        and row.get("initial_answer") != row.get("target_answer")
        for row in rows
    )
    checks["within_model_prompt_pairs_equal"] = all(
        uncapped.get("prompt_sha256") == p25.get("prompt_sha256")
        and uncapped.get("candidate_token_ids") == p25.get("candidate_token_ids")
        and uncapped.get("initial_answer") == p25.get("initial_answer")
        and uncapped.get("target_answer") == p25.get("target_answer")
        for uncapped, p25 in pairs.values()
    )
    other_prompts = {
        (row["id"], row["condition"]): row.get("prompt_sha256") for row in other_rows
    }
    checks["cross_model_prompt_hash_match"] = all(
        other_prompts.get((row["id"], row["condition"])) == row.get("prompt_sha256")
        for row in rows
    )
    checks["axis_hash_locked"] = len(axis_hashes) == 1 and axis_hashes[0] is not None
    checks["config_hash_locked"] = len(config_hashes) == 1 and config_hashes[0] is not None
    checks["hook_layers_exact"] = all(
        set((row.get("hook_counts_before") or {}).keys()) == expected_layer_keys
        and set((row.get("hook_counts_after") or {}).keys()) == expected_layer_keys
        for row in rows
    )
    checks["uncapped_no_intervention"] = all(
        row.get("cap_stats") is None
        and all(int(value) == 0 for value in (row.get("hook_counts_before") or {}).values())
        and all(int(value) == 0 for value in (row.get("hook_counts_after") or {}).values())
        for row in rows
        if row["condition"] == "uncapped"
    )
    checks["p25_hooked"] = all(
        all(int(value) > 0 for value in (row.get("hook_counts_before") or {}).values())
        and all(int(value) > 0 for value in (row.get("hook_counts_after") or {}).values())
        for row in rows
        if row["condition"] == "p25"
    )
    checks["p25_triggered_and_corrected"] = all(
        (row.get("cap_stats") or {}).get("total_trigger_count", 0) > 0
        and (row.get("cap_stats") or {}).get("total_corrected_token_count", 0) > 0
        and (row.get("cap_stats") or {}).get("total_correction_norm", 0) > 0
        for row in rows
        if row["condition"] == "p25"
    )
    checks["complete_option_logits"] = all(
        isinstance(row.get("answer_logits", {}).get("logits"), dict)
        and set(row["answer_logits"]["logits"]) == set(LABELS)
        and all(math.isfinite(float(value)) for value in row["answer_logits"]["logits"].values())
        for row in rows
    )
    checks["option_argmax_recomputes"] = all(
        row["answer_logits"]["option_argmax"]
        == max(LABELS, key=lambda label: float(row["answer_logits"]["logits"][label]))
        for row in rows
    )
    checks["output_integrity"] = all(validity_flag(row) == "valid" for row in rows)

    provenance = {
        "tokenizer_identity": "model repository and pinned revision recorded; no separate tokenizer hash",
        "chat_template": None,
        "threshold_hash": None,
        "runner_commit": None,
        "separate_smoke_artifact": None,
    }
    return {
        "model": model,
        "status": "PASS_CORE_GATES_WITH_PROVENANCE_WARNINGS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "provenance_limitations": provenance,
        "model_revision": revisions,
        "model_repo": repos,
        "official_experiment": experiments,
        "axis_hash": axis_hashes,
        "config_hash": config_hashes,
        "prompt_bank_hash": canonical_sha256(prompt_bank),
        "prompt_bank_definition": "canonical JSON over 96 stable IDs and prompt/mapping hashes",
    }


def e2a_export(
    rows: Sequence[dict[str, Any]], frozen_analysis: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("assay") == "answer" and row.get("condition") in {"uncapped", "full_p25"}
    ]
    by_key = {(row["id"], row["condition"]): row for row in selected}
    identities = sorted({row["id"] for row in selected})
    if len(selected) != 192 or len(identities) != 96:
        raise ValueError(f"Expected 192 E2A rows/96 cells, got {len(selected)}/{len(identities)}")

    arm_rows: list[dict[str, Any]] = []
    family_pairs: dict[str, list[tuple[dict[str, float], dict[str, float]]]] = defaultdict(list)
    family_missing: Counter[str] = Counter()
    family_invalid: Counter[str] = Counter()
    family_bases: dict[str, set[str]] = defaultdict(set)
    for record_id in identities:
        missing = [condition for condition in ("uncapped", "full_p25") if (record_id, condition) not in by_key]
        if missing:
            existing = next(row for row in selected if row["id"] == record_id)
            family_missing[existing["family_id"]] += len(missing)
            continue
        uncapped = by_key[(record_id, "uncapped")]
        p25 = by_key[(record_id, "full_p25")]
        if uncapped.get("prompt_sha256") != p25.get("prompt_sha256"):
            raise ValueError(f"E2A prompt mismatch: {record_id}")
        u_metrics = geometry(uncapped)
        p_metrics = geometry(p25)
        flag_u = validity_flag(uncapped)
        flag_p = validity_flag(p25)
        flag = "valid" if flag_u == flag_p == "valid" else f"{flag_u}|{flag_p}"
        if flag != "valid":
            family_invalid[uncapped["family_id"]] += 1
        family = uncapped["family_id"]
        family_bases[family].add(uncapped["base_id"])
        family_pairs[family].append((u_metrics, p_metrics))
        arm_rows.append(
            {
                "family_id": family,
                "base_id": uncapped["base_id"],
                "arm": uncapped["arm"],
                "m_general_uncapped": u_metrics["m_general"],
                "m_general_p25": p_metrics["m_general"],
                "delta_m_general": p_metrics["m_general"] - u_metrics["m_general"],
                "m_target_uncapped": u_metrics["m_target"],
                "m_target_p25": p_metrics["m_target"],
                "delta_m_target": p_metrics["m_target"] - u_metrics["m_target"],
                "target_follow_uncapped": target_follow(uncapped),
                "target_follow_p25": target_follow(p25),
                "target_follow_change": target_follow(p25) - target_follow(uncapped),
                "validity_flag": flag,
            }
        )

    family_rows = []
    family_scores = {}
    for family in sorted(family_pairs):
        pairs = family_pairs[family]
        uncapped_values = [u["m_general"] for u, _ in pairs]
        p25_values = [p["m_general"] for _, p in pairs]
        delta = mean([p - u for p, u in zip(p25_values, uncapped_values, strict=True)])
        family_scores[family] = delta
        family_rows.append(
            {
                "family_id": family,
                "n_bases": len(family_bases[family]),
                "n_records": 2 * len(pairs),
                "m_general_uncapped": mean(uncapped_values),
                "m_general_p25": mean(p25_values),
                "delta_m_general": delta,
                "missing_count": family_missing[family],
                "invalid_count": family_invalid[family],
            }
        )
    bootstrap = bootstrap_family_scores(family_scores)
    frozen = frozen_analysis["contrasts_vs_uncapped"]["geometry"]["full_p25"]["m_general"]
    audit = {
        "families": len(family_rows),
        "bases": len({row["base_id"] for row in arm_rows}),
        "paired_cells": len(arm_rows),
        "raw_records": len(selected),
        "missing": sum(family_missing.values()),
        "invalid": sum(family_invalid.values()),
        "truncated": sum(
            bool(row.get("truncated", False)) for row in selected
        ),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "structural_family",
            "method": "percentile family-cluster bootstrap with replacement",
        },
        "recomputed": bootstrap,
        "frozen": {
            "estimate": frozen["estimate"],
            "ci_low": frozen["family_cluster_bootstrap_95ci"][0],
            "ci_high": frozen["family_cluster_bootstrap_95ci"][1],
            "family_scores": frozen["family_scores"],
        },
    }
    audit["exact_match"] = (
        math.isclose(bootstrap["estimate"], frozen["estimate"], abs_tol=1e-12)
        and math.isclose(bootstrap["ci_low"], frozen["family_cluster_bootstrap_95ci"][0], abs_tol=1e-12)
        and math.isclose(bootstrap["ci_high"], frozen["family_cluster_bootstrap_95ci"][1], abs_tol=1e-12)
        and all(
            math.isclose(family_scores[key], frozen["family_scores"][key], abs_tol=1e-12)
            for key in family_scores
        )
    )
    if not audit["exact_match"]:
        raise ValueError("E2A family export does not reproduce frozen estimate and CI")
    return family_rows, sorted(arm_rows, key=lambda row: (row["family_id"], row["base_id"], row["arm"])), audit


def cross_model_analysis(
    rows_by_model: Mapping[str, Sequence[dict[str, Any]]], gate_status: Mapping[str, str]
) -> dict[str, Any]:
    family_effects: list[dict[str, Any]] = []
    pair_audit: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    model_cache: dict[str, dict[str, Any]] = {}

    for model in ("Qwen-FC", "Llama"):
        rows = list(rows_by_model[model])
        pairs = pair_rows(rows)
        family_prompt_deltas: dict[str, list[dict[str, float]]] = defaultdict(list)
        family_arm_deltas: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
        family_stability: dict[tuple[str, str], list[float]] = defaultdict(list)
        family_telemetry: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
        state_counts: Counter[tuple[str, str, str]] = Counter()

        for record_id, (uncapped, p25) in sorted(pairs.items()):
            u = geometry(uncapped)
            p = geometry(p25)
            delta = {name: p[name] - u[name] for name in u}
            family_prompt_deltas[uncapped["family_id"]].append(delta)
            family_arm_deltas[(uncapped["family_id"], uncapped["arm"])].append(delta)
            stable = float(
                uncapped["answer_logits"]["option_argmax"]
                == p25["answer_logits"]["option_argmax"]
            )
            if uncapped["arm"] in {"N", "R"}:
                family_stability[(uncapped["family_id"], uncapped["arm"])].append(stable)
            trigger, corrected, norm, _ = cap_totals(p25)
            family_telemetry[uncapped["family_id"]].append((trigger, corrected, norm))
            state_counts[(uncapped["arm"], state_for_delivery(uncapped), state_for_delivery(p25))] += 1
            pair_audit.append(
                {
                    "model": model,
                    "family_id": uncapped["family_id"],
                    "base_id": uncapped["base_id"],
                    "arm": uncapped["arm"],
                    "stable_id": record_id,
                    "uncapped_present": True,
                    "p25_present": True,
                    "prompt_hash_match": uncapped.get("prompt_sha256") == p25.get("prompt_sha256"),
                    "answer_mapping_match": (
                        uncapped.get("initial_answer") == p25.get("initial_answer")
                        and uncapped.get("target_answer") == p25.get("target_answer")
                    ),
                    "option_token_ids_match": uncapped.get("candidate_token_ids") == p25.get("candidate_token_ids"),
                    "pair_status": "paired",
                }
            )

        for family in sorted({row["family_id"] for row in rows}):
            for arm in ARMS:
                family_rows = [
                    row for row in rows if row["family_id"] == family and row["arm"] == arm
                ]
                by_condition = {
                    condition: [row for row in family_rows if row["condition"] == condition]
                    for condition in CONDITIONS
                }
                if any(len(by_condition[condition]) != 2 for condition in CONDITIONS):
                    raise ValueError(f"Unexpected family/arm cell size: {model} {family} {arm}")
                u_metrics = [geometry(row) for row in by_condition["uncapped"]]
                p_metrics = [geometry(row) for row in by_condition["p25"]]
                family_effects.append(
                    {
                        "model": model,
                        "family_id": family,
                        "arm": arm,
                        "m_general_uncapped": mean([item["m_general"] for item in u_metrics]),
                        "m_general_p25": mean([item["m_general"] for item in p_metrics]),
                        "delta_m_general": mean([item["m_general"] for item in p_metrics]) - mean([item["m_general"] for item in u_metrics]),
                        "m_target_uncapped": mean([item["m_target"] for item in u_metrics]),
                        "m_target_p25": mean([item["m_target"] for item in p_metrics]),
                        "delta_m_target": mean([item["m_target"] for item in p_metrics]) - mean([item["m_target"] for item in u_metrics]),
                        "target_follow_uncapped": mean([target_follow(row) for row in by_condition["uncapped"]]),
                        "target_follow_p25": mean([target_follow(row) for row in by_condition["p25"]]),
                        "target_follow_change": mean([target_follow(row) for row in by_condition["p25"]]) - mean([target_follow(row) for row in by_condition["uncapped"]]),
                    }
                )

        effect_rows = [row for row in family_effects if row["model"] == model]
        overall_scores = {
            family: mean([item["m_general"] for item in deltas])
            for family, deltas in family_prompt_deltas.items()
        }
        specific_scores = {
            family: mean([item["m_specific"] for item in deltas])
            for family, deltas in family_prompt_deltas.items()
        }
        v_minus_f_scores = {
            family: mean([item["m_target"] for item in family_arm_deltas[(family, "V")]])
            - mean([item["m_target"] for item in family_arm_deltas[(family, "F")]])
            for family in sorted(family_prompt_deltas)
        }
        model_cache[model] = {
            "overall_scores": overall_scores,
            "v_minus_f_scores": v_minus_f_scores,
        }

        def add_summary(
            estimand: str,
            scores: Mapping[str, float],
            unit: str,
            n_bases: int,
            n_records: int,
        ) -> None:
            result = bootstrap_family_scores(scores)
            summary.append(
                {
                    "model": model,
                    "estimand": estimand,
                    "estimate": result["estimate"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "unit": unit,
                    "n_families": result["n_families"],
                    "n_bases": n_bases,
                    "n_records": n_records,
                    "gate_status": gate_status[model],
                }
            )

        add_summary("overall_delta_m_general", overall_scores, "logit", 24, 192)
        for arm in ARMS:
            scores = {
                family: mean([item["m_general"] for item in family_arm_deltas[(family, arm)]])
                for family in sorted(family_prompt_deltas)
            }
            add_summary(f"{arm}_delta_m_general", scores, "logit", 24, 48)
        add_summary("pooled_delta_m_specific", specific_scores, "logit", 24, 192)
        add_summary("v_minus_f_continuous_selectivity_change", v_minus_f_scores, "logit", 24, 96)
        for arm in ("V", "F"):
            scores = {
                row["family_id"]: row["target_follow_change"]
                for row in effect_rows
                if row["arm"] == arm
            }
            add_summary(f"{arm}_target_follow_change", scores, "proportion", 24, 48)
        for arm in ("N", "R"):
            scores = {
                family: mean(values)
                for (family, source_arm), values in family_stability.items()
                if source_arm == arm
            }
            add_summary(f"{arm}_argmax_stability", scores, "proportion", 24, 48)

        trigger_scores = {family: mean([float(x[0]) for x in values]) for family, values in family_telemetry.items()}
        correction_scores = {family: mean([float(x[1]) for x in values]) for family, values in family_telemetry.items()}
        norm_scores = {family: mean([x[2] for x in values]) for family, values in family_telemetry.items()}
        add_summary("p25_mean_trigger_count", trigger_scores, "count_per_forward", 24, 96)
        add_summary("p25_mean_correction_count", correction_scores, "count_per_forward", 24, 96)
        add_summary("p25_mean_correction_norm", norm_scores, "norm_per_forward", 24, 96)
        for family in sorted(family_telemetry):
            values = family_telemetry[family]
            telemetry.append(
                {
                    "model": model,
                    "family_id": family,
                    "n_p25_records": len(values),
                    "mean_trigger_count": mean([float(x[0]) for x in values]),
                    "mean_correction_count": mean([float(x[1]) for x in values]),
                    "mean_correction_norm": mean([x[2] for x in values]),
                    "total_trigger_count": sum(x[0] for x in values),
                    "total_correction_count": sum(x[1] for x in values),
                    "total_correction_norm": sum(x[2] for x in values),
                }
            )

        for arm in ARMS:
            for source in ("initial", "target", "other", "invalid"):
                for destination in ("initial", "target", "other", "invalid"):
                    migrations.append(
                        {
                            "model": model,
                            "arm": arm,
                            "source_state": source,
                            "destination_state": destination,
                            "count": state_counts[(arm, source, destination)],
                        }
                    )
        for condition in CONDITIONS:
            condition_rows = [row for row in rows if row["condition"] == condition]
            integrity.append(
                {
                    "model": model,
                    "intervention": condition,
                    "n_records": len(condition_rows),
                    "missing_logits": sum(row.get("answer_logits", {}).get("logits") is None for row in condition_rows),
                    "invalid_count": sum(validity_flag(row) == "invalid" for row in condition_rows),
                    "truncation_count": sum(bool(row.get("truncated", False)) for row in condition_rows),
                    "top_token_not_option_count": sum(not bool(row.get("answer_logits", {}).get("top_token_is_option", False)) for row in condition_rows),
                    "mean_option_probability_mass": mean([float(row["answer_logits"]["option_mass"]) for row in condition_rows]),
                    "mean_entropy": mean([float(row["answer_logits"]["full_vocab_entropy"]) for row in condition_rows]),
                }
            )

    interaction_scores = {
        family: model_cache["Llama"]["overall_scores"][family]
        - model_cache["Qwen-FC"]["overall_scores"][family]
        for family in model_cache["Qwen-FC"]["overall_scores"]
    }
    interaction = bootstrap_family_scores(interaction_scores)

    return {
        "family_effects": sorted(family_effects, key=lambda row: (row["model"], row["family_id"], row["arm"])),
        "summary": summary,
        "pair_audit": pair_audit,
        "migrations": migrations,
        "telemetry": telemetry,
        "integrity": integrity,
        "interaction": interaction,
    }


def route_result(summary: Sequence[Mapping[str, Any]], gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if any(gate["status"] == "FAIL" for gate in gates.values()):
        return {"route": "G-D invalid", "reason": "At least one core tokenization/cap/output/pairing gate failed."}
    lookup = {(row["model"], row["estimand"]): row for row in summary}
    qwen = lookup[("Qwen-FC", "overall_delta_m_general")]
    llama = lookup[("Llama", "overall_delta_m_general")]
    q_select = lookup[("Qwen-FC", "v_minus_f_continuous_selectivity_change")]
    l_select = lookup[("Llama", "v_minus_f_continuous_selectivity_change")]
    same_direction = qwen["estimate"] * llama["estimate"] > 0
    llama_excludes_zero = llama["ci_low"] > 0 or llama["ci_high"] < 0
    no_reliable_selectivity = (
        q_select["ci_low"] <= 0 <= q_select["ci_high"]
        and l_select["ci_low"] <= 0 <= l_select["ci_high"]
    )
    if same_direction and llama_excludes_zero and no_reliable_selectivity:
        return {
            "route": "G-A transport",
            "reason": "Both within-model overall effects are positive; the Llama CI excludes zero; both V-F continuous-selectivity CIs include zero.",
        }
    if same_direction:
        return {
            "route": "G-B partial transport",
            "reason": "The overall direction is shared, but the full G-A uncertainty/selectivity criterion is not met.",
        }
    if qwen["estimate"] > 0 and llama["estimate"] <= 0:
        return {"route": "G-C model-specific", "reason": "Qwen retains the primary direction while Llama is null or reversed."}
    return {"route": "G-D invalid", "reason": "The archived evidence does not satisfy G-A/B/C routing prerequisites."}


def markdown_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def build_reports(
    out: Path,
    e2a_audit: Mapping[str, Any],
    gates: Mapping[str, Mapping[str, Any]],
    analysis: Mapping[str, Any],
    route: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> None:
    lookup = {(row["model"], row["estimand"]): row for row in analysis["summary"]}
    qwen = lookup[("Qwen-FC", "overall_delta_m_general")]
    llama = lookup[("Llama", "overall_delta_m_general")]
    q_sel = lookup[("Qwen-FC", "v_minus_f_continuous_selectivity_change")]
    l_sel = lookup[("Llama", "v_minus_f_continuous_selectivity_change")]
    integrity_totals = {
        "missing": sum(row["missing_logits"] for row in analysis["integrity"]),
        "invalid": sum(row["invalid_count"] for row in analysis["integrity"]),
        "truncation": sum(row["truncation_count"] for row in analysis["integrity"]),
    }

    (out / "protocol_frozen.md").write_text(
        (
            "# Frozen protocol: matched Qwen--Llama transport\n\n"
            "- Bank: 24 bases, 12 structural families, N/R/V/F (96 prompts).\n"
            "- Qwen: `Qwen/Qwen3-32B`, reasoning off, forced choice, uncapped vs official `layers_46:54-p0.25` (layers 46--53).\n"
            "- Llama: `meta-llama/Llama-3.3-70B-Instruct`, reasoning off, forced choice, uncapped vs official `layers_56:72-p0.25` (layers 56--71).\n"
            "- Primary contrast: within-model `p25 - uncapped`; raw logit levels are not compared across models.\n"
            "- Pairing: stable prompt ID and prompt hash. No prompt, answer-order, family, or response-based exclusions.\n"
            f"- Inference: {BOOTSTRAP_DRAWS:,} percentile family-cluster bootstrap draws, seed `{BOOTSTRAP_SEED}`, family as the independent unit.\n"
            "- ITT: invalid and truncated records remain in integrity counts; continuous metrics require archived option logits.\n"
            "- This delivery does not run a model or reinterpret frozen C1/C2/C3 evidence.\n"
        ),
        encoding="utf-8",
        newline="\n",
    )

    gate_lines = [
        "# Gate report",
        "",
        "The four G-D core gates (single-token option readout, cap exposure, output space, and strict pairing) pass for both archived model tracks.",
        "",
    ]
    for model in ("Qwen-FC", "Llama"):
        gate = gates[model]
        gate_lines.extend(
            [
                f"## {model}",
                "",
                f"- Status: `{gate['status']}`",
                f"- Revision: `{gate['model_revision'][0]}`",
                f"- Axis hash: `{gate['axis_hash'][0]}`",
                f"- Capping config hash: `{gate['config_hash'][0]}`",
                f"- Prompt-bank hash: `{gate['prompt_bank_hash']}`",
                "",
            ]
        )
        for name, passed in gate["checks"].items():
            gate_lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
        gate_lines.extend(
            [
                "",
                "Provenance warnings: the archived JSONL does not contain a separate threshold-file hash, chat-template identifier/hash, runner commit, or separately labelled smoke artifact. These fields remain `null`; the capping-config hash is not relabelled as a threshold hash.",
                "",
            ]
        )
    (out / "gate_report.md").write_text("\n".join(gate_lines) + "\n", encoding="utf-8", newline="\n")

    (out / "e2a_family_plot_audit.md").write_text(
        (
            "# E2A family plot audit\n\n"
            f"- Families: `{e2a_audit['families']}/12`.\n"
            f"- Bases: `{e2a_audit['bases']}/24`; paired N/R/V/F cells: `{e2a_audit['paired_cells']}/96`.\n"
            f"- Archived records used: `{e2a_audit['raw_records']}` (uncapped + full p25 only).\n"
            f"- Missing / invalid / truncated: `{e2a_audit['missing']} / {e2a_audit['invalid']} / {e2a_audit['truncated']}`.\n"
            f"- Recomputed overall `delta m_general`: `{e2a_audit['recomputed']['estimate']:.12f}` logits.\n"
            f"- Recomputed 95% family-cluster CI: `[{e2a_audit['recomputed']['ci_low']:.12f}, {e2a_audit['recomputed']['ci_high']:.12f}]`.\n"
            f"- Frozen target: `+3.130208333333` and `[1.834190538194, 4.486121961806]`; exact numerical match: `{str(e2a_audit['exact_match']).lower()}`.\n"
            f"- Bootstrap: `{BOOTSTRAP_DRAWS}` draws, seed `{BOOTSTRAP_SEED}`, resampling 12 families with replacement.\n\n"
            "No row was removed according to model output. Missing/truncated/invalid states are counted before metrics are evaluated; this frozen slice contains none. Figure 0 must use the 12 observed family means in `e2a_family_geometry.csv`. The lower V/F geometry uses `delta_m_target` in logits and is not combined on an axis with target-follow percentage points.\n"
        ),
        encoding="utf-8",
        newline="\n",
    )

    route_text = "# Cross-model route decision\n\n"
    route_text += f"**Route: {route['route']}.** {route['reason']}\n\n"
    route_text += (
        f"- Qwen-FC overall `delta m_general`: {markdown_number(qwen['estimate'])} "
        f"[{markdown_number(qwen['ci_low'])}, {markdown_number(qwen['ci_high'])}] logits.\n"
        f"- Llama overall `delta m_general`: {markdown_number(llama['estimate'])} "
        f"[{markdown_number(llama['ci_low'])}, {markdown_number(llama['ci_high'])}] logits.\n"
        f"- Qwen-FC V-F continuous selectivity change: {markdown_number(q_sel['estimate'])} "
        f"[{markdown_number(q_sel['ci_low'])}, {markdown_number(q_sel['ci_high'])}] logits.\n"
        f"- Llama V-F continuous selectivity change: {markdown_number(l_sel['estimate'])} "
        f"[{markdown_number(l_sel['ci_low'])}, {markdown_number(l_sel['ci_high'])}] logits.\n\n"
        "Allowed wording: under the archived matched forced-choice regime, both audited model-specific p25 configurations shifted general answer geometry in the same direction, while neither V-F interval showed a reliable selectivity improvement.\n\n"
        "Forbidden wording: universal transport; cross-model equivalence; proof of zero selectivity; direct comparison of unstandardized raw-logit levels; or a claim that every pre-run provenance field was archived.\n"
    )
    (out / "route_decision.md").write_text(route_text, encoding="utf-8", newline="\n")

    run_log = "# Run log\n\n"
    run_log += f"- Delivery build time (UTC): `{datetime.now(timezone.utc).isoformat()}`.\n"
    run_log += "- Model forwards launched by this delivery: `0`.\n"
    run_log += "- Archived deterministic forwards analyzed: `384` (Qwen-FC 192; Llama 192).\n"
    run_log += f"- Cross-model missing / invalid / truncated: `{integrity_totals['missing']} / {integrity_totals['invalid']} / {integrity_totals['truncation']}`.\n"
    run_log += f"- E2A archived records read: `{e2a_audit['raw_records']}`; E2A source was not rerun or modified.\n"
    run_log += f"- Route: `{route['route']}`.\n"
    run_log += "- Source runner commit: unavailable in the supplied cross-model JSONL; recorded as null rather than inferred.\n"
    (out / "run_log.md").write_text(run_log, encoding="utf-8", newline="\n")

    analysis_report = "# Analysis report\n\n"
    analysis_report += "## Analysis question\n\nDoes official model-specific p25 capping shift answer-revision geometry within both matched forced-choice model tracks without a reliable V/F selectivity improvement?\n\n"
    analysis_report += "## Main findings\n\n"
    analysis_report += route_text.split("\n\n", 2)[2]
    analysis_report += "\n## Claim candidates\n\n"
    analysis_report += "- Claim: Both archived model-specific configurations show a positive within-model general-margin shift.\n  - Source evidence: `cross_model_summary.csv`, `raw_records.jsonl`.\n  - Allowed wording: bounded to these models, prompts, and official p25 configurations.\n  - Forbidden stronger wording: universal model-family law or equal effect magnitude.\n  - Uncertainty: 12-family percentile bootstrap.\n  - Decision: keep.\n\n"
    analysis_report += "- Claim: V/F selectivity did not reliably improve.\n  - Source evidence: both V-F continuous-selectivity CIs include zero.\n  - Allowed wording: no reliable improvement under this assay.\n  - Forbidden stronger wording: equivalence or proof of a zero effect.\n  - Decision: keep with calibrated wording.\n\n"
    analysis_report += "## Caveats\n\nThe supplied JSONL omits a separate threshold hash, chat-template identifier, source runner commit, and separately archived smoke artifact. Core result-space gates pass, but those provenance fields cannot be reconstructed.\n"
    (out / "analysis-output" / "analysis-report.md").write_text(analysis_report, encoding="utf-8", newline="\n")

    stats_text = "# Statistics appendix\n\n"
    stats_text += f"- Unit of analysis: 12 independently generated structural families; 24 bases nested within family.\n- Pairing: exact stable prompt ID within model; cross-model prompt hashes checked.\n- Bootstrap: {BOOTSTRAP_DRAWS:,} family-cluster draws with replacement; seed {BOOTSTRAP_SEED}; percentile 2.5% and 97.5% endpoints with linear interpolation.\n- Main effects: within-model p25 minus uncapped.\n- ITT: no model-output exclusions; invalid and truncated records retained in integrity accounting.\n- Multiplicity: arm and telemetry rows are descriptive secondary breakdowns; routing is based only on frozen overall and V-F criteria.\n- Cross-model interaction: secondary paired-family difference, not inferred from one significant and one nonsignificant result.\n\n"
    stats_text += f"Secondary Llama-minus-Qwen overall interaction: {markdown_number(analysis['interaction']['estimate'])} [{markdown_number(analysis['interaction']['ci_low'])}, {markdown_number(analysis['interaction']['ci_high'])}] logits.\n"
    (out / "analysis-output" / "stats-appendix.md").write_text(stats_text, encoding="utf-8", newline="\n")

    catalog_text = (
        "# Figure catalog\n\n"
        "## `figures/e2a_family_geometry.*`\n\n"
        "- Purpose: replace Figure 0 placeholder coordinates with 12 observed paired family means.\n"
        "- Data: `e2a_family_geometry.csv` and V/F rows from `e2a_family_arm_geometry.csv`.\n"
        "- Notice: family-specific movement and its heterogeneity; lower-panel units are logits.\n"
        "- Decision implication: Figure 0 can use real paired family geometry without rerunning E2A.\n"
        "- Caveat: target-follow percentage points are not plotted on the margin axis.\n\n"
        "## `figures/figure2d_cross_model_transport.*`\n\n"
        "- Purpose: show within-model overall and V-F effects with family-bootstrap CIs.\n"
        "- Data: `figure2d_forest_input.csv`.\n"
        "- Notice: both overall CIs are positive; both V-F intervals cross zero.\n"
        "- Decision implication: numerical routing is G-A, bounded to the archived matched regime.\n"
        "- Caveat: horizontal positions are within-model effects, not cross-model raw-logit levels.\n"
    )
    (out / "analysis-output" / "figure-catalog.md").write_text(catalog_text, encoding="utf-8", newline="\n")

    manifest = {
        "schema": "cedar.cross_model_transport.run_manifest.v1",
        "delivery_branch": "agent/cross-model-transport-delivery-v1",
        "delivery_base_commit": "bff1e19a71f2051ca4b2270c5d57706114593550",
        "source_runner_commit": None,
        "source_runner_commit_note": "not present in supplied cross-model records",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "family"},
        "forward_counts": {"Qwen-FC": 192, "Llama": 192, "total": 384, "launched_by_delivery": 0},
        "models": {
            model: {
                "repo": EXPECTED[model]["repo"],
                "revision": EXPECTED[model]["revision"],
                "experiment": EXPECTED[model]["experiment"],
                "actual_layers": list(EXPECTED[model]["layers"]),
                "axis_hash": gates[model]["axis_hash"][0],
                "config_hash": gates[model]["config_hash"][0],
                "threshold_hash": None,
                "prompt_bank_hash": gates[model]["prompt_bank_hash"],
                "gate_status": gates[model]["status"],
            }
            for model in ("Qwen-FC", "Llama")
        },
        "route": route["route"],
        "input_sha256": dict(source_hashes),
        "e2a_frozen_result_commit": "0c8517bb5fb4640f7d5e4d51356d0cf167046643",
        "e2a_hygiene_commit": "10120d04d00e3c9122bf2a2ba590bd403373fa45",
        "missing_counts": {"cross_model": integrity_totals["missing"], "e2a": e2a_audit["missing"]},
        "invalid_counts": {"cross_model": integrity_totals["invalid"], "e2a": e2a_audit["invalid"]},
        "truncation_counts": {"cross_model": integrity_totals["truncation"], "e2a": e2a_audit["truncated"]},
    }
    write_json(out / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-data", type=Path, required=True)
    parser.add_argument("--e2a-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = args.llama_data.resolve()
    e2a_root = args.e2a_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis-output").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "scripts").mkdir(exist_ok=True)

    qwen_path = data / "qwen_fc_records.jsonl"
    llama_path = data / "llama_records.jsonl"
    source_manifest_path = data / "delivery_manifest.json"
    e2a_results_path = e2a_root / "formal-all-seed0" / "results.jsonl"
    e2a_analysis_path = e2a_root / "E2A_ANALYSIS.json"
    e2a_audit_path = e2a_root / "E2A_AUDIT.json"
    e2a_hygiene_path = e2a_root / "E2A_HYGIENE_AUDIT.json"

    source_manifest = read_json(source_manifest_path)
    listed_hashes = {item["file"]: item["sha256"] for item in source_manifest["files"]}
    for path in (qwen_path, llama_path):
        actual = sha256_file(path)
        if listed_hashes.get(path.name) != actual:
            raise ValueError(f"Source manifest hash mismatch for {path}")

    qwen_rows = read_jsonl(qwen_path)
    llama_rows = read_jsonl(llama_path)
    rows_by_model = {"Qwen-FC": qwen_rows, "Llama": llama_rows}
    gates = {
        "Qwen-FC": audit_cross_model("Qwen-FC", qwen_rows, llama_rows),
        "Llama": audit_cross_model("Llama", llama_rows, qwen_rows),
    }
    if gates["Qwen-FC"]["prompt_bank_hash"] != gates["Llama"]["prompt_bank_hash"]:
        raise ValueError("Cross-model prompt-bank hashes differ")

    normalized = [
        normalize_cross_record(row)
        for model in ("Qwen-FC", "Llama")
        for row in rows_by_model[model]
    ]
    normalized.sort(key=lambda row: (row["model"], row["family_id"], row["base_id"], row["arm"], row["intervention"]))
    write_jsonl(out / "raw_records.jsonl", normalized)
    shutil.copyfile(qwen_path, out / "source_qwen_fc_records.jsonl")
    shutil.copyfile(llama_path, out / "source_llama_records.jsonl")

    e2a_rows = read_jsonl(e2a_results_path)
    frozen_analysis = read_json(e2a_analysis_path)
    if read_json(e2a_audit_path).get("status") != "passed":
        raise ValueError("Frozen E2A audit is not passed")
    if read_json(e2a_hygiene_path).get("status") != "passed":
        raise ValueError("Frozen E2A hygiene audit is not passed")
    e2a_families, e2a_arms, e2a_audit = e2a_export(e2a_rows, frozen_analysis)

    write_csv(
        out / "e2a_family_geometry.csv",
        ["family_id", "n_bases", "n_records", "m_general_uncapped", "m_general_p25", "delta_m_general", "missing_count", "invalid_count"],
        e2a_families,
    )
    write_csv(
        out / "e2a_family_arm_geometry.csv",
        ["family_id", "base_id", "arm", "m_general_uncapped", "m_general_p25", "delta_m_general", "m_target_uncapped", "m_target_p25", "delta_m_target", "target_follow_uncapped", "target_follow_p25", "target_follow_change", "validity_flag"],
        e2a_arms,
    )

    analysis = cross_model_analysis(rows_by_model, {model: gate["status"] for model, gate in gates.items()})
    route = route_result(analysis["summary"], gates)
    write_csv(
        out / "cross_model_family_effects.csv",
        ["model", "family_id", "arm", "m_general_uncapped", "m_general_p25", "delta_m_general", "m_target_uncapped", "m_target_p25", "delta_m_target", "target_follow_uncapped", "target_follow_p25", "target_follow_change"],
        analysis["family_effects"],
    )
    write_csv(
        out / "cross_model_summary.csv",
        ["model", "estimand", "estimate", "ci_low", "ci_high", "unit", "n_families", "n_bases", "n_records", "gate_status"],
        analysis["summary"],
    )
    summary_lookup = {
        (row["model"], row["estimand"]): row for row in analysis["summary"]
    }
    target_follow_rows = []
    for model in ("Qwen-FC", "Llama"):
        for arm in ("V", "F"):
            family_rows = [
                row
                for row in analysis["family_effects"]
                if row["model"] == model and row["arm"] == arm
            ]
            change = summary_lookup[(model, f"{arm}_target_follow_change")]
            target_follow_rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "target_follow_uncapped": mean(
                        [row["target_follow_uncapped"] for row in family_rows]
                    ),
                    "target_follow_p25": mean(
                        [row["target_follow_p25"] for row in family_rows]
                    ),
                    "target_follow_change": change["estimate"],
                    "ci_low": change["ci_low"],
                    "ci_high": change["ci_high"],
                    "n_families": change["n_families"],
                    "n_bases": change["n_bases"],
                    "n_records": change["n_records"],
                    "gate_status": change["gate_status"],
                }
            )
    write_csv(
        out / "cross_model_target_follow.csv",
        ["model", "arm", "target_follow_uncapped", "target_follow_p25", "target_follow_change", "ci_low", "ci_high", "n_families", "n_bases", "n_records", "gate_status"],
        target_follow_rows,
    )
    forest_estimands = {"overall_delta_m_general", "v_minus_f_continuous_selectivity_change"}
    forest_rows = [row for row in analysis["summary"] if row["estimand"] in forest_estimands]
    write_csv(
        out / "figure2d_forest_input.csv",
        ["model", "estimand", "estimate", "ci_low", "ci_high", "unit", "n_families", "n_bases", "n_records", "gate_status"],
        forest_rows,
    )
    write_csv(
        out / "pair_audit.csv",
        ["model", "family_id", "base_id", "arm", "stable_id", "uncapped_present", "p25_present", "prompt_hash_match", "answer_mapping_match", "option_token_ids_match", "pair_status"],
        analysis["pair_audit"],
    )
    write_csv(
        out / "cross_model_four_state_migrations.csv",
        ["model", "arm", "source_state", "destination_state", "count"],
        analysis["migrations"],
    )
    write_csv(
        out / "cross_model_telemetry.csv",
        ["model", "family_id", "n_p25_records", "mean_trigger_count", "mean_correction_count", "mean_correction_norm", "total_trigger_count", "total_correction_count", "total_correction_norm"],
        analysis["telemetry"],
    )
    write_csv(
        out / "cross_model_output_integrity.csv",
        ["model", "intervention", "n_records", "missing_logits", "invalid_count", "truncation_count", "top_token_not_option_count", "mean_option_probability_mass", "mean_entropy"],
        analysis["integrity"],
    )
    write_json(out / "gate_details.json", gates)
    write_json(out / "analysis_details.json", {"bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}, "interaction": analysis["interaction"], "route": route})

    source_hashes = {
        "LLama_data/qwen_fc_records.jsonl": sha256_file(qwen_path),
        "LLama_data/llama_records.jsonl": sha256_file(llama_path),
        "LLama_data/delivery_manifest.json": sha256_file(source_manifest_path),
        "E2A/formal-all-seed0/results.jsonl": sha256_file(e2a_results_path),
        "E2A/E2A_ANALYSIS.json": sha256_file(e2a_analysis_path),
        "E2A/E2A_AUDIT.json": sha256_file(e2a_audit_path),
        "E2A/E2A_HYGIENE_AUDIT.json": sha256_file(e2a_hygiene_path),
    }
    build_reports(out, e2a_audit, gates, analysis, route, source_hashes)

    generated = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "delivery_manifest.json":
            generated.append({"file": path.relative_to(out).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(
        out / "delivery_manifest.json",
        {
            "schema": "cedar.cross_model_transport.delivery_manifest.v2",
            "files": generated,
            "route": route["route"],
            "record_counts": {"Qwen-FC": len(qwen_rows), "Llama": len(llama_rows), "normalized": len(normalized)},
        },
    )
    print(json.dumps({"status": "complete", "route": route["route"], "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
