#!/usr/bin/env python3
# ruff: noqa: E402
"""Frozen family-clustered analysis for E2 new-salt confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "e2_new_salt_confirmation"
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from e2_new_salt_run_common import (
    ARMS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONDITIONS,
    FORMAL_RESULT_ROWS,
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
)

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "e2_new_salt_confirmation"


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "sd": stdev(values) if len(values) > 1 else None,
        "median": median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def logits(row: Mapping[str, Any]) -> Mapping[str, float] | None:
    payload = row.get("answer_logits")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("logits"), Mapping):
        return None
    return payload["logits"]


def metric(row: Mapping[str, Any], name: str) -> float | None:
    values = logits(row)
    if values is None:
        return None
    initial = str(row["initial_answer"])
    target = str(row["target_answer"])
    background = [label for label in "ABCD" if label not in {initial, target}]
    initial_logit = float(values[initial])
    target_logit = float(values[target])
    background_mean = mean(float(values[label]) for label in background)
    if name == "m_general":
        return mean(float(values[label]) for label in "ABCD" if label != initial) - initial_logit
    if name == "m_target":
        return target_logit - initial_logit
    if name == "m_background":
        return background_mean - initial_logit
    if name == "m_specific":
        return target_logit - background_mean
    raise ValueError(f"Unknown metric: {name}")


def target_follow(row: Mapping[str, Any]) -> int:
    return int(row.get("outcome_class") == "induced_target")


def invalid(row: Mapping[str, Any]) -> int:
    return int(row.get("outcome_class") == "invalid_or_abstain")


def pair_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index = {(row["id"], row["condition"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate result cells")
    return index


def paired_records(
    rows: Sequence[dict[str, Any]],
    value: Callable[[Mapping[str, Any]], float | None],
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[list[tuple[str, float]], list[str]]:
    index = pair_index(rows)
    source = [
        row
        for row in rows
        if row["condition"] == "uncapped" and (predicate is None or predicate(row))
    ]
    records = []
    missing = []
    for left in source:
        right = index[(left["id"], "full_p25")]
        left_value = value(left)
        right_value = value(right)
        if left_value is None or right_value is None:
            missing.append(left["id"])
        else:
            records.append((left["family_id"], float(right_value) - float(left_value)))
    return records, missing


def bootstrap_summary(
    records: Sequence[tuple[str, float]], *, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for family, value in records:
        by_family[family].append(value)
    family_scores = {family: mean(values) for family, values in sorted(by_family.items())}
    families = sorted(family_scores)
    if not families:
        return {
            "estimate": None,
            "family_cluster_bootstrap_95ci": [None, None],
            "families": 0,
            "records": 0,
        }
    estimate = mean(family_scores.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [rng.choice(families) for _ in families]
        draws.append(mean(family_scores[family] for family in sampled))
    draws.sort()
    lower = draws[int(0.025 * BOOTSTRAP_DRAWS)]
    upper = draws[int(0.975 * BOOTSTRAP_DRAWS) - 1]
    return {
        "estimate": estimate,
        "family_cluster_bootstrap_95ci": [lower, upper],
        "families": len(families),
        "records": len(records),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": seed,
        "positive_families": sum(value > 0 for value in family_scores.values()),
        "negative_families": sum(value < 0 for value in family_scores.values()),
        "tied_families": sum(value == 0 for value in family_scores.values()),
        "family_scores": family_scores,
    }


def paired_metric_summary(
    rows: Sequence[dict[str, Any]],
    name: str,
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    records, missing = paired_records(rows, lambda row: metric(row, name), predicate)
    result = bootstrap_summary(records)
    result.update(
        {
            "metric": name,
            "definition": "full_p25 minus uncapped; prompt-paired and family-clustered",
            "missing_pair_ids": missing,
            "complete_pairs": len(records),
        }
    )
    return result


def condition_summary(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    return {
        "rows": len(selected),
        "outcome_counts": dict(sorted(Counter(row["outcome_class"] for row in selected).items())),
        "target_rate": mean(target_follow(row) for row in selected),
        "invalid_rate": mean(invalid(row) for row in selected),
        "truncation_rate": mean(bool(row.get("truncated")) for row in selected),
        "parse_failure_rate": mean(
            row.get("final_answer_parse_status") != "passed" for row in selected
        ),
        "m_general": descriptive(
            [value for row in selected if (value := metric(row, "m_general")) is not None]
        ),
        "m_target": descriptive(
            [value for row in selected if (value := metric(row, "m_target")) is not None]
        ),
        "m_background": descriptive(
            [value for row in selected if (value := metric(row, "m_background")) is not None]
        ),
        "m_specific": descriptive(
            [value for row in selected if (value := metric(row, "m_specific")) is not None]
        ),
        "option_mass": descriptive(
            [
                float(row["answer_logits"]["option_mass"])
                for row in selected
                if isinstance(row.get("answer_logits"), Mapping)
            ]
        ),
        "entropy": descriptive(
            [
                float(row["answer_logits"]["full_vocab_entropy"])
                for row in selected
                if isinstance(row.get("answer_logits"), Mapping)
            ]
        ),
        "reasoning_tokens": descriptive(
            [float(row["reasoning_token_count"]) for row in selected]
        ),
        "top_token_is_option_rate": mean(
            bool(row["answer_logits"]["top_token_is_option"])
            for row in selected
            if isinstance(row.get("answer_logits"), Mapping)
        ),
    }


def binary_effect(
    rows: Sequence[dict[str, Any]],
    indicator: Callable[[Mapping[str, Any]], int],
    predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    records, missing = paired_records(rows, lambda row: float(indicator(row)), predicate)
    result = bootstrap_summary(records)
    result.update({"missing_pair_ids": missing, "complete_pairs": len(records)})
    return result


def transitions(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    index = pair_index(rows)
    values = Counter()
    for left in rows:
        if left["condition"] != "uncapped":
            continue
        right = index[(left["id"], "full_p25")]
        values[f"{left['outcome_class']} -> {right['outcome_class']}"] += 1
    return dict(sorted(values.items()))


def leave_one_family_out(records: Sequence[tuple[str, float]]) -> dict[str, float]:
    families = sorted({family for family, _ in records})
    return {
        omitted: mean(value for family, value in records if family != omitted)
        for omitted in families
    }


def primary_by_stratum(
    rows: Sequence[dict[str, Any]], bases: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, Any]:
    values = sorted({str(base[field]) for base in bases.values()})
    return {
        value: paired_metric_summary(
            rows,
            "m_general",
            lambda row, value=value: str(bases[row["base_id"]][field]) == value,
        )
        for value in values
    }


def cap_telemetry(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == "full_p25"]
    payloads = [row["cap_stats"]["pooled"] for row in selected]
    total = sum(int(item["total_positions"]) for item in payloads)
    triggered = sum(int(item["triggered_positions"]) for item in payloads)
    correction = sum(float(item["projection_correction_sum"]) for item in payloads)
    return {
        "rows": len(payloads),
        "total_positions": total,
        "triggered_positions": triggered,
        "trigger_rate": triggered / total if total else 0.0,
        "projection_correction_sum": correction,
        "mean_projection_correction": correction / total if total else 0.0,
    }


def decision(primary: Mapping[str, Any]) -> str:
    estimate = primary["estimate"]
    lower, upper = primary["family_cluster_bootstrap_95ci"]
    if primary["complete_pairs"] != 96:
        return "inconclusive_missing_primary_logits"
    if estimate > 0 and lower > 0:
        return "positive_confirmation"
    if estimate > 0 and lower <= 0 <= upper:
        return "directionally_consistent_not_confirmed"
    if estimate <= 0 and lower <= 0 <= upper:
        return "not_confirmed"
    if upper < 0:
        return "directional_contradiction"
    return "not_confirmed"


def write_per_prompt(
    output_root: Path,
    rows: Sequence[dict[str, Any]],
    bases: Mapping[str, Mapping[str, Any]],
) -> None:
    index = pair_index(rows)
    path = output_root / "per_prompt_results.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "base_id",
                "family_id",
                "arm",
                "variant",
                "semantic_f_class",
                "initial_answer",
                "target_answer",
                "uncapped_outcome",
                "full_p25_outcome",
                "uncapped_m_general",
                "full_p25_m_general",
                "delta_m_general",
                "uncapped_truncated",
                "full_p25_truncated",
            ],
        )
        writer.writeheader()
        for left in rows:
            if left["condition"] != "uncapped":
                continue
            right = index[(left["id"], "full_p25")]
            left_metric = metric(left, "m_general")
            right_metric = metric(right, "m_general")
            writer.writerow(
                {
                    "id": left["id"],
                    "base_id": left["base_id"],
                    "family_id": left["family_id"],
                    "arm": left["arm"],
                    "variant": left["variant"],
                    "semantic_f_class": bases[left["base_id"]]["error_class"],
                    "initial_answer": left["initial_answer"],
                    "target_answer": left["target_answer"],
                    "uncapped_outcome": left["outcome_class"],
                    "full_p25_outcome": right["outcome_class"],
                    "uncapped_m_general": left_metric,
                    "full_p25_m_general": right_metric,
                    "delta_m_general": (
                        right_metric - left_metric
                        if left_metric is not None and right_metric is not None
                        else None
                    ),
                    "uncapped_truncated": left["truncated"],
                    "full_p25_truncated": right["truncated"],
                }
            )


def analyze(output_root: Path) -> dict[str, Any]:
    audit = read_json(output_root / "FORMAL_AUDIT.json")
    if audit.get("status") != "passed":
        raise ValueError("Formal engineering audit must pass before analysis")
    results_path = output_root / "formal-paired-uncapped-full-p25-seed0" / "results.jsonl"
    rows = read_jsonl(results_path)
    if len(rows) != FORMAL_RESULT_ROWS:
        raise ValueError("Formal result matrix is incomplete")
    bases = {row["base_id"]: row for row in read_jsonl(DATA_ROOT / "bases24.jsonl")}
    primary_records, primary_missing = paired_records(
        rows, lambda row: metric(row, "m_general")
    )
    primary = bootstrap_summary(primary_records)
    primary.update(
        {
            "estimand": "Delta m_general = m_general(full_p25) - m_general(uncapped)",
            "complete_pairs": len(primary_records),
            "missing_pair_ids": primary_missing,
        }
    )
    metric_effects = {
        name: paired_metric_summary(rows, name)
        for name in ("m_target", "m_background", "m_specific")
    }
    arm_effects = {
        arm: paired_metric_summary(rows, "m_general", lambda row, arm=arm: row["arm"] == arm)
        for arm in ARMS
    }
    arm_summaries = {
        arm: {
            condition: condition_summary(
                [row for row in rows if row["arm"] == arm], condition
            )
            for condition in CONDITIONS
        }
        for arm in ARMS
    }
    v_target = binary_effect(rows, target_follow, lambda row: row["arm"] == "V")
    f_target = binary_effect(rows, target_follow, lambda row: row["arm"] == "F")
    vf_selectivity_records = []
    index = pair_index(rows)
    for base_id in sorted(bases):
        v_u = index[(f"{base_id}-V", "uncapped")]
        v_p = index[(f"{base_id}-V", "full_p25")]
        f_u = index[(f"{base_id}-F", "uncapped")]
        f_p = index[(f"{base_id}-F", "full_p25")]
        vf_selectivity_records.append(
            (
                v_u["family_id"],
                target_follow(v_p)
                - target_follow(v_u)
                - target_follow(f_p)
                + target_follow(f_u),
            )
        )
    common_update = {
        "V": v_target,
        "F": f_target,
        "mean_common_update": (v_target["estimate"] + f_target["estimate"]) / 2,
    }
    completion = {
        condition: {
            "truncations": [
                row["id"]
                for row in rows
                if row["condition"] == condition and row.get("truncated")
            ],
            "parse_failures": [
                row["id"]
                for row in rows
                if row["condition"] == condition
                and row.get("final_answer_parse_status") != "passed"
            ],
            "missing_logits": [
                row["id"]
                for row in rows
                if row["condition"] == condition and logits(row) is None
            ],
            "invalid_rate": mean(
                invalid(row) for row in rows if row["condition"] == condition
            ),
        }
        for condition in CONDITIONS
    }
    result = {
        "schema": "cedar.e2_new_salt_confirmation.analysis.v1",
        "status": "complete",
        "scope": "new_salt_unselected_instance_confirmation",
        "claim_boundary": (
            "same task grammar and family structures; not independent-family or "
            "external confirmation"
        ),
        "conditions": list(CONDITIONS),
        "seed": 0,
        "primary": primary,
        "decision_classification": decision(primary),
        "secondary_metric_effects": metric_effects,
        "arm_specific_m_general_effects": arm_effects,
        "arm_condition_summaries": arm_summaries,
        "target_follow_effects": {"V": v_target, "F": f_target},
        "common_update": common_update,
        "v_minus_f_selectivity_change": bootstrap_summary(vf_selectivity_records),
        "four_state_transitions": transitions(rows),
        "completion_integrity": completion,
        "stratified_primary": {
            "variant": primary_by_stratum(rows, bases, "variant"),
            "semantic_f_class": primary_by_stratum(rows, bases, "error_class"),
            "initial_answer": {
                label: paired_metric_summary(
                    rows,
                    "m_general",
                    lambda row, label=label: row["initial_answer"] == label,
                )
                for label in "ABCD"
            },
            "target_answer": {
                label: paired_metric_summary(
                    rows,
                    "m_general",
                    lambda row, label=label: row["target_answer"] == label,
                )
                for label in "ABCD"
            },
        },
        "leave_one_family_out_primary": leave_one_family_out(primary_records),
        "p25_telemetry": cap_telemetry(rows),
        "review_documentation_limitation": read_json(
            DATA_ROOT / "human_review_result.json"
        )["protocol_deviation"],
        "additional_runs_authorized": False,
    }
    write_json(output_root / "E2_NEW_SALT_ANALYSIS.json", result)
    write_per_prompt(output_root, rows, bases)
    lower, upper = primary["family_cluster_bootstrap_95ci"]
    summary = f"""# E2 new-salt confirmation summary

This is a new-salt, unselected instance confirmation drawn from the same task
grammar and family structures. It is not an independent-family or external
confirmation.

- Decision: `{result['decision_classification']}`
- Primary Delta m_general: `{primary['estimate']}`
- Family-bootstrap 95% CI: `[{lower}, {upper}]`
- Complete primary pairs: `{primary['complete_pairs']}/96`
- Review limitation: external expert review was user-attested, but the item-level
  workbook was not completed and numeric naturalness scores are unavailable.
- Additional salts, seeds, p1, patching, LoFa, and J-Lens were not run or authorized.
"""
    (output_root / "E2_NEW_SALT_ANALYSIS.md").write_text(
        summary, encoding="utf-8", newline="\n"
    )
    delivery_paths = [
        output_root / "SMOKE_AUDIT.json",
        output_root / "FORMAL_AUDIT.json",
        output_root / "E2_NEW_SALT_ANALYSIS.json",
        output_root / "E2_NEW_SALT_ANALYSIS.md",
        output_root / "per_prompt_results.csv",
        results_path,
    ]
    write_json(
        output_root / "DELIVERY_MANIFEST.json",
        {
            "schema": "cedar.e2_new_salt_confirmation.delivery.v1",
            "status": "complete",
            "files": {
                str(path.relative_to(output_root)): file_sha256(path)
                for path in delivery_paths
            },
            "additional_runs_authorized": False,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = analyze(args.output_root.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision_classification"],
                "primary": result["primary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
