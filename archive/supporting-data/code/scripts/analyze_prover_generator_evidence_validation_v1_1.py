#!/usr/bin/env python3
"""Strict paired ITT analysis for the Evidence-only uncapped/p25 validation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_evidence_validation_v1_1"
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from evidence_validation_common import CONDITIONS, EVIDENCE_ARMS, read_json

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "prover_generator_evidence_validation_v1_1"
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20260724


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def target(row: Mapping[str, Any]) -> int:
    return int(row.get("outcome_class") == "induced_target")


def initial(row: Mapping[str, Any]) -> int:
    return int(row.get("outcome_class") == "initial")


def invalid(row: Mapping[str, Any]) -> int:
    return int(row.get("outcome_class") == "invalid_or_abstain")


def truncated(row: Mapping[str, Any]) -> int:
    return int(bool(row.get("truncated")))


def margin(row: Mapping[str, Any]) -> float | None:
    payload = row.get("answer_logits")
    return float(payload["target_minus_initial"]) if isinstance(payload, Mapping) else None


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "sd": stdev(values) if len(values) > 1 else None,
        "median": median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def arm_summary(rows: Sequence[dict[str, Any]], arm: str, condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm and row["condition"] == condition]
    margins = [value for row in selected if (value := margin(row)) is not None]
    masses = [float(row["answer_logits"]["option_mass"]) for row in selected if isinstance(row.get("answer_logits"), Mapping)]
    entropies = [float(row["answer_logits"]["full_vocab_entropy"]) for row in selected if isinstance(row.get("answer_logits"), Mapping)]
    return {
        "rows": len(selected),
        "outcome_counts": dict(sorted(Counter(row["outcome_class"] for row in selected).items())),
        "target_rate_itt": sum(map(target, selected)) / len(selected),
        "initial_rate_itt": sum(map(initial, selected)) / len(selected),
        "other_wrong_rate_itt": sum(row["outcome_class"] == "other_wrong" for row in selected) / len(selected),
        "invalid_rate_itt": sum(map(invalid, selected)) / len(selected),
        "truncation_rate": sum(map(truncated, selected)) / len(selected),
        "strict_parse_rate": sum(row.get("final_answer_parse_status") == "passed" for row in selected) / len(selected),
        "target_minus_initial_margin": descriptive(margins),
        "option_mass": descriptive(masses),
        "full_vocab_entropy": descriptive(entropies),
    }


def paired_by_id(rows: Sequence[dict[str, Any]], arm: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected = [row for row in rows if row["arm"] == arm]
    index = {(row["id"], row["condition"]): row for row in selected}
    ids = sorted({row["id"] for row in selected})
    pairs = [(index[(item, "uncapped")], index[(item, "p25")]) for item in ids]
    if len(pairs) != 24:
        raise ValueError(f"{arm}: expected 24 complete condition pairs")
    return pairs


def family_cluster_bootstrap(records: Sequence[tuple[str, float]]) -> list[float]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for family, value in records:
        by_family[family].append(value)
    families = sorted(by_family)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [rng.choice(families) for _ in families]
        values = [value for family in sampled for value in by_family[family]]
        draws.append(mean(values))
    draws.sort()
    return draws


def family_cluster_summary(records: Sequence[tuple[str, float]]) -> dict[str, Any]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for family, value in records:
        by_family[family].append(value)
    scores = {family: mean(values) for family, values in sorted(by_family.items())}
    draws = family_cluster_bootstrap(records)
    return {
        "estimate": mean(value for _, value in records),
        "family_cluster_bootstrap_95ci": [draws[500], draws[19499]],
        "positive_families": sum(value > 0 for value in scores.values()),
        "negative_families": sum(value < 0 for value in scores.values()),
        "tied_families": sum(value == 0 for value in scores.values()),
        "family_scores": scores,
    }


def exact_mcnemar(positive: int, negative: int) -> float:
    discordant = positive + negative
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(positive, negative) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def family_signflip_p(records: Sequence[tuple[str, float]]) -> float:
    by_family: dict[str, list[float]] = defaultdict(list)
    for family, value in records:
        by_family[family].append(value)
    scores = [mean(values) for _, values in sorted(by_family.items())]
    observed = abs(mean(scores))
    extreme = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(scores)):
        total += 1
        if abs(mean(sign * value for sign, value in zip(signs, scores, strict=True))) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def binary_condition_contrast(rows: Sequence[dict[str, Any]], arm: str, indicator: Callable[[Mapping[str, Any]], int]) -> dict[str, Any]:
    pairs = paired_by_id(rows, arm)
    records = [(left["family_id"], indicator(right) - indicator(left)) for left, right in pairs]
    transitions = Counter(f"{indicator(left)}->{indicator(right)}" for left, right in pairs)
    positive = sum(value > 0 for _, value in records)
    negative = sum(value < 0 for _, value in records)
    result = family_cluster_summary(records)
    result.update(
        {
            "arm": arm,
            "paired_bases": len(pairs),
            "definition": "p25 minus uncapped",
            "transitions_uncapped_to_p25": dict(sorted(transitions.items())),
            "positive_bases": positive,
            "negative_bases": negative,
            "tied_bases": len(records) - positive - negative,
            "exact_mcnemar_two_sided_p": exact_mcnemar(positive, negative),
            "family_signflip_two_sided_p": family_signflip_p(records),
        }
    )
    return result


def continuous_condition_contrast(rows: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    pairs = paired_by_id(rows, arm)
    complete = [(left, right) for left, right in pairs if margin(left) is not None and margin(right) is not None]
    records = [(left["family_id"], float(margin(right)) - float(margin(left))) for left, right in complete]
    if not records:
        return {"arm": arm, "paired_bases": 0, "estimate": None}
    result = family_cluster_summary(records)
    result.update({"arm": arm, "paired_bases": len(records), "definition": "p25 minus uncapped target-minus-initial margin", "family_signflip_two_sided_p": family_signflip_p(records)})
    return result


def discrimination_records(rows: Sequence[dict[str, Any]], indicator: Callable[[Mapping[str, Any]], int]) -> list[tuple[str, float]]:
    index = {(row["base_id"], row["arm"], row["condition"]): row for row in rows}
    bases = sorted({row["base_id"] for row in rows})
    records = []
    for base in bases:
        cells = {key: index[(base, key[0], key[1])] for key in (("V", "uncapped"), ("F", "uncapped"), ("V", "p25"), ("F", "p25"))}
        value = (
            indicator(cells[("V", "p25")])
            - indicator(cells[("F", "p25")])
            - indicator(cells[("V", "uncapped")])
            + indicator(cells[("F", "uncapped")])
        )
        records.append((cells[("V", "uncapped")]["family_id"], value))
    return records


def discrimination_contrast(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = discrimination_records(rows, target)
    result = family_cluster_summary(records)
    result.update(
        {
            "definition": "(V-F target following under p25) minus (V-F target following under uncapped)",
            "paired_bases": len(records),
            "family_signflip_two_sided_p": family_signflip_p(records),
        }
    )
    return result


def margin_discrimination_contrast(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    index = {(row["base_id"], row["arm"], row["condition"]): row for row in rows}
    records: list[tuple[str, float]] = []
    for base in sorted({row["base_id"] for row in rows}):
        cells = [
            index[(base, arm, condition)]
            for arm, condition in (("V", "uncapped"), ("F", "uncapped"), ("V", "p25"), ("F", "p25"))
        ]
        values = [margin(row) for row in cells]
        if any(value is None for value in values):
            continue
        value = float(values[2]) - float(values[3]) - float(values[0]) + float(values[1])
        records.append((cells[0]["family_id"], value))
    if not records:
        return {"paired_bases": 0, "estimate": None}
    result = family_cluster_summary(records)
    result.update(
        {
            "definition": "p25-minus-uncapped change in the continuous V-minus-F target-minus-initial margin",
            "paired_bases": len(records),
            "family_signflip_two_sided_p": family_signflip_p(records),
        }
    )
    return result


def rate_bounds(rows: Sequence[dict[str, Any]], arm: str, condition: str) -> list[float]:
    selected = [row for row in rows if row["arm"] == arm and row["condition"] == condition]
    observed = sum(map(target, selected))
    uncertain = sum(map(truncated, selected))
    return [observed / len(selected), (observed + uncertain) / len(selected)]


def subtract_bounds(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [left[0] - right[1], left[1] - right[0]]


def sensitivity_bounds(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    arm_condition = {arm: {condition: rate_bounds(rows, arm, condition) for condition in CONDITIONS} for arm in EVIDENCE_ARMS}
    cap_effects = {arm: subtract_bounds(arm_condition[arm]["p25"], arm_condition[arm]["uncapped"]) for arm in EVIDENCE_ARMS}
    discrimination = {}
    for condition in CONDITIONS:
        discrimination[condition] = subtract_bounds(arm_condition["V"][condition], arm_condition["F"][condition])
    return {
        "interpretation": "lower/upper target-following bounds treat every truncated response as non-target/target respectively; non-truncation invalids remain non-target",
        "arm_condition_target_bounds": arm_condition,
        "p25_minus_uncapped_bounds": cap_effects,
        "v_minus_f_bounds": discrimination,
        "discrimination_change_bounds": subtract_bounds(discrimination["p25"], discrimination["uncapped"]),
    }


def stratified(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("variant", "position_transition"):
        values = sorted({str(row[field]) for row in rows})
        output[field] = {}
        for value in values:
            subset = [row for row in rows if str(row[field]) == value]
            output[field][value] = {
                arm: {
                    condition: {
                        "n": sum(row["arm"] == arm and row["condition"] == condition for row in subset),
                        "target_rate": mean(target(row) for row in subset if row["arm"] == arm and row["condition"] == condition),
                        "invalid_rate": mean(invalid(row) for row in subset if row["arm"] == arm and row["condition"] == condition),
                    }
                    for condition in CONDITIONS
                }
                for arm in EVIDENCE_ARMS
            }
    return output


def cap_exposure(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        payloads = [row["cap_stats"]["pooled"] for row in selected if isinstance(row.get("cap_stats"), Mapping)]
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

    p25 = [row for row in rows if row["condition"] == "p25"]
    return {
        "definition": "actual p25 decoder-layer token positions pooled over official layers 46-53",
        "overall": summarize(p25),
        "by_arm": {arm: summarize([row for row in p25 if row["arm"] == arm]) for arm in EVIDENCE_ARMS},
    }


def outcome_label(v_delta: float, f_delta: float, invalid_delta: float) -> str:
    if f_delta > 0:
        label = "increased_false_argument_vulnerability"
    elif v_delta < 0 and f_delta < 0:
        label = "false_argument_reduction_with_valid_update_cost"
    elif v_delta == 0 and f_delta < 0:
        label = "selective_false_argument_reduction"
    elif v_delta == 0 and f_delta == 0:
        label = "no_observed_discrete_v_or_f_change"
    else:
        label = "mixed_or_asymmetric_effect"
    return f"{label}_plus_completion_cost" if invalid_delta > 0 else label


def analyze(output_root: Path) -> dict[str, Any]:
    audit = read_json(output_root / "FORMAL_AUDIT.json")
    if audit.get("status") != "passed":
        raise ValueError("Formal engineering audit must pass before analysis")
    rows = read_jsonl(output_root / "formal-paired-uncapped-p25-seed0" / "results.jsonl")
    summaries = {arm: {condition: arm_summary(rows, arm, condition) for condition in CONDITIONS} for arm in EVIDENCE_ARMS}
    target_effects = {arm: binary_condition_contrast(rows, arm, target) for arm in EVIDENCE_ARMS}
    initial_effects = {arm: binary_condition_contrast(rows, arm, initial) for arm in EVIDENCE_ARMS}
    invalid_effects = {arm: binary_condition_contrast(rows, arm, invalid) for arm in EVIDENCE_ARMS}
    margin_effects = {arm: continuous_condition_contrast(rows, arm) for arm in EVIDENCE_ARMS}
    discrimination = discrimination_contrast(rows)
    margin_discrimination = margin_discrimination_contrast(rows)
    raw_primary_p = {
        "V_target": target_effects["V"]["family_signflip_two_sided_p"],
        "F_target": target_effects["F"]["family_signflip_two_sided_p"],
        "V_minus_F_change": discrimination["family_signflip_two_sided_p"],
    }
    adjusted_primary_p = holm_adjust(raw_primary_p)
    truncation_by_condition = {
        condition: sum(row["condition"] == condition and row.get("truncated") for row in rows) / 96
        for condition in CONDITIONS
    }
    truncation_by_arm = {
        arm: {condition: summaries[arm][condition]["truncation_rate"] for condition in CONDITIONS}
        for arm in EVIDENCE_ARMS
    }
    overall_gap = truncation_by_condition["p25"] - truncation_by_condition["uncapped"]
    arm_gaps = {arm: values["p25"] - values["uncapped"] for arm, values in truncation_by_arm.items()}
    completion_warning = abs(overall_gap) > 0.05 or any(abs(value) > 0.05 for value in arm_gaps.values())
    total_invalid = {
        condition: sum(row["condition"] == condition and invalid(row) for row in rows) / 96
        for condition in CONDITIONS
    }
    v_delta = target_effects["V"]["estimate"]
    f_delta = target_effects["F"]["estimate"]
    invalid_delta = total_invalid["p25"] - total_invalid["uncapped"]
    result = {
        "schema": "cedar.prover_generator.evidence_validation_v1_1.analysis.v1",
        "status": "complete",
        "scope": "prespecified_internal_development_validation",
        "condition": "paired_uncapped_p25",
        "seed": 0,
        "unit_of_analysis": "24 bases paired across conditions, clustered in 12 prespecified families; one deterministic seed",
        "primary_estimand": "intention-to-treat discrete outcome with invalid/truncated retained",
        "arm_summaries": summaries,
        "target_following_cap_effects": target_effects,
        "initial_answer_cap_effects": initial_effects,
        "invalid_cap_effects": invalid_effects,
        "margin_cap_effects_complete_cases": margin_effects,
        "evidence_discrimination_change": discrimination,
        "margin_discrimination_change_complete_cases": margin_discrimination,
        "primary_family_signflip_p_values": {"raw": raw_primary_p, "holm_adjusted": adjusted_primary_p},
        "truncation": {
            "by_condition": truncation_by_condition,
            "p25_minus_uncapped_overall": overall_gap,
            "by_arm": truncation_by_arm,
            "p25_minus_uncapped_by_arm": arm_gaps,
            "completion_confound_warning": completion_warning,
            "warning_threshold": 0.05,
        },
        "sensitivity_bounds": sensitivity_bounds(rows),
        "stratified": stratified(rows),
        "actual_p25_cap_exposure": cap_exposure(rows),
        "decision_classification": outcome_label(v_delta, f_delta, invalid_delta),
        "data_product_status": "core_evidence_dataset_complete_after_result_commit",
        "social_module_status": "excluded_from_formal_prover_evaluation; use natural external social benchmarks",
        "new_salt_confirmation_status": "optional_submission_enhancement_not_run",
        "p25_authorized_for_additional_development_retesting": False,
        "p1_authorized": False,
        "limitations": [
            "The validation families were prespecified and untouched by model execution, but share structure classes with calibration families.",
            "One deterministic seed measures dataset-level paired variation, not run-to-run stochastic variation.",
            "Family-cluster intervals quantify uncertainty over the 12-family internal validation sample, not a universal task population.",
            "Any truncation imbalance must be reported jointly with behavioral effects.",
        ],
    }
    output_path = output_root / "EVIDENCE_VALIDATION_ANALYSIS.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = analyze(args.output_root.resolve())
    print(
        {
            "status": result["status"],
            "V_p25_minus_uncapped": result["target_following_cap_effects"]["V"]["estimate"],
            "F_p25_minus_uncapped": result["target_following_cap_effects"]["F"]["estimate"],
            "discrimination_change": result["evidence_discrimination_change"]["estimate"],
            "completion_confound_warning": result["truncation"]["completion_confound_warning"],
            "decision": result["decision_classification"],
        }
    )


if __name__ == "__main__":
    main()
