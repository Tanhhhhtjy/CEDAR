#!/usr/bin/env python3
"""Classify five-dose responses and assemble a family-grouped enriched form."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_m2_2b_v5_dose_response_bank import (
    CAP_ORDER,
    DATASET_VERSION,
    HERE,
    OUTPUT,
    SOURCE_ARMS,
    atomic_write_json,
    atomic_write_jsonl,
    candidate_base_id,
    fixed_initial_user,
    load_jsonl,
)

PATTERNS_OUTPUT = HERE / "response_patterns468.jsonl"
SELECTION_OUTPUT = HERE / "selection_report.json"
SELECTED_OUTPUT = HERE / "m2_2b_v5_dose_response_selected180_arms.jsonl"
TARGET_V = (36, 32, 28, 24, 20)
TARGET_F = (18, 14, 10, 6, 2)
TARGET_S = (18, 14, 10, 6, 2)
BEAM_WIDTH = 12000
CLIFF_MIN_TRANSITIONS = 12
CLIFF_SHARE = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--allow-approximate", action="store_true")
    return parser.parse_args()


def parse_final_answer(response: str) -> str | None:
    text = response.rsplit("</think>", 1)[-1].strip()
    objects = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for raw in reversed(objects):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        answer = payload.get("answer")
        if answer in {"A", "B", "C"}:
            return str(answer)
    return None


def load_results(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            item_id = row.get("id")
            condition = row.get("cap") or row.get("condition")
            if not isinstance(item_id, str) or condition not in CAP_ORDER:
                raise ValueError(f"Malformed result row in {path}: {row}")
            key = (item_id, str(condition))
            if key in observed:
                raise ValueError(f"Duplicate result {key}")
            observed[key] = row
    return observed


def response_bit(candidate: dict[str, Any], answer: str) -> int:
    if candidate["arm"] == "V":
        return int(answer == candidate["updated_answer_id"])
    if candidate["arm"] == "N":
        return int(answer == candidate["initial_answer_id"])
    return int(answer == candidate["induced_target_answer_id"])


def monotone_nonincreasing(bits: tuple[int, ...]) -> bool:
    return all(left >= right for left, right in zip(bits, bits[1:], strict=False))


def cliff_diagnostic(patterns: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen pre-selection cliff rule to complete V/F/S patterns."""
    intervals = [
        f"{left}->{right}" for left, right in zip(CAP_ORDER, CAP_ORDER[1:], strict=False)
    ]
    by_arm: dict[str, dict[str, Any]] = {}
    qualified: list[dict[str, Any]] = []
    for arm in ("V", "F", "S"):
        counts = [0] * len(intervals)
        for row in patterns:
            if row["arm"] != arm or not row["complete"] or not row["monotone_nonincreasing"]:
                continue
            bits = [int(value) for value in row["response_bits"]]
            transitions = [
                index
                for index, (left, right) in enumerate(zip(bits, bits[1:], strict=False))
                if left == 1 and right == 0
            ]
            if len(transitions) == 1:
                counts[transitions[0]] += 1
        total = sum(counts)
        maximum = max(counts, default=0)
        share = maximum / total if total else 0.0
        interval_index = counts.index(maximum) if maximum else None
        arm_report = {
            "single_transition_records": total,
            "interval_counts": dict(zip(intervals, counts, strict=True)),
            "largest_interval_share": round(share, 6),
            "qualifies": total >= CLIFF_MIN_TRANSITIONS and share >= CLIFF_SHARE,
        }
        by_arm[arm] = arm_report
        if arm_report["qualifies"] and interval_index is not None:
            qualified.append(
                {
                    "arm": arm,
                    "interval_index": interval_index,
                    "interval": intervals[interval_index],
                    "transition_count": counts[interval_index],
                    "share": round(share, 6),
                }
            )
    arm_order = {arm: index for index, arm in enumerate(("V", "F", "S"))}
    chosen = min(
        qualified,
        key=lambda row: (-row["transition_count"], row["interval_index"], arm_order[row["arm"]]),
        default=None,
    )
    return {
        "status": "refinement_required" if chosen else "not_triggered",
        "minimum_single_transition_records": CLIFF_MIN_TRANSITIONS,
        "minimum_interval_share": CLIFF_SHARE,
        "by_arm": by_arm,
        "qualified_arm_intervals": qualified,
        "chosen_refinement_interval": chosen,
        "evaluated_before_level_selection": True,
    }


def classify(
    candidates: list[dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        answers: list[str | None] = []
        bits: list[int | None] = []
        for condition in CAP_ORDER:
            result = results.get((candidate["id"], condition))
            answer = (
                parse_final_answer(str(result.get("response", ""))) if result is not None else None
            )
            answers.append(answer)
            bits.append(response_bit(candidate, answer) if answer is not None else None)
        complete = all(bit is not None for bit in bits)
        bit_tuple = tuple(int(bit) for bit in bits) if complete else ()
        rows.append(
            {
                "id": candidate["id"],
                "source_base_id": candidate["source_base_id"],
                "candidate_base_id": candidate["candidate_base_id"],
                "family": candidate["template_family"],
                "template_variant": candidate["template_variant"],
                "arm": candidate["arm"],
                "gradient_level": candidate["gradient_level"],
                "gradient_level_name": candidate["gradient_level_name"],
                "conditions": list(CAP_ORDER),
                "answers": answers,
                "response_bits": list(bits),
                "pattern": "".join(str(bit) for bit in bit_tuple) if complete else None,
                "complete": complete,
                "monotone_nonincreasing": complete and monotone_nonincreasing(bit_tuple),
            }
        )
    return rows


def vector_sum(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def squared_distance(observed: tuple[int, ...], target: tuple[float, ...]) -> float:
    return sum((value - expected) ** 2 for value, expected in zip(observed, target, strict=True))


def family_choices(
    patterns: list[dict[str, Any]],
    *,
    axis: str,
) -> dict[str, dict[int, dict[str, tuple[int, ...]]]]:
    relevant_arms = {"V", "F"} if axis == "VF" else {"S"}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in patterns:
        if row["arm"] in relevant_arms:
            grouped[(row["family"], row["gradient_level"], row["arm"])].append(row)
    choices: dict[str, dict[int, dict[str, tuple[int, ...]]]] = defaultdict(dict)
    families = sorted({key[0] for key in grouped})
    for family in families:
        for level in range(4):
            payload: dict[str, tuple[int, ...]] = {}
            valid = True
            for arm in sorted(relevant_arms):
                rows = grouped.get((family, level, arm), [])
                if len(rows) != 2 or any(
                    not row["complete"] or not row["monotone_nonincreasing"] for row in rows
                ):
                    valid = False
                    break
                aggregate = tuple(
                    sum(int(row["response_bits"][index]) for row in rows)
                    for index in range(len(CAP_ORDER))
                )
                payload[arm] = aggregate
            if valid:
                choices[family][level] = payload
    return dict(choices)


def select_levels(
    choices: dict[str, dict[int, dict[str, tuple[int, ...]]]],
    *,
    targets: dict[str, tuple[int, ...]],
) -> tuple[dict[str, int], dict[str, tuple[int, ...]], float]:
    families = sorted(choices)
    if len(families) != 18:
        raise ValueError(f"Expected 18 choice families, found {len(families)}")
    for family in families:
        if not choices[family]:
            raise ValueError(f"No monotone candidate level for {family}")

    zero = tuple(0 for _ in CAP_ORDER)
    states: list[tuple[float, dict[str, tuple[int, ...]], dict[str, int]]] = [
        (0.0, {arm: zero for arm in targets}, {})
    ]
    for index, family in enumerate(families, start=1):
        expanded: dict[
            tuple[tuple[int, ...], ...], tuple[float, dict[str, tuple[int, ...]], dict[str, int]]
        ] = {}
        for _, totals, selected in states:
            for level, payload in sorted(choices[family].items()):
                new_totals = {arm: vector_sum(totals[arm], payload[arm]) for arm in targets}
                scaled_targets = {
                    arm: tuple(value * index / len(families) for value in targets[arm])
                    for arm in targets
                }
                score = sum(
                    squared_distance(new_totals[arm], scaled_targets[arm]) for arm in targets
                )
                new_selected = {**selected, family: level}
                key = tuple(new_totals[arm] for arm in sorted(targets))
                candidate_state = (score, new_totals, new_selected)
                incumbent = expanded.get(key)
                if incumbent is None or tuple(new_selected.items()) < tuple(incumbent[2].items()):
                    expanded[key] = candidate_state
        states = sorted(
            expanded.values(),
            key=lambda state: (state[0], tuple(state[2].items())),
        )[:BEAM_WIDTH]
    final = min(
        states,
        key=lambda state: (
            sum(squared_distance(state[1][arm], targets[arm]) for arm in targets),
            tuple(state[2].items()),
        ),
    )
    final_distance = sum(squared_distance(final[1][arm], targets[arm]) for arm in targets)
    return final[2], final[1], final_distance


def selected_source_row(source: dict[str, Any], arm: str) -> dict[str, Any]:
    row = copy.deepcopy(source)
    initial_user, repaired = fixed_initial_user(source["base_id"], source["initial_user"])
    row.update(
        {
            "dataset_version": DATASET_VERSION.replace("candidate-bank", "selected-form"),
            "split": "development_dose_response_enriched_selected_form",
            "status": "selected_from_complete_five-dose_development_scan",
            "id": f"{candidate_base_id(source['base_id'])}-{arm}",
            "candidate_base_id": candidate_base_id(source["base_id"]),
            "source_base_id": source["base_id"],
            "arm": arm,
            "initial_user": initial_user,
            "selection_status": "selected_family_group",
            "initial_task_ambiguity_repaired": repaired,
            "cap_scan_conditions": list(CAP_ORDER),
        }
    )
    return row


def assemble_selected(
    candidates: list[dict[str, Any]],
    vf_levels: dict[str, int],
    s_levels: dict[str, int],
) -> list[dict[str, Any]]:
    candidate_index = {
        (row["source_base_id"], row["arm"], row["gradient_level"]): row for row in candidates
    }
    source_grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(SOURCE_ARMS):
        source_grouped[row["base_id"]][row["arm"]] = row
    selected: list[dict[str, Any]] = []
    for source_base_id in sorted(source_grouped):
        family = source_grouped[source_base_id]["V"]["template_family"]
        n_source = copy.deepcopy(candidate_index[(source_base_id, "N", None)])
        n_source.update(
            {
                "dataset_version": DATASET_VERSION.replace("candidate-bank", "selected-form"),
                "split": "development_dose_response_enriched_selected_form",
                "status": "selected_from_complete_five-dose_development_scan",
                "source_candidate_id": n_source["id"],
                "selection_status": "selected_family_group_control_not_used_for_level_selection",
            }
        )
        selected.append(n_source)
        selected.append(selected_source_row(source_grouped[source_base_id]["R"], "R"))
        for arm, level in (
            ("V", vf_levels[family]),
            ("F", vf_levels[family]),
            ("S", s_levels[family]),
        ):
            source = copy.deepcopy(candidate_index[(source_base_id, arm, level)])
            source.update(
                {
                    "dataset_version": DATASET_VERSION.replace("candidate-bank", "selected-form"),
                    "split": "development_dose_response_enriched_selected_form",
                    "status": "selected_from_complete_five-dose_development_scan",
                    "source_candidate_id": source["id"],
                    "id": f"{candidate_base_id(source_base_id)}-{arm}",
                    "selection_status": "selected_family_group",
                }
            )
            selected.append(source)
    return selected


def main() -> None:
    args = parse_args()
    candidates = load_jsonl(OUTPUT)
    results = load_results(args.results)
    patterns = classify(candidates, results)
    atomic_write_jsonl(PATTERNS_OUTPUT, patterns)

    missing = [row["id"] for row in patterns if not row["complete"]]
    cliff = cliff_diagnostic(patterns)
    if not missing and cliff["status"] == "refinement_required":
        report = {
            "status": "cliff_refinement_required_before_selection",
            "scope": "response-enriched development challenge set; no levels selected",
            "conditions": list(CAP_ORDER),
            "candidate_rows": len(candidates),
            "expected_result_rows": len(candidates) * len(CAP_ORDER),
            "observed_result_rows": len(results),
            "missing_candidate_ids": missing,
            "cliff_diagnostic": cliff,
            "selection_performed": False,
            "next_step": "apply the frozen two-point rule in RUN_CONTRACT.md to all 468 records",
        }
        atomic_write_json(SELECTION_OUTPUT, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit("Predeclared cliff rule triggered; no candidate levels were selected.")
    vf_choices = family_choices(patterns, axis="VF")
    s_choices = family_choices(patterns, axis="S")
    vf_selected, vf_achieved, vf_distance = select_levels(
        vf_choices, targets={"V": TARGET_V, "F": TARGET_F}
    )
    s_selected, s_achieved, s_distance = select_levels(s_choices, targets={"S": TARGET_S})
    exact = vf_distance == 0 and s_distance == 0 and not missing
    n_patterns = [row for row in patterns if row["arm"] == "N"]
    n_correct = [
        sum(int(row["response_bits"][index]) for row in n_patterns if row["complete"])
        for index in range(len(CAP_ORDER))
    ]
    report = {
        "status": "exact_curve_selected" if exact else "candidate_scan_complete_curve_not_exact",
        "scope": "response-enriched development challenge set; descriptive on the scanned bank",
        "conditions": list(CAP_ORDER),
        "candidate_rows": len(candidates),
        "expected_result_rows": len(candidates) * len(CAP_ORDER),
        "observed_result_rows": len(results),
        "missing_candidate_ids": missing,
        "cliff_diagnostic": cliff,
        "target_counts": {"V": list(TARGET_V), "F": list(TARGET_F), "S": list(TARGET_S)},
        "achieved_counts": {
            "V": list(vf_achieved["V"]),
            "F": list(vf_achieved["F"]),
            "S": list(s_achieved["S"]),
        },
        "N_maintain_control": {
            "items": len(n_patterns),
            "correct_maintain_counts": n_correct,
            "unintended_change_or_invalid_counts": [36 - value for value in n_correct],
            "used_for_level_selection": False,
            "interpretation_limit": (
                "A flat N curve checks answer stability but cannot by itself rule out "
                "general answer inertia."
            ),
        },
        "squared_distance": {"VF": vf_distance, "S": s_distance},
        "selected_vf_level_by_family": vf_selected,
        "selected_s_level_by_family": s_selected,
        "selection_unit": "family; both parameter variants share one level",
        "nonmonotone_candidates_excluded": sum(
            row["arm"] in {"V", "F", "S"}
            and row["complete"]
            and not row["monotone_nonincreasing"]
            for row in patterns
        ),
        "raw_outputs_used_for_selection": True,
    }
    atomic_write_json(SELECTION_OUTPUT, report)
    if not exact and not args.allow_approximate:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(
            "Exact curve quotas were not available; inspect response_patterns468.jsonl "
            "and add candidates only for missing bins."
        )
    selected = assemble_selected(candidates, vf_selected, s_selected)
    atomic_write_jsonl(SELECTED_OUTPUT, selected)
    print(json.dumps({**report, "selected_rows": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
