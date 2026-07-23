#!/usr/bin/env python3
"""Analyze the two completed Step 5 uncapped development cells."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import atomic_write_json, file_sha256

try:
    from scripts import run_prover_generator_step5_uncapped as runner
except ImportError:
    import run_prover_generator_step5_uncapped as runner

MODES = ("thinking", "nonthinking")
OUTCOMES = ("initial", "induced_target", "other_wrong", "invalid_or_abstain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=runner.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def answer_margin(row: Mapping[str, Any]) -> float | None:
    payload = row.get("answer_logits")
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("target_minus_initial")
    return float(value) if value is not None else None


def arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["outcome_class"]) for row in rows)
    margins = [value for row in rows if (value := answer_margin(row)) is not None]
    option_mass = [
        float(row["answer_logits"]["option_mass"])
        for row in rows
        if isinstance(row.get("answer_logits"), Mapping)
    ]
    entropy = [
        float(row["answer_logits"]["full_vocab_entropy"])
        for row in rows
        if isinstance(row.get("answer_logits"), Mapping)
    ]
    target_conditional = [
        float(row["answer_logits"]["conditional_probabilities"][row["target_answer"]])
        for row in rows
        if isinstance(row.get("answer_logits"), Mapping)
    ]
    initial_conditional = [
        float(row["answer_logits"]["conditional_probabilities"][row["initial_answer"]])
        for row in rows
        if isinstance(row.get("answer_logits"), Mapping)
    ]
    return {
        "rows": len(rows),
        "outcome_counts": {label: counts.get(label, 0) for label in OUTCOMES},
        "outcome_rates": {
            label: rate(counts.get(label, 0), len(rows)) for label in OUTCOMES
        },
        "gold_correct_count": sum(bool(row.get("gold_correct")) for row in rows),
        "gold_accuracy": rate(
            sum(bool(row.get("gold_correct")) for row in rows), len(rows)
        ),
        "strict_parse_count": sum(
            row.get("final_answer_parse_status") == "passed" for row in rows
        ),
        "strict_parse_rate": rate(
            sum(row.get("final_answer_parse_status") == "passed" for row in rows),
            len(rows),
        ),
        "answer_margin_count": len(margins),
        "target_minus_initial_mean": mean_or_none(margins),
        "target_minus_initial_median": median_or_none(margins),
        "option_mass_mean": mean_or_none(option_mass),
        "full_vocab_entropy_mean": mean_or_none(entropy),
        "conditional_target_mass_mean": mean_or_none(target_conditional),
        "conditional_initial_mass_mean": mean_or_none(initial_conditional),
        "response_tokens_mean": mean_or_none(
            [float(row["response_tokens"]) for row in rows]
        ),
        "reasoning_tokens_mean": mean_or_none(
            [float(row["reasoning_token_count"]) for row in rows]
        ),
        "other_wrong_rate": rate(
            sum(bool(row.get("other_wrong_indicator")) for row in rows), len(rows)
        ),
        "invalid_or_abstain_rate": rate(
            sum(bool(row.get("invalid_or_abstain_indicator")) for row in rows),
            len(rows),
        ),
        "synchronized_other_error_rate": rate(
            sum(bool(row.get("synchronized_other_error")) for row in rows), len(rows)
        ),
    }


def indexed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    output = {(str(row["base_id"]), str(row["arm"])): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("Step 5 analysis requires unique base/arm rows")
    return output


def paired_contrast(
    rows: Sequence[Mapping[str, Any]], left_arm: str, right_arm: str
) -> dict[str, Any]:
    values = indexed(rows)
    bases = sorted({base for base, arm in values if arm == left_arm})
    discrete_target: list[float] = []
    discrete_initial: list[float] = []
    margin: list[float] = []
    choice_changed: list[float] = []
    for base in bases:
        left = values[(base, left_arm)]
        right = values[(base, right_arm)]
        discrete_target.append(
            float(left["outcome_class"] == "induced_target")
            - float(right["outcome_class"] == "induced_target")
        )
        discrete_initial.append(
            float(left["outcome_class"] == "initial")
            - float(right["outcome_class"] == "initial")
        )
        choice_changed.append(float(left.get("chosen_option") != right.get("chosen_option")))
        left_margin = answer_margin(left)
        right_margin = answer_margin(right)
        if left_margin is not None and right_margin is not None:
            margin.append(left_margin - right_margin)
    return {
        "left_minus_right": f"{left_arm}-{right_arm}",
        "paired_bases": len(bases),
        "induced_target_rate_difference": mean_or_none(discrete_target),
        "initial_rate_difference": mean_or_none(discrete_initial),
        "target_minus_initial_margin_difference": mean_or_none(margin),
        "margin_paired_bases": len(margin),
        "choice_changed_rate": mean_or_none(choice_changed),
    }


def cross_mode_contrast(
    thinking: Sequence[Mapping[str, Any]],
    nonthinking: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    left = {str(row["base_id"]): row for row in thinking if row["arm"] == arm}
    right = {str(row["base_id"]): row for row in nonthinking if row["arm"] == arm}
    if set(left) != set(right):
        raise ValueError(f"Cross-mode {arm} base identity differs")
    target_differences = []
    initial_differences = []
    margin_differences = []
    choice_changes = []
    for base in sorted(left):
        a, b = left[base], right[base]
        target_differences.append(
            float(a["outcome_class"] == "induced_target")
            - float(b["outcome_class"] == "induced_target")
        )
        initial_differences.append(
            float(a["outcome_class"] == "initial")
            - float(b["outcome_class"] == "initial")
        )
        choice_changes.append(float(a.get("chosen_option") != b.get("chosen_option")))
        a_margin, b_margin = answer_margin(a), answer_margin(b)
        if a_margin is not None and b_margin is not None:
            margin_differences.append(a_margin - b_margin)
    return {
        "thinking_minus_nonthinking": arm,
        "paired_bases": len(left),
        "induced_target_rate_difference": mean_or_none(target_differences),
        "initial_rate_difference": mean_or_none(initial_differences),
        "target_minus_initial_margin_difference": mean_or_none(margin_differences),
        "margin_paired_bases": len(margin_differences),
        "choice_changed_rate": mean_or_none(choice_changes),
    }


def calibration_flags(cells: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in MODES:
        arms = cells[mode]["arms"]
        v_target = arms["V"]["outcome_rates"]["induced_target"]
        f_target = arms["F"]["outcome_rates"]["induced_target"]
        s_target = arms["S"]["outcome_rates"]["induced_target"]
        n_initial = arms["N"]["outcome_rates"]["initial"]
        r_initial = arms["R"]["outcome_rates"]["initial"]
        margin_complete = {
            arm: arms[arm]["answer_margin_count"] == arms[arm]["rows"]
            for arm in runner.ARMS
        }
        output[mode] = {
            "v_at_100pct_ceiling": v_target == 1.0,
            "f_at_0pct_floor": f_target == 0.0,
            "s_has_interior_dynamic_range": s_target is not None and 0.0 < s_target < 1.0,
            "n_initial_rate": n_initial,
            "r_initial_rate": r_initial,
            "n_r_initial_rate_absolute_difference": (
                abs(float(n_initial) - float(r_initial))
                if n_initial is not None and r_initial is not None
                else None
            ),
            "answer_margin_complete_by_arm": margin_complete,
            "all_answer_margins_complete": all(margin_complete.values()),
        }
    output["thinking_f_and_s_both_at_floor"] = (
        output["thinking"]["f_at_0pct_floor"]
        and cells["thinking"]["arms"]["S"]["outcome_rates"]["induced_target"] == 0.0
    )
    output["automatic_p25_authorization"] = False
    output["p25_decision"] = "requires_human_review_after_uncapped_analysis"
    return output


def load_results(output_root: Path, mode: str) -> list[dict[str, Any]]:
    run_dir = output_root / runner.run_name("formal", mode)
    results_path = run_dir / "results.jsonl"
    runtime_path = run_dir / "runtime.json"
    if not results_path.is_file() or not runtime_path.is_file():
        raise ValueError(f"Missing formal {mode} artifacts")
    runtime = runner.read_json(runtime_path)
    if (
        runtime.get("complete_results") != 288
        or runtime.get("results_sha256") != file_sha256(results_path)
    ):
        raise ValueError(f"Formal {mode} runtime is incomplete or stale")
    rows = runner.read_jsonl(results_path)
    if len(rows) != 288:
        raise ValueError(f"Formal {mode} requires 288 rows")
    return rows


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# CEDAR Step 5 Uncapped Development Calibration",
        "",
        f"Status: {report['status']}",
        "",
        "This is development calibration, not confirmatory evidence and not p25 approval.",
        "",
        "## Mode summaries",
        "",
        "| mode | arm | initial | target | other | invalid | mean margin |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for arm in runner.ARMS:
            summary = report["cells"][mode]["arms"][arm]
            rates = summary["outcome_rates"]
            margin = summary["target_minus_initial_mean"]
            lines.append(
                "| {mode} | {arm} | {initial:.3f} | {target:.3f} | {other:.3f} | "
                "{invalid:.3f} | {margin} |".format(
                    mode=mode,
                    arm=arm,
                    initial=rates["initial"] or 0.0,
                    target=rates["induced_target"] or 0.0,
                    other=rates["other_wrong"] or 0.0,
                    invalid=rates["invalid_or_abstain"] or 0.0,
                    margin="NA" if margin is None else f"{margin:.4f}",
                )
            )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "The report never authorizes p25 automatically. Review the continuous and discrete "
            "flags together before any new intervention run.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    audit_path = args.output_root / "FORMAL_CROSS_MODE_AUDIT.json"
    if not audit_path.is_file() or runner.read_json(audit_path).get("status") != "passed":
        raise ValueError("Step 5 analysis requires a passed formal cross-mode audit")
    source_rows = runner.select_rows(runner.validate_inputs(), "formal")
    source_ids = [row["id"] for row in source_rows]
    results = {mode: load_results(args.output_root, mode) for mode in MODES}
    for mode in MODES:
        if [row["id"] for row in results[mode]] != source_ids:
            raise ValueError(f"Formal {mode} result order differs from frozen data")

    cells: dict[str, Any] = {}
    for mode in MODES:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results[mode]:
            grouped[str(row["arm"])].append(row)
        cells[mode] = {
            "rows": len(results[mode]),
            "arms": {arm: arm_summary(grouped[arm]) for arm in runner.ARMS},
            "within_mode_contrasts": {
                "valid_vs_invalid_revision": paired_contrast(results[mode], "V", "F"),
                "social_vs_answer_only": paired_contrast(results[mode], "S", "AO"),
                "recheck_vs_neutral": paired_contrast(results[mode], "R", "N"),
                "answer_exposure_vs_neutral": paired_contrast(results[mode], "AO", "N"),
            },
        }
    cross_mode = {
        arm: cross_mode_contrast(results["thinking"], results["nonthinking"], arm)
        for arm in runner.ARMS
    }
    report = {
        "schema": "cedar.prover_generator.step5.uncapped_analysis.v1",
        "status": "complete",
        "scope": "development calibration; no item selection; not confirmatory",
        "condition": "uncapped",
        "data_freeze_commit": runner.DATA_FREEZE_COMMIT,
        "formal_audit_path": str(audit_path),
        "formal_audit_sha256": file_sha256(audit_path),
        "cells": cells,
        "cross_mode_contrasts": cross_mode,
        "calibration_flags": calibration_flags(cells),
        "prohibited_inference": [
            "no automatic p25 approval",
            "no held-out confirmation claim",
            "no item deletion or rewriting from model outputs",
            "no mechanism claim",
        ],
    }
    output = args.report or args.output_root / "STEP5_UNCAPPED_ANALYSIS.json"
    markdown = args.markdown or args.output_root / "STEP5_UNCAPPED_ANALYSIS.md"
    atomic_write_json(output, report)
    write_markdown(markdown, report)
    print(
        json.dumps(
            {"status": report["status"], "report": str(output), "markdown": str(markdown)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
