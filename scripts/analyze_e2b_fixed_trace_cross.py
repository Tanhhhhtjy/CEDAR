#!/usr/bin/env python3
"""Frozen family-clustered analysis for E2B fixed-trace teacher forcing."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cross"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from e2b_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    TRACE_SOURCES,
    family_bootstrap,
    geometry,
    read_json,
    read_jsonl,
    require,
    write_json,
)

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/e2b_fixed_trace_cross"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _metric(row: Mapping[str, Any], name: str) -> float:
    if name in {"m_target", "m_background", "m_general", "m_specific"}:
        return geometry(row)[name]
    diagnostics = row["logit_diagnostics"]
    if name == "option_mass":
        return float(diagnostics["option_mass"])
    if name == "full_vocab_entropy":
        return float(diagnostics["full_vocab_entropy"])
    raise ValueError(f"Unknown metric: {name}")


def paired_family_scores(
    rows: Sequence[dict[str, Any]],
    *,
    trace_source: str,
    indicator: Callable[[Mapping[str, Any]], float],
    arm: str | None = None,
) -> dict[str, float]:
    selected = [
        row
        for row in rows
        if row["trace_source"] == trace_source and (arm is None or row["arm"] == arm)
    ]
    by_cell = {(row["source_id"], row["forward_condition"]): row for row in selected}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        if row["forward_condition"] != "uncapped":
            continue
        treated = by_cell[(row["source_id"], "full_p25")]
        grouped[row["family_id"]].append(indicator(treated) - indicator(row))
    scores = {family: statistics.fmean(values) for family, values in grouped.items()}
    require(len(scores) == 12, f"Expected 12 family scores, found {len(scores)}")
    return scores


def bootstrap_summary(scores: Mapping[str, float]) -> dict[str, Any]:
    return family_bootstrap(scores)


def combined_family_scores(
    left: Mapping[str, float], right: Mapping[str, float], operation: str
) -> dict[str, float]:
    require(set(left) == set(right), "Primary family sets differ")
    if operation == "mean":
        return {family: (left[family] + right[family]) / 2 for family in left}
    if operation == "difference":
        return {family: right[family] - left[family] for family in left}
    raise ValueError(operation)


def leave_one_family_out(scores: Mapping[str, float]) -> dict[str, float]:
    return {
        omitted: statistics.fmean(
            value for family, value in scores.items() if family != omitted
        )
        for omitted in sorted(scores)
    }


def vf_selectivity_scores(
    rows: Sequence[dict[str, Any]], *, trace_source: str
) -> dict[str, float]:
    by_arm = {
        arm: paired_family_scores(
            rows,
            trace_source=trace_source,
            indicator=lambda row: _metric(row, "m_target"),
            arm=arm,
        )
        for arm in ("V", "F")
    }
    return {
        family: by_arm["V"][family] - by_arm["F"][family]
        for family in by_arm["V"]
    }


def integrity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for trace_source in TRACE_SOURCES:
        for condition in ("uncapped", "full_p25"):
            selected = [
                row
                for row in rows
                if row["trace_source"] == trace_source
                and row["forward_condition"] == condition
            ]
            key = f"{trace_source}::{condition}"
            grouped[key] = {
                "rows": len(selected),
                "option_argmax": dict(
                    Counter(row["logit_diagnostics"]["option_argmax"] for row in selected)
                ),
                "top_token_is_option_rate": statistics.fmean(
                    float(row["logit_diagnostics"]["top_token_is_option"])
                    for row in selected
                ),
                "mean_option_mass": statistics.fmean(
                    float(row["logit_diagnostics"]["option_mass"]) for row in selected
                ),
                "mean_full_vocab_entropy": statistics.fmean(
                    float(row["logit_diagnostics"]["full_vocab_entropy"])
                    for row in selected
                ),
            }
    return grouped


def analyze(output_root: Path) -> dict[str, Any]:
    audit = read_json(output_root / "E2B_AUDIT.json")
    require(audit.get("status") == "passed", "E2B analysis requires a passed formal audit")
    rows = read_jsonl(output_root / "formal-fixed-trace-seed0/results.jsonl")
    require(len(rows) == 384, "E2B analysis requires 384 results")

    primary_scores = {
        source: paired_family_scores(
            rows,
            trace_source=source,
            indicator=lambda row: _metric(row, "m_general"),
        )
        for source in TRACE_SOURCES
    }
    primary = {
        "delta_general_U": bootstrap_summary(primary_scores["e2a_uncapped"]),
        "delta_general_P": bootstrap_summary(primary_scores["e2a_full_p25"]),
    }
    mean_scores = combined_family_scores(
        primary_scores["e2a_uncapped"], primary_scores["e2a_full_p25"], "mean"
    )
    interaction_scores = combined_family_scores(
        primary_scores["e2a_uncapped"], primary_scores["e2a_full_p25"], "difference"
    )

    metric_effects: dict[str, Any] = {}
    for metric in (
        "m_target",
        "m_background",
        "m_specific",
        "option_mass",
        "full_vocab_entropy",
    ):
        metric_effects[metric] = {
            source: bootstrap_summary(
                paired_family_scores(
                    rows,
                    trace_source=source,
                    indicator=lambda row, metric=metric: _metric(row, metric),
                )
            )
            for source in TRACE_SOURCES
        }

    arm_breakdowns: dict[str, Any] = {}
    for source in TRACE_SOURCES:
        arm_breakdowns[source] = {
            arm: bootstrap_summary(
                paired_family_scores(
                    rows,
                    trace_source=source,
                    indicator=lambda row: _metric(row, "m_general"),
                    arm=arm,
                )
            )
            for arm in ("N", "R", "V", "F")
        }

    vf = {
        source: {
            "V_target_minus_initial_cap_effect": bootstrap_summary(
                paired_family_scores(
                    rows,
                    trace_source=source,
                    indicator=lambda row: _metric(row, "m_target"),
                    arm="V",
                )
            ),
            "F_target_minus_initial_cap_effect": bootstrap_summary(
                paired_family_scores(
                    rows,
                    trace_source=source,
                    indicator=lambda row: _metric(row, "m_target"),
                    arm="F",
                )
            ),
            "V_minus_F_selectivity_change": bootstrap_summary(
                vf_selectivity_scores(rows, trace_source=source)
            ),
        }
        for source in TRACE_SOURCES
    }

    both_positive = all(
        summary["family_cluster_bootstrap_95ci"][0] > 0 for summary in primary.values()
    )
    both_include_zero = all(
        summary["family_cluster_bootstrap_95ci"][0] <= 0
        <= summary["family_cluster_bootstrap_95ci"][1]
        for summary in primary.values()
    )
    if both_positive:
        interpretation = "fixed_trace_positive_both_primary_intervals"
    elif both_include_zero:
        interpretation = "fixed_trace_effect_not_established"
    else:
        interpretation = "trace_source_dependent_or_mixed"

    analysis = {
        "schema": "cedar.e2b_fixed_trace_cross.analysis.v1",
        "status": "complete",
        "scope": "controlled_fixed_trace_diagnostic",
        "not_a_natural_direct_effect": True,
        "rows": len(rows),
        "families": 12,
        "bootstrap": {
            "unit": "family",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "interval": "two_sided_percentile_95ci",
        },
        "primary_estimands": primary,
        "primary_interpretation": interpretation,
        "secondary": {
            "mean_primary_contrast": bootstrap_summary(mean_scores),
            "trace_source_by_cap_interaction_P_minus_U": bootstrap_summary(
                interaction_scores
            ),
            "metric_effects": metric_effects,
            "arm_m_general_breakdowns": arm_breakdowns,
            "vf_target_and_selectivity": vf,
            "output_space_integrity": integrity(rows),
            "leave_one_family_out": {
                "delta_general_U": leave_one_family_out(primary_scores["e2a_uncapped"]),
                "delta_general_P": leave_one_family_out(primary_scores["e2a_full_p25"]),
            },
        },
        "claim_contract": {
            "allowed": [
                "conditional on the frozen generated trace",
                "controlled fixed-trace diagnostic",
            ],
            "forbidden": [
                "natural direct effect",
                "complete mediation",
                "effect occurs only at the answer layer",
                "final-layer circuit",
                "independent confirmation",
            ],
        },
    }
    write_json(output_root / "E2B_ANALYSIS.json", analysis)
    markdown = "\n".join(
        [
            "# E2B Fixed-Trace Cross Analysis",
            "",
            "This is a controlled fixed-trace diagnostic, not a natural direct-effect "
            "or confirmation analysis.",
            "",
            f"- delta_general_U: {primary['delta_general_U']['estimate']:.6f} "
            f"[{primary['delta_general_U']['family_cluster_bootstrap_95ci'][0]:.6f}, "
            f"{primary['delta_general_U']['family_cluster_bootstrap_95ci'][1]:.6f}]",
            f"- delta_general_P: {primary['delta_general_P']['estimate']:.6f} "
            f"[{primary['delta_general_P']['family_cluster_bootstrap_95ci'][0]:.6f}, "
            f"{primary['delta_general_P']['family_cluster_bootstrap_95ci'][1]:.6f}]",
            f"- interpretation: {interpretation}",
            "",
        ]
    )
    (output_root / "E2B_ANALYSIS.md").write_text(markdown, encoding="utf-8", newline="\n")
    return analysis


def main() -> None:
    args = parse_args()
    analysis = analyze(args.output_root.resolve())
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
