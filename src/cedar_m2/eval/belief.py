"""Preregistered behavior-level statistics for M2.2 Belief-R scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cedar_m2.runners.belief import STEPS


def _mean(values: Sequence[float | bool]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(np.mean(np.asarray(values, dtype=float)))


def _paired_bootstrap_lower_bound(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> float:
    """Return the one-sided 95% lower bound for an item-paired mean difference."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sequence")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.05))


def _index_records(
    records: Sequence[dict[str, Any]],
    conditions: Sequence[str],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, str]]:
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("M2.2 conditions must be non-empty and unique")
    condition_set = set(conditions)
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    targets: dict[str, str] = {}
    for record in records:
        sample_id = record.get("sample_id")
        condition = record.get("condition")
        step = record.get("step")
        target = record.get("target")
        if not isinstance(sample_id, str) or condition not in condition_set or step not in STEPS:
            raise ValueError("Malformed M2.2 score record key")
        if target not in {"BU", "BM"}:
            raise ValueError(f"Unknown M2.2 target {target!r}")
        if not isinstance(record.get("correct"), bool):
            raise ValueError("M2.2 score record must contain a boolean correct field")
        margin = record.get("margin")
        if (
            isinstance(margin, bool)
            or not isinstance(margin, int | float)
            or not np.isfinite(margin)
        ):
            raise ValueError("M2.2 score record must contain a numeric margin")
        key = (sample_id, condition, step)
        if key in indexed:
            raise ValueError(f"Duplicate M2.2 score record {key}")
        if sample_id in targets and targets[sample_id] != target:
            raise ValueError(f"Inconsistent M2.2 target for {sample_id}")
        indexed[key] = record
        targets[sample_id] = target

    expected = {
        (sample_id, condition, step)
        for sample_id in targets
        for condition in conditions
        for step in STEPS
    }
    if set(indexed) != expected:
        missing = len(expected - set(indexed))
        extra = len(set(indexed) - expected)
        raise ValueError(f"Incomplete M2.2 score matrix: missing={missing}, extra={extra}")
    return indexed, targets


def summarize_belief(
    records: Sequence[dict[str, Any]],
    conditions: Sequence[str],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260711,
) -> dict[str, Any]:
    """Compute M2.2 accuracies, paired changes, and preregistered gates."""
    if not records:
        raise ValueError("No M2.2 score records found")
    if "uncapped" not in conditions:
        raise ValueError("M2.2 summaries require an uncapped condition")
    indexed, targets = _index_records(records, conditions)
    sample_ids = sorted(targets)
    by_target = {
        target: [sample_id for sample_id in sample_ids if targets[sample_id] == target]
        for target in ("BU", "BM")
    }

    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        trajectory = {
            sample_id: bool(indexed[(sample_id, condition, "time_t")]["correct"])
            and bool(indexed[(sample_id, condition, "time_t1")]["correct"])
            for sample_id in sample_ids
        }
        target_accuracy: dict[str, float | None] = {}
        target_trajectory: dict[str, float | None] = {}
        target_margin: dict[str, float | None] = {}
        for target, ids in by_target.items():
            target_accuracy[target] = (
                _mean([indexed[(sample_id, condition, "time_t1")]["correct"] for sample_id in ids])
                if ids
                else None
            )
            target_trajectory[target] = (
                _mean([trajectory[sample_id] for sample_id in ids]) if ids else None
            )
            target_margin[target] = (
                _mean([indexed[(sample_id, condition, "time_t1")]["margin"] for sample_id in ids])
                if ids
                else None
            )
        bu_accuracy = target_accuracy["BU"]
        bm_accuracy = target_accuracy["BM"]
        condition_metrics[condition] = {
            "time_t_accuracy": _mean(
                [indexed[(sample_id, condition, "time_t")]["correct"] for sample_id in sample_ids]
            ),
            "time_t1_accuracy": target_accuracy,
            "breu": (
                float((bu_accuracy + bm_accuracy) / 2)
                if bu_accuracy is not None and bm_accuracy is not None
                else None
            ),
            "trajectory_accuracy": {
                "overall": _mean(list(trajectory.values())),
                **target_trajectory,
            },
            "time_t1_mean_margin": target_margin,
        }

    comparisons: dict[str, dict[str, Any]] = {}
    bu_ids = by_target["BU"]
    if bu_ids:
        uncapped_bu = condition_metrics["uncapped"]["time_t1_accuracy"]["BU"]
        assert uncapped_bu is not None
        for offset, condition in enumerate(conditions):
            if condition == "uncapped":
                continue
            capped_bu = condition_metrics[condition]["time_t1_accuracy"]["BU"]
            assert capped_bu is not None
            correctness_differences = [
                float(indexed[(sample_id, "uncapped", "time_t1")]["correct"])
                - float(indexed[(sample_id, condition, "time_t1")]["correct"])
                for sample_id in bu_ids
            ]
            margin_differences = [
                float(indexed[(sample_id, "uncapped", "time_t1")]["margin"])
                - float(indexed[(sample_id, condition, "time_t1")]["margin"])
                for sample_id in bu_ids
            ]
            comparisons[condition] = {
                "delta_bu": float(uncapped_bu - capped_bu),
                "delta_bu_one_sided_95ci_lower": _paired_bootstrap_lower_bound(
                    correctness_differences,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + offset,
                ),
                "bu_retention": float(capped_bu / uncapped_bu) if uncapped_bu > 0 else None,
                "mean_margin_drop_bu": _mean(margin_differences),
            }
            bm_ids = by_target["BM"]
            if bm_ids:
                uncapped_bm = condition_metrics["uncapped"]["time_t1_accuracy"]["BM"]
                capped_bm = condition_metrics[condition]["time_t1_accuracy"]["BM"]
                assert uncapped_bm is not None and capped_bm is not None
                comparisons[condition].update(
                    {
                        "delta_bm": float(uncapped_bm - capped_bm),
                        "mean_margin_drop_bm": _mean(
                            [
                                float(indexed[(sample_id, "uncapped", "time_t1")]["margin"])
                                - float(indexed[(sample_id, condition, "time_t1")]["margin"])
                                for sample_id in bm_ids
                            ]
                        ),
                    }
                )

    summary: dict[str, Any] = {
        "unit": "Belief-R item",
        "items": len(sample_ids),
        "target_counts": {target: len(ids) for target, ids in by_target.items()},
        "conditions": condition_metrics,
        "comparisons_to_uncapped": comparisons,
    }
    if tuple(conditions) == ("uncapped",):
        trajectory = condition_metrics["uncapped"]["trajectory_accuracy"]["overall"]
        bu_accuracy = condition_metrics["uncapped"]["time_t1_accuracy"]["BU"]
        has_bu_headroom = bu_accuracy is not None and bu_accuracy >= 0.10
        summary["development_gate"] = {
            "uncapped_trajectory_accuracy_at_least_50pct": trajectory >= 0.50,
            "uncapped_bu_accuracy_at_least_10pct": has_bu_headroom,
            "primary_delta_bu_is_identifiable": has_bu_headroom,
            "prompt_valid": trajectory >= 0.50 and has_bu_headroom,
        }
    elif "p25" in comparisons:
        p25 = comparisons["p25"]
        established = (
            p25["delta_bu"] >= 0.10 and p25["delta_bu_one_sided_95ci_lower"] > 0
        )
        if established:
            next_step = "cost_established"
        elif 0.05 <= p25["delta_bu"] < 0.10:
            next_step = "expand_bu"
        elif p25["delta_bu"] < 0.05:
            next_step = "evaluate_strongest_measured_cap"
        else:
            next_step = "p25_effect_uncertain"
        summary["pilot_gate"] = {
            "p25_delta_bu_at_least_10pp": p25["delta_bu"] >= 0.10,
            "p25_one_sided_ci_lower_above_zero": p25[
                "delta_bu_one_sided_95ci_lower"
            ]
            > 0,
            "cost_established": established,
            "next_step": next_step,
        }
    return summary
