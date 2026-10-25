#!/usr/bin/env python3
"""Recompute the answer-head affine-contraction diagnostics from delivered records.

The analysis always uses the frozen ``initial_answer`` stored with an item.  It
never substitutes the uncapped option argmax.  This script performs CPU-only
post-processing and does not load a model or an activation vector.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

OPTIONS = ("A", "B", "C", "D")
ARMS = ("N", "R", "V", "F")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def option_vector(record: dict[str, Any], field: str) -> np.ndarray:
    logits = record[field]["logits"]
    return np.asarray([logits[o] for o in OPTIONS], dtype=float)


def affine_fit(
    uncapped: dict[str, Any],
    capped: dict[str, Any],
    *,
    logit_field: str,
    initial_answer: str,
) -> dict[str, float]:
    x = option_vector(uncapped, logit_field)
    y = option_vector(capped, logit_field)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    total = float(np.square(y - y.mean()).sum())
    unexplained = float(np.square(residual).sum())
    initial_index = OPTIONS.index(initial_answer)
    alternatives = [i for i in range(len(OPTIONS)) if i != initial_index]
    margin_u = float(np.mean(x[alternatives] - x[initial_index]))
    margin_c = float(np.mean(y[alternatives] - y[initial_index]))
    observed_change = margin_c - margin_u
    affine_change = (float(slope) - 1.0) * margin_u
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 - unexplained / total if total else float("nan"),
        "mean_absolute_residual": float(np.abs(residual).mean()),
        "observed_delta_m_general": observed_change,
        "affine_delta_m_general": affine_change,
        "non_affine_delta_m_general": observed_change - affine_change,
    }


def paired_records(
    records: Iterable[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    condition_field: str,
    uncapped_name: str = "uncapped",
    capped_name: str = "full_p25",
) -> list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = tuple(record.get(field) for field in key_fields)
        grouped[key][record[condition_field]] = record
    return [
        (key, cells[uncapped_name], cells[capped_name])
        for key, cells in grouped.items()
        if uncapped_name in cells and capped_name in cells
    ]


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    array = finite(values)
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def summarize_fits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "status": "not_estimable"}
    observed = finite(row["observed_delta_m_general"] for row in rows)
    predicted = finite(row["affine_delta_m_general"] for row in rows)
    explained = None if abs(float(observed.mean())) < 1e-12 else float(predicted.mean() / observed.mean())
    correlation = None
    if observed.size > 1 and np.std(observed) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
    return {
        "n": len(rows),
        "slope": describe(row["slope"] for row in rows),
        "r_squared": describe(row["r_squared"] for row in rows),
        "fraction_slope_below_one": float(np.mean(finite(row["slope"] for row in rows) < 1)),
        "mean_absolute_residual": float(
            np.mean(finite(row["mean_absolute_residual"] for row in rows))
        ),
        "mean_observed_delta_m_general": float(observed.mean()),
        "mean_affine_delta_m_general": float(predicted.mean()),
        "affine_explained_share": explained,
        "observed_affine_correlation": correlation,
    }


def family_bootstrap(
    rows: list[dict[str, Any]],
    statistic,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    families = sorted({row["family_id"] for row in rows})
    by_family = {family: [row for row in rows if row["family_id"] == family] for family in families}
    estimates = np.empty(draws, dtype=float)
    for index in range(draws):
        sampled = rng.choice(families, size=len(families), replace=True)
        resample = [row for family in sampled for row in by_family[family]]
        estimates[index] = statistic(resample)
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def by_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if arm_rows:
            result[arm] = summarize_fits(arm_rows)
    return result


def fit_standard_records(
    records: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    condition_field: str,
    logit_field: str,
    capped_name: str = "full_p25",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, uncapped, capped in paired_records(
        records,
        key_fields=key_fields,
        condition_field=condition_field,
        capped_name=capped_name,
    ):
        if uncapped.get(logit_field) is None or capped.get(logit_field) is None:
            continue
        initial = uncapped.get("initial_answer") or capped.get("initial_answer")
        if initial not in OPTIONS:
            raise ValueError(f"missing frozen initial_answer for pair {key}")
        row = affine_fit(
            uncapped,
            capped,
            logit_field=logit_field,
            initial_answer=initial,
        )
        row.update(
            {
                "key": list(key),
                "arm": uncapped.get("arm"),
                "family_id": uncapped.get("family_id"),
            }
        )
        rows.append(row)
    return rows


def analyze_rg1(package: Path, initial_answers: dict[tuple[str, str], str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(package / "runs/downstream_propagation/records.jsonl")
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            record["source_id"],
            record["arm"],
            record["trace_source"],
            record.get("variant"),
        )
        grouped[key][record["forward_condition"]] = record
    for key, cells in grouped.items():
        if "uncapped" not in cells or "full_p25" not in cells:
            continue
        uncapped, capped = cells["uncapped"], cells["full_p25"]
        initial = initial_answers[(uncapped["base_id"], uncapped["arm"])]
        row = affine_fit(
            uncapped,
            capped,
            logit_field="current_logits",
            initial_answer=initial,
        )
        row.update(
            {
                "key": list(key),
                "arm": uncapped["arm"],
                "family_id": uncapped["family_id"],
                "trace_source": uncapped["trace_source"],
            }
        )
        rows.append(row)

    controls: dict[str, Any] = {}
    control_names = [
        "uncapped_noop_repeat",
        "random_s20260727_pos",
        "random_s20260727_neg",
        "random_s20260728_pos",
        "random_s20260728_neg",
    ]
    for name in control_names:
        fits: list[dict[str, float]] = []
        for key, cells in grouped.items():
            if "uncapped" not in cells or name not in cells:
                continue
            initial = initial_answers[(cells["uncapped"]["base_id"], cells["uncapped"]["arm"])]
            fits.append(
                affine_fit(
                    cells["uncapped"],
                    cells[name],
                    logit_field="current_logits",
                    initial_answer=initial,
                )
            )
        controls[name] = summarize_fits(fits)
    return rows, controls


def add_exposure(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    capped = {
        (r["base_id"], r["arm"]): r
        for r in records
        if r.get("condition") == "full_p25"
    }
    for row in rows:
        base_id, arm = row["key"][0], row["arm"]
        stats = capped[(base_id, arm)].get("cap_stats") or {}
        correction = 0.0
        triggered = 0.0
        positions = 0.0
        for layer in (stats.get("by_layer") or {}).values():
            correction += float(layer.get("projection_correction_sum", 0.0))
            triggered += float(layer.get("triggered_positions", 0.0))
            positions += float(layer.get("total_positions", 0.0))
        row["total_projection_correction"] = correction
        row["trigger_rate"] = triggered / positions if positions else float("nan")
        row["triggerable_positions"] = positions


def contrast(
    rows: list[dict[str, Any]],
    field: str,
    first: str,
    second: str,
    *,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    def statistic(sample: list[dict[str, Any]]) -> float:
        a = np.mean([row[field] for row in sample if row["arm"] == first])
        b = np.mean([row[field] for row in sample if row["arm"] == second])
        return float(a - b)

    estimate = statistic(rows)
    low, high = family_bootstrap(rows, statistic, draws=draws, rng=rng)
    return {"estimate": estimate, "ci_low": low, "ci_high": high}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="root of the anonymous supporting-data package",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON (default: runs/scale_contraction/SCALE_CONTRACTION_ANALYSIS.json)",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    args = parser.parse_args()
    package = args.package_root.resolve()

    qwen_records = read_jsonl(package / "runs/cross_model/qwen_fc_records.jsonl")
    llama_records = read_jsonl(package / "runs/cross_model/llama_records.jsonl")
    qwen_initial = {(r["base_id"], r["arm"]): r["initial_answer"] for r in qwen_records}

    qwen_rows = fit_standard_records(
        qwen_records,
        key_fields=("base_id", "arm", "variant"),
        condition_field="condition",
        logit_field="answer_logits",
        capped_name="p25",
    )
    llama_rows = fit_standard_records(
        llama_records,
        key_fields=("base_id", "arm", "variant"),
        condition_field="condition",
        logit_field="answer_logits",
        capped_name="p25",
    )
    new_salt_records = read_jsonl(
        package
        / "runs/evidence_new_salt/formal-paired-uncapped-full-p25-seed0/results.jsonl"
    )
    new_salt_rows = fit_standard_records(
        new_salt_records,
        key_fields=("base_id", "arm"),
        condition_field="condition",
        logit_field="answer_logits",
    )
    add_exposure(new_salt_rows, new_salt_records)
    rg1_rows, controls = analyze_rg1(package, qwen_initial)

    social: dict[str, Any] = {}
    for arm_name, arm_dir in (("AO", "direct"), ("S", "direct")):
        uncapped = read_jsonl(package / f"runs/social/{arm_dir}-uncapped/results.jsonl")
        capped = read_jsonl(package / f"runs/social/{arm_dir}-p25/results.jsonl")
        wanted = arm_name
        cells = [r for r in uncapped + capped if r.get("arm") == wanted]
        rows = fit_standard_records(
            cells,
            key_fields=("base_id", "arm"),
            condition_field="condition",
            logit_field="answer_logits",
            capped_name="p25",
        )
        social[arm_name] = summarize_fits(rows)

    bootstrap_rng = np.random.default_rng(args.bootstrap_seed)
    arm_slopes: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in new_salt_rows if row["arm"] == arm]
        estimate = float(np.mean([row["slope"] for row in arm_rows]))

        def stat(sample: list[dict[str, Any]], selected: str = arm) -> float:
            return float(np.mean([row["slope"] for row in sample if row["arm"] == selected]))

        low, high = family_bootstrap(
            new_salt_rows,
            stat,
            draws=args.bootstrap_draws,
            rng=bootstrap_rng,
        )
        arm_slopes[arm] = {"estimate": estimate, "ci_low": low, "ci_high": high}

    # Preserve the frozen analysis order because one RNG stream is shared by
    # all reported family-cluster intervals.
    posthoc_contrasts: dict[str, Any] = {}
    for name, field, first, second in (
        ("slope_V_minus_F", "slope", "V", "F"),
        ("slope_V_minus_R", "slope", "V", "R"),
        ("slope_F_minus_N", "slope", "F", "N"),
        ("residual_V_minus_R", "mean_absolute_residual", "V", "R"),
        ("residual_F_minus_R", "mean_absolute_residual", "F", "R"),
        ("residual_V_minus_F", "mean_absolute_residual", "V", "F"),
    ):
        posthoc_contrasts[name] = contrast(
            new_salt_rows,
            field,
            first,
            second,
            draws=args.bootstrap_draws,
            rng=bootstrap_rng,
        )

    correction_contrasts: dict[str, Any] = {}
    for name, first, second in (
        ("F_minus_R", "F", "R"),
        ("V_minus_R", "V", "R"),
        ("V_minus_F", "V", "F"),
    ):
        correction_contrasts[name] = contrast(
            new_salt_rows,
            "total_projection_correction",
            first,
            second,
            draws=args.bootstrap_draws,
            rng=bootstrap_rng,
        )

    slope_values = np.asarray([row["slope"] for row in new_salt_rows])
    corrections = np.asarray([row["total_projection_correction"] for row in new_salt_rows])
    positions = np.asarray([row["triggerable_positions"] for row in new_salt_rows])

    result = {
        "schema": "cedar.answer_head_affine_contraction.analysis.v1",
        "definition": {
            "fit": "capped_option_logit = slope * uncapped_option_logit + intercept + residual",
            "initial_answer_source": "frozen initial_answer field; never uncapped argmax",
            "bootstrap": {
                "unit": "structural_family",
                "method": "percentile cluster bootstrap",
                "draws": args.bootstrap_draws,
                "seed": args.bootstrap_seed,
            },
            "status": "descriptive_post_hoc",
        },
        "assay_summaries": {
            "rg1_fixed_trace": summarize_fits(rg1_rows),
            "qwen_forced_choice": summarize_fits(qwen_rows),
            "llama_forced_choice": summarize_fits(llama_rows),
            "new_salt_generation": summarize_fits(new_salt_rows),
            "social": social,
        },
        "affine_explained_share_by_arm": {
            "rg1_fixed_trace": by_arm(rg1_rows),
            "new_salt_generation": by_arm(new_salt_rows),
            "qwen_forced_choice": by_arm(qwen_rows),
            "llama_forced_choice": by_arm(llama_rows),
        },
        "new_salt_arm_slopes": arm_slopes,
        "new_salt_posthoc_contrasts": posthoc_contrasts,
        "new_salt_exposure": {
            "by_arm": {
                arm: {
                    "mean_total_projection_correction": float(
                        np.mean(
                            [
                                row["total_projection_correction"]
                                for row in new_salt_rows
                                if row["arm"] == arm
                            ]
                        )
                    ),
                    "mean_trigger_rate": float(
                        np.mean(
                            [row["trigger_rate"] for row in new_salt_rows if row["arm"] == arm]
                        )
                    ),
                    "mean_triggerable_positions": float(
                        np.mean(
                            [
                                row["triggerable_positions"]
                                for row in new_salt_rows
                                if row["arm"] == arm
                            ]
                        )
                    ),
                }
                for arm in ARMS
            },
            "correction_contrasts": correction_contrasts,
            "correlation_total_correction_with_slope": float(
                np.corrcoef(corrections, slope_values)[0, 1]
            ),
            "correlation_position_count_with_slope": float(
                np.corrcoef(positions, slope_values)[0, 1]
            ),
        },
        "rg1_controls": controls,
        "record_counts": {
            "rg1_pairs": len(rg1_rows),
            "qwen_pairs": len(qwen_rows),
            "llama_pairs": len(llama_rows),
            "new_salt_pairs": len(new_salt_rows),
            "social_pairs_per_arm": {
                arm: summary["n"] for arm, summary in social.items()
            },
        },
    }

    output = args.output or (
        package / "runs/scale_contraction/SCALE_CONTRACTION_ANALYSIS.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
