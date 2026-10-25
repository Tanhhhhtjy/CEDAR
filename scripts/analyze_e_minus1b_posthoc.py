#!/usr/bin/env python3
"""Run the frozen, zero-GPU E-1b.1 diagnostic posthoc audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cedar_m2.checkpoint import atomic_write_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "b3f9d2a83d305ecf0bcc012546c029f413c213d7"
SOURCE_DIR = ROOT / "artifacts" / "m2" / "e_minus1b_philpapers_cap_audit"
DEFAULT_OUTPUT = ROOT / "artifacts" / "m2" / "e_minus1b_posthoc_audit"
PROTOCOL = ROOT / "docs" / "E_MINUS1B_POSTHOC_AUDIT.md"
LAYERS = tuple(range(46, 54))
SAMPLE_SEEDS = (42, 7, 123)
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_HASHES = {
    "row_results.jsonl": "da6e01c4543173da62172b9cefa6f8512c22e1d21bbdfc51b4f4b6ca13bf00c4",
    "pair_results.jsonl": "3fa946a5a393e78e55d596ead63e71e47510a3b333550632941a9cc84963f2fa",
    "prepare_audit.json": "f08ccb4516f19657ab90cf3372c8da63fc4e3d72223c3a40638349c0fbb785d8",
    "runtime_manifest.json": "3691dc222c33b004e26b4adb0fe419cf3ac98a91eec9b13cfccdf6de9d7cff33",
}
TRANSITIONS = (
    "stable_sycophantic",
    "sycophantic_to_non_sycophantic",
    "non_sycophantic_to_sycophantic",
    "stable_non_sycophantic",
)
COLORS = {
    "stable_sycophantic": "#8b8b8b",
    "sycophantic_to_non_sycophantic": "#168a45",
    "non_sycophantic_to_sycophantic": "#d84b16",
    "stable_non_sycophantic": "#1665d8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_blob_sha256(commit: str, path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def rounded(value: float | None, digits: int = 10) -> float | None:
    return None if value is None else round(float(value), digits)


def percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="linear"))


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": rounded(float(array.mean())),
        "std": rounded(float(array.std(ddof=1))) if len(values) > 1 else 0.0,
        "min": rounded(float(array.min())),
        "q25": rounded(float(np.quantile(array, 0.25))),
        "median": rounded(float(np.median(array))),
        "q75": rounded(float(np.quantile(array, 0.75))),
        "max": rounded(float(array.max())),
    }


def exact_mcnemar(syc_to_non: int, non_to_syc: int) -> dict[str, Any]:
    discordant = syc_to_non + non_to_syc
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(syc_to_non, non_to_syc)
        lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2 * lower_tail)
    return {
        "test": "two-sided exact McNemar binomial test",
        "sycophantic_to_non_sycophantic": syc_to_non,
        "non_sycophantic_to_sycophantic": non_to_syc,
        "discordant": discordant,
        "odds_ratio_non_to_syc_over_syc_to_non": (
            rounded(non_to_syc / syc_to_non) if syc_to_non else None
        ),
        "p_value": rounded(p_value),
        "cluster_warning": (
            "Auxiliary only: this exact test treats 900 prompt-level pairs as paired units but "
            "does not adjust for original/swapped clustering."
        ),
    }


def validate_inputs(source_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    file_checks: dict[str, Any] = {}
    for name, expected in EXPECTED_HASHES.items():
        path = source_dir / name
        exists = path.is_file()
        actual = file_sha256(path) if exists else None
        blob = git_blob_sha256(SOURCE_COMMIT, path) if exists else None
        passed = exists and actual == expected and blob == expected
        file_checks[name] = {
            "exists": exists,
            "expected_sha256": expected,
            "working_tree_sha256": actual,
            "source_commit_blob_sha256": blob,
            "passed": passed,
        }
        if not passed:
            failures.append(f"hash_or_blob:{name}")

    keys = [(row.get("sample_seed"), row.get("question_id"), row.get("condition")) for row in rows]
    counts = Counter(keys)
    condition_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    pair_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    math_failures = 0
    for row in rows:
        seed = int(row.get("sample_seed", -1))
        question_id = int(row.get("question_id", -1))
        condition_groups[(seed, question_id)].append(row)
        if row.get("condition") == "uncapped_monitor":
            pair_groups[(seed, str(row.get("pair_id")))].append(row)
        syc = row.get("sycophantic_answer")
        non = row.get("non_sycophantic_answer")
        if {syc, non} != {"A", "B"}:
            math_failures += 1
            continue
        logits = {"A": float(row["A_logit"]), "B": float(row["B_logit"])}
        expected_margin = logits[str(syc)] - logits[str(non)]
        expected_prediction = "A" if logits["A"] >= logits["B"] else "B"
        if (
            abs(expected_margin - float(row["sycophancy_margin"])) > 1e-7
            or row.get("predicted_forced_choice") != expected_prediction
        ):
            math_failures += 1

    condition_pair_failures = sum(
        len(group) != 2 or {row.get("condition") for row in group} != {
            "uncapped_monitor",
            "p25_cap",
        }
        for group in condition_groups.values()
    )
    order_pair_failures = sum(
        len(group) != 2 or {row.get("variant") for row in group} != {"original", "swapped"}
        for group in pair_groups.values()
    )
    seed_condition_rows = {
        str(seed): {
            condition: sum(
                row.get("sample_seed") == seed and row.get("condition") == condition
                for row in rows
            )
            for condition in ("uncapped_monitor", "p25_cap")
        }
        for seed in SAMPLE_SEEDS
    }
    checks = {
        "source_commit_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0,
        "frozen_file_hashes_and_git_blobs": all(item["passed"] for item in file_checks.values()),
        "1800_condition_rows": len(rows) == 1800,
        "keys_unique": len(counts) == 1800 and all(count == 1 for count in counts.values()),
        "900_complete_condition_pairs": len(condition_groups) == 900
        and condition_pair_failures == 0,
        "450_complete_original_swapped_clusters": len(pair_groups) == 450
        and order_pair_failures == 0,
        "three_seed_counts": all(
            seed_condition_rows[str(seed)] == {"uncapped_monitor": 300, "p25_cap": 300}
            for seed in SAMPLE_SEEDS
        ),
        "answer_mapping_and_margin_math": math_failures == 0,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    return {
        "status": "passed" if not failures else "failed",
        "source_commit": SOURCE_COMMIT,
        "posthoc": True,
        "file_checks": file_checks,
        "checks": checks,
        "counts": {
            "condition_rows": len(rows),
            "prompt_condition_pairs": len(condition_groups),
            "original_swapped_clusters": len(pair_groups),
            "seed_condition_rows": seed_condition_rows,
        },
        "failures": failures,
    }


def transition_name(before_syc: bool, after_syc: bool) -> str:
    if before_syc and after_syc:
        return "stable_sycophantic"
    if before_syc and not after_syc:
        return "sycophantic_to_non_sycophantic"
    if not before_syc and after_syc:
        return "non_sycophantic_to_sycophantic"
    return "stable_non_sycophantic"


def normalized_exposure(row: dict[str, Any], *, actual: bool) -> dict[str, Any]:
    prompt_tokens = int(row["prompt_token_count"])
    if actual:
        source = row["actual_cap_stats"]["by_layer"]
        total_name, triggered_name = "total_positions", "triggered_positions"
    else:
        source = row["axis_telemetry"]["phases"]["prefill"]["reference_cap_exposure"][
            "p25"
        ]
        total_name, triggered_name = "count", "triggered"
    layer_values: dict[str, Any] = {}
    for layer in map(str, LAYERS):
        item = source[layer]
        total = int(item[total_name])
        triggered = int(item[triggered_name])
        correction = float(item["correction_sum"])
        if total != prompt_tokens:
            raise ValueError(f"Layer {layer}: {total} positions != {prompt_tokens} prompt tokens")
        layer_values[layer] = {
            "total_positions": total,
            "triggered_positions": triggered,
            "trigger_rate": triggered / total if total else 0.0,
            "correction_sum": correction,
            "normalized_correction": correction / total if total else 0.0,
        }

    def aggregate(layers: tuple[int, ...]) -> dict[str, Any]:
        selected = [layer_values[str(layer)] for layer in layers]
        total = sum(int(item["total_positions"]) for item in selected)
        triggered = sum(int(item["triggered_positions"]) for item in selected)
        correction = sum(float(item["correction_sum"]) for item in selected)
        return {
            "layers": list(layers),
            "total_positions": total,
            "triggered_positions": triggered,
            "trigger_rate": triggered / total if total else 0.0,
            "correction_sum": correction,
            "normalized_correction": correction / total if total else 0.0,
        }

    pooled = aggregate(LAYERS)
    return {
        "by_layer": layer_values,
        "pooled": {
            **pooled,
            "mean_correction_per_token": pooled["correction_sum"] / prompt_tokens,
        },
        "layers_46_49": aggregate(tuple(range(46, 50))),
        "layers_50_53": aggregate(tuple(range(50, 54))),
        "layer_46": aggregate((46,)),
        "layers_47_53": aggregate(tuple(range(47, 54))),
    }


def build_transition_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (int(row["sample_seed"]), int(row["question_id"]), str(row["condition"])): row
        for row in rows
    }
    uncapped_rows = [row for row in rows if row["condition"] == "uncapped_monitor"]
    transitions: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    for uncapped in uncapped_rows:
        seed = int(uncapped["sample_seed"])
        question_id = int(uncapped["question_id"])
        capped = by_key[(seed, question_id, "p25_cap")]
        syc = str(uncapped["sycophantic_answer"])
        non = str(uncapped["non_sycophantic_answer"])
        before_syc = uncapped["predicted_forced_choice"] == syc
        after_syc = capped["predicted_forced_choice"] == syc
        transition = transition_name(before_syc, after_syc)
        uncapped_logits = {"A": float(uncapped["A_logit"]), "B": float(uncapped["B_logit"])}
        capped_logits = {"A": float(capped["A_logit"]), "B": float(capped["B_logit"])}
        cf = normalized_exposure(uncapped, actual=False)
        actual = normalized_exposure(capped, actual=True)
        shared = {
            "sample_seed": seed,
            "pair_id": uncapped["pair_id"],
            "base_id": uncapped["base_id"],
            "orig_id": uncapped["orig_id"],
            "question_id": question_id,
            "variant": uncapped["variant"],
            "transition": transition,
        }
        transition_row = {
            **shared,
            "sycophantic_answer": syc,
            "non_sycophantic_answer": non,
            "uncapped_syc_logit": uncapped_logits[syc],
            "uncapped_non_syc_logit": uncapped_logits[non],
            "p25_syc_logit": capped_logits[syc],
            "p25_non_syc_logit": capped_logits[non],
            "delta_syc_logit": capped_logits[syc] - uncapped_logits[syc],
            "delta_non_syc_logit": capped_logits[non] - uncapped_logits[non],
            "uncapped_margin": float(uncapped["sycophancy_margin"]),
            "abs_uncapped_margin": abs(float(uncapped["sycophancy_margin"])),
            "p25_margin": float(capped["sycophancy_margin"]),
            "delta_margin": float(capped["sycophancy_margin"])
            - float(uncapped["sycophancy_margin"]),
            "crossed_choice_boundary": before_syc != after_syc,
            "delta_AB_mass": float(capped["AB_probability_mass"])
            - float(uncapped["AB_probability_mass"]),
            "delta_entropy": float(capped["full_vocab_entropy"])
            - float(uncapped["full_vocab_entropy"]),
            "top_token_changed": capped["full_vocab_top1_token_id"]
            != uncapped["full_vocab_top1_token_id"],
            "uncapped_sycophantic_choice": before_syc,
            "p25_sycophantic_choice": after_syc,
        }
        exposure_row = {
            **shared,
            "prompt_tokens": int(uncapped["prompt_token_count"]),
            "counterfactual": cf,
            "actual_p25": actual,
            "position_level_availability": {
                "last_input_token": "not_available",
                "near_AB_decision": "not_available",
                "chat_template_vs_user_body": "not_available",
                "reason": "E-1b stores layer aggregates but no token-position exposure arrays",
            },
        }
        transitions.append(transition_row)
        exposures.append(exposure_row)
    if len(transitions) != 900 or len(exposures) != 900:
        raise ValueError("Expected 900 prompt-level condition pairs")
    return transitions, exposures


def stratified_cluster_indices(transitions: list[dict[str, Any]]) -> dict[int, list[np.ndarray]]:
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(transitions):
        grouped[(int(row["sample_seed"]), str(row["pair_id"]))].append(index)
    by_seed: dict[int, list[np.ndarray]] = {seed: [] for seed in SAMPLE_SEEDS}
    for (seed, _pair_id), indices in grouped.items():
        if len(indices) != 2 or {transitions[index]["variant"] for index in indices} != {
            "original",
            "swapped",
        }:
            raise ValueError("Bootstrap cluster is not a complete original/swapped pair")
        by_seed[seed].append(np.asarray(indices, dtype=np.int64))
    if any(len(by_seed[seed]) != 150 for seed in SAMPLE_SEEDS):
        raise ValueError("Expected 150 original/swapped clusters per seed")
    return by_seed


def _sample_indices(
    rng: np.random.Generator,
    clusters: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    by_seed: dict[int, np.ndarray] = {}
    for seed in SAMPLE_SEEDS:
        seed_clusters = clusters[seed]
        selected = rng.integers(0, len(seed_clusters), size=len(seed_clusters))
        by_seed[seed] = np.concatenate([seed_clusters[index] for index in selected])
    return np.concatenate([by_seed[seed] for seed in SAMPLE_SEEDS]), by_seed


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return None
    return float(np.dot(centered, y - y.mean()) / denominator)


def run_bootstrap(
    transitions: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    clusters = stratified_cluster_indices(transitions)
    before = np.asarray([float(row["uncapped_sycophantic_choice"]) for row in transitions])
    after = np.asarray([float(row["p25_sycophantic_choice"]) for row in transitions])
    delta_choice = after - before
    categories = np.asarray([row["transition"] for row in transitions], dtype=object)

    exposure_by_question = {int(row["question_id"]): row for row in exposures}
    feature_names = (
        "abs_uncapped_margin",
        "normalized_mean_correction",
        "layer_46_normalized_correction",
        "layers_47_53_normalized_correction",
        "layers_46_49_normalized_correction",
        "layers_50_53_normalized_correction",
    )
    matrix_rows: list[list[float]] = []
    for transition in transitions:
        exposure = exposure_by_question[int(transition["question_id"])]["counterfactual"]
        matrix_rows.append(
            [
                float(transition["abs_uncapped_margin"]),
                float(exposure["pooled"]["normalized_correction"]),
                float(exposure["layer_46"]["normalized_correction"]),
                float(exposure["layers_47_53"]["normalized_correction"]),
                float(exposure["layers_46_49"]["normalized_correction"]),
                float(exposure["layers_50_53"]["normalized_correction"]),
            ]
        )
    feature_matrix = np.asarray(matrix_rows, dtype=float)
    continuous_outcomes = {
        "delta_margin": np.asarray([float(row["delta_margin"]) for row in transitions]),
        "delta_entropy": np.asarray([float(row["delta_entropy"]) for row in transitions]),
        "top_token_changed": np.asarray(
            [float(row["top_token_changed"]) for row in transitions]
        ),
        "net_away_from_sycophancy": -delta_choice,
    }
    x = feature_matrix[:, feature_names.index("normalized_mean_correction")]
    contrasts = {
        "non_syc_to_syc_vs_stable_non_syc": (
            "non_sycophantic_to_sycophantic",
            "stable_non_sycophantic",
        ),
        "syc_to_non_syc_vs_stable_syc": (
            "sycophantic_to_non_sycophantic",
            "stable_sycophantic",
        ),
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    overall_replicates: list[float] = []
    seed_replicates: dict[int, list[float]] = {seed: [] for seed in SAMPLE_SEEDS}
    contrast_replicates: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {
            feature: {"mean_difference": [], "median_difference": []}
            for feature in feature_names
        }
        for name in contrasts
    }
    slope_replicates: dict[str, list[float]] = {name: [] for name in continuous_outcomes}
    for _replicate in range(BOOTSTRAP_REPLICATES):
        sampled, sampled_by_seed = _sample_indices(rng, clusters)
        overall_replicates.append(float(delta_choice[sampled].mean()))
        for seed in SAMPLE_SEEDS:
            seed_replicates[seed].append(float(delta_choice[sampled_by_seed[seed]].mean()))
        sampled_categories = categories[sampled]
        sampled_features = feature_matrix[sampled]
        for contrast_name, (left_name, right_name) in contrasts.items():
            left = sampled_categories == left_name
            right = sampled_categories == right_name
            if not left.any() or not right.any():
                continue
            for feature_index, feature in enumerate(feature_names):
                left_values = sampled_features[left, feature_index]
                right_values = sampled_features[right, feature_index]
                contrast_replicates[contrast_name][feature]["mean_difference"].append(
                    float(left_values.mean() - right_values.mean())
                )
                contrast_replicates[contrast_name][feature]["median_difference"].append(
                    float(np.median(left_values) - np.median(right_values))
                )
        sampled_x = x[sampled]
        for name, outcome in continuous_outcomes.items():
            estimate = _slope(sampled_x, outcome[sampled])
            if estimate is not None:
                slope_replicates[name].append(estimate)

    def interval(values: list[float]) -> list[float]:
        return [rounded(percentile(values, 0.025)), rounded(percentile(values, 0.975))]

    counts = Counter(row["transition"] for row in transitions)
    point = float(delta_choice.mean())
    ci = interval(overall_replicates)
    if float(ci[0]) > 0:
        reliability = "reliable_increase"
    elif float(ci[1]) <= 0:
        reliability = "does_not_support_increase"
    else:
        reliability = "positive_direction_candidate_CI_includes_zero"
    discrete = {
        "status": "passed",
        "estimand": "P(syc|p25) - P(syc|uncapped)",
        "condition_paired_prompts": 900,
        "bootstrap_unit": (
            "150 original/swapped base clusters sampled within each of three sample seeds; "
            "both variants retained together"
        ),
        "point_estimate": rounded(point),
        "point_estimate_percentage_points": rounded(100 * point),
        "ci95": ci,
        "ci95_percentage_points": [rounded(100 * float(value)) for value in ci],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": BOOTSTRAP_SEED,
        "classification": reliability,
        "uncapped_sycophantic_rate": rounded(float(before.mean())),
        "p25_sycophantic_rate": rounded(float(after.mean())),
        "transition_counts": {name: counts[name] for name in TRANSITIONS},
        "by_seed": {},
        "exact_paired_auxiliary": exact_mcnemar(
            counts["sycophantic_to_non_sycophantic"],
            counts["non_sycophantic_to_sycophantic"],
        ),
    }
    for seed in SAMPLE_SEEDS:
        mask = np.asarray([row["sample_seed"] == seed for row in transitions])
        seed_point = float(delta_choice[mask].mean())
        discrete["by_seed"][str(seed)] = {
            "prompts": int(mask.sum()),
            "point_estimate": rounded(seed_point),
            "point_estimate_percentage_points": rounded(100 * seed_point),
            "ci95": interval(seed_replicates[seed]),
            "uncapped_sycophantic_rate": rounded(float(before[mask].mean())),
            "p25_sycophantic_rate": rounded(float(after[mask].mean())),
        }

    observed_contrasts: dict[str, Any] = {}
    for contrast_name, (left_name, right_name) in contrasts.items():
        left = categories == left_name
        right = categories == right_name
        feature_reports: dict[str, Any] = {}
        for feature_index, feature in enumerate(feature_names):
            left_values = feature_matrix[left, feature_index]
            right_values = feature_matrix[right, feature_index]
            replicates = contrast_replicates[contrast_name][feature]
            feature_reports[feature] = {
                "left": left_name,
                "right": right_name,
                "left_n": int(left.sum()),
                "right_n": int(right.sum()),
                "left_mean": rounded(float(left_values.mean())),
                "right_mean": rounded(float(right_values.mean())),
                "mean_difference": rounded(float(left_values.mean() - right_values.mean())),
                "mean_difference_ci95": interval(replicates["mean_difference"]),
                "left_median": rounded(float(np.median(left_values))),
                "right_median": rounded(float(np.median(right_values))),
                "median_difference": rounded(
                    float(np.median(left_values) - np.median(right_values))
                ),
                "median_difference_ci95": interval(replicates["median_difference"]),
            }
        observed_contrasts[contrast_name] = feature_reports

    continuous: dict[str, Any] = {}
    for name, outcome in continuous_outcomes.items():
        observed_slope = _slope(x, outcome)
        correlation = float(np.corrcoef(x, outcome)[0, 1]) if outcome.std() else None
        continuous[name] = {
            "predictor": "uncapped normalized mean correction per token-layer position",
            "slope": rounded(observed_slope),
            "slope_ci95": interval(slope_replicates[name]),
            "pearson_correlation": rounded(correlation),
            "cluster_bootstrap_replicates": len(slope_replicates[name]),
        }
    effects = {
        "status": "passed",
        "posthoc": True,
        "primary_predictor": "uncapped counterfactual normalized mean correction",
        "actual_p25_exposure_used_as_predictor": False,
        "position_level_exposure": "not_available",
        "predefined_transition_contrasts": observed_contrasts,
        "continuous_associations": continuous,
        "interpretation_boundary": (
            "Small transition groups and posthoc design make these descriptive mechanism "
            "diagnostics, not confirmatory causal estimates."
        ),
    }
    return discrete, effects


def transition_group_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "uncapped_margin",
        "abs_uncapped_margin",
        "delta_syc_logit",
        "delta_non_syc_logit",
        "delta_margin",
        "delta_AB_mass",
        "delta_entropy",
    )
    output: dict[str, Any] = {}
    for transition in TRANSITIONS:
        selected = [row for row in transitions if row["transition"] == transition]
        output[transition] = {
            "rows": len(selected),
            "metrics": {
                field: describe([float(row[field]) for row in selected]) for field in fields
            },
            "top_token_change_rate": rounded(
                statistics.fmean(float(row["top_token_changed"]) for row in selected)
            ),
        }
    return output


def _svg_to_png(svg: str, path: Path) -> None:
    temporary = path.with_suffix(".tmp.svg")
    temporary.write_text(svg, encoding="utf-8", newline="\n")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(temporary),
                "-frames:v",
                "1",
                str(path),
            ],
            check=True,
        )
    finally:
        temporary.unlink(missing_ok=True)


def scatter_plot(transitions: list[dict[str, Any]], path: Path) -> None:
    width, height = 1000, 720
    left, top, plot_width, plot_height = 90, 65, 850, 570
    x_values = [float(row["uncapped_margin"]) for row in transitions]
    y_values = [float(row["delta_margin"]) for row in transitions]
    x_limit = max(abs(min(x_values)), abs(max(x_values))) * 1.08
    y_limit = max(abs(min(y_values)), abs(max(y_values))) * 1.08

    def x_pos(value: float) -> float:
        return left + (value + x_limit) / (2 * x_limit) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_limit - value) / (2 * y_limit) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="32" text-anchor="middle" font-family="sans-serif" '
        'font-size="21" font-weight="bold">Uncapped margin vs p25 delta margin</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="#fafafa" stroke="#bbb"/>',
        f'<line x1="{x_pos(0)}" y1="{top}" x2="{x_pos(0)}" y2="{top + plot_height}" '
        'stroke="#777" stroke-dasharray="5,4"/>',
        f'<line x1="{left}" y1="{y_pos(0)}" x2="{left + plot_width}" y2="{y_pos(0)}" '
        'stroke="#777" stroke-dasharray="5,4"/>',
        f'<line x1="{x_pos(-min(x_limit, y_limit))}" y1="{y_pos(min(x_limit, y_limit))}" '
        f'x2="{x_pos(min(x_limit, y_limit))}" y2="{y_pos(-min(x_limit, y_limit))}" '
        'stroke="#111" stroke-dasharray="7,5"/>',
    ]
    for row in transitions:
        color = COLORS[str(row["transition"])]
        radius = 4.0 if row["crossed_choice_boundary"] else 2.2
        svg.append(
            f'<circle cx="{x_pos(float(row["uncapped_margin"])):.2f}" '
            f'cy="{y_pos(float(row["delta_margin"])):.2f}" r="{radius}" '
            f'fill="{color}" fill-opacity="0.62"/>'
        )
    for index, transition in enumerate(TRANSITIONS):
        y = 665 + (index // 2) * 20
        x = 110 + (index % 2) * 390
        svg.extend(
            [
                f'<circle cx="{x}" cy="{y - 4}" r="4" fill="{COLORS[transition]}"/>',
                f'<text x="{x + 10}" y="{y}" font-family="sans-serif" '
                f'font-size="11">{transition}</text>',
            ]
        )
    svg.extend(
        [
            '<text x="515" y="708" text-anchor="middle" font-family="sans-serif" '
            'font-size="13">uncapped sycophancy margin</text>',
            '<text x="20" y="350" text-anchor="middle" font-family="sans-serif" '
            'font-size="13" transform="rotate(-90 20 350)">p25 - uncapped margin</text>',
            '<text x="735" y="88" font-family="sans-serif" font-size="11">'
            'dashed diagonal: p25 margin = 0</text>',
            "</svg>",
        ]
    )
    _svg_to_png("\n".join(svg), path)


def boxplot_distributions(transitions: list[dict[str, Any]], path: Path) -> None:
    width, height = 1100, 650
    panels = (
        ("Uncapped margin", "uncapped_margin", 60),
        ("Absolute uncapped margin", "abs_uncapped_margin", 580),
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="21" font-weight="bold">Margin distribution by transition</text>',
    ]
    for title, field, left in panels:
        values = [float(row[field]) for row in transitions]
        minimum, maximum = min(values), max(values)
        if minimum == maximum:
            maximum += 1
        plot_top, plot_height = 80, 470

        def y_pos(
            value: float,
            low: float = minimum,
            high: float = maximum,
            origin: float = plot_top,
            height: float = plot_height,
        ) -> float:
            return origin + (high - value) / (high - low) * height

        svg.extend(
            [
                f'<rect x="{left}" y="60" width="480" height="535" fill="#fafafa" '
                'stroke="#ccc"/>',
                f'<text x="{left + 240}" y="82" text-anchor="middle" '
                f'font-family="sans-serif" font-size="16">{title}</text>',
            ]
        )
        for index, transition in enumerate(TRANSITIONS):
            selected = np.asarray(
                [float(row[field]) for row in transitions if row["transition"] == transition]
            )
            q1, median, q3 = np.quantile(selected, [0.25, 0.5, 0.75])
            low, high = selected.min(), selected.max()
            x = left + 75 + index * 105
            svg.extend(
                [
                    f'<line x1="{x}" y1="{y_pos(low)}" x2="{x}" y2="{y_pos(high)}" '
                    f'stroke="{COLORS[transition]}"/>',
                    f'<rect x="{x - 25}" y="{y_pos(q3)}" width="50" '
                    f'height="{max(y_pos(q1) - y_pos(q3), 1)}" fill="{COLORS[transition]}" '
                    'fill-opacity="0.25" stroke="#333"/>',
                    f'<line x1="{x - 25}" y1="{y_pos(median)}" x2="{x + 25}" '
                    f'y2="{y_pos(median)}" stroke="#111" stroke-width="2"/>',
                    f'<text x="{x}" y="575" text-anchor="middle" font-family="sans-serif" '
                    f'font-size="10">{index + 1}: n={len(selected)}</text>',
                ]
            )
    svg.extend(
        [
            '<text x="550" y="628" text-anchor="middle" font-family="sans-serif" '
            'font-size="11">1 stable syc; 2 syc→non-syc; 3 non-syc→syc; 4 stable non-syc</text>',
            "</svg>",
        ]
    )
    _svg_to_png("\n".join(svg), path)


def layerwise_plot(
    transitions: list[dict[str, Any]], exposures: list[dict[str, Any]], path: Path
) -> None:
    by_question = {int(row["question_id"]): row for row in exposures}
    series: dict[str, list[float]] = {}
    for transition in TRANSITIONS:
        selected = [row for row in transitions if row["transition"] == transition]
        series[transition] = [
            statistics.fmean(
                float(
                    by_question[int(row["question_id"])]["counterfactual"]["by_layer"][
                        str(layer)
                    ]["normalized_correction"]
                )
                for row in selected
            )
            for layer in LAYERS
        ]
    width, height = 1000, 650
    left, top, plot_width, plot_height = 85, 70, 850, 480
    all_values = [value for values in series.values() for value in values]
    low, high = min(all_values), max(all_values)
    padding = max((high - low) * 0.12, 1e-8)
    low, high = low - padding, high + padding

    def x_pos(index: int) -> float:
        return left + index / (len(LAYERS) - 1) * plot_width

    def y_pos(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="32" text-anchor="middle" font-family="sans-serif" '
        'font-size="21" font-weight="bold">Uncapped counterfactual exposure by layer</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="#fafafa" stroke="#bbb"/>',
    ]
    for index, layer in enumerate(LAYERS):
        x = x_pos(index)
        svg.append(
            f'<text x="{x}" y="{top + plot_height + 20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{layer}</text>'
        )
    for series_index, transition in enumerate(TRANSITIONS):
        values = series[transition]
        points = " ".join(
            f"{x_pos(index):.2f},{y_pos(value):.2f}" for index, value in enumerate(values)
        )
        color = COLORS[transition]
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        for index, value in enumerate(values):
            svg.append(
                f'<circle cx="{x_pos(index):.2f}" cy="{y_pos(value):.2f}" r="3" '
                f'fill="{color}"/>'
            )
        svg.append(
            f'<text x="{110 + (series_index % 2) * 400}" '
            f'y="{590 + (series_index // 2) * 22}" font-family="sans-serif" '
            f'font-size="11" fill="{color}">{transition}</text>'
        )
    svg.extend(
        [
            '<text x="510" y="635" text-anchor="middle" font-family="sans-serif" '
            'font-size="13">decoder layer</text>',
            '<text x="20" y="320" text-anchor="middle" font-family="sans-serif" '
            'font-size="13" transform="rotate(-90 20 320)">correction_sum / layer positions</text>',
            "</svg>",
        ]
    )
    _svg_to_png("\n".join(svg), path)


def summary_markdown(
    discrete: dict[str, Any],
    group_summary: dict[str, Any],
    effects: dict[str, Any],
    input_audit: dict[str, Any],
) -> str:
    non_to_syc = group_summary["non_sycophantic_to_sycophantic"]
    stable_non = group_summary["stable_non_sycophantic"]
    flip_abs = non_to_syc["metrics"]["abs_uncapped_margin"]
    stable_abs = stable_non["metrics"]["abs_uncapped_margin"]
    contrast = effects["predefined_transition_contrasts"][
        "non_syc_to_syc_vs_stable_non_syc"
    ]
    entropy = effects["continuous_associations"]["delta_entropy"]
    net = effects["continuous_associations"]["net_away_from_sycophancy"]
    lines = [
        "# E−1b.1 零 GPU 补充审计摘要",
        "",
        "> 本轮是在看到 E−1b 结果后设计的诊断性补充分析，不属于预注册确认性结果。",
        "",
        f"- 输入审计：`{input_audit['status']}`",
        "- GPU/model execution：`false`",
        f"- condition-paired prompts：{discrete['condition_paired_prompts']}",
        "- original/swapped bootstrap clusters：450（每 seed 150）",
        f"- bootstrap：{discrete['bootstrap_replicates']}，seed `{BOOTSTRAP_SEED}`",
        "",
        "## 1. +1.44pp 是否可靠？",
        "",
        f"点估计为 `{discrete['point_estimate_percentage_points']:+.4f}pp`；cluster-bootstrap "
        f"95% CI 为 `[{discrete['ci95_percentage_points'][0]:+.4f}, "
        f"{discrete['ci95_percentage_points'][1]:+.4f}]pp`。冻结口径分类为 "
        f"`{discrete['classification']}`。",
        "",
        f"辅助 exact McNemar：12 条 syc→non-syc、25 条 non-syc→syc，"
        f"p=`{discrete['exact_paired_auxiliary']['p_value']:.6f}`；该检验不校正"
        "original/swapped cluster，只作辅助。",
        "",
        "| seed | uncapped syc | p25 syc | delta pp | 95% cluster CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in SAMPLE_SEEDS:
        item = discrete["by_seed"][str(seed)]
        lines.append(
            f"| {seed} | {item['uncapped_sycophantic_rate']:.4f} | "
            f"{item['p25_sycophantic_rate']:.4f} | "
            f"{item['point_estimate_percentage_points']:+.4f} | "
            f"[{100 * item['ci95'][0]:+.4f}, {100 * item['ci95'][1]:+.4f}]pp |"
        )
    lines.extend(
        [
            "",
            "## 2. 为什么连续 margin 与离散选择相反？",
            "",
            f"non-syc→syc 的初始 |margin| 中位数为 `{flip_abs['median']:.6f}`，"
            f"stable non-syc 为 `{stable_abs['median']:.6f}`。翻转主要由靠近零边界的"
            "少量题决定；平均 margin 则由大量未翻转题的连续位移主导。因此两者不是同一"
            "estimand，方向可以相反。",
            "",
            "四类完整分布见 `transition_level_results.jsonl` 和两张 margin 图。",
            "",
            "## 3. 定向谄媚偏移还是一般扰动？",
            "",
            f"normalized exposure→delta entropy 的斜率为 `{entropy['slope']}`，95% CI "
            f"`{entropy['slope_ci95']}`；exposure→净离开谄媚的斜率为 `{net['slope']}`，"
            f"95% CI `{net['slope_ci95']}`。结合 A/B mass、entropy 与 top-token 诊断，"
            "若前者清楚而后者不清楚，应优先解释为一般输出分布扰动。",
            "",
            "## 4. 哪些层或阶段最有信息？",
            "",
            "预先冻结的层组比较如下（non-syc→syc 减 stable non-syc）：",
            "",
            "| 特征 | 均值差 | 95% cluster CI |",
            "|---|---:|---:|",
        ]
    )
    for field in (
        "normalized_mean_correction",
        "layer_46_normalized_correction",
        "layers_47_53_normalized_correction",
        "layers_46_49_normalized_correction",
        "layers_50_53_normalized_correction",
    ):
        item = contrast[field]
        lines.append(
            f"| {field} | {item['mean_difference']:+.6f} | "
            f"[{item['mean_difference_ci95'][0]:+.6f}, "
            f"{item['mean_difference_ci95'][1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "日志没有 token-position arrays，因此最后输入 token、pre-answer、chat template"
            "与用户正文分区均为 `not_available`；本轮没有为补字段重跑模型。",
            "",
            "## 5. 允许与禁止的结论",
            "",
            "允许：报告 PhilPapers 上离散 syc rate 的 cluster-bootstrap 可靠性；说明连续"
            "margin 与边界翻转衡量不同对象；报告 normalized exposure 与输出扰动的描述性"
            "关联。",
            "",
            "禁止：把 posthoc 关联写成确认性机制；把 margin 下降写成回答质量改善；宣称"
            "p25 普遍提高或降低谄媚；外推到正确更新、自由生成或安全性。",
            "",
            "E−1b.1 到此停止；未运行 p1、新 seed、thinking、生成或机制工具。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.source_dir.resolve() != SOURCE_DIR.resolve():
        raise ValueError("E-1b.1 only accepts the frozen E-1b artifact directory")
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().isoformat()
    rows = read_jsonl(args.source_dir / "row_results.jsonl")
    input_audit = validate_inputs(args.source_dir, rows)
    atomic_write_json(args.output_dir / "INPUT_AUDIT.json", input_audit)
    if input_audit["status"] != "passed":
        raise RuntimeError("Frozen E-1b inputs failed identity or completeness audit")

    transitions, exposures = build_transition_rows(rows)
    atomic_write_jsonl(args.output_dir / "transition_level_results.jsonl", transitions)
    atomic_write_jsonl(args.output_dir / "normalized_exposure_results.jsonl", exposures)
    discrete, effects = run_bootstrap(transitions, exposures)
    group_summary = transition_group_summary(transitions)
    effects["transition_group_summary"] = group_summary
    atomic_write_json(args.output_dir / "discrete_sycophancy_bootstrap.json", discrete)
    atomic_write_json(args.output_dir / "exposure_effect_summary.json", effects)
    scatter_plot(transitions, args.output_dir / "uncapped_margin_vs_delta_margin.png")
    boxplot_distributions(
        transitions,
        args.output_dir / "margin_distribution_by_transition.png",
    )
    layerwise_plot(transitions, exposures, args.output_dir / "layerwise_exposure.png")
    (args.output_dir / "E_MINUS1B_POSTHOC_SUMMARY_ZH.md").write_text(
        summary_markdown(discrete, group_summary, effects, input_audit),
        encoding="utf-8",
        newline="\n",
    )
    output_names = (
        "INPUT_AUDIT.json",
        "discrete_sycophancy_bootstrap.json",
        "transition_level_results.jsonl",
        "normalized_exposure_results.jsonl",
        "exposure_effect_summary.json",
        "uncapped_margin_vs_delta_margin.png",
        "margin_distribution_by_transition.png",
        "layerwise_exposure.png",
        "E_MINUS1B_POSTHOC_SUMMARY_ZH.md",
    )
    manifest = {
        "status": "passed",
        "analysis_kind": "posthoc diagnostic; not preregistered confirmatory analysis",
        "started_at": started,
        "completed_at": datetime.now().astimezone().isoformat(),
        "source_commit": SOURCE_COMMIT,
        "analysis_commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gpu_or_model_execution": False,
        "new_model_outputs": 0,
        "content_retries": 0,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "outputs": {
            name: {
                "path": str(args.output_dir / name),
                "sha256": file_sha256(args.output_dir / name),
            }
            for name in output_names
        },
        "mandatory_stop": True,
        "prohibited_runs_executed": [],
    }
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "delta_syc_rate_pp": discrete["point_estimate_percentage_points"],
                "ci95_pp": discrete["ci95_percentage_points"],
                "classification": discrete["classification"],
                "exact_mcnemar_p": discrete["exact_paired_auxiliary"]["p_value"],
                "gpu_or_model_execution": False,
                "mandatory_stop": True,
            }
        )
    )


if __name__ == "__main__":
    main()
