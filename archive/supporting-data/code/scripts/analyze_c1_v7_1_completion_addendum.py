#!/usr/bin/env python3
"""Zero-GPU completion and answer-policy addendum for frozen C1/V7.1 results."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import html
import importlib.metadata
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.stats import binomtest, spearmanr
from transformers import AutoTokenizer

from cedar_m2.checkpoint import atomic_write_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "c1_v7_1_zero_gpu_addendum"

CUE_COMMIT = "1336f9dda660876dcfccb39842569eb93946c8f3"
NO_CUE_COMMIT = "deb4932318b11906c1008849e8c97ec960fb1d90"
CUE_ROOT = "artifacts/m2/c1_v7_heldout_confirmation"
NO_CUE_ROOT = "artifacts/m2/c1_v7_1_prompt_cue_ablation"
CUE_SENTENCE = " No new task information is supplied."

ARMS = ("N", "AO", "S")
CONTEXTS = ("cue", "no_cue")
CONDITIONS = ("uncapped", "p25")
SCORINGS = ("strict", "lenient")
CATEGORIES = ("initial", "target", "other_wrong", "invalid_or_truncated")
VALID_CATEGORIES = frozenset(CATEGORIES[:-1])
LAYERS = tuple(range(46, 54))
EARLY_LAYERS = tuple(range(46, 50))
LATE_LAYERS = tuple(range(50, 54))
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260722
TOKENIZER_REPO = "Qwen/Qwen3-32B"
TOKENIZER_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"

EXPECTED_RESULT_SHA256 = {
    "cue_uncapped_results": "d25d8017bde03ccbce4f60588fce2c7d616f3acfde5991901de91a1efcfb78a7",
    "cue_p25_results": "e86787777bbe24dc6088e7e098d69a8bc4cb063e5ffaf893384b0d08d6ddfc25",
    "no_cue_uncapped_results": (
        "849e47fc5ded864ad03d5e0b742a3c5cb999a0005cbb91ca0d38f77782644b22"
    ),
    "no_cue_p25_results": "b2d2b0c98bca4d8d58268dbb4af133ff29c80b4344821c11825e04a2c5f746e3",
}

INPUT_SPECS = {
    "cue_uncapped_results": (CUE_COMMIT, f"{CUE_ROOT}/formal-uncapped/results.jsonl"),
    "cue_uncapped_runtime": (
        CUE_COMMIT,
        f"{CUE_ROOT}/formal-uncapped/record_runtime.jsonl",
    ),
    "cue_p25_results": (CUE_COMMIT, f"{CUE_ROOT}/formal-p25/results.jsonl"),
    "cue_p25_runtime": (CUE_COMMIT, f"{CUE_ROOT}/formal-p25/record_runtime.jsonl"),
    "cue_audit": (CUE_COMMIT, f"{CUE_ROOT}/CROSS_CONDITION_AUDIT.json"),
    "no_cue_uncapped_results": (
        NO_CUE_COMMIT,
        f"{NO_CUE_ROOT}/formal-uncapped/results.jsonl",
    ),
    "no_cue_uncapped_runtime": (
        NO_CUE_COMMIT,
        f"{NO_CUE_ROOT}/formal-uncapped/record_runtime.jsonl",
    ),
    "no_cue_p25_results": (NO_CUE_COMMIT, f"{NO_CUE_ROOT}/formal-p25/results.jsonl"),
    "no_cue_p25_runtime": (
        NO_CUE_COMMIT,
        f"{NO_CUE_ROOT}/formal-p25/record_runtime.jsonl",
    ),
    "no_cue_analysis": (NO_CUE_COMMIT, f"{NO_CUE_ROOT}/C1_V7_1_ANALYSIS.json"),
    "no_cue_audit": (NO_CUE_COMMIT, f"{NO_CUE_ROOT}/CROSS_CONDITION_AUDIT.json"),
}


def run_git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def git_blob(commit: str, path: str) -> bytes:
    try:
        return run_git("show", f"{commit}:{path}")
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Cannot read frozen Git object {commit}:{path}: {message}") from exc


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must contain JSON objects")
    return rows


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def response_sha256(record: Mapping[str, Any]) -> str:
    return sha256_bytes(str(record.get("response", "")).encode("utf-8"))


def index_unique(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    indexed = {str(row.get("id")): row for row in rows}
    if len(indexed) != len(rows) or "None" in indexed:
        raise ValueError(f"{label} requires unique non-null IDs")
    return indexed


def validate_runtime(
    results: Sequence[Mapping[str, Any]],
    runtime_envelopes: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    runtime_rows = [envelope.get("result") for envelope in runtime_envelopes]
    if len(results) != len(runtime_rows):
        raise ValueError(f"{label}: results/runtime row count mismatch")
    for index, (result, runtime) in enumerate(zip(results, runtime_rows, strict=True)):
        if not isinstance(runtime, dict):
            raise ValueError(f"{label}: runtime row {index} lacks result")
        if runtime.get("id") != result.get("id"):
            raise ValueError(f"{label}: runtime ID mismatch at row {index}")
        if response_sha256(runtime) != response_sha256(result):
            raise ValueError(f"{label}: response hash mismatch for {result.get('id')}")
        if runtime.get("finish_reason") != result.get("finish_reason"):
            raise ValueError(f"{label}: finish reason mismatch for {result.get('id')}")
        if runtime != result:
            raise ValueError(f"{label}: materialized result differs from runtime checkpoint")


def aggregate_reference_exposure(
    record: Mapping[str, Any], phase: str, layers: Sequence[int] = LAYERS
) -> dict[str, float | int]:
    try:
        values = record["axis_telemetry"]["phases"][phase]["reference_cap_exposure"]["p25"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{record.get('id')}: missing {phase} reference exposure") from exc
    total = triggered = 0
    correction = 0.0
    for layer in layers:
        item = values[str(layer)]
        count = int(item["count"])
        layer_triggered = int(item["triggered"])
        layer_sum = float(item["correction_sum"])
        if not math.isclose(
            layer_sum,
            float(item["mean_correction_all_positions"]) * count,
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{record.get('id')}: inconsistent correction aggregate at layer {layer}"
            )
        total += count
        triggered += layer_triggered
        correction += layer_sum
    return {
        "count": total,
        "triggered": triggered,
        "trigger_rate": triggered / total,
        "correction_sum": correction,
        "mean_correction_per_position": correction / total,
    }


def aggregate_actual_cap(
    record: Mapping[str, Any], layers: Sequence[int] = LAYERS
) -> dict[str, float | int]:
    cap_stats = record.get("cap_stats")
    if not isinstance(cap_stats, dict):
        raise ValueError(f"{record.get('id')}: missing actual cap_stats")
    by_layer = cap_stats.get("by_layer")
    if not isinstance(by_layer, dict):
        raise ValueError(f"{record.get('id')}: malformed cap_stats.by_layer")
    total = triggered = 0
    correction = 0.0
    for layer in layers:
        item = by_layer[str(layer)]
        positions = int(item["total_positions"])
        total += positions
        triggered += int(item["triggered_positions"])
        correction += float(item["projection_correction_sum"])
    return {
        "count": total,
        "triggered": triggered,
        "trigger_rate": triggered / total,
        "correction_sum": correction,
        "mean_correction_per_position": correction / total,
    }


def validate_phase_counts(record: Mapping[str, Any]) -> None:
    phases = record.get("axis_telemetry", {}).get("phases", {})
    prompt_tokens = int(record["prompt_tokens"])
    response_tokens = int(record["response_tokens"])
    for phase, expected in (("prefill", prompt_tokens), ("decode", response_tokens - 1)):
        exposure = phases.get(phase, {}).get("reference_cap_exposure", {}).get("p25", {})
        if set(exposure) != {str(layer) for layer in LAYERS}:
            raise ValueError(f"{record.get('id')}: incomplete {phase} layer telemetry")
        for layer in LAYERS:
            if int(exposure[str(layer)]["count"]) != expected:
                raise ValueError(
                    f"{record.get('id')}: {phase} layer {layer} count does not equal {expected}"
                )
    if record.get("condition") == "p25":
        expected_actual = prompt_tokens + response_tokens - 1
        by_layer = record["cap_stats"]["by_layer"]
        if any(int(by_layer[str(layer)]["total_positions"]) != expected_actual for layer in LAYERS):
            raise ValueError(f"{record.get('id')}: actual cap positions include unexpected tokens")


def load_and_validate_inputs() -> dict[str, Any]:
    raw_inputs = {
        label: git_blob(commit, path) for label, (commit, path) in INPUT_SPECS.items()
    }
    for label, expected in EXPECTED_RESULT_SHA256.items():
        observed = sha256_bytes(raw_inputs[label])
        if observed != expected:
            raise ValueError(f"{label}: frozen SHA-256 mismatch: {observed} != {expected}")

    audits = {
        "cue": parse_json(raw_inputs["cue_audit"], "cue audit"),
        "no_cue": parse_json(raw_inputs["no_cue_audit"], "no-cue audit"),
    }
    for context, audit in audits.items():
        if audit.get("status") != "passed" or audit.get("content_retries") != 0:
            raise ValueError(f"{context}: audit is not passed with zero content retries")
    prior_analysis = parse_json(raw_inputs["no_cue_analysis"], "no-cue prior analysis")
    if prior_analysis.get("status") != "complete":
        raise ValueError("Frozen V7.1 analysis is incomplete")

    rows: dict[str, list[dict[str, Any]]] = {}
    for context in CONTEXTS:
        for condition in CONDITIONS:
            key = f"{context}_{condition}"
            results_key = f"{key}_results"
            runtime_key = f"{key}_runtime"
            rows[key] = parse_jsonl(raw_inputs[results_key], results_key)
            runtime = parse_jsonl(raw_inputs[runtime_key], runtime_key)
            validate_runtime(rows[key], runtime, key)

    if any(len(rows[f"no_cue_{condition}"]) != 144 for condition in CONDITIONS):
        raise ValueError("No-cue formal conditions must contain 144 records each")
    if any(len(rows[f"cue_{condition}"]) != 192 for condition in CONDITIONS):
        raise ValueError("Cue formal conditions must contain 192 records each")

    for context in CONTEXTS:
        for condition in CONDITIONS:
            for record in rows[f"{context}_{condition}"]:
                cap = record.get("cap_stats")
                if condition == "uncapped" and cap is not None:
                    raise ValueError(f"{record.get('id')}: uncapped record has actual cap_stats")
                if condition == "p25" and not isinstance(cap, dict):
                    raise ValueError(f"{record.get('id')}: p25 record lacks actual cap_stats")
                validate_phase_counts(record)

    cue = {
        condition: [row for row in rows[f"cue_{condition}"] if row.get("arm") in ARMS]
        for condition in CONDITIONS
    }
    no_cue = {condition: rows[f"no_cue_{condition}"] for condition in CONDITIONS}
    if any(len(cue[condition]) != 144 for condition in CONDITIONS):
        raise ValueError("Cue N/AO/S subset must contain 144 records")

    paired: list[dict[str, Any]] = []
    maps = {
        context: {condition: index_unique(data[condition], f"{context}_{condition}")
                  for condition in CONDITIONS}
        for context, data in (("cue", cue), ("no_cue", no_cue))
    }
    no_order = [str(row["id"]) for row in no_cue["uncapped"]]
    cue_source_order = [str(row["id"]) for row in cue["uncapped"]]
    if [item.removesuffix("-NOCUE") for item in no_order] != cue_source_order:
        raise ValueError("Cue/no-cue N/AO/S record order is not one-to-one")

    metadata = (
        "base_id",
        "family_id",
        "arm",
        "initial_answer",
        "target_answer",
        "distractor_answer",
        "gold_answer",
    )
    for no_id in no_order:
        if not no_id.endswith("-NOCUE"):
            raise ValueError(f"Malformed no-cue ID: {no_id}")
        source_id = no_id.removesuffix("-NOCUE")
        records = {
            "cue_uncapped": maps["cue"]["uncapped"][source_id],
            "cue_p25": maps["cue"]["p25"][source_id],
            "no_cue_uncapped": maps["no_cue"]["uncapped"][no_id],
            "no_cue_p25": maps["no_cue"]["p25"][no_id],
        }
        identity = tuple(records["no_cue_uncapped"].get(field) for field in metadata)
        if any(
            tuple(record.get(field) for field in metadata) != identity
            for record in records.values()
        ):
            raise ValueError(f"{no_id}: cue/no-cue identity metadata mismatch")
        if records["cue_uncapped"]["prompt"].replace(CUE_SENTENCE, "") != records[
            "no_cue_uncapped"
        ]["prompt"]:
            raise ValueError(f"{no_id}: no-cue prompt does not differ by only the frozen sentence")
        for context in CONTEXTS:
            uncapped = records[f"{context}_uncapped"]
            p25 = records[f"{context}_p25"]
            for field in (*metadata, "prompt", "prompt_sha256", "prompt_token_ids_sha256"):
                if uncapped.get(field) != p25.get(field):
                    raise ValueError(f"{no_id}: {context} cross-condition mismatch in {field}")
        for scoring in SCORINGS:
            for record in records.values():
                if record.get(f"{scoring}_category") not in CATEGORIES:
                    raise ValueError(f"{no_id}: invalid {scoring} category")
        paired.append(
            {
                "id": no_id,
                "source_id": source_id,
                "base_id": identity[0],
                "family_id": identity[1],
                "arm": identity[2],
                "initial_answer": identity[3],
                "target_answer": identity[4],
                "distractor_answer": identity[5],
                "records": records,
            }
        )

    if len(paired) != 144 or len({pair["family_id"] for pair in paired}) != 24:
        raise ValueError("Expected 144 pairs across 24 families")
    provenance = {}
    for label, raw in raw_inputs.items():
        commit, path = INPUT_SPECS[label]
        provenance[label] = {
            "commit": commit,
            "path": path,
            "git_blob_id": run_git("rev-parse", f"{commit}:{path}").decode().strip(),
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
        }
    return {"pairs": paired, "input_provenance": provenance, "audits": audits}


def describe(values: Sequence[float | int]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "p90": None,
            "max": None,
        }
    q1, median, q3, p90 = np.quantile(array, (0.25, 0.5, 0.75, 0.9))
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p90": float(p90),
        "max": float(np.max(array)),
    }


def family_bootstrap(
    pairs: Sequence[Mapping[str, Any]],
    statistic: Callable[[Sequence[Mapping[str, Any]]], float],
    *,
    samples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["family_id"])].append(pair)
    families = sorted(grouped)
    if len(families) != 24:
        raise ValueError(f"Family bootstrap expected 24 clusters, found {len(families)}")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = rng.choice(families, size=len(families), replace=True)
        sample_rows = [row for family in selected for row in grouped[str(family)]]
        draws.append(statistic(sample_rows))
    return {
        "estimate": float(statistic(pairs)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "samples": samples,
        "seed": seed,
    }


def record_for(pair: Mapping[str, Any], context: str, condition: str) -> Mapping[str, Any]:
    return pair["records"][f"{context}_{condition}"]


def category_for(
    pair: Mapping[str, Any], context: str, condition: str, scoring: str
) -> str:
    return str(record_for(pair, context, condition)[f"{scoring}_category"])


def transition_counts(
    pairs: Sequence[Mapping[str, Any]], context: str, arm: str, scoring: str
) -> dict[str, dict[str, int]]:
    selected = [pair for pair in pairs if pair["arm"] == arm]
    matrix = {before: {after: 0 for after in CATEGORIES} for before in CATEGORIES}
    for pair in selected:
        before = category_for(pair, context, "uncapped", scoring)
        after = category_for(pair, context, "p25", scoring)
        matrix[before][after] += 1
    return matrix


def rate_delta(
    pairs: Sequence[Mapping[str, Any]],
    context: str,
    arm: str,
    scoring: str,
    predicate: Callable[[str], bool],
) -> float:
    selected = [pair for pair in pairs if pair["arm"] == arm]
    before = np.mean(
        [predicate(category_for(pair, context, "uncapped", scoring)) for pair in selected]
    )
    after = np.mean([predicate(category_for(pair, context, "p25", scoring)) for pair in selected])
    return float(after - before)


def analyze_transitions(
    pairs: Sequence[Mapping[str, Any]], samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    highlighted = {
        "initial->target",
        "target->initial",
        "initial->other_wrong",
        "target->other_wrong",
        "invalid_or_truncated->target",
        "invalid_or_truncated->initial",
        "initial->invalid_or_truncated",
        "target->invalid_or_truncated",
        "other_wrong->invalid_or_truncated",
        "initial->initial",
        "target->target",
    }
    for context in CONTEXTS:
        report[context] = {}
        for scoring in SCORINGS:
            report[context][scoring] = {}
            for arm in ARMS:
                matrix = transition_counts(pairs, context, arm, scoring)
                selected = [pair for pair in pairs if pair["arm"] == arm]
                def target_stat(
                    rows: Sequence[Mapping[str, Any]],
                    c: str = context,
                    a: str = arm,
                    s: str = scoring,
                ) -> float:
                    return rate_delta(rows, c, a, s, lambda category: category == "target")

                def completion_stat(
                    rows: Sequence[Mapping[str, Any]],
                    c: str = context,
                    a: str = arm,
                    s: str = scoring,
                ) -> float:
                    return rate_delta(rows, c, a, s, lambda category: category in VALID_CATEGORIES)
                report[context][scoring][arm] = {
                    "n": len(selected),
                    "matrix": matrix,
                    "delta_target": family_bootstrap(
                        pairs, target_stat, samples=samples, seed=BOOTSTRAP_SEED
                    ),
                    "delta_completion": family_bootstrap(
                        pairs, completion_stat, samples=samples, seed=BOOTSTRAP_SEED
                    ),
                }
                for before in CATEGORIES:
                    for after in CATEGORIES:
                        transition = f"{before}->{after}"
                        csv_rows.append(
                            {
                                "context": context,
                                "scoring": scoring,
                                "arm": arm,
                                "from_category": before,
                                "to_category": after,
                                "count": matrix[before][after],
                                "total": len(selected),
                                "rate": matrix[before][after] / len(selected),
                                "required_named_transition": transition in highlighted,
                            }
                        )
    return report, csv_rows


def complete_case_analysis(
    pairs: Sequence[Mapping[str, Any]], samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    s_pairs = [pair for pair in pairs if pair["arm"] == "S"]
    complete = [
        pair
        for pair in s_pairs
        if category_for(pair, "no_cue", "uncapped", "strict") in {"initial", "target"}
    ]
    before_target = sum(
        category_for(pair, "no_cue", "uncapped", "strict") == "target" for pair in complete
    )
    after_target = sum(
        category_for(pair, "no_cue", "p25", "strict") == "target" for pair in complete
    )
    initial_target = sum(
        category_for(pair, "no_cue", "uncapped", "strict") == "initial"
        and category_for(pair, "no_cue", "p25", "strict") == "target"
        for pair in complete
    )
    target_initial = sum(
        category_for(pair, "no_cue", "uncapped", "strict") == "target"
        and category_for(pair, "no_cue", "p25", "strict") == "initial"
        for pair in complete
    )
    observed = (len(complete), before_target, after_target, initial_target, target_initial)
    expected = (42, 23, 21, 7, 9)
    if observed != expected:
        raise ValueError(
            f"Frozen complete-case counts changed: observed={observed}, expected={expected}"
        )

    def statistic(rows: Sequence[Mapping[str, Any]]) -> float:
        subset = [
            pair
            for pair in rows
            if pair["arm"] == "S"
            and category_for(pair, "no_cue", "uncapped", "strict") in {"initial", "target"}
        ]
        return float(
            np.mean(
                [category_for(pair, "no_cue", "p25", "strict") == "target" for pair in subset]
            )
            - np.mean(
                [
                    category_for(pair, "no_cue", "uncapped", "strict") == "target"
                    for pair in subset
                ]
            )
        )

    bootstrap = family_bootstrap(pairs, statistic, samples=samples)
    loo = {}
    for family in sorted({str(pair["family_id"]) for pair in pairs}):
        loo[family] = statistic([pair for pair in pairs if str(pair["family_id"]) != family])
    p_value = float(binomtest(initial_target, initial_target + target_initial, 0.5).pvalue)
    rows = []
    for pair in complete:
        before = category_for(pair, "no_cue", "uncapped", "strict")
        after = category_for(pair, "no_cue", "p25", "strict")
        rows.append(
            {
                "id": pair["id"],
                "base_id": pair["base_id"],
                "family_id": pair["family_id"],
                "initial_answer": pair["initial_answer"],
                "target_answer": pair["target_answer"],
                "uncapped_category": before,
                "p25_category": after,
                "transition": f"{before}->{after}",
            }
        )
    return {
        "analysis_role": "sensitivity_only_not_ITT",
        "selection_rule": "no-cue S uncapped strict category is initial or target",
        "n": len(complete),
        "uncapped_target_n": before_target,
        "p25_target_n": after_target,
        "paired_delta": after_target / len(complete) - before_target / len(complete),
        "initial_to_target": initial_target,
        "target_to_initial": target_initial,
        "discordant_pairs": initial_target + target_initial,
        "exact_two_sided_mcnemar_binomial_p": p_value,
        "family_bootstrap": bootstrap,
        "leave_one_family_out": {
            "values": loo,
            "min": min(loo.values()),
            "max": max(loo.values()),
        },
    }, rows


def bounds_for(
    pairs: Sequence[Mapping[str, Any]], context: str, scoring: str
) -> dict[str, float | int]:
    selected = [pair for pair in pairs if pair["arm"] == "S"]
    n = len(selected)
    uncapped_target = sum(
        category_for(pair, context, "uncapped", scoring) == "target" for pair in selected
    )
    uncapped_invalid = sum(
        category_for(pair, context, "uncapped", scoring) == "invalid_or_truncated"
        for pair in selected
    )
    p25_target = sum(
        category_for(pair, context, "p25", scoring) == "target" for pair in selected
    )
    bounds = partial_identification_bounds(
        n=n,
        uncapped_target=uncapped_target,
        uncapped_invalid=uncapped_invalid,
        p25_target=p25_target,
    )
    return {
        "n": n,
        "uncapped_target": uncapped_target,
        "uncapped_invalid": uncapped_invalid,
        "p25_target": p25_target,
        **bounds,
    }


def partial_identification_bounds(
    *, n: int, uncapped_target: int, uncapped_invalid: int, p25_target: int
) -> dict[str, float]:
    if n <= 0:
        raise ValueError("Partial-identification bound requires n > 0")
    counts = (uncapped_target, uncapped_invalid, p25_target)
    if any(count < 0 or count > n for count in counts):
        raise ValueError("Partial-identification counts must lie in [0, n]")
    if uncapped_target + uncapped_invalid > n:
        raise ValueError("Uncapped target and invalid counts cannot exceed n")
    return {
        "lower": p25_target / n - (uncapped_target + uncapped_invalid) / n,
        "upper": p25_target / n - uncapped_target / n,
    }


def analyze_missingness_bounds(
    pairs: Sequence[Mapping[str, Any]], samples: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scoring in SCORINGS:
        point = {context: bounds_for(pairs, context, scoring) for context in CONTEXTS}
        no_cue = point["no_cue"]
        cue = point["cue"]
        interaction = {
            "lower": float(no_cue["lower"] - cue["upper"]),
            "upper": float(no_cue["upper"] - cue["lower"]),
            "definition": "no-cue cap-effect bounds minus cue cap-effect bounds",
        }

        def bound_stat(
            rows: Sequence[Mapping[str, Any]],
            context: str,
            side: str,
            scoring_name: str = scoring,
        ) -> float:
            return float(bounds_for(rows, context, scoring_name)[side])

        boot = {
            context: {
                side: family_bootstrap(
                    pairs,
                    lambda rows, c=context, s=side: bound_stat(rows, c, s),
                    samples=samples,
                )
                for side in ("lower", "upper")
            }
            for context in CONTEXTS
        }
        interaction_boot = {
            "lower": family_bootstrap(
                pairs,
                lambda rows, scoring_name=scoring: float(
                    bounds_for(rows, "no_cue", scoring_name)["lower"]
                    - bounds_for(rows, "cue", scoring_name)["upper"]
                ),
                samples=samples,
            ),
            "upper": family_bootstrap(
                pairs,
                lambda rows, scoring_name=scoring: float(
                    bounds_for(rows, "no_cue", scoring_name)["upper"]
                    - bounds_for(rows, "cue", scoring_name)["lower"]
                ),
                samples=samples,
            ),
        }
        output[scoring] = {
            "point_bounds": point,
            "family_bootstrap": boot,
            "cue_by_cap_interaction_bounds": interaction,
            "cue_by_cap_interaction_bootstrap": interaction_boot,
        }
    strict = output["strict"]["point_bounds"]
    expected_no_cue = (-0.0625, 0.0625)
    observed_no_cue = (strict["no_cue"]["lower"], strict["no_cue"]["upper"])
    if not all(math.isclose(a, b) for a, b in zip(observed_no_cue, expected_no_cue, strict=True)):
        raise ValueError(f"No-cue S missingness bounds changed: {observed_no_cue}")
    return output


@functools.lru_cache(maxsize=1)
def fixed_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(
        TOKENIZER_REPO,
        revision=TOKENIZER_REVISION,
        local_files_only=True,
    )


def token_count(text: str) -> int:
    return len(fixed_tokenizer().encode(text, add_special_tokens=False))


def split_thinking_response(record: Mapping[str, Any]) -> tuple[str, str] | None:
    response = str(record.get("response", ""))
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return None
    prefix, remainder = response.split("<think>", 1)
    thinking, final = remainder.split("</think>", 1)
    if prefix:
        return None
    return thinking, final


def final_token_count(record: Mapping[str, Any]) -> int | None:
    boundary = record.get("boundary", {})
    close_index = boundary.get("think_close_last_token_index")
    if boundary.get("valid_thinking_boundary") and isinstance(close_index, int):
        value = int(record["response_tokens"]) - close_index - 1
    else:
        segments = split_thinking_response(record)
        if segments is None:
            return None
        _, final = segments
        eos = 1 if record.get("finish_reason") == "eos_token" else 0
        value = token_count(final) + eos
    if value < 0:
        raise ValueError(f"{record.get('id')}: invalid final-token boundary")
    return value


def length_value(record: Mapping[str, Any], metric: str) -> int | None:
    if metric == "response_tokens":
        return int(record["response_tokens"])
    if metric == "thinking_tokens":
        boundary = record.get("boundary", {})
        if not boundary.get("valid_thinking_boundary"):
            if bool(record.get("truncated")) and record.get("finish_reason") == "length":
                return int(record["response_tokens"])
            segments = split_thinking_response(record)
            return token_count(segments[0]) if segments is not None else None
        value = record.get("boundary", {}).get("thinking_tokens")
        return int(value) if isinstance(value, int | float) else None
    if metric == "final_tokens":
        return final_token_count(record)
    raise ValueError(f"Unknown length metric: {metric}")


def paired_length_delta(pair: Mapping[str, Any], context: str, metric: str) -> float | None:
    before = length_value(record_for(pair, context, "uncapped"), metric)
    after = length_value(record_for(pair, context, "p25"), metric)
    if before is None or after is None:
        return None
    return float(after - before)


def analyze_lengths(
    pairs: Sequence[Mapping[str, Any]], samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {"cells": {}, "paired_deltas": {}, "difference_in_differences": {}}
    csv_rows: list[dict[str, Any]] = []
    metrics = ("response_tokens", "thinking_tokens", "final_tokens")
    for context in CONTEXTS:
        report["cells"][context] = {}
        report["paired_deltas"][context] = {}
        for arm in ARMS:
            selected = [pair for pair in pairs if pair["arm"] == arm]
            report["cells"][context][arm] = {}
            report["paired_deltas"][context][arm] = {}
            for condition in CONDITIONS:
                records = [record_for(pair, context, condition) for pair in selected]
                cell = {
                    metric: describe(
                        [
                            value
                            for record in records
                            if (value := length_value(record, metric)) is not None
                        ]
                    )
                    for metric in metrics
                }
                cell.update(
                    {
                        "truncated_n": sum(bool(record.get("truncated")) for record in records),
                        "truncated_rate": float(
                            np.mean([bool(record.get("truncated")) for record in records])
                        ),
                        "invalid_n": sum(
                            record.get("strict_category") == "invalid_or_truncated"
                            for record in records
                        ),
                        "invalid_rate": float(
                            np.mean(
                                [
                                    record.get("strict_category") == "invalid_or_truncated"
                                    for record in records
                                ]
                            )
                        ),
                        "valid_thinking_boundary_n": sum(
                            bool(record.get("boundary", {}).get("valid_thinking_boundary"))
                            for record in records
                        ),
                        "finish_reasons": dict(
                            sorted(
                                Counter(
                                    str(record.get("finish_reason")) for record in records
                                ).items()
                            )
                        ),
                    }
                )
                report["cells"][context][arm][condition] = cell
                for metric in metrics:
                    stats = cell[metric]
                    csv_rows.append(
                        {
                            "row_type": "cell",
                            "context": context,
                            "arm": arm,
                            "condition": condition,
                            "metric": metric,
                            **stats,
                            "paired_ci_low": "",
                            "paired_ci_high": "",
                            "truncated_rate": cell["truncated_rate"],
                            "invalid_rate": cell["invalid_rate"],
                        }
                    )
            for metric in metrics:
                available = [
                    value
                    for pair in selected
                    if (value := paired_length_delta(pair, context, metric)) is not None
                ]

                def statistic(
                    rows: Sequence[Mapping[str, Any]],
                    c: str = context,
                    a: str = arm,
                    m: str = metric,
                ) -> float:
                    values = [
                        value
                        for pair in rows
                        if pair["arm"] == a
                        and (value := paired_length_delta(pair, c, m)) is not None
                    ]
                    return float(np.mean(values))

                bootstrap = family_bootstrap(pairs, statistic, samples=samples)
                delta = {"distribution": describe(available), "mean_family_bootstrap": bootstrap}
                report["paired_deltas"][context][arm][metric] = delta
                csv_rows.append(
                    {
                        "row_type": "paired_delta_p25_minus_uncapped",
                        "context": context,
                        "arm": arm,
                        "condition": "paired",
                        "metric": metric,
                        **delta["distribution"],
                        "paired_ci_low": bootstrap["ci_low"],
                        "paired_ci_high": bootstrap["ci_high"],
                        "truncated_rate": "",
                        "invalid_rate": "",
                    }
                )

        report["difference_in_differences"][context] = {}
        for metric in metrics:
            for contrast, left, right in (("S_minus_AO", "S", "AO"), ("S_minus_N", "S", "N")):

                def did_stat(
                    rows: Sequence[Mapping[str, Any]],
                    c: str = context,
                    m: str = metric,
                    a: str = left,
                    b: str = right,
                ) -> float:
                    arm_values = {}
                    for arm_name in (a, b):
                        arm_values[arm_name] = [
                            value
                            for pair in rows
                            if pair["arm"] == arm_name
                            and (value := paired_length_delta(pair, c, m)) is not None
                        ]
                    return float(np.mean(arm_values[a]) - np.mean(arm_values[b]))

                report["difference_in_differences"][context][f"{metric}_{contrast}"] = (
                    family_bootstrap(pairs, did_stat, samples=samples)
                )
    report["right_censoring_note"] = (
        "Outputs ending at max_new_tokens are right-censored; observed mean differences are "
        "descriptive and are not complete latent-length effects."
    )
    return report, csv_rows


def paired_margin_delta(
    pair: Mapping[str, Any], context: str, field: str
) -> float | None:
    values = []
    for condition in CONDITIONS:
        block = record_for(pair, context, condition).get(field)
        value = block.get("target_minus_initial") if isinstance(block, dict) else None
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return None
        values.append(float(value))
    return values[1] - values[0]


def analyze_margins(
    pairs: Sequence[Mapping[str, Any]], samples: int
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for context in CONTEXTS:
        report[context] = {}
        for arm in ARMS:
            report[context][arm] = {}
            for field in ("actual_answer_logits", "scaffold_answer_logits"):
                values = [
                    value
                    for pair in pairs
                    if pair["arm"] == arm
                    and (value := paired_margin_delta(pair, context, field)) is not None
                ]

                def statistic(
                    rows: Sequence[Mapping[str, Any]],
                    c: str = context,
                    a: str = arm,
                    f: str = field,
                ) -> float:
                    available = [
                        value
                        for pair in rows
                        if pair["arm"] == a
                        and (value := paired_margin_delta(pair, c, f)) is not None
                    ]
                    return float(np.mean(available))

                report[context][arm][field] = {
                    "distribution": describe(values),
                    "mean_family_bootstrap": family_bootstrap(
                        pairs, statistic, samples=samples
                    ),
                    "selection": "both conditions have a finite margin",
                }
    return report


def finite_or_none(value: float | np.floating[Any]) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def spearman_report(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {
        "n": len(x),
        "rho": finite_or_none(result.statistic),
        "p_value": finite_or_none(result.pvalue),
    }


def clustered_ols(
    rows: Sequence[Mapping[str, Any]],
    predictors: Sequence[tuple[str, Callable[[Mapping[str, Any]], float]]],
) -> dict[str, Any]:
    names = ["intercept", *(name for name, _ in predictors)]
    x = np.asarray(
        [[1.0, *(float(function(row)) for _, function in predictors)] for row in rows],
        dtype=float,
    )
    y = np.asarray([float(row["length_delta"]) for row in rows], dtype=float)
    groups = np.asarray([str(row["family_id"]) for row in rows])
    if len(rows) <= x.shape[1] or len(set(groups)) < 2:
        raise ValueError("Insufficient rows/clusters for regression")
    result = sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
    )
    intervals = np.asarray(result.conf_int(alpha=0.05), dtype=float)
    coefficients = {}
    for index, name in enumerate(names):
        coefficients[name] = {
            "estimate": finite_or_none(result.params[index]),
            "cluster_se": finite_or_none(result.bse[index]),
            "p_value": finite_or_none(result.pvalues[index]),
            "ci_low": finite_or_none(intervals[index, 0]),
            "ci_high": finite_or_none(intervals[index, 1]),
        }
    return {
        "n": len(rows),
        "family_clusters": len(set(groups)),
        "formula_columns": names,
        "coefficients": coefficients,
        "r_squared": finite_or_none(result.rsquared),
        "covariance": "family-cluster robust with finite-sample correction",
    }


def exposure_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for pair in pairs:
        for context in CONTEXTS:
            uncapped = record_for(pair, context, "uncapped")
            p25 = record_for(pair, context, "p25")
            prefill = aggregate_reference_exposure(uncapped, "prefill")
            prefill_early = aggregate_reference_exposure(uncapped, "prefill", EARLY_LAYERS)
            prefill_late = aggregate_reference_exposure(uncapped, "prefill", LATE_LAYERS)
            decode = aggregate_reference_exposure(uncapped, "decode")
            p25_prefill = aggregate_reference_exposure(p25, "prefill")
            p25_decode = aggregate_reference_exposure(p25, "decode")
            actual = aggregate_actual_cap(p25)
            actual_early = aggregate_actual_cap(p25, EARLY_LAYERS)
            actual_late = aggregate_actual_cap(p25, LATE_LAYERS)
            before = str(uncapped["strict_category"])
            after = str(p25["strict_category"])
            output.append(
                {
                    "id": pair["id"],
                    "source_id": pair["source_id"],
                    "base_id": pair["base_id"],
                    "family_id": pair["family_id"],
                    "arm": pair["arm"],
                    "context": context,
                    "transition": f"{before}->{after}",
                    "uncapped_truncated": bool(uncapped.get("truncated")),
                    "prompt_tokens": int(uncapped["prompt_tokens"]),
                    "uncapped_response_tokens": int(uncapped["response_tokens"]),
                    "p25_response_tokens": int(p25["response_tokens"]),
                    "uncapped_thinking_tokens_observed_or_censor_bound": length_value(
                        uncapped, "thinking_tokens"
                    ),
                    "p25_thinking_tokens_observed_or_censor_bound": length_value(
                        p25, "thinking_tokens"
                    ),
                    "length_delta": int(p25["response_tokens"]) - int(uncapped["response_tokens"]),
                    "prefill_trigger_rate": prefill["trigger_rate"],
                    "prefill_mean_correction": prefill["mean_correction_per_position"],
                    "prefill_early_mean_correction": prefill_early[
                        "mean_correction_per_position"
                    ],
                    "prefill_late_mean_correction": prefill_late["mean_correction_per_position"],
                    "uncapped_decode_trigger_rate": decode["trigger_rate"],
                    "uncapped_decode_mean_correction": decode[
                        "mean_correction_per_position"
                    ],
                    "uncapped_decode_correction_sum": decode["correction_sum"],
                    "p25_monitor_prefill_trigger_rate": p25_prefill["trigger_rate"],
                    "p25_monitor_prefill_mean_correction": p25_prefill[
                        "mean_correction_per_position"
                    ],
                    "p25_monitor_decode_trigger_rate": p25_decode["trigger_rate"],
                    "p25_monitor_decode_mean_correction": p25_decode[
                        "mean_correction_per_position"
                    ],
                    "p25_actual_trigger_rate": actual["trigger_rate"],
                    "p25_actual_mean_correction": actual["mean_correction_per_position"],
                    "p25_actual_correction_sum": actual["correction_sum"],
                    "p25_actual_early_mean_correction": actual_early[
                        "mean_correction_per_position"
                    ],
                    "p25_actual_late_mean_correction": actual_late[
                        "mean_correction_per_position"
                    ],
                }
            )
    return output


def arm_interaction_predictors(name: str) -> list[tuple[str, Callable[[Mapping[str, Any]], float]]]:
    return [
        (name, lambda row, n=name: float(row[n])),
        ("arm_AO", lambda row: float(row["arm"] == "AO")),
        ("arm_S", lambda row: float(row["arm"] == "S")),
        (f"{name}:arm_AO", lambda row, n=name: float(row[n]) * float(row["arm"] == "AO")),
        (f"{name}:arm_S", lambda row, n=name: float(row[n]) * float(row["arm"] == "S")),
    ]


def analyze_exposure(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = exposure_rows(pairs)
    models: dict[str, Any] = {}
    correlations: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    for context in CONTEXTS:
        selected = [row for row in rows if row["context"] == context]
        models[context] = {
            "main_prefill_mean_correction_by_arm": clustered_ols(
                selected, arm_interaction_predictors("prefill_mean_correction")
            ),
            "prefill_trigger_rate_by_arm": clustered_ols(
                selected, arm_interaction_predictors("prefill_trigger_rate")
            ),
            "early_late_prefill": clustered_ols(
                selected,
                [
                    (
                        "prefill_early_mean_correction",
                        lambda row: float(row["prefill_early_mean_correction"]),
                    ),
                    (
                        "prefill_late_mean_correction",
                        lambda row: float(row["prefill_late_mean_correction"]),
                    ),
                    ("arm_AO", lambda row: float(row["arm"] == "AO")),
                    ("arm_S", lambda row: float(row["arm"] == "S")),
                ],
            ),
            "p25_actual_mean_correction_post_treatment_descriptive": clustered_ols(
                selected, arm_interaction_predictors("p25_actual_mean_correction")
            ),
        }
        correlations[context] = {}
        for predictor in (
            "prefill_mean_correction",
            "prefill_trigger_rate",
            "p25_actual_mean_correction",
            "uncapped_decode_correction_sum",
        ):
            correlations[context][predictor] = spearman_report(
                [float(row[predictor]) for row in selected],
                [float(row["length_delta"]) for row in selected],
            )
        transitions[context] = {}
        for transition in sorted({str(row["transition"]) for row in selected}):
            group = [row for row in selected if row["transition"] == transition]
            transitions[context][transition] = {
                "n": len(group),
                "prefill_mean_correction": describe(
                    [float(row["prefill_mean_correction"]) for row in group]
                ),
                "length_delta": describe([float(row["length_delta"]) for row in group]),
            }

    no_cue_without_truncation = [
        row for row in rows if row["context"] == "no_cue" and not row["uncapped_truncated"]
    ]
    models["no_cue_excluding_six_uncapped_truncations"] = {
        "n": len(no_cue_without_truncation),
        "main_prefill_mean_correction_by_arm": clustered_ols(
            no_cue_without_truncation,
            arm_interaction_predictors("prefill_mean_correction"),
        ),
        "spearman": spearman_report(
            [float(row["prefill_mean_correction"]) for row in no_cue_without_truncation],
            [float(row["length_delta"]) for row in no_cue_without_truncation],
        ),
    }
    return {
        "analysis_boundary": {
            "main_predictor": "uncapped counterfactual prefill p25 exposure",
            "decode_correction_sum_is_descriptive_only": True,
            "p25_actual_exposure_is_post_treatment_descriptive_only": True,
            "causal_token_localization_not_claimed": True,
        },
        "models": models,
        "spearman": correlations,
        "transition_groups": transitions,
        "record_level": rows,
    }


def truncation_audit_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        pair
        for pair in pairs
        if pair["arm"] == "S"
        and category_for(pair, "no_cue", "uncapped", "strict") == "invalid_or_truncated"
    ]
    expected_ids = {
        "C1V7-004-S-NOCUE",
        "C1V7-006-S-NOCUE",
        "C1V7-007-S-NOCUE",
        "C1V7-010-S-NOCUE",
        "C1V7-016-S-NOCUE",
        "C1V7-040-S-NOCUE",
    }
    if {str(pair["id"]) for pair in selected} != expected_ids:
        raise ValueError("Frozen six no-cue S truncation IDs changed")
    output = []
    for pair in selected:
        uncapped = record_for(pair, "no_cue", "uncapped")
        p25 = record_for(pair, "no_cue", "p25")
        actual = aggregate_actual_cap(p25)
        output.append(
            {
                "id": pair["id"],
                "family_id": pair["family_id"],
                "initial_answer": pair["initial_answer"],
                "target_answer": pair["target_answer"],
                "uncapped_response_tokens": uncapped["response_tokens"],
                "uncapped_finish_reason": uncapped["finish_reason"],
                "uncapped_tail_300_chars": str(uncapped["response"])[-300:],
                "uncapped_last_observable_stance_note": (
                    "Unclosed thinking discusses conflict between initial "
                    f"{pair['initial_answer']} "
                    f"and target {pair['target_answer']}; no final JSON was observed or imputed."
                ),
                "p25_category": p25["strict_category"],
                "p25_answer": p25["strict_answer"],
                "p25_response_tokens": p25["response_tokens"],
                "p25_finish_reason": p25["finish_reason"],
                "p25_actual_trigger_rate": actual["trigger_rate"],
                "p25_actual_mean_correction_per_position": actual[
                    "mean_correction_per_position"
                ],
                "p25_actual_correction_sum": actual["correction_sum"],
                "final_label_imputed": False,
            }
        )
    return output


def xml_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def step_path(
    values: Sequence[int],
    x_map: Callable[[float], float],
    y_map: Callable[[float], float],
) -> str:
    if not values:
        return ""
    ordered = sorted(values)
    points = [(0.0, 0.0)]
    for index, value in enumerate(ordered, start=1):
        points.append((float(value), (index - 1) / len(ordered)))
        points.append((float(value), index / len(ordered)))
    return " ".join(
        ("M" if index == 0 else "L") + f" {x_map(x):.2f} {y_map(y):.2f}"
        for index, (x, y) in enumerate(points)
    )


def termination_path(
    records: Sequence[Mapping[str, Any]],
    x_map: Callable[[float], float],
    y_map: Callable[[float], float],
) -> str:
    n = len(records)
    events = sorted(
        int(record["response_tokens"])
        for record in records
        if not bool(record.get("truncated")) and record.get("finish_reason") != "length"
    )
    points = [(0.0, 0.0)]
    for index, value in enumerate(events, start=1):
        points.append((float(value), (index - 1) / n))
        points.append((float(value), index / n))
    return " ".join(
        ("M" if index == 0 else "L") + f" {x_map(x):.2f} {y_map(y):.2f}"
        for index, (x, y) in enumerate(points)
    )


def write_length_svg(
    path: Path, pairs: Sequence[Mapping[str, Any]], *, termination: bool
) -> None:
    width, height = 1120, 520
    margin_x, plot_w, plot_h = 70, 450, 390
    colors = {"N": "#1f77b4", "AO": "#ff7f0e", "S": "#2ca02c"}
    dashes = {"uncapped": "", "p25": "6 4"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,sans-serif;font-size:12px}</style>',
    ]
    for panel, context in enumerate(CONTEXTS):
        left = margin_x + panel * 540
        top = 55
        def x_map(value: float, panel_left: int = left) -> float:
            return panel_left + min(max(value, 0.0), 2048.0) / 2048.0 * plot_w

        def y_map(value: float, panel_top: int = top) -> float:
            return panel_top + plot_h - min(max(value, 0.0), 1.0) * plot_h
        parts.extend(
            [
                f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
                'fill="none" stroke="#444"/>',
                f'<text x="{left + plot_w / 2}" y="25" text-anchor="middle" '
                f'font-size="15">{xml_text(context)}</text>',
            ]
        )
        for tick in (0, 512, 1024, 1536, 2048):
            x = x_map(tick)
            parts.append(
                f'<line x1="{x}" y1="{top + plot_h}" x2="{x}" '
                f'y2="{top + plot_h + 5}" stroke="#444"/>'
            )
            parts.append(
                f'<text x="{x}" y="{top + plot_h + 20}" '
                f'text-anchor="middle">{tick}</text>'
            )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y_map(tick)
            parts.append(f'<line x1="{left - 5}" y1="{y}" x2="{left}" y2="{y}" stroke="#444"/>')
            parts.append(f'<text x="{left - 9}" y="{y + 4}" text-anchor="end">{tick:.2f}</text>')
        for arm in ARMS:
            for condition in CONDITIONS:
                records = [
                    record_for(pair, context, condition)
                    for pair in pairs
                    if pair["arm"] == arm
                ]
                if termination:
                    curve = termination_path(records, x_map, y_map)
                else:
                    curve = step_path(
                        [int(record["response_tokens"]) for record in records], x_map, y_map
                    )
                dash = f' stroke-dasharray="{dashes[condition]}"' if dashes[condition] else ""
                parts.append(
                    f'<path d="{curve}" fill="none" stroke="{colors[arm]}" '
                    f'stroke-width="1.8"{dash}/>'
                )
        parts.append(
            f'<text x="{left + plot_w / 2}" y="{top + plot_h + 48}" '
            'text-anchor="middle">response tokens</text>'
        )
    y_label = "cumulative observed termination" if termination else "empirical cumulative fraction"
    parts.append(
        '<text x="18" y="260" transform="rotate(-90 18 260)" '
        f'text-anchor="middle">{y_label}</text>'
    )
    legend_x, legend_y = 350, 495
    for index, arm in enumerate(ARMS):
        x = legend_x + index * 105
        parts.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 25}" y2="{legend_y}" '
            f'stroke="{colors[arm]}" stroke-width="2"/>'
        )
        parts.append(f'<text x="{x + 30}" y="{legend_y + 4}">{arm}</text>')
    parts.append('<line x1="680" y1="495" x2="705" y2="495" stroke="#444" stroke-width="2"/>')
    parts.append('<text x="710" y="499">uncapped</text>')
    parts.append(
        '<line x1="800" y1="495" x2="825" y2="495" stroke="#444" '
        'stroke-width="2" stroke-dasharray="6 4"/>'
    )
    parts.append('<text x="830" y="499">p25</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def analysis_provenance(
    input_provenance: Mapping[str, Any], *, require_committed_script: bool = True
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    relative = script_path.relative_to(ROOT).as_posix()
    head = run_git("rev-parse", "HEAD").decode().strip()
    local_script = script_path.read_bytes()
    script_matches_head = False
    try:
        committed_script = git_blob(head, relative)
        script_matches_head = committed_script == local_script
    except ValueError:
        if require_committed_script:
            raise
    if require_committed_script and not script_matches_head:
        raise ValueError("Analysis script must be committed unchanged before frozen output")
    return {
        "source_commits": {"cue": CUE_COMMIT, "no_cue": NO_CUE_COMMIT},
        "input_blobs": input_provenance,
        "analysis_script_commit": head,
        "analysis_script_path": relative,
        "analysis_script_sha256": sha256_bytes(local_script),
        "analysis_script_matches_head": script_matches_head,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "pandas": package_version("pandas"),
        "transformers": package_version("transformers"),
        "pillow": package_version("Pillow"),
        "matplotlib": package_version("matplotlib"),
        "tokenizer_fallback": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "local_files_only": True,
            "purpose": "recount closed thinking/final segments when runtime round-trip failed",
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "platform": platform.platform(),
        "gpu_used": False,
    }


def write_summary(path: Path, report: Mapping[str, Any]) -> None:
    complete = report["answer_policy_effect"]["complete_case42"]
    bounds = report["missingness_bounds"]["strict"]["point_bounds"]
    lengths = report["completion_deliberation_effect"]["paired_deltas"]["no_cue"]
    margin = report["answer_policy_effect"]["paired_margin"]["no_cue"]["S"]

    def pp(value: float) -> str:
        return f"{100 * value:+.2f}pp"

    lines = [
        "# C1/V7.1 零 GPU 完成过程补充分析",
        "",
        "本报告只复算冻结的 V7/V7.1 输出，没有加载模型、使用 GPU 或运行 capping。",
        "",
        "## 核心结果",
        "",
        (
            "- no-cue S 的 48 条 ITT：target 由 23/48 变为 26/48；completion 由 "
            "42/48 变为 48/48。"
        ),
        (
            f"- 由 uncapped 完成状态预先定义的 42 条 sensitivity：target 由 "
            f"{complete['uncapped_target_n']}/42 变为 {complete['p25_target_n']}/42，"
            f"paired delta {pp(complete['paired_delta'])}；initial→target "
            f"{complete['initial_to_target']} 条，target→initial "
            f"{complete['target_to_initial']} 条，exact p="
            f"{complete['exact_two_sided_mcnemar_binomial_p']:.4f}。"
        ),
        (
            f"- no-cue S 的截断识别边界为 [{pp(bounds['no_cue']['lower'])}, "
            f"{pp(bounds['no_cue']['upper'])}]；cue S 为 "
            f"[{pp(bounds['cue']['lower'])}, {pp(bounds['cue']['upper'])}]。"
        ),
        (
            f"- no-cue observed response length：N "
            f"{lengths['N']['response_tokens']['mean_family_bootstrap']['estimate']:+.1f}，"
            f"AO {lengths['AO']['response_tokens']['mean_family_bootstrap']['estimate']:+.1f}，"
            "S "
            f"{lengths['S']['response_tokens']['mean_family_bootstrap']['estimate']:+.1f} "
            "tokens。"
        ),
        (
            "- no-cue S 的 actual answer-boundary margin 只在两侧均可用的记录中变化均值为 "
            f"{margin['actual_answer_logits']['mean_family_bootstrap']['estimate']:+.3f} logit。"
        ),
        "",
        "## 解释边界",
        "",
        (
            "48 条表面 +6.25pp 同时包含 6 条 uncapped 截断恢复；在 42 条基线完整记录中，"
            "净迁移方向反而是 7 条 initial→target 对 9 条 target→initial。因而不能把全部 "
            "invalid→target 直接解释成更顺从。"
        ),
        (
            "2048-token 截断是右删失；长度均值是观察长度，不是潜在完整思考长度。exposure "
            "模型是相关分析，不定位导致翻转的 token，也不提供因果中介结论。"
        ),
        "",
        (
            "完整 strict/lenient 迁移、bootstrap、长度、margin、exposure 和六条截断审计"
            "见同目录机器可读文件。"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_delivery_manifest(output_root: Path, files: Sequence[Path]) -> dict[str, Any]:
    entries = []
    for path in sorted(files):
        try:
            relative_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative_path = path.relative_to(output_root).as_posix()
        entries.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "status": "complete",
        "manifest_self_excluded_to_avoid_recursive_hash": True,
        "files": entries,
    }


def analyze(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    require_committed_script: bool = True,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    loaded = load_and_validate_inputs()
    pairs = loaded["pairs"]
    transition_report, transition_rows = analyze_transitions(pairs, bootstrap_samples)
    complete_report, complete_rows = complete_case_analysis(pairs, bootstrap_samples)
    missingness = analyze_missingness_bounds(pairs, bootstrap_samples)
    length_report, length_rows = analyze_lengths(pairs, bootstrap_samples)
    margins = analyze_margins(pairs, bootstrap_samples)
    exposure = analyze_exposure(pairs)
    truncations = truncation_audit_rows(pairs)
    provenance = analysis_provenance(
        loaded["input_provenance"], require_committed_script=require_committed_script
    )

    report = {
        "status": "complete",
        "analysis_kind": "zero-GPU completion and answer-policy decomposition",
        "rows": 144,
        "families": 24,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "provenance": provenance,
        "input_audits": loaded["audits"],
        "four_state_transitions": transition_report,
        "answer_policy_effect": {
            "complete_case42": complete_report,
            "paired_margin": margins,
        },
        "missingness_bounds": missingness,
        "completion_deliberation_effect": length_report,
        "analysis_separation": {
            "invalid_to_target_is_completion_plus_observed_answer_not_pure_answer_policy": True,
            "complete_case42_is_sensitivity_not_primary_ITT": True,
            "right_censored_lengths_not_imputed": True,
        },
        "stopping_rule": "Step 1 complete; no new model or capping run authorized",
    }

    output_root.mkdir(parents=True, exist_ok=True)
    figures = output_root / "figures"
    paths = {
        "analysis": output_root / "analysis.json",
        "transitions": output_root / "four_state_transitions.csv",
        "complete": output_root / "complete_case42.csv",
        "bounds": output_root / "missingness_bounds.json",
        "lengths": output_root / "length_by_arm.csv",
        "exposure": output_root / "exposure_length_models.json",
        "truncations": output_root / "truncation_case_audit.csv",
        "summary": output_root / "C1_V7_1_ZERO_GPU_ADDENDUM_SUMMARY_ZH.md",
        "length_figure": figures / "response_length_ecdf.svg",
        "termination_figure": figures / "termination_cumulative.svg",
    }
    atomic_write_json(paths["analysis"], report)
    write_csv(paths["transitions"], transition_rows)
    write_csv(paths["complete"], complete_rows)
    atomic_write_json(paths["bounds"], missingness)
    write_csv(paths["lengths"], length_rows)
    atomic_write_json(paths["exposure"], exposure)
    write_csv(paths["truncations"], truncations)
    write_summary(paths["summary"], report)
    write_length_svg(paths["length_figure"], pairs, termination=False)
    write_length_svg(paths["termination_figure"], pairs, termination=True)
    manifest = build_delivery_manifest(output_root, list(paths.values()))
    atomic_write_json(output_root / "delivery_manifest.json", manifest)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(output_root=args.output_root, bootstrap_samples=args.bootstrap_samples)
    concise = {
        "status": report["status"],
        "rows": report["rows"],
        "families": report["families"],
        "bootstrap_samples": report["bootstrap_samples"],
        "output_root": args.output_root.as_posix(),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
