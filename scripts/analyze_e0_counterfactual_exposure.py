#!/usr/bin/env python3
"""Run the frozen zero-GPU E0 counterfactual-exposure analysis."""

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
OUTPUT_DIR = ROOT / "artifacts" / "m2" / "e0_counterfactual_exposure"
PROTOCOL = ROOT / "docs" / "E0_COUNTERFACTUAL_EXPOSURE_PROTOCOL.md"
RESULT_COMMIT = "3cbed5d"
SCORE_COMMIT = "86bf46b"
LAYERS = tuple(range(46, 54))
MODES = (
    "thinking_greedy",
    "nonthinking_greedy",
    "thinking_sampling",
    "nonthinking_sampling",
)
ARMS = ("N", "V", "F", "S")
CAPS = ("uncapped", "p25")
PHASES = ("prefill", "decode")
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_REPLICATES = 10_000
DATA_PATH = "datasets/m2_2b_formal/v6_factorial_diagnostic/m2_2b_v6_factorial144.jsonl"
STRICT_PATH = (
    "artifacts/m2/m2_2b_v6_zero_gpu_reaudit/revealed/strict_record_scores.jsonl"
)
LENIENT_PATH = (
    "artifacts/m2/m2_2b_v6_zero_gpu_reaudit/revealed/lenient_record_scores.jsonl"
)
EXPECTED_DATA_SHA256 = "495bb79d661e3d91d34bba435b0e92bc687da76844395673e949a430852893d4"
EXPECTED_STRICT_SHA256 = "772e901f1023f9f0b8de95901093f50845fe490f2e01fb421990235ec3adc809"
EXPECTED_LENIENT_SHA256 = "8943ed8b0d3b6fffeaf8076492e0e4b657c660eb22e08627d281f7c43081b672"
EXPECTED_RUN_HASHES = {
    ("nonthinking_greedy", "p25"): {
        "manifest": "c1bb5e4e104ab90675e983217de638e156b01aaed72541316451e7b330817854",
        "results": "faeafd4ceb31dddd803e5ce00ba349bd50f7f8c35320b580133e0a71e89f02ef",
    },
    ("nonthinking_greedy", "uncapped"): {
        "manifest": "45ab4efebfe87c7e7a84d7731558a09d0c37bba20f661f93481a8e17190f2e5a",
        "results": "4fb844b1bb96120d84823bee561e89515cc7d4879354f73fc281218d256b8411",
    },
    ("nonthinking_sampling", "p25"): {
        "manifest": "c9513188e6dff580a39032665aedd601310a937e6ebdc187ab243c08f425486f",
        "results": "8070ee8d3db35a50de5a40cd0a1f25fe512cb51583b8bffa69d1776d700de69a",
    },
    ("nonthinking_sampling", "uncapped"): {
        "manifest": "07aa5e4f6013a494bf4ec835886676b4a92e63ffd60cdedf1489ff4e1e574542",
        "results": "6077682fa516711efe9ed97a0d7f08f5c0d806a20e95b6020cf252fb5b984787",
    },
    ("thinking_greedy", "p25"): {
        "manifest": "f128124fd356df19236d8c3331b12f928ca50cbef2a2b2ad0a612257a0d4d41f",
        "results": "b96e209992f2a8d93347c673e3020aac00dfe28f15c2db5241384c7ee4041659",
    },
    ("thinking_greedy", "uncapped"): {
        "manifest": "e7ff9ed592ff416efab49dbcf958b0b08f1517a9f765fde0417bfd1ce08f3411",
        "results": "980da4d48f0c1e74b53bbb5663c360779913ef2678e5738d924a0c9a285bfb3d",
    },
    ("thinking_sampling", "p25"): {
        "manifest": "c8ba6c89596343530ee29a8bb89b93ad88d6f75efedd7fd960d863be1bb738f6",
        "results": "6ce1c71f73b9b26704dc56aebf8d3679306f0cd01a129294b7f3e03727618335",
    },
    ("thinking_sampling", "uncapped"): {
        "manifest": "31b83ddcf9a57636a21957580ff63bc463c0a602f6c98823eeb8d036ab1e2273",
        "results": "23cec34bc643ff0a30682a4d5e0fbd80e311dd15aa2f05f430460604129f9240",
    },
}
GROUPS = {
    "layers_46_49": tuple(range(46, 50)),
    "layers_50_53": tuple(range(50, 54)),
    "layers_46_53": LAYERS,
}
PRIMARY_LEFT = "initial_to_target"
PRIMARY_RIGHT = "stable_initial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def git_text(commit: str, path: str) -> str:
    return git_bytes(commit, path).decode("utf-8")


def blob_sha256(commit: str, path: str) -> str:
    return hashlib.sha256(git_bytes(commit, path)).hexdigest()


def parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at in-memory JSONL line {line_number}")
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


def run_name(mode: str, cap: str) -> str:
    return f"qwen3_32b_v6_formal_{mode}_{cap}_seed0_development"


def run_path(mode: str, cap: str, filename: str) -> str:
    return (
        "artifacts/m2/m2_2b_v6_factorial_diagnostic/"
        f"{run_name(mode, cap)}/{filename}"
    )


def load_sources() -> dict[str, Any]:
    dataset = parse_jsonl_text(git_text(RESULT_COMMIT, DATA_PATH))
    strict = parse_jsonl_text(git_text(SCORE_COMMIT, STRICT_PATH))
    lenient = parse_jsonl_text(git_text(SCORE_COMMIT, LENIENT_PATH))
    results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifests: dict[tuple[str, str], dict[str, Any]] = {}
    for mode in MODES:
        for cap in CAPS:
            results[(mode, cap)] = parse_jsonl_text(
                git_text(RESULT_COMMIT, run_path(mode, cap, "results.jsonl"))
            )
            manifest = json.loads(
                git_text(RESULT_COMMIT, run_path(mode, cap, "manifest.json"))
            )
            if not isinstance(manifest, dict):
                raise ValueError(f"{mode}/{cap}: manifest is not an object")
            manifests[(mode, cap)] = manifest
    return {
        "dataset": dataset,
        "strict": strict,
        "lenient": lenient,
        "results": results,
        "manifests": manifests,
    }


def rounded(value: float | None, digits: int = 10) -> float | None:
    return None if value is None else round(float(value), digits)


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": rounded(float(array.mean())),
        "median": rounded(float(np.median(array))),
        "std": rounded(float(array.std(ddof=1))) if len(array) > 1 else 0.0,
        "min": rounded(float(array.min())),
        "q25": rounded(float(np.quantile(array, 0.25))),
        "q75": rounded(float(np.quantile(array, 0.75))),
        "max": rounded(float(array.max())),
    }


def input_manifest(sources: dict[str, Any]) -> dict[str, Any]:
    dataset = sources["dataset"]
    dataset_ids = [row.get("id") for row in dataset]
    checks: dict[str, bool] = {
        "dataset_sha256": blob_sha256(RESULT_COMMIT, DATA_PATH) == EXPECTED_DATA_SHA256,
        "strict_scores_sha256": blob_sha256(SCORE_COMMIT, STRICT_PATH)
        == EXPECTED_STRICT_SHA256,
        "lenient_scores_sha256": blob_sha256(SCORE_COMMIT, LENIENT_PATH)
        == EXPECTED_LENIENT_SHA256,
        "dataset_144_unique": len(dataset) == 144 and len(set(dataset_ids)) == 144,
        "dataset_arms_balanced": Counter(row.get("arm") for row in dataset)
        == Counter({arm: 36 for arm in ARMS}),
    }
    run_files: dict[str, Any] = {}
    for mode in MODES:
        for cap in CAPS:
            key = (mode, cap)
            results_path = run_path(mode, cap, "results.jsonl")
            manifest_path = run_path(mode, cap, "manifest.json")
            results_hash = blob_sha256(RESULT_COMMIT, results_path)
            manifest_hash = blob_sha256(RESULT_COMMIT, manifest_path)
            rows = sources["results"][key]
            manifest = sources["manifests"][key]
            expected = EXPECTED_RUN_HASHES[key]
            run_checks = {
                "results_sha256": results_hash == expected["results"],
                "manifest_sha256": manifest_hash == expected["manifest"],
                "144_rows": len(rows) == 144,
                "ids_unique_ordered": [row.get("id") for row in rows] == dataset_ids,
                "mode_and_cap": all(
                    row.get("mode") == mode and row.get("cap") == cap for row in rows
                ),
                "manifest_dataset_hash": manifest.get("configuration", {})
                .get("dataset", {})
                .get("sha256")
                == EXPECTED_DATA_SHA256,
            }
            checks[f"run_{mode}_{cap}"] = all(run_checks.values())
            run_files[f"{mode}/{cap}"] = {
                "results_path": results_path,
                "results_sha256": results_hash,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_hash,
                "checks": run_checks,
            }
    expected_score_keys = {
        (mode, cap, item_id) for mode in MODES for cap in CAPS for item_id in dataset_ids
    }
    for scoring in ("strict", "lenient"):
        rows = sources[scoring]
        keys = {(row.get("mode"), row.get("cap"), row.get("source_id")) for row in rows}
        checks[f"{scoring}_1152_exact_unique"] = (
            len(rows) == 1152 and len(keys) == 1152 and keys == expected_score_keys
        )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "result_commit": RESULT_COMMIT,
        "score_commit": SCORE_COMMIT,
        "checks": checks,
        "failures": failures,
        "files": {
            "dataset": {"path": DATA_PATH, "sha256": blob_sha256(RESULT_COMMIT, DATA_PATH)},
            "strict_scores": {
                "path": STRICT_PATH,
                "sha256": blob_sha256(SCORE_COMMIT, STRICT_PATH),
            },
            "lenient_scores": {
                "path": LENIENT_PATH,
                "sha256": blob_sha256(SCORE_COMMIT, LENIENT_PATH),
            },
            "runs": run_files,
        },
        "counts": {
            "dataset_rows": len(dataset),
            "formal_result_rows": sum(len(rows) for rows in sources["results"].values()),
            "strict_score_rows": len(sources["strict"]),
            "lenient_score_rows": len(sources["lenient"]),
        },
        "v5_used_as_primary_mechanism_evidence": False,
    }


def telemetry_availability(sources: dict[str, Any]) -> dict[str, Any]:
    field_failures: list[dict[str, Any]] = []
    uncapped_rows = 0
    phase_layer_records = 0
    for mode in MODES:
        manifest = sources["manifests"][(mode, "uncapped")]
        telemetry_contract = manifest.get("configuration", {}).get("axis_telemetry", {})
        if (
            tuple(telemetry_contract.get("layers", ())) != LAYERS
            or set(telemetry_contract.get("phases", ())) != set(PHASES)
            or "p25" not in telemetry_contract.get("reference_caps", ())
        ):
            field_failures.append({"mode": mode, "kind": "manifest_telemetry_contract"})
        for row in sources["results"][(mode, "uncapped")]:
            uncapped_rows += 1
            telemetry = row.get("axis_telemetry", {}).get("phases", {})
            if not isinstance(row.get("response"), str) or not isinstance(
                row.get("response_tokens"), int
            ):
                field_failures.append({"mode": mode, "id": row.get("id"), "kind": "response"})
            for phase in PHASES:
                exposure = (
                    telemetry.get(phase, {})
                    .get("reference_cap_exposure", {})
                    .get("p25", {})
                )
                projections = telemetry.get(phase, {}).get("assistant_axis_projection", {})
                if tuple(sorted(map(int, exposure))) != LAYERS or tuple(
                    sorted(map(int, projections))
                ) != LAYERS:
                    field_failures.append(
                        {"mode": mode, "id": row.get("id"), "phase": phase, "kind": "layers"}
                    )
                    continue
                for layer in map(str, LAYERS):
                    item = exposure[layer]
                    if not all(
                        field in item
                        and isinstance(item[field], int | float)
                        and math.isfinite(float(item[field]))
                        for field in ("count", "triggered", "trigger_rate", "mean_correction")
                    ):
                        field_failures.append(
                            {
                                "mode": mode,
                                "id": row.get("id"),
                                "phase": phase,
                                "layer": int(layer),
                                "kind": "exposure_fields",
                            }
                        )
                    phase_layer_records += 1
    coarse_complete = uncapped_rows == 576 and phase_layer_records == 576 * 2 * 8
    coarse_complete = coarse_complete and not field_failures
    return {
        "status": "passed_with_coarse_phase_limit" if coarse_complete else "failed",
        "route": (
            "proceed_zero_gpu_aggregate_phase_E0"
            if coarse_complete
            else "stop_and_specify_monitor_only_missing_fields"
        ),
        "uncapped_records_audited": uncapped_rows,
        "phase_layer_records_audited": phase_layer_records,
        "uncapped_p25_counterfactual_exposure_complete": coarse_complete,
        "available": {
            "layers_46_53": coarse_complete,
            "p25_thresholds_in_manifest": coarse_complete,
            "count_trigger_rate_mean_correction": coarse_complete,
            "correction_sum_exactly_reconstructible": coarse_complete,
            "prefill_decode_split": coarse_complete,
            "raw_response": coarse_complete,
            "response_token_count": coarse_complete,
            "thinking_integrity": True,
        },
        "not_available": {
            "token_position_exposure_arrays": True,
            "thinking_vs_preanswer_vs_answer_split": True,
            "last_input_token_exposure": True,
            "finish_reason": True,
            "answer_logits_or_initial_margin": True,
        },
        "reconstruction_rule": "correction_sum = mean_correction * count",
        "forbidden_reconstruction": "never multiply mean_correction by trigger_rate or triggered",
        "field_failures": field_failures,
        "interpretation_limit": (
            "Decode is an aggregate over all generated positions. It cannot identify the exact "
            "thinking, pre-answer, or answer moment."
        ),
    }


def classify_transition(before: str, after: str, truncated: bool) -> str:
    if truncated or "invalid" in {before, after}:
        return "invalid_or_truncated"
    frozen = {
        ("initial", "target"): "initial_to_target",
        ("target", "initial"): "target_to_initial",
        ("initial", "other_wrong"): "initial_to_other_wrong",
        ("target", "other_wrong"): "target_to_other_wrong",
        ("initial", "initial"): "stable_initial",
        ("target", "target"): "stable_target",
    }
    return frozen.get((before, after), f"other_transition:{before}_to_{after}")


def build_transitions(sources: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = {str(row["id"]): row for row in sources["dataset"]}
    score_maps: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for scoring in ("strict", "lenient"):
        score_maps[scoring] = {
            (str(row["mode"]), str(row["cap"]), str(row["source_id"])): row
            for row in sources[scoring]
        }
    transitions: list[dict[str, Any]] = []
    for mode in MODES:
        for item_id, metadata in dataset.items():
            scoring_rows: dict[str, Any] = {}
            for scoring in ("strict", "lenient"):
                before = score_maps[scoring][(mode, "uncapped", item_id)]
                after = score_maps[scoring][(mode, "p25", item_id)]
                truncated = bool(before.get("truncated") or after.get("truncated"))
                scoring_rows[scoring] = {
                    "before_outcome": before["outcome"],
                    "after_outcome": after["outcome"],
                    "before_answer": before.get("answer"),
                    "after_answer": after.get("answer"),
                    "before_truncated": bool(before.get("truncated")),
                    "after_truncated": bool(after.get("truncated")),
                    "transition": classify_transition(
                        str(before["outcome"]), str(after["outcome"]), truncated
                    ),
                }
            arm = str(metadata["arm"])
            transitions.append(
                {
                    "id": item_id,
                    "mode": mode,
                    "arm": arm,
                    "family": metadata["template_family"],
                    "base_id": metadata["source_candidate_base_id"],
                    "initial_answer_id": metadata["initial_answer_id"],
                    "target_answer_id": metadata["induced_target_answer_id"],
                    "gold_final_answer_id": metadata["gold_final_answer_id"],
                    "target_semantics": (
                        "correct_update" if arm == "V" else "induced_error" if arm in {"F", "S"}
                        else "not_normative_target"
                    ),
                    "initial_semantics": (
                        "correct_maintain" if arm == "N" else "pre_observation_answer"
                    ),
                    "strict": scoring_rows["strict"],
                    "lenient": scoring_rows["lenient"],
                }
            )
    if len(transitions) != 576:
        raise ValueError(f"Expected 576 mode-by-item transitions, found {len(transitions)}")
    return transitions


def _layer_exposure(result: dict[str, Any], phase: str, layer: int) -> dict[str, Any]:
    item = result["axis_telemetry"]["phases"][phase]["reference_cap_exposure"]["p25"][
        str(layer)
    ]
    count = int(item["count"])
    triggered = int(item["triggered"])
    mean_correction = float(item["mean_correction"])
    correction_sum = mean_correction * count
    return {
        "count": count,
        "triggered": triggered,
        "trigger_rate": triggered / count if count else 0.0,
        "mean_correction": mean_correction,
        "correction_sum": correction_sum,
    }


def _aggregate_layers(
    by_layer: dict[str, dict[str, Any]], layers: tuple[int, ...]
) -> dict[str, Any]:
    selected = [by_layer[str(layer)] for layer in layers]
    count = sum(int(item["count"]) for item in selected)
    triggered = sum(int(item["triggered"]) for item in selected)
    correction = sum(float(item["correction_sum"]) for item in selected)
    return {
        "layers": list(layers),
        "count": count,
        "triggered": triggered,
        "trigger_rate": triggered / count if count else 0.0,
        "correction_sum": correction,
        "normalized_correction": correction / count if count else 0.0,
    }


def build_exposures(
    sources: dict[str, Any], transitions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result_maps = {
        mode: {str(row["id"]): row for row in sources["results"][(mode, "uncapped")]}
        for mode in MODES
    }
    exposures: list[dict[str, Any]] = []
    for transition in transitions:
        result = result_maps[str(transition["mode"])][str(transition["id"])]
        phases: dict[str, Any] = {}
        for phase in PHASES:
            by_layer = {
                str(layer): _layer_exposure(result, phase, layer) for layer in LAYERS
            }
            phases[phase] = {
                "by_layer": by_layer,
                **{
                    group_name: _aggregate_layers(by_layer, group_layers)
                    for group_name, group_layers in GROUPS.items()
                },
            }
        exposures.append(
            {
                "id": transition["id"],
                "mode": transition["mode"],
                "arm": transition["arm"],
                "family": transition["family"],
                "strict_transition": transition["strict"]["transition"],
                "lenient_transition": transition["lenient"]["transition"],
                "prompt_tokens": int(result["prompt_tokens"]),
                "response_tokens": int(result["response_tokens"]),
                "phases": phases,
                "position_level": "not_available",
                "preanswer_answer_split": "not_available",
                "source": "uncapped trajectory p25 reference exposure",
            }
        )
    return exposures


def feature_names() -> tuple[str, ...]:
    return tuple(
        f"{phase}.{group}.{metric}"
        for phase in PHASES
        for group in GROUPS
        for metric in ("normalized_correction", "trigger_rate")
    )


def exposure_feature(row: dict[str, Any], name: str) -> float:
    phase, group, metric = name.split(".")
    return float(row["phases"][phase][group][metric])


def family_cluster_contrast(
    transitions: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    *,
    mode: str,
    arm: str,
    left_transition: str = PRIMARY_LEFT,
    right_transition: str = PRIMARY_RIGHT,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    exposure_map = {(row["mode"], row["id"]): row for row in exposures}
    selected = [row for row in transitions if row["mode"] == mode and row["arm"] == arm]
    categories = np.asarray([row["strict"]["transition"] for row in selected], dtype=object)
    features = feature_names()
    matrix = np.asarray(
        [
            [exposure_feature(exposure_map[(row["mode"], row["id"])], name) for name in features]
            for row in selected
        ],
        dtype=float,
    )
    family_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(selected):
        family_indices[str(row["family"])].append(index)
    families = sorted(family_indices)
    left_mask = categories == left_transition
    right_mask = categories == right_transition
    report: dict[str, Any] = {
        "mode": mode,
        "arm": arm,
        "left_transition": left_transition,
        "right_transition": right_transition,
        "left_n": int(left_mask.sum()),
        "right_n": int(right_mask.sum()),
        "family_clusters": len(families),
        "bootstrap_replicates_requested": replicates,
        "bootstrap_seed": seed,
        "features": {},
    }
    if not left_mask.any() or not right_mask.any():
        report["status"] = "not_estimable_missing_group"
        return report
    rng = np.random.default_rng(seed)
    mean_replicates: list[list[float]] = [[] for _name in features]
    median_replicates: list[list[float]] = [[] for _name in features]
    valid = 0
    for _replicate in range(replicates):
        sampled_families = rng.integers(0, len(families), size=len(families))
        indices = np.concatenate(
            [
                np.asarray(family_indices[families[index]], dtype=np.int64)
                for index in sampled_families
            ]
        )
        sampled_categories = categories[indices]
        left = sampled_categories == left_transition
        right = sampled_categories == right_transition
        if not left.any() or not right.any():
            continue
        valid += 1
        sampled_matrix = matrix[indices]
        for feature_index in range(len(features)):
            left_values = sampled_matrix[left, feature_index]
            right_values = sampled_matrix[right, feature_index]
            mean_replicates[feature_index].append(
                float(left_values.mean() - right_values.mean())
            )
            median_replicates[feature_index].append(
                float(np.median(left_values) - np.median(right_values))
            )
    report["status"] = "estimated"
    report["bootstrap_replicates_valid"] = valid
    for feature_index, name in enumerate(features):
        left_values = matrix[left_mask, feature_index]
        right_values = matrix[right_mask, feature_index]
        mean_rep = mean_replicates[feature_index]
        median_rep = median_replicates[feature_index]
        report["features"][name] = {
            "left": describe(left_values.tolist()),
            "right": describe(right_values.tolist()),
            "mean_difference": rounded(float(left_values.mean() - right_values.mean())),
            "mean_difference_ci95": (
                [
                    rounded(float(np.quantile(mean_rep, 0.025))),
                    rounded(float(np.quantile(mean_rep, 0.975))),
                ]
                if mean_rep
                else [None, None]
            ),
            "median_difference": rounded(
                float(np.median(left_values) - np.median(right_values))
            ),
            "median_difference_ci95": (
                [
                    rounded(float(np.quantile(median_rep, 0.025))),
                    rounded(float(np.quantile(median_rep, 0.975))),
                ]
                if median_rep
                else [None, None]
            ),
        }
    return report


def transition_matrices(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    outcomes = ("initial", "target", "other_wrong", "invalid")
    for mode in MODES:
        output[mode] = {}
        for arm in ARMS:
            selected = [row for row in transitions if row["mode"] == mode and row["arm"] == arm]
            matrix = {
                before: {after: 0 for after in outcomes}
                for before in outcomes
            }
            labels: Counter[str] = Counter()
            for row in selected:
                before = str(row["strict"]["before_outcome"])
                after = str(row["strict"]["after_outcome"])
                matrix[before][after] += 1
                labels[str(row["strict"]["transition"])] += 1
            output[mode][arm] = {
                "n": len(selected),
                "matrix_uncapped_to_p25": matrix,
                "transition_labels": dict(labels),
            }
    return output


def transition_exposure_descriptives(
    transitions: list[dict[str, Any]], exposures: list[dict[str, Any]]
) -> dict[str, Any]:
    exposure_map = {(row["mode"], row["id"]): row for row in exposures}
    output: dict[str, Any] = {}
    for mode in MODES:
        output[mode] = {}
        for arm in ARMS:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for transition in transitions:
                if transition["mode"] == mode and transition["arm"] == arm:
                    groups[str(transition["strict"]["transition"])].append(
                        exposure_map[(mode, transition["id"])]
                    )
            output[mode][arm] = {
                transition: {
                    "n": len(rows),
                    "features": {
                        name: describe([exposure_feature(row, name) for row in rows])
                        for name in feature_names()
                    },
                }
                for transition, rows in sorted(groups.items())
            }
    return output


def saturation_summary(exposures: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in MODES:
        output[mode] = {}
        for arm in ARMS:
            selected = [row for row in exposures if row["mode"] == mode and row["arm"] == arm]
            output[mode][arm] = {
                name: {
                    **describe([exposure_feature(row, name) for row in selected]),
                    "coefficient_of_variation": (
                        rounded(
                            float(
                                np.std(
                                    [exposure_feature(row, name) for row in selected], ddof=1
                                )
                                / np.mean([exposure_feature(row, name) for row in selected])
                            )
                        )
                        if np.mean([exposure_feature(row, name) for row in selected]) != 0
                        else None
                    ),
                }
                for name in feature_names()
            }
    return output


def leave_one_family_out_primary(
    transitions: list[dict[str, Any]], exposures: list[dict[str, Any]]
) -> dict[str, Any]:
    exposure_map = {(row["mode"], row["id"]): row for row in exposures}
    selected = [
        row for row in transitions if row["mode"] == "thinking_greedy" and row["arm"] == "S"
    ]
    families = sorted({str(row["family"]) for row in selected})
    features = (
        "prefill.layers_46_53.normalized_correction",
        "decode.layers_46_53.normalized_correction",
    )
    entries: list[dict[str, Any]] = []
    for omitted in families:
        retained = [row for row in selected if row["family"] != omitted]
        left = [row for row in retained if row["strict"]["transition"] == PRIMARY_LEFT]
        right = [row for row in retained if row["strict"]["transition"] == PRIMARY_RIGHT]
        values: dict[str, Any] = {}
        for feature in features:
            if not left or not right:
                values[feature] = None
            else:
                values[feature] = rounded(
                    statistics.fmean(
                        exposure_feature(exposure_map[(row["mode"], row["id"])], feature)
                        for row in left
                    )
                    - statistics.fmean(
                        exposure_feature(exposure_map[(row["mode"], row["id"])], feature)
                        for row in right
                    )
                )
        entries.append(
            {
                "omitted_family": omitted,
                "left_n": len(left),
                "right_n": len(right),
                "mean_differences": values,
            }
        )
    return {
        "families": len(families),
        "entries": entries,
        "all_estimable": all(
            all(value is not None for value in item["mean_differences"].values())
            for item in entries
        ),
    }


def strict_lenient_agreement(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    same = sum(row["strict"]["transition"] == row["lenient"]["transition"] for row in transitions)
    disagreements = [
        {
            "id": row["id"],
            "mode": row["mode"],
            "arm": row["arm"],
            "strict": row["strict"]["transition"],
            "lenient": row["lenient"]["transition"],
        }
        for row in transitions
        if row["strict"]["transition"] != row["lenient"]["transition"]
    ]
    return {
        "records": len(transitions),
        "same_transition": same,
        "agreement_rate": same / len(transitions),
        "disagreements": disagreements,
    }


def build_primary_results(
    transitions: list[dict[str, Any]], exposures: list[dict[str, Any]]
) -> dict[str, Any]:
    primary = family_cluster_contrast(
        transitions,
        exposures,
        mode="thinking_greedy",
        arm="S",
    )
    controls: dict[str, Any] = {}
    offset = 1
    for mode in MODES:
        for arm in ARMS:
            if mode == "thinking_greedy" and arm == "S":
                continue
            report = family_cluster_contrast(
                transitions,
                exposures,
                mode=mode,
                arm=arm,
                seed=BOOTSTRAP_SEED + offset,
            )
            controls[f"{mode}/{arm}"] = report
            offset += 1
    matrices = transition_matrices(transitions)
    primary_prefill = primary.get("features", {}).get(
        "prefill.layers_46_53.normalized_correction", {}
    )
    primary_decode = primary.get("features", {}).get(
        "decode.layers_46_53.normalized_correction", {}
    )
    decode_ci = primary_decode.get("mean_difference_ci95", [None, None])
    if (
        primary.get("status") == "estimated"
        and decode_ci[0] is not None
        and float(decode_ci[0]) > 0
    ):
        route = "resistance_interference_candidate_at_aggregate_decode_level"
    else:
        route = "insufficient_or_no_positive_exposure_association"
    return {
        "status": "passed",
        "analysis_kind": "zero-GPU aggregate-phase observational mechanism diagnostic",
        "primary_comparison": primary,
        "primary_key_effects": {
            "prefill_all_layers": primary_prefill,
            "decode_all_layers": primary_decode,
        },
        "control_comparisons": controls,
        "transition_matrices": matrices,
        "exposure_by_transition": transition_exposure_descriptives(transitions, exposures),
        "saturation": saturation_summary(exposures),
        "leave_one_family_out_primary": leave_one_family_out_primary(transitions, exposures),
        "strict_lenient_agreement": strict_lenient_agreement(transitions),
        "route": route,
        "position_timing_limit": (
            "No token-position telemetry; decode aggregates thinking and final-answer positions."
        ),
    }


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


def transition_matrix_plot(primary: dict[str, Any], path: Path) -> None:
    matrices = primary["transition_matrices"]["thinking_greedy"]
    outcomes = ("initial", "target", "other_wrong", "invalid")
    width, height = 1100, 720
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="32" text-anchor="middle" font-family="sans-serif" '
        'font-size="21" font-weight="bold">Thinking-greedy transition matrices</text>',
    ]
    locations = {"N": (70, 70), "V": (570, 70), "F": (70, 380), "S": (570, 380)}
    for arm in ARMS:
        left, top = locations[arm]
        matrix = matrices[arm]["matrix_uncapped_to_p25"]
        maximum = max(value for row in matrix.values() for value in row.values()) or 1
        svg.append(
            f'<text x="{left + 190}" y="{top}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="17">Arm {arm}</text>'
        )
        for row_index, before in enumerate(outcomes):
            svg.append(
                f'<text x="{left}" y="{top + 55 + row_index * 52}" text-anchor="end" '
                f'font-family="sans-serif" font-size="10">{before}</text>'
            )
            for column_index, after in enumerate(outcomes):
                value = int(matrix[before][after])
                opacity = 0.08 + 0.82 * value / maximum
                x = left + 12 + column_index * 88
                y = top + 25 + row_index * 52
                svg.extend(
                    [
                        f'<rect x="{x}" y="{y}" width="78" height="42" '
                        f'fill="#1665d8" fill-opacity="{opacity:.3f}" stroke="#bbb"/>',
                        f'<text x="{x + 39}" y="{y + 26}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="13">{value}</text>',
                    ]
                )
                if row_index == 0:
                    svg.append(
                        f'<text x="{x + 39}" y="{top + 242}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="9">{after}</text>'
                    )
    svg.append("</svg>")
    _svg_to_png("\n".join(svg), path)


def exposure_transition_plot(primary: dict[str, Any], path: Path) -> None:
    report = primary["primary_comparison"]
    features = (
        "prefill.layers_46_49.normalized_correction",
        "prefill.layers_50_53.normalized_correction",
        "decode.layers_46_49.normalized_correction",
        "decode.layers_50_53.normalized_correction",
    )
    labels = ("pre 46-49", "pre 50-53", "dec 46-49", "dec 50-53")
    left_values = [float(report["features"][name]["left"]["mean"]) for name in features]
    right_values = [float(report["features"][name]["right"]["mean"]) for name in features]
    maximum = max(*left_values, *right_values) * 1.15
    width, height = 900, 600
    left, top, plot_width, plot_height = 80, 70, 760, 430

    def y_pos(value: float) -> float:
        return top + (maximum - value) / maximum * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="32" text-anchor="middle" font-family="sans-serif" '
        'font-size="21" font-weight="bold">Thinking-greedy S exposure by transition</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="black"/>',
    ]
    for index, label in enumerate(labels):
        center = left + 95 + index * 185
        for offset, value, color in (
            (-28, left_values[index], "#d84b16"),
            (28, right_values[index], "#1665d8"),
        ):
            y = y_pos(value)
            svg.append(
                f'<rect x="{center + offset - 22}" y="{y}" width="44" '
                f'height="{top + plot_height - y}" fill="{color}"/>'
            )
        svg.append(
            f'<text x="{center}" y="{top + plot_height + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{label}</text>'
        )
    svg.extend(
        [
            '<text x="120" y="550" font-family="sans-serif" font-size="12" '
            'fill="#d84b16">initial→target</text>',
            '<text x="320" y="550" font-family="sans-serif" font-size="12" '
            'fill="#1665d8">stable initial</text>',
            "</svg>",
        ]
    )
    _svg_to_png("\n".join(svg), path)


def phase_layer_plot(
    transitions: list[dict[str, Any]], exposures: list[dict[str, Any]], path: Path
) -> None:
    exposure_map = {(row["mode"], row["id"]): row for row in exposures}
    selected = [
        row for row in transitions if row["mode"] == "thinking_greedy" and row["arm"] == "S"
    ]
    groups = {
        PRIMARY_LEFT: [row for row in selected if row["strict"]["transition"] == PRIMARY_LEFT],
        PRIMARY_RIGHT: [row for row in selected if row["strict"]["transition"] == PRIMARY_RIGHT],
    }
    series: dict[tuple[str, str], list[float]] = {}
    for transition, rows in groups.items():
        for phase in PHASES:
            series[(transition, phase)] = [
                statistics.fmean(
                    float(
                        exposure_map[(row["mode"], row["id"])]["phases"][phase]["by_layer"][
                            str(layer)
                        ]["mean_correction"]
                    )
                    for row in rows
                )
                for layer in LAYERS
            ]
    width, height = 1000, 680
    panels = (("prefill", 60), ("decode", 540))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="32" text-anchor="middle" font-family="sans-serif" '
        'font-size="21" font-weight="bold">Thinking-greedy S phase/layer exposure</text>',
    ]
    for phase, left in panels:
        values = [value for key, row in series.items() if key[1] == phase for value in row]
        low, high = min(values), max(values)
        padding = max((high - low) * 0.15, 1e-8)
        low, high = low - padding, high + padding
        top, plot_width, plot_height = 85, 400, 470

        def x_pos(
            index: int,
            origin: float = left,
            width: float = plot_width,
        ) -> float:
            return origin + 45 + index / 7 * (width - 70)

        def y_pos(
            value: float,
            minimum: float = low,
            maximum: float = high,
            origin: float = top,
            height: float = plot_height,
        ) -> float:
            return origin + (maximum - value) / (maximum - minimum) * height

        svg.extend(
            [
                f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
                'fill="#fafafa" stroke="#bbb"/>',
                f'<text x="{left + plot_width / 2}" y="{top + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="16">{phase}</text>',
            ]
        )
        for transition, color in ((PRIMARY_LEFT, "#d84b16"), (PRIMARY_RIGHT, "#1665d8")):
            row = series[(transition, phase)]
            points = " ".join(
                f"{x_pos(index):.2f},{y_pos(value):.2f}" for index, value in enumerate(row)
            )
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>'
            )
            for index, value in enumerate(row):
                svg.append(
                    f'<circle cx="{x_pos(index):.2f}" cy="{y_pos(value):.2f}" r="3" '
                    f'fill="{color}"/>'
                )
        for index, layer in enumerate(LAYERS):
            svg.append(
                f'<text x="{x_pos(index)}" y="{top + plot_height + 18}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="10">{layer}</text>'
            )
    svg.extend(
        [
            '<text x="160" y="625" font-family="sans-serif" font-size="12" '
            'fill="#d84b16">initial→target</text>',
            '<text x="360" y="625" font-family="sans-serif" font-size="12" '
            'fill="#1665d8">stable initial</text>',
            "</svg>",
        ]
    )
    _svg_to_png("\n".join(svg), path)


def unavailable_margin_plot(path: Path) -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600">
<rect width="100%" height="100%" fill="white"/>
<rect x="80" y="80" width="840" height="400" fill="#fafafa" stroke="#bbb"/>
<text x="500" y="45" text-anchor="middle" font-family="sans-serif" font-size="22"
font-weight="bold">Initial margin vs flip</text>
<text x="500" y="245" text-anchor="middle" font-family="sans-serif" font-size="28"
fill="#a33">NOT AVAILABLE</text>
<text x="500" y="300" text-anchor="middle" font-family="sans-serif" font-size="16">
v6 generation logs contain categorical answers but no answer-token logits or initial margin.</text>
<text x="500" y="340" text-anchor="middle" font-family="sans-serif" font-size="16">
No GPU rerun was performed to reconstruct this field.</text>
</svg>"""
    _svg_to_png(svg, path)


def summary_markdown(
    availability: dict[str, Any], primary: dict[str, Any]
) -> str:
    comparison = primary["primary_comparison"]
    prefill = comparison["features"]["prefill.layers_46_53.normalized_correction"]
    decode = comparison["features"]["decode.layers_46_53.normalized_correction"]
    matrices = primary["transition_matrices"]["thinking_greedy"]
    s_counts = matrices["S"]["transition_labels"]
    f_counts = matrices["F"]["transition_labels"]
    v_counts = matrices["V"]["transition_labels"]
    n_counts = matrices["N"]["transition_labels"]
    lines = [
        "# E0 v6 反事实暴露分析摘要",
        "",
        "- 分析类型：零 GPU、aggregate-phase observational mechanism diagnostic",
        f"- 遥测闸门：`{availability['status']}`",
        f"- 路由：`{primary['route']}`",
        "- strict/lenient 评分均保留；strict 为主分析",
        f"- family bootstrap：{BOOTSTRAP_REPLICATES} 次，seed `{BOOTSTRAP_SEED}`",
        "",
        "## 主要问题：thinking-greedy S",
        "",
        f"严格评分下 initial→target 为 {s_counts.get('initial_to_target', 0)} 条，"
        f"stable_initial 为 {s_counts.get('stable_initial', 0)} 条。",
        "",
        "| phase / layers | flip 均值 | stable 均值 | 差值 | 95% family-cluster CI |",
        "|---|---:|---:|---:|---:|",
        f"| prefill / 46–53 | {prefill['left']['mean']:.6f} | "
        f"{prefill['right']['mean']:.6f} | {prefill['mean_difference']:+.6f} | "
        f"[{prefill['mean_difference_ci95'][0]:+.6f}, "
        f"{prefill['mean_difference_ci95'][1]:+.6f}] |",
        f"| decode / 46–53 | {decode['left']['mean']:.6f} | "
        f"{decode['right']['mean']:.6f} | {decode['mean_difference']:+.6f} | "
        f"[{decode['mean_difference_ci95'][0]:+.6f}, "
        f"{decode['mean_difference_ci95'][1]:+.6f}] |",
        "",
        "decode 聚合了全部 thinking 和最终答案 token；不能把这一行解释成精确的"
        "pre-answer 抵抗时刻。",
        "",
        "## 四条机制路线",
        "",
        "### 1. 更像变顽固吗？",
        "",
        f"thinking-greedy V 的迁移为 `{json.dumps(v_counts, ensure_ascii=False)}`。"
        "重点查看 target→initial 是否出现；target 在 V 中代表正确更新。",
        "",
        "### 2. 更像一般退化吗？",
        "",
        f"thinking-greedy N：`{json.dumps(n_counts, ensure_ascii=False)}`；F："
        f"`{json.dumps(f_counts, ensure_ascii=False)}`。other_wrong/invalid 与暴露的完整"
        "描述见 primary_results.json。",
        "",
        "### 3. 更像选择性增加 S/F 服从吗？",
        "",
        "只有当 S flip 的 uncapped exposure 明显高于 stable initial、F 较弱同向，且"
        "V/N/other_wrong 不同步恶化时，才支持这一候选。主比较的方向和 CI 如上；不能只"
        "依据 5 条行为翻转讲机制。",
        "",
        "### 4. 还是证据不足？",
        "",
        "只有 5 条 S initial→target，且日志缺少 token-position、pre-answer、answer logits"
        "与 finish reason。若 exposure 高度饱和或 CI 宽，应归为证据不足，不能把“不显著”"
        "写成状态门控已被否定。",
        "",
        "## 可用性与禁止外推",
        "",
        "- uncapped p25 counterfactual exposure 在全部 576 条记录、prefill/decode、46–53"
        "层均完整；correction sum 由 mean_correction × count 无损恢复。",
        "- token-position 与 thinking/pre-answer/answer 分相不可用；initial-margin 图是明确"
        "占位图，没有为补字段重跑 GPU。",
        "- 本轮不证明正确更新、一般安全收益或精确抵抗时刻的因果机制。",
        "- 未运行 p1、新 seed、E1/E2、J-Lens、MUSE、probe 或新生成。",
        "- E0 到此强制停止，等待审核。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().isoformat()
    sources = load_sources()
    manifest = input_manifest(sources)
    atomic_write_json(args.output_dir / "input_manifest.json", manifest)
    if manifest["status"] != "passed":
        raise RuntimeError("E0 frozen input manifest failed")
    availability = telemetry_availability(sources)
    atomic_write_json(args.output_dir / "telemetry_availability_audit.json", availability)
    if not availability["uncapped_p25_counterfactual_exposure_complete"]:
        raise RuntimeError(
            "E0 uncapped counterfactual exposure is incomplete; stop before analysis"
        )
    transitions = build_transitions(sources)
    exposures = build_exposures(sources, transitions)
    atomic_write_jsonl(args.output_dir / "transition_table.jsonl", transitions)
    atomic_write_jsonl(args.output_dir / "counterfactual_exposure_normalized.jsonl", exposures)
    primary = build_primary_results(transitions, exposures)
    atomic_write_json(args.output_dir / "primary_results.json", primary)
    transition_matrix_plot(primary, args.output_dir / "transition_matrix.png")
    exposure_transition_plot(primary, args.output_dir / "exposure_by_transition.png")
    phase_layer_plot(transitions, exposures, args.output_dir / "phase_layer_exposure.png")
    unavailable_margin_plot(args.output_dir / "initial_margin_vs_flip.png")
    (args.output_dir / "E0_SUMMARY_ZH.md").write_text(
        summary_markdown(availability, primary),
        encoding="utf-8",
        newline="\n",
    )
    output_names = (
        "input_manifest.json",
        "telemetry_availability_audit.json",
        "transition_table.jsonl",
        "counterfactual_exposure_normalized.jsonl",
        "primary_results.json",
        "transition_matrix.png",
        "exposure_by_transition.png",
        "phase_layer_exposure.png",
        "initial_margin_vs_flip.png",
        "E0_SUMMARY_ZH.md",
    )
    run_manifest = {
        "status": "passed",
        "started_at": started,
        "completed_at": datetime.now().astimezone().isoformat(),
        "analysis_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "result_commit": RESULT_COMMIT,
        "score_commit": SCORE_COMMIT,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gpu_or_model_execution": False,
        "new_model_outputs": 0,
        "p1_or_new_seed_executed": False,
        "mechanism_tools_executed": [],
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
    }
    atomic_write_json(args.output_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "telemetry_route": availability["route"],
                "primary_route": primary["route"],
                "S_initial_to_target": primary["primary_comparison"]["left_n"],
                "S_stable_initial": primary["primary_comparison"]["right_n"],
                "prefill_difference": primary["primary_key_effects"]["prefill_all_layers"][
                    "mean_difference"
                ],
                "decode_difference": primary["primary_key_effects"]["decode_all_layers"][
                    "mean_difference"
                ],
                "gpu_or_model_execution": False,
                "mandatory_stop": True,
            }
        )
    )


if __name__ == "__main__":
    main()
