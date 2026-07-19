#!/usr/bin/env python3
"""Analyze the frozen E1 token-resolved, within-prompt geometry assay."""

from __future__ import annotations

import argparse
import json
import os
import platform
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cedar_m2.checkpoint import atomic_write_json, file_sha256, object_sha256

try:
    from scripts import run_e1_token_resolved_geometry as runner
except ModuleNotFoundError:
    import run_e1_token_resolved_geometry as runner  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "m2" / "e1_token_resolved_geometry"
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_REPLICATES = 10_000

LAYER_GROUPS = {
    "layers_46_49": tuple(range(46, 50)),
    "layers_50_53": tuple(range(50, 54)),
    "layers_46_53": tuple(range(46, 54)),
}
WINDOWS = (
    "prefill_original",
    "prefill_challenge",
    "thinking_early",
    "thinking_middle",
    "thinking_late",
    "preanswer_32",
    "answer_prefix",
    "answer_choice",
    "post_answer",
    "all_decode",
)
MEASURES = ("correction", "axis", "distance")
PRIMARY_FEATURE = "preanswer_32.layers_50_53.correction"
PREFILL_FEATURE = "prefill_challenge.layers_50_53.correction"
EARLY_LAYER_FEATURE = "preanswer_32.layers_46_49.correction"
FULL_DECODE_FEATURE = "all_decode.layers_50_53.correction"
PRIMARY_AXIS_FEATURE = "preanswer_32.layers_50_53.axis"
PHASE_INTERACTION_FEATURE = "interaction.preanswer_minus_prefill.layers_50_53.correction"
LAYER_INTERACTION_FEATURE = "interaction.layers_50_53_minus_46_49.preanswer.correction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "final"), default="gate")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_spec(
    stage: str,
    formal: Sequence[dict[str, Any]],
    axis_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[int, ...], tuple[str, ...], bool]:
    rows = runner._stage_rows(stage, formal, axis_rows)
    if stage == "smoke":
        return rows, (0,), ("no_hook", "monitor_only"), False
    if stage == "bridge":
        return rows, (0,), ("monitor_only",), False
    if stage == "sample-s":
        return rows, runner.SAMPLE_SEEDS, ("monitor_only",), True
    if stage == "control-f":
        return rows, runner.F_CONTROL_SEEDS, ("monitor_only",), True
    raise ValueError(stage)


def _stage_fingerprint(
    run_dir: Path,
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    paths: Sequence[str],
    do_sample: bool,
) -> str:
    return object_sha256(
        {
            "input_manifest": file_sha256(run_dir / "INPUT_MANIFEST.json"),
            "stage": stage,
            "rows": [row["id"] for row in rows],
            "seeds": list(seeds),
            "paths": list(paths),
            "do_sample": do_sample,
        }
    )


def load_stage(
    run_dir: Path,
    stage: str,
    formal: Sequence[dict[str, Any]],
    axis_rows: Sequence[dict[str, Any]],
    *,
    required: bool,
) -> list[dict[str, Any]]:
    rows, seeds, paths, do_sample = _stage_spec(stage, formal, axis_rows)
    fingerprint = _stage_fingerprint(run_dir, stage, rows, seeds, paths, do_sample)
    store = runner.TrajectoryStore(run_dir / "records" / stage, fingerprint=fingerprint)
    loaded: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        for seed in seeds:
            for path in paths:
                key = runner._trajectory_key(stage, row, seed, path)
                envelope = store.read(key)
                if envelope is None:
                    missing.append(key)
                    continue
                _, arrays_path = store.paths(key)
                with np.load(arrays_path, allow_pickle=False) as handle:
                    arrays = {name: handle[name] for name in handle.files}
                loaded.append(
                    {
                        "key": key,
                        "result": envelope["result"],
                        "arrays": arrays,
                        "arrays_path": arrays_path,
                        "arrays_sha256": envelope["arrays_sha256"],
                    }
                )
    if missing and required:
        raise RuntimeError(f"{stage}: missing {len(missing)} trajectories; first={missing[0]}")
    return loaded


def _layer_columns(layer_ids: np.ndarray, layers: Sequence[int]) -> list[int]:
    mapping = {int(layer): index for index, layer in enumerate(layer_ids.tolist())}
    if not set(layers) <= set(mapping):
        raise ValueError(f"Layer group {layers} unavailable in {sorted(mapping)}")
    return [mapping[layer] for layer in layers]


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def trajectory_features(record: Mapping[str, Any]) -> dict[str, Any]:
    arrays = record["arrays"]
    layer_ids = arrays["layer_ids"]
    features: dict[str, Any] = {}
    prompt_codes = arrays["prefill_phase_codes"]
    response_codes = arrays["response_phase_codes"]
    observed_indices = arrays.get(
        "decode_observed_response_indices",
        np.empty((0,), dtype=np.int32),
    )
    preanswer = arrays["response_preanswer32_mask"]
    decode_response_codes = (
        response_codes[observed_indices]
        if observed_indices.size
        else response_codes[:0]
    )
    decode_preanswer = preanswer[observed_indices] if observed_indices.size else preanswer[:0]

    masks: dict[str, tuple[str, np.ndarray]] = {
        "prefill_original": (
            "prefill",
            prompt_codes == runner.PHASE_CODES["prefill_original"],
        ),
        "prefill_challenge": (
            "prefill",
            prompt_codes == runner.PHASE_CODES["prefill_challenge"],
        ),
        "thinking_early": (
            "decode",
            decode_response_codes == runner.PHASE_CODES["thinking_early"],
        ),
        "thinking_middle": (
            "decode",
            decode_response_codes == runner.PHASE_CODES["thinking_middle"],
        ),
        "thinking_late": (
            "decode",
            decode_response_codes == runner.PHASE_CODES["thinking_late"],
        ),
        "preanswer_32": ("decode", decode_preanswer.astype(bool)),
        "answer_prefix": (
            "decode",
            decode_response_codes == runner.PHASE_CODES["answer_prefix"],
        ),
        "answer_choice": (
            "decode",
            decode_response_codes == runner.PHASE_CODES["answer_choice"],
        ),
        "post_answer": (
            "decode",
            decode_response_codes == runner.PHASE_CODES["post_answer"],
        ),
        "all_decode": ("decode", np.ones(observed_indices.size, dtype=bool)),
    }
    for window, (source, mask) in masks.items():
        for group, layers in LAYER_GROUPS.items():
            columns = _layer_columns(layer_ids, layers)
            for measure in MEASURES:
                matrix = arrays.get(f"{source}_{measure}")
                key = f"{window}.{group}.{measure}"
                count_key = f"{window}.{group}.positions"
                if matrix is None or mask.shape[0] != matrix.shape[0] or not np.any(mask):
                    features[key] = None
                    features.setdefault(count_key, 0)
                    continue
                selected = matrix[mask][:, columns]
                features[key] = _finite_mean(selected)
                features[count_key] = int(selected.size)

    primary = features.get(PRIMARY_FEATURE)
    prefill = features.get(PREFILL_FEATURE)
    early = features.get(EARLY_LAYER_FEATURE)
    features[PHASE_INTERACTION_FEATURE] = (
        primary - prefill if primary is not None and prefill is not None else None
    )
    features[LAYER_INTERACTION_FEATURE] = (
        primary - early if primary is not None and early is not None else None
    )
    for layer in runner.LAYERS:
        column = _layer_columns(layer_ids, (layer,))[0]
        for measure in ("correction", "axis", "distance"):
            matrix = arrays.get(f"decode_{measure}")
            key = f"preanswer_32.layer_{layer}.{measure}"
            if matrix is None or not np.any(decode_preanswer):
                features[key] = None
            else:
                features[key] = _finite_mean(matrix[decode_preanswer, column])
    return features


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def enrich_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        result = record["result"]
        features = trajectory_features(record) if result["monitor_enabled"] else {}
        enriched.append({**record, "features": features})
    return enriched


def build_behavior_counts(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    by_stage_arm: dict[str, Counter[str]] = defaultdict(Counter)
    finish: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        result = record["result"]
        if not result["monitor_enabled"]:
            continue
        stage = result["stage"]
        behavior = result.get("behavior") or "not_applicable"
        by_stage[stage][behavior] += 1
        by_stage_arm[f"{stage}/{result.get('arm')}"][behavior] += 1
        finish[stage][result["finish_reason"]] += 1
    return {
        "by_stage": {key: dict(value) for key, value in sorted(by_stage.items())},
        "by_stage_arm": {
            key: dict(value) for key, value in sorted(by_stage_arm.items())
        },
        "finish_reasons": {key: dict(value) for key, value in sorted(finish.items())},
    }


def mixed_prompt_gate(sample_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_prompt: dict[str, Counter[str]] = defaultdict(Counter)
    for record in sample_records:
        result = record["result"]
        by_prompt[result["id"]][result["behavior"]] += 1
    prompt_rows: list[dict[str, Any]] = []
    mixed = 0
    mixed_minority_three = 0
    total_resist = 0
    total_yield = 0
    for prompt_id, counts in sorted(by_prompt.items()):
        resist = counts["resist"]
        yielded = counts["yield"]
        total_resist += resist
        total_yield += yielded
        is_mixed = resist > 0 and yielded > 0
        minority = min(resist, yielded) if is_mixed else 0
        mixed += int(is_mixed)
        mixed_minority_three += int(is_mixed and minority >= 3)
        prompt_rows.append(
            {
                "id": prompt_id,
                "counts": dict(counts),
                "mixed": is_mixed,
                "minority_count": minority,
            }
        )
    checks = {
        "at_least_10_mixed_prompts": mixed >= 10,
        "at_least_40_resist": total_resist >= 40,
        "at_least_40_yield": total_yield >= 40,
        "at_least_10_mixed_prompts_minority_ge_3": mixed_minority_three >= 10,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed_insufficient_variation",
        "checks": checks,
        "sample_trajectories": len(sample_records),
        "prompts": len(by_prompt),
        "mixed_prompts": mixed,
        "mixed_prompts_minority_ge_3": mixed_minority_three,
        "total_resist": total_resist,
        "total_yield": total_yield,
        "prompt_counts": prompt_rows,
        "temperature_or_seed_adaptation_allowed": False,
    }


def _mixed_groups(
    records: Sequence[dict[str, Any]], feature: str
) -> list[dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        result = record["result"]
        value = record["features"].get(feature)
        if result["behavior"] in {"resist", "yield"} and value is not None:
            by_prompt[result["id"]].append(record)
    groups: list[dict[str, Any]] = []
    for prompt_id, prompt_records in sorted(by_prompt.items()):
        resist = [
            float(record["features"][feature])
            for record in prompt_records
            if record["result"]["behavior"] == "resist"
        ]
        yielded = [
            float(record["features"][feature])
            for record in prompt_records
            if record["result"]["behavior"] == "yield"
        ]
        if not resist or not yielded:
            continue
        groups.append(
            {
                "id": prompt_id,
                "family": prompt_records[0]["result"]["template_family"],
                "resist": np.asarray(resist, dtype=float),
                "yield": np.asarray(yielded, dtype=float),
                "difference": float(np.mean(resist) - np.mean(yielded)),
            }
        )
    return groups


def hierarchical_bootstrap(
    records: Sequence[dict[str, Any]],
    feature: str,
    all_families: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    groups = _mixed_groups(records, feature)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_family[group["family"]].append(group)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    families = list(all_families)
    for _ in range(replicates):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        prompt_differences: list[float] = []
        for family in sampled_families:
            for group in by_family.get(str(family), []):
                resist = rng.choice(group["resist"], size=group["resist"].size, replace=True)
                yielded = rng.choice(group["yield"], size=group["yield"].size, replace=True)
                prompt_differences.append(float(resist.mean() - yielded.mean()))
        if prompt_differences:
            estimates.append(float(np.mean(prompt_differences)))
    prompt_diffs = [float(group["difference"]) for group in groups]
    array = np.asarray(estimates, dtype=float)
    return {
        "feature": feature,
        "estimand": "prompt-equal mean of resist minus yield",
        "mixed_prompts": len(groups),
        "families_with_mixed_prompts": len(by_family),
        "family_clusters_total": len(families),
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": int(array.size),
        "bootstrap_seed": seed,
        "point_estimate": float(np.mean(prompt_diffs)) if prompt_diffs else None,
        "ci95": (
            [float(value) for value in np.quantile(array, [0.025, 0.975])]
            if array.size
            else None
        ),
        "prompt_difference_summary": _summarize(prompt_diffs),
        "prompt_differences": [
            {
                "id": group["id"],
                "family": group["family"],
                "resist_n": int(group["resist"].size),
                "yield_n": int(group["yield"].size),
                "difference": group["difference"],
            }
            for group in groups
        ],
    }


def conditional_logit(
    records: Sequence[dict[str, Any]],
    feature: str,
    *,
    include_length: bool,
) -> dict[str, Any]:
    eligible = [
        record
        for record in records
        if record["result"]["behavior"] in {"resist", "yield"}
        and record["features"].get(feature) is not None
    ]
    prompt_outcomes: dict[str, set[str]] = defaultdict(set)
    for record in eligible:
        prompt_outcomes[record["result"]["id"]].add(record["result"]["behavior"])
    eligible = [
        record
        for record in eligible
        if prompt_outcomes[record["result"]["id"]] == {"resist", "yield"}
    ]
    if not eligible:
        return {"status": "not_estimable", "reason": "no mixed prompts"}
    feature_values = np.asarray([record["features"][feature] for record in eligible], dtype=float)
    feature_mean = float(feature_values.mean())
    feature_sd = float(feature_values.std(ddof=1))
    if feature_sd == 0:
        return {"status": "not_estimable", "reason": "zero feature variance"}
    columns = [(feature_values - feature_mean) / feature_sd]
    names = [f"z:{feature}"]
    if include_length:
        lengths = np.asarray(
            [record["result"]["boundary"]["thinking_tokens"] for record in eligible],
            dtype=float,
        )
        length_sd = float(lengths.std(ddof=1))
        columns.append((lengths - lengths.mean()) / length_sd if length_sd else lengths * 0)
        names.append("z:thinking_tokens")
    seeds = sorted({int(record["result"]["seed"]) for record in eligible})
    for seed in seeds[1:]:
        columns.append(
            np.asarray([int(record["result"]["seed"] == seed) for record in eligible], dtype=float)
        )
        names.append(f"seed_{seed}")
    exog = np.column_stack(columns)
    outcome = np.asarray(
        [int(record["result"]["behavior"] == "yield") for record in eligible], dtype=int
    )
    prompt_ids = sorted({record["result"]["id"] for record in eligible})
    prompt_map = {value: index for index, value in enumerate(prompt_ids)}
    groups = np.asarray([prompt_map[record["result"]["id"]] for record in eligible])
    try:
        from statsmodels.discrete.conditional_models import ConditionalLogit

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ConditionalLogit(outcome, exog, groups=groups).fit(
                method="bfgs", maxiter=1000, disp=False
            )
        params = np.asarray(fitted.params, dtype=float)
        standard_errors = np.asarray(fitted.bse, dtype=float)
        return {
            "status": "estimated",
            "records": len(eligible),
            "mixed_prompts": len(prompt_ids),
            "outcome": "yield=1,resist=0",
            "prompt_effect": "conditioned out",
            "seed_fixed_effects": seeds,
            "feature_standardization": {"mean": feature_mean, "sd": feature_sd},
            "coefficients": {
                name: {
                    "estimate": float(params[index]),
                    "standard_error": float(standard_errors[index]),
                    "ci95": [
                        float(params[index] - 1.96 * standard_errors[index]),
                        float(params[index] + 1.96 * standard_errors[index]),
                    ],
                }
                for index, name in enumerate(names)
            },
            "converged": bool(fitted.mle_retvals.get("converged", False)),
        }
    except Exception as error:  # numerical separation is a reportable assay result
        return {
            "status": "not_estimable",
            "records": len(eligible),
            "mixed_prompts": len(prompt_ids),
            "error": f"{type(error).__name__}: {error}",
        }


def within_prompt_margin_association(
    records: Sequence[dict[str, Any]], feature: str, margin_source: str
) -> dict[str, Any]:
    pairs: list[tuple[str, float, float]] = []
    for record in records:
        diagnostics = record["result"].get(margin_source)
        value = record["features"].get(feature)
        if diagnostics is None or diagnostics.get("target_minus_initial") is None or value is None:
            continue
        pairs.append(
            (
                record["result"]["id"],
                float(value),
                float(diagnostics["target_minus_initial"]),
            )
        )
    by_prompt: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for prompt_id, exposure, margin in pairs:
        by_prompt[prompt_id].append((exposure, margin))
    centered_x: list[float] = []
    centered_y: list[float] = []
    for values in by_prompt.values():
        x = np.asarray([value[0] for value in values], dtype=float)
        y = np.asarray([value[1] for value in values], dtype=float)
        centered_x.extend((x - x.mean()).tolist())
        centered_y.extend((y - y.mean()).tolist())
    x_array = np.asarray(centered_x, dtype=float)
    y_array = np.asarray(centered_y, dtype=float)
    if x_array.size < 3 or np.allclose(x_array, 0) or np.allclose(y_array, 0):
        return {"status": "not_estimable", "records": int(x_array.size)}
    slope = float(np.dot(x_array, y_array) / np.dot(x_array, x_array))
    correlation = float(np.corrcoef(x_array, y_array)[0, 1])
    return {
        "status": "estimated",
        "records": int(x_array.size),
        "prompts": len(by_prompt),
        "feature": feature,
        "margin_source": margin_source,
        "method": "within-prompt demeaned OLS/correlation",
        "slope": slope,
        "correlation": correlation,
    }


def length_matched_analysis(
    records: Sequence[dict[str, Any]], feature: str, all_families: Sequence[str]
) -> dict[str, Any]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if (
            record["result"]["behavior"] in {"resist", "yield"}
            and record["features"].get(feature) is not None
        ):
            by_prompt[record["result"]["id"]].append(record)
    matched: list[dict[str, Any]] = []
    for prompt_id, prompt_records in sorted(by_prompt.items()):
        resist = [r for r in prompt_records if r["result"]["behavior"] == "resist"]
        yielded = [r for r in prompt_records if r["result"]["behavior"] == "yield"]
        candidates = sorted(
            (
                abs(
                    int(left["result"]["boundary"]["thinking_tokens"])
                    - int(right["result"]["boundary"]["thinking_tokens"])
                ),
                int(left["result"]["seed"]),
                int(right["result"]["seed"]),
                left,
                right,
            )
            for left in resist
            for right in yielded
        )
        used_left: set[int] = set()
        used_right: set[int] = set()
        for length_difference, left_seed, right_seed, left, right in candidates:
            if left_seed in used_left or right_seed in used_right:
                continue
            used_left.add(left_seed)
            used_right.add(right_seed)
            matched.append(
                {
                    "id": prompt_id,
                    "family": left["result"]["template_family"],
                    "resist_seed": left_seed,
                    "yield_seed": right_seed,
                    "thinking_token_difference": length_difference,
                    "feature_difference": float(
                        left["features"][feature] - right["features"][feature]
                    ),
                }
            )
    prompt_means: list[dict[str, Any]] = []
    for prompt_id in sorted({row["id"] for row in matched}):
        rows = [row for row in matched if row["id"] == prompt_id]
        prompt_means.append(
            {
                "id": prompt_id,
                "family": rows[0]["family"],
                "difference": float(np.mean([row["feature_difference"] for row in rows])),
                "pairs": len(rows),
            }
        )
    family_values: dict[str, list[float]] = defaultdict(list)
    for row in prompt_means:
        family_values[row["family"]].append(row["difference"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates: list[float] = []
    families = list(all_families)
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(families, size=len(families), replace=True)
        values = [value for family in selected for value in family_values.get(str(family), [])]
        if values:
            estimates.append(float(np.mean(values)))
    estimate_array = np.asarray(estimates, dtype=float)
    return {
        "feature": feature,
        "matching": "within prompt greedy nearest absolute thinking-token distance; no reuse",
        "matched_pairs": len(matched),
        "matched_prompts": len(prompt_means),
        "mean_abs_thinking_token_difference": (
            float(np.mean([row["thinking_token_difference"] for row in matched]))
            if matched
            else None
        ),
        "point_estimate": (
            float(np.mean([row["difference"] for row in prompt_means]))
            if prompt_means
            else None
        ),
        "ci95": (
            [float(value) for value in np.quantile(estimate_array, [0.025, 0.975])]
            if estimate_array.size
            else None
        ),
        "prompt_differences": prompt_means,
    }


def _append_array(target: dict[str, list[np.ndarray]], name: str, value: np.ndarray) -> None:
    target.setdefault(name, []).append(value)


def consolidate_trajectories(
    run_dir: Path, records: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    index_rows: list[dict[str, Any]] = []
    pieces: dict[str, list[np.ndarray]] = {}
    prefill_offset = 0
    decode_offset = 0
    response_offset = 0
    abc_offset = 0
    for trajectory_index, record in enumerate(records):
        result = record["result"]
        arrays = record["arrays"]
        prompt_count = arrays["prompt_token_ids"].shape[0]
        response_count = arrays["response_token_ids"].shape[0]
        prefill_count = arrays.get("prefill_correction", np.empty((0,))).shape[0]
        decode_count = arrays.get("decode_correction", np.empty((0,))).shape[0]
        abc_count = arrays["abc_logits_by_step"].shape[0]
        for measure in (*MEASURES, "cap", "trigger"):
            if f"prefill_{measure}" in arrays:
                _append_array(pieces, f"prefill_{measure}", arrays[f"prefill_{measure}"])
            if f"decode_{measure}" in arrays:
                _append_array(pieces, f"decode_{measure}", arrays[f"decode_{measure}"])
        _append_array(pieces, "prompt_token_ids", arrays["prompt_token_ids"])
        _append_array(pieces, "prefill_phase_codes", arrays["prefill_phase_codes"])
        _append_array(
            pieces,
            "prefill_trajectory_index",
            np.full(prompt_count, trajectory_index, dtype=np.int32),
        )
        _append_array(
            pieces,
            "prefill_token_position",
            np.arange(prompt_count, dtype=np.int32),
        )
        _append_array(pieces, "response_token_ids", arrays["response_token_ids"])
        _append_array(pieces, "response_phase_codes", arrays["response_phase_codes"])
        _append_array(
            pieces,
            "response_preanswer32_mask",
            arrays["response_preanswer32_mask"],
        )
        _append_array(
            pieces,
            "response_trajectory_index",
            np.full(response_count, trajectory_index, dtype=np.int32),
        )
        _append_array(
            pieces,
            "response_token_position",
            np.arange(response_count, dtype=np.int32),
        )
        _append_array(pieces, "abc_logits_by_step", arrays["abc_logits_by_step"])
        _append_array(
            pieces,
            "abc_trajectory_index",
            np.full(abc_count, trajectory_index, dtype=np.int32),
        )
        _append_array(
            pieces,
            "abc_step",
            np.arange(abc_count, dtype=np.int32),
        )
        if decode_count:
            observed = arrays["decode_observed_response_indices"]
            _append_array(
                pieces,
                "decode_trajectory_index",
                np.full(decode_count, trajectory_index, dtype=np.int32),
            )
            _append_array(pieces, "decode_response_position", observed)
        index_rows.append(
            {
                "trajectory_index": trajectory_index,
                "stage": result["stage"],
                "path": result["path"],
                "id": result["id"],
                "arm": result.get("arm"),
                "family": result.get("template_family"),
                "seed": result["seed"],
                "behavior": result.get("behavior"),
                "strict_answer": result.get("strict_answer"),
                "finish_reason": result["finish_reason"],
                "truncated": result["truncated"],
                "prompt_tokens": prompt_count,
                "response_tokens": response_count,
                "thinking_tokens": result["boundary"].get("thinking_tokens"),
                "prompt_token_texts": result["prompt_token_texts"],
                "response_token_texts": result["response_token_texts"],
                "prefill_array_range": [prefill_offset, prefill_offset + prefill_count],
                "decode_array_range": [decode_offset, decode_offset + decode_count],
                "response_array_range": [response_offset, response_offset + response_count],
                "abc_logit_array_range": [abc_offset, abc_offset + abc_count],
                "source_record_sha256": record["arrays_sha256"],
            }
        )
        prefill_offset += prefill_count
        decode_offset += decode_count
        response_offset += response_count
        abc_offset += abc_count
    consolidated: dict[str, np.ndarray] = {
        "layer_ids": np.asarray(runner.LAYERS, dtype=np.int16),
        "abc_token_ids": np.asarray(runner.ABC_TOKEN_IDS, dtype=np.int32),
        "thresholds": records[0]["arrays"]["thresholds"].astype(np.float32),
    }
    for name, values in pieces.items():
        consolidated[name] = np.concatenate(values, axis=0) if values else np.empty((0,))
    runner.atomic_write_npz(run_dir / "token_projection_arrays.npz", consolidated)
    _atomic_write_jsonl(run_dir / "trajectory_index.jsonl", index_rows)
    return index_rows, consolidated


def answer_margin_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        result = record["result"]
        rows.append(
            {
                "stage": result["stage"],
                "id": result["id"],
                "arm": result.get("arm"),
                "family": result.get("template_family"),
                "seed": result["seed"],
                "behavior": result.get("behavior"),
                "strict_answer": result.get("strict_answer"),
                "finish_reason": result["finish_reason"],
                "response_tokens": result["response_tokens"],
                "thinking_tokens": result["boundary"].get("thinking_tokens"),
                "actual_answer_logits": result.get("actual_answer_logits"),
                "scaffold_answer_logits": result.get("scaffold_answer_logits"),
                "primary_preanswer_correction": record["features"].get(PRIMARY_FEATURE),
                "primary_preanswer_axis_projection": record["features"].get(
                    PRIMARY_AXIS_FEATURE
                ),
                "prefill_challenge_correction": record["features"].get(PREFILL_FEATURE),
                "all_decode_correction": record["features"].get(FULL_DECODE_FEATURE),
            }
        )
    return rows


def _feature_report(
    records: Sequence[dict[str, Any]], feature: str, families: Sequence[str]
) -> dict[str, Any]:
    report = hierarchical_bootstrap(records, feature, families)
    groups = _mixed_groups(records, feature)
    report["resist_trajectory_summary"] = _summarize(
        [float(value) for group in groups for value in group["resist"]]
    )
    report["yield_trajectory_summary"] = _summarize(
        [float(value) for group in groups for value in group["yield"]]
    )
    return report


def analyze(
    run_dir: Path,
    sample_records: Sequence[dict[str, Any]],
    control_records: Sequence[dict[str, Any]],
    all_records: Sequence[dict[str, Any]],
    gate: Mapping[str, Any],
    *,
    final_requested: bool,
) -> dict[str, Any]:
    formal, _ = runner.validate_input_rows()
    families = sorted({row["template_family"] for row in formal})
    gate_passed = gate["status"] == "passed"
    primary_status = "estimated" if gate_passed else "continuous_only_insufficient_variation"

    primary_features = [
        PRIMARY_FEATURE,
        PRIMARY_AXIS_FEATURE,
        PREFILL_FEATURE,
        EARLY_LAYER_FEATURE,
        FULL_DECODE_FEATURE,
        PHASE_INTERACTION_FEATURE,
        LAYER_INTERACTION_FEATURE,
        *(f"preanswer_32.layer_{layer}.correction" for layer in runner.LAYERS),
        *(f"preanswer_32.layer_{layer}.axis" for layer in runner.LAYERS),
    ]
    feature_reports = {
        feature: _feature_report(sample_records, feature, families)
        for feature in primary_features
    }
    primary = {
        "status": primary_status,
        "estimand": "within-prompt resist minus yield",
        "primary_feature": feature_reports[PRIMARY_FEATURE],
        "assistant_axis_projection": feature_reports[PRIMARY_AXIS_FEATURE],
        "prefill_challenge": feature_reports[PREFILL_FEATURE],
        "all_decode": feature_reports[FULL_DECODE_FEATURE],
        "all_layers": {
            str(layer): {
                "correction": feature_reports[f"preanswer_32.layer_{layer}.correction"],
                "axis": feature_reports[f"preanswer_32.layer_{layer}.axis"],
            }
            for layer in runner.LAYERS
        },
        "conditional_logit": conditional_logit(
            sample_records, PRIMARY_FEATURE, include_length=False
        ),
        "conditional_logit_with_length": conditional_logit(
            sample_records, PRIMARY_FEATURE, include_length=True
        ),
        "actual_margin_association": within_prompt_margin_association(
            sample_records, PRIMARY_FEATURE, "actual_answer_logits"
        ),
        "scaffold_margin_association": within_prompt_margin_association(
            sample_records, PRIMARY_FEATURE, "scaffold_answer_logits"
        ),
    }
    phase_interaction = {
        "status": primary_status,
        "definition": "D_preanswer32 - D_prefill_challenge; layers 50-53",
        "result": feature_reports[PHASE_INTERACTION_FEATURE],
        "claim_allowed_only_if_ci_lower_gt_zero": True,
    }
    layer_interaction = {
        "status": primary_status,
        "definition": "D_layers50-53 - D_layers46-49; preanswer_32",
        "result": feature_reports[LAYER_INTERACTION_FEATURE],
        "claim_allowed_only_if_ci_lower_gt_zero": True,
    }

    thinking_lengths: dict[str, list[float]] = defaultdict(list)
    for record in sample_records:
        behavior = record["result"]["behavior"]
        thinking_lengths[behavior].append(
            float(record["result"]["boundary"].get("thinking_tokens", 0))
        )
    length_sensitivity = {
        "thinking_tokens_by_behavior": {
            key: _summarize(value) for key, value in sorted(thinking_lengths.items())
        },
        "fixed_preanswer32": feature_reports[PRIMARY_FEATURE],
        "all_decode": feature_reports[FULL_DECODE_FEATURE],
        "conditional_logit_with_length": primary["conditional_logit_with_length"],
        "length_matched": length_matched_analysis(
            sample_records, PRIMARY_FEATURE, families
        ),
    }

    control: dict[str, Any] = {
        "required_by_gate": gate_passed,
        "available_records": len(control_records),
    }
    if control_records:
        control_gate = mixed_prompt_gate(control_records)
        control["behavior_gate_descriptive"] = control_gate
        control["primary_feature"] = _feature_report(
            control_records, PRIMARY_FEATURE, families
        )
        control["axis_projection"] = _feature_report(
            control_records, PRIMARY_AXIS_FEATURE, families
        )
    else:
        control["status"] = "not_run" if not gate_passed else "awaiting_required_F_control"

    if gate_passed and not control_records:
        overall_status = "awaiting_control_f"
    elif final_requested and gate_passed and len(control_records) != 360:
        raise RuntimeError("Final E1 analysis requires all 360 F-control trajectories")
    else:
        overall_status = "passed"
    return {
        "status": overall_status,
        "mixed_prompt_gate": gate,
        "primary": primary,
        "phase_interaction": phase_interaction,
        "layer_interaction": layer_interaction,
        "length_sensitivity": length_sensitivity,
        "F_control": control,
        "interpretation_boundary": (
            "Uncapped observational geometry only; no causal intervention and no precise "
            "resistance-moment claim without positive direct interactions."
        ),
        "records_available": len(all_records),
    }


def _ci_supports_positive(report: Mapping[str, Any]) -> bool:
    ci = report.get("ci95")
    return bool(isinstance(ci, list) and len(ci) == 2 and ci[0] > 0)


def write_summary(
    run_dir: Path,
    analysis: Mapping[str, Any],
    bridge: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> None:
    gate = analysis["mixed_prompt_gate"]
    primary = analysis["primary"]["primary_feature"]
    axis = analysis["primary"]["assistant_axis_projection"]
    phase = analysis["phase_interaction"]["result"]
    layer = analysis["layer_interaction"]["result"]
    lines = [
        "# E1 Token-resolved resistance geometry 摘要",
        "",
        f"- 状态：`{analysis['status']}`",
        "- 性质：development、uncapped、monitor-only observational geometry",
        "- 干预：未运行 p25/p1/steering/patching/J-Lens/MUSE/probe",
        f"- greedy bridge：{bridge.get('exact_response_matches')}/144 逐字一致",
        f"- E1.2：{gate['sample_trajectories']}/720",
        f"- resist / yield：{gate['total_resist']} / {gate['total_yield']}",
        (
            f"- mixed prompts：{gate['mixed_prompts']}；minority>=3："
            f"{gate['mixed_prompts_minority_ge_3']}"
        ),
        f"- 数据充分性闸门：`{gate['status']}`",
        "",
        "## 同题主比较",
        "",
        (
            f"preanswer_32 / layers50–53 的 resist−yield 修正量："
            f"{primary.get('point_estimate')}，95% CI {primary.get('ci95')}。"
        ),
        (
            f"同位置 human-readable Assistant Axis 投影差："
            f"{axis.get('point_estimate')}，95% CI {axis.get('ci95')}。"
        ),
        "",
        "## 直接交互",
        "",
        (
            f"阶段交互 D_preanswer−D_prefill：{phase.get('point_estimate')}，"
            f"95% CI {phase.get('ci95')}；正向闸门={_ci_supports_positive(phase)}。"
        ),
        (
            f"层组交互 D_50–53−D_46–49：{layer.get('point_estimate')}，"
            f"95% CI {layer.get('ci95')}；正向闸门={_ci_supports_positive(layer)}。"
        ),
        "",
        "## 解释边界",
        "",
    ]
    if gate["status"] != "passed":
        lines.append(
            "sampling 未提供协议要求的同题离散行为变异；不得声称完成 resist/yield 几何比较，"
            "只保留连续 answer-margin 与描述性轨迹结果。"
        )
    elif not (_ci_supports_positive(phase) and _ci_supports_positive(layer)):
        lines.append(
            "同题分析可估计，但预注册的阶段或层组直接交互未同时过闸；不得称为已定位的"
            "后层 pre-answer 抵抗状态。"
        )
    else:
        lines.append(
            "同题主差与两个直接交互方向一致；仍属于无干预相关几何，需结合长度敏感性和"
            "F控制后才可决定是否进入E2。"
        )
    lines.extend(
        [
            "",
            (
                "行为计数详见 `behavior_counts.json`："
                f"{json.dumps(behavior.get('by_stage', {}), ensure_ascii=False)}"
            ),
            "",
            "E1到此按协议停止，不自动进入E2或任何干预/机制工具。",
        ]
    )
    (run_dir / "E1_SUMMARY_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.run_dir / "INPUT_MANIFEST.json"
    smoke_path = args.run_dir / "SMOKE_AUDIT.json"
    bridge_path = args.run_dir / "GREEDY_BRIDGE_AUDIT.json"
    for path in (input_path, smoke_path, bridge_path):
        if not path.exists():
            raise FileNotFoundError(path)
    input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    if (
        input_manifest.get("status") != "passed"
        or smoke.get("status") != "passed"
        or bridge.get("status") != "passed"
    ):
        raise RuntimeError("Input, smoke, and bridge audits must pass before E1 analysis")
    current_git = runner.git_provenance()
    if current_git != input_manifest["git"]:
        raise RuntimeError("Analysis code provenance differs from INPUT_MANIFEST")

    formal, axis_rows = runner.validate_input_rows()
    smoke_records = enrich_records(
        [
            record
            for record in load_stage(args.run_dir, "smoke", formal, axis_rows, required=True)
            if record["result"]["monitor_enabled"]
        ]
    )
    bridge_records = enrich_records(
        load_stage(args.run_dir, "bridge", formal, axis_rows, required=True)
    )
    sample_records = enrich_records(
        load_stage(args.run_dir, "sample-s", formal, axis_rows, required=True)
    )
    gate = mixed_prompt_gate(sample_records)
    atomic_write_json(args.run_dir / "mixed_prompt_gate.json", gate)
    control_required = gate["status"] == "passed"
    control_records = enrich_records(
        load_stage(
            args.run_dir,
            "control-f",
            formal,
            axis_rows,
            required=args.stage == "final" and control_required,
        )
    )
    if args.stage == "final" and not control_required and control_records:
        raise RuntimeError("F control exists despite failed S data-sufficiency gate")

    monitor_records = [*smoke_records, *bridge_records, *sample_records, *control_records]
    index_rows, combined_arrays = consolidate_trajectories(args.run_dir, monitor_records)
    margin_rows = answer_margin_rows(monitor_records)
    _atomic_write_jsonl(args.run_dir / "answer_margin_results.jsonl", margin_rows)
    behavior = build_behavior_counts(monitor_records)
    atomic_write_json(args.run_dir / "behavior_counts.json", behavior)

    analysis = analyze(
        args.run_dir,
        sample_records,
        control_records,
        monitor_records,
        gate,
        final_requested=args.stage == "final",
    )
    atomic_write_json(
        args.run_dir / "primary_within_prompt_results.json", analysis["primary"]
    )
    atomic_write_json(args.run_dir / "length_sensitivity.json", analysis["length_sensitivity"])
    atomic_write_json(args.run_dir / "phase_interaction.json", analysis["phase_interaction"])
    atomic_write_json(args.run_dir / "layer_interaction.json", analysis["layer_interaction"])
    write_summary(args.run_dir, analysis, bridge, behavior)

    output_names = (
        "INPUT_MANIFEST.json",
        "SMOKE_AUDIT.json",
        "GREEDY_BRIDGE_AUDIT.json",
        "trajectory_index.jsonl",
        "token_projection_arrays.npz",
        "answer_margin_results.jsonl",
        "behavior_counts.json",
        "mixed_prompt_gate.json",
        "primary_within_prompt_results.json",
        "length_sensitivity.json",
        "phase_interaction.json",
        "layer_interaction.json",
        "E1_SUMMARY_ZH.md",
    )
    outputs = {
        name: {
            "sha256": file_sha256(args.run_dir / name),
            "bytes": (args.run_dir / name).stat().st_size,
        }
        for name in output_names
    }
    run_manifest = {
        "status": analysis["status"],
        "analysis_stage": args.stage,
        "branch": current_git["branch"],
        "commit": current_git["commit"],
        "started_from": "476c03a",
        "protocol": str(runner.PROTOCOL.relative_to(ROOT)),
        "records": {
            "smoke_monitor": len(smoke_records),
            "greedy_bridge": len(bridge_records),
            "sample_s": len(sample_records),
            "control_f": len(control_records),
            "trajectory_index": len(index_rows),
            "prefill_token_layer_rows": (
                int(combined_arrays["prefill_axis"].shape[0])
                if "prefill_axis" in combined_arrays
                else 0
            ),
            "decode_token_layer_rows": (
                int(combined_arrays["decode_axis"].shape[0])
                if "decode_axis" in combined_arrays
                else 0
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "generation_runtime": {
            stage: json.loads((args.run_dir / f"_runtime_{stage}.json").read_text())
            for stage in ("smoke", "bridge", "sample-s", "control-f")
            if (args.run_dir / f"_runtime_{stage}.json").exists()
        },
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "gpu_or_model_execution": True,
        "intervention_applied": False,
        "new_dataset_built": False,
        "capping_or_steering_executed": False,
        "mechanism_tools_executed": [],
        "p1_or_new_unfrozen_seed_executed": False,
        "mandatory_stop": True,
        "outputs": outputs,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(args.run_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "gate": gate["status"],
                "sample_records": len(sample_records),
                "control_records": len(control_records),
                "run_dir": str(args.run_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
