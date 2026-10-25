#!/usr/bin/env python3
"""Engineering audit for RG1-P Stage A."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/rg1p_stagea"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from rg1p_common import (  # noqa: E402
    CAP_LAYERS,
    CAPTURE_LAYERS,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_SMOKE_CELLS,
    FINAL_LENS_LOGIT_TOLERANCE,
    NOOP_CONDITION,
    OFFICIAL_CONDITIONS,
    RANDOM_CONDITIONS,
    RANDOM_SEEDS,
    REALIZED_NORM_ABS_TOLERANCE,
    REALIZED_NORM_REL_TOLERANCE,
    REPLAY_TOLERANCE,
    SOURCE_RESULTS,
    file_sha256,
    read_json,
    read_jsonl,
    require,
    tensor_sha256,
    write_json,
)

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/rg1p_stagea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.float()).all().item())


def bf16_alignment_diagnostics(
    cap_input: torch.Tensor, cap_output: torch.Tensor, axis: torch.Tensor
) -> dict[str, float | bool]:
    """Audit an axis correction at the precision seen by the BF16 model.

    Small dense axis updates are rounded componentwise when written back to a
    BF16 residual. Their raw cosine can therefore be low even when every
    off-axis component is explainable by one final rounding step. The gate
    requires a positive axis projection and bounds the off-axis residual by
    the exact half-ULP L2 envelope of the stored output.
    """
    require(cap_input.dtype == cap_output.dtype == torch.bfloat16, "Alignment dtype changed")
    unit = axis.float() / torch.linalg.vector_norm(axis.float())
    correction = cap_output.float() - cap_input.float()
    norm = float(torch.linalg.vector_norm(correction).item())
    if norm == 0:
        return {
            "correction_norm": 0.0,
            "cosine": 1.0,
            "parallel_projection": 0.0,
            "off_axis_norm": 0.0,
            "half_ulp_l2_envelope": 0.0,
            "off_axis_to_envelope": 0.0,
            "passed": True,
        }
    parallel = float(torch.dot(correction, unit).item())
    off_axis = correction - parallel * unit
    off_axis_norm = float(torch.linalg.vector_norm(off_axis).item())
    positive = torch.full_like(cap_output, float("inf"))
    negative = torch.full_like(cap_output, float("-inf"))
    upward = torch.nextafter(cap_output, positive).float() - cap_output.float()
    downward = cap_output.float() - torch.nextafter(cap_output, negative).float()
    half_ulp = 0.5 * torch.maximum(torch.abs(upward), torch.abs(downward))
    envelope = float(torch.linalg.vector_norm(half_ulp).item())
    ratio = off_axis_norm / envelope if envelope else math.inf
    return {
        "correction_norm": norm,
        "cosine": parallel / norm,
        "parallel_projection": parallel,
        "off_axis_norm": off_axis_norm,
        "half_ulp_l2_envelope": envelope,
        "off_axis_to_envelope": ratio,
        "passed": parallel > 0 and off_axis_norm <= envelope + 1e-6,
    }


def audit(scope: str, output_root: Path) -> dict[str, Any]:
    expected = EXPECTED_SMOKE_CELLS if scope == "smoke" else EXPECTED_FORMAL_CELLS
    run_dir = output_root / f"{scope}-stagea-seed0"
    results_path = run_dir / "results.jsonl"
    runtime_path = run_dir / "runtime.json"
    require(results_path.is_file(), "RG1-P results file missing")
    require(runtime_path.is_file(), "RG1-P runtime file missing")
    rows = read_jsonl(results_path)
    runtime = read_json(runtime_path)
    failures: list[str] = []
    warnings: list[str] = []

    if len(rows) != expected:
        failures.append(f"rows:{len(rows)}!={expected}")
    cell_ids = [row.get("cell_id") for row in rows]
    if len(set(cell_ids)) != len(cell_ids):
        failures.append("duplicate_cell_ids")
    conditions = Counter(row.get("forward_condition") for row in rows)
    per_official = 16 if scope == "smoke" else 192
    per_random = 8 if scope == "smoke" else 96
    expected_conditions = {
        "uncapped": per_official,
        "full_p25": per_official,
        NOOP_CONDITION: per_random,
        **{condition: per_random for condition in RANDOM_CONDITIONS},
    }
    if conditions != expected_conditions:
        failures.append(f"condition_counts:{dict(conditions)}")
    kinds = Counter(row.get("cell_kind") for row in rows)
    expected_kinds = {
        "official": 32 if scope == "smoke" else 384,
        "engineering_noop": 8 if scope == "smoke" else 96,
        "random_reference": 32 if scope == "smoke" else 384,
    }
    if kinds != expected_kinds:
        failures.append(f"cell_kind_counts:{dict(kinds)}")

    anchors = {
        (row["source_id"], row["trace_source"], row["forward_condition"]): row
        for row in read_jsonl(SOURCE_RESULTS)
    }
    replay_failures: list[dict[str, Any]] = []
    replay_max_abs = 0.0
    tensor_failures: list[str] = []
    nonfinite_tensors: list[str] = []
    hook_leaks: list[str] = []
    processor_failures: list[str] = []
    schedule_failures: list[str] = []
    norm_failures: list[dict[str, Any]] = []
    random_sign_pair_failures: list[dict[str, Any]] = []
    terminal_alignment_failures: list[dict[str, Any]] = []
    terminal_alignment_diagnostics: list[dict[str, Any]] = []
    noop_failures: list[dict[str, Any]] = []
    lens_anchor_failures: list[dict[str, Any]] = []
    top_token_invalid: list[str] = []

    axes = load_file(DATA / "official_axis_units.safetensors", device="cpu")[
        "official_axis_units"
    ].float()
    require(list(axes.shape) == [len(CAP_LAYERS), 5120], "Official axis tensor shape changed")
    loaded_tensors: dict[str, dict[str, torch.Tensor]] = {}
    row_by_cell = {str(row["cell_id"]): row for row in rows}

    for row in rows:
        cell_id = str(row["cell_id"])
        hook_before = row.get("hook_counts_before", {})
        hook_after = row.get("hook_counts_after", {})
        if hook_before != hook_after or any(int(value) != 0 for value in hook_after.values()):
            hook_leaks.append(cell_id)
        if not row.get("processor_order", {}).get("capture_before_force_all"):
            processor_failures.append(cell_id)
        if row.get("forced_trace_token_ids_sha256") != row.get("trace_prefix_token_ids_sha256"):
            failures.append(f"forced_trace_hash:{cell_id}")
        if row.get("candidate_token_ids") != {"A": 32, "B": 33, "C": 34, "D": 35}:
            failures.append(f"candidate_token_ids:{cell_id}")
        if not row.get("logit_diagnostics", {}).get("top_token_is_option"):
            top_token_invalid.append(cell_id)
        logits = row.get("logit_diagnostics", {}).get("logits", {})
        if set(logits) != {"A", "B", "C", "D"} or not all(
            math.isfinite(float(value)) for value in logits.values()
        ):
            failures.append(f"invalid_logits:{cell_id}")

        if row["forward_condition"] in {*OFFICIAL_CONDITIONS, NOOP_CONDITION}:
            anchor_condition = (
                "uncapped"
                if row["forward_condition"] == NOOP_CONDITION
                else row["forward_condition"]
            )
            key = (row["source_id"], row["trace_source"], anchor_condition)
            anchor = anchors.get(key)
            if anchor is None:
                replay_failures.append({"cell_id": cell_id, "reason": "missing_anchor"})
            else:
                errors = {
                    label: abs(
                        float(logits[label]) - float(anchor["logit_diagnostics"]["logits"][label])
                    )
                    for label in ("A", "B", "C", "D")
                }
                maximum = max(errors.values())
                replay_max_abs = max(replay_max_abs, maximum)
                if maximum > REPLAY_TOLERANCE:
                    replay_failures.append(
                        {"cell_id": cell_id, "max_abs": maximum, "by_label": errors}
                    )

        tensor_path = run_dir / str(row.get("tensor_file", ""))
        if not tensor_path.is_file():
            tensor_failures.append(f"missing:{cell_id}")
            continue
        if file_sha256(tensor_path) != row.get("tensor_file_sha256"):
            tensor_failures.append(f"hash:{cell_id}")
            continue
        tensors = load_file(tensor_path, device="cpu")
        loaded_tensors[cell_id] = tensors
        if sorted(tensors) != row.get("tensor_keys"):
            tensor_failures.append(f"recorded_keys:{cell_id}")
        if {name: list(value.shape) for name, value in tensors.items()} != row.get(
            "tensor_shapes"
        ):
            tensor_failures.append(f"recorded_shapes:{cell_id}")
        if list(tensors.get("block_outputs", torch.empty(0)).shape) != [
            len(CAPTURE_LAYERS),
            5120,
        ]:
            tensor_failures.append(f"block_shape:{cell_id}")
        if tensors.get("block_outputs", torch.empty(0)).dtype != torch.bfloat16:
            tensor_failures.append(f"block_dtype:{cell_id}")
        for name, tensor in tensors.items():
            if not _finite_tensor(tensor):
                nonfinite_tensors.append(f"{cell_id}:{name}")
        if row["forward_condition"] == "full_p25":
            required = {
                "block_outputs",
                "official_schedule_norms",
                "cap_input_terminal",
                "cap_output_terminal",
            }
            if set(tensors) != required:
                tensor_failures.append(f"p25_keys:{cell_id}:{sorted(tensors)}")
            schedule = tensors.get("official_schedule_norms", torch.empty(0))
            if list(schedule.shape) != [len(CAP_LAYERS), row["causal_schedule_positions"]]:
                tensor_failures.append(f"schedule_shape:{cell_id}")
            telemetry = row.get("official_schedule") or {}
            if telemetry.get("schedule_sha256") != tensor_sha256(schedule):
                schedule_failures.append(f"official_schedule_hash:{cell_id}")
            cap_input = tensors.get("cap_input_terminal", torch.empty(0)).float()
            cap_output = tensors.get("cap_output_terminal", torch.empty(0)).float()
            if (
                list(cap_input.shape) == [len(CAP_LAYERS), 5120]
                and cap_output.shape == cap_input.shape
            ):
                for index, layer in enumerate(CAP_LAYERS):
                    values = bf16_alignment_diagnostics(
                        tensors["cap_input_terminal"][index],
                        tensors["cap_output_terminal"][index],
                        axes[index],
                    )
                    item = {"cell_id": cell_id, "layer": layer, **values}
                    terminal_alignment_diagnostics.append(item)
                    if not values["passed"]:
                        terminal_alignment_failures.append(item)
        elif row["forward_condition"] in RANDOM_CONDITIONS:
            required = {
                "block_outputs",
                "injection_input_terminal",
                "injection_output_terminal",
                "random_realized_norms",
            }
            if set(tensors) != required:
                tensor_failures.append(f"random_keys:{cell_id}:{sorted(tensors)}")
            source_cell = row.get("random_schedule_source_cell")
            source_row = row_by_cell.get(str(source_cell))
            if source_row is None:
                schedule_failures.append(f"missing_schedule_source:{cell_id}")
            else:
                source_telemetry = source_row.get("official_schedule") or {}
                injection = row.get("random_injection") or {}
                source_tensors = loaded_tensors.get(str(source_cell))
                if source_tensors is None:
                    schedule_failures.append(f"missing_schedule_tensor:{cell_id}")
                    source_schedule = torch.empty(0)
                else:
                    source_schedule = source_tensors.get(
                        "official_schedule_norms", torch.empty(0)
                    ).float()
                realized_norms = tensors.get("random_realized_norms", torch.empty(0)).float()
                if injection.get("schedule_sha256") != source_telemetry.get("schedule_sha256"):
                    schedule_failures.append(f"schedule_hash_mismatch:{cell_id}")
                if injection.get("realized_norms_sha256") != tensor_sha256(realized_norms):
                    schedule_failures.append(f"realized_norm_hash_mismatch:{cell_id}")
                if injection.get("positions_per_layer") != source_telemetry.get(
                    "positions_per_layer"
                ):
                    schedule_failures.append(f"schedule_position_mismatch:{cell_id}")
                if injection.get("triggered_by_layer") != source_telemetry.get(
                    "triggered_by_layer"
                ):
                    schedule_failures.append(f"schedule_trigger_mismatch:{cell_id}")
                if realized_norms.shape != source_schedule.shape:
                    schedule_failures.append(f"realized_norm_shape_mismatch:{cell_id}")
                else:
                    if not torch.equal(realized_norms > 0, source_schedule > 0):
                        schedule_failures.append(f"injection_position_mismatch:{cell_id}")
                    tensor_abs = torch.abs(realized_norms - source_schedule)
                    tensor_rel = torch.zeros_like(tensor_abs)
                    nonzero = source_schedule > 0
                    tensor_rel[nonzero] = tensor_abs[nonzero] / source_schedule[nonzero]
                    tensor_max_abs = float(tensor_abs.max().item())
                    tensor_max_rel = float(tensor_rel.max().item())
                    tensor_violations = (
                        tensor_abs > REALIZED_NORM_ABS_TOLERANCE
                    ) & (tensor_rel > REALIZED_NORM_REL_TOLERANCE)
                    if torch.any(tensor_violations):
                        norm_failures.append(
                            {
                                "cell_id": cell_id,
                                "source": "independent_tensor_recompute",
                                "violating_positions": int(
                                    torch.count_nonzero(tensor_violations).item()
                                ),
                                "realized_max_abs": tensor_max_abs,
                                "realized_max_rel": tensor_max_rel,
                            }
                        )
                requested = float(injection.get("requested_norm_max_abs_error", math.inf))
                realized_abs = float(injection.get("realized_norm_max_abs_error", math.inf))
                realized_rel = float(injection.get("realized_norm_max_relative_error", math.inf))
                direction_min = float(injection.get("realized_direction_cosine_min", -math.inf))
                if not math.isfinite(direction_min) or direction_min <= 0:
                    schedule_failures.append(f"random_direction_sign:{cell_id}:{direction_min}")
                if requested > 1e-5:
                    norm_failures.append(
                        {
                            "cell_id": cell_id,
                            "requested_max_abs": requested,
                            "realized_max_abs": realized_abs,
                            "realized_max_rel": realized_rel,
                        }
                    )
        elif row["forward_condition"] in {"uncapped", NOOP_CONDITION}:
            required = {"block_outputs"}
            if set(tensors) != required:
                tensor_failures.append(f"baseline_keys:{cell_id}:{sorted(tensors)}")

    for row in rows:
        if row["forward_condition"] != "uncapped" or row["trace_source"] != "e2a_uncapped":
            continue
        for seed in RANDOM_SEEDS:
            positive = loaded_tensors.get(f"{row['trace_id']}::random_s{seed}_pos")
            negative = loaded_tensors.get(f"{row['trace_id']}::random_s{seed}_neg")
            if positive is None or negative is None:
                continue
            pos_norms = positive["random_realized_norms"].float()
            neg_norms = negative["random_realized_norms"].float()
            difference = torch.abs(pos_norms - neg_norms)
            denominator = torch.maximum(pos_norms, neg_norms)
            relative = torch.zeros_like(difference)
            nonzero = denominator > 0
            relative[nonzero] = difference[nonzero] / denominator[nonzero]
            max_abs = float(difference.max().item())
            max_rel = float(relative.max().item())
            sign_violations = (
                difference > 2 * REALIZED_NORM_ABS_TOLERANCE
            ) & (relative > 2 * REALIZED_NORM_REL_TOLERANCE)
            if torch.any(sign_violations):
                random_sign_pair_failures.append(
                    {
                        "trace_id": row["trace_id"],
                        "seed": seed,
                        "violating_positions": int(torch.count_nonzero(sign_violations).item()),
                        "max_abs": max_abs,
                        "max_rel": max_rel,
                    }
                )

    layer_metrics_path = run_dir / "layer_metrics.csv"
    lens_rows: list[dict[str, str]] = []
    expected_lens_pairs = 24 if scope == "smoke" else 288
    expected_official_pairs = 16 if scope == "smoke" else 192
    expected_noop_pairs = 8 if scope == "smoke" else 96
    if not layer_metrics_path.is_file():
        failures.append("missing_layer_metrics_csv")
    else:
        with layer_metrics_path.open(encoding="utf-8", newline="") as handle:
            lens_rows = list(csv.DictReader(handle))
        if len(lens_rows) != expected_lens_pairs * len(CAPTURE_LAYERS):
            failures.append(f"layer_metric_rows:{len(lens_rows)}")
        pair_ids = {row["pair_id"] for row in lens_rows}
        if len(pair_ids) != expected_lens_pairs:
            failures.append(f"layer_metric_pairs:{len(pair_ids)}")
        if len({(row["pair_id"], row["layer"]) for row in lens_rows}) != len(lens_rows):
            failures.append("duplicate_layer_metric_rows")
        for pair_id in pair_ids:
            coverage = {
                int(row["layer"]) for row in lens_rows if row["pair_id"] == pair_id
            }
            if coverage != set(CAPTURE_LAYERS):
                failures.append(f"layer_metric_coverage:{pair_id}")
        pair_kinds = Counter(
            next(row["pair_kind"] for row in lens_rows if row["pair_id"] == pair_id)
            for pair_id in pair_ids
        )
        if pair_kinds != {
            "official_p25": expected_official_pairs,
            "engineering_noop": expected_noop_pairs,
        }:
            failures.append(f"layer_metric_pair_kinds:{dict(pair_kinds)}")
        for row in lens_rows:
            numeric_keys = (
                "kl_opt",
                "kl_vocab",
                "delta_m_general_mean_noninitial",
                "delta_h_l2",
                "state_cosine",
                "uncapped_state_l2",
                "treatment_state_l2",
            )
            if not all(math.isfinite(float(row[key])) for key in numeric_keys):
                failures.append(f"nonfinite_lens:{row['pair_id']}:{row['layer']}")
            if int(row["layer"]) == 63:
                maximum = max(
                    float(row["final_layer_uncapped_anchor_error"]),
                    float(row["final_layer_treatment_anchor_error"]),
                )
                if maximum >= FINAL_LENS_LOGIT_TOLERANCE:
                    lens_anchor_failures.append(
                        {"pair_id": row["pair_id"], "max_abs": maximum}
                    )
            if row["pair_kind"] == "engineering_noop" and (
                float(row["kl_opt"]) >= 1e-6 or float(row["delta_h_l2"]) >= 1e-4
            ):
                noop_failures.append(
                    {
                        "pair_id": row["pair_id"],
                        "layer": int(row["layer"]),
                        "kl_opt": float(row["kl_opt"]),
                        "delta_h_l2": float(row["delta_h_l2"]),
                    }
                )

    results_hash_ok = runtime.get("results_sha256") == file_sha256(results_path)
    if not results_hash_ok:
        failures.append("runtime_results_hash")
    if runtime.get("complete_results") != expected:
        failures.append(f"runtime_count:{runtime.get('complete_results')}")
    if runtime.get("tensor_files") != expected:
        failures.append(f"runtime_tensor_count:{runtime.get('tensor_files')}")
    if runtime.get("layer_lens_pairs") != expected_lens_pairs:
        failures.append(f"runtime_lens_pairs:{runtime.get('layer_lens_pairs')}")
    if layer_metrics_path.is_file() and runtime.get("layer_metrics_sha256") != file_sha256(
        layer_metrics_path
    ):
        failures.append("runtime_layer_metrics_hash")

    failures.extend(f"replay:{item['cell_id']}" for item in replay_failures)
    failures.extend(f"tensor:{item}" for item in tensor_failures)
    failures.extend(f"nonfinite:{item}" for item in nonfinite_tensors)
    failures.extend(f"hook_leak:{item}" for item in hook_leaks)
    failures.extend(f"processor:{item}" for item in processor_failures)
    failures.extend(f"schedule:{item}" for item in schedule_failures)
    failures.extend(f"norm:{item['cell_id']}" for item in norm_failures)
    failures.extend(
        f"random_sign_pair:{item['trace_id']}:{item['seed']}"
        for item in random_sign_pair_failures
    )
    failures.extend(
        f"alignment:{item['cell_id']}:{item['layer']}" for item in terminal_alignment_failures
    )
    failures.extend(f"noop:{item['pair_id']}:{item['layer']}" for item in noop_failures)
    failures.extend(f"lens_anchor:{item['pair_id']}" for item in lens_anchor_failures)
    if top_token_invalid:
        warnings.append(f"top_token_not_option:{len(top_token_invalid)}")

    audit_record = {
        "schema": "cedar.rg1p_stagea.audit.v1",
        "scope": scope,
        "status": "passed" if not failures else "failed",
        "counts": {
            "expected": expected,
            "rows": len(rows),
            "conditions": dict(conditions),
            "cell_kinds": dict(kinds),
            "tensor_files": len(loaded_tensors),
            "nonfinite_tensors": len(nonfinite_tensors),
            "hook_leaks": len(hook_leaks),
            "schedule_failures": len(schedule_failures),
            "norm_failures": len(norm_failures),
            "random_sign_pair_failures": len(random_sign_pair_failures),
            "noop_failures": len(noop_failures),
            "lens_anchor_failures": len(lens_anchor_failures),
        },
        "replay_gate": {
            "checked": sum(
                1
                for row in rows
                if row["forward_condition"] in {*OFFICIAL_CONDITIONS, NOOP_CONDITION}
            ),
            "failures": replay_failures,
            "max_abs_logit_error": replay_max_abs,
            "tolerance": REPLAY_TOLERANCE,
            "passed": not replay_failures,
        },
        "schedule_match": {
            "failures": schedule_failures,
            "norm_failures": norm_failures,
            "random_sign_pair_failures": random_sign_pair_failures,
            "bf16_absolute_tolerance": REALIZED_NORM_ABS_TOLERANCE,
            "bf16_relative_tolerance": REALIZED_NORM_REL_TOLERANCE,
            "application": "per_position_absolute_or_relative",
            "passed": (
                not schedule_failures and not norm_failures and not random_sign_pair_failures
            ),
        },
        "official_terminal_alignment": {
            "failures": terminal_alignment_failures,
            "criterion": "positive_axis_projection_and_off_axis_within_half_ulp_l2_envelope",
            "minimum_realized_cosine": min(
                (float(item["cosine"]) for item in terminal_alignment_diagnostics),
                default=1.0,
            ),
            "maximum_off_axis_to_envelope": max(
                (float(item["off_axis_to_envelope"]) for item in terminal_alignment_diagnostics),
                default=0.0,
            ),
            "passed": not terminal_alignment_failures,
        },
        "engineering_noop_control": {
            "failures": noop_failures,
            "kl_opt_threshold": 1e-6,
            "delta_h_l2_threshold": 1e-4,
            "passed": not noop_failures,
        },
        "final_layer_logit_lens_anchor": {
            "failures": lens_anchor_failures,
            "max_abs_tolerance": FINAL_LENS_LOGIT_TOLERANCE,
            "passed": not lens_anchor_failures,
        },
        "output_space_integrity": {
            "top_token_not_option_count": len(top_token_invalid),
            "top_token_not_option_cells": top_token_invalid,
        },
        "results_hash_validation": {
            "status": "passed" if results_hash_ok else "failed",
            "runtime_sha256": runtime.get("results_sha256"),
            "actual_sha256": file_sha256(results_path),
        },
        "failures": failures,
        "warnings": warnings,
    }
    output_name = "SMOKE_AUDIT.json" if scope == "smoke" else "FORMAL_AUDIT.json"
    write_json(output_root / output_name, audit_record)
    return audit_record


def main() -> None:
    args = parse_args()
    result = audit(args.scope, args.output_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
