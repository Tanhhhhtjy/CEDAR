#!/usr/bin/env python3
"""Audit E2B v1.1 cache-faithful forced replay."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cached"
V1_DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cross"
for path in (DATA, V1_DATA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e2b_cached_common import (  # noqa: E402
    CAP_LAYERS,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_SMOKE_CELLS,
    FORWARD_CONDITIONS,
    LABELS,
    REPLAY_TOLERANCE,
    SOURCE_TRACES,
    TRACE_SOURCES,
    canonical_text_sha256,
    file_sha256,
    ordered_cells,
    read_json,
    read_jsonl,
    write_json,
)
from e2b_common import close  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/e2b_fixed_trace_cached_v1_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def finite_logits(row: dict[str, Any]) -> bool:
    logits = (row.get("logit_diagnostics") or {}).get("logits")
    return (
        isinstance(logits, dict)
        and set(logits) == set(LABELS)
        and all(math.isfinite(float(value)) for value in logits.values())
    )


def audit(output_root: Path, *, scope: str) -> dict[str, Any]:
    run_dir = output_root / f"{scope}-fixed-trace-seed0"
    results_path = run_dir / "results.jsonl"
    runtime_path = run_dir / "runtime.json"
    traces = read_jsonl(SOURCE_TRACES)
    manifest = read_json(DATA / "manifest.json")
    expected_matrix = ordered_cells(
        traces, scope=scope, smoke_ids=manifest["smoke_prompt_ids"]
    )
    expected_keys = {(trace["trace_id"], condition) for trace, condition in expected_matrix}
    trace_by_id = {trace["trace_id"]: trace for trace in traces}
    rows = read_jsonl(results_path) if results_path.is_file() else []
    runtime = read_json(runtime_path) if runtime_path.is_file() else {}
    failures: list[str] = []
    warnings: list[str] = []
    expected_count = EXPECTED_SMOKE_CELLS if scope == "smoke" else EXPECTED_FORMAL_CELLS

    if len(rows) != expected_count:
        failures.append(f"row_count:{len(rows)}!={expected_count}")
    observed_keys = {(row.get("trace_id"), row.get("forward_condition")) for row in rows}
    if len(observed_keys) != len(rows):
        failures.append("duplicate_cells")
    if observed_keys != expected_keys:
        failures.append(
            f"matrix_mismatch:missing={len(expected_keys-observed_keys)}:"
            f"extra={len(observed_keys-expected_keys)}"
        )
    if Counter(row.get("forward_condition") for row in rows) != Counter(
        {condition: expected_count // 2 for condition in FORWARD_CONDITIONS}
    ):
        failures.append("forward_condition_balance")
    if Counter(row.get("trace_source") for row in rows) != Counter(
        {source: expected_count // 2 for source in TRACE_SOURCES}
    ):
        failures.append("trace_source_balance")

    hash_validation: dict[str, Any] = {"status": "failed"}
    if not runtime_path.is_file():
        failures.append("runtime_missing")
    elif not results_path.is_file():
        failures.append("results_missing")
    else:
        raw = file_sha256(results_path)
        canonical = canonical_text_sha256(results_path)
        stored = runtime.get("results_sha256")
        match_mode = (
            "raw_file"
            if stored == raw
            else "canonical_lf_after_checkout_normalization"
            if stored == canonical
            else None
        )
        if match_mode is None:
            failures.append("runtime_results_hash_mismatch")
        if runtime.get("complete_results") != len(rows):
            failures.append("runtime_complete_results_mismatch")
        hash_validation = {
            "status": "passed" if match_mode else "failed",
            "match_mode": match_mode,
            "runtime_sha256": stored,
            "raw_file_sha256": raw,
            "canonical_lf_sha256": canonical,
        }

    expected_layers = {str(layer) for layer in CAP_LAYERS}
    replay_failures: list[str] = []
    replay_checked = 0
    uncapped_checked = 0
    p25_checked = 0
    for row in rows:
        cell_id = str(row.get("cell_id"))
        trace = trace_by_id.get(str(row.get("trace_id")))
        if trace is None:
            failures.append(f"unknown_trace:{cell_id}")
            continue
        for field in (
            "source_id",
            "trace_source",
            "family_id",
            "base_id",
            "variant",
            "arm",
            "initial_answer",
            "target_answer",
            "prompt_sha256",
            "prompt_token_ids_sha256",
            "trace_prefix_text_sha256",
            "trace_prefix_token_ids_sha256",
            "teacher_forced_input_token_ids_sha256",
            "candidate_token_ids",
        ):
            if row.get(field) != trace.get(field):
                failures.append(f"trace_field:{cell_id}:{field}")
        if row.get("replay_engine") != "model.generate_kv_cache_forced_tokens":
            failures.append(f"replay_engine:{cell_id}")
        if row.get("model_generate_used") is not True:
            failures.append(f"model_generate_not_used:{cell_id}")
        if row.get("natural_generation") is not False or row.get("generated_text") is not False:
            failures.append(f"natural_or_text_generation:{cell_id}")
        if row.get("retry_count") != 0 or row.get("answer_repair_performed") is not False:
            failures.append(f"retry_or_repair:{cell_id}")
        if row.get("forced_trace_token_ids_sha256") != trace["trace_prefix_token_ids_sha256"]:
            failures.append(f"forced_trace_hash:{cell_id}")
        if row.get("forced_trace_tokens") != trace["trace_prefix_tokens"]:
            failures.append(f"forced_trace_length:{cell_id}")
        if row.get("terminal_probe_token_id") != trace["generated_label_token_id"]:
            failures.append(f"terminal_probe_token:{cell_id}")
        if row.get("terminal_probe_label") != trace["generated_label"]:
            failures.append(f"terminal_probe_label:{cell_id}")
        order = row.get("processor_order") or {}
        expected_processor_calls = trace["trace_prefix_tokens"] + 1
        if order.get("capture_before_force_all") is not True or order.get("violations") != []:
            failures.append(f"processor_order:{cell_id}")
        if order.get("capture_calls") != expected_processor_calls:
            failures.append(f"capture_calls:{cell_id}")
        if order.get("force_calls") != expected_processor_calls:
            failures.append(f"force_calls:{cell_id}")
        if order.get("capture_call_index") != trace["trace_prefix_tokens"]:
            failures.append(f"capture_boundary:{cell_id}")
        if not finite_logits(row):
            failures.append(f"nonfinite_or_missing_logits:{cell_id}")
        if row.get("hook_counts_before") != row.get("hook_counts_after"):
            failures.append(f"hook_leak:{cell_id}")
        if any(int(value) != 0 for value in (row.get("hook_counts_after") or {}).values()):
            failures.append(f"residual_hook:{cell_id}")

        monitor = row.get("monitor_telemetry") or {}
        if monitor.get("apply_changes") is not False:
            failures.append(f"monitor_not_read_only:{cell_id}")
        if set(monitor.get("by_layer", {})) != expected_layers:
            failures.append(f"monitor_layer_set:{cell_id}")
        for layer in expected_layers:
            phases = monitor.get("by_layer", {}).get(layer, {})
            prefill = phases.get("prefill", {})
            decode = phases.get("decode", {})
            if prefill.get("forward_calls") != 1:
                failures.append(f"prefill_forward_count:{cell_id}:{layer}")
            if prefill.get("observed_positions") != row.get("prompt_tokens"):
                failures.append(f"prefill_position_count:{cell_id}:{layer}")
            if decode.get("forward_calls") != row.get("trace_prefix_tokens"):
                failures.append(f"decode_forward_count:{cell_id}:{layer}")
            if decode.get("observed_positions") != row.get("trace_prefix_tokens"):
                failures.append(f"decode_position_count:{cell_id}:{layer}")
            for phase_name, phase in (("prefill", prefill), ("decode", decode)):
                for key in (
                    "trigger_rate",
                    "projection_correction_sum",
                    "mean_projection_correction",
                ):
                    value = phase.get(key)
                    if value is None or not math.isfinite(float(value)):
                        failures.append(f"nonfinite_monitor:{cell_id}:{layer}:{phase_name}:{key}")

        condition = row.get("forward_condition")
        full = row.get("full_cap_stats")
        if condition == "uncapped":
            if full is not None:
                failures.append(f"uncapped_has_intervention:{cell_id}")
        elif condition == "full_p25":
            if not isinstance(full, dict) or set(full.get("by_layer", {})) != expected_layers:
                failures.append(f"full_p25_layer_set:{cell_id}")
            else:
                for layer, stats in full["by_layer"].items():
                    if stats.get("total_positions") != row.get("teacher_forced_input_tokens"):
                        failures.append(f"cap_position_count:{cell_id}:{layer}")
                    for key in (
                        "trigger_rate",
                        "projection_correction_sum",
                        "mean_projection_correction",
                    ):
                        if not math.isfinite(float(stats.get(key, math.nan))):
                            failures.append(f"nonfinite_cap:{cell_id}:{layer}:{key}")
        else:
            failures.append(f"unknown_condition:{cell_id}")

        if row.get("is_diagonal_replay"):
            replay_checked += 1
            if condition == "uncapped":
                uncapped_checked += 1
            else:
                p25_checked += 1
            observed = row.get("logit_diagnostics", {}).get("logits", {})
            anchor = trace.get("anchor_answer_logits", {}).get("logits", {})
            if set(observed) != set(anchor) or any(
                not close(float(observed[label]), float(anchor[label])) for label in observed
            ):
                deltas = {
                    label: float(observed.get(label, math.nan))
                    - float(anchor.get(label, math.nan))
                    for label in LABELS
                }
                replay_failures.append(f"{cell_id}:{deltas}")

    expected_diagonal = expected_count // 2
    if replay_checked != expected_diagonal:
        failures.append(f"replay_count:{replay_checked}!={expected_diagonal}")
    if replay_failures:
        failures.extend(f"replay_logits:{value}" for value in replay_failures)

    runtime_p25 = runtime.get("official_p25") or {}
    if runtime_p25.get("axis_sha256") != manifest["assets"]["axis_sha256"]:
        failures.append("runtime_axis_hash")
    if runtime_p25.get("capping_config_sha256") != manifest["assets"]["capping_config_sha256"]:
        failures.append("runtime_cap_config_hash")
    if set(runtime_p25.get("layers") or {}) != expected_layers:
        failures.append("runtime_p25_layer_set")
    if runtime.get("replay_engine") != "model.generate_kv_cache_forced_tokens":
        failures.append("runtime_replay_engine")
    if runtime.get("processor_order_violations") not in (None, []):
        failures.append("runtime_processor_order_violations")

    report = {
        "schema": "cedar.e2b_fixed_trace_cached.audit.v1_1",
        "scope": scope,
        "status": "passed" if not failures else "failed",
        "counts": {
            "rows": len(rows),
            "expected": expected_count,
            "trace_sources": dict(Counter(row.get("trace_source") for row in rows)),
            "forward_conditions": dict(
                Counter(row.get("forward_condition") for row in rows)
            ),
            "nonfinite_logits": sum(not finite_logits(row) for row in rows),
            "hook_leaks": sum(
                row.get("hook_counts_before") != row.get("hook_counts_after") for row in rows
            ),
            "processor_order_failures": sum(
                (row.get("processor_order") or {}).get("capture_before_force_all") is not True
                for row in rows
            ),
        },
        "replay_gate": {
            "tolerance": REPLAY_TOLERANCE,
            "checked": replay_checked,
            "expected": expected_diagonal,
            "uncapped_checked": uncapped_checked,
            "full_p25_checked": p25_checked,
            "passed": replay_checked == expected_diagonal and not replay_failures,
            "failures": replay_failures,
        },
        "results_hash_validation": hash_validation,
        "forced_trace_contract": {
            "capture_before_force": True,
            "terminal_probe_excluded_from_trace_hash": True,
            "natural_generation": False,
        },
        "failures": failures,
        "warnings": warnings,
    }
    output_name = "SMOKE_AUDIT.json" if scope == "smoke" else "E2B_AUDIT.json"
    write_json(output_root / output_name, report)
    return report


def main() -> None:
    args = parse_args()
    report = audit(args.output_root.resolve(), scope=args.scope)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
