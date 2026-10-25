"""Contracts shared by the E2B cache-faithful replay amendment."""

from __future__ import annotations

import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V1_DATA = HERE.parent / "e2b_fixed_trace_cross"
if str(V1_DATA) not in sys.path:
    sys.path.insert(0, str(V1_DATA))

import e2b_common as v1  # noqa: E402

SCHEMA = "cedar.e2b_fixed_trace_cached.v1_1"
BRANCH = "agent/e2b-fixed-trace-cached-v1-1"
BASE_RESULT_COMMIT = "e2366b2b33011f6d795d7fbe7484be9e9752799b"
AMENDMENT = ROOT / "docs/E2B_FIXED_TRACE_CACHED_REPLAY_AMENDMENT.md"
V1_FAILURE_AUDIT = ROOT / "artifacts/m2/e2b_fixed_trace_cross/SMOKE_AUDIT.json"
V1_FAILURE_REPORT = ROOT / "artifacts/m2/e2b_fixed_trace_cross/E2B_SMOKE_FAILURE_REPORT.md"
V1_SMOKE_RESULTS = (
    ROOT
    / "artifacts/m2/e2b_fixed_trace_cross/smoke-fixed-trace-seed0/results.jsonl"
)
SOURCE_TRACES = V1_DATA / "frozen_traces192.jsonl"
SOURCE_MATRIX = V1_DATA / "formal_matrix384.jsonl"
SOURCE_MANIFEST = V1_DATA / "manifest.json"

MODEL_REPO = v1.MODEL_REPO
MODEL_REVISION = v1.MODEL_REVISION
MODEL_DTYPE = v1.MODEL_DTYPE
ATTENTION_IMPLEMENTATION = v1.ATTENTION_IMPLEMENTATION
CAP_EXPERIMENT = v1.CAP_EXPERIMENT
CAP_LAYERS = v1.CAP_LAYERS
EXPECTED_AXIS_SHA256 = v1.EXPECTED_AXIS_SHA256
EXPECTED_CAP_CONFIG_SHA256 = v1.EXPECTED_CAP_CONFIG_SHA256
LABELS = v1.LABELS
TRACE_SOURCES = v1.TRACE_SOURCES
FORWARD_CONDITIONS = v1.FORWARD_CONDITIONS
EXPECTED_PROMPTS = v1.EXPECTED_PROMPTS
EXPECTED_TRACES = v1.EXPECTED_TRACES
EXPECTED_FORMAL_CELLS = v1.EXPECTED_FORMAL_CELLS
EXPECTED_SMOKE_PROMPTS = v1.EXPECTED_SMOKE_PROMPTS
EXPECTED_SMOKE_CELLS = v1.EXPECTED_SMOKE_CELLS
BOOTSTRAP_DRAWS = v1.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = v1.BOOTSTRAP_SEED
REPLAY_TOLERANCE = v1.REPLAY_TOLERANCE

canonical_text_sha256 = v1.canonical_text_sha256
file_sha256 = v1.file_sha256
object_sha256 = v1.object_sha256
ordered_cells = v1.ordered_cells
read_json = v1.read_json
read_jsonl = v1.read_jsonl
require = v1.require
write_json = v1.write_json
geometry = v1.geometry


def replay_geometry(logits: Mapping[str, float], row: Mapping[str, Any]) -> dict[str, float]:
    return geometry(
        {
            "initial_answer": row["initial_answer"],
            "target_answer": row["target_answer"],
            "logit_diagnostics": {"logits": dict(logits)},
        }
    )


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right, strict=True)
    )
    denominator = (
        sum((x - mean_left) ** 2 for x in left)
        * sum((y - mean_right) ** 2 for y in right)
    ) ** 0.5
    return numerator / denominator if denominator else None


def v1_smoke_replay_diagnostic() -> dict[str, Any]:
    rows = read_jsonl(V1_SMOKE_RESULTS)
    diagonal = [row for row in rows if row.get("is_diagonal_replay")]
    require(len(diagonal) == 16, "Expected 16 v1 diagonal smoke cells")
    cells: list[dict[str, Any]] = []
    for row in diagonal:
        observed = row["logit_diagnostics"]["logits"]
        anchor = row["anchor_answer_logits"]["logits"]
        raw = {label: float(observed[label]) - float(anchor[label]) for label in LABELS}
        common_shift = statistics.fmean(raw.values())
        centered = {label: raw[label] - common_shift for label in LABELS}
        observed_geometry = replay_geometry(observed, row)
        anchor_geometry = replay_geometry(anchor, row)
        cells.append(
            {
                "cell_id": row["cell_id"],
                "condition": row["forward_condition"],
                "trace_prefix_tokens": row["trace_prefix_tokens"],
                "raw_logit_deltas": raw,
                "common_shift": common_shift,
                "raw_max_abs": max(abs(value) for value in raw.values()),
                "centered_max_abs": max(abs(value) for value in centered.values()),
                "metric_errors": {
                    metric: observed_geometry[metric] - anchor_geometry[metric]
                    for metric in observed_geometry
                },
            }
        )

    summary: dict[str, Any] = {}
    for condition in ("all", *FORWARD_CONDITIONS):
        selected = (
            cells if condition == "all" else [row for row in cells if row["condition"] == condition]
        )
        item: dict[str, Any] = {"cells": len(selected)}
        for metric in (
            "raw_max_abs",
            "centered_max_abs",
            "common_shift",
            "m_general",
            "m_target",
            "m_background",
            "m_specific",
        ):
            values = [
                float(row[metric])
                if metric in row
                else float(row["metric_errors"][metric])
                for row in selected
            ]
            item[metric] = {
                "mean": statistics.fmean(values),
                "mean_abs": statistics.fmean(abs(value) for value in values),
                "max_abs": max(abs(value) for value in values),
            }
        lengths = [float(row["trace_prefix_tokens"]) for row in selected]
        item["pearson_trace_length_vs_raw_max_abs"] = _correlation(
            lengths, [float(row["raw_max_abs"]) for row in selected]
        )
        item["pearson_trace_length_vs_abs_m_general_error"] = _correlation(
            lengths, [abs(float(row["metric_errors"]["m_general"])) for row in selected]
        )
        summary[condition] = item
    return {
        "schema": "cedar.e2b_fixed_trace_cached.v1_smoke_diagnostic.v1",
        "gpu_used": False,
        "source_results_canonical_lf_sha256": canonical_text_sha256(V1_SMOKE_RESULTS),
        "conclusion": "not_common_shift_only",
        "summary": summary,
        "cells": cells,
    }


def validate_v1_failure_evidence() -> dict[str, Any]:
    require(V1_FAILURE_AUDIT.is_file(), "v1 smoke audit missing")
    require(V1_FAILURE_REPORT.is_file(), "v1 smoke failure report missing")
    audit = read_json(V1_FAILURE_AUDIT)
    require(audit.get("status") == "failed", "v1 smoke audit is not failed")
    require(audit.get("counts", {}).get("rows") == 32, "v1 smoke count changed")
    replay = audit.get("replay_gate", {})
    require(replay.get("checked") == 16, "v1 replay count changed")
    require(len(replay.get("failures", [])) == 14, "v1 replay failure count changed")
    return {
        "audit_canonical_lf_sha256": canonical_text_sha256(V1_FAILURE_AUDIT),
        "failure_report_canonical_lf_sha256": canonical_text_sha256(V1_FAILURE_REPORT),
        "smoke_results_canonical_lf_sha256": canonical_text_sha256(V1_SMOKE_RESULTS),
        "rows": 32,
        "diagonal_checked": 16,
        "diagonal_failed": 14,
    }
