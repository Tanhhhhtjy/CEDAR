#!/usr/bin/env python3
"""Engineering and anchor-equivalence audit for E2A results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "m2_2b_formal" / "e2_phase_validity_choice"
if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from e2_common import (  # noqa: E402
    ANCHOR_RESULTS,
    FORMAL_CONDITIONS,
    read_json,
    read_jsonl,
    write_json,
)

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "e2_phase_validity_choice"


def _close(left: float, right: float, tolerance: float = 1e-4) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def anchor_equivalence(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    anchor = {(row["id"], row["condition"]): row for row in read_jsonl(ANCHOR_RESULTS)}
    failures: list[str] = []
    checked = 0
    for row in rows:
        if row.get("assay") != "answer" or row.get("condition") not in {
            "uncapped",
            "full_p25",
        }:
            continue
        anchor_condition = "p25" if row["condition"] == "full_p25" else "uncapped"
        old = anchor.get((row["id"], anchor_condition))
        if old is None:
            failures.append(f"anchor_missing:{row['id']}::{anchor_condition}")
            continue
        checked += 1
        exact_fields = (
            "prompt_sha256",
            "prompt_token_ids_sha256",
            "response_token_ids_sha256",
            "chosen_option",
            "outcome_class",
            "final_answer_parse_status",
            "truncated",
            "selected_answer_logit_index",
        )
        for field in exact_fields:
            if row.get(field) != old.get(field):
                failures.append(f"anchor_field:{row['id']}::{row['condition']}::{field}")
        new_logits = (row.get("answer_logits") or {}).get("logits", {})
        old_logits = (old.get("answer_logits") or {}).get("logits", {})
        if set(new_logits) != set(old_logits) or any(
            not _close(float(new_logits[label]), float(old_logits[label])) for label in new_logits
        ):
            failures.append(f"anchor_logits:{row['id']}::{row['condition']}")
    return {
        "checked": checked,
        "expected": 192,
        "passed": checked == 192 and not failures,
    }, failures


def audit(output_root: Path, *, scope: str, assay: str) -> dict[str, Any]:
    run_dir = output_root / f"{scope}-{assay}-seed0"
    results_path = run_dir / "results.jsonl"
    runtime_path = run_dir / "runtime.json"
    failures: list[str] = []
    warnings: list[str] = []
    rows = read_jsonl(results_path) if results_path.is_file() else []
    runtime = read_json(runtime_path) if runtime_path.is_file() else {}
    expected = 576 if scope == "formal" and assay == "all" else None
    if expected is not None and len(rows) != expected:
        failures.append(f"row_count:{len(rows)}!={expected}")
    keys = [(row.get("assay"), row.get("id"), row.get("condition")) for row in rows]
    if len(keys) != len(set(keys)):
        failures.append("duplicate_cells")
    if any(row.get("content_retry_count") != 0 for row in rows):
        failures.append("content_retry_present")
    if any(row.get("answer_repair_performed") is not False for row in rows):
        failures.append("answer_repair_present")
    if any(row.get("hook_counts_before") != row.get("hook_counts_after") for row in rows):
        failures.append("hook_leak")

    for row in rows:
        condition = row.get("condition")
        phase = row.get("phase_telemetry")
        full = row.get("full_cap_stats")
        prefix = f"{row.get('assay')}::{row.get('id')}::{condition}"
        if condition in {"uncapped", "nohook"}:
            if phase is not None or full is not None:
                failures.append(f"unexpected_hook:{prefix}")
            continue
        if condition == "full_p25":
            if phase is None or full is None:
                failures.append(f"missing_full_telemetry:{prefix}")
                continue
            if phase.get("apply_changes") is not False or phase.get("enabled_phases") != [
                "decode",
                "prefill",
            ]:
                failures.append(f"full_monitor_mode:{prefix}")
            for layer, full_layer in full.get("by_layer", {}).items():
                phase_layer = phase.get("by_layer", {}).get(layer, {})
                prefill = phase_layer.get("prefill", {})
                decode = phase_layer.get("decode", {})
                monitored_total = prefill.get("applied_positions", 0) + decode.get(
                    "applied_positions", 0
                )
                monitored_triggered = prefill.get("triggered_positions", 0) + decode.get(
                    "triggered_positions", 0
                )
                monitored_correction = prefill.get("projection_correction_sum", 0.0) + decode.get(
                    "projection_correction_sum", 0.0
                )
                if monitored_total != full_layer.get("total_positions"):
                    failures.append(f"full_phase_total_mismatch:{prefix}::L{layer}")
                if monitored_triggered != full_layer.get("triggered_positions"):
                    failures.append(f"full_phase_trigger_mismatch:{prefix}::L{layer}")
                if not _close(
                    float(monitored_correction),
                    float(full_layer.get("projection_correction_sum", math.nan)),
                ):
                    failures.append(f"full_phase_correction_mismatch:{prefix}::L{layer}")
        elif condition == "monitor_only":
            if phase is None or full is not None or phase.get("apply_changes") is not False:
                failures.append(f"monitor_mode:{prefix}")
        elif condition in {"prefill_only", "decode_only"}:
            expected_phase = "prefill" if condition == "prefill_only" else "decode"
            other_phase = "decode" if expected_phase == "prefill" else "prefill"
            if phase is None or full is not None:
                failures.append(f"phase_telemetry_missing:{prefix}")
                continue
            if phase.get("apply_changes") is not True or phase.get("enabled_phases") != [
                expected_phase
            ]:
                failures.append(f"phase_mode:{prefix}")
            for layer, stats in phase.get("by_layer", {}).items():
                if stats[expected_phase]["applied_positions"] <= 0:
                    failures.append(f"phase_not_applied:{prefix}::L{layer}")
                if stats[other_phase]["applied_positions"] != 0:
                    failures.append(f"wrong_phase_applied:{prefix}::L{layer}")
                if stats[other_phase]["observed_positions"] <= 0:
                    failures.append(f"other_phase_not_observed:{prefix}::L{layer}")

    anchor = {"checked": 0, "expected": 0, "passed": scope != "formal"}
    if scope == "formal" and assay == "all" and len(rows) == 576:
        anchor, anchor_failures = anchor_equivalence(rows)
        failures.extend(anchor_failures)
    condition_counts = Counter(row.get("condition") for row in rows)
    if scope == "formal" and assay == "all":
        expected_conditions = Counter({condition: 144 for condition in FORMAL_CONDITIONS})
        if condition_counts != expected_conditions:
            failures.append("formal_condition_balance")
        if Counter(row.get("assay") for row in rows) != Counter({"answer": 384, "validity": 192}):
            failures.append("formal_assay_balance")
    result = {
        "schema": "cedar.e2_phase_validity_choice.audit.v1",
        "status": "passed" if not failures else "failed",
        "scope": scope,
        "assay": assay,
        "counts": {
            "rows": len(rows),
            "conditions": dict(sorted(condition_counts.items())),
            "assays": dict(sorted(Counter(row.get("assay") for row in rows).items())),
            "truncations": list(runtime.get("truncations", [])),
            "parse_failures": list(runtime.get("parse_failures", [])),
            "missing_logits": list(runtime.get("missing_logits", [])),
        },
        "anchor_equivalence": anchor,
        "failures": failures,
        "warnings": warnings,
    }
    write_json(
        output_root / ("E2A_AUDIT.json" if scope == "formal" else "SMOKE_AUDIT.json"), result
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--assay", choices=("answer", "validity", "all"), default="all")
    args = parser.parse_args()
    result = audit(args.output_root, scope=args.scope, assay=args.assay)
    print(
        json.dumps(
            {"status": result["status"], "failures": len(result["failures"])}, sort_keys=True
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
