#!/usr/bin/env python3
# ruff: noqa: E402
"""Engineering audit for E2 new-salt smoke and formal runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "e2_new_salt_confirmation"
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from e2_new_salt_run_common import (
    ARMS,
    CAP_LAYERS,
    CONDITIONS,
    FORMAL_RESULT_ROWS,
    MAX_NEW_TOKENS,
    MODEL_REPO,
    MODEL_REVISION,
    SMOKE_RESULT_ROWS,
    file_sha256,
    ordered_run_cells,
    read_json,
    select_rows,
    write_json,
)

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "e2_new_salt_confirmation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def logits_finite(row: Mapping[str, Any]) -> bool:
    payload = row.get("answer_logits")
    if payload is None:
        return True
    values = payload.get("logits", {})
    return set(values) == {"A", "B", "C", "D"} and all(
        math.isfinite(float(value)) for value in values.values()
    )


def audit(scope: str, output_root: Path) -> dict[str, Any]:
    run_dir = output_root / f"{scope}-paired-uncapped-full-p25-seed0"
    required = {
        name: run_dir / name
        for name in ("manifest.json", "results.jsonl", "record_runtime.jsonl", "runtime.json")
    }
    failures = [f"missing:{name}" for name, path in required.items() if not path.is_file()]
    output_name = "SMOKE_AUDIT.json" if scope == "smoke" else "FORMAL_AUDIT.json"
    if failures:
        result = {
            "schema": "cedar.e2_new_salt_confirmation.audit.v1",
            "scope": scope,
            "status": "failed",
            "failures": failures,
            "warnings": [],
        }
        write_json(output_root / output_name, result)
        return result

    rows = read_jsonl(required["results.jsonl"])
    runtime = read_json(required["runtime.json"])
    manifest = read_json(required["manifest.json"])
    config = manifest.get("configuration", {})
    source = select_rows(scope)
    expected_cells = ordered_run_cells(source)
    expected = [(row["id"], condition) for row, condition in expected_cells]
    observed = [(row.get("id"), row.get("condition")) for row in rows]
    by_key = {(row.get("id"), row.get("condition")): row for row in rows}
    paired = []
    for source_row in source:
        uncapped = by_key.get((source_row["id"], "uncapped"))
        p25 = by_key.get((source_row["id"], "full_p25"))
        paired.append(
            bool(
                uncapped
                and p25
                and uncapped.get("prompt_sha256") == p25.get("prompt_sha256")
                and uncapped.get("prompt_token_ids_sha256")
                == p25.get("prompt_token_ids_sha256")
                and uncapped.get("record_seed") == p25.get("record_seed")
                and uncapped.get("initial_answer") == p25.get("initial_answer")
                and uncapped.get("target_answer") == p25.get("target_answer")
                and uncapped.get("gold_answer") == p25.get("gold_answer")
            )
        )
    p25_rows = [row for row in rows if row.get("condition") == "full_p25"]
    uncapped_rows = [row for row in rows if row.get("condition") == "uncapped"]
    expected_rows = SMOKE_RESULT_ROWS if scope == "smoke" else FORMAL_RESULT_ROWS
    per_arm_condition = expected_rows // (len(ARMS) * len(CONDITIONS))
    truncations = [
        f"{row['id']}::{row['condition']}" for row in rows if row.get("truncated")
    ]
    parse_failures = [
        f"{row['id']}::{row['condition']}"
        for row in rows
        if row.get("final_answer_parse_status") != "passed"
    ]
    missing_logits = [
        f"{row['id']}::{row['condition']}"
        for row in rows
        if row.get("answer_logits") is None
    ]
    checks = {
        "complete_rows": len(rows) == expected_rows,
        "unique_cells": len(set(observed)) == expected_rows,
        "frozen_paired_order": observed == expected,
        "condition_balance": Counter(row.get("condition") for row in rows)
        == Counter({condition: expected_rows // 2 for condition in CONDITIONS}),
        "arm_condition_balance": Counter(
            (row.get("arm"), row.get("condition")) for row in rows
        )
        == Counter(
            {
                (arm, condition): per_arm_condition
                for arm in ARMS
                for condition in CONDITIONS
            }
        ),
        "thinking_only": all(
            row.get("mode") == "thinking" and row.get("thinking_enabled") is True
            for row in rows
        ),
        "paired_prompt_identity": len(paired) == len(source) and all(paired),
        "uncapped_has_no_cap_stats": all(row.get("cap_stats") is None for row in uncapped_rows),
        "p25_has_complete_cap_stats": len(p25_rows) == expected_rows // 2
        and all(
            row.get("cap_stats") is not None
            and tuple(sorted(int(layer) for layer in row["cap_stats"]["by_layer"]))
            == CAP_LAYERS
            for row in p25_rows
        ),
        "p25_intervention_triggered": runtime.get("full_p25_triggered_positions", 0) > 0,
        "finite_logits_when_present": all(logits_finite(row) for row in rows),
        "no_hook_leaks": all(
            row.get("hook_counts_before") == row.get("hook_counts_after") for row in rows
        ),
        "strict_itt_truncation_classification": all(
            not row.get("truncated")
            or (
                row.get("outcome_class") == "invalid_or_abstain"
                and row.get("chosen_option") is None
            )
            for row in rows
        ),
        "no_retry_or_repair": all(
            row.get("content_retry_count") == 0
            and row.get("answer_repair_performed") is False
            for row in rows
        ),
        "runtime_complete": runtime.get("complete_results") == expected_rows,
        "runtime_results_hash": runtime.get("results_sha256")
        == file_sha256(required["results.jsonl"]),
        "runtime_record_hash": runtime.get("record_runtime_sha256")
        == file_sha256(required["record_runtime.jsonl"]),
        "single_matrix_completion": runtime.get(
            "single_matrix_completed_without_content_selection"
        )
        is True,
        "scope_frozen": config.get("scope") == scope
        and config.get("conditions") == list(CONDITIONS),
        "generation_frozen": config.get("generation", {}).get("max_new_tokens")
        == MAX_NEW_TOKENS
        and config.get("generation", {}).get("do_sample") is False
        and config.get("generation", {}).get("num_beams") == 1,
        "model_frozen": config.get("model", {}).get("repo_id") == MODEL_REPO
        and config.get("model", {}).get("revision") == MODEL_REVISION,
        "prohibited_extensions": all(
            token in config.get("prohibited", [])
            for token in ("p1", "additional_salt", "additional_seed", "patching")
        ),
    }
    if scope == "smoke":
        checks.update(
            {
                "smoke_no_truncations": not truncations,
                "smoke_no_parse_failures": not parse_failures,
                "smoke_no_missing_logits": not missing_logits,
            }
        )
    failures.extend(name for name, passed in checks.items() if not passed)
    warnings = []
    if truncations:
        warnings.append(f"{len(truncations)} truncations retained as ITT invalid_or_abstain")
    if parse_failures:
        warnings.append(f"{len(parse_failures)} strict parse failures retained")
    if missing_logits:
        warnings.append(f"{len(missing_logits)} missing answer-logit captures")
    result = {
        "schema": "cedar.e2_new_salt_confirmation.audit.v1",
        "scope": scope,
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "counts": {
            "rows": len(rows),
            "conditions": dict(sorted(Counter(row.get("condition") for row in rows).items())),
            "arms_by_condition": {
                f"{arm}/{condition}": sum(
                    row.get("arm") == arm and row.get("condition") == condition
                    for row in rows
                )
                for arm in ARMS
                for condition in CONDITIONS
            },
            "truncations": truncations,
            "parse_failures": parse_failures,
            "missing_logits": missing_logits,
            "hook_leaks": [
                f"{row['id']}::{row['condition']}"
                for row in rows
                if row.get("hook_counts_before") != row.get("hook_counts_after")
            ],
        },
        "hashes": {
            "results": file_sha256(required["results.jsonl"]),
            "record_runtime": file_sha256(required["record_runtime.jsonl"]),
        },
    }
    write_json(output_root / output_name, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = audit(args.scope, args.output_root.resolve())
    print(
        json.dumps(
            {
                "scope": args.scope,
                "status": result["status"],
                "failures": result["failures"],
                "warnings": result["warnings"],
                "counts": result.get("counts"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
