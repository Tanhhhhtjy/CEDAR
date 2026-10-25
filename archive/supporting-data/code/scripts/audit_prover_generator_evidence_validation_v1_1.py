#!/usr/bin/env python3
"""Engineering audit for the 192-cell paired Evidence validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_evidence_validation_v1_1"
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from evidence_validation_common import (
    CAP_LAYERS,
    CONDITIONS,
    EVIDENCE_ARMS,
    EXPECTED_RESULT_ROWS,
    MODEL_REPO,
    MODEL_REVISION,
    file_sha256,
    ordered_run_cells,
    read_json,
    select_validation_rows,
)

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "prover_generator_evidence_validation_v1_1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(output_root: Path) -> dict[str, Any]:
    run_dir = output_root / "formal-paired-uncapped-p25-seed0"
    required = {name: run_dir / name for name in ("manifest.json", "results.jsonl", "record_runtime.jsonl", "runtime.json")}
    failures = [f"missing {name}" for name, path in required.items() if not path.is_file()]
    if failures:
        result = {"schema": "cedar.prover_generator.evidence_validation_v1_1.audit.v1", "status": "failed", "failures": failures, "warnings": []}
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "FORMAL_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return result

    rows = read_jsonl(required["results.jsonl"])
    manifest = read_json(required["manifest.json"])
    runtime = read_json(required["runtime.json"])
    config = manifest.get("configuration", {})
    source = select_validation_rows()
    expected = [(row["id"], condition) for row, condition in ordered_run_cells(source)]
    observed = [(row.get("id"), row.get("condition")) for row in rows]
    by_key = {(row.get("id"), row.get("condition")): row for row in rows}
    paired_checks = []
    for source_row in source:
        uncapped = by_key.get((source_row["id"], "uncapped"))
        p25 = by_key.get((source_row["id"], "p25"))
        paired_checks.append(
            bool(
                uncapped
                and p25
                and uncapped.get("prompt_sha256") == p25.get("prompt_sha256")
                and uncapped.get("prompt_token_ids_sha256") == p25.get("prompt_token_ids_sha256")
                and uncapped.get("record_seed") == p25.get("record_seed")
                and uncapped.get("initial_answer") == p25.get("initial_answer")
                and uncapped.get("target_answer") == p25.get("target_answer")
                and uncapped.get("gold_answer") == p25.get("gold_answer")
            )
        )
    p25_rows = [row for row in rows if row.get("condition") == "p25"]
    uncapped_rows = [row for row in rows if row.get("condition") == "uncapped"]
    p25_layer_checks = [
        row.get("cap_stats") is not None
        and tuple(sorted(int(layer) for layer in row["cap_stats"].get("by_layer", {}))) == CAP_LAYERS
        for row in p25_rows
    ]
    checks = {
        "rows_192": len(rows) == EXPECTED_RESULT_ROWS,
        "unique_cells": len(set(observed)) == EXPECTED_RESULT_ROWS,
        "frozen_paired_order": observed == expected,
        "condition_balance": Counter(row.get("condition") for row in rows) == Counter({condition: 96 for condition in CONDITIONS}),
        "arm_condition_balance": Counter((row.get("arm"), row.get("condition")) for row in rows) == Counter({(arm, condition): 24 for arm in EVIDENCE_ARMS for condition in CONDITIONS}),
        "validation_split_only": all(row.get("development_split") == "development_validation" for row in rows),
        "thinking_only": all(row.get("mode") == "thinking" and row.get("thinking_enabled") is True for row in rows),
        "uncapped_then_p25_pair_identity": all(paired_checks) and len(paired_checks) == 96,
        "uncapped_has_no_cap_stats": all(row.get("cap_stats") is None for row in uncapped_rows),
        "p25_has_complete_cap_stats": len(p25_layer_checks) == 96 and all(p25_layer_checks),
        "p25_intervention_triggered": runtime.get("p25_triggered_positions", 0) > 0,
        "no_hook_leaks": all(row.get("hook_counts_before") == row.get("hook_counts_after") for row in rows),
        "strict_itt_truncation_classification": all(
            not row.get("truncated")
            or (row.get("outcome_class") == "invalid_or_abstain" and row.get("chosen_option") is None)
            for row in rows
        ),
        "no_retry_or_repair": all(row.get("content_retry_count") == 0 and row.get("answer_repair_performed") is False for row in rows),
        "runtime_complete": runtime.get("complete_results") == EXPECTED_RESULT_ROWS,
        "runtime_results_hash": runtime.get("results_sha256") == file_sha256(required["results.jsonl"]),
        "runtime_record_hash": runtime.get("record_runtime_sha256") == file_sha256(required["record_runtime.jsonl"]),
        "single_matrix_completion": runtime.get("single_matrix_completed_without_content_selection") is True,
        "scope_authorization": config.get("scope") == "evidence_only_internal_validation" and config.get("conditions") == list(CONDITIONS),
        "generation_frozen": config.get("generation", {}).get("max_new_tokens") == 4096 and config.get("generation", {}).get("do_sample") is False and config.get("generation", {}).get("num_beams") == 1,
        "model_frozen": config.get("model", {}).get("repo_id") == MODEL_REPO and config.get("model", {}).get("revision") == MODEL_REVISION,
        "p1_and_social_prohibited": "p1" in config.get("prohibited", []) and "AO" in config.get("prohibited", []) and "S" in config.get("prohibited", []),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    warnings = []
    truncations = [f"{row['id']}::{row['condition']}" for row in rows if row.get("truncated")]
    parse_failures = [f"{row['id']}::{row['condition']}" for row in rows if row.get("final_answer_parse_status") != "passed"]
    missing_logits = [f"{row['id']}::{row['condition']}" for row in rows if row.get("answer_logits") is None]
    if truncations:
        warnings.append(f"{len(truncations)} truncations retained as ITT invalid_or_abstain")
    if parse_failures:
        warnings.append(f"{len(parse_failures)} strict parse failures")
    if missing_logits:
        warnings.append(f"{len(missing_logits)} missing answer-logit captures")
    result = {
        "schema": "cedar.prover_generator.evidence_validation_v1_1.audit.v1",
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "counts": {
            "rows": len(rows),
            "conditions": dict(sorted(Counter(row.get("condition") for row in rows).items())),
            "arms_by_condition": {f"{arm}/{condition}": sum(row.get("arm") == arm and row.get("condition") == condition for row in rows) for arm in EVIDENCE_ARMS for condition in CONDITIONS},
            "truncations": truncations,
            "parse_failures": parse_failures,
            "missing_logits": missing_logits,
        },
        "hashes": {"results": file_sha256(required["results.jsonl"]), "record_runtime": file_sha256(required["record_runtime.jsonl"])},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "FORMAL_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = audit(args.output_root.resolve())
    print({"status": result["status"], "failures": result["failures"], "warnings": result["warnings"], "counts": result.get("counts")})


if __name__ == "__main__":
    main()
