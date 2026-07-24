#!/usr/bin/env python3
"""Cross-track engineering audit for v1.1 uncapped template calibration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "prover_generator_v1_1_uncapped"
TRACKS = {
    "evidence": {"mode": "thinking", "arms": ("N", "R", "V", "F")},
    "social-candidates": {"mode": "nonthinking", "arms": ("N", "AO", "S1", "S2")},
}
EXPECTED = {"smoke": 16, "formal": 96}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(scope: str, track: str) -> str:
    return f"v1_1_template_calibration_{scope}_{track.replace('-', '_')}_uncapped_seed0"


def audit_cell(output_root: Path, scope: str, track: str) -> tuple[dict[str, Any], list[str], list[str]]:
    run_dir = output_root / run_name(scope, track)
    required = {name: run_dir / name for name in ("manifest.json", "results.jsonl", "record_runtime.jsonl", "runtime.json")}
    failures = [f"{track}: missing {name}" for name, path in required.items() if not path.is_file()]
    if failures:
        return {"run_dir": str(run_dir)}, failures, []
    manifest = read_json(required["manifest.json"])
    runtime = read_json(required["runtime.json"])
    rows = read_jsonl(required["results.jsonl"])
    config = manifest.get("configuration", {})
    expected = EXPECTED[scope]
    arms = TRACKS[track]["arms"]
    per_arm = expected // len(arms)
    checks = {
        "rows": len(rows) == expected,
        "unique_ids": len({row.get("id") for row in rows}) == expected,
        "arm_balance": Counter(row.get("arm") for row in rows) == Counter({arm: per_arm for arm in arms}),
        "uncapped": all(row.get("condition") == "uncapped" for row in rows),
        "track": all(row.get("track") == track for row in rows),
        "mode": all(row.get("mode") == TRACKS[track]["mode"] for row in rows),
        "calibration_split": all(row.get("development_split") == "template_calibration" for row in rows),
        "max_new_tokens_4096": config.get("generation", {}).get("max_new_tokens") == 4096,
        "no_retry_or_repair": all(row.get("content_retry_count") == 0 and row.get("answer_repair_performed") is False for row in rows),
        "runtime_complete": runtime.get("complete_results") == expected,
        "runtime_results_hash": runtime.get("results_sha256") == file_sha256(required["results.jsonl"]),
        "p25_prohibited": "p25" in config.get("prohibited_conditions", []) and config.get("condition") == "uncapped",
    }
    failures.extend(f"{track}: {name}" for name, passed in checks.items() if not passed)
    parse_failures = [row["id"] for row in rows if row.get("final_answer_parse_status") != "passed"]
    missing_logits = [row["id"] for row in rows if row.get("answer_logits") is None]
    truncated = [row["id"] for row in rows if row.get("truncated")]
    warnings = []
    if parse_failures:
        warnings.append(f"{track}: {len(parse_failures)} strict parse failures")
    if missing_logits:
        warnings.append(f"{track}: {len(missing_logits)} missing answer-logit captures")
    if truncated:
        warnings.append(f"{track}: {len(truncated)} truncated responses")
    return {
        "run_dir": str(run_dir),
        "checks": checks,
        "results": len(rows),
        "arm_counts": dict(sorted(Counter(row.get("arm") for row in rows).items())),
        "outcome_counts": dict(sorted(Counter(row.get("outcome_class") for row in rows).items())),
        "parse_failures": len(parse_failures),
        "missing_answer_logits": len(missing_logits),
        "truncated": len(truncated),
        "hashes": {name.removesuffix(".json").removesuffix(".jsonl"): file_sha256(path) for name, path in required.items()},
        "runner_commit": config.get("git", {}).get("commit"),
    }, failures, warnings


def audit(output_root: Path, scope: str) -> dict[str, Any]:
    cells = {}
    failures: list[str] = []
    warnings: list[str] = []
    for track in TRACKS:
        cell, cell_failures, cell_warnings = audit_cell(output_root, scope, track)
        cells[track] = cell
        failures.extend(cell_failures)
        warnings.extend(cell_warnings)
    result = {
        "schema": "cedar.prover_generator_development.v1_1.cross_track_audit.v1",
        "phase": "template-calibration",
        "scope": scope,
        "condition": "uncapped",
        "status": "passed" if not failures else "failed",
        "cells": cells,
        "failures": failures,
        "warnings": warnings,
        "p25_authorized": False,
    }
    name = "TEMPLATE_CALIBRATION_SMOKE_AUDIT.json" if scope == "smoke" else "TEMPLATE_CALIBRATION_FORMAL_AUDIT.json"
    (output_root / name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = audit(args.output_root.resolve(), args.scope)
    print({"status": result["status"], "scope": result["scope"], "failures": result["failures"], "warnings": result["warnings"]})


if __name__ == "__main__":
    main()
