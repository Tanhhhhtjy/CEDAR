#!/usr/bin/env python3
"""Audit cross-condition input and batching identity for the M2.2b v5 scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import atomic_write_json, file_sha256, object_sha256

CONDITIONS = ("uncapped", "p75", "p50", "p25", "p1")
EXPECTED_ROWS = 468
EXPECTED_BATCHES = EXPECTED_ROWS // 2
RUN_NAMES = {
    condition: f"qwen3_32b_v5_dose_bank_{condition}_thinking_greedy_seed0_development"
    for condition in CONDITIONS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/m2/m2_2b_v5_dose_response_bank"),
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    signatures: dict[str, dict[str, Any]] = {}
    run_reports: dict[str, dict[str, Any]] = {}

    for condition in CONDITIONS:
        run_dir = args.output_root / RUN_NAMES[condition]
        manifest_path = run_dir / "manifest.json"
        runtime_path = run_dir / "runtime.json"
        results_path = run_dir / "qwen_thinking_results.jsonl"
        batches_path = run_dir / "batch_runtime.jsonl"
        for path in (manifest_path, runtime_path, results_path, batches_path):
            require(path.exists(), f"{condition}: missing {path}", failures)
        required_paths = (manifest_path, runtime_path, results_path, batches_path)
        if not all(path.exists() for path in required_paths):
            continue

        manifest = load_json(manifest_path)
        runtime = load_json(runtime_path)
        configuration = manifest.get("configuration", {})
        dataset = configuration.get("dataset", {})
        generation = configuration.get("generation", {})
        results = load_jsonl(results_path)
        batches = load_jsonl(batches_path)

        require(
            configuration.get("condition") == condition,
            f"{condition}: manifest label",
            failures,
        )
        require(len(results) == EXPECTED_ROWS, f"{condition}: results={len(results)}", failures)
        require(len(batches) == EXPECTED_BATCHES, f"{condition}: batches={len(batches)}", failures)
        require(
            [row.get("cap") for row in results] == [condition] * EXPECTED_ROWS,
            f"{condition}: result condition labels",
            failures,
        )
        require(
            runtime.get("complete_results") == EXPECTED_ROWS,
            f"{condition}: incomplete runtime",
            failures,
        )
        require(
            runtime.get("tokenizer_padding_side") == "left"
            and generation.get("padding_side") == "left",
            f"{condition}: padding is not frozen left",
            failures,
        )
        require(
            runtime.get("results_sha256") == file_sha256(results_path),
            f"{condition}: result hash mismatch",
            failures,
        )

        batch_layout = [
            {
                "ids": row.get("ids"),
                "prompt_padded_width": row.get("prompt_padded_width"),
                "prompt_nonpadding_tokens": row.get("prompt_nonpadding_tokens"),
            }
            for row in batches
        ]
        flattened_ids = [item for batch in batch_layout for item in (batch["ids"] or [])]
        require(
            flattened_ids == [row.get("id") for row in results],
            f"{condition}: batch/result order mismatch",
            failures,
        )
        signatures[condition] = {
            "dataset_sha256": dataset.get("sha256"),
            "row_ids_sha256": dataset.get("row_ids_sha256"),
            "batch_membership_sha256_manifest": dataset.get("batch_membership_sha256"),
            "batch_membership_sha256_runtime": runtime.get("batch_membership_sha256"),
            "rendered_prompts_sha256": runtime.get("rendered_prompts_sha256"),
            "tokenized_input_ids_sha256": runtime.get("tokenized_input_ids_sha256"),
            "batch_layout_sha256": object_sha256(batch_layout),
            "tokenizer_padding_side": runtime.get("tokenizer_padding_side"),
            "tokenizer_pad_token_id": runtime.get("tokenizer_pad_token_id"),
            "model": configuration.get("model"),
            "messages": configuration.get("messages"),
            "generation": generation,
        }
        require(
            signatures[condition]["batch_membership_sha256_manifest"]
            == signatures[condition]["batch_membership_sha256_runtime"],
            f"{condition}: runtime batch membership changed",
            failures,
        )
        if condition != "uncapped":
            require(
                bool(runtime.get("cap_trigger_stats")),
                f"{condition}: missing cap stats",
                failures,
            )
        run_reports[condition] = {
            "run_dir": str(run_dir),
            "results_sha256": file_sha256(results_path),
            "complete_results": len(results),
            "complete_batches": len(batches),
            "input_signature": signatures[condition],
        }

    if len(signatures) == len(CONDITIONS):
        baseline = signatures["uncapped"]
        for condition in CONDITIONS[1:]:
            require(
                signatures[condition] == baseline,
                f"{condition}: tokenization, padding, batch layout, or frozen config differs",
                failures,
            )

    report = {
        "status": "passed" if not failures else "failed",
        "scope": "cross-condition input identity; responses and cap exposure may differ",
        "conditions": list(CONDITIONS),
        "expected_rows_per_condition": EXPECTED_ROWS,
        "left_padding_required": True,
        "runs": run_reports,
        "failures": failures,
    }
    report_path = args.report or args.output_root / "cross_condition_input_audit.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
