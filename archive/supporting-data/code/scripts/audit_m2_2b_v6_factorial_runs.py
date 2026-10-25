#!/usr/bin/env python3
"""Audit input identity and completeness within every v6 cap comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import file_sha256, object_sha256

ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "thinking_greedy",
    "thinking_sampling",
    "nonthinking_greedy",
    "nonthinking_sampling",
)
CAPS = ("uncapped", "p25", "p1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("formal", "axis"), default="formal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "m2" / "m2_2b_v6_factorial_diagnostic",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    args = parse_args()
    expected = 144 if args.dataset == "formal" else 24
    failures: list[str] = []
    missing: list[str] = []
    missing_cells: list[str] = []
    cells: dict[str, dict[str, Any]] = {}

    for mode in MODES:
        cells[mode] = {}
        for cap in CAPS:
            name = f"qwen3_32b_v6_{args.dataset}_{mode}_{cap}_seed{args.seed}_development"
            run_dir = args.output_root / name
            paths = {
                "manifest": run_dir / "manifest.json",
                "runtime": run_dir / "runtime.json",
                "results": run_dir / "results.jsonl",
            }
            absent = [str(path) for path in paths.values() if not path.exists()]
            if absent:
                missing.extend(absent)
                missing_cells.append(f"{mode}/{cap}")
                continue
            manifest = read_json(paths["manifest"])
            runtime = read_json(paths["runtime"])
            results = read_jsonl(paths["results"])
            configuration = manifest["configuration"]
            require(configuration.get("condition") == cap, f"{name}: condition", failures)
            require(configuration.get("mode", {}).get("name") == mode, f"{name}: mode", failures)
            require(len(results) == expected, f"{name}: results={len(results)}", failures)
            require(
                runtime.get("complete_results") == expected,
                f"{name}: runtime incomplete",
                failures,
            )
            require(
                runtime.get("results_sha256") == file_sha256(paths["results"]),
                f"{name}: results hash",
                failures,
            )
            require(
                [row.get("record_seed") for row in results]
                == [args.seed * 1_000_003 + index for index in range(expected)],
                f"{name}: per-record seeds",
                failures,
            )
            require(
                not runtime.get("mode_integrity_failures"),
                f"{name}: mode integrity failures",
                failures,
            )
            require(
                object_sha256([row.get("id") for row in results])
                == configuration.get("dataset", {}).get("row_ids_sha256"),
                f"{name}: result ID order",
                failures,
            )
            signature = {
                "dataset_sha256": configuration.get("dataset", {}).get("sha256"),
                "row_ids_sha256": configuration.get("dataset", {}).get("row_ids_sha256"),
                "mode": configuration.get("mode"),
                "model": configuration.get("model"),
                "generation": configuration.get("generation"),
                "rendered_prompts_sha256": runtime.get("rendered_prompts_sha256"),
                "tokenized_input_ids_sha256": runtime.get("tokenized_input_ids_sha256"),
                "padding_side": runtime.get("tokenizer_padding_side"),
                "pad_token_id": runtime.get("tokenizer_pad_token_id"),
            }
            cells[mode][cap] = {
                "run_dir": str(run_dir),
                "results": len(results),
                "results_sha256": file_sha256(paths["results"]),
                "signature": signature,
            }

        if "uncapped" in cells[mode]:
            reference = cells[mode]["uncapped"]["signature"]
            for cap in ("p25", "p1"):
                if cap not in cells[mode]:
                    continue
                require(
                    cells[mode][cap]["signature"] == reference,
                    f"{mode}/{cap}: inputs differ from uncapped",
                    failures,
                )

    if args.allow_incomplete:
        required = {
            f"{mode}/{cap}"
            for mode in ("thinking_sampling", "nonthinking_sampling")
            for cap in ("uncapped", "p25")
        }
        absent_required = sorted(required & set(missing_cells))
        if absent_required:
            failures.append(f"missing mandatory primary cells: {absent_required}")
    elif missing:
        failures.append(f"missing {len(missing)} run files")
    report = {
        "status": "passed" if not failures else "failed",
        "dataset": args.dataset,
        "seed": args.seed,
        "failures": failures,
        "missing": missing,
        "missing_cells": missing_cells,
        "cells": cells,
    }
    report_name = f"v6_cross_cell_audit_{args.dataset}_seed{args.seed}.json"
    output = args.report or args.output_root / report_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "report": str(output)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
