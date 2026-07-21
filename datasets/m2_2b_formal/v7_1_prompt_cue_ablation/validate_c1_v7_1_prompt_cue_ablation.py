#!/usr/bin/env python3
"""Validate that V7.1 changes only the frozen epistemic-status cue."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_c1_v7_1_prompt_cue_ablation import (
    ARMS,
    CUE,
    MANIFEST,
    OUTPUT,
    SOURCE,
    VERSION,
    file_sha256,
    object_sha256,
    read_jsonl,
)


def comparable(row: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(row)
    for field in (
        "dataset_version",
        "id",
        "prompt_variant",
        "removed_text",
        "source_id",
        "source_row_sha256",
    ):
        value.pop(field, None)
    return value


def validate(source_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    source_rows = read_jsonl(source_path)
    rows = read_jsonl(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_by_id = {row["id"]: row for row in source_rows}
    failures: list[str] = []
    if len(rows) != 144:
        failures.append(f"row_count={len(rows)}")
    if Counter(row.get("arm") for row in rows) != Counter({arm: 48 for arm in ARMS}):
        failures.append("arm_balance")
    if len({row.get("base_id") for row in rows}) != 48:
        failures.append("base_count")
    if len({row.get("family_id") for row in rows}) != 24:
        failures.append("family_count")
    if len({row.get("id") for row in rows}) != len(rows):
        failures.append("duplicate_ids")
    for row in rows:
        source = source_by_id.get(row.get("source_id"))
        if source is None:
            failures.append(f"{row.get('id')}:missing_source")
            continue
        if source.get("arm") not in ARMS:
            failures.append(f"{row['id']}:source_arm")
        if row.get("dataset_version") != VERSION:
            failures.append(f"{row['id']}:version")
        if row.get("source_row_sha256") != object_sha256(source):
            failures.append(f"{row['id']}:source_hash")
        source_final = source["messages"][-1]["content"]
        row_final = row["messages"][-1]["content"]
        if source_final.count(CUE) != 1 or CUE.strip() in row_final:
            failures.append(f"{row['id']}:cue_count")
        expected = copy.deepcopy(source)
        expected["messages"][-1]["content"] = source_final.replace(CUE, "", 1)
        if comparable(row) != comparable(expected):
            failures.append(f"{row['id']}:nonminimal_change")
    if manifest.get("source_sha256") != file_sha256(source_path):
        failures.append("manifest_source_hash")
    if manifest.get("output_sha256") != file_sha256(output_path):
        failures.append("manifest_output_hash")
    if manifest.get("other_prompt_edits") != 0:
        failures.append("manifest_other_edits")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "rows": len(rows),
        "arms": Counter(row.get("arm") for row in rows),
        "bases": len({row.get("base_id") for row in rows}),
        "families": len({row.get("family_id") for row in rows}),
        "source_sha256": file_sha256(source_path),
        "output_sha256": file_sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    report = validate(args.source, args.data, args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
