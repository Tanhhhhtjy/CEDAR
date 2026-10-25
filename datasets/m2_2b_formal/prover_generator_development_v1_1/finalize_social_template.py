#!/usr/bin/env python3
"""Freeze one globally selected social template after template calibration."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

from development_v1_1_common import (
    FINAL_ARMS,
    HERE,
    SCHEMA,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    write_json,
    write_jsonl,
)


def read_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def finalize(root: Path, calibration_audit: Path, output_root: Path) -> dict[str, Any]:
    audit = read_json(calibration_audit)
    selected = audit.get("selected_social_template")
    if audit.get("status") != "passed" or selected not in {"S1", "S2"}:
        raise ValueError("A passed calibration audit with one global S1/S2 selection is required")
    if audit.get("itemwise_template_selection") is not False:
        raise ValueError("Itemwise social-template selection is prohibited")
    if audit.get("selection_rule") != "weakest_globally_passing_template_s1_wins_ties":
        raise ValueError("Calibration audit does not implement the frozen global selection rule")
    candidates = read_jsonl(root / "development_candidates336.jsonl")
    release: list[dict[str, Any]] = []
    for row in candidates:
        if row["arm"] in {"S1", "S2"} and row["arm"] != selected:
            continue
        item = copy.deepcopy(row)
        if item["arm"] == selected:
            item["arm"] = "S"
            item["id"] = f"{item['base_id']}-S"
            item["social_template"] = selected
            item["prompt_canonical_sha256"] = canonical_sha256(item["messages"])
        release.append(item)
    counts = Counter(row["arm"] for row in release)
    if len(release) != 288 or counts != Counter({arm: 48 for arm in FINAL_ARMS}):
        raise RuntimeError(f"Final release matrix is invalid: {counts}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "development_arms288.jsonl", release)
    manifest = {
        "schema": SCHEMA,
        "status": "social_template_frozen",
        "selected_social_template": selected,
        "selection_audit_path": str(calibration_audit.resolve()),
        "selection_audit_raw_sha256": file_sha256(calibration_audit),
        "itemwise_template_selection": False,
        "rows": len(release),
        "arm_counts": dict(sorted(counts.items())),
        "arms_object_sha256": canonical_sha256(release),
        "ids_object_sha256": canonical_sha256([row["id"] for row in release]),
        "p25_authorized": False,
        "held_out_confirmation": False,
    }
    write_json(output_root / "release_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--calibration-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "release")
    args = parser.parse_args()
    result = finalize(args.root.resolve(), args.calibration_audit.resolve(), args.output_root.resolve())
    print(result)


if __name__ == "__main__":
    main()
