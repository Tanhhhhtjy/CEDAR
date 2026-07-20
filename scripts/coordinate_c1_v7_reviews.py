#!/usr/bin/env python3
"""Coordinate C1/V7 blind reviews without invoking a model or changing labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "datasets" / "m2_2b_formal" / "v7_heldout_confirmation"
_VALIDATOR: ModuleType | None = None


def load_validator() -> ModuleType:
    global _VALIDATOR
    if _VALIDATOR is not None:
        return _VALIDATOR
    path = DEFAULT_DATA_DIR / "validate_m2_2b_v7_heldout.py"
    spec = importlib.util.spec_from_file_location("c1_v7_review_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _VALIDATOR = module
    return _VALIDATOR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite state file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_jsonl_exclusive(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite private labels: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def review_paths(data_dir: Path) -> dict[str, Path]:
    private = data_dir / "private"
    return {
        "bases": data_dir / "m2_2b_v7_heldout48_base.jsonl",
        "review_a": data_dir / "blind_review_a.csv",
        "review_b": data_dir / "blind_review_b.csv",
        "key": private / "blind_review_key.jsonl",
        "packet_manifest": data_dir / "review_packet_manifest.json",
        "completed_manifest": data_dir / "completed_review_manifest.json",
        "agreement": data_dir / "review_agreement_report.json",
        "disputes": private / "adjudication_disputes.csv",
        "final_labels": private / "final_review_labels.jsonl",
        "adjudication_completed": private / "adjudication_completed.json",
        "adjudication_manifest": data_dir / "adjudication_manifest.json",
        "finalized_manifest": data_dir / "finalized_review_manifest.json",
    }


def load_context(
    data_dir: Path, *, require_complete: bool
) -> tuple[ModuleType, list[dict[str, Any]]]:
    validator = load_validator()
    paths = review_paths(data_dir)
    bases = validator.load_jsonl(paths["bases"])
    validator.validate_review_key_binding(paths["key"], paths["bases"], bases)
    for reviewer in ("review_a", "review_b"):
        validator.validate_review_sheet(paths[reviewer], require_complete=require_complete)
        validator.validate_review_packet_content(paths[reviewer], paths["key"], bases)
    tracked = validator.tracked_private_files(data_dir)
    if tracked:
        raise RuntimeError(f"private review files are Git-tracked: {tracked}")
    return validator, bases


def snapshot_packet(data_dir: Path) -> dict[str, Any]:
    validator, _ = load_context(data_dir, require_complete=False)
    paths = review_paths(data_dir)
    for reviewer in ("review_a", "review_b"):
        report = validator.validate_review_sheet(paths[reviewer], require_complete=False)
        if report["status"] != "blank":
            raise RuntimeError("review packet snapshot requires two blank review sheets")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet = {
        "state": "blank_review_packet",
        "source_manifest_sha256": sha256(manifest_path),
        "base_sha256": sha256(paths["bases"]),
        "review_a_sha256": sha256(paths["review_a"]),
        "review_b_sha256": sha256(paths["review_b"]),
        "key_sha256_private_not_committed": sha256(paths["key"]),
        "review_rows_each": 48,
        "labels_blank": True,
        "qwen_or_gpu_execution": False,
        "source_manifest": manifest,
    }
    write_json_exclusive(paths["packet_manifest"], packet)
    return packet


def score_only(
    data_dir: Path,
    *,
    received_a_at: str,
    received_b_at: str,
) -> dict[str, Any]:
    validator, bases = load_context(data_dir, require_complete=True)
    paths = review_paths(data_dir)
    if not paths["packet_manifest"].is_file():
        raise RuntimeError("blank review_packet_manifest.json is missing")
    result = validator.score_independent_reviews(
        paths["review_a"], paths["review_b"], paths["key"], bases
    )
    completed = {
        "state": "raw_completed_reviews_before_adjudication",
        "received_a_at": received_a_at,
        "received_b_at": received_b_at,
        "review_a_sha256": sha256(paths["review_a"]),
        "review_b_sha256": sha256(paths["review_b"]),
        "base_sha256": sha256(paths["bases"]),
        "packet_manifest_sha256": sha256(paths["packet_manifest"]),
        "reviewers_independent_attestation_required": True,
        "raw_labels_modified_by_coordinator": False,
        "qwen_or_gpu_execution": False,
    }
    report = {
        **result,
        "state": "scored_raw_reviews_no_adjudication",
        "completed_review_manifest_sha256_pending_write": True,
        "qwen_or_gpu_execution": False,
    }
    write_json_exclusive(paths["completed_manifest"], completed)
    report["completed_review_manifest_sha256_pending_write"] = False
    report["completed_review_manifest_sha256"] = sha256(paths["completed_manifest"])
    write_json_exclusive(paths["agreement"], report)
    return report


def reviewed_fields(validator: ModuleType) -> tuple[str, ...]:
    return (
        "original_supported_answer",
        "v_supported_answer",
        *validator.BOOLEAN_REVIEW_FIELDS,
    )


def prepare_adjudication(data_dir: Path) -> dict[str, Any]:
    validator, bases = load_context(data_dir, require_complete=True)
    paths = review_paths(data_dir)
    if not paths["agreement"].is_file():
        raise RuntimeError("run --score-only before preparing adjudication")
    agreement = json.loads(paths["agreement"].read_text(encoding="utf-8"))
    if agreement.get("status") != "passed":
        raise RuntimeError("raw agreement gate failed; adjudication cannot repair it")
    left = {row["blind_item_id"]: row for row in validator.read_review_sheet(paths["review_a"])}
    right = {row["blind_item_id"]: row for row in validator.read_review_sheet(paths["review_b"])}
    disputes = [
        {
            "blind_item_id": blind_id,
            "field": field,
            "reviewer_a_label": left[blind_id][field],
            "reviewer_b_label": right[blind_id][field],
            "adjudicated_value": "",
            "rationale": "",
            "adjudicator": "",
            "adjudicated_at": "",
        }
        for blind_id in sorted(left)
        for field in reviewed_fields(validator)
        if left[blind_id][field].strip().lower() != right[blind_id][field].strip().lower()
    ]
    if len(disputes) != agreement["disputed_cells"]:
        raise RuntimeError("recomputed dispute count differs from frozen agreement report")
    if paths["disputes"].exists():
        raise RuntimeError(f"refusing to overwrite adjudication sheet: {paths['disputes']}")
    paths["disputes"].parent.mkdir(parents=True, exist_ok=True)
    with paths["disputes"].open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(disputes[0])
            if disputes
            else (
                "blind_item_id",
                "field",
                "reviewer_a_label",
                "reviewer_b_label",
                "adjudicated_value",
                "rationale",
                "adjudicator",
                "adjudicated_at",
            ),
        )
        writer.writeheader()
        writer.writerows(disputes)
    manifest = {
        "state": "awaiting_human_adjudication",
        "disputed_cells": len(disputes),
        "agreement_report_sha256": sha256(paths["agreement"]),
        "completed_review_manifest_sha256": sha256(paths["completed_manifest"]),
        "adjudication_sheet_sha256_private_not_committed": sha256(paths["disputes"]),
        "agreed_raw_labels_locked": True,
        "automatic_adjudication": False,
        "qwen_or_gpu_execution": False,
        "base_rows_checked": len(bases),
    }
    write_json_exclusive(paths["adjudication_manifest"], manifest)
    return manifest


def read_adjudication(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finalize(data_dir: Path) -> dict[str, Any]:
    validator, bases = load_context(data_dir, require_complete=True)
    paths = review_paths(data_dir)
    if not paths["adjudication_manifest"].is_file() or not paths["disputes"].is_file():
        raise RuntimeError("prepare and complete human adjudication before finalizing")
    left = {row["blind_item_id"]: row for row in validator.read_review_sheet(paths["review_a"])}
    right = {row["blind_item_id"]: row for row in validator.read_review_sheet(paths["review_b"])}
    expected = {
        (blind_id, field)
        for blind_id in left
        for field in reviewed_fields(validator)
        if left[blind_id][field].strip().lower() != right[blind_id][field].strip().lower()
    }
    adjudication_rows = read_adjudication(paths["disputes"])
    decisions: dict[tuple[str, str], str] = {}
    for row in adjudication_rows:
        key = (row.get("blind_item_id", ""), row.get("field", ""))
        if key in decisions or key not in expected:
            raise RuntimeError(f"unexpected or duplicate adjudication cell: {key}")
        for field in ("adjudicated_value", "rationale", "adjudicator", "adjudicated_at"):
            if not row.get(field, "").strip():
                raise RuntimeError(f"incomplete adjudication {key}: missing {field}")
        decisions[key] = row["adjudicated_value"].strip()
    if set(decisions) != expected:
        raise RuntimeError("adjudication does not cover every and only disputed cell")
    final_rows: list[dict[str, str]] = []
    for blind_id in sorted(left):
        final: dict[str, str] = {"blind_item_id": blind_id}
        for field in reviewed_fields(validator):
            left_value = left[blind_id][field].strip()
            right_value = right[blind_id][field].strip()
            if left_value.lower() == right_value.lower():
                final[field] = left_value
            else:
                final[field] = decisions[(blind_id, field)]
        final_rows.append(final)
    temporary_labels = paths["final_labels"].with_name(".final_review_labels.validation.tmp")
    temporary_labels.parent.mkdir(parents=True, exist_ok=True)
    temporary_labels.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in final_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    try:
        acceptance = validator.validate_final_review_labels(
            temporary_labels,
            paths["review_a"],
            paths["review_b"],
            paths["key"],
            bases,
        )
    except Exception:
        temporary_labels.unlink(missing_ok=True)
        raise
    if paths["final_labels"].exists() or paths["adjudication_completed"].exists():
        temporary_labels.unlink(missing_ok=True)
        raise RuntimeError("refusing to overwrite finalized private review state")
    os.replace(temporary_labels, paths["final_labels"])
    completed = {
        "status": "complete",
        "disputed_cells": len(expected),
        "adjudicated_cells": len(decisions),
        "unresolved_cells": 0,
        "raw_reviews_frozen_before_adjudication": True,
        "reviewers_independent_attestation": True,
        "adjudication_sheet_sha256": sha256(paths["disputes"]),
        "final_labels_sha256": sha256(paths["final_labels"]),
    }
    write_json_exclusive(paths["adjudication_completed"], completed)
    validator.validate_private_adjudication(
        paths["adjudication_completed"], expected_disputes=len(expected)
    )
    manifest = {
        "state": "adjudicated_acceptance_contract_passed",
        "acceptance": acceptance,
        "raw_review_a_sha256": sha256(paths["review_a"]),
        "raw_review_b_sha256": sha256(paths["review_b"]),
        "final_labels_sha256_private_not_committed": sha256(paths["final_labels"]),
        "adjudication_completed_sha256_private_not_committed": sha256(
            paths["adjudication_completed"]
        ),
        "agreed_raw_labels_preserved": True,
        "qwen_or_gpu_execution": False,
    }
    write_json_exclusive(paths["finalized_manifest"], manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--snapshot-packet", action="store_true")
    action.add_argument("--score-only", action="store_true")
    action.add_argument("--prepare-adjudication", action="store_true")
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--received-a-at")
    parser.add_argument("--received-b-at")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.snapshot_packet:
        result = snapshot_packet(args.data_dir)
    elif args.score_only:
        if not args.received_a_at or not args.received_b_at:
            raise RuntimeError("--score-only requires both reviewer receipt timestamps")
        result = score_only(
            args.data_dir,
            received_a_at=args.received_a_at,
            received_b_at=args.received_b_at,
        )
    elif args.prepare_adjudication:
        result = prepare_adjudication(args.data_dir)
    else:
        result = finalize(args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
