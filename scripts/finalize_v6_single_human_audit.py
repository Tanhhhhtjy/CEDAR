#!/usr/bin/env python3
"""Finalize v6 blind labels from exact parsing plus one blind human audit.

This script is deliberately pre-key: it does not accept, locate, or read the
private blind key and it does not recover any experimental condition.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts" / "m2" / "m2_2b_v6_zero_gpu_reaudit"
BASELINE = "7e26853f86d4449d7e3130c4bd378e6f1dada473"

# Load the frozen pure-standard-library helper directly by path. Importing the
# top-level cedar_m2 package would import torch even though this pre-key step
# performs no tensor, model, CUDA, or GPU work.
HELPER_PATH = ROOT / "src" / "cedar_m2" / "eval" / "v6_zero_gpu.py"
SPEC = importlib.util.spec_from_file_location("cedar_v6_zero_gpu_pure", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load frozen helper: {HELPER_PATH}")
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)
ANNOTATION_FIELDS = HELPER.ANNOTATION_FIELDS
file_sha256 = HELPER.file_sha256
portable_text_file_sha256 = HELPER.portable_text_file_sha256
read_csv = HELPER.read_csv
read_jsonl = HELPER.read_jsonl
strict_json_answer = HELPER.strict_json_answer
validate_annotation_rows = HELPER.validate_annotation_rows
write_csv = HELPER.write_csv
write_json = HELPER.write_json
write_jsonl = HELPER.write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    packet_path = root / "blind" / "annotation_packet.jsonl"
    freeze_path = root / "input_freeze_manifest.json"
    selection_path = root / "hybrid" / "selection_manifest.json"
    audit_template_path = root / "hybrid" / "blind" / "hybrid_annotator_template.csv"
    attestation_path = root / "hybrid" / "human_review_attestation.json"

    packet = read_jsonl(packet_path)
    freeze = load_json(freeze_path)
    selection = load_json(selection_path)
    attestation = load_json(attestation_path)

    if freeze.get("packet_sha256") != portable_text_file_sha256(packet_path):
        raise ValueError("Frozen packet hash mismatch")
    if len(packet) != 1152 or len({row["blind_id"] for row in packet}) != 1152:
        raise ValueError("Expected 1152 unique blind packet rows")
    if selection.get("baseline_commit") != BASELINE:
        raise ValueError("Hybrid selection baseline mismatch")
    if selection.get("private_key_read") is not False:
        raise ValueError("Hybrid selection was not pre-key")
    if attestation.get("status") != "complete":
        raise ValueError("Human review attestation is incomplete")
    if attestation.get("reviewed_rows") != 174:
        raise ValueError("Human review must cover 174 selected rows")
    if attestation.get("review_materials_included_private_key") is not False:
        raise ValueError("Human review materials included private key")
    findings = attestation.get("findings", {})
    if findings != {
        "legal_complete_unique_answer_json": 173,
        "empty_final_surface": 1,
        "other_anomaly": 0,
    }:
        raise ValueError(f"Unexpected human review findings: {findings}")

    selected_rows = read_csv(audit_template_path)
    selected_ids = [row["blind_id"] for row in selected_rows]
    if len(selected_ids) != 174 or len(set(selected_ids)) != 174:
        raise ValueError("Hybrid audit template must contain 174 unique rows")
    if any(
        row.get(field, "").strip()
        for row in selected_rows
        for field in ANNOTATION_FIELDS[1:]
    ):
        raise ValueError("Hybrid audit template unexpectedly contains manual labels")
    selected_set = set(selected_ids)

    labels: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    parse_counts: Counter[str] = Counter()
    empty_ids: list[str] = []
    selected_strict = 0
    for row in packet:
        blind_id = row["blind_id"]
        surface = row["final_surface"]
        answer = strict_json_answer(surface)
        if answer is not None:
            parse_counts["strict_exact_json"] += 1
            in_audit = blind_id in selected_set
            selected_strict += int(in_audit)
            labels.append(
                {
                    "blind_id": blind_id,
                    "answer_extracted": answer,
                    "json_valid": True,
                    "complete": True,
                    "ambiguity_reason": "",
                    "notes": "",
                }
            )
            provenance.append(
                {
                    "blind_id": blind_id,
                    "label_source": (
                        "machine_validated_by_single_human_audit"
                        if in_audit
                        else "machine_exact_json"
                    ),
                    "strict_parser_answer": answer,
                }
            )
            continue
        if surface.strip() == "":
            parse_counts["empty_final_surface"] += 1
            empty_ids.append(blind_id)
            if blind_id not in selected_set:
                raise ValueError("Empty final surface was not included in mandatory audit")
            labels.append(
                {
                    "blind_id": blind_id,
                    "answer_extracted": "",
                    "json_valid": False,
                    "complete": False,
                    "ambiguity_reason": "incomplete_or_truncated",
                    "notes": "",
                }
            )
            provenance.append(
                {
                    "blind_id": blind_id,
                    "label_source": "single_human_mandatory_empty",
                    "strict_parser_answer": None,
                }
            )
            continue
        raise ValueError(f"Unexpected non-strict non-empty output: {blind_id}")

    if parse_counts != Counter({"strict_exact_json": 1151, "empty_final_surface": 1}):
        raise ValueError(f"Unexpected parse counts: {parse_counts}")
    if selected_strict != 173 or len(empty_ids) != 1:
        raise ValueError(
            f"Unexpected selected audit composition: strict={selected_strict}, empty={empty_ids}"
        )

    expected_ids = [row["blind_id"] for row in packet]
    adjudication_dir = root / "adjudication"
    labels_path = adjudication_dir / "adjudicated_labels.csv"
    provenance_path = adjudication_dir / "label_provenance.jsonl"
    report_path = adjudication_dir / "agreement_report.json"
    write_csv(labels_path, labels, ANNOTATION_FIELDS)
    validate_annotation_rows(read_csv(labels_path), expected_ids)
    write_jsonl(provenance_path, provenance)

    report = {
        "status": "complete",
        "annotation_design": "exact_parser_plus_single_human_blind_audit",
        "not_double_annotated": True,
        "cohen_kappa": None,
        "cohen_kappa_reason": "No second independent annotation was performed.",
        "records": len(packet),
        "packet_sha256": portable_text_file_sha256(packet_path),
        "packet_worktree_raw_sha256": file_sha256(packet_path),
        "selection_manifest_sha256": portable_text_file_sha256(selection_path),
        "audit_template_sha256": portable_text_file_sha256(audit_template_path),
        "human_review_attestation_sha256": portable_text_file_sha256(attestation_path),
        "adjudicated_sha256": portable_text_file_sha256(labels_path),
        "label_provenance_sha256": portable_text_file_sha256(provenance_path),
        "parse_counts": dict(parse_counts),
        "human_audit": {
            "reviewed": 174,
            "strict_exact_json": 173,
            "mandatory_empty": 1,
            "machine_human_discrepancies_reported": 0,
        },
        "empty_blind_id": empty_ids[0],
        "private_key_read": False,
        "gpu_or_model_execution": False,
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "records": len(labels),
                "parse_counts": dict(parse_counts),
                "adjudicated_labels": str(labels_path),
                "adjudicated_sha256": report["adjudicated_sha256"],
                "private_key_read": False,
                "gpu_or_model_execution": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
