#!/usr/bin/env python3
"""Validate two independent blind extractions and reconcile disagreements while still blind."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cedar_m2.eval.v6_zero_gpu import (
    ANNOTATION_FIELDS,
    cohen_kappa,
    file_sha256,
    read_csv,
    read_jsonl,
    validate_annotation_rows,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts" / "m2" / "m2_2b_v6_zero_gpu_reaudit"
CORE_FIELDS = ("answer_extracted", "json_valid", "complete", "ambiguity_reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--annotator-a", type=Path)
    parser.add_argument("--annotator-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    return parser.parse_args()


def rate(matches: int, total: int) -> dict[str, Any]:
    return {"matches": matches, "total": total, "rate": matches / total if total else None}


def core_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left[field] == right[field] for field in CORE_FIELDS)


def main() -> None:
    args = parse_args()
    packet_path = args.root / "blind" / "annotation_packet.jsonl"
    packet = read_jsonl(packet_path)
    expected_ids = [row["blind_id"] for row in packet]
    if len(expected_ids) != 1152 or len(set(expected_ids)) != 1152:
        raise ValueError("Blind packet must contain 1152 unique IDs")
    surfaces = {row["blind_id"]: row["final_surface"] for row in packet}

    a_path = args.annotator_a or args.root / "blind" / "annotator_a.csv"
    b_path = args.annotator_b or args.root / "blind" / "annotator_b.csv"
    a = validate_annotation_rows(read_csv(a_path), expected_ids)
    b = validate_annotation_rows(read_csv(b_path), expected_ids)

    disagreements = [
        blind_id
        for blind_id in expected_ids
        if not core_equal(a[blind_id], b[blind_id])
    ]
    agreement = {
        "status": "needs_adjudication" if disagreements else "complete_no_disagreements",
        "blind_only": True,
        "records": len(expected_ids),
        "packet_sha256": file_sha256(packet_path),
        "annotator_a_sha256": file_sha256(a_path),
        "annotator_b_sha256": file_sha256(b_path),
        "all_core_fields": rate(len(expected_ids) - len(disagreements), len(expected_ids)),
        "answer": rate(
            sum(
                a[item]["answer_extracted"] == b[item]["answer_extracted"]
                for item in expected_ids
            ),
            len(expected_ids),
        ),
        "json_valid": rate(
            sum(a[item]["json_valid"] == b[item]["json_valid"] for item in expected_ids),
            len(expected_ids),
        ),
        "complete": rate(
            sum(a[item]["complete"] == b[item]["complete"] for item in expected_ids),
            len(expected_ids),
        ),
        "kappa": {
            "answer": cohen_kappa(
                [a[item]["answer_extracted"] for item in expected_ids],
                [b[item]["answer_extracted"] for item in expected_ids],
            ),
            "json_valid": cohen_kappa(
                [a[item]["json_valid"] for item in expected_ids],
                [b[item]["json_valid"] for item in expected_ids],
            ),
            "complete": cohen_kappa(
                [a[item]["complete"] for item in expected_ids],
                [b[item]["complete"] for item in expected_ids],
            ),
        },
        "disagreements": len(disagreements),
        "ambiguity_reason_a": dict(Counter(a[item]["ambiguity_reason"] for item in expected_ids)),
        "ambiguity_reason_b": dict(Counter(b[item]["ambiguity_reason"] for item in expected_ids)),
    }

    adjudication_dir = args.root / "adjudication"
    disagreement_path = adjudication_dir / "disagreement_template.csv"
    disagreement_fields = (
        "blind_id",
        "final_surface",
        "a_answer_extracted",
        "a_json_valid",
        "a_complete",
        "a_ambiguity_reason",
        "b_answer_extracted",
        "b_json_valid",
        "b_complete",
        "b_ambiguity_reason",
        *ANNOTATION_FIELDS[1:],
    )
    disagreement_rows = []
    for blind_id in disagreements:
        disagreement_rows.append(
            {
                "blind_id": blind_id,
                "final_surface": surfaces[blind_id],
                **{f"a_{field}": a[blind_id][field] for field in CORE_FIELDS},
                **{f"b_{field}": b[blind_id][field] for field in CORE_FIELDS},
                "answer_extracted": "",
                "json_valid": "",
                "complete": "",
                "ambiguity_reason": "",
                "notes": "",
            }
        )
    write_csv(disagreement_path, disagreement_rows, disagreement_fields)

    adjudicated = {item: dict(a[item]) for item in expected_ids if item not in disagreements}
    if disagreements:
        if args.adjudication is None:
            agreement["disagreement_template"] = str(disagreement_path)
            write_json(adjudication_dir / "agreement_report.json", agreement)
            print(json.dumps({"status": agreement["status"], "disagreements": len(disagreements)}))
            return
        resolved = validate_annotation_rows(read_csv(args.adjudication), disagreements)
        adjudicated.update(resolved)
        agreement["adjudication_input_sha256"] = file_sha256(args.adjudication)

    if set(adjudicated) != set(expected_ids):
        raise ValueError("Adjudicated labels are incomplete")
    output_rows = [adjudicated[item] for item in expected_ids]
    output_path = adjudication_dir / "adjudicated_labels.csv"
    write_csv(output_path, output_rows, ANNOTATION_FIELDS)
    agreement["status"] = "complete"
    agreement["adjudicated_sha256"] = file_sha256(output_path)
    agreement["ambiguity_reason_adjudicated"] = dict(
        Counter(adjudicated[item]["ambiguity_reason"] for item in expected_ids)
    )
    write_json(adjudication_dir / "agreement_report.json", agreement)
    print(
        json.dumps(
            {
                "status": "complete",
                "disagreements": len(disagreements),
                "adjudicated": str(output_path),
            }
        )
    )


if __name__ == "__main__":
    main()
