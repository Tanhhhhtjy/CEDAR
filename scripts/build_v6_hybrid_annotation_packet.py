#!/usr/bin/env python3
"""Build the pre-key hybrid audit packet for the v6 zero-GPU re-audit.

This script reads only the two public blind CSV templates. It deliberately does
not accept a private-key path or any experimental metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASELINE = "7e26853f86d4449d7e3130c4bd378e6f1dada473"
SELECTION_PREFIX = f"CEDAR|V6-ZERO-GPU-HYBRID-AUDIT|{BASELINE}|"
DISPLAY_PREFIX = f"CEDAR|V6-ZERO-GPU-HYBRID-DISPLAY|{BASELINE}|"
LABEL_COLUMNS = (
    "answer_extracted",
    "json_valid",
    "complete",
    "ambiguity_reason",
    "notes",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_text_sha256(path: Path) -> str:
    value = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(value).hexdigest()


def read_blind_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = ["blind_id", "final_surface", *LABEL_COLUMNS]
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    if list(rows[0]) != expected:
        raise ValueError(f"unexpected columns in {path}: {list(rows[0])}")
    return rows


def strict_parse(surface: str) -> tuple[bool, dict[str, Any] | None, str]:
    try:
        obj = json.loads(surface.strip())
    except Exception:
        return False, None, "json_parse_error"
    if not isinstance(obj, dict):
        return False, None, "not_object"
    if set(obj) != {"answer", "confidence", "reason"}:
        return False, None, "keys_not_exact"
    if not isinstance(obj["answer"], str) or obj["answer"] not in {"A", "B", "C"}:
        return False, None, "bad_answer"
    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False, None, "bad_confidence_type"
    if not 0 <= confidence <= 100:
        return False, None, "bad_confidence_range"
    if not isinstance(obj["reason"], str) or not obj["reason"].strip():
        return False, None, "bad_reason"
    return True, obj, "strict_machine_ok"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows_a = read_blind_csv(args.annotator_a)
    rows_b = read_blind_csv(args.annotator_b)
    if len(rows_a) != 1152 or len(rows_b) != 1152:
        raise ValueError(f"expected 1152 rows per template, got {len(rows_a)} and {len(rows_b)}")
    if any(row[column].strip() for row in rows_a + rows_b for column in LABEL_COLUMNS):
        raise ValueError("source templates already contain labels")
    visible_a = [(row["blind_id"], row["final_surface"]) for row in rows_a]
    visible_b = [(row["blind_id"], row["final_surface"]) for row in rows_b]
    if visible_a != visible_b:
        raise ValueError("annotator A/B visible packets differ")
    ids = [row["blind_id"] for row in rows_a]
    if len(set(ids)) != 1152:
        raise ValueError("blind IDs are not unique")

    parsed_rows: list[dict[str, str]] = []
    strict_rows: list[dict[str, str]] = []
    mandatory_rows: list[dict[str, str]] = []
    failure_counts: dict[str, int] = {}
    for row in rows_a:
        ok, obj, parse_class = strict_parse(row["final_surface"])
        failure_counts[parse_class] = failure_counts.get(parse_class, 0) + 1
        parsed = {
            "blind_id": row["blind_id"],
            "final_surface": row["final_surface"],
            "machine_parse_class": parse_class,
            "machine_answer_extracted": obj["answer"] if ok and obj is not None else "",
            "machine_json_valid": "true" if ok else "",
            "machine_complete": "true" if ok else "",
            "selection_hash": sha256_text(SELECTION_PREFIX + row["blind_id"]),
            "display_hash": sha256_text(DISPLAY_PREFIX + row["blind_id"]),
            "selected_for_human": "",
            "selection_reason": "",
        }
        parsed_rows.append(parsed)
        (strict_rows if ok else mandatory_rows).append(parsed)

    audit_n = math.ceil(0.15 * len(strict_rows))
    strict_ranked = sorted(strict_rows, key=lambda row: (row["selection_hash"], row["blind_id"]))
    sampled = strict_ranked[:audit_n]
    sampled_ids = {row["blind_id"] for row in sampled}
    mandatory_ids = {row["blind_id"] for row in mandatory_rows}
    for row in parsed_rows:
        if row["blind_id"] in mandatory_ids:
            row["selected_for_human"] = "true"
            row["selection_reason"] = "mandatory_human_review"
        elif row["blind_id"] in sampled_ids:
            row["selected_for_human"] = "true"
            row["selection_reason"] = "strict_machine_random_audit_15pct"
        else:
            row["selected_for_human"] = "false"

    selected = sorted(
        (row for row in parsed_rows if row["selected_for_human"] == "true"),
        key=lambda row: (row["display_hash"], row["blind_id"]),
    )
    blind_rows = [
        {
            "blind_id": row["blind_id"],
            "final_surface": row["final_surface"],
            **{column: "" for column in LABEL_COLUMNS},
        }
        for row in selected
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "blind" / "hybrid_annotator_template.csv",
        blind_rows,
        ["blind_id", "final_surface", *LABEL_COLUMNS],
    )
    write_csv(
        args.output_dir / "machine" / "machine_full_parse.csv",
        parsed_rows,
        list(parsed_rows[0]),
    )
    selected_id_payload = "\n".join(row["blind_id"] for row in selected) + "\n"
    manifest = {
        "schema_version": "v6-zero-gpu-hybrid-audit-1",
        "baseline_commit": BASELINE,
        "source_a": {
            "path": str(args.annotator_a).replace("\\", "/"),
            "rows": len(rows_a),
            "sha256": portable_text_sha256(args.annotator_a),
        },
        "source_b": {
            "path": str(args.annotator_b).replace("\\", "/"),
            "rows": len(rows_b),
            "sha256": portable_text_sha256(args.annotator_b),
        },
        "selection": {
            "selection_prefix": SELECTION_PREFIX,
            "display_prefix": DISPLAY_PREFIX,
            "strict_machine_ok": len(strict_rows),
            "mandatory_human_review": len(mandatory_rows),
            "audit_fraction": 0.15,
            "strict_random_audit": audit_n,
            "human_rows_per_annotator": len(selected),
            "selected_ids_ordered_sha256": sha256_text(selected_id_payload),
        },
        "parse_class_counts": dict(sorted(failure_counts.items())),
        "private_key_read": False,
        "labels_prefilled": False,
    }
    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"] = {
        "blind_template_sha256": sha256_file(args.output_dir / "blind" / "hybrid_annotator_template.csv"),
        "machine_parse_sha256": sha256_file(args.output_dir / "machine" / "machine_full_parse.csv"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
