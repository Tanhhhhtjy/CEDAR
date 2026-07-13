#!/usr/bin/env python3
"""Build the prompt-only neutral derivative for the paired v4 calibration."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import file_sha256, object_sha256
from cedar_m2.data.m2_2b_v4_neutral import (
    ALLOWED_CHANGED_FIELDS,
    ARMS,
    N_PROMPT,
    PROMPT_CONTRACT_SHA256,
    R_PROMPT,
    SYSTEM,
    changed_fields,
    neutralize_rows,
    split_vf_followup,
    text_sha256,
    unchanged_projection,
)

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "m2_2b_v4_public_poc180_arms.jsonl"
OUTPUT = HERE / "m2_2b_v4_public_poc180_neutral_prompt_arms.jsonl"
AUDIT = HERE / "neutral_prompt_diff_audit.json"
SOURCE_COMMIT = "a6d71b656cd9682ec3facbb4d74aba653a5d0d74"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_audit(
    source_rows: list[dict[str, Any]],
    neutral_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    violations: list[str] = []
    records: list[dict[str, Any]] = []
    vf_content_preserved = 0
    expected_ids = [row.get("id") for row in source_rows]
    observed_ids = [row.get("id") for row in neutral_rows]
    if expected_ids != observed_ids:
        violations.append("ID sequence changed")

    for source, neutral in zip(source_rows, neutral_rows, strict=True):
        item_id = str(source.get("id"))
        observed_changes = changed_fields(source, neutral)
        unexpected = sorted(set(observed_changes) - ALLOWED_CHANGED_FIELDS)
        if unexpected:
            violations.append(f"{item_id}: unexpected changed fields {unexpected}")
        if neutral.get("system") != SYSTEM:
            violations.append(f"{item_id}: system does not match frozen P0")
        if unchanged_projection(source) != unchanged_projection(neutral):
            violations.append(f"{item_id}: unchanged field projection differs")

        record: dict[str, Any] = {
            "id": item_id,
            "arm": source.get("arm"),
            "changed_fields": observed_changes,
            "unchanged_projection_sha256": object_sha256(unchanged_projection(source)),
            "source_system_sha256": text_sha256(str(source.get("system", ""))),
            "neutral_system_sha256": text_sha256(str(neutral.get("system", ""))),
            "source_followup_sha256": text_sha256(str(source.get("followup_user", ""))),
            "neutral_followup_sha256": text_sha256(str(neutral.get("followup_user", ""))),
        }
        if source.get("arm") in {"V", "F"}:
            source_observation, source_candidate = split_vf_followup(
                str(source["followup_user"])
            )
            neutral_observation, neutral_candidate = split_vf_followup(
                str(neutral["followup_user"])
            )
            if (source_observation, source_candidate) != (
                neutral_observation,
                neutral_candidate,
            ):
                violations.append(f"{item_id}: V/F inner content changed")
            else:
                vf_content_preserved += 1
            record.update(
                {
                    "observation_sha256": text_sha256(source_observation),
                    "candidate_reasoning_sha256": text_sha256(source_candidate),
                }
            )
        records.append(record)

    changed_field_counts = Counter(
        field for record in records for field in record["changed_fields"]
    )
    contract_counts = Counter(row.get("arm") for row in neutral_rows)
    if contract_counts != Counter({arm: 36 for arm in ARMS}):
        violations.append(f"Arm balance changed: {dict(contract_counts)}")
    if any(row["system"] != SYSTEM for row in neutral_rows):
        violations.append("Not all rows use the frozen P0 system")
    if any(row["followup_user"] != N_PROMPT for row in neutral_rows if row["arm"] == "N"):
        violations.append("N prompt mismatch")
    if any(row["followup_user"] != R_PROMPT for row in neutral_rows if row["arm"] == "R"):
        violations.append("R prompt mismatch")

    return {
        "status": "passed" if not violations else "failed",
        "source_commit": SOURCE_COMMIT,
        "source_path": str(SOURCE.relative_to(HERE.parents[3])),
        "neutral_path": str(OUTPUT.relative_to(HERE.parents[3])),
        "contract_sources": [
            "docs/M2_2B_V3_HELDOUT_INTAKE.md",
            "datasets/m2_2b_formal/v3_heldout/build_m2_2b_v3_heldout.py",
        ],
        "prompt_contract_sha256": PROMPT_CONTRACT_SHA256,
        "rows": len(neutral_rows),
        "base_units": len({row["base_id"] for row in neutral_rows}),
        "arms": dict(sorted(contract_counts.items())),
        "ids_and_record_order_unchanged": expected_ids == observed_ids,
        "allowed_changed_fields": sorted(ALLOWED_CHANGED_FIELDS),
        "observed_changed_field_counts": dict(sorted(changed_field_counts.items())),
        "vf_observation_and_candidate_preserved": vf_content_preserved,
        "unchanged_field_violations": len(violations),
        "violations": violations,
        "source_sha256": file_sha256(SOURCE),
        "neutral_sha256": file_sha256(OUTPUT),
        "records": records,
    }


def main() -> None:
    source_rows = load_jsonl(SOURCE)
    if len(source_rows) != 180:
        raise ValueError(f"Expected 180 source rows, found {len(source_rows)}")
    neutral_rows = neutralize_rows(source_rows)
    atomic_write_jsonl(OUTPUT, neutral_rows)
    audit = build_audit(source_rows, neutral_rows)
    if audit["status"] != "passed":
        raise RuntimeError(json.dumps(audit["violations"], ensure_ascii=False))
    atomic_write_json(AUDIT, audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "rows": audit["rows"],
                "base_units": audit["base_units"],
                "changed_field_counts": audit["observed_changed_field_counts"],
                "vf_inner_content_preserved": audit[
                    "vf_observation_and_candidate_preserved"
                ],
                "neutral_sha256": audit["neutral_sha256"],
                "prompt_contract_sha256": audit["prompt_contract_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
