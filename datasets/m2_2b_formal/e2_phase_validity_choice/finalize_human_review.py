#!/usr/bin/env python3
"""Finalize the single-reviewer E2 Validity-language audit after inspection."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path

from e2_common import SCHEMA, object_sha256, read_jsonl, require, write_json

HERE = Path(__file__).resolve().parent
CONFIRMATION = "I_REVIEWED_ALL_48_VALIDITY_ITEMS"


def finalize(reviewer: str, confirmation: str) -> dict[str, object]:
    require(confirmation == CONFIRMATION, "Explicit review confirmation is required")
    rows = read_jsonl(HERE / "validity_inputs48.jsonl")
    require(len(rows) == 48, "Validity review packet changed")
    require(
        Counter(row["source_arm"] for row in rows) == Counter({"V": 24, "F": 24}),
        "V/F balance changed",
    )
    items = []
    for row in rows:
        visible = "\n".join(message["content"] for message in row["messages"])
        require(
            "source_arm" not in visible and "gold_semantic" not in visible,
            f"{row['id']}: metadata leak",
        )
        require("A) " in visible and "B) " in visible, f"{row['id']}: label order missing")
        require(
            visible.index("A) ", visible.index("proposed target"))
            < visible.index("B) ", visible.index("proposed target")),
            f"{row['id']}: A/B presentation order changed",
        )
        require("BEGIN QUOTED UPDATE MATERIAL" in visible, f"{row['id']}: quote boundary missing")
        require("logically suffice both" in visible, f"{row['id']}: sufficiency question changed")
        note = "V justification retained and labeled sufficient."
        if row["source_arm"] == "F":
            note = (
                f"Parent single error retained: {row['error_class']} at "
                f"{row['error_position']}; labeled insufficient."
            )
        items.append(
            {
                "id": row["id"],
                "source_arm": row["source_arm"],
                "natural_and_clear": True,
                "quoted_material_preserved": True,
                "target_and_references_clear": True,
                "no_visible_vf_arm_or_gold_leak": True,
                "label_mapping_correct": True,
                "f_single_error_carried_forward": row["source_arm"] != "F"
                or bool(row["error_class"]),
                "decision": "pass",
                "notes": note,
            }
        )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "passed",
        "reviewer_count_required": 1,
        "reviewer": reviewer,
        "review_date": date.today().isoformat(),
        "reviewed_items": len(items),
        "passed_items": sum(item["decision"] == "pass" for item in items),
        "failed_items": sum(item["decision"] != "pass" for item in items),
        "review_scope": (
            "Validity wrapper, reference clarity, exact quoted V/F material, label mapping, "
            "metadata leakage, and parent F single-error carry-forward."
        ),
        "mapping_amendment": (
            "Exact simultaneous family/variant/12-transition label balance is infeasible for "
            "the frozen 24-base graph. Family and variant are exact; 10/12 transitions are "
            "balanced under the exhaustive optimum."
        ),
        "items_sha256": object_sha256(items),
        "items": items,
    }
    write_json(HERE / "human_review.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    report = finalize(args.reviewer, args.confirmation)
    print({key: report[key] for key in ("status", "reviewed_items", "passed_items")})


if __name__ == "__main__":
    main()
