#!/usr/bin/env python3
"""Validate the four fixed, zero-GPU C1/V7 Phase 2 authoring sentinels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from validate_m2_2b_v7_heldout import (
    LABELS,
    MIGRATIONS,
    VERSION,
    ValidationError,
    _require_keys,
    _require_text,
    _validate_construction_certificate,
    _validate_frozen_treatments,
    _validate_messages,
    load_jsonl,
    validate_family_cards,
    validate_overlap_audit,
    validate_overlap_matrix,
)

HERE = Path(__file__).resolve().parent
SENTINEL_PATH = HERE / "phase2_sentinel4_draft.jsonl"
EXPECTED = {
    "C1V7-001": ("C1V7-F01", 1),
    "C1V7-021": ("C1V7-F11", 1),
    "C1V7-033": ("C1V7-F17", 1),
    "C1V7-048": ("C1V7-F24", 2),
}
REQUIRED = (
    "dataset_version",
    "base_id",
    "family_id",
    "variant",
    "stratum",
    "initial_answer",
    "target_answer",
    "distractor_answer",
    "answer_migration",
    "options",
    "original_gold_rationale",
    "updated_gold_rationale",
    "messages_common",
    "v_new_evidence",
    "v_followup",
    "construction_certificate",
    "new_structure_attestation",
)


def validate_sentinels(
    rows: list[dict[str, Any]], cards: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(rows) != 4:
        raise ValidationError(f"expected 4 sentinel bases, found {len(rows)}")
    row_by_id = {str(row.get("base_id", "")): row for row in rows}
    if len(row_by_id) != 4 or set(row_by_id) != set(EXPECTED):
        raise ValidationError(f"sentinel base IDs must be exactly {sorted(EXPECTED)}")
    card_by_id = {str(card["family_id"]): card for card in cards}
    for base_id, (expected_family, expected_variant) in EXPECTED.items():
        base = row_by_id[base_id]
        where = f"sentinel/{base_id}"
        _require_keys(base, REQUIRED, where)
        if base["dataset_version"] != VERSION:
            raise ValidationError(f"{where}: wrong dataset version")
        if (base["family_id"], base["variant"]) != (
            expected_family,
            expected_variant,
        ):
            raise ValidationError(f"{where}: wrong frozen family/variant")
        if base["stratum"] != card_by_id[expected_family]["stratum"]:
            raise ValidationError(f"{where}: base/family stratum mismatch")
        answers = (
            base["initial_answer"],
            base["target_answer"],
            base["distractor_answer"],
        )
        if set(answers) != set(LABELS):
            raise ValidationError(f"{where}: answers must permute A/B/C")
        migration = f"{answers[0]}_to_{answers[1]}"
        if base["answer_migration"] != migration or migration not in MIGRATIONS:
            raise ValidationError(f"{where}: invalid answer migration")
        options = base["options"]
        if not isinstance(options, dict) or set(options) != set(LABELS):
            raise ValidationError(f"{where}: options must contain A/B/C exactly")
        texts = [str(options[label]).strip() for label in LABELS]
        if any(not value for value in texts) or len(set(texts)) != 3:
            raise ValidationError(f"{where}: option texts must be non-empty and unique")
        _require_text(base, "original_gold_rationale", where)
        _require_text(base, "updated_gold_rationale", where)
        if base["new_structure_attestation"] is not True:
            raise ValidationError(f"{where}: new_structure_attestation must be true")
        _validate_messages(base, where)
        _validate_frozen_treatments(base, where)
        _validate_construction_certificate(base, where)
    return {
        "status": "passed",
        "sentinels": 4,
        "base_ids": sorted(row_by_id),
        "qwen_or_gpu_execution": False,
    }


def main() -> None:
    cards = load_jsonl(HERE / "family_cards24.jsonl")
    validate_family_cards(cards, require_approved=True)
    validate_overlap_matrix(HERE / "DEVELOPMENT_OVERLAP_MATRIX.md", require_approved=True)
    validate_overlap_audit(
        HERE / "development_overlap_audit24.jsonl",
        cards,
        require_approved=True,
    )
    report = validate_sentinels(load_jsonl(SENTINEL_PATH), cards)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
