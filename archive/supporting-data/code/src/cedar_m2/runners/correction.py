"""Resumable uncapped scoring for the objective correction screen."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from torch import nn

from cedar_m2.checkpoint import RecordStore, object_sha256
from cedar_m2.data.correction_screen import CorrectionCase
from cedar_m2.eval.correction import score_correction_conversations


def correction_score_key(item_id: str, variant: str) -> dict[str, str]:
    return {"item_id": item_id, "variant": variant}


def run_correction_scoring_stage(
    model: nn.Module,
    tokenizer: Any,
    cases: Sequence[CorrectionCase],
    store: RecordStore,
    *,
    batch_size: int,
) -> dict[str, int]:
    """Score missing conversations and atomically checkpoint each result."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    expected_keys = [correction_score_key(case.item_id, case.variant) for case in cases]
    already_complete = sum(store.contains(key) for key in expected_keys)
    newly_scored = 0
    batches = (len(cases) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(cases), batch_size), start=1):
        chunk = cases[start : start + batch_size]
        missing = [
            case
            for case in chunk
            if not store.contains(correction_score_key(case.item_id, case.variant))
        ]
        if not missing:
            continue
        scores = score_correction_conversations(
            model,
            tokenizer,
            [case.messages for case in missing],
            correct_labels=[case.correct_label for case in missing],
            batch_size=batch_size,
        )
        for case, score in zip(missing, scores, strict=True):
            payload = {
                "item_id": case.item_id,
                "category": case.category,
                "variant": case.variant,
                "correct_label": case.correct_label,
                "initial_label": case.initial_label,
                "challenged_label": case.challenged_label,
                "conversation_sha256": object_sha256(list(case.messages)),
                **score.to_dict(),
            }
            store.write(correction_score_key(case.item_id, case.variant), payload)
            newly_scored += 1
        print(
            json.dumps(
                {
                    "stage": "score",
                    "batch": batch_number,
                    "batches": batches,
                    "variant": missing[0].variant,
                    "scored": len(missing),
                }
            ),
            flush=True,
        )

    return {"already_complete": already_complete, "newly_scored": newly_scored}
