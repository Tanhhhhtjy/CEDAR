"""Resumable deterministic scoring for the M2.2 Belief-R arm."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

from torch import nn

from cedar_m2.checkpoint import RecordStore
from cedar_m2.data.belief_r import BeliefPair, BeliefSplits
from cedar_m2.eval.mcq import score_mcq_prompts
from cedar_m2.interventions.capping import ActivationCapper, LayerCap

CONDITION_EXPERIMENTS: dict[str, str | None] = {
    "uncapped": None,
    "p1": "layers_46:54-p0.01",
    "p25": "layers_46:54-p0.25",
    "p50": "layers_46:54-p0.5",
    "p75": "layers_46:54-p0.75",
}
SPLIT_CONDITIONS = {
    "development": ("uncapped",),
    "pilot": tuple(CONDITION_EXPERIMENTS),
    "extension": ("uncapped", "p25"),
}
STEPS = ("time_t", "time_t1")


def pairs_for_split(splits: BeliefSplits, split: str) -> tuple[BeliefPair, ...]:
    """Return the preregistered item sequence for one M2.2 phase."""
    if split == "development":
        return splits.development
    if split == "pilot":
        return (*splits.pilot_bu, *splits.pilot_bm)
    if split == "extension":
        return splits.extension_bu
    raise ValueError(f"Unknown M2.2 split {split!r}")


def conditions_for_split(split: str) -> tuple[str, ...]:
    """Return conditions allowed in one phase without exposing later phases."""
    try:
        return SPLIT_CONDITIONS[split]
    except KeyError as error:
        raise ValueError(f"Unknown M2.2 split {split!r}") from error


def expected_score_count(item_count: int, conditions: Sequence[str]) -> int:
    """Return the exact number of item-condition-step records."""
    return item_count * len(conditions) * len(STEPS)


def score_key(sample_id: str, condition: str, step: str) -> dict[str, str]:
    """Build the deterministic checkpoint key for one score."""
    return {"sample_id": sample_id, "condition": condition, "step": step}


def _chunks(items: Sequence[BeliefPair], size: int) -> list[Sequence[BeliefPair]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[start : start + size] for start in range(0, len(items), size)]


def run_belief_scoring_stage(
    model: nn.Module,
    tokenizer: Any,
    pairs: Sequence[BeliefPair],
    conditions: Sequence[str],
    caps_by_condition: Mapping[str, Mapping[int, LayerCap]],
    store: RecordStore,
    *,
    batch_size: int,
) -> dict[str, int]:
    """Score missing records and checkpoint every deterministic batch."""
    unknown = set(conditions) - set(CONDITION_EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown M2.2 conditions: {sorted(unknown)}")
    for condition in conditions:
        if condition != "uncapped" and condition not in caps_by_condition:
            raise ValueError(f"Missing capping assets for condition {condition!r}")

    expected_keys = [
        score_key(pair.sample_id, condition, step)
        for condition in conditions
        for step in STEPS
        for pair in pairs
    ]
    already_complete = sum(store.contains(key) for key in expected_keys)
    newly_scored = 0
    batch_number = 0
    total_batches = len(conditions) * len(STEPS) * len(_chunks(pairs, batch_size))

    for condition in conditions:
        context = (
            nullcontext()
            if condition == "uncapped"
            else ActivationCapper(
                model,
                caps_by_condition[condition],
                positions="all",
                track_stats=False,
            )
        )
        with context:
            for step in STEPS:
                for chunk in _chunks(pairs, batch_size):
                    batch_number += 1
                    missing = [
                        pair
                        for pair in chunk
                        if not store.contains(score_key(pair.sample_id, condition, step))
                    ]
                    if not missing:
                        continue
                    rows = [pair.time_t if step == "time_t" else pair.time_t1 for pair in missing]
                    scores = score_mcq_prompts(
                        model,
                        tokenizer,
                        [row.question for row in rows],
                        correct_labels=[row.ground_truth for row in rows],
                        batch_size=batch_size,
                    )
                    for pair, row, score in zip(missing, rows, scores, strict=True):
                        payload = {
                            "sample_id": pair.sample_id,
                            "target": pair.target,
                            "condition": condition,
                            "step": step,
                            "modus": row.modus,
                            "relation_type": row.relation_type,
                            "agreement_level": row.agreement_level,
                            "ground_truth": row.ground_truth,
                            "question_sha256": hashlib.sha256(row.question.encode()).hexdigest(),
                            **score.to_dict(),
                        }
                        store.write(score_key(pair.sample_id, condition, step), payload)
                        newly_scored += 1
                    print(
                        json.dumps(
                            {
                                "stage": "score",
                                "batch": batch_number,
                                "batches": total_batches,
                                "condition": condition,
                                "step": step,
                                "scored": len(missing),
                            }
                        ),
                        flush=True,
                    )

    return {"already_complete": already_complete, "newly_scored": newly_scored}
