"""Validated project-local formal-domain items derived from Belief-R."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cedar_m2.data.belief_r import BeliefPair


@dataclass(frozen=True)
class FormalItem:
    """One self-contained ``time_t1`` formal-domain multiple-choice item."""

    dataset_id: str
    modus: str
    question: str
    options: tuple[str, str, str]
    ground_truth: str
    should_update: bool
    agreement_level: int

    @property
    def sample_id(self) -> str:
        return f"{self.dataset_id}:{self.modus}"


def read_formal_items(path: str | Path) -> list[FormalItem]:
    """Read the checked-in formal JSONL and reject malformed or duplicate items."""
    required = {
        "id",
        "question",
        "options",
        "ground_truth",
        "should_update",
        "modus",
        "agreement_lv",
    }
    items: list[FormalItem] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not required <= set(record):
                raise ValueError(f"Malformed formal-domain record on line {line_number}")
            options = record["options"]
            if not isinstance(options, dict) or set(options) != {"a", "b", "c"}:
                raise ValueError(f"Malformed formal-domain options on line {line_number}")
            ground_truth = str(record["ground_truth"]).strip().upper()
            if ground_truth not in {"A", "B", "C"}:
                raise ValueError(
                    f"Unexpected formal-domain ground truth {ground_truth!r} on line {line_number}"
                )
            should_update = record["should_update"]
            if not isinstance(should_update, bool) or should_update != (ground_truth == "C"):
                raise ValueError(f"Inconsistent should_update on line {line_number}")
            agreement_level = record["agreement_lv"]
            if isinstance(agreement_level, bool) or not isinstance(agreement_level, int):
                raise ValueError(f"Invalid agreement level on line {line_number}")

            item = FormalItem(
                dataset_id=str(record["id"]).strip(),
                modus=str(record["modus"]).strip(),
                question=str(record["question"]).strip(),
                options=tuple(str(options[label]).strip() for label in ("a", "b", "c")),
                ground_truth=ground_truth,
                should_update=should_update,
                agreement_level=agreement_level,
            )
            if not item.dataset_id or not item.modus or not item.question:
                raise ValueError(f"Blank formal-domain field on line {line_number}")
            items.append(item)

    ids = [item.sample_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Formal-domain item IDs must be unique by dataset_id + modus")
    return items


def validate_formal_items(
    items: Sequence[FormalItem],
    pairs: Sequence[BeliefPair],
) -> tuple[FormalItem, ...]:
    """Validate selected frozen pairs against their checked-in ``time_t1`` payloads."""
    item_by_id = {item.sample_id: item for item in items}
    selected: list[FormalItem] = []
    for pair in pairs:
        try:
            item = item_by_id[pair.sample_id]
        except KeyError as error:
            raise ValueError(f"Formal-domain data is missing {pair.sample_id}") from error
        source = pair.time_t1
        expected_options = tuple(value.strip() for value in source.options)
        if (
            item.dataset_id != source.dataset_id
            or item.modus != source.modus
            or item.question != source.question.strip()
            or item.options != expected_options
            or item.ground_truth != source.ground_truth
            or item.should_update != (pair.target == "BU")
            or str(item.agreement_level) != source.agreement_level
        ):
            raise ValueError(
                f"Formal-domain payload differs from pinned Belief-R for {pair.sample_id}"
            )
        selected.append(item)
    return tuple(selected)
