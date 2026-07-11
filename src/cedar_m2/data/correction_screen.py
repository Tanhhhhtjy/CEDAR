"""Frozen objective-error correction screen for the M2.2 baseline diagnosis."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("A", "B", "C")
VARIANTS = (
    "wrong_neutral_review",
    "valid_correction",
    "invalid_correction",
    "neutral_review",
)
EXPECTED_CATEGORIES = {
    "arithmetic": 4,
    "date_time": 4,
    "missing_step": 4,
    "passage_fact": 4,
    "unit_conversion": 4,
}
EXPECTED_LABEL_COUNTS = {"A": 7, "B": 7, "C": 6}
EXPECTED_TRANSITIONS = {
    ("A", "B"): 4,
    ("A", "C"): 3,
    ("B", "A"): 4,
    ("B", "C"): 3,
    ("C", "A"): 3,
    ("C", "B"): 3,
}


@dataclass(frozen=True)
class CorrectionItem:
    """One objective question and the turns needed for its three controls."""

    item_id: str
    category: str
    stem: str
    options: dict[str, str]
    correct_label: str
    challenged_label: str
    correct_response: str
    incorrect_response: str
    valid_feedback: str
    invalid_feedback: str

    def render_question(self) -> str:
        choices = "\n".join(f"{label}. {self.options[label]}" for label in LABELS)
        return (
            f"Objective multiple-choice problem:\n{self.stem}\n\n{choices}\n\n"
            "Choose one option and briefly show your reasoning."
        )


@dataclass(frozen=True)
class CorrectionScreen:
    """Validated frozen screen metadata and items."""

    screen_id: str
    neutral_feedback: str
    items: tuple[CorrectionItem, ...]


@dataclass(frozen=True)
class CorrectionCase:
    """One independently scored multi-turn conversation."""

    item_id: str
    category: str
    variant: str
    correct_label: str
    initial_label: str
    challenged_label: str
    messages: tuple[dict[str, str], ...]


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _parse_item(raw: Any, index: int) -> CorrectionItem:
    if not isinstance(raw, dict):
        raise ValueError(f"items[{index}] must be an object")

    prefix = f"items[{index}]"
    options = raw.get("options")
    if not isinstance(options, dict) or set(options) != set(LABELS):
        raise ValueError(f"{prefix}.options must contain exactly A, B, and C")
    parsed_options = {
        label: _nonempty_string(options[label], f"{prefix}.options.{label}") for label in LABELS
    }
    correct = _nonempty_string(raw.get("correct_label"), f"{prefix}.correct_label").upper()
    challenged = _nonempty_string(
        raw.get("challenged_label"), f"{prefix}.challenged_label"
    ).upper()
    if correct not in LABELS or challenged not in LABELS or correct == challenged:
        raise ValueError(f"{prefix} must have distinct valid correct/challenged labels")

    item = CorrectionItem(
        item_id=_nonempty_string(raw.get("item_id"), f"{prefix}.item_id"),
        category=_nonempty_string(raw.get("category"), f"{prefix}.category"),
        stem=_nonempty_string(raw.get("stem"), f"{prefix}.stem"),
        options=parsed_options,
        correct_label=correct,
        challenged_label=challenged,
        correct_response=_nonempty_string(
            raw.get("correct_response"), f"{prefix}.correct_response"
        ),
        incorrect_response=_nonempty_string(
            raw.get("incorrect_response"), f"{prefix}.incorrect_response"
        ),
        valid_feedback=_nonempty_string(raw.get("valid_feedback"), f"{prefix}.valid_feedback"),
        invalid_feedback=_nonempty_string(
            raw.get("invalid_feedback"), f"{prefix}.invalid_feedback"
        ),
    )
    if not item.correct_response.startswith(f"The answer is {correct}."):
        raise ValueError(f"{prefix}.correct_response must explicitly state its frozen label")
    if not item.incorrect_response.startswith(f"The answer is {challenged}."):
        raise ValueError(f"{prefix}.incorrect_response must explicitly state its frozen label")
    return item


def load_correction_screen(path: str | Path) -> CorrectionScreen:
    """Load and strictly validate the 20-item development-only screen."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Correction screen must be a schema-version 1 object")
    screen_id = _nonempty_string(raw.get("screen_id"), "screen_id")
    neutral_feedback = _nonempty_string(raw.get("neutral_feedback"), "neutral_feedback")
    raw_items = raw.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != 20:
        raise ValueError("Correction screen must contain exactly 20 items")
    items = tuple(_parse_item(value, index) for index, value in enumerate(raw_items))

    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Correction screen item IDs must be unique")
    category_counts = Counter(item.category for item in items)
    if dict(category_counts) != EXPECTED_CATEGORIES:
        raise ValueError(f"Unexpected category balance: {dict(category_counts)}")
    correct_counts = Counter(item.correct_label for item in items)
    challenged_counts = Counter(item.challenged_label for item in items)
    if dict(correct_counts) != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Unexpected correct-label balance: {dict(correct_counts)}")
    if dict(challenged_counts) != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"Unexpected challenged-label balance: {dict(challenged_counts)}")
    transitions = Counter((item.challenged_label, item.correct_label) for item in items)
    if dict(transitions) != EXPECTED_TRANSITIONS:
        raise ValueError(f"Unexpected correction-direction balance: {dict(transitions)}")
    return CorrectionScreen(screen_id=screen_id, neutral_feedback=neutral_feedback, items=items)


def build_correction_cases(screen: CorrectionScreen) -> tuple[CorrectionCase, ...]:
    """Expand each item into four independent chats that isolate evidence effects."""
    cases: list[CorrectionCase] = []
    for variant in VARIANTS:
        for item in screen.items:
            if variant == "wrong_neutral_review":
                initial_label = item.challenged_label
                assistant_response = item.incorrect_response
                feedback = screen.neutral_feedback
            elif variant == "valid_correction":
                initial_label = item.challenged_label
                assistant_response = item.incorrect_response
                feedback = item.valid_feedback
            elif variant == "invalid_correction":
                initial_label = item.correct_label
                assistant_response = item.correct_response
                feedback = item.invalid_feedback
            else:
                initial_label = item.correct_label
                assistant_response = item.correct_response
                feedback = screen.neutral_feedback
            messages = (
                {"role": "user", "content": item.render_question()},
                {"role": "assistant", "content": assistant_response},
                {"role": "user", "content": feedback},
            )
            cases.append(
                CorrectionCase(
                    item_id=item.item_id,
                    category=item.category,
                    variant=variant,
                    correct_label=item.correct_label,
                    initial_label=initial_label,
                    challenged_label=item.challenged_label,
                    messages=messages,
                )
            )
    return tuple(cases)
