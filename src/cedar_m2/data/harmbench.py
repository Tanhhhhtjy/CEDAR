"""Pinned HarmBench text-behavior loading with a test-split firewall."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

HARMBENCH_REPO_URL = "https://github.com/centerforaisafety/HarmBench.git"
HARMBENCH_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
HARMBENCH_SPLIT_FILES = {
    "validation": "data/behavior_datasets/harmbench_behaviors_text_val.csv",
    "test": "data/behavior_datasets/harmbench_behaviors_text_test.csv",
}
HARMBENCH_EXPECTED_COUNTS = {"validation": 41, "test": 159}
HARMBENCH_SEMANTIC_CATEGORIES = (
    "chemical_biological",
    "cybercrime_intrusion",
    "harassment_bullying",
    "harmful",
    "illegal",
    "misinformation_disinformation",
)
_REQUIRED_FIELDS = {
    "Behavior",
    "FunctionalCategory",
    "SemanticCategory",
    "Tags",
    "ContextString",
    "BehaviorID",
}


@dataclass(frozen=True)
class HarmBenchBehavior:
    """One standard, text-only HarmBench behavior."""

    behavior_id: str
    behavior: str
    semantic_category: str
    tags: tuple[str, ...]
    context: str


def checkout_revision(root: str | Path) -> str:
    """Return the Git revision of a local HarmBench checkout."""
    checkout = Path(root)
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"HarmBench root is not a readable Git checkout: {checkout}") from error
    return completed.stdout.strip()


def verify_checkout(root: str | Path) -> None:
    """Reject any HarmBench checkout that is not at the preregistered revision."""
    revision = checkout_revision(root)
    if revision != HARMBENCH_REVISION:
        raise ValueError(
            f"HarmBench checkout is at {revision}, expected pinned revision {HARMBENCH_REVISION}"
        )


def read_harmbench_behaviors(path: str | Path) -> list[HarmBenchBehavior]:
    """Read only standard text behaviors from a HarmBench CSV."""
    behaviors: list[HarmBenchBehavior] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not _REQUIRED_FIELDS <= fields:
            missing = sorted(_REQUIRED_FIELDS - fields)
            raise ValueError(f"HarmBench CSV is missing fields: {missing}")

        for record in reader:
            if record["FunctionalCategory"].strip() != "standard":
                continue
            category = record["SemanticCategory"].strip()
            if category not in HARMBENCH_SEMANTIC_CATEGORIES:
                raise ValueError(f"Unknown HarmBench semantic category {category!r}")
            behavior_id = record["BehaviorID"].strip()
            behavior = record["Behavior"].strip()
            if not behavior_id or not behavior:
                raise ValueError("HarmBench behavior ID and text must be non-empty")
            tags = tuple(tag.strip() for tag in record["Tags"].split(",") if tag.strip())
            behaviors.append(
                HarmBenchBehavior(
                    behavior_id=behavior_id,
                    behavior=behavior,
                    semantic_category=category,
                    tags=tags,
                    context=record["ContextString"].strip(),
                )
            )

    ids = [item.behavior_id for item in behaviors]
    if len(ids) != len(set(ids)):
        raise ValueError("HarmBench standard behavior IDs must be unique")
    return behaviors


def load_harmbench_split(
    root: str | Path,
    split: str,
    *,
    allow_test: bool = False,
) -> tuple[Path, list[HarmBenchBehavior]]:
    """Load a pinned split while requiring explicit authorization for test."""
    if split not in HARMBENCH_SPLIT_FILES:
        raise ValueError(f"Unknown HarmBench split {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError(
            "HarmBench test is sealed; pass allow_test=True only after template freeze"
        )

    checkout = Path(root)
    verify_checkout(checkout)
    path = checkout / HARMBENCH_SPLIT_FILES[split]
    behaviors = read_harmbench_behaviors(path)
    expected = HARMBENCH_EXPECTED_COUNTS[split]
    if len(behaviors) != expected:
        raise ValueError(f"Expected {expected} standard {split} behaviors, found {len(behaviors)}")
    return path, behaviors
