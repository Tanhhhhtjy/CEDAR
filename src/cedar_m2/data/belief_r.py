"""Belief-R parsing, time-step pairing, and preregistered M2 splits."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

BELIEF_R_REPO = "CAiRE/belief_r"
BELIEF_R_REVISION = "3719f5804c63318037465fecf298a7fd78d99121"
PROTOCOL_SEED = 20260711
PROTOCOL_SPLIT_COUNTS = {
    "development": 50,
    "pilot_bu": 100,
    "pilot_bm": 100,
    "extension_bu": 100,
}


@dataclass(frozen=True)
class BeliefRow:
    question: str
    ground_truth: str
    step: str
    modus: str
    relation_type: str
    agreement_level: str
    atomic_idx: str
    dataset_id: str
    options: tuple[str, str, str] = ("", "", "")

    @property
    def key(self) -> str:
        return f"{self.dataset_id}:{self.modus}"


@dataclass(frozen=True)
class BeliefPair:
    time_t: BeliefRow
    time_t1: BeliefRow

    def __post_init__(self) -> None:
        if self.time_t.key != self.time_t1.key:
            raise ValueError("BeliefPair rows must share dataset_id and modus")
        if self.time_t.step != "time_t" or self.time_t1.step != "time_t1":
            raise ValueError("BeliefPair must contain time_t followed by time_t1")

    @property
    def sample_id(self) -> str:
        return self.time_t.key

    @property
    def target(self) -> str:
        return "BU" if self.time_t1.ground_truth == "C" else "BM"

    @property
    def strength(self) -> str:
        return self.time_t1.dataset_id.rsplit("-", 1)[-1]


@dataclass(frozen=True)
class BeliefSplits:
    development: tuple[BeliefPair, ...]
    pilot_bu: tuple[BeliefPair, ...]
    pilot_bm: tuple[BeliefPair, ...]
    extension_bu: tuple[BeliefPair, ...]

    def all_sample_ids(self) -> list[str]:
        groups = (self.development, self.pilot_bu, self.pilot_bm, self.extension_bu)
        return [item.sample_id for group in groups for item in group]

    def manifest(self) -> dict[str, object]:
        def records(items: Iterable[BeliefPair]) -> list[dict[str, str]]:
            return [
                {
                    "sample_id": item.sample_id,
                    "target": item.target,
                    "modus": item.time_t1.modus,
                    "relation_type": item.time_t1.relation_type,
                    "agreement_level": item.time_t1.agreement_level,
                    "ground_truth_t": item.time_t.ground_truth,
                    "ground_truth_t1": item.time_t1.ground_truth,
                }
                for item in items
            ]

        return {
            "source": {"repo_id": BELIEF_R_REPO, "revision": BELIEF_R_REVISION},
            "development": records(self.development),
            "pilot_bu": records(self.pilot_bu),
            "pilot_bm": records(self.pilot_bm),
            "extension_bu": records(self.extension_bu),
        }


def download_belief_r(
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Download only the pinned Belief-R CSV and return its local path."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=BELIEF_R_REPO,
        repo_type="dataset",
        filename="test.csv",
        revision=BELIEF_R_REVISION,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )
    return Path(path)


def read_belief_rows(path: str | Path) -> list[BeliefRow]:
    """Read the public CSV without modifying its natural-language questions."""
    rows: list[BeliefRow] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            ground_truth = record["ground_truth"].strip().upper()
            if ground_truth not in {"A", "B", "C"}:
                raise ValueError(f"Unexpected ground truth {ground_truth!r}")
            rows.append(
                BeliefRow(
                    question=record["questions"],
                    ground_truth=ground_truth,
                    step=record["step"],
                    modus=record["modus"],
                    relation_type=record["types_of_relation"],
                    agreement_level=record["agreement_lv"],
                    atomic_idx=record["atomic_idx"],
                    dataset_id=record["dataset_id"],
                    options=(record["a"], record["b"], record["c"]),
                )
            )
    return rows


def pair_belief_rows(rows: Iterable[BeliefRow]) -> list[BeliefPair]:
    """Pair unique time_t and time_t1 rows; unpaired rows are excluded."""
    indexed: dict[tuple[str, str], BeliefRow] = {}
    for row in rows:
        index_key = (row.key, row.step)
        if index_key in indexed:
            raise ValueError(f"Duplicate Belief-R row for {index_key}")
        indexed[index_key] = row

    pairs: list[BeliefPair] = []
    keys = sorted({key for key, _step in indexed})
    for key in keys:
        time_t = indexed.get((key, "time_t"))
        time_t1 = indexed.get((key, "time_t1"))
        if time_t is not None and time_t1 is not None:
            pairs.append(BeliefPair(time_t=time_t, time_t1=time_t1))
    return pairs


Item = TypeVar("Item")
Stratum = tuple[str, ...]


def proportional_stratified_sample(
    items: Sequence[Item],
    count: int,
    *,
    stratum: Callable[[Item], Stratum],
    identity: Callable[[Item], str],
    seed: int,
) -> list[Item]:
    """Sample exactly ``count`` items with deterministic proportional allocation."""
    if count < 0 or count > len(items):
        raise ValueError(f"Cannot sample {count} items from a population of {len(items)}")
    if count == 0:
        return []

    groups: dict[Stratum, list[Item]] = defaultdict(list)
    for item in sorted(items, key=identity):
        groups[stratum(item)].append(item)

    rng = random.Random(seed)
    for group_items in groups.values():
        rng.shuffle(group_items)

    total = len(items)
    ideal = {key: count * len(group) / total for key, group in groups.items()}
    allocation = {key: math.floor(value) for key, value in ideal.items()}
    remaining = count - sum(allocation.values())
    remainder_order = sorted(
        groups,
        key=lambda key: (-(ideal[key] - allocation[key]), key),
    )
    for key in remainder_order[:remaining]:
        allocation[key] += 1

    sampled = [item for key in sorted(groups) for item in groups[key][: allocation[key]]]
    rng.shuffle(sampled)
    return sampled


def _eligible_protocol_pairs(
    pairs: Iterable[BeliefPair],
) -> tuple[list[BeliefPair], list[BeliefPair]]:
    eligible = [
        pair
        for pair in pairs
        if pair.strength == "strong" and pair.time_t1.agreement_level in {"4", "5", "6"}
    ]
    return (
        [pair for pair in eligible if pair.target == "BU"],
        [pair for pair in eligible if pair.target == "BM"],
    )


def build_protocol_splits(
    pairs: Sequence[BeliefPair], *, seed: int = PROTOCOL_SEED
) -> BeliefSplits:
    """Build the frozen 50-dev, 100-BU, 100-BM, and 100-extension split."""
    bu, bm = _eligible_protocol_pairs(pairs)

    def bu_stratum(item: BeliefPair) -> Stratum:
        return (
            item.time_t1.modus,
            item.time_t1.agreement_level,
            item.time_t1.relation_type,
        )

    def bm_stratum(item: BeliefPair) -> Stratum:
        return (
            item.time_t1.modus,
            item.time_t1.ground_truth,
            item.time_t1.relation_type,
        )

    def identity(item: BeliefPair) -> str:
        return item.sample_id

    dev_bu = proportional_stratified_sample(
        bu, 25, stratum=bu_stratum, identity=identity, seed=seed
    )
    dev_bm = proportional_stratified_sample(
        bm, 25, stratum=bm_stratum, identity=identity, seed=seed + 1
    )
    dev_ids = {item.sample_id for item in [*dev_bu, *dev_bm]}

    remaining_bu = [item for item in bu if item.sample_id not in dev_ids]
    remaining_bm = [item for item in bm if item.sample_id not in dev_ids]
    pilot_bu = proportional_stratified_sample(
        remaining_bu, 100, stratum=bu_stratum, identity=identity, seed=seed + 2
    )
    pilot_bm = proportional_stratified_sample(
        remaining_bm, 100, stratum=bm_stratum, identity=identity, seed=seed + 3
    )

    used_ids = (
        dev_ids | {item.sample_id for item in pilot_bu} | {item.sample_id for item in pilot_bm}
    )
    extension_pool = [item for item in remaining_bu if item.sample_id not in used_ids]
    extension_bu = proportional_stratified_sample(
        extension_pool, 100, stratum=bu_stratum, identity=identity, seed=seed + 4
    )

    splits = BeliefSplits(
        development=tuple([*dev_bu, *dev_bm]),
        pilot_bu=tuple(pilot_bu),
        pilot_bm=tuple(pilot_bm),
        extension_bu=tuple(extension_bu),
    )
    ids = splits.all_sample_ids()
    if len(ids) != len(set(ids)):
        raise AssertionError("Protocol splits overlap")
    return splits


def load_protocol_splits(
    path: str | Path,
    pairs: Sequence[BeliefPair],
) -> BeliefSplits:
    """Load and strictly validate a frozen split manifest against source pairs."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    source = document.get("source")
    expected_source = {"repo_id": BELIEF_R_REPO, "revision": BELIEF_R_REVISION}
    if source != expected_source:
        raise ValueError(f"Belief-R split source differs from pinned source: {source!r}")
    if document.get("seed") != PROTOCOL_SEED:
        raise ValueError(f"Belief-R split seed must be {PROTOCOL_SEED}")

    pair_by_id = {pair.sample_id: pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("Belief-R source pairs contain duplicate sample IDs")

    def load_group(name: str, *, expected_target: str | None) -> tuple[BeliefPair, ...]:
        records = document.get(name)
        if not isinstance(records, list):
            raise ValueError(f"Belief-R split manifest field {name!r} must be a list")
        selected: list[BeliefPair] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Belief-R split {name!r} contains a malformed record")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in pair_by_id:
                raise ValueError(f"Belief-R split {name!r} has unknown sample {sample_id!r}")
            pair = pair_by_id[sample_id]
            expected = {
                "sample_id": pair.sample_id,
                "target": pair.target,
                "modus": pair.time_t1.modus,
                "relation_type": pair.time_t1.relation_type,
                "agreement_level": pair.time_t1.agreement_level,
                "ground_truth_t": pair.time_t.ground_truth,
                "ground_truth_t1": pair.time_t1.ground_truth,
            }
            if dict(record) != expected:
                raise ValueError(f"Belief-R split metadata mismatch for {sample_id}")
            if expected_target is not None and pair.target != expected_target:
                raise ValueError(
                    f"Belief-R split {name!r} expected target {expected_target}, "
                    f"found {pair.target} for {sample_id}"
                )
            selected.append(pair)
        return tuple(selected)

    development = load_group("development", expected_target=None)
    pilot_bu = load_group("pilot_bu", expected_target="BU")
    pilot_bm = load_group("pilot_bm", expected_target="BM")
    extension_bu = load_group("extension_bu", expected_target="BU")
    splits = BeliefSplits(
        development=development,
        pilot_bu=pilot_bu,
        pilot_bm=pilot_bm,
        extension_bu=extension_bu,
    )

    counts = document.get("counts")
    expected_counts = {
        "paired_items": len(pairs),
        **PROTOCOL_SPLIT_COUNTS,
    }
    if not isinstance(counts, Mapping) or any(
        counts.get(name) != count for name, count in expected_counts.items()
    ):
        raise ValueError("Belief-R split manifest counts do not match its records")

    ids = splits.all_sample_ids()
    if len(ids) != len(set(ids)):
        raise ValueError("Belief-R frozen splits overlap")

    rebuilt = build_protocol_splits(pairs, seed=PROTOCOL_SEED)
    for name in PROTOCOL_SPLIT_COUNTS:
        loaded_ids = [pair.sample_id for pair in getattr(splits, name)]
        rebuilt_ids = [pair.sample_id for pair in getattr(rebuilt, name)]
        if loaded_ids != rebuilt_ids:
            raise ValueError(
                f"Belief-R frozen split {name!r} differs from the preregistered selection"
            )
    return splits
